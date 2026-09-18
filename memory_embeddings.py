"""Durable fact-embedding jobs and pgvector cosine profile retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import time
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

import models
from database import SessionLocal
from profile_semantic import SemanticSettings, embed_texts, semantic_settings


EMBEDDING_DIMENSIONS = 768
RUNNABLE_STATUSES = {models.OutboxStatus.PENDING, models.OutboxStatus.RETRY}


def aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def bounded_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def embedding_max_attempts() -> int:
    return bounded_integer("MEMORY_EMBEDDING_MAX_ATTEMPTS", 5, 1, 20)


def embedding_top_k() -> int:
    return bounded_integer("MEMORY_PROFILE_VECTOR_TOP_K", 20, 1, 100)


def fact_embedding_text(fact: models.MemoryFact) -> str:
    value = json.dumps(
        fact.value_json.get("value"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"category: {fact.category}\nkey: {fact.key}\nvalue: {value}"


def embedding_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def enqueue_fact_embedding(
    db: Session,
    fact: models.MemoryFact,
    *,
    settings: SemanticSettings | None = None,
) -> models.MemoryEmbeddingJob | None:
    """Ensure a durable job exists in the fact-write transaction."""
    active_settings = settings or semantic_settings()
    if not active_settings.enabled:
        return None
    if not active_settings.model:
        raise ValueError("embedding model cannot be empty")
    text = fact_embedding_text(fact)
    content_hash = embedding_content_hash(text)
    embedding = db.get(
        models.MemoryEmbedding,
        {"memory_id": fact.id, "model": active_settings.model},
    )
    if embedding is not None and embedding.content_hash == content_hash:
        return None

    job = db.get(
        models.MemoryEmbeddingJob,
        {"memory_id": fact.id, "model": active_settings.model},
    )
    now = datetime.now(timezone.utc)
    if job is None:
        job = models.MemoryEmbeddingJob(
            memory_id=fact.id,
            model=active_settings.model,
            user_id=fact.user_id,
            content_hash=content_hash,
            status=models.OutboxStatus.PENDING,
            attempt_count=0,
            max_attempts=embedding_max_attempts(),
            available_at=now,
            created_at=now,
            updated_at=now,
        )
        db.add(job)
    elif job.content_hash != content_hash or job.status in {
        models.OutboxStatus.COMPLETED,
        models.OutboxStatus.FAILED,
        models.OutboxStatus.CANCELED,
    }:
        job.user_id = fact.user_id
        job.content_hash = content_hash
        job.status = models.OutboxStatus.PENDING
        job.attempt_count = 0
        job.max_attempts = embedding_max_attempts()
        job.available_at = now
        job.locked_at = None
        job.locked_by = None
        job.last_error = None
        job.completed_at = None
        job.updated_at = now
    else:
        # A matching pending/retry/processing job already owns this work.
        return None
    return job


def enqueue_missing_embeddings(
    db: Session,
    *,
    user_id: str | None = None,
    settings: SemanticSettings | None = None,
) -> int:
    active_settings = settings or semantic_settings()
    if not active_settings.enabled:
        raise ValueError("Set MEMORY_PROFILE_SEMANTIC_PROVIDER=ollama first")
    now = datetime.now(timezone.utc)
    query = db.query(models.MemoryFact).filter(
        models.MemoryFact.status == models.FactStatus.ACTIVE,
        models.MemoryFact.valid_from <= now,
        or_(models.MemoryFact.valid_to.is_(None), models.MemoryFact.valid_to > now),
        or_(models.MemoryFact.expires_at.is_(None), models.MemoryFact.expires_at > now),
    )
    if user_id is not None:
        query = query.filter(models.MemoryFact.user_id == user_id)
    queued = 0
    for fact in query.order_by(models.MemoryFact.updated_at.desc()).all():
        if enqueue_fact_embedding(db, fact, settings=active_settings) is not None:
            queued += 1
    db.commit()
    return queued


def embedding_status(
    db: Session,
    *,
    user_id: str | None = None,
    settings: SemanticSettings | None = None,
) -> dict[str, Any]:
    """Return index coverage and durable-job counts without calling Ollama."""
    active_settings = settings or semantic_settings()
    now = datetime.now(timezone.utc)
    fact_query = db.query(models.MemoryFact.id).filter(
        models.MemoryFact.status == models.FactStatus.ACTIVE,
        models.MemoryFact.valid_from <= now,
        or_(models.MemoryFact.valid_to.is_(None), models.MemoryFact.valid_to > now),
        or_(models.MemoryFact.expires_at.is_(None), models.MemoryFact.expires_at > now),
    )
    embedding_query = (
        db.query(models.MemoryEmbedding.memory_id)
        .join(models.MemoryFact, models.MemoryFact.id == models.MemoryEmbedding.memory_id)
        .filter(
            models.MemoryEmbedding.model == active_settings.model,
            models.MemoryFact.status == models.FactStatus.ACTIVE,
            models.MemoryFact.valid_from <= now,
            or_(models.MemoryFact.valid_to.is_(None), models.MemoryFact.valid_to > now),
            or_(models.MemoryFact.expires_at.is_(None), models.MemoryFact.expires_at > now),
        )
    )
    job_query = db.query(
        models.MemoryEmbeddingJob.status,
        func.count(models.MemoryEmbeddingJob.memory_id),
    ).filter(models.MemoryEmbeddingJob.model == active_settings.model)
    if user_id is not None:
        fact_query = fact_query.filter(models.MemoryFact.user_id == user_id)
        embedding_query = embedding_query.filter(models.MemoryEmbedding.user_id == user_id)
        job_query = job_query.filter(models.MemoryEmbeddingJob.user_id == user_id)
    job_counts = {
        status.value: count
        for status, count in job_query.group_by(models.MemoryEmbeddingJob.status).all()
    }
    active_fact_count = fact_query.count()
    indexed_count = embedding_query.count()
    return {
        "enabled": active_settings.enabled,
        "provider": active_settings.provider,
        "model": active_settings.model if active_settings.enabled else None,
        "backend": "pgvector_cosine",
        "dimensions": EMBEDDING_DIMENSIONS,
        "user_id": user_id,
        "active_fact_count": active_fact_count,
        "indexed_count": indexed_count,
        "missing_count": max(0, active_fact_count - indexed_count),
        "job_counts": job_counts,
    }


def claim_next_embedding_job(
    db: Session,
    *,
    worker_id: str,
    model: str | None = None,
    lease_seconds: int = 300,
    now: datetime | None = None,
) -> models.MemoryEmbeddingJob | None:
    if not worker_id.strip() or len(worker_id) > 128:
        raise ValueError("worker_id must contain 1-128 characters")
    claim_time = aware_utc(now or datetime.now(timezone.utc))
    stale_before = claim_time - timedelta(seconds=max(1, lease_seconds))
    conditions = [
            models.MemoryEmbeddingJob.available_at <= claim_time,
            or_(
                models.MemoryEmbeddingJob.status.in_(RUNNABLE_STATUSES),
                (
                    (models.MemoryEmbeddingJob.status == models.OutboxStatus.PROCESSING)
                    & models.MemoryEmbeddingJob.locked_at.is_not(None)
                    & (models.MemoryEmbeddingJob.locked_at <= stale_before)
                ),
            ),
    ]
    if model is not None:
        conditions.append(models.MemoryEmbeddingJob.model == model)
    statement = (
        select(models.MemoryEmbeddingJob)
        .where(*conditions)
        .order_by(
            models.MemoryEmbeddingJob.created_at.asc(),
            models.MemoryEmbeddingJob.memory_id.asc(),
        )
        .limit(1)
    )
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        statement = statement.with_for_update(skip_locked=True)
    job = db.scalars(statement).first()
    if job is None:
        return None
    job.status = models.OutboxStatus.PROCESSING
    job.attempt_count += 1
    job.locked_at = claim_time
    job.locked_by = worker_id
    job.last_error = None
    job.updated_at = claim_time
    db.commit()
    db.refresh(job)
    return job


def _fail_embedding_job(
    db: Session,
    *,
    memory_id: str,
    model: str,
    worker_id: str,
    error: Exception,
    base_retry_seconds: int,
) -> models.MemoryEmbeddingJob | None:
    job = db.get(
        models.MemoryEmbeddingJob,
        {"memory_id": memory_id, "model": model},
    )
    if job is None:
        return None
    if job.status != models.OutboxStatus.PROCESSING or job.locked_by != worker_id:
        return job
    terminal = job.attempt_count >= job.max_attempts
    now = datetime.now(timezone.utc)
    job.status = models.OutboxStatus.FAILED if terminal else models.OutboxStatus.RETRY
    delay = min(300, max(0, base_retry_seconds) * (2 ** max(0, job.attempt_count - 1)))
    job.available_at = now if terminal else now + timedelta(seconds=delay)
    job.locked_at = None
    job.locked_by = None
    job.last_error = f"{type(error).__name__}: {error}"[:2_000]
    job.updated_at = now
    db.commit()
    db.refresh(job)
    return job


def serialize_embedding_job(job: models.MemoryEmbeddingJob) -> dict[str, Any]:
    created_at = aware_utc(job.created_at)
    completed_at = aware_utc(job.completed_at) if job.completed_at else None
    now = datetime.now(timezone.utc)
    return {
        "memory_id": job.memory_id,
        "model": job.model,
        "user_id": job.user_id,
        "status": job.status.value,
        "attempt_count": job.attempt_count,
        "max_attempts": job.max_attempts,
        "last_error": job.last_error,
        "completed_at": completed_at.isoformat() if completed_at else None,
        "created_at": created_at.isoformat(),
        "timing": {
            "enqueue_to_completion_ms": (
                round(max(0.0, (completed_at - created_at).total_seconds()) * 1000, 3)
                if completed_at is not None
                else None
            ),
            "current_age_ms": (
                None
                if completed_at is not None
                else round(max(0.0, (now - created_at).total_seconds()) * 1000, 3)
            ),
        },
    }


def process_one_embedding_job(
    *,
    session_factory: sessionmaker[Session] = SessionLocal,
    worker_id: str,
    lease_seconds: int = 300,
    base_retry_seconds: int = 2,
    settings: SemanticSettings | None = None,
) -> dict[str, Any] | None:
    db = session_factory()
    claimed_key: tuple[str, str] | None = None
    try:
        active_settings = settings or semantic_settings()
        if not active_settings.enabled:
            return None
        job = claim_next_embedding_job(
            db,
            worker_id=worker_id,
            model=active_settings.model,
            lease_seconds=lease_seconds,
        )
        if job is None:
            return None
        claimed_key = (job.memory_id, job.model)
        fact = db.get(models.MemoryFact, job.memory_id)
        if fact is None:
            return None
        if fact.user_id != job.user_id:
            raise ValueError("embedding job ownership does not match memory fact")
        if fact.status != models.FactStatus.ACTIVE:
            job.status = models.OutboxStatus.CANCELED
            job.completed_at = datetime.now(timezone.utc)
            job.locked_at = None
            job.locked_by = None
            job.updated_at = job.completed_at
            db.commit()
            return {"job": serialize_embedding_job(job), "indexed": False}

        text = fact_embedding_text(fact)
        content_hash = embedding_content_hash(text)
        if content_hash != job.content_hash:
            raise ValueError("embedding job content hash does not match memory fact")
        vector = embed_texts([text], settings=active_settings)[0]
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding dimensions must be {EMBEDDING_DIMENSIONS}, got {len(vector)}"
            )
        row = db.get(
            models.MemoryEmbedding,
            {"memory_id": fact.id, "model": job.model},
        )
        now = datetime.now(timezone.utc)
        if row is None:
            row = models.MemoryEmbedding(
                memory_id=fact.id,
                model=job.model,
                user_id=fact.user_id,
                content_hash=content_hash,
                dimensions=EMBEDDING_DIMENSIONS,
                embedding=list(vector),
                created_at=now,
                updated_at=now,
            )
            db.add(row)
        else:
            row.user_id = fact.user_id
            row.content_hash = content_hash
            row.dimensions = EMBEDDING_DIMENSIONS
            row.embedding = list(vector)
            row.updated_at = now
        job.status = models.OutboxStatus.COMPLETED
        job.completed_at = now
        job.locked_at = None
        job.locked_by = None
        job.last_error = None
        job.updated_at = now
        db.commit()
        db.refresh(job)
        return {"job": serialize_embedding_job(job), "indexed": True}
    except Exception as exc:
        db.rollback()
        if claimed_key is None:
            raise
        failed = _fail_embedding_job(
            db,
            memory_id=claimed_key[0],
            model=claimed_key[1],
            worker_id=worker_id,
            error=exc,
            base_retry_seconds=base_retry_seconds,
        )
        return {
            "job": serialize_embedding_job(failed) if failed is not None else None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        db.close()


@dataclass(frozen=True)
class PersistentSemanticResult:
    scores: dict[str, float]
    provider: str
    model: str | None
    enabled: bool
    available: bool
    candidate_count: int
    indexed_count: int
    query_embedding_ms: float
    vector_search_ms: float
    backend: str
    reason: str | None = None


def score_profile_memories_pgvector(
    db: Session,
    *,
    user_id: str,
    query: str | None,
    settings: SemanticSettings | None = None,
) -> PersistentSemanticResult:
    active_settings = settings or semantic_settings()
    base = {
        "provider": active_settings.provider,
        "model": active_settings.model if active_settings.enabled else None,
        "enabled": active_settings.enabled,
        "backend": "pgvector_cosine",
    }
    if not active_settings.enabled:
        return PersistentSemanticResult(
            scores={}, available=False, candidate_count=0, indexed_count=0,
            query_embedding_ms=0.0, vector_search_ms=0.0, reason="disabled", **base,
        )
    if not query or not query.strip():
        return PersistentSemanticResult(
            scores={}, available=False, candidate_count=0, indexed_count=0,
            query_embedding_ms=0.0, vector_search_ms=0.0, reason="query_missing", **base,
        )
    if db.bind is None or db.bind.dialect.name != "postgresql":
        return PersistentSemanticResult(
            scores={}, available=False, candidate_count=0, indexed_count=0,
            query_embedding_ms=0.0, vector_search_ms=0.0,
            reason="postgresql_pgvector_required", **base,
        )

    now = datetime.now(timezone.utc)
    active_filter = (
        (models.MemoryFact.user_id == user_id),
        (models.MemoryFact.status == models.FactStatus.ACTIVE),
        (models.MemoryFact.valid_from <= now),
        or_(models.MemoryFact.valid_to.is_(None), models.MemoryFact.valid_to > now),
        or_(models.MemoryFact.expires_at.is_(None), models.MemoryFact.expires_at > now),
        (models.MemoryEmbedding.user_id == user_id),
        (models.MemoryEmbedding.model == active_settings.model),
    )
    indexed_count = (
        db.query(models.MemoryEmbedding.memory_id)
        .join(models.MemoryFact, models.MemoryFact.id == models.MemoryEmbedding.memory_id)
        .filter(*active_filter)
        .count()
    )
    if indexed_count == 0:
        return PersistentSemanticResult(
            scores={}, available=True, candidate_count=0, indexed_count=0,
            query_embedding_ms=0.0, vector_search_ms=0.0,
            reason="no_indexed_memories", **base,
        )

    try:
        started = time.perf_counter()
        query_vector = embed_texts([query], settings=active_settings)[0]
        query_embedding_ms = (time.perf_counter() - started) * 1000
        if len(query_vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"query dimensions must be {EMBEDDING_DIMENSIONS}, got {len(query_vector)}"
            )
        distance = models.MemoryEmbedding.embedding.cosine_distance(list(query_vector))
        started = time.perf_counter()
        rows = (
            db.query(models.MemoryEmbedding.memory_id, distance.label("distance"))
            .join(models.MemoryFact, models.MemoryFact.id == models.MemoryEmbedding.memory_id)
            .filter(*active_filter)
            .order_by(distance.asc())
            .limit(embedding_top_k())
            .all()
        )
        vector_search_ms = (time.perf_counter() - started) * 1000
        scores = {
            memory_id: max(-1.0, min(1.0, 1.0 - float(distance_value)))
            for memory_id, distance_value in rows
            if distance_value is not None and math.isfinite(float(distance_value))
        }
    except Exception as exc:
        return PersistentSemanticResult(
            scores={}, available=False, candidate_count=indexed_count,
            indexed_count=indexed_count, query_embedding_ms=0.0,
            vector_search_ms=0.0, reason=f"{type(exc).__name__}: {exc}", **base,
        )
    return PersistentSemanticResult(
        scores=scores,
        available=True,
        candidate_count=len(scores),
        indexed_count=indexed_count,
        query_embedding_ms=round(query_embedding_ms, 3),
        vector_search_ms=round(vector_search_ms, 3),
        reason=None,
        **base,
    )
