"""Durable PostgreSQL outbox primitives for asynchronous memory extraction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.orm import Session, aliased

import models


RUNNABLE_STATUSES = {
    models.OutboxStatus.PENDING,
    models.OutboxStatus.RETRY,
}
UNFINISHED_STATUSES = {
    models.OutboxStatus.PENDING,
    models.OutboxStatus.RETRY,
    models.OutboxStatus.PROCESSING,
}


def aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def serialize_outbox_job(job: models.MemoryOutbox) -> dict[str, Any]:
    return {
        "event_id": job.event_id,
        "user_id": job.user_id,
        "session_id": job.session_id,
        "status": job.status.value,
        "attempt_count": job.attempt_count,
        "max_attempts": job.max_attempts,
        "available_at": aware_utc(job.available_at).isoformat(),
        "locked_at": aware_utc(job.locked_at).isoformat() if job.locked_at else None,
        "locked_by": job.locked_by,
        "last_error": job.last_error,
        "completed_at": (
            aware_utc(job.completed_at).isoformat() if job.completed_at else None
        ),
        "created_at": aware_utc(job.created_at).isoformat(),
        "updated_at": aware_utc(job.updated_at).isoformat(),
    }


def claim_next_outbox_job(
    db: Session,
    *,
    worker_id: str,
    lease_seconds: int = 300,
    now: datetime | None = None,
) -> models.MemoryOutbox | None:
    """Atomically claim the oldest eligible job while preserving session order."""
    if not worker_id.strip() or len(worker_id) > 128:
        raise ValueError("worker_id must be a nonempty string of at most 128 characters")
    if lease_seconds < 1:
        raise ValueError("lease_seconds must be positive")
    claim_time = aware_utc(now or datetime.now(timezone.utc))
    stale_before = claim_time - timedelta(seconds=lease_seconds)

    job = aliased(models.MemoryOutbox, name="outbox_job")
    earlier = aliased(models.MemoryOutbox, name="earlier_outbox_job")
    earlier_unfinished = exists(
        select(1).where(
            earlier.user_id == job.user_id,
            earlier.session_id == job.session_id,
            earlier.status.in_(UNFINISHED_STATUSES),
            or_(
                earlier.created_at < job.created_at,
                and_(
                    earlier.created_at == job.created_at,
                    earlier.event_id < job.event_id,
                ),
            ),
        )
    )
    runnable = or_(
        job.status.in_(RUNNABLE_STATUSES),
        and_(
            job.status == models.OutboxStatus.PROCESSING,
            job.locked_at.is_not(None),
            job.locked_at <= stale_before,
        ),
    )
    statement = (
        select(job)
        .where(
            runnable,
            job.available_at <= claim_time,
            ~earlier_unfinished,
        )
        .order_by(job.created_at.asc(), job.event_id.asc())
        .limit(1)
    )
    # PostgreSQL workers do not wait on a row already being claimed by another
    # process. SQLite is retained only for unit tests and has no SKIP LOCKED.
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        statement = statement.with_for_update(skip_locked=True)

    claimed = db.scalars(statement).first()
    if claimed is None:
        return None
    claimed.status = models.OutboxStatus.PROCESSING
    claimed.attempt_count += 1
    claimed.locked_at = claim_time
    claimed.locked_by = worker_id
    claimed.last_error = None
    claimed.updated_at = claim_time
    db.commit()
    db.refresh(claimed)
    return claimed


def complete_outbox_job(
    db: Session,
    *,
    event_id: str,
    worker_id: str,
    now: datetime | None = None,
) -> models.MemoryOutbox:
    completed_at = aware_utc(now or datetime.now(timezone.utc))
    job = db.get(models.MemoryOutbox, event_id)
    if job is None:
        raise ValueError("Outbox job no longer exists")
    if job.status == models.OutboxStatus.COMPLETED:
        return job
    if job.status != models.OutboxStatus.PROCESSING or job.locked_by != worker_id:
        raise ValueError("Outbox job is not owned by this worker")
    job.status = models.OutboxStatus.COMPLETED
    job.completed_at = completed_at
    job.locked_at = None
    job.locked_by = None
    job.last_error = None
    job.updated_at = completed_at
    db.commit()
    db.refresh(job)
    return job


def fail_outbox_job(
    db: Session,
    *,
    event_id: str,
    worker_id: str,
    error: str,
    retryable: bool = True,
    base_retry_seconds: int = 2,
    now: datetime | None = None,
) -> models.MemoryOutbox | None:
    failure_time = aware_utc(now or datetime.now(timezone.utc))
    job = db.get(models.MemoryOutbox, event_id)
    if job is None:
        return None
    if job.status != models.OutboxStatus.PROCESSING or job.locked_by != worker_id:
        return job
    terminal = not retryable or job.attempt_count >= job.max_attempts
    job.status = (
        models.OutboxStatus.FAILED if terminal else models.OutboxStatus.RETRY
    )
    delay = min(300, max(0, base_retry_seconds) * (2 ** max(0, job.attempt_count - 1)))
    job.available_at = failure_time if terminal else failure_time + timedelta(seconds=delay)
    job.locked_at = None
    job.locked_by = None
    job.last_error = error.strip()[:2_000] or "Unknown worker error"
    job.updated_at = failure_time
    db.commit()
    db.refresh(job)
    return job
