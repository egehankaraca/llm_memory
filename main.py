from datetime import datetime, timedelta, timezone
from dataclasses import replace
from typing import Any
import json
import os
import uuid

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import models
from database import get_db
from memory_analyzer import MemoryDecision, analyze_message, get_analyzer_status
from memory_analyzer import stable_statement_key
from memory_consolidator import (
    ActiveMemory,
    ConsolidationAction,
    ConsolidationPlan,
    canonical_category,
    resolve_consolidation,
)
from memory_retriever import (
    ProfileMemory,
    estimate_tokens as estimate_profile_tokens,
    select_profile_memories,
)
from memory_outbox import serialize_outbox_job
from temporary_memory import (
    StaleTemporaryMemoryError,
    load_temporary_memories,
    serialize_temporary_memory,
    upsert_temporary_memory,
)


app = FastAPI(title="LLM Memory Service API", version="0.12.0")


DEFAULT_WINDOW_MAX_MESSAGES = 10
DEFAULT_WINDOW_MAX_TOKENS = 2_000
DEFAULT_CONVERSATION_TTL_HOURS = 24
DEFAULT_CONSOLIDATION_MAX_FACTS = 50
DEFAULT_PROFILE_MAX_FACTS = 20
DEFAULT_PROFILE_MAX_TOKENS = 1_500
DEFAULT_PINNED_PROFILE_CATEGORIES = (
    "communication,accessibility,emergency_contact"
)
ESTIMATED_CHARACTERS_PER_TOKEN = 3


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def bounded_environment_integer(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def conversation_window_limits() -> tuple[int, int]:
    return (
        bounded_environment_integer(
            "MEMORY_WINDOW_MAX_MESSAGES",
            DEFAULT_WINDOW_MAX_MESSAGES,
            1,
            50,
        ),
        bounded_environment_integer(
            "MEMORY_WINDOW_MAX_TOKENS",
            DEFAULT_WINDOW_MAX_TOKENS,
            128,
            16_000,
        ),
    )


def conversation_ttl_hours() -> int:
    return bounded_environment_integer(
        "MEMORY_CONVERSATION_TTL_HOURS",
        DEFAULT_CONVERSATION_TTL_HOURS,
        1,
        24 * 30,
    )


def temporary_memory_limits() -> tuple[int, int]:
    return (
        bounded_environment_integer("MEMORY_TEMPORARY_MAX_ITEMS", 20, 1, 100),
        bounded_environment_integer("MEMORY_TEMPORARY_MAX_TOKENS", 1_500, 128, 16_000),
    )


def consolidation_max_facts() -> int:
    return bounded_environment_integer(
        "MEMORY_CONSOLIDATION_MAX_FACTS",
        DEFAULT_CONSOLIDATION_MAX_FACTS,
        1,
        200,
    )


def outbox_max_attempts() -> int:
    return bounded_environment_integer(
        "MEMORY_OUTBOX_MAX_ATTEMPTS",
        5,
        1,
        20,
    )


def profile_retrieval_settings() -> tuple[int, int, set[str]]:
    categories = {
        category.strip().casefold()
        for category in os.getenv(
            "MEMORY_PROFILE_PINNED_CATEGORIES",
            DEFAULT_PINNED_PROFILE_CATEGORIES,
        ).split(",")
        if category.strip()
    }
    return (
        bounded_environment_integer(
            "MEMORY_PROFILE_MAX_FACTS",
            DEFAULT_PROFILE_MAX_FACTS,
            1,
            100,
        ),
        bounded_environment_integer(
            "MEMORY_PROFILE_MAX_TOKENS",
            DEFAULT_PROFILE_MAX_TOKENS,
            128,
            16_000,
        ),
        categories,
    )


def estimate_token_count(content: str) -> int:
    """Conservative tokenizer-free estimate suitable for bounded local context."""
    return max(
        1,
        (len(content) + ESTIMATED_CHARACTERS_PER_TOKEN - 1)
        // ESTIMATED_CHARACTERS_PER_TOKEN,
    )


def require_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Datetime değeri timezone içermelidir")
    return value.astimezone(timezone.utc)


def stored_utc_datetime(value: datetime) -> datetime:
    """SQLite returns UTC database timestamps without their timezone."""
    return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None
            else value.astimezone(timezone.utc))


def parse_json_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def filter_unexpired_observations(
    observations: list[Any], now: datetime
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        expires_at = parse_json_datetime(observation.get("expires_at"))
        if expires_at is not None and expires_at > now:
            result.append(observation)
    return result


class FactCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    category: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    key: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    value: Any
    sensitivity: models.Sensitivity = models.Sensitivity.NORMAL
    verification_status: models.VerificationStatus = (
        models.VerificationStatus.UNVERIFIED
    )
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_event_id: str | None = Field(default=None, min_length=1, max_length=36)
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: datetime | None) -> datetime | None:
        return require_aware_datetime(value) if value is not None else None


class TemporaryObservation(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1, max_length=100)
    value: Any
    observed_at: datetime | None = None
    expires_at: datetime

    @field_validator("observed_at", "expires_at")
    @classmethod
    def validate_datetimes(cls, value: datetime | None) -> datetime | None:
        return require_aware_datetime(value) if value is not None else None


class SessionStatePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    active_goal: dict[str, Any] | None = None
    temporary_observations: list[TemporaryObservation] = Field(default_factory=list)


class SessionUpdate(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    state: SessionStatePayload
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: datetime) -> datetime:
        return require_aware_datetime(value)


class ContextRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    query: str | None = Field(default=None, max_length=10_000)
    include_unconfirmed_sensitive_history: bool = False


class TemporaryMemoryCreate(FactCreate):
    session_id: str = Field(min_length=1, max_length=128)
    category: str = Field(default="session", min_length=1, max_length=100,
                          pattern=r"^[A-Za-z0-9_.-]+$")
    occurred_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        return require_aware_datetime(value)


class EventCreate(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=36)
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    event_type: str = Field(
        min_length=1,
        max_length=50,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    payload: dict[str, Any]
    occurred_at: datetime = Field(default_factory=utc_now)
    retention_until: datetime | None = None

    @field_validator("occurred_at", "retention_until")
    @classmethod
    def validate_datetimes(cls, value: datetime | None) -> datetime | None:
        return require_aware_datetime(value) if value is not None else None


class ConversationMessage(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2_000)


class ConversationMessageCreate(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=36)
    user_id: str = Field(min_length=1, max_length=128)
    role: models.ConversationRole
    content: str = Field(min_length=1, max_length=10_000)
    parent_message_id: str | None = Field(default=None, min_length=1, max_length=36)
    occurred_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None

    @field_validator("occurred_at", "expires_at")
    @classmethod
    def validate_datetimes(cls, value: datetime | None) -> datetime | None:
        return require_aware_datetime(value) if value is not None else None


class InteractionProcessRequest(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=36)
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=10_000)
    recent_messages: list[ConversationMessage] | None = Field(
        default=None,
        max_length=10,
    )
    occurred_at: datetime = Field(default_factory=utc_now)

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        return require_aware_datetime(value)


class CandidateDecisionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)


def serialize_fact(fact: models.MemoryFact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "category": fact.category,
        "key": fact.key,
        "value": fact.value_json.get("value"),
        "sensitivity": fact.sensitivity.value,
        "verification_status": fact.verification_status.value,
        "confidence": float(fact.confidence),
        "status": fact.status.value,
        "source_event_id": fact.source_event_id,
        "supersedes_id": fact.supersedes_id,
        "valid_from": fact.valid_from.isoformat(),
        "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
        "expires_at": fact.expires_at.isoformat() if fact.expires_at else None,
    }


def serialize_candidate(candidate: models.MemoryCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "event_id": candidate.source_event_id,
        "memory_type": candidate.memory_type.value,
        "category": candidate.category,
        "key": candidate.key,
        "value": (
            candidate.value_json.get("value") if candidate.value_json else None
        ),
        "sensitivity": candidate.sensitivity.value,
        "confidence": float(candidate.confidence),
        "requires_confirmation": candidate.requires_confirmation,
        "analyzer_source": candidate.analyzer_source,
        "analysis": candidate.analysis_json,
        "consolidation_action": candidate.consolidation_action,
        "consolidates_fact_id": candidate.consolidates_fact_id,
        "status": candidate.status.value,
        "reason": candidate.reason,
        "expires_at": (
            candidate.expires_at.isoformat() if candidate.expires_at else None
        ),
        "applied_ref": candidate.applied_ref,
    }


def load_active_memories(
    user_id: str,
    db: Session,
    *,
    limit: int | None = None,
) -> list[models.MemoryFact]:
    now = utc_now()
    query = db.query(models.MemoryFact).filter(
        models.MemoryFact.user_id == user_id,
        models.MemoryFact.status == models.FactStatus.ACTIVE,
        models.MemoryFact.valid_from <= now,
        or_(
            models.MemoryFact.valid_to.is_(None),
            models.MemoryFact.valid_to > now,
        ),
        or_(
            models.MemoryFact.expires_at.is_(None),
            models.MemoryFact.expires_at > now,
        ),
    ).order_by(models.MemoryFact.updated_at.desc())
    if limit is not None:
        query = query.limit(limit)
    return query.all()


def active_memory_views(facts: list[models.MemoryFact]) -> list[ActiveMemory]:
    return [
        ActiveMemory(
            id=fact.id,
            category=fact.category,
            key=fact.key,
            value=fact.value_json.get("value"),
        )
        for fact in facts
    ]


def profile_memory_views(facts: list[models.MemoryFact]) -> list[ProfileMemory]:
    return [
        ProfileMemory(
            id=fact.id,
            category=fact.category,
            key=fact.key,
            value=fact.value_json.get("value"),
            confidence=float(fact.confidence),
            verification_status=fact.verification_status.value,
            updated_at=fact.updated_at,
        )
        for fact in facts
    ]


def existing_memories_for_analyzer(
    facts: list[models.MemoryFact],
) -> list[dict[str, Any]]:
    protected = models.SENSITIVITIES_REQUIRING_CONFIRMATION
    return [
        {
            "id": fact.id,
            "category": fact.category,
            "key": fact.key,
            "value": "[protected]"
            if fact.sensitivity in protected
            else fact.value_json.get("value"),
        }
        for fact in facts
    ]


def consolidation_metadata(plan: ConsolidationPlan) -> dict[str, Any]:
    return {
        "version": "1",
        "action": plan.action.value,
        "matched_fact_id": plan.matched_fact_id,
        "matched_by": plan.matched_by,
        "similarity": plan.similarity,
        "reason": plan.reason,
    }


def serialize_conversation_message(
    message: models.ConversationMessage,
    *,
    content: str | None = None,
    estimated_tokens: int | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "message_id": message.id,
        "role": message.role.value,
        "content": message.content if content is None else content,
        "estimated_tokens": message.estimated_tokens
        if estimated_tokens is None
        else estimated_tokens,
        "occurred_at": message.occurred_at.isoformat(),
        "expires_at": message.expires_at.isoformat(),
        "truncated": truncated,
    }


def assert_session_owner(session_id: str, user_id: str, db: Session) -> None:
    session_state = db.get(models.SessionState, session_id)
    if session_state is not None and session_state.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session başka bir kullanıcıya aittir",
        )

    message_owner = (
        db.query(models.ConversationMessage.user_id)
        .filter(models.ConversationMessage.session_id == session_id)
        .first()
    )
    if message_owner is not None and message_owner[0] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session başka bir kullanıcıya aittir",
        )

    temporary_owner = (
        db.query(models.TemporaryMemory.user_id)
        .filter(models.TemporaryMemory.session_id == session_id)
        .first()
    )
    if temporary_owner is not None and temporary_owner[0] != user_id:
        raise HTTPException(status_code=403, detail="Session başka bir kullanıcıya aittir")


def bound_external_messages(
    messages: list[dict[str, str]],
    max_messages: int,
    max_tokens: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    used_tokens = 0
    for message in reversed(messages[-max_messages:]):
        content = message["content"]
        estimated_tokens = estimate_token_count(content)
        remaining_tokens = max_tokens - used_tokens
        if remaining_tokens <= 0:
            break
        if estimated_tokens > remaining_tokens:
            if selected:
                break
            character_limit = max(
                1,
                remaining_tokens * ESTIMATED_CHARACTERS_PER_TOKEN - 1,
            )
            content = content[:character_limit] + "…"
            estimated_tokens = estimate_token_count(content)
        selected.append({"role": message["role"], "content": content})
        used_tokens += estimated_tokens
    selected.reverse()
    return selected


def build_memory_analysis_context(
    recent_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Keep only dialogue that can resolve the current user's omitted context.

    User-only ingestion events are not a dialogue. Feeding unrelated statements
    to the extractor can make a complete assertion look like an answer or a
    question. The last assistant turn is the natural boundary for contextual
    replies such as "Evet, bugün yapalım".
    """
    for index in range(len(recent_messages) - 1, -1, -1):
        if recent_messages[index].get("role") == "assistant":
            return recent_messages[index:]
    return []


def build_conversation_window(
    user_id: str,
    session_id: str,
    db: Session,
    *,
    exclude_message_id: str | None = None,
    exclude_unconfirmed_sensitive: bool = False,
) -> dict[str, Any]:
    now = utc_now()
    max_messages, max_tokens = conversation_window_limits()
    query = db.query(models.ConversationMessage).filter(
        models.ConversationMessage.user_id == user_id,
        models.ConversationMessage.session_id == session_id,
        models.ConversationMessage.expires_at > now,
    )
    if exclude_message_id is not None:
        query = query.filter(models.ConversationMessage.id != exclude_message_id)
    if exclude_unconfirmed_sensitive:
        now = utc_now()
        protected_events = (
            db.query(models.MemoryCandidate.source_event_id)
            .filter(
                models.MemoryCandidate.user_id == user_id,
                models.MemoryCandidate.sensitivity.in_(models.SENSITIVITIES_REQUIRING_CONFIRMATION),
                or_(
                    models.MemoryCandidate.status.in_([
                        models.CandidateStatus.PENDING, models.CandidateStatus.REJECTED,
                    ]),
                    and_(
                        models.MemoryCandidate.status == models.CandidateStatus.CONFIRMED,
                        models.MemoryCandidate.expires_at.is_not(None),
                        models.MemoryCandidate.expires_at <= now,
                    ),
                ),
            )
        )
        query = query.filter(
            ~models.ConversationMessage.id.in_(protected_events),
            or_(
                models.ConversationMessage.parent_message_id.is_(None),
                ~models.ConversationMessage.parent_message_id.in_(protected_events),
            ),
        )
        # Asynchronous events have not passed sensitivity policy yet. Keep both
        # the raw user message and its linked assistant response out of future
        # model context until the worker completes the analysis.
        unfinished_events = (
            db.query(models.MemoryOutbox.event_id)
            .filter(
                models.MemoryOutbox.user_id == user_id,
                models.MemoryOutbox.session_id == session_id,
                models.MemoryOutbox.status.in_([
                    models.OutboxStatus.PENDING,
                    models.OutboxStatus.PROCESSING,
                    models.OutboxStatus.RETRY,
                    models.OutboxStatus.FAILED,
                    models.OutboxStatus.CANCELED,
                ]),
            )
        )
        query = query.filter(
            ~models.ConversationMessage.id.in_(unfinished_events),
            or_(
                models.ConversationMessage.parent_message_id.is_(None),
                ~models.ConversationMessage.parent_message_id.in_(unfinished_events),
            ),
        )
    rows = (
        query.order_by(
            models.ConversationMessage.occurred_at.desc(),
            models.ConversationMessage.created_at.desc(),
        )
        .limit(max_messages)
        .all()
    )

    selected: list[dict[str, Any]] = []
    used_tokens = 0
    for message in rows:
        content = message.content
        estimated_tokens = message.estimated_tokens
        remaining_tokens = max_tokens - used_tokens
        if remaining_tokens <= 0:
            break
        truncated = False
        if estimated_tokens > remaining_tokens:
            if selected:
                break
            character_limit = max(
                1,
                remaining_tokens * ESTIMATED_CHARACTERS_PER_TOKEN - 1,
            )
            content = content[:character_limit] + "…"
            estimated_tokens = estimate_token_count(content)
            truncated = True
        selected.append(
            serialize_conversation_message(
                message,
                content=content,
                estimated_tokens=estimated_tokens,
                truncated=truncated,
            )
        )
        used_tokens += estimated_tokens
    selected.reverse()
    return {
        "messages": selected,
        "message_count": len(selected),
        "estimated_tokens": used_tokens,
        "max_messages": max_messages,
        "max_tokens": max_tokens,
    }


def apply_short_term_decision(
    decision: MemoryDecision,
    user_id: str,
    session_id: str,
    db: Session,
    *,
    source_event_id: str | None = None,
    source_text: str | None = None,
    occurred_at: datetime | None = None,
    verification_status: models.VerificationStatus = models.VerificationStatus.UNVERIFIED,
) -> dict[str, Any]:
    assert_session_owner(session_id, user_id, db)
    if decision.expires_at is None:
        raise HTTPException(status_code=422, detail="Geçici memory için expires_at gerekli")
    key = decision.key
    requested_key = key
    slot_action = "model_key"
    category = decision.category or "session"
    metadata = decision.analysis_metadata or {}
    if key in {None, "active_goal", "none", "unknown"}:
        seed = json.dumps(decision.value, ensure_ascii=False, sort_keys=True, default=str)
        key = stable_statement_key("temporary", seed)
        slot_action = "fallback_key"

    start = occurred_at or utc_now()
    existing = (
        db.query(models.TemporaryMemory)
        .filter(
            models.TemporaryMemory.user_id == user_id,
            models.TemporaryMemory.session_id == session_id,
            models.TemporaryMemory.category == category,
            models.TemporaryMemory.key == key,
            models.TemporaryMemory.status == models.FactStatus.ACTIVE,
            models.TemporaryMemory.expires_at > start,
        )
        .first()
    )
    if existing is not None:
        same_value = existing.value_json == {"value": decision.value}
        if same_value:
            slot_action = "same_slot"
        else:
            # A small model can occasionally reuse an unrelated semantic key.
            # Never let that silently delete a different active intention.
            seed = json.dumps(
                {
                    "value": decision.value,
                    "evidence_text": metadata.get("evidence_text") or source_text,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            key = stable_statement_key("temporary", seed)
            slot_action = "collision_disambiguated"
    try:
        row = upsert_temporary_memory(
            db, user_id=user_id, session_id=session_id,
            category=category, key=key, value=decision.value,
            sensitivity=decision.sensitivity, verification_status=verification_status,
            confidence=decision.confidence, source_event_id=source_event_id,
            occurred_at=start, expires_at=decision.expires_at,
        )
    except StaleTemporaryMemoryError as exc:
        # Analyzer writes reaching here are exact-value repeats of the same
        # resolved slot. Reuse the newer record without rolling its timestamp,
        # expiry, provenance, or verification state backward.
        row = db.get(models.TemporaryMemory, exc.newer_memory_id)
        if row is None:
            raise
        slot_action = "stale_reused_newer"
    payload = serialize_temporary_memory(row)
    payload["_slot_resolution"] = {
        "action": slot_action,
        "requested_key": requested_key,
        "resolved_key": row.key,
    }
    return payload


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/analyzer/status")
def analyzer_status() -> dict[str, Any]:
    """Report whether the configured analyzer and model are ready."""
    result = get_analyzer_status()
    result["consolidation_version"] = "1"
    return result


@app.post(
    "/v1/sessions/{session_id}/messages",
    status_code=status.HTTP_201_CREATED,
)
def create_conversation_message(
    session_id: str,
    message: ConversationMessageCreate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Append one idempotent user/assistant message to the TTL conversation log."""
    existing = db.get(models.ConversationMessage, message.message_id)
    if existing is not None:
        same_message = (
            existing.user_id == message.user_id
            and existing.session_id == session_id
            and existing.role == message.role
            and existing.content == message.content
            and existing.parent_message_id == message.parent_message_id
        )
        if not same_message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Message ID farklı bir mesaj için kullanılıyor",
            )
        return {
            "status": "duplicate",
            "message": serialize_conversation_message(existing),
        }

    assert_session_owner(session_id, message.user_id, db)
    parent = None
    if message.parent_message_id is not None:
        parent = db.get(models.ConversationMessage, message.parent_message_id)
        if (
            parent is None
            or parent.user_id != message.user_id
            or parent.session_id != session_id
            or parent.role != models.ConversationRole.USER
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="parent_message_id aynı kullanıcı/session içindeki user mesajı olmalıdır",
            )
    now = utc_now()
    expires_at = message.expires_at or (
        message.occurred_at + timedelta(hours=conversation_ttl_hours())
    )
    if expires_at <= now or expires_at <= message.occurred_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message expires_at hem occurred_at hem de şimdiki zamandan sonra olmalıdır",
        )

    db.query(models.ConversationMessage).filter(
        models.ConversationMessage.expires_at <= now
    ).delete(synchronize_session=False)
    stored = models.ConversationMessage(
        id=message.message_id,
        user_id=message.user_id,
        session_id=session_id,
        role=message.role,
        content=message.content,
        parent_message_id=message.parent_message_id,
        estimated_tokens=estimate_token_count(message.content),
        occurred_at=message.occurred_at,
        expires_at=expires_at,
    )
    db.add(stored)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation message kaydedilemedi",
        ) from exc
    return {
        "status": "created",
        "message": serialize_conversation_message(stored),
    }


@app.get("/v1/sessions/{session_id}/messages")
def list_conversation_messages(
    session_id: str,
    user_id: str = Query(min_length=1, max_length=128),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return the active bounded conversation window in chronological order."""
    assert_session_owner(session_id, user_id, db)
    return build_conversation_window(user_id, session_id, db)


@app.post("/v1/temporary-memories", status_code=status.HTTP_201_CREATED)
def create_temporary_memory(
    memory: TemporaryMemoryCreate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Write one explicitly identified, finite-lived item from a trusted backend."""
    assert_session_owner(memory.session_id, memory.user_id, db)
    if memory.expires_at <= utc_now():
        raise HTTPException(status_code=422, detail="Geçici memory expires_at gelecekte olmalıdır")
    if (memory.sensitivity in models.SENSITIVITIES_REQUIRING_CONFIRMATION
            and memory.verification_status == models.VerificationStatus.UNVERIFIED):
        raise HTTPException(status_code=422, detail="Hassas bilgi interactions:process ve açık onay gerektirir")
    try:
        row = upsert_temporary_memory(
            db, user_id=memory.user_id, session_id=memory.session_id,
            category=memory.category, key=memory.key, value=memory.value,
            sensitivity=memory.sensitivity, verification_status=memory.verification_status,
            confidence=memory.confidence, source_event_id=memory.source_event_id,
            occurred_at=memory.occurred_at, expires_at=memory.expires_at,
        )
        db.commit()
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Geçici memory aynı anda güncellendi") from exc
    return {"status": "success", "temporary_memory": serialize_temporary_memory(row)}


@app.get("/v1/sessions/{session_id}/temporary-memories")
def list_session_temporary_memories(
    session_id: str,
    user_id: str = Query(min_length=1, max_length=128),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    assert_session_owner(session_id, user_id, db)
    max_items, max_tokens = temporary_memory_limits()
    rows = load_temporary_memories(
        db, user_id=user_id, session_id=session_id, now=utc_now(),
        max_items=max_items, max_tokens=max_tokens,
    )
    return {
        "temporary_memories": [serialize_temporary_memory(row) for row in rows],
        "memory_count": len(rows), "max_items": max_items, "max_tokens": max_tokens,
    }


@app.post("/v1/memories", status_code=status.HTTP_201_CREATED)
def create_memory_fact(
    fact: FactCreate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return _write_memory_fact(fact, db)


def _write_memory_fact(
    fact: FactCreate,
    db: Session,
    *,
    commit: bool = True,
) -> dict[str, Any]:
    """Create or supersede a structured long-term fact."""
    now = utc_now()
    value_json = {"value": fact.value}

    try:
        active_facts = (
            db.query(models.MemoryFact)
            .filter(
                models.MemoryFact.user_id == fact.user_id,
                models.MemoryFact.category == fact.category,
                models.MemoryFact.key == fact.key,
                models.MemoryFact.status == models.FactStatus.ACTIVE,
            )
            .order_by(models.MemoryFact.created_at.desc())
            .with_for_update()
            .all()
        )

        live_facts: list[models.MemoryFact] = []
        for existing in active_facts:
            if existing.expires_at is not None and stored_utc_datetime(existing.expires_at) <= now:
                existing.status = models.FactStatus.EXPIRED
                existing.valid_to = existing.expires_at
                existing.updated_at = now
            else:
                live_facts.append(existing)

        current = live_facts[0] if live_facts else None
        unchanged = (
            current is not None
            and current.value_json == value_json
            and current.sensitivity == fact.sensitivity
            and current.verification_status == fact.verification_status
            and float(current.confidence) == fact.confidence
            and current.expires_at == fact.expires_at
        )

        if unchanged:
            for duplicate in live_facts[1:]:
                duplicate.status = models.FactStatus.SUPERSEDED
                duplicate.valid_to = now
                duplicate.updated_at = now
            db.commit() if commit else db.flush()
            return {
                "status": "unchanged",
                "fact_id": current.id,
                "supersedes_id": current.supersedes_id,
            }

        for existing in live_facts:
            existing.status = models.FactStatus.SUPERSEDED
            existing.valid_to = now
            existing.updated_at = now

        new_fact = models.MemoryFact(
            id=str(uuid.uuid4()),
            user_id=fact.user_id,
            category=fact.category,
            key=fact.key,
            value_json=value_json,
            sensitivity=fact.sensitivity,
            verification_status=fact.verification_status,
            confidence=fact.confidence,
            source_event_id=fact.source_event_id,
            supersedes_id=current.id if current is not None else None,
            valid_from=now,
            expires_at=fact.expires_at,
            status=models.FactStatus.ACTIVE,
        )
        db.add(new_fact)
        db.commit() if commit else db.flush()
        return {
            "status": "created",
            "fact_id": new_fact.id,
            "supersedes_id": new_fact.supersedes_id,
        }
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Fact aynı anda başka bir istek tarafından güncellendi",
        ) from exc


@app.put("/v1/sessions/{session_id}/state")
def update_session_state(
    session_id: str,
    session_data: SessionUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Create or replace one user's short-term session state."""
    assert_session_owner(session_id, session_data.user_id, db)
    now = utc_now()
    if session_data.expires_at <= now:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Session expires_at gelecekte olmalıdır",
        )

    db_session = (
        db.query(models.SessionState)
        .filter(models.SessionState.session_id == session_id)
        .with_for_update()
        .first()
    )

    if db_session is not None and db_session.user_id != session_data.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session başka bir kullanıcıya aittir",
        )

    state = session_data.state.model_dump(mode="json")
    state["temporary_observations"] = filter_unexpired_observations(
        state.get("temporary_observations", []),
        now,
    )

    if db_session is None:
        db_session = models.SessionState(
            session_id=session_id,
            user_id=session_data.user_id,
            state_json=state,
            expires_at=session_data.expires_at,
            version=1,
        )
        db.add(db_session)
    else:
        db_session.state_json = state
        db_session.expires_at = session_data.expires_at
        db_session.version += 1
        db_session.updated_at = now

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session aynı anda başka bir istek tarafından güncellendi",
        ) from exc

    return {"status": "success", "version": db_session.version}


@app.post("/v1/events", status_code=status.HTTP_201_CREATED)
def create_memory_event(
    event: EventCreate,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Append an idempotent interaction event."""
    existing = db.get(models.MemoryEvent, event.event_id)
    if existing is not None:
        if existing.user_id != event.user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Event ID başka bir kullanıcı tarafından kullanılıyor",
            )
        return {"status": "duplicate", "event_id": existing.id}

    if event.session_id is not None:
        session = db.get(models.SessionState, event.session_id)
        if session is not None and session.user_id != event.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session başka bir kullanıcıya aittir",
            )

    db.add(
        models.MemoryEvent(
            id=event.event_id,
            user_id=event.user_id,
            session_id=event.session_id,
            event_type=event.event_type,
            payload_json=event.payload,
            occurred_at=event.occurred_at,
            retention_until=event.retention_until,
        )
    )
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Event kaydedilemedi",
        ) from exc

    return {"status": "created", "event_id": event.event_id}


@app.post("/v1/context:build")
def build_context(
    request: ContextRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Build a query-aware bounded context from profile and session memory."""
    now = utc_now()
    assert_session_owner(request.session_id, request.user_id, db)
    active_facts = load_active_memories(request.user_id, db)
    max_facts, max_profile_tokens, pinned_categories = (
        profile_retrieval_settings()
    )
    retrieval = select_profile_memories(
        profile_memory_views(active_facts),
        query=request.query,
        max_facts=max_facts,
        max_tokens=max_profile_tokens,
        pinned_categories=pinned_categories,
        now=now,
    )

    # The legacy retriever ranks unchanged values. Account for the attributed
    # transport wrapper separately; never truncate a fact to make it fit.
    active_by_id = {fact.id: fact for fact in active_facts}
    source_event_ids = {
        active_by_id[selected.memory.id].source_event_id
        for selected in retrieval.selected
        if active_by_id[selected.memory.id].source_event_id is not None
    }
    owned_source_event_ids = {
        event_id
        for (event_id,) in (
            db.query(models.MemoryEvent.id)
            .filter(
                models.MemoryEvent.user_id == request.user_id,
                models.MemoryEvent.id.in_(source_event_ids),
            )
            .all()
        )
    } if source_event_ids else set()
    profile_facts: list[dict[str, Any]] = []
    packed_selected = []
    for selected in retrieval.selected:
        memory = selected.memory
        source_event_id = active_by_id[memory.id].source_event_id
        attributed_fact = {
            "fact_id": memory.id,
            "owner_user_id": request.user_id,
            "category": memory.category,
            "key": memory.key,
            "value": memory.value,
            "provenance": {
                "source_event_id": (
                    source_event_id
                    if source_event_id in owned_source_event_ids
                    else None
                ),
                "verification_status": memory.verification_status,
                "confidence": memory.confidence,
            },
        }
        if estimate_profile_tokens([*profile_facts, attributed_fact]) <= max_profile_tokens:
            profile_facts.append(attributed_fact)
            packed_selected.append(replace(
                selected, estimated_tokens=estimate_profile_tokens(attributed_fact)
            ))
    retrieval = replace(
        retrieval,
        selected=tuple(packed_selected),
        estimated_tokens=estimate_profile_tokens(profile_facts),
    )

    profile_data: dict[str, dict[str, Any]] = {}
    for selected in retrieval.selected:
        memory = selected.memory
        profile_data.setdefault(memory.category, {})[memory.key] = (
            memory.value
        )

    session_state = (
        db.query(models.SessionState)
        .filter(
            models.SessionState.session_id == request.session_id,
            models.SessionState.user_id == request.user_id,
            models.SessionState.expires_at > now,
        )
        .first()
    )
    session_data = session_state.state_json if session_state else {}
    observations = filter_unexpired_observations(
        session_data.get("temporary_observations", []),
        now,
    )
    conversation_window = build_conversation_window(
        request.user_id,
        request.session_id,
        db,
        exclude_unconfirmed_sensitive=not request.include_unconfirmed_sensitive_history,
    )
    max_temporary_items, max_temporary_tokens = temporary_memory_limits()
    temporary_rows = load_temporary_memories(
        db, user_id=request.user_id, session_id=request.session_id, now=now,
        max_items=max_temporary_items, max_tokens=max_temporary_tokens,
    )
    temporary_items = [serialize_temporary_memory(row) for row in temporary_rows]
    # Keep the old single-goal projection for older clients. New clients must
    # consume temporary_memories; this projection is never written back.
    latest_goal = next((row.value_json.get("value") for row in temporary_rows
                        if row.sensitivity not in models.SENSITIVITIES_REQUIRING_CONFIRMATION), None)
    if not isinstance(latest_goal, dict):
        latest_goal = {"description": latest_goal} if latest_goal is not None else None
    legacy_goal = session_data.get("active_goal") or {}
    if isinstance(legacy_goal, dict):
        legacy_expiry = parse_json_datetime(legacy_goal.get("expires_at"))
        if legacy_expiry is not None and legacy_expiry <= now:
            legacy_goal = {}
    else:
        legacy_goal = {}
    has_temporary_history = db.query(models.TemporaryMemory.id).filter(
        models.TemporaryMemory.user_id == request.user_id,
        models.TemporaryMemory.session_id == request.session_id,
    ).first() is not None

    return {
        "as_of": now.isoformat(),
        "user_id": request.user_id,
        "session_id": request.session_id,
        "profile": profile_data,
        "profile_facts": profile_facts,
        "session": latest_goal if latest_goal is not None else {} if has_temporary_history else legacy_goal,
        "temporary_memories": temporary_items,
        "temporary_memory_budget": {
            "max_items": max_temporary_items, "max_tokens": max_temporary_tokens,
            "selected_item_count": len(temporary_items),
            "estimated_tokens": estimate_profile_tokens({"temporary_memories": temporary_items}),
        },
        "temporary_observations": observations,
        "recent_messages": [
            {"role": message["role"], "content": message["content"]}
            for message in conversation_window["messages"]
        ],
        "conversation_window": {
            key: value
            for key, value in conversation_window.items()
            if key != "messages"
        },
        "message_refs": [
            message["message_id"] for message in conversation_window["messages"]
        ],
        "history_policy": {
            "include_unconfirmed_sensitive_history": request.include_unconfirmed_sensitive_history,
        },
        "profile_retrieval": {
            "strategy": "deterministic_lexical_v1",
            "query_used": retrieval.query_used,
            "eligible_fact_count": retrieval.eligible_fact_count,
            "candidate_fact_count": retrieval.candidate_fact_count,
            "selected_fact_count": len(retrieval.selected),
            "omitted_fact_count": retrieval.omitted_fact_count,
            "estimated_tokens": retrieval.estimated_tokens,
            "max_facts": retrieval.max_facts,
            "max_tokens": retrieval.max_tokens,
            "pinned_categories": sorted(pinned_categories),
            "selected": [
                {
                    "fact_id": selected.memory.id,
                    "category": selected.memory.category,
                    "key": selected.memory.key,
                    "score": selected.score,
                    "estimated_tokens": selected.estimated_tokens,
                    "reasons": list(selected.reasons),
                }
                for selected in retrieval.selected
            ],
        },
        "memory_refs": [
            selected.memory.id for selected in retrieval.selected
        ],
    }


@app.post(
    "/v1/interactions:enqueue",
    status_code=status.HTTP_202_ACCEPTED,
)
def enqueue_interaction(
    interaction: InteractionProcessRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Durably enqueue analysis without calling the memory model in this request."""
    assert_session_owner(interaction.session_id, interaction.user_id, db)
    existing_event = db.get(models.MemoryEvent, interaction.event_id)
    if existing_event is not None and (
        existing_event.user_id != interaction.user_id
        or existing_event.session_id not in {None, interaction.session_id}
        or existing_event.event_type != "user_message"
        or existing_event.payload_json.get("text") != interaction.text
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Event ID farklı bir interaction için kullanılıyor",
        )

    existing_message = db.get(models.ConversationMessage, interaction.event_id)
    if existing_message is not None and (
        existing_message.user_id != interaction.user_id
        or existing_message.session_id != interaction.session_id
        or existing_message.role != models.ConversationRole.USER
        or existing_message.content != interaction.text
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Message ID farklı bir interaction için kullanılıyor",
        )

    existing_job = db.get(models.MemoryOutbox, interaction.event_id)
    if existing_job is not None:
        if (
            existing_job.user_id != interaction.user_id
            or existing_job.session_id != interaction.session_id
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Event ID başka bir outbox işi tarafından kullanılıyor",
            )
        return {
            "status": "duplicate",
            "event_id": interaction.event_id,
            "job": serialize_outbox_job(existing_job),
        }

    # Capture only dialogue that existed before this event. This makes delayed
    # extraction deterministic and prevents future messages leaking backward.
    max_messages, max_tokens = conversation_window_limits()
    if interaction.recent_messages is not None:
        recent_messages = bound_external_messages(
            [message.model_dump() for message in interaction.recent_messages],
            max_messages,
            max_tokens,
        )
    else:
        stored_window = build_conversation_window(
            interaction.user_id,
            interaction.session_id,
            db,
            exclude_message_id=interaction.event_id,
            exclude_unconfirmed_sensitive=True,
        )
        recent_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in stored_window["messages"]
        ]
    analysis_context = build_memory_analysis_context(recent_messages)

    now = utc_now()
    message_expiry = interaction.occurred_at + timedelta(
        hours=conversation_ttl_hours()
    )
    if message_expiry <= now or message_expiry <= interaction.occurred_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Interaction konuşma TTL süresi dolmuş olamaz",
        )

    if existing_event is None:
        db.add(models.MemoryEvent(
            id=interaction.event_id,
            user_id=interaction.user_id,
            session_id=interaction.session_id,
            event_type="user_message",
            payload_json={"text": interaction.text},
            occurred_at=interaction.occurred_at,
        ))
    if existing_message is None:
        db.add(models.ConversationMessage(
            id=interaction.event_id,
            user_id=interaction.user_id,
            session_id=interaction.session_id,
            role=models.ConversationRole.USER,
            content=interaction.text,
            estimated_tokens=estimate_token_count(interaction.text),
            occurred_at=interaction.occurred_at,
            expires_at=message_expiry,
        ))

    already_processed = db.query(models.MemoryCandidate.id).filter(
        models.MemoryCandidate.source_event_id == interaction.event_id,
        models.MemoryCandidate.user_id == interaction.user_id,
    ).first() is not None
    job = models.MemoryOutbox(
        event_id=interaction.event_id,
        user_id=interaction.user_id,
        session_id=interaction.session_id,
        status=(
            models.OutboxStatus.COMPLETED
            if already_processed
            else models.OutboxStatus.PENDING
        ),
        analysis_context_json=analysis_context,
        attempt_count=0,
        max_attempts=outbox_max_attempts(),
        available_at=now,
        completed_at=now if already_processed else None,
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Interaction aynı anda başka bir istek tarafından kuyruğa alındı",
        ) from exc
    db.refresh(job)
    return {
        "status": "already_processed" if already_processed else "queued",
        "event_id": interaction.event_id,
        "job": serialize_outbox_job(job),
    }


@app.get("/v1/interactions/{event_id}/status")
def interaction_status(
    event_id: str,
    user_id: str = Query(min_length=1, max_length=128),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    job = db.get(models.MemoryOutbox, event_id)
    if job is None or job.user_id != user_id:
        raise HTTPException(status_code=404, detail="Outbox interaction bulunamadı")
    candidates = (
        db.query(models.MemoryCandidate)
        .filter(
            models.MemoryCandidate.source_event_id == event_id,
            models.MemoryCandidate.user_id == user_id,
        )
        .order_by(models.MemoryCandidate.decision_index)
        .all()
    )
    return {
        "job": serialize_outbox_job(job),
        "decisions": [serialize_candidate(candidate) for candidate in candidates],
    }


@app.post("/v1/interactions:process", status_code=status.HTTP_201_CREATED)
def process_interaction(
    interaction: InteractionProcessRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Classify a message and route it to short-, long-, or review memory."""
    existing_event = db.get(models.MemoryEvent, interaction.event_id)
    if existing_event is not None and existing_event.user_id != interaction.user_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Event ID başka bir kullanıcı tarafından kullanılıyor",
        )
    if existing_event is not None and (
        existing_event.session_id not in {None, interaction.session_id}
        or existing_event.event_type != "user_message"
        or existing_event.payload_json.get("text") != interaction.text
    ):
        raise HTTPException(status_code=409, detail="Event ID farklı bir interaction için kullanılıyor")

    existing_candidates = (
        db.query(models.MemoryCandidate)
        .filter(
            models.MemoryCandidate.source_event_id == interaction.event_id,
            models.MemoryCandidate.user_id == interaction.user_id,
        )
        .order_by(models.MemoryCandidate.decision_index)
        .all()
    )
    if existing_candidates:
        if any(
            candidate.session_id != interaction.session_id
            for candidate in existing_candidates
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Event ID başka bir session interaction'ı tarafından kullanılıyor",
            )
        return {
            "status": "duplicate",
            "event_id": interaction.event_id,
            "decisions": [
                serialize_candidate(candidate) for candidate in existing_candidates
            ],
        }

    if existing_event is not None:
        # A partial ingest retry keeps the original source time and TTL. The
        # caller may have omitted occurred_at on both HTTP requests.
        interaction = interaction.model_copy(update={
            "occurred_at": stored_utc_datetime(existing_event.occurred_at),
        })

    assert_session_owner(interaction.session_id, interaction.user_id, db)
    max_messages, max_tokens = conversation_window_limits()
    if interaction.recent_messages is not None:
        recent_messages = bound_external_messages(
            [message.model_dump() for message in interaction.recent_messages],
            max_messages,
            max_tokens,
        )
    else:
        stored_window = build_conversation_window(
            interaction.user_id,
            interaction.session_id,
            db,
            exclude_message_id=interaction.event_id,
        )
        recent_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in stored_window["messages"]
        ]

    if existing_event is None:
        create_memory_event(
            EventCreate(
                event_id=interaction.event_id,
                user_id=interaction.user_id,
                session_id=interaction.session_id,
                event_type="user_message",
                payload={"text": interaction.text},
                occurred_at=interaction.occurred_at,
            ),
            db,
        )

    create_conversation_message(
        interaction.session_id,
        ConversationMessageCreate(
            message_id=interaction.event_id,
            user_id=interaction.user_id,
            role=models.ConversationRole.USER,
            content=interaction.text,
            occurred_at=interaction.occurred_at,
        ),
        db,
    )

    active_facts_for_analysis = load_active_memories(
        interaction.user_id,
        db,
        limit=consolidation_max_facts(),
    )
    max_temporary_items, max_temporary_tokens = temporary_memory_limits()
    active_temporary_for_analysis = load_temporary_memories(
        db, user_id=interaction.user_id, session_id=interaction.session_id,
        now=utc_now(), max_items=max_temporary_items, max_tokens=max_temporary_tokens,
    )
    analysis_context = build_memory_analysis_context(recent_messages)
    decisions = analyze_message(
        interaction.text,
        interaction.occurred_at,
        analysis_context,
        [*existing_memories_for_analyzer(active_facts_for_analysis),
         *existing_memories_for_analyzer(active_temporary_for_analysis)],
    )
    candidates: list[models.MemoryCandidate] = []

    for index, decision in enumerate(decisions):
        candidate_status = models.CandidateStatus.PENDING
        applied_ref: str | None = None
        candidate_category = decision.category
        candidate_key = decision.key
        candidate_reason = decision.reason
        candidate_analysis = dict(decision.analysis_metadata or {})
        candidate_analysis["analysis_context"] = {
            "strategy": "last_assistant_suffix_v1",
            "message_count": len(analysis_context),
        }
        consolidation_action = "not_applicable"
        consolidates_fact_id: str | None = None

        plan: ConsolidationPlan | None = None
        if (
            decision.memory_type
            in {
                models.CandidateMemoryType.LONG_TERM,
                models.CandidateMemoryType.SENSITIVE,
            }
            and decision.category is not None
            and decision.key is not None
            and decision.expires_at is None
        ):
            active_facts = load_active_memories(
                interaction.user_id,
                db,
                limit=consolidation_max_facts(),
            )
            plan = resolve_consolidation(
                category=decision.category,
                key=decision.key,
                value=decision.value,
                source_text=interaction.text,
                active_memories=active_memory_views(active_facts),
                matched_memory_id=candidate_analysis.get("matched_memory_id"),
                relation_to_existing=candidate_analysis.get(
                    "relation_to_existing",
                    "none",
                ),
                # Presentation punctuation is not a semantic profile change for
                # ordinary memories. Protected literal values (credentials,
                # financial data, addresses, etc.) always remain exact.
                ignore_terminal_sentence_punctuation=(
                    decision.sensitivity
                    not in models.SENSITIVITIES_REQUIRING_CONFIRMATION
                    and canonical_category(decision.category)
                    in {"routine", "preference", "communication", "relationship"}
                ),
            )
            candidate_category = plan.category
            candidate_key = plan.key
            consolidation_action = plan.action.value
            consolidates_fact_id = plan.matched_fact_id
            candidate_analysis["consolidation"] = consolidation_metadata(plan)
            candidate_reason = (
                f"{decision.reason} Consolidation: {plan.reason}"
            )[:500]

        requires_confirmation = decision.requires_confirmation or (
            decision.memory_type == models.CandidateMemoryType.SENSITIVE
            or decision.sensitivity
            in models.SENSITIVITIES_REQUIRING_CONFIRMATION
        )
        if (
            plan is not None
            and plan.action
            == ConsolidationAction.CONFLICT_REQUIRES_CONFIRMATION
        ):
            requires_confirmation = True

        if (
            decision.memory_type == models.CandidateMemoryType.LONG_TERM
            and not requires_confirmation
            and decision.expires_at is None
        ):
            if (
                plan is not None
                and plan.action == ConsolidationAction.UNCHANGED
                and plan.matched_fact_id is not None
            ):
                applied_ref = plan.matched_fact_id
            else:
                result = _write_memory_fact(
                    FactCreate(
                        user_id=interaction.user_id,
                        category=candidate_category or "uncategorized",
                        key=candidate_key or f"fact_{interaction.event_id}",
                        value=decision.value,
                        sensitivity=decision.sensitivity,
                        verification_status=models.VerificationStatus.UNVERIFIED,
                        confidence=decision.confidence,
                        source_event_id=interaction.event_id,
                        expires_at=decision.expires_at,
                    ),
                    db,
                    commit=False,
                )
                applied_ref = result["fact_id"]
            candidate_status = models.CandidateStatus.AUTO_APPLIED
        elif (
            (decision.memory_type == models.CandidateMemoryType.SHORT_TERM
             or decision.expires_at is not None)
            and not requires_confirmation
            and decision.memory_type != models.CandidateMemoryType.DISCARD
        ):
            if decision.expires_at is not None and require_aware_datetime(decision.expires_at) <= utc_now():
                candidate_status = models.CandidateStatus.IGNORED
                candidate_analysis["expired_at_ingestion"] = True
            else:
                result = apply_short_term_decision(
                    decision, interaction.user_id, interaction.session_id, db,
                    source_event_id=interaction.event_id,
                    source_text=interaction.text,
                    occurred_at=interaction.occurred_at,
                )
                candidate_status = models.CandidateStatus.AUTO_APPLIED
                applied_ref = result["memory_id"]
                candidate_category = result["category"]
                candidate_key = result["key"]
                candidate_analysis["temporary_slot"] = result["_slot_resolution"]
        elif decision.memory_type == models.CandidateMemoryType.DISCARD:
            candidate_status = models.CandidateStatus.IGNORED

        candidate_analysis["storage_destination"] = (
            "none" if decision.memory_type == models.CandidateMemoryType.DISCARD
            else "temporary" if decision.expires_at is not None
            or decision.memory_type == models.CandidateMemoryType.SHORT_TERM
            else "profile"
        )

        candidate = models.MemoryCandidate(
            id=str(uuid.uuid4()),
            user_id=interaction.user_id,
            session_id=interaction.session_id,
            source_event_id=interaction.event_id,
            decision_index=index,
            memory_type=decision.memory_type,
            category=candidate_category,
            key=candidate_key,
            value_json={"value": decision.value}
            if decision.value is not None
            else None,
            sensitivity=decision.sensitivity,
            confidence=decision.confidence,
            requires_confirmation=requires_confirmation,
            analyzer_source=decision.analyzer_source,
            analysis_json=candidate_analysis,
            consolidation_action=consolidation_action,
            consolidates_fact_id=consolidates_fact_id,
            status=candidate_status,
            reason=candidate_reason,
            expires_at=decision.expires_at,
            applied_ref=applied_ref,
            resolved_at=utc_now()
            if candidate_status
            in {models.CandidateStatus.AUTO_APPLIED, models.CandidateStatus.IGNORED}
            else None,
        )
        db.add(candidate)
        candidates.append(candidate)

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Interaction aynı anda başka bir istek tarafından işlendi",
        ) from exc

    return {
        "status": "processed",
        "event_id": interaction.event_id,
        "decisions": [serialize_candidate(candidate) for candidate in candidates],
    }


@app.get("/v1/candidates")
def list_memory_candidates(
    user_id: str = Query(min_length=1, max_length=128),
    candidate_status: models.CandidateStatus | None = Query(
        default=None,
        alias="status",
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    query = db.query(models.MemoryCandidate).filter(
        models.MemoryCandidate.user_id == user_id
    )
    if candidate_status is not None:
        query = query.filter(models.MemoryCandidate.status == candidate_status)

    candidates = (
        query.order_by(models.MemoryCandidate.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "items": [serialize_candidate(candidate) for candidate in candidates],
        "limit": limit,
        "offset": offset,
    }


@app.post("/v1/candidates/{candidate_id}:confirm")
def confirm_memory_candidate(
    candidate_id: str,
    request: CandidateDecisionRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    candidate = (
        db.query(models.MemoryCandidate)
        .filter(
            models.MemoryCandidate.id == candidate_id,
            models.MemoryCandidate.user_id == request.user_id,
        )
        .with_for_update()
        .first()
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="Memory candidate bulunamadı")
    if candidate.status == models.CandidateStatus.CONFIRMED:
        return serialize_candidate(candidate)
    if candidate.status != models.CandidateStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Yalnızca pending candidate onaylanabilir",
        )
    if candidate.category is None or candidate.key is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Candidate kalıcı memory alanlarını içermiyor",
        )

    value = candidate.value_json.get("value") if candidate.value_json else None
    if candidate.expires_at is not None:
        expiry = stored_utc_datetime(candidate.expires_at)
        if expiry <= utc_now():
            raise HTTPException(status_code=422, detail="Süresi dolan memory onaylanamaz; yeni beyan gerekli")
        if candidate.session_id is None:
            raise HTTPException(status_code=422, detail="Geçici memory için session gerekli")
        source = db.get(models.MemoryEvent, candidate.source_event_id)
        result = apply_short_term_decision(
            MemoryDecision(
                memory_type=models.CandidateMemoryType.SHORT_TERM,
                category=candidate.category, key=candidate.key, value=value,
                sensitivity=candidate.sensitivity, confidence=float(candidate.confidence),
                requires_confirmation=False, expires_at=expiry, reason="Açık kullanıcı onayı",
                analyzer_source=candidate.analyzer_source,
                analysis_metadata=dict(candidate.analysis_json or {}),
            ),
            request.user_id, candidate.session_id, db,
            source_event_id=candidate.source_event_id,
            source_text=(
                source.payload_json.get("text")
                if source is not None and isinstance(source.payload_json, dict)
                else None
            ),
            occurred_at=stored_utc_datetime(source.occurred_at) if source is not None else utc_now(),
            verification_status=models.VerificationStatus.USER_CONFIRMED,
        )
        applied_ref = result["memory_id"]
        candidate.category = result["category"]
        candidate.key = result["key"]
        candidate.analysis_json = {
            **(candidate.analysis_json or {}),
            "temporary_slot": result["_slot_resolution"],
        }
    else:
        fact_result = _write_memory_fact(
            FactCreate(
                user_id=request.user_id, category=candidate.category, key=candidate.key,
                value=value, sensitivity=candidate.sensitivity,
                verification_status=models.VerificationStatus.USER_CONFIRMED,
                confidence=float(candidate.confidence), source_event_id=candidate.source_event_id,
            ),
            db, commit=False,
        )
        applied_ref = fact_result["fact_id"]
    candidate.status = models.CandidateStatus.CONFIRMED
    candidate.applied_ref = applied_ref
    candidate.resolved_at = utc_now()
    db.commit()
    return serialize_candidate(candidate)


@app.post("/v1/candidates/{candidate_id}:reject")
def reject_memory_candidate(
    candidate_id: str,
    request: CandidateDecisionRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    candidate = (
        db.query(models.MemoryCandidate)
        .filter(
            models.MemoryCandidate.id == candidate_id,
            models.MemoryCandidate.user_id == request.user_id,
        )
        .with_for_update()
        .first()
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="Memory candidate bulunamadı")
    if candidate.status == models.CandidateStatus.REJECTED:
        return serialize_candidate(candidate)
    if candidate.status != models.CandidateStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Yalnızca pending candidate reddedilebilir",
        )

    candidate.status = models.CandidateStatus.REJECTED
    candidate.resolved_at = utc_now()
    db.commit()
    return serialize_candidate(candidate)


@app.get("/v1/users/{user_id}/memories")
def list_user_memories(
    user_id: str,
    include_inactive: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    now = utc_now()
    query = db.query(models.MemoryFact).filter(models.MemoryFact.user_id == user_id)
    if not include_inactive:
        query = query.filter(
            models.MemoryFact.status == models.FactStatus.ACTIVE,
            or_(
                models.MemoryFact.valid_to.is_(None),
                models.MemoryFact.valid_to > now,
            ),
            or_(
                models.MemoryFact.expires_at.is_(None),
                models.MemoryFact.expires_at > now,
            ),
        )

    facts = (
        query.order_by(models.MemoryFact.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "items": [serialize_fact(fact) for fact in facts],
        "limit": limit,
        "offset": offset,
    }


@app.delete("/v1/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(
    memory_id: str,
    user_id: str = Query(min_length=1, max_length=128),
    db: Session = Depends(get_db),
) -> Response:
    fact = (
        db.query(models.MemoryFact)
        .filter(
            models.MemoryFact.id == memory_id,
            models.MemoryFact.user_id == user_id,
        )
        .first()
    )
    if fact is None:
        raise HTTPException(status_code=404, detail="Memory bulunamadı")
    db.delete(fact)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.delete(
    "/v1/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_session(
    session_id: str,
    user_id: str = Query(min_length=1, max_length=128),
    db: Session = Depends(get_db),
) -> Response:
    assert_session_owner(session_id, user_id, db)
    session = (
        db.query(models.SessionState)
        .filter(
            models.SessionState.session_id == session_id,
            models.SessionState.user_id == user_id,
        )
        .first()
    )
    deleted_messages = (
        db.query(models.ConversationMessage)
        .filter(
            models.ConversationMessage.session_id == session_id,
            models.ConversationMessage.user_id == user_id,
        )
        .delete(synchronize_session=False)
    )
    deleted_temporary = db.query(models.TemporaryMemory).filter(
        models.TemporaryMemory.session_id == session_id,
        models.TemporaryMemory.user_id == user_id,
    ).delete(synchronize_session=False)
    db.query(models.MemoryOutbox).filter(
        models.MemoryOutbox.session_id == session_id,
        models.MemoryOutbox.user_id == user_id,
        models.MemoryOutbox.status.in_([
            models.OutboxStatus.PENDING,
            models.OutboxStatus.PROCESSING,
            models.OutboxStatus.RETRY,
        ]),
    ).update(
        {
            models.MemoryOutbox.status: models.OutboxStatus.CANCELED,
            models.MemoryOutbox.locked_at: None,
            models.MemoryOutbox.locked_by: None,
            models.MemoryOutbox.updated_at: utc_now(),
        },
        synchronize_session=False,
    )
    if session is None and deleted_messages == 0 and deleted_temporary == 0:
        raise HTTPException(status_code=404, detail="Session bulunamadı")
    if session is not None:
        db.delete(session)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
