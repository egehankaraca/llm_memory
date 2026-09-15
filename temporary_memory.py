"""Per-item short-term memory persistence and bounded session retrieval."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from memory_retriever import estimate_tokens
from models import (
    FactStatus,
    MemoryEvent,
    Sensitivity,
    TemporaryMemory,
    VerificationStatus,
)


TEMPORARY_CANDIDATE_MULTIPLIER = 5
MAX_TEMPORARY_CANDIDATES = 500


class StaleTemporaryMemoryError(ValueError):
    """Raised when a late event tries to replace a newer active slot."""

    def __init__(self, newer_memory_id: str) -> None:
        self.newer_memory_id = newer_memory_id
        super().__init__("occurred_at is older than the active temporary memory")


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("Memory timestamps must be finite datetime values")
    if value.tzinfo is None or value.utcoffset() is None:
        # SQLite strips timezone information from UTC DateTime columns.
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def serialize_temporary_memory(row: TemporaryMemory) -> dict[str, Any]:
    """Keep the complete value with its owner, expiry and source attribution."""
    return {
        "memory_id": row.id,
        "owner_user_id": row.user_id,
        "session_id": row.session_id,
        "category": row.category,
        "key": row.key,
        "value": row.value_json.get("value"),
        "occurred_at": _aware_utc(row.occurred_at).isoformat(),
        "expires_at": _aware_utc(row.expires_at).isoformat(),
        "sensitivity": Sensitivity(row.sensitivity).value,
        "provenance": {
            "source_event_id": row.source_event_id,
            "verification_status": VerificationStatus(row.verification_status).value,
            "confidence": float(row.confidence),
        },
    }


def upsert_temporary_memory(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    category: str,
    key: str,
    value: Any,
    sensitivity: Sensitivity,
    verification_status: VerificationStatus,
    confidence: float,
    source_event_id: str | None,
    occurred_at: datetime,
    expires_at: datetime,
) -> TemporaryMemory:
    """Replace only the matching session slot without deleting its history.

    Flushes but never commits; the caller owns the surrounding transaction.
    """
    for name, field, max_length in (
        ("user_id", user_id, 128),
        ("session_id", session_id, 128),
        ("category", category, 100),
        ("key", key, 100),
    ):
        if not isinstance(field, str) or not field.strip() or len(field) > max_length:
            raise ValueError(f"{name} must be a nonempty string of at most {max_length} characters")
    start = _aware_utc(occurred_at)
    expiry = _aware_utc(expires_at)
    if expiry <= start:
        raise ValueError("expires_at must be later than occurred_at")
    if not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be finite and between 0 and 1")
    sensitivity = Sensitivity(sensitivity)
    verification_status = VerificationStatus(verification_status)
    if source_event_id is not None:
        if not isinstance(source_event_id, str) or not source_event_id.strip():
            raise ValueError("source_event_id must be a nonempty string when supplied")
        source = db.get(MemoryEvent, source_event_id)
        if source is None or source.user_id != user_id:
            raise ValueError("source_event_id must belong to the memory owner")
        if source.session_id is not None and source.session_id != session_id:
            raise ValueError("source_event_id must belong to the memory session")
        previous_attempt = db.scalars(
            select(TemporaryMemory).where(
                TemporaryMemory.user_id == user_id,
                TemporaryMemory.session_id == session_id,
                TemporaryMemory.category == category,
                TemporaryMemory.key == key,
                TemporaryMemory.source_event_id == source_event_id,
            )
        ).first()
        if previous_attempt is not None:
            if (
                previous_attempt.status == FactStatus.ACTIVE
                and previous_attempt.value_json == {"value": value}
                and _aware_utc(previous_attempt.occurred_at) == start
                and _aware_utc(previous_attempt.expires_at) == expiry
                and previous_attempt.sensitivity == sensitivity
                and previous_attempt.verification_status == verification_status
                and round(float(previous_attempt.confidence), 3) == round(float(confidence), 3)
            ):
                return previous_attempt
            raise ValueError("source event has already produced a different memory for this slot")

    # Release the partial-unique-index slot before inserting a replacement.
    # Expired rows remain available for audit, but are never eligible context.
    existing_rows = db.scalars(
        select(TemporaryMemory)
        .where(
            TemporaryMemory.user_id == user_id,
            TemporaryMemory.session_id == session_id,
            TemporaryMemory.category == category,
            TemporaryMemory.key == key,
            TemporaryMemory.status == FactStatus.ACTIVE,
        )
        .with_for_update()
    ).all()
    newer = next(
        (
            row
            for row in sorted(
                existing_rows,
                key=lambda item: _aware_utc(item.occurred_at),
                reverse=True,
            )
            if _aware_utc(row.occurred_at) > start
        ),
        None,
    )
    if newer is not None:
        raise StaleTemporaryMemoryError(newer.id)
    for previous in existing_rows:
        previous.status = (
            FactStatus.EXPIRED
            if _aware_utc(previous.expires_at) <= start
            else FactStatus.SUPERSEDED
        )
    db.flush()

    row = TemporaryMemory(
        user_id=user_id,
        session_id=session_id,
        category=category,
        key=key,
        value_json={"value": value},
        sensitivity=sensitivity,
        verification_status=verification_status,
        confidence=confidence,
        source_event_id=source_event_id,
        occurred_at=start,
        expires_at=expiry,
        status=FactStatus.ACTIVE,
    )
    db.add(row)
    db.flush()
    return row


def load_temporary_memories(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    now: datetime,
    max_items: int = 20,
    max_tokens: int = 1500,
) -> list[TemporaryMemory]:
    """Select newest whole records within item and serialized-wrapper budgets.

    Scan at most five candidates per requested item, capped at 500. If recent
    oversized records exhaust that bounded pool, older records are omitted
    rather than extending retrieval into an unbounded history scan.
    """
    if not isinstance(max_items, int) or not isinstance(max_tokens, int):
        raise ValueError("Temporary memory budgets must be integers")
    if max_items < 0 or max_tokens < 0:
        raise ValueError("Temporary memory budgets must be nonnegative")
    retrieval_time = _aware_utc(now)
    if max_items == 0 or max_tokens == 0:
        return []
    rows = db.scalars(
        select(TemporaryMemory)
        .where(
            TemporaryMemory.user_id == user_id,
            TemporaryMemory.session_id == session_id,
            TemporaryMemory.status == FactStatus.ACTIVE,
            TemporaryMemory.occurred_at <= retrieval_time,
            TemporaryMemory.expires_at > retrieval_time,
        )
        .order_by(
            TemporaryMemory.occurred_at.desc(),
            TemporaryMemory.created_at.desc(),
            TemporaryMemory.id.desc(),
        )
        .limit(min(MAX_TEMPORARY_CANDIDATES, max_items * TEMPORARY_CANDIDATE_MULTIPLIER))
    )
    selected: list[TemporaryMemory] = []
    serialized: list[dict[str, Any]] = []
    for row in rows:
        payload = serialize_temporary_memory(row)
        if estimate_tokens({"temporary_memories": [*serialized, payload]}) > max_tokens:
            continue
        selected.append(row)
        serialized.append(payload)
        if len(selected) >= max_items:
            break
    return selected
