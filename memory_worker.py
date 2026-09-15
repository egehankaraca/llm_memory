"""One durable asynchronous memory worker, reusable by CLI and tests."""

from __future__ import annotations

import os
import socket
import uuid
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session, sessionmaker

import main
import models
from database import SessionLocal
from memory_outbox import (
    claim_next_outbox_job,
    complete_outbox_job,
    fail_outbox_job,
    serialize_outbox_job,
)


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"[:128]


def process_one_outbox_job(
    *,
    session_factory: sessionmaker[Session] = SessionLocal,
    worker_id: str,
    lease_seconds: int = 300,
    base_retry_seconds: int = 2,
) -> dict[str, Any] | None:
    """Claim and process at most one job; return None when the queue is empty."""
    db = session_factory()
    claimed_event_id: str | None = None
    try:
        job = claim_next_outbox_job(
            db,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        if job is None:
            return None
        claimed_event_id = job.event_id
        event = db.get(models.MemoryEvent, job.event_id)
        if event is None:
            raise ValueError("Outbox event is missing")
        if (
            event.user_id != job.user_id
            or event.session_id != job.session_id
            or event.event_type != "user_message"
            or not isinstance(event.payload_json, dict)
            or not isinstance(event.payload_json.get("text"), str)
        ):
            raise ValueError("Outbox event payload or ownership is invalid")

        snapshot = job.analysis_context_json
        if not isinstance(snapshot, list) or any(
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
            for message in snapshot
        ):
            raise ValueError("Outbox analysis context is invalid")

        result = main.process_interaction(
            main.InteractionProcessRequest(
                event_id=event.id,
                user_id=event.user_id,
                session_id=job.session_id,
                text=event.payload_json["text"],
                occurred_at=main.stored_utc_datetime(event.occurred_at),
                recent_messages=[
                    main.ConversationMessage.model_validate(message)
                    for message in snapshot
                ],
            ),
            db,
        )
        completed = complete_outbox_job(
            db,
            event_id=job.event_id,
            worker_id=worker_id,
        )
        return {
            "job": serialize_outbox_job(completed),
            "interaction": result,
        }
    except Exception as exc:
        db.rollback()
        if claimed_event_id is not None:
            # Validation and ownership errors are permanent; model/network/DB
            # failures remain retryable until the job's attempt budget ends.
            retryable = not isinstance(exc, (ValueError, HTTPException))
            failed = fail_outbox_job(
                db,
                event_id=claimed_event_id,
                worker_id=worker_id,
                error=f"{type(exc).__name__}: {exc}",
                retryable=retryable,
                base_retry_seconds=base_retry_seconds,
            )
            return {
                "job": serialize_outbox_job(failed) if failed is not None else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        raise
    finally:
        db.close()
