"""Add independently expiring session-scoped memory items.

Revision ID: 20260914_07
Revises: 20260911_06
Create Date: 2026-09-14

Existing session_states JSON is intentionally preserved unchanged. New item
storage can coexist with legacy session data without a lossy backfill.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260914_07"
down_revision: str | None = "20260911_06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "temporary_memories",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("value_json", sa.JSON(), nullable=False),
        sa.Column(
            "sensitivity",
            sa.Enum(
                "normal", "personal", "health", "emergency_contact", "location",
                "financial", "credential", name="temporary_memory_sensitivity",
                native_enum=False, create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "verification_status",
            sa.Enum(
                "unverified", "user_confirmed", "caregiver_confirmed", "system_verified",
                name="temporary_memory_verification_status",
                native_enum=False, create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "active", "superseded", "expired", "revoked",
                name="temporary_memory_status", native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("source_event_id", sa.String(length=36), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_temporary_memories_confidence_range",
        ),
        sa.CheckConstraint(
            "expires_at > occurred_at",
            name="ck_temporary_memories_positive_ttl",
        ),
        sa.ForeignKeyConstraint(
            ["source_event_id"], ["memory_events.id"], ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_temporary_memories_active_owner_slot",
        "temporary_memories",
        ["user_id", "session_id", "category", "key"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        sqlite_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "ix_temporary_memories_owner_expiry",
        "temporary_memories", ["user_id", "session_id", "status", "expires_at"],
    )
    op.create_index(
        "ix_temporary_memories_expires", "temporary_memories", ["expires_at"],
    )


def downgrade() -> None:
    op.drop_table("temporary_memories")
