"""Create the initial versioned memory schema.

Revision ID: 20260910_01
Revises:
Create Date: 2026-09-10
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260910_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=True),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_events_user_id", "memory_events", ["user_id"])
    op.create_index(
        "ix_memory_events_session_id", "memory_events", ["session_id"]
    )
    op.create_index(
        "ix_memory_events_user_occurred",
        "memory_events",
        ["user_id", "occurred_at"],
    )

    op.create_table(
        "session_states",
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("state_json", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "version > 0", name="ck_session_states_positive_version"
        ),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("ix_session_states_user_id", "session_states", ["user_id"])
    op.create_index(
        "ix_session_states_user_expires",
        "session_states",
        ["user_id", "expires_at"],
    )

    op.create_table(
        "memory_facts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("value_json", sa.JSON(), nullable=False),
        sa.Column(
            "sensitivity",
            sa.Enum(
                "normal",
                "personal",
                "health",
                "emergency_contact",
                name="memory_sensitivity",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "verification_status",
            sa.Enum(
                "unverified",
                "user_confirmed",
                "caregiver_confirmed",
                "system_verified",
                name="memory_verification_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "active",
                "superseded",
                "expired",
                "revoked",
                name="memory_fact_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("source_event_id", sa.String(length=36), nullable=True),
        sa.Column("supersedes_id", sa.String(length=36), nullable=True),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_facts_confidence_range",
        ),
        sa.ForeignKeyConstraint(
            ["source_event_id"], ["memory_events.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_id"], ["memory_facts.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_facts_user_id", "memory_facts", ["user_id"])
    op.create_index(
        "ix_memory_facts_user_status",
        "memory_facts",
        ["user_id", "status"],
    )
    op.create_index(
        "uq_memory_facts_active_user_category_key",
        "memory_facts",
        ["user_id", "category", "key"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_table("memory_facts")
    op.drop_table("session_states")
    op.drop_table("memory_events")
