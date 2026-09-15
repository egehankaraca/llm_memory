"""Add a durable asynchronous memory-analysis outbox.

Revision ID: 20260915_09
Revises: 20260915_08
Create Date: 2026-09-15
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260915_09"
down_revision: str | None = "20260915_08"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_outbox",
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "processing",
                "retry",
                "completed",
                "failed",
                "canceled",
                name="memory_outbox_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("analysis_context_json", sa.JSON(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=128), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "attempt_count >= 0 AND max_attempts > 0",
            name="ck_memory_outbox_attempts",
        ),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["memory_events.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index(
        "ix_memory_outbox_user_id",
        "memory_outbox",
        ["user_id"],
    )
    op.create_index(
        "ix_memory_outbox_session_id",
        "memory_outbox",
        ["session_id"],
    )
    op.create_index(
        "ix_memory_outbox_ready",
        "memory_outbox",
        ["status", "available_at", "created_at"],
    )
    op.create_index(
        "ix_memory_outbox_session_order",
        "memory_outbox",
        ["user_id", "session_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_outbox_session_order", table_name="memory_outbox")
    op.drop_index("ix_memory_outbox_ready", table_name="memory_outbox")
    op.drop_index("ix_memory_outbox_session_id", table_name="memory_outbox")
    op.drop_index("ix_memory_outbox_user_id", table_name="memory_outbox")
    op.drop_table("memory_outbox")
