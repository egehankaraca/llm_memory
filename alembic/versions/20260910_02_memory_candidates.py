"""Add analyzed memory candidates.

Revision ID: 20260910_02
Revises: 20260910_01
Create Date: 2026-09-10
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260910_02"
down_revision: str | None = "20260910_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_candidates",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("source_event_id", sa.String(length=36), nullable=False),
        sa.Column("decision_index", sa.Integer(), nullable=False),
        sa.Column(
            "memory_type",
            sa.Enum(
                "short_term",
                "long_term",
                "sensitive",
                "discard",
                name="candidate_memory_type",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("key", sa.String(length=100), nullable=True),
        sa.Column("value_json", sa.JSON(), nullable=True),
        sa.Column(
            "sensitivity",
            sa.Enum(
                "normal",
                "personal",
                "health",
                "emergency_contact",
                name="candidate_sensitivity",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column("requires_confirmation", sa.Boolean(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "auto_applied",
                "confirmed",
                "rejected",
                "ignored",
                name="memory_candidate_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("reason", sa.String(length=500), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("applied_ref", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_candidates_confidence_range",
        ),
        sa.ForeignKeyConstraint(
            ["source_event_id"],
            ["memory_events.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_event_id",
            "decision_index",
            name="uq_memory_candidates_event_decision",
        ),
    )
    op.create_index(
        "ix_memory_candidates_user_id", "memory_candidates", ["user_id"]
    )
    op.create_index(
        "ix_memory_candidates_session_id", "memory_candidates", ["session_id"]
    )
    op.create_index(
        "ix_memory_candidates_user_status",
        "memory_candidates",
        ["user_id", "status", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("memory_candidates")
