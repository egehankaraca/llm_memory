"""Separate memory scope from sensitivity and add episodic storage.

Revision ID: 20260917_12
Revises: 20260916_11
Create Date: 2026-09-17
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260917_12"
down_revision: str | None = "20260916_11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    scope_enum = sa.Enum(
        "profile",
        "episode",
        "session",
        "discard",
        name="candidate_memory_scope",
        native_enum=False,
        create_constraint=True,
    )
    op.add_column(
        "memory_candidates",
        sa.Column(
            "scope",
            scope_enum,
            nullable=False,
            server_default="discard",
        ),
    )
    op.execute(
        "UPDATE memory_candidates SET scope = CASE "
        "WHEN memory_type = 'discard' THEN 'discard' "
        "WHEN memory_type = 'short_term' OR expires_at IS NOT NULL THEN 'session' "
        "ELSE 'profile' END"
    )
    op.alter_column("memory_candidates", "scope", server_default=None)

    op.create_table(
        "memory_episodes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=True),
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
                "location",
                "financial",
                "credential",
                name="memory_episode_sensitivity",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "verification_status",
            sa.Enum(
                "unverified",
                "user_asserted",
                "user_confirmed",
                "caregiver_confirmed",
                "system_verified",
                name="memory_episode_verification_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column("source_event_id", sa.String(length=36), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_episodes_confidence_range",
        ),
        sa.ForeignKeyConstraint(
            ["source_event_id"],
            ["memory_events.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_memory_episodes_user_id",
        "memory_episodes",
        ["user_id"],
    )
    op.create_index(
        "ix_memory_episodes_session_id",
        "memory_episodes",
        ["session_id"],
    )
    op.create_index(
        "ix_memory_episodes_user_occurred",
        "memory_episodes",
        ["user_id", "occurred_at"],
    )
    op.create_index(
        "ix_memory_episodes_user_retention",
        "memory_episodes",
        ["user_id", "retention_until"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_episodes_user_retention",
        table_name="memory_episodes",
    )
    op.drop_index(
        "ix_memory_episodes_user_occurred",
        table_name="memory_episodes",
    )
    op.drop_index("ix_memory_episodes_session_id", table_name="memory_episodes")
    op.drop_index("ix_memory_episodes_user_id", table_name="memory_episodes")
    op.drop_table("memory_episodes")
    op.drop_column("memory_candidates", "scope")
