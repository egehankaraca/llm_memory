"""Add bounded short-term conversation messages.

Revision ID: 20260911_05
Revises: 20260911_04
Create Date: 2026-09-11
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260911_05"
down_revision: str | None = "20260911_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column(
            "role",
            sa.Enum(
                "user",
                "assistant",
                name="conversation_message_role",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("estimated_tokens", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "estimated_tokens > 0",
            name="ck_conversation_messages_positive_tokens",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversation_messages_user_id",
        "conversation_messages",
        ["user_id"],
    )
    op.create_index(
        "ix_conversation_messages_session_id",
        "conversation_messages",
        ["session_id"],
    )
    op.create_index(
        "ix_conversation_messages_window",
        "conversation_messages",
        ["user_id", "session_id", "occurred_at"],
    )
    op.create_index(
        "ix_conversation_messages_expires",
        "conversation_messages",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_table("conversation_messages")
