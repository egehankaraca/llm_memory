"""Link assistant messages to the user turn that produced them.

Revision ID: 20260915_08
Revises: 20260914_07
Create Date: 2026-09-15
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260915_08"
down_revision: str | None = "20260914_07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversation_messages",
        sa.Column("parent_message_id", sa.String(length=36), nullable=True),
    )
    op.create_foreign_key(
        "fk_conversation_messages_parent",
        "conversation_messages",
        "conversation_messages",
        ["parent_message_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_conversation_messages_parent_message_id",
        "conversation_messages",
        ["parent_message_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_messages_parent_message_id",
        table_name="conversation_messages",
    )
    op.drop_constraint(
        "fk_conversation_messages_parent",
        "conversation_messages",
        type_="foreignkey",
    )
    op.drop_column("conversation_messages", "parent_message_id")
