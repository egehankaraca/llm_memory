"""Track memory conflict and consolidation decisions.

Revision ID: 20260911_06
Revises: 20260911_05
Create Date: 2026-09-11
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260911_06"
down_revision: str | None = "20260911_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "memory_candidates",
        sa.Column(
            "consolidation_action",
            sa.String(length=50),
            nullable=False,
            server_default="not_applicable",
        ),
    )
    op.alter_column(
        "memory_candidates",
        "consolidation_action",
        server_default=None,
    )
    op.add_column(
        "memory_candidates",
        sa.Column("consolidates_fact_id", sa.String(length=36), nullable=True),
    )
    op.create_foreign_key(
        "fk_memory_candidates_consolidates_fact",
        "memory_candidates",
        "memory_facts",
        ["consolidates_fact_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_memory_candidates_consolidates_fact",
        "memory_candidates",
        type_="foreignkey",
    )
    op.drop_column("memory_candidates", "consolidates_fact_id")
    op.drop_column("memory_candidates", "consolidation_action")
