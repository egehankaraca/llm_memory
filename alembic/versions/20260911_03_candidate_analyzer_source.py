"""Track which analyzer produced each memory candidate.

Revision ID: 20260911_03
Revises: 20260910_02
Create Date: 2026-09-11
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260911_03"
down_revision: str | None = "20260910_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "memory_candidates",
        sa.Column(
            "analyzer_source",
            sa.String(length=50),
            nullable=False,
            server_default="rules",
        ),
    )
    op.alter_column("memory_candidates", "analyzer_source", server_default=None)


def downgrade() -> None:
    op.drop_column("memory_candidates", "analyzer_source")
