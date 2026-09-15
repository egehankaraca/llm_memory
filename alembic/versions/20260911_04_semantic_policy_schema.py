"""Add semantic policy metadata and sensitivity domains.

Revision ID: 20260911_04
Revises: 20260911_03
Create Date: 2026-09-11
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260911_04"
down_revision: str | None = "20260911_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


EXPANDED_VALUES = (
    "'normal', 'personal', 'health', 'emergency_contact', "
    "'location', 'financial', 'credential'"
)
ORIGINAL_VALUES = "'normal', 'personal', 'health', 'emergency_contact'"


def upgrade() -> None:
    op.drop_constraint("memory_sensitivity", "memory_facts", type_="check")
    op.create_check_constraint(
        "memory_sensitivity",
        "memory_facts",
        f"sensitivity IN ({EXPANDED_VALUES})",
    )
    op.drop_constraint("candidate_sensitivity", "memory_candidates", type_="check")
    op.create_check_constraint(
        "candidate_sensitivity",
        "memory_candidates",
        f"sensitivity IN ({EXPANDED_VALUES})",
    )
    op.add_column(
        "memory_candidates",
        sa.Column("analysis_json", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.execute(
        "UPDATE memory_facts SET sensitivity = 'personal' "
        "WHERE sensitivity IN ('location', 'financial', 'credential')"
    )
    op.execute(
        "UPDATE memory_candidates SET sensitivity = 'personal' "
        "WHERE sensitivity IN ('location', 'financial', 'credential')"
    )
    op.drop_column("memory_candidates", "analysis_json")
    op.drop_constraint("candidate_sensitivity", "memory_candidates", type_="check")
    op.create_check_constraint(
        "candidate_sensitivity",
        "memory_candidates",
        f"sensitivity IN ({ORIGINAL_VALUES})",
    )
    op.drop_constraint("memory_sensitivity", "memory_facts", type_="check")
    op.create_check_constraint(
        "memory_sensitivity",
        "memory_facts",
        f"sensitivity IN ({ORIGINAL_VALUES})",
    )
