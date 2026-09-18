"""Remove the user-confirmation candidate flow.

Revision ID: 20260918_13
Revises: 20260917_12
Create Date: 2026-09-18
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260918_13"
down_revision: str | None = "20260917_12"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Preserve the audit record while collapsing all legacy review outcomes to
    # the two states produced by the current automatic policy.
    op.execute(
        "UPDATE memory_candidates SET status = CASE "
        "WHEN status = 'confirmed' THEN 'auto_applied' ELSE 'ignored' END "
        "WHERE status IN ('pending', 'confirmed', 'rejected')"
    )
    op.drop_constraint(
        "memory_candidate_status", "memory_candidates", type_="check"
    )
    op.create_check_constraint(
        "memory_candidate_status",
        "memory_candidates",
        "status IN ('auto_applied', 'ignored')",
    )
    op.drop_column("memory_candidates", "requires_confirmation")


def downgrade() -> None:
    op.add_column(
        "memory_candidates",
        sa.Column(
            "requires_confirmation",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.alter_column(
        "memory_candidates", "requires_confirmation", server_default=None
    )
    op.drop_constraint(
        "memory_candidate_status", "memory_candidates", type_="check"
    )
    op.create_check_constraint(
        "memory_candidate_status",
        "memory_candidates",
        "status IN ('pending', 'auto_applied', 'confirmed', 'rejected', 'ignored')",
    )
