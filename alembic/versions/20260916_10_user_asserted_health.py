"""Distinguish a direct user assertion from explicit confirmation.

Revision ID: 20260916_10
Revises: 20260915_09
Create Date: 2026-09-16
"""

from collections.abc import Sequence

from alembic import op


revision: str = "20260916_10"
down_revision: str | None = "20260915_09"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


MEMORY_VALUES = (
    "'unverified', 'user_asserted', 'user_confirmed', "
    "'caregiver_confirmed', 'system_verified'"
)
OLD_MEMORY_VALUES = (
    "'unverified', 'user_confirmed', 'caregiver_confirmed', 'system_verified'"
)


def replace_verification_constraint(
    table_name: str,
    constraint_name: str,
    values: str,
) -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table(table_name) as batch:
            batch.drop_constraint(constraint_name, type_="check")
            batch.create_check_constraint(
                constraint_name,
                f"verification_status IN ({values})",
            )
        return
    op.drop_constraint(constraint_name, table_name, type_="check")
    op.create_check_constraint(
        constraint_name,
        table_name,
        f"verification_status IN ({values})",
    )


def upgrade() -> None:
    replace_verification_constraint(
        "memory_facts",
        "memory_verification_status",
        MEMORY_VALUES,
    )
    replace_verification_constraint(
        "temporary_memories",
        "temporary_memory_verification_status",
        MEMORY_VALUES,
    )


def downgrade() -> None:
    op.execute(
        "UPDATE memory_facts SET verification_status = 'unverified' "
        "WHERE verification_status = 'user_asserted'"
    )
    op.execute(
        "UPDATE temporary_memories SET verification_status = 'unverified' "
        "WHERE verification_status = 'user_asserted'"
    )
    replace_verification_constraint(
        "memory_facts",
        "memory_verification_status",
        OLD_MEMORY_VALUES,
    )
    replace_verification_constraint(
        "temporary_memories",
        "temporary_memory_verification_status",
        OLD_MEMORY_VALUES,
    )
