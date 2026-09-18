"""Add persistent pgvector profile embeddings and their durable jobs.

Revision ID: 20260916_11
Revises: 20260916_10
Create Date: 2026-09-16
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa


revision: str = "20260916_11"
down_revision: str | None = "20260916_10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        embedding_type = Vector(768)
    else:
        # SQLite remains a unit-test backend. Production retrieval requires
        # PostgreSQL + pgvector, but JSON keeps metadata.create/migrations usable.
        embedding_type = sa.JSON()

    op.create_table(
        "memory_embeddings",
        sa.Column("memory_id", sa.String(length=36), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", embedding_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "dimensions = 768",
            name="ck_memory_embeddings_dimensions",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memory_facts.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("memory_id", "model"),
    )
    op.create_index(
        "ix_memory_embeddings_user_model",
        "memory_embeddings",
        ["user_id", "model"],
    )
    if bind.dialect.name == "postgresql":
        # pgvector's cosine HNSW index accelerates semantic top-k retrieval as
        # the table grows. PostgreSQL may still choose a seq scan for tiny demos.
        op.execute(
            "CREATE INDEX ix_memory_embeddings_hnsw_cosine "
            "ON memory_embeddings USING hnsw (embedding vector_cosine_ops)"
        )

    op.create_table(
        "memory_embedding_jobs",
        sa.Column("memory_id", sa.String(length=36), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "processing",
                "retry",
                "completed",
                "failed",
                "canceled",
                name="memory_embedding_job_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
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
            name="ck_memory_embedding_jobs_attempts",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memory_facts.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("memory_id", "model"),
    )
    op.create_index(
        "ix_memory_embedding_jobs_ready",
        "memory_embedding_jobs",
        ["status", "available_at", "created_at"],
    )
    op.create_index(
        "ix_memory_embedding_jobs_user",
        "memory_embedding_jobs",
        ["user_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_embedding_jobs_user",
        table_name="memory_embedding_jobs",
    )
    op.drop_index(
        "ix_memory_embedding_jobs_ready",
        table_name="memory_embedding_jobs",
    )
    op.drop_table("memory_embedding_jobs")
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_memory_embeddings_hnsw_cosine")
    op.drop_index(
        "ix_memory_embeddings_user_model",
        table_name="memory_embeddings",
    )
    op.drop_table("memory_embeddings")
    # Do not drop the shared vector extension on downgrade.
