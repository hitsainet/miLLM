"""Add GGUF file selection to models, and widen the uniqueness constraint.

Revision ID: 014
Revises: 013
Create Date: 2026-09-07

A GGUF repository holds many MUTUALLY EXCLUSIVE quantizations of one model, and
the `quantizationtype` enum is far too coarse to tell them apart: Q4_K_M, Q4_K_S
and Q4_0 are all "Q4". Under the old `UNIQUE (repo_id, quantization)` only one of
them could exist as a row, and they all resolved to the same cache directory, so
deleting one destroyed the others.

`gguf_label` is NOT NULL with a '' default, deliberately. Postgres treats NULLs
as DISTINCT in a unique index, so a nullable third column would stop the
constraint deduplicating ordinary whole-repo downloads — quietly permitting the
duplicate rows it exists to prevent. '' means "the whole repo", which is exactly
what every existing row is.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column(
            "gguf_label",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
    )
    op.add_column("models", sa.Column("gguf_files", JSONB(), nullable=True))
    op.add_column("models", sa.Column("revision", sa.String(length=100), nullable=True))

    # Replace the constraint rather than adding a second one: leaving the old
    # two-column constraint in place would keep rejecting a second quantization
    # from the same repo, which is the entire point of this migration.
    op.drop_constraint("uq_repo_quantization", "models", type_="unique")
    op.create_unique_constraint(
        "uq_repo_quantization",
        "models",
        ["repo_id", "quantization", "gguf_label"],
    )


def downgrade() -> None:
    # Downgrading collapses the new dimension, so rows that differ ONLY by
    # gguf_label would violate the narrower constraint. Fail loudly here rather
    # than letting the constraint creation fail halfway with a confusing error:
    # the operator must decide which quantization to keep.
    conn = op.get_bind()
    duplicates = conn.execute(
        sa.text(
            """
            SELECT repo_id, quantization, COUNT(*) AS n
            FROM models
            WHERE repo_id IS NOT NULL
            GROUP BY repo_id, quantization
            HAVING COUNT(*) > 1
            """
        )
    ).fetchall()
    if duplicates:
        listing = ", ".join(f"{r[0]}/{r[1]} x{r[2]}" for r in duplicates)
        raise RuntimeError(
            "Cannot downgrade: these (repo_id, quantization) pairs hold more than "
            f"one GGUF quantization and would collide: {listing}. "
            "Delete the unwanted models first."
        )

    op.drop_constraint("uq_repo_quantization", "models", type_="unique")
    op.create_unique_constraint(
        "uq_repo_quantization",
        "models",
        ["repo_id", "quantization"],
    )
    op.drop_column("models", "revision")
    op.drop_column("models", "gguf_files")
    op.drop_column("models", "gguf_label")
