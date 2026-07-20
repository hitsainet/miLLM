"""Add circuits table (Feature 13: Circuit Import).

A circuit is a multi-layer graph over several SAEs — unlike a cluster it is
NOT a profiles row, so it gets its own table. ``circuit_meta`` stores the full
original mistudio.circuit-definition/v1 document verbatim for lossless
re-export; ``rung`` is the circuit-level evidence rung (MIN over edges) cached
for list/filter queries; ``per_sae_warnings`` records the per-referenced-SAE
compatibility verdicts from import.

The partial unique index mirrors ``idx_active_profile``: at most one circuit
may be active at a time.

Revision ID: 011
Revises: 010
Create Date: 2026-07-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "circuits",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        # Full original circuit-definition/v1 document (lossless re-export).
        sa.Column(
            "circuit_meta",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        # Cached evidence rung = MIN over edges (0..3); empty edges ⇒ 0.
        sa.Column("rung", sa.Integer(), server_default="0", nullable=False),
        sa.Column("edge_count", sa.Integer(), server_default="0", nullable=False),
        # Layers the circuit references, e.g. [10, 13].
        sa.Column(
            "layers",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        # Per-referenced-SAE compatibility verdicts recorded at import:
        # [{sae_id, layer, verdict: bind|warn|block|unbound, reason?}]
        sa.Column(
            "per_sae_warnings",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        # True only when EVERY referenced SAE binds (full multi-SAE serving);
        # otherwise activation degrades to the per-layer cluster slice.
        sa.Column("serveable", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="false", nullable=False),
        # "full" | "slice_fallback" | null (not currently serving).
        sa.Column("serving_mode", sa.String(20), nullable=True),
        sa.Column("intensity", sa.Float(), server_default="1.0", nullable=False),
        sa.Column(
            "provenance",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_circuits_name", "circuits", ["name"], unique=True)
    op.create_index("idx_circuits_rung", "circuits", ["rung"])
    # At most one active circuit (mirrors idx_active_profile).
    op.create_index(
        "uq_circuits_active",
        "circuits",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
        sqlite_where=sa.text("is_active = 1"),
    )


def downgrade() -> None:
    op.drop_index("uq_circuits_active", table_name="circuits")
    op.drop_index("idx_circuits_rung", table_name="circuits")
    op.drop_index("idx_circuits_name", table_name="circuits")
    op.drop_table("circuits")
