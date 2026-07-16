"""Add cluster columns to profiles table (Feature 8: Cluster Import).

Extends profiles to carry imported cluster definitions: source_kind
discriminates manual vs cluster rows, cluster_meta holds the full original
mistudio.cluster-definition/v1 document (lossless re-export), intensity is
the current lambda dial, and sensing_enabled is consumed by Feature 11.

Revision ID: 007
Revises: 006
Create Date: 2026-07-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "profiles",
        sa.Column("source_kind", sa.String(20), server_default="manual", nullable=False),
    )
    op.add_column(
        "profiles",
        sa.Column(
            "cluster_meta",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )
    op.add_column(
        "profiles",
        sa.Column("intensity", sa.Float(), server_default="1.0", nullable=False),
    )
    op.add_column(
        "profiles",
        sa.Column("sensing_enabled", sa.Boolean(), server_default="false", nullable=False),
    )
    op.create_index("idx_profiles_source_kind", "profiles", ["source_kind"])


def downgrade() -> None:
    op.drop_index("idx_profiles_source_kind", table_name="profiles")
    op.drop_column("profiles", "sensing_enabled")
    op.drop_column("profiles", "intensity")
    op.drop_column("profiles", "cluster_meta")
    op.drop_column("profiles", "source_kind")
