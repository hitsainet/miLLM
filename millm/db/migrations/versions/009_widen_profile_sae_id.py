"""Widen profiles.sae_id to match SAE.id (Feature 8/10 live-E2E fix).

profiles.sae_id was VARCHAR(50) while sae.id is VARCHAR(100): binding a
profile/cluster to an SAE with a real HF-derived id (e.g.
'mistudio--sae-lfm2.5-1.2b-instruct--layer_12--width_8k--res', 59 chars)
raised StringDataRightTruncationError — every bound cluster import 500'd.

Revision ID: 009
Revises: 008
Create Date: 2026-07-16
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "profiles",
        "sae_id",
        existing_type=sa.String(50),
        type_=sa.String(100),
        existing_nullable=True,
    )


def downgrade() -> None:
    # Only safe when no value exceeds 50 chars; truncation would corrupt ids.
    op.alter_column(
        "profiles",
        "sae_id",
        existing_type=sa.String(100),
        type_=sa.String(50),
        existing_nullable=True,
    )
