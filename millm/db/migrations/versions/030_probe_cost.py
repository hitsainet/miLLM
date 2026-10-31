"""What an armed probe costs per forward pass (Feature 37, FR-37.2).

`probes.cost` (JSON, nullable): measured at arm on the probe's layer's card, cleared whenever the
probe disarms. NO BACKFILL: every probe is disarmed by the restart that runs this migration, and a
cost is never trusted across a restart.

Revision ID: 030
Revises: 029
Create Date: 2026-10-21
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "probes",
        sa.Column("cost", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("probes", "cost")
