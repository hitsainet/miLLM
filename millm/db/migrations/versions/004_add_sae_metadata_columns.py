"""Add width and average_l0 columns to saes table.

Revision ID: 004
Revises: 003
Create Date: 2026-01-31
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add width and average_l0 columns to saes table."""
    op.add_column("saes", sa.Column("width", sa.String(50), nullable=True))
    op.add_column("saes", sa.Column("average_l0", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Remove width and average_l0 columns from saes table."""
    op.drop_column("saes", "average_l0")
    op.drop_column("saes", "width")
