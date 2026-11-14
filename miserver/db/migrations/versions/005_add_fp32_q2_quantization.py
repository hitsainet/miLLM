"""Add FP32 and Q2 to quantizationtype enum.

Revision ID: 005
Revises: 004
Create Date: 2026-02-08
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new enum values to the existing PostgreSQL enum type
    op.execute("ALTER TYPE quantizationtype ADD VALUE IF NOT EXISTS 'FP32'")
    op.execute("ALTER TYPE quantizationtype ADD VALUE IF NOT EXISTS 'Q2'")


def downgrade() -> None:
    # PostgreSQL does not support removing values from enum types.
    # To downgrade, ensure no rows use FP32 or Q2, then recreate the type.
    pass
