"""Create profiles table

Revision ID: 003
Revises: 002
Create Date: 2026-01-30

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the profiles table for steering configuration persistence."""
    op.create_table(
        "profiles",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("model_id", sa.String(100), nullable=True),
        sa.Column("sae_id", sa.String(50), nullable=True),
        sa.Column("layer", sa.Integer(), nullable=True),
        sa.Column("steering", JSONB(), nullable=False, server_default="{}"),
        sa.Column("is_active", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )

    # Create index on name for fast lookups
    op.create_index("idx_profiles_name", "profiles", ["name"])

    # Create partial unique index for single active profile
    # This ensures only one profile can be active at a time
    op.execute(
        """
        CREATE UNIQUE INDEX idx_single_active_profile
        ON profiles (is_active)
        WHERE is_active = true
        """
    )


def downgrade() -> None:
    """Drop the profiles table."""
    # Drop partial unique index
    op.execute("DROP INDEX IF EXISTS idx_single_active_profile")

    # Drop name index
    op.drop_index("idx_profiles_name")

    # Drop profiles table
    op.drop_table("profiles")
