"""Create SAE tables

Revision ID: 002
Revises: 001
Create Date: 2026-01-30

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the SAE and SAE attachment tables."""
    # Create SAE status enum
    op.execute("CREATE TYPE saestatus AS ENUM ('downloading', 'cached', 'attached', 'error')")

    # Create SAEs table
    op.create_table(
        "saes",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("repository_id", sa.String(255), nullable=False),
        sa.Column("revision", sa.String(100), nullable=False, server_default="main"),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("format", sa.String(50), nullable=False, server_default="saelens"),
        sa.Column("d_in", sa.Integer(), nullable=False),
        sa.Column("d_sae", sa.Integer(), nullable=False),
        sa.Column("trained_on", sa.String(255), nullable=True),
        sa.Column("trained_layer", sa.Integer(), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("cache_path", sa.String(500), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "downloading",
                "cached",
                "attached",
                "error",
                name="saestatus",
                create_type=False,
            ),
            server_default="cached",
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )

    # Create indexes for SAEs table
    op.create_index("idx_saes_repository_id", "saes", ["repository_id"])
    op.create_index("idx_saes_status", "saes", ["status"])

    # Create SAE attachments table
    op.create_table(
        "sae_attachments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "sae_id",
            sa.String(100),
            sa.ForeignKey("saes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "model_id",
            sa.Integer(),
            sa.ForeignKey("models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("layer", sa.Integer(), nullable=False),
        sa.Column("attached_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("detached_at", sa.DateTime(), nullable=True),
        sa.Column("memory_usage_mb", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    )

    # Create indexes for SAE attachments table
    op.create_index("idx_sae_attachments_sae_id", "sae_attachments", ["sae_id"])

    # Create partial unique index for single active attachment
    # This ensures only one active attachment can exist at a time
    op.execute(
        """
        CREATE UNIQUE INDEX idx_single_active_attachment
        ON sae_attachments (is_active)
        WHERE is_active = true
        """
    )


def downgrade() -> None:
    """Drop the SAE tables and enum types."""
    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_single_active_attachment")
    op.drop_index("idx_sae_attachments_sae_id")

    # Drop SAE attachments table
    op.drop_table("sae_attachments")

    # Drop SAE indexes
    op.drop_index("idx_saes_status")
    op.drop_index("idx_saes_repository_id")

    # Drop SAEs table
    op.drop_table("saes")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS saestatus")
