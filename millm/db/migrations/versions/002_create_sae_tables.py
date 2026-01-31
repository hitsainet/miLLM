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
    # Create SAE status enum using DO block for safety
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE saestatus AS ENUM ('downloading', 'cached', 'attached', 'error');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create SAEs table using raw SQL
    op.execute("""
        CREATE TABLE saes (
            id VARCHAR(100) PRIMARY KEY,
            repository_id VARCHAR(255) NOT NULL,
            revision VARCHAR(100) NOT NULL DEFAULT 'main',
            name VARCHAR(255) NOT NULL,
            format VARCHAR(50) NOT NULL DEFAULT 'saelens',
            d_in INTEGER NOT NULL,
            d_sae INTEGER NOT NULL,
            trained_on VARCHAR(255),
            trained_layer INTEGER,
            file_size_bytes BIGINT,
            cache_path VARCHAR(500) NOT NULL,
            status saestatus NOT NULL DEFAULT 'cached',
            error_message TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    # Create indexes for SAEs table
    op.execute("CREATE INDEX idx_saes_repository_id ON saes (repository_id)")
    op.execute("CREATE INDEX idx_saes_status ON saes (status)")

    # Create SAE attachments table using raw SQL
    op.execute("""
        CREATE TABLE sae_attachments (
            id SERIAL PRIMARY KEY,
            sae_id VARCHAR(100) NOT NULL REFERENCES saes(id) ON DELETE CASCADE,
            model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
            layer INTEGER NOT NULL,
            attached_at TIMESTAMP NOT NULL DEFAULT NOW(),
            detached_at TIMESTAMP,
            memory_usage_mb INTEGER,
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        )
    """)

    # Create indexes for SAE attachments table
    op.execute("CREATE INDEX idx_sae_attachments_sae_id ON sae_attachments (sae_id)")

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
    # Drop tables (cascades indexes)
    op.execute("DROP TABLE IF EXISTS sae_attachments CASCADE")
    op.execute("DROP TABLE IF EXISTS saes CASCADE")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS saestatus")
