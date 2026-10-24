"""Create models table

Revision ID: 001
Revises:
Create Date: 2026-01-30

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the models table with all columns and constraints."""
    # Create enum types using DO blocks to check existence first
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE modelsource AS ENUM ('huggingface', 'local');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE quantizationtype AS ENUM ('Q4', 'Q8', 'FP16');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE modelstatus AS ENUM ('downloading', 'ready', 'loading', 'loaded', 'error');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create models table using raw SQL to avoid SQLAlchemy enum recreation issues
    op.execute("""
        CREATE TABLE models (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            source modelsource NOT NULL,
            repo_id VARCHAR(255),
            local_path VARCHAR(500),
            params VARCHAR(50),
            architecture VARCHAR(100),
            quantization quantizationtype NOT NULL,
            disk_size_mb INTEGER,
            estimated_memory_mb INTEGER,
            cache_path VARCHAR(500) NOT NULL,
            config_json JSONB,
            trust_remote_code BOOLEAN NOT NULL DEFAULT FALSE,
            status modelstatus NOT NULL DEFAULT 'ready',
            error_message TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
            loaded_at TIMESTAMP,
            CONSTRAINT uq_repo_quantization UNIQUE (repo_id, quantization),
            CONSTRAINT uq_local_path UNIQUE (local_path)
        )
    """)

    # Create indexes
    op.execute("CREATE INDEX idx_models_status ON models (status)")
    op.execute("CREATE INDEX idx_models_repo_id ON models (repo_id)")
    op.execute("CREATE INDEX idx_models_source ON models (source)")


def downgrade() -> None:
    """Drop the models table and enum types."""
    # Drop table (cascades indexes and constraints)
    op.execute("DROP TABLE IF EXISTS models CASCADE")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS modelstatus")
    op.execute("DROP TYPE IF EXISTS quantizationtype")
    op.execute("DROP TYPE IF EXISTS modelsource")
