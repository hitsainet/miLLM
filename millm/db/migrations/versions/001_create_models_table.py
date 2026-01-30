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
    # Create enum types
    op.execute("CREATE TYPE modelsource AS ENUM ('huggingface', 'local')")
    op.execute("CREATE TYPE quantizationtype AS ENUM ('Q4', 'Q8', 'FP16')")
    op.execute(
        "CREATE TYPE modelstatus AS ENUM ('downloading', 'ready', 'loading', 'loaded', 'error')"
    )

    # Create models table
    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "source",
            sa.Enum("huggingface", "local", name="modelsource", create_type=False),
            nullable=False,
        ),
        sa.Column("repo_id", sa.String(255), nullable=True),
        sa.Column("local_path", sa.String(500), nullable=True),
        sa.Column("params", sa.String(50), nullable=True),
        sa.Column("architecture", sa.String(100), nullable=True),
        sa.Column(
            "quantization",
            sa.Enum("Q4", "Q8", "FP16", name="quantizationtype", create_type=False),
            nullable=False,
        ),
        sa.Column("disk_size_mb", sa.Integer(), nullable=True),
        sa.Column("estimated_memory_mb", sa.Integer(), nullable=True),
        sa.Column("cache_path", sa.String(500), nullable=False),
        sa.Column("config_json", JSONB(), nullable=True),
        sa.Column("trust_remote_code", sa.Boolean(), default=False, nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "downloading",
                "ready",
                "loading",
                "loaded",
                "error",
                name="modelstatus",
                create_type=False,
            ),
            default="ready",
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("loaded_at", sa.DateTime(), nullable=True),
    )

    # Create unique constraints
    op.create_unique_constraint(
        "uq_repo_quantization", "models", ["repo_id", "quantization"]
    )
    op.create_unique_constraint("uq_local_path", "models", ["local_path"])

    # Create indexes for common queries
    op.create_index("idx_models_status", "models", ["status"])
    op.create_index("idx_models_repo_id", "models", ["repo_id"])
    op.create_index("idx_models_source", "models", ["source"])


def downgrade() -> None:
    """Drop the models table and enum types."""
    # Drop indexes
    op.drop_index("idx_models_source")
    op.drop_index("idx_models_repo_id")
    op.drop_index("idx_models_status")

    # Drop constraints
    op.drop_constraint("uq_local_path", "models", type_="unique")
    op.drop_constraint("uq_repo_quantization", "models", type_="unique")

    # Drop table
    op.drop_table("models")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS modelstatus")
    op.execute("DROP TYPE IF EXISTS quantizationtype")
    op.execute("DROP TYPE IF EXISTS modelsource")
