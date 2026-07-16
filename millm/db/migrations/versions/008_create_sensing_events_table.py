"""Create sensing_events table (Feature 11: Co-Activation Sensing).

Persists bounded per-request cluster co-activation events: which members
fired together, where in the sequence (debounced token span), the ±K token
context window, and a human-readable summary. Rows cascade with their
profile and are pruned by per-cluster cap + age on flush/read.

Revision ID: 008
Revises: 007
Create Date: 2026-07-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sensing_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "profile_id",
            sa.String(50),
            sa.ForeignKey("profiles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("phase", sa.String(10), nullable=False),  # 'prefill' | 'decode'
        sa.Column("pos_start", sa.Integer(), nullable=False),
        sa.Column("pos_end", sa.Integer(), nullable=False),
        sa.Column(
            "fired_members",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("fired_count", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("ambient_fired_count", sa.Integer(), nullable=True),
        sa.Column("context_text", sa.Text(), nullable=True),
        sa.Column(
            "context_token_ids",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("summary", sa.String(300), nullable=False),
        sa.Column("truncated", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_sensing_events_profile_created",
        "sensing_events",
        ["profile_id", "created_at"],
    )
    op.create_index(
        "idx_sensing_events_request", "sensing_events", ["request_id"]
    )


def downgrade() -> None:
    op.drop_index("idx_sensing_events_request", table_name="sensing_events")
    op.drop_index(
        "idx_sensing_events_profile_created", table_name="sensing_events"
    )
    op.drop_table("sensing_events")
