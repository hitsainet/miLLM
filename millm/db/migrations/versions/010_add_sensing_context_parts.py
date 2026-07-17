"""Add sensing_events.context_parts (span highlighting).

Stores the context window as three separately decoded segments
{before, span, after} so the UI can highlight the fired span (the prime
token) — token boundaries don't map to character offsets in the joined
context_text, so the split must happen at decode time.

Revision ID: 010
Revises: 009
Create Date: 2026-07-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sensing_events",
        sa.Column(
            "context_parts",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("sensing_events", "context_parts")
