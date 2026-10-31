"""`app_settings.last_check_error`: a failed check of a saved setting is recorded with it (Feature 34).

Before this, the outcome of a FAILED Hugging Face token test lived only in process memory: a restart
forgot it, so the Settings page could show a rejected token as merely "never verified". The column
holds a short reason (`invalid` / `network`), never the Hub's message and never the token.

No backfill: NULL means "no failed check recorded", which is what every existing row has.

Revision ID: 026
Revises: 025
Create Date: 2026-10-18
"""

import sqlalchemy as sa
from alembic import op

revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "app_settings",
        sa.Column("last_check_error", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "last_check_error")
