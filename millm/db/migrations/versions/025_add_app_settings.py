"""Operator settings stored in the database: `app_settings` (Feature 34).

The first row is the Hugging Face token saved on the Admin UI's Settings page, so the operator need
not paste one into every download. Its value is the AES-256-GCM envelope from
`millm/core/secret_box.py`; the table never holds a secret in plaintext.

No backfill: the table starts empty, which means "nothing saved" — the `HF_TOKEN` environment tier
and anonymous access behave exactly as before this migration.

Revision ID: 025
Revises: 024
Create Date: 2026-10-18
"""

import sqlalchemy as sa
from alembic import op

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verified_user", sa.String(length=255), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
