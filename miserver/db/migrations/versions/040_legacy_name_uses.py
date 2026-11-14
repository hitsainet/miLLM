"""040 (ENH-009 rebrand, FR-040.10; FTDD §4.1; TID §4): `legacy_name_uses`, the persistent record of
old (miLLM) names still in use, so the health answer's `last_seen` survives a restart and P7 can retire
an alias on evidence.

One row per `(kind, old_name)`: `env` (an old environment name read through its fallback) or `host`
(a request under an old hostname), plus one `meta` row whose `first_seen_at` is when observation
began (`legacy_names_meta.observing_since`). Names, times and a count only: never a value, a client
address, a path or request content. A few dozen rows ever; no index beyond the primary key.

Downgrade drops the table; nothing else reads it.

Revision ID: 040
Revises: 039
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None

TABLE = "legacy_name_uses"


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("kind", sa.String(8), primary_key=True),
        sa.Column("old_name", sa.String(255), primary_key=True),
        sa.Column("new_name", sa.String(255), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("count", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.execute(sa.text(
        f"INSERT INTO {TABLE} (kind, old_name, new_name, first_seen_at, last_seen_at, count) "
        "VALUES ('meta', 'observing_since', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)"))


def downgrade() -> None:
    op.drop_table(TABLE)
