"""039 S3: operator-code documents beside their rule set (FTDD §4.4; TD39-34; OQ-39.7 (a)).

One nullable column, nothing else changed:

- `rule_sets.code_modules` — `{code_sha256: canonical UTF-8 text}`, one entry per code-rule
  document the import carried, each verified against its hash before it was stored. The
  `rule-runner` worker receives these bytes over its socket and never reads the database.

NULL for every set imported before S3 (and for a set whose code rules arrived without their
documents: those stay pending, `rule_worker_unavailable`). Downgrade drops the column. Renumbered
from the FTDD's `035` (FTASKS renumbering note: S1 is `034`, S2 `035`).

Revision ID: 036
Revises: 035
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None

JSONVariant = sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True),
                                                      "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("rule_sets") as batch:
        batch.add_column(sa.Column("code_modules", JSONVariant, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("rule_sets") as batch:
        batch.drop_column("code_modules")
