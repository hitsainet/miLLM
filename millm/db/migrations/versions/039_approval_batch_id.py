"""ENH-006 D79: `approval_requests.batch_id`, set when the operator approves every waiting
settings change at once ("Approve all settings changes").

Each request in such a batch is still approved and recorded on its own (its own status, result,
`decided_by` and `decided_at`); `batch_id` (`bat_<32 hex>`) is the one value they share, so History
can say which requests one approval decided. NULL for a request decided on its own, which is every
row that exists when this migration runs. Nullable, no default, no index: nothing filters on it.

Downgrade drops the column; the requests themselves and their decisions are kept.

Revision ID: 039
Revises: 038
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None

TABLE = "approval_requests"
COLUMN = "batch_id"


def upgrade() -> None:
    op.add_column(TABLE, sa.Column(COLUMN, sa.String(40), nullable=True))


def downgrade() -> None:
    op.drop_column(TABLE, COLUMN)
