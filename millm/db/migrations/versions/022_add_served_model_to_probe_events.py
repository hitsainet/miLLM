"""Which model produced the residual a probe event read (033 D-14, addendum D-36).

A probe armed on a base model keeps scoring while an adapter is applied: the residual it reads
then comes from a model it was never fitted on, so the verdict is `provisional` and carries the
served name (the adapter's own name). That reached the per-request verdict and stopped there; the
event row, which is what the operator's history and the event list show, could not say which
model a past verdict was about. This column records it.

NULLABLE WITH NO DEFAULT, AND NO BACKFILL: NULL means "not recorded", which is true of every row
written before this migration. Backfilling the base's name would claim a fact nobody observed,
and for events recorded while an adapter was applied it would be the wrong fact.

String(255), the width of `models.name`: the served name is a row name, a checkpoint manifest's
name or an adapter manifest's name, and the longest of those a row can carry is 255.

Revision ID: 022
Revises: 021
Create Date: 2026-10-14
"""

from alembic import op
import sqlalchemy as sa

revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("probe_events", sa.Column("served_model", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("probe_events", "served_model")
