"""Batch-marked sensing and circuit-edge sensing events (Feature 26, FTASKS 0.4).

The 0.4 spike's answer, read from the retention code: both event tables keep the newest
SENSING_MAX_EVENTS_PER_CLUSTER / CIRCUIT_SENSING_MAX_EVENTS_PER_CIRCUIT (1,000) per profile or
circuit, counted across ALL rows, and a request records up to 20 events. A batch generation row
flows through the synchronous path and records sensing events as live traffic, so ~50 firing
rows of a 5,000-row batch would evict an operator's entire live history. T-70's marking is
therefore applied here too: `origin` (server default `live` — true of every existing row, since
no batch API existed before migration 019), `batch_id`, `batch_line`, and an index for the
per-origin cap.

Revision ID: 020
Revises: 019
Create Date: 2026-10-07
"""

from alembic import op
import sqlalchemy as sa

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None

_TABLES = (
    ("sensing_events", "idx_sensing_events_profile_origin_created", "profile_id"),
    ("circuit_edge_sensing_events", "idx_circuit_edge_events_circuit_origin_created", "circuit_id"),
)


def upgrade() -> None:
    for table, index, owner in _TABLES:
        op.add_column(
            table, sa.Column("origin", sa.String(length=8), nullable=False, server_default="live")
        )
        op.add_column(table, sa.Column("batch_id", sa.String(length=32), nullable=True))
        op.add_column(table, sa.Column("batch_line", sa.Integer(), nullable=True))
        op.create_index(index, table, [owner, "origin", "created_at"])


def downgrade() -> None:
    for table, index, _owner in reversed(_TABLES):
        op.drop_index(index, table_name=table)
        op.drop_column(table, "batch_line")
        op.drop_column(table, "batch_id")
        op.drop_column(table, "origin")
