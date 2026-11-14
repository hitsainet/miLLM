"""A re-run batch row cannot record its sensing or circuit-edge events twice (026 F-1).

T-70 (migration 019) made "a batch row re-run after a crash records no second probe event" a
database fact with a partial unique index on probe events. Sensing and circuit-edge sensing
events got the batch marking in migration 020 but no such guard, so a row whose chunk died
after its events were written but before its state was recorded (the exactly-once window the
runner re-runs) recorded every event again, and an operator's per-line counts were doubled.

The repositories now skip every event for a (owner, batch, line) that already has events — the
re-run is the same line, and a line's events are written together in one flush. These partial
unique indexes are the backstop for the identical rows: live rows (batch_id NULL) are untouched.

Revision ID: 023
Revises: 022
Create Date: 2026-10-16
"""

from alembic import op
import sqlalchemy as sa

revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None

_INDEXES = (
    ("uq_sensing_events_batch_line", "sensing_events",
     ["profile_id", "batch_id", "batch_line", "phase", "pos_start", "pos_end"]),
    ("uq_circuit_edge_events_batch_line", "circuit_edge_sensing_events",
     ["circuit_id", "batch_id", "batch_line", "phase", "edge_key", "up_pos", "down_pos"]),
)


def upgrade() -> None:
    for name, table, columns in _INDEXES:
        op.create_index(
            name, table, columns, unique=True,
            postgresql_where=sa.text("batch_id IS NOT NULL"),
            sqlite_where=sa.text("batch_id IS NOT NULL"),
        )


def downgrade() -> None:
    for name, table, _columns in reversed(_INDEXES):
        op.drop_index(name, table_name=table)
