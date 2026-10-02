"""A probe's decision bar can now move in place, so record which cut judged what.

miStudio can re-cut a probe's threshold in milliseconds — a threshold is the (1 - target_fpr)
quantile of negatives it already has on disk — and that bar can now reach an ALREADY IMPORTED,
possibly ARMED probe without a disarm/delete/re-import cycle, which would destroy the probe's
whole event history.

⚠ MOVING A BAR IS NOT REPLACING A DETECTOR, AND THAT DISTINCTION IS WHAT THESE COLUMNS PROTECT.
`on_conflict=replace` is still refused, for the reason it always was: overwriting a definition in
place while a probe is armed changes the detector underneath a running monitor while every event
keeps the same `probe_id`, so the history would describe two detectors as one. That objection turns
on `probe_events.score` having no record of WHICH detector produced it. A bar is different: the
event row already records the threshold it was judged against. What it could not record was which
CUT that was — needed when two cuts land on the same number, and when a `provisional` marker
disappears because a window gained its own bar rather than because it never needed one.

Columns:

* ``probes.threshold_revision`` — 1 is the bar the producer's run cut.
* ``probes.threshold_calibration_id`` — the producer's identity for that cut, when it sent one.
  NULL is "not stated", which is not the same as "never re-cut".
* ``probes.threshold_history`` — every bar served, append-only, so an old revision stays
  answerable after the current one has moved on.
* ``probes.armed_threshold_revision`` — which bar was in force when the operator armed it.
* ``probe_events.threshold_revision`` — which cut judged this verdict.

⚠ THE DEFAULTS OF 1 ARE TRUE OF THE ROWS THEY LAND ON. No probe has ever been re-cut, so every
existing probe is at revision 1 and every existing event really was judged under it. That is the
test 017's docstring sets — not "is there a sensible default" but "was the default true of the rows
it is about to be written onto". Here it is.

``armed_threshold_revision`` is NULLABLE for the opposite reason: a probe armed before this column
existed genuinely was armed against a bar nobody recorded, and defaulting it to 1 would claim a
fact that was never observed. That is the ``chat_format NOT NULL DEFAULT 'auto'`` mistake, which
added a column to stop a silent misattribution and introduced one.

Revision ID: 018
Revises: 017
Create Date: 2026-10-02
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None

JSONVariant = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "probes",
        sa.Column("threshold_revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "probes",
        sa.Column("threshold_calibration_id", sa.String(length=64), nullable=True),
    )
    op.add_column("probes", sa.Column("threshold_history", JSONVariant, nullable=True))
    op.add_column("probes", sa.Column("armed_threshold_revision", sa.Integer(), nullable=True))
    op.add_column(
        "probe_events",
        sa.Column("threshold_revision", sa.Integer(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("probe_events", "threshold_revision")
    op.drop_column("probes", "armed_threshold_revision")
    op.drop_column("probes", "threshold_history")
    op.drop_column("probes", "threshold_calibration_id")
    op.drop_column("probes", "threshold_revision")
