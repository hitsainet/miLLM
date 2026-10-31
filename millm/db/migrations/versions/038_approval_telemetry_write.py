"""ENH-006 D76: the approval queue accepts `telemetry_write`.

An AI agent's change to a telemetry setting (ENH-006 D75 made them changeable while miLLM runs)
waits in miLLM's approval queue like the six ENH-008 actions. Only the action CHECK constraint
changes: it is dropped and created again with the seventh name. Nothing else in the table moves.

The lists are written out LITERALLY, never imported from the model (a migration records what was
applied). `tests/schema/test_approvals_migration.py` pins `ACTIONS` here equal to
`millm.db.models.approval.APPROVAL_ACTION_NAMES`, and 037's list to the six it shipped with.

Downgrade DELETES every `telemetry_write` request (pending or decided) before restoring 037's
constraint, which would otherwise refuse those rows.

Revision ID: 038
Revises: 037
"""

from __future__ import annotations

from alembic import op

revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None

CONSTRAINT = "ck_approval_requests_action"
PREVIOUS = ("model_load", "model_unload", "model_delete",
            "probe_limits_write", "capacity_measure", "code_rule_import")
ACTIONS = PREVIOUS + ("telemetry_write",)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    op.drop_constraint(CONSTRAINT, "approval_requests", type_="check")
    op.create_check_constraint(CONSTRAINT, "approval_requests", _in("action", ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM approval_requests WHERE action = 'telemetry_write'")
    op.drop_constraint(CONSTRAINT, "approval_requests", type_="check")
    op.create_check_constraint(CONSTRAINT, "approval_requests", _in("action", PREVIOUS))
