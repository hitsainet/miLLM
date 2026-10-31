"""ENH-008 D1: miLLM's own approval queue, `approval_requests`.

An AI agent's call to a gated route is stored here and waits for the operator
(`millm.api.approval_gate`, `millm.services.approval_service`). One table, nothing else changed.

The CHECK lists are written out LITERALLY, never imported from the model: a migration is a record
of what was applied, and must not change when the model's tuples do.
`tests/schema/test_approvals_migration.py` pins these lists equal to
`millm.db.models.approval.APPROVAL_STATUSES` and `APPROVAL_ACTION_NAMES`.

Downgrade drops the table (and with it every recorded decision).

Revision ID: 037
Revises: 036
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None

STATUSES = ("pending", "executing", "executed", "failed", "rejected", "expired")
ACTIONS = ("model_load", "model_unload", "model_delete",
           "probe_limits_write", "capacity_measure", "code_rule_import")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    op.create_table(
        "approval_requests",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("executor_key", sa.String(200), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("secret_payload", sa.JSON(), nullable=True),
        sa.Column("request_digest", sa.String(64), nullable=False),
        sa.Column("requested_by", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("decided_by", sa.String(64), nullable=True),
        sa.Column("reason", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(_in("status", STATUSES), name="ck_approval_requests_status"),
        sa.CheckConstraint(_in("action", ACTIONS), name="ck_approval_requests_action"),
    )
    op.create_index("ix_approval_requests_status_created_at", "approval_requests",
                    ["status", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_approval_requests_status_created_at", table_name="approval_requests")
    op.drop_table("approval_requests")
