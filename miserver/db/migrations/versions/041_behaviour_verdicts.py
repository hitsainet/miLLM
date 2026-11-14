"""041 (ENH-006 G19; operator decisions G19-1 … G19-6, 2026-11-08, revising D9): behaviour verdicts.

- `behaviour_verdicts`: one row per behaviour key with an attached `mistudio.behaviour-verdict/v1`
  definition (as received, with its sha256 and what the attach found);
- `behaviour_verdict_history`: append-only, one row per attach or clear;
- `behaviour_verdict_results`: one row per recorded `kind: "behaviour"` result, pruned with the probe
  events' retention. No `Text` column and no string wider than 64: a verdict carries no text.

Downgrade drops the three tables; nothing else reads them.

Revision ID: 041
Revises: 040
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def _json() -> sa.types.TypeEngine:
    from sqlalchemy.dialects.postgresql import JSONB

    return sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


def upgrade() -> None:
    op.create_table(
        "behaviour_verdicts",
        sa.Column("behaviour_key", sa.String(64), primary_key=True),
        sa.Column("definition", _json(), nullable=False),
        sa.Column("definition_sha256", sa.String(64), nullable=False),
        sa.Column("report_id", sa.String(64), nullable=True),
        sa.Column("attach_report", _json(), nullable=True),
        sa.Column("attached_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("attached_by", sa.String(64), nullable=True),
        sa.Column("reason", sa.String(500), nullable=True),
    )
    op.create_table(
        "behaviour_verdict_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("behaviour_key", sa.String(64), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("definition_sha256", sa.String(64), nullable=True),
        sa.Column("report_id", sa.String(64), nullable=True),
        sa.Column("by", sa.String(64), nullable=True),
        sa.Column("reason", sa.String(500), nullable=True),
    )
    op.create_index("ix_behaviour_verdict_history_key", "behaviour_verdict_history",
                    ["behaviour_key", "changed_at"])
    op.create_table(
        "behaviour_verdict_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("result", _json(), nullable=False),
        sa.Column("result_id", sa.String(36), nullable=False),
        sa.Column("trace_id", sa.String(32), nullable=True),
        sa.Column("span_id", sa.String(16), nullable=True),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome", sa.String(12), nullable=False),
        sa.Column("reason", sa.String(64), nullable=True),
        sa.Column("status", sa.String(12), nullable=False, server_default="final"),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("session_id", sa.String(24), nullable=True),
        sa.Column("origin", sa.String(8), nullable=False, server_default="live"),
        sa.Column("batch_id", sa.String(32), nullable=True),
        sa.Column("batch_line", sa.Integer(), nullable=True),
        sa.Column("behaviour_key", sa.String(64), nullable=False),
        sa.Column("window", sa.String(16), nullable=False),
        sa.Column("fires", sa.Boolean(), nullable=True),
        sa.Column("decided_by", sa.String(24), nullable=False),
        sa.Column("definition_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("uq_behaviour_verdict_results_result_id", "behaviour_verdict_results",
                    ["result_id"], unique=True)
    op.create_index("ix_behaviour_verdict_results_key", "behaviour_verdict_results",
                    ["behaviour_key", "origin", "created_at"])
    op.create_index("ix_behaviour_verdict_results_request", "behaviour_verdict_results",
                    ["request_id"])
    op.create_index("ix_behaviour_verdict_results_created", "behaviour_verdict_results",
                    ["created_at"])


def downgrade() -> None:
    op.drop_table("behaviour_verdict_results")
    op.drop_table("behaviour_verdict_history")
    op.drop_table("behaviour_verdicts")
