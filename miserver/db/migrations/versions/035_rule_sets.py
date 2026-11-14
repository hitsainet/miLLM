"""039 S2: rule sets, behaviour attachments and rule results (FTDD §4.3).

Four new tables, nothing existing changed:

- `rule_sets` — imported miDataworks rule-set documents; the producer's identity digest is unique.
- `behaviour_rule_signals` — one row per behaviour key with any attachment.
- `behaviour_rule_signal_history` — append-only.
- `rule_signal_results` — recorded `kind: "rule"` results. No `Text` column and no string wider
  than 64 (a rule result carries a count, never a value); no foreign key to `rule_sets` (a deleted
  set's results keep its id and digest).

Downgrade drops the four tables. Renumbered from the FTDD's `034` (FTASKS renumbering note: 038
took `032`, the S1 migration is `034`).

Revision ID: 035
Revises: 034
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None

JSONVariant = sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True),
                                                      "postgresql")


def upgrade() -> None:
    op.create_table(
        "rule_sets",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("behaviour_key", sa.String(length=64), nullable=False),
        sa.Column("origin", sa.String(length=16), nullable=False),
        sa.Column("digest", sa.String(length=64), nullable=False),
        sa.Column("document", JSONVariant, nullable=False),
        sa.Column("provenance", JSONVariant, nullable=True),
        sa.Column("vector_status", sa.String(length=16), server_default="passed", nullable=False),
        sa.Column("vector_report", JSONVariant, nullable=True),
        sa.Column("imported_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("uq_rule_sets_digest", "rule_sets", ["digest"], unique=True)
    op.create_index("ix_rule_sets_behaviour_key", "rule_sets", ["behaviour_key", "id"],
                    unique=False)

    op.create_table(
        "behaviour_rule_signals",
        sa.Column("behaviour_key", sa.String(length=64), nullable=False),
        sa.Column("rule_set_ids", JSONVariant, nullable=False),
        sa.Column("signals", JSONVariant, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.PrimaryKeyConstraint("behaviour_key"),
    )

    op.create_table(
        "behaviour_rule_signal_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("behaviour_key", sa.String(length=64), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("action", sa.String(length=24), nullable=False),
        sa.Column("rule_set_ids", JSONVariant, nullable=False),
        sa.Column("signals", JSONVariant, nullable=True),
        sa.Column("by", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_behaviour_rule_signal_history_key", "behaviour_rule_signal_history",
                    ["behaviour_key", "changed_at"], unique=False)

    op.create_table(
        "rule_signal_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("result", JSONVariant, nullable=False),
        sa.Column("result_id", sa.String(length=36), nullable=False),
        sa.Column("trace_id", sa.String(length=32), nullable=True),
        sa.Column("span_id", sa.String(length=16), nullable=True),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome", sa.String(length=12), nullable=False),
        sa.Column("reason", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=12), server_default="final", nullable=False),
        sa.Column("persistence", sa.String(length=14), server_default="persisted",
                  nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("session_id", sa.String(length=24), nullable=True),
        sa.Column("turn_index", sa.Integer(), nullable=True),
        sa.Column("origin", sa.String(length=8), server_default="live", nullable=False),
        sa.Column("batch_id", sa.String(length=32), nullable=True),
        sa.Column("batch_line", sa.Integer(), nullable=True),
        sa.Column("behaviour_key", sa.String(length=64), nullable=True),
        sa.Column("rule_set_id", sa.String(length=24), nullable=False),
        sa.Column("rule_set_digest", sa.String(length=64), nullable=False),
        sa.Column("signal", sa.String(length=64), nullable=False),
        sa.Column("form", sa.String(length=24), nullable=False),
        sa.Column("window", sa.String(length=16), nullable=False),
        sa.Column("fires", sa.Boolean(), nullable=True),
        sa.Column("count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["probe_sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("uq_rule_signal_results_result_id", "rule_signal_results", ["result_id"],
                    unique=True)
    op.create_index("ix_rule_signal_results_request_id", "rule_signal_results", ["request_id"],
                    unique=False)
    op.create_index("ix_rule_signal_results_behaviour_created", "rule_signal_results",
                    ["behaviour_key", "created_at"], unique=False)
    op.create_index("ix_rule_signal_results_signal_origin_created", "rule_signal_results",
                    ["rule_set_id", "signal", "origin", "created_at"], unique=False)
    op.create_index("ix_rule_signal_results_session_turn", "rule_signal_results",
                    ["session_id", "turn_index"], unique=False)
    op.create_index(
        "uq_rule_signal_results_batch_line", "rule_signal_results",
        ["rule_set_id", "signal", "batch_id", "batch_line", "window"], unique=True,
        postgresql_where=sa.text("batch_id IS NOT NULL"),
        sqlite_where=sa.text("batch_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_table("rule_signal_results")
    op.drop_table("behaviour_rule_signal_history")
    op.drop_table("behaviour_rule_signals")
    op.drop_table("rule_sets")
