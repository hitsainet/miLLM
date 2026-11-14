"""Add probe monitors (Feature 24: Probe Monitor Runtime).

A probe is a small linear readout trained in miStudio and imported here as a self-describing
``mistudio.probe-definition/v1`` document. The whole document is stored in ``definition``; the
typed columns beside it are a PROJECTION used for queries and for the arm-time identity check,
not a second source of truth. Anything additive a newer miStudio emits survives in ``definition``
because the mirror allows extra fields.

``probe_events`` is its own table rather than a reuse of ``sensing_events``: sensing events are
keyed to a cluster profile and carry co-activation members, which a probe has none of. Bending one
table to hold both would mean half its columns are null for either kind.

⚠ ``rung`` IS DENORMALISED ONTO EVERY EVENT, deliberately, exactly as ``circuit_edge_sensing_event``
denormalises ``edge_rung``. A probe's rung can rise (a re-import after more evaluations, a judge
run), and an event must keep describing the evidence that was true WHEN IT WAS OBSERVED. Reading
today's rung against a month-old event would retroactively upgrade what the probe claimed at the
time — the precise overclaim the evidence ladder exists to prevent.

Revision ID: 016
Revises: 015
Create Date: 2026-09-27
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite test DBs).
JSONVariant = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "probes",
        sa.Column("id", sa.String(24), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("definition", JSONVariant, nullable=False),
        # --- identity: every one of these is compared at arm time (FR-24.3) -----------
        sa.Column("hf_id", sa.String(255), nullable=False),
        sa.Column("revision", sa.String(100), nullable=True),
        sa.Column("d_model", sa.Integer(), nullable=False),
        sa.Column("n_layers", sa.Integer(), nullable=False),
        sa.Column("template_sha256", sa.String(64), nullable=True),
        # --- read point and readout ---------------------------------------------------
        sa.Column("layer", sa.Integer(), nullable=False),
        sa.Column("rule", sa.String(32), nullable=False),
        sa.Column("streamable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("scope", sa.String(32), nullable=False),
        sa.Column("basis", sa.String(32), nullable=False, server_default="residual"),
        sa.Column("sae_ref", JSONVariant, nullable=True),
        # --- decision -----------------------------------------------------------------
        # `threshold` is nullable and that is meaningful: NULL means no threshold was placed, so
        # the probe ranks but does not decide. It is not a threshold of 0.
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("target_fpr", sa.Float(), nullable=True),
        # --- evidence -----------------------------------------------------------------
        sa.Column("rung", sa.Integer(), nullable=False),
        sa.Column("definition_acknowledgement", JSONVariant, nullable=True),
        sa.Column("arm_acknowledgement", JSONVariant, nullable=True),
        # --- runtime ------------------------------------------------------------------
        sa.Column("parity", JSONVariant, nullable=True),
        sa.Column("armed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("paused_reason", sa.String(64), nullable=True),
        sa.Column("provenance", JSONVariant, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_probes_name", "probes", ["name"], unique=True)
    op.create_index("ix_probes_armed", "probes", ["armed"])
    # ⚠ NO INDEX ON sae_ref->>'hf_repo'. The FTDD asked for one, for the missing-SAE lookup
    # (FR-24.15). It is not worth it and it is not free: a JSONB expression index exists only on
    # PostgreSQL, so the ORM and the migrations would describe different schemas on SQLite — which
    # the drift ratchet correctly refused. The lookup it serves scans a table bounded by
    # PROBE_MAX_ARMED (8) armed probes and realistically a few dozen imported ones; a sequential
    # scan over that is free, and buying nothing with a cross-dialect divergence is a bad trade.

    op.create_table(
        "probe_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "probe_id",
            sa.String(24),
            sa.ForeignKey("probes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # The `/v1` completion id, which is the ONLY link between a response and its verdict.
        sa.Column("request_id", sa.String(64), nullable=True),
        # `scored=False` always carries a reason. A probe never goes silently quiet (BR-006).
        sa.Column("scored", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("not_scored_reason", sa.String(64), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("verdict", sa.Boolean(), nullable=True),
        sa.Column("rung", sa.Integer(), nullable=True),
        sa.Column("top_positions", JSONVariant, nullable=True),
        sa.Column("n_scored_tokens", sa.Integer(), nullable=True),
        sa.Column("context_text", sa.Text(), nullable=True),
        sa.Column("context_token_ids", JSONVariant, nullable=True),
        sa.Column("summary", sa.String(300), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_probe_events_probe_created", "probe_events", ["probe_id", "created_at"])
    op.create_index("ix_probe_events_request_id", "probe_events", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_probe_events_request_id", table_name="probe_events")
    op.drop_index("ix_probe_events_probe_created", table_name="probe_events")
    op.drop_table("probe_events")
    op.drop_index("ix_probes_armed", table_name="probes")
    op.drop_index("ix_probes_name", table_name="probes")
    op.drop_table("probes")
