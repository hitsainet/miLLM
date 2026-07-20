"""Add circuit edge sensing (Feature 15: Circuit Edge Sensing).

Feature 11 sensing answers "did all the members of a cluster co-fire?".
An edge is a directed, evidence-graded claim — *upstream fired, and then
downstream fired within a token lag* — so its events carry BOTH endpoints with
their own layer/feature/position/activation, plus the observed ``token_lag``.

``edge_rung`` and ``edge_rung_language`` are denormalised onto every row on
purpose. The rung of a circuit can change (a re-import at a higher rung, an
edge re-validated), and an event must keep describing the evidence that was
true WHEN IT WAS OBSERVED. Reading today's rung against a months-old event
would retroactively upgrade the claim — the exact overclaim the evidence
ladder exists to prevent.

``circuits.sensing_enabled`` mirrors ``profiles.sensing_enabled``: persistent
operator INTENT, reported distinctly from runtime ``armed`` (a circuit can be
enabled but not armed because it is not active, or because its SAE set is not
attached).

Revision ID: 012
Revises: 011
Create Date: 2026-07-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite test DBs) — mirrors
# sensing_events, which is the table this one is modelled on.
JSONVariant = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "circuits",
        sa.Column(
            "sensing_enabled",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )

    op.create_table(
        "circuit_edge_sensing_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("circuit_id", sa.String(length=50), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("phase", sa.String(length=10), nullable=False),
        # Stable identity for the edge within its circuit, synthesised as
        # "{up_idx}@{up_layer}->{down_idx}@{down_layer}". circuit-definition/v1
        # edges carry no id of their own.
        sa.Column("edge_key", sa.String(length=128), nullable=False),
        # Upstream endpoint
        sa.Column("up_layer", sa.Integer(), nullable=False),
        sa.Column("up_feature_idx", sa.Integer(), nullable=False),
        sa.Column("up_pos", sa.Integer(), nullable=False),
        sa.Column("up_act", sa.Float(), nullable=False),
        # Downstream endpoint
        sa.Column("down_layer", sa.Integer(), nullable=False),
        sa.Column("down_feature_idx", sa.Integer(), nullable=False),
        sa.Column("down_pos", sa.Integer(), nullable=False),
        sa.Column("down_act", sa.Float(), nullable=False),
        # down_pos - up_pos; always >= 1 (strict ordering) and <= the configured
        # lag window. Stored rather than derived so queries can filter on it.
        sa.Column("token_lag", sa.Integer(), nullable=False),
        # Evidence as of observation — see module docstring.
        sa.Column("edge_rung", sa.Integer(), nullable=False),
        sa.Column("edge_rung_language", sa.String(length=64), nullable=False),
        sa.Column("edge_type", sa.String(length=32), nullable=True),
        sa.Column("ambient_fired_count", sa.Integer(), nullable=True),
        sa.Column("context_text", sa.Text(), nullable=True),
        sa.Column("context_token_ids", JSONVariant, nullable=True),
        # {before, span, after} — span covers up_pos..down_pos inclusive.
        sa.Column("context_parts", JSONVariant, nullable=True),
        sa.Column("summary", sa.String(length=300), nullable=False),
        sa.Column(
            "truncated", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["circuit_id"], ["circuits.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # (circuit_id, created_at) drives the list query; the id tiebreak matters
    # because one flush inserts many rows with an identical created_at.
    op.create_index(
        "idx_circuit_edge_events_circuit_created",
        "circuit_edge_sensing_events",
        ["circuit_id", "created_at"],
    )
    op.create_index(
        "idx_circuit_edge_events_request",
        "circuit_edge_sensing_events",
        ["request_id"],
    )
    # Supports "show me every observation of THIS edge".
    op.create_index(
        "idx_circuit_edge_events_edge",
        "circuit_edge_sensing_events",
        ["circuit_id", "edge_key"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_circuit_edge_events_edge", table_name="circuit_edge_sensing_events"
    )
    op.drop_index(
        "idx_circuit_edge_events_request", table_name="circuit_edge_sensing_events"
    )
    op.drop_index(
        "idx_circuit_edge_events_circuit_created",
        table_name="circuit_edge_sensing_events",
    )
    op.drop_table("circuit_edge_sensing_events")
    op.drop_column("circuits", "sensing_enabled")
