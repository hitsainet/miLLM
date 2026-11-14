"""Add circuit layer claims (Feature 19: Concurrent Circuit Serving).

Until now at most ONE circuit could be active, enforced by the partial unique
index ``uq_circuits_active``. That index is what this migration removes, and
the claim table is what replaces it — moving the constraint from "one circuit"
to "one circuit PER LAYER", which is the unit contention actually has.

The unit is the layer because steering composes additively into a single
per-layer dict:

    modified = original + Σ(strength_i × W_dec[i])      # sae_wrapper.py:444

Two circuits on the same layer sum, and nothing bounds that sum — the ±200
clamp bounds each member individually. The GPU close-out measured what that
costs: two layers at strength 5 produced degenerate output (repeated " lé"
tokens) where one layer was indistinguishable from baseline. So layers are
claimed exclusively by default and composition requires an explicit override.

DOWNGRADE ORDERING IS LOAD-BEARING. Recreating ``uq_circuits_active`` while two
circuits are active FAILS — the index cannot be built over rows that violate
it. So the downgrade must deactivate all but the most recently activated
circuit FIRST, then drop the table, then recreate the index. A downgrade that
merely reverses the upgrade statements bricks any database that used the
feature it is downgrading away from.

Revision ID: 013
Revises: 012
Create Date: 2026-07-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"
    json_type = sa.JSON() if is_sqlite else postgresql.JSONB()

    op.create_table(
        "circuit_layer_claims",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("circuit_id", sa.String(length=64), nullable=False),
        sa.Column("layer", sa.Integer(), nullable=False),
        sa.Column(
            "claimed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "composed",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        # The feature indices this circuit wrote on this layer, so a release
        # can remove ONLY its own keys and leave a co-tenant's intact.
        sa.Column("steering_keys", json_type, nullable=True),
        sa.ForeignKeyConstraint(
            ["circuit_id"], ["circuits.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_circuit_layer_claims_circuit_id",
        "circuit_layer_claims",
        ["circuit_id"],
    )
    op.create_index(
        "idx_circuit_layer_claims_live",
        "circuit_layer_claims",
        ["layer", "released_at"],
    )
    # At most one EXCLUSIVE live claim per layer, enforced by the DATABASE so a
    # race between two concurrent activations is decided here rather than in a
    # service-level check-then-act window a second request can slip through.
    #
    # BOTH dialect predicates are required. Without `sqlite_where` the index is
    # unconditional on SQLite: every released claim collides, and every
    # contention test passes for the wrong reason.
    op.create_index(
        "uq_circuit_layer_claim_live",
        "circuit_layer_claims",
        ["layer"],
        unique=True,
        postgresql_where=sa.text("released_at IS NULL AND composed = false"),
        sqlite_where=sa.text("released_at IS NULL AND composed = 0"),
    )

    # Backfill: the (at most one) currently-active circuit already holds its
    # layers in fact, so it must hold them in the new model too. Without this,
    # the first activation after the migration sees an unclaimed layer and the
    # incumbent silently loses the protection it had a moment earlier.
    circuits = sa.table(
        "circuits",
        sa.column("id", sa.String),
        sa.column("is_active", sa.Boolean),
        sa.column("layers", json_type),
    )
    rows = bind.execute(
        sa.select(circuits.c.id, circuits.c.layers).where(
            circuits.c.is_active.is_(True)
        )
    ).fetchall()
    for circuit_id, layers in rows:
        if isinstance(layers, str):
            import json as _json

            layers = _json.loads(layers)
        for layer in layers or []:
            bind.execute(
                sa.text(
                    "INSERT INTO circuit_layer_claims "
                    "(circuit_id, layer, composed) "
                    "VALUES (:cid, :layer, false)"
                    if not is_sqlite
                    else "INSERT INTO circuit_layer_claims "
                    "(circuit_id, layer, composed) "
                    "VALUES (:cid, :layer, 0)"
                ),
                {"cid": circuit_id, "layer": int(layer)},
            )

    # Only now drop the single-active constraint: if the backfill above failed
    # we want the old guarantee still standing.
    op.drop_index("uq_circuits_active", table_name="circuits")


def downgrade() -> None:
    bind = op.get_bind()

    # ORDER IS LOAD-BEARING — see the module docstring. Recreating a partial
    # unique index over `is_active` while TWO circuits are active fails, so the
    # extra actives must go first.
    #
    # Which one survives is a real decision, not an implementation detail: keep
    # the MOST RECENTLY ACTIVATED, because that is the one an operator most
    # recently asked for. `claimed_at` is the activation record; circuits with
    # no claim row (activated before this migration, or never) sort last.
    circuits = sa.table(
        "circuits",
        sa.column("id", sa.String),
        sa.column("is_active", sa.Boolean),
    )
    active = bind.execute(
        sa.text(
            "SELECT c.id, MAX(cl.claimed_at) AS latest "
            "FROM circuits c "
            "LEFT JOIN circuit_layer_claims cl ON cl.circuit_id = c.id "
            "WHERE c.is_active = true "
            "GROUP BY c.id "
            "ORDER BY latest DESC NULLS LAST, c.id DESC"
            if bind.dialect.name != "sqlite"
            else "SELECT c.id, MAX(cl.claimed_at) AS latest "
            "FROM circuits c "
            "LEFT JOIN circuit_layer_claims cl ON cl.circuit_id = c.id "
            "WHERE c.is_active = 1 "
            "GROUP BY c.id "
            "ORDER BY latest IS NULL, latest DESC, c.id DESC"
        )
    ).fetchall()

    for circuit_id, _latest in active[1:]:
        bind.execute(
            sa.update(circuits)
            .where(circuits.c.id == circuit_id)
            .values(is_active=False)
        )

    op.drop_index("uq_circuit_layer_claim_live", table_name="circuit_layer_claims")
    op.drop_index("idx_circuit_layer_claims_live", table_name="circuit_layer_claims")
    op.drop_index(
        "ix_circuit_layer_claims_circuit_id", table_name="circuit_layer_claims"
    )
    op.drop_table("circuit_layer_claims")

    # Safe now: at most one row satisfies the predicate.
    op.create_index(
        "uq_circuits_active",
        "circuits",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
        sqlite_where=sa.text("is_active = 1"),
    )
