"""Behaviour probe sets and per-model limits in behaviours (Feature 39 slice S1).

**Numbering.** The 039 documents call this migration `033` (FTDD §4.2, FTID §4); miLLM 038 took `033`
for R-039 (`033_probe_identity_recompute`) before 039 was built, so S1 is `034` and S2 `035`
(recorded in the 039 FTASKS, 0.5).

* `probes.behaviour_version` — a PROJECTION of `definition.behaviour.version` (NULL = uncatalogued,
  which the selector treats as a version of its own, OQ-39.3 (a)), backfilled once from the stored
  document by a dialect-portable loop, as migration `029` did for the key; plus the index the
  selector reads, `(behaviour_key, behaviour_version)`.
* `probes.last_cost` — the latest arm OR dry-run cost measurement with the fingerprint it was taken
  under. Evidence, never cleared by a disarm (TD39-11), unlike `cost`.
* `model_probe_limits` — `max_behaviours` (the count gate, in behaviours, D13) and the approval
  record: `source`, `set_by`, `approved_at`, `approved_measurement_id`, `fingerprint`,
  `recommendation`.
* `model_probe_limits_history` — append-only by service rule (the `env_override_changes` pattern of
  migration `028`): every change of a model's limits, old and new.
* `probe_capacity_measurements.lines` — the per-rule cost lines (`mean`, `max`, `attention`).

⚠ **`max_armed` IS NULLED.** A per-model `max_armed` was a probe count; the count gate now counts
behaviours, so carrying a probe count over as a behaviour count would admit (or refuse) the wrong
thing (TD39-14). Each non-NULL value is written to the history first, and the downgrade restores
it from those rows.

Revision ID: 034
Revises: 033
Create Date: 2026-10-24
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None

JSONVariant = sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True),
                                                      "postgresql")

#: The history `source` the migration writes, and the one its downgrade restores from.
SOURCE = "migration-034"
REASON = ("D13: the count limit counts behaviours, so a per-model probe count is not carried over "
          "as a behaviour count; restored by the downgrade")


def _behaviour_version(definition: Any) -> Optional[int]:
    """`definition.behaviour.version` when it is a positive integer, else None. FROZEN copy."""
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except ValueError:
            return None
    if not isinstance(definition, dict):
        return None
    behaviour = definition.get("behaviour")
    if not isinstance(behaviour, dict) or not behaviour.get("key"):
        return None
    version = behaviour.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        return None
    return version


def _as_dict(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def upgrade() -> None:
    op.add_column("probes", sa.Column("behaviour_version", sa.Integer(), nullable=True))
    op.add_column("probes", sa.Column("last_cost", JSONVariant, nullable=True))
    op.create_index("ix_probes_behaviour_key_version", "probes",
                    ["behaviour_key", "behaviour_version"], unique=False)

    op.add_column("model_probe_limits", sa.Column("max_behaviours", sa.Integer(), nullable=True))
    op.add_column("model_probe_limits", sa.Column("source", sa.String(length=24), nullable=True))
    op.add_column("model_probe_limits", sa.Column("set_by", sa.String(length=64), nullable=True))
    op.add_column("model_probe_limits",
                  sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("model_probe_limits",
                  sa.Column("approved_measurement_id", sa.Integer(), nullable=True))
    op.add_column("model_probe_limits", sa.Column("fingerprint", JSONVariant, nullable=True))
    op.add_column("model_probe_limits", sa.Column("recommendation", JSONVariant, nullable=True))
    with op.batch_alter_table("model_probe_limits") as batch:
        batch.create_foreign_key(
            "model_probe_limits_approved_measurement_id_fkey", "probe_capacity_measurements",
            ["approved_measurement_id"], ["id"], ondelete="SET NULL",
        )

    op.create_table(
        "model_probe_limits_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("old", JSONVariant, nullable=True),
        sa.Column("new", JSONVariant, nullable=True),
        sa.Column("source", sa.String(length=24), nullable=True),
        sa.Column("set_by", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_probe_limits_history_model", "model_probe_limits_history",
                    ["model_id", "changed_at"], unique=False)

    op.add_column("probe_capacity_measurements", sa.Column("lines", JSONVariant, nullable=True))

    bind = op.get_bind()
    # ── backfill: the version projection ───────────────────────────────────────────────
    probes = sa.table(
        "probes",
        sa.column("id", sa.String()),
        sa.column("definition", sa.JSON()),
        sa.column("behaviour_version", sa.Integer()),
    )
    for probe_id, definition in bind.execute(sa.select(probes.c.id, probes.c.definition)).all():
        version = _behaviour_version(definition)
        if version is not None:
            bind.execute(
                probes.update().where(probes.c.id == probe_id).values(behaviour_version=version)
            )

    # ── max_armed → history → NULL (TD39-14) ───────────────────────────────────────────
    limits = sa.table(
        "model_probe_limits",
        sa.column("model_id", sa.Integer()),
        sa.column("max_armed", sa.Integer()),
    )
    history = sa.table(
        "model_probe_limits_history",
        sa.column("model_id", sa.Integer()),
        sa.column("changed_at", sa.DateTime(timezone=True)),
        sa.column("old", sa.JSON()),
        sa.column("new", sa.JSON()),
        sa.column("source", sa.String()),
        sa.column("set_by", sa.String()),
        sa.column("reason", sa.String()),
    )
    now = datetime.now(timezone.utc)
    carried = bind.execute(
        sa.select(limits.c.model_id, limits.c.max_armed).where(limits.c.max_armed.is_not(None))
    ).all()
    for model_id, max_armed in carried:
        bind.execute(history.insert().values(
            model_id=model_id, changed_at=now, old={"max_armed": int(max_armed)},
            new={"max_armed": None}, source=SOURCE, set_by="migration", reason=REASON,
        ))
    if carried:
        # ⚠ AFTER the history rows: the downgrade restores from them.
        bind.execute(limits.update().where(limits.c.max_armed.is_not(None))
                     .values(max_armed=None))


def downgrade() -> None:
    bind = op.get_bind()
    limits = sa.table(
        "model_probe_limits",
        sa.column("model_id", sa.Integer()),
        sa.column("max_armed", sa.Integer()),
    )
    history = sa.table(
        "model_probe_limits_history",
        sa.column("id", sa.Integer()),
        sa.column("model_id", sa.Integer()),
        sa.column("old", sa.JSON()),
        sa.column("source", sa.String()),
    )
    rows = bind.execute(
        sa.select(history.c.model_id, history.c.old).where(history.c.source == SOURCE)
        .order_by(history.c.id)
    ).all()
    for model_id, old in rows:
        value = (_as_dict(old) or {}).get("max_armed")
        if isinstance(value, int):
            bind.execute(limits.update().where(limits.c.model_id == model_id)
                         .values(max_armed=value))

    # Batch operations throughout: SQLite cannot drop a column that a constraint or an index
    # names in place, and the unit-test database is SQLite (PostgreSQL runs them as ALTERs).
    with op.batch_alter_table("probe_capacity_measurements") as batch:
        batch.drop_column("lines")
    op.drop_index("ix_model_probe_limits_history_model", table_name="model_probe_limits_history")
    op.drop_table("model_probe_limits_history")
    with op.batch_alter_table("model_probe_limits") as batch:
        batch.drop_constraint("model_probe_limits_approved_measurement_id_fkey",
                              type_="foreignkey")
    with op.batch_alter_table("model_probe_limits") as batch:
        for column in ("recommendation", "fingerprint", "approved_measurement_id", "approved_at",
                       "set_by", "source", "max_behaviours"):
            batch.drop_column(column)
    op.drop_index("ix_probes_behaviour_key_version", table_name="probes")
    with op.batch_alter_table("probes") as batch:
        batch.drop_column("last_cost")
        batch.drop_column("behaviour_version")
