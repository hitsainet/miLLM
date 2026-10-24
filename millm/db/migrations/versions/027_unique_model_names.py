"""One served name, one row: tag duplicate model names with their precision, then make `name` unique.

Migration 015 tagged GGUF rows `repo:QUANT`. A non-GGUF registration still took the bare repo tail,
so two precisions of one repository shared a name. Observed live 2026-10-18 (suite decision D56):

     7 | jevify-gemma4-e4b | Q8   (bitsandbytes)
    13 | jevify-gemma4-e4b | FP16

`/v1/models` listed the name once; `system_fingerprint` named the OTHER row's quantization (the
carrier lookup went by name and took `.first()` with no ORDER BY); a request naming the model with
nothing resident auto-loaded whichever row Postgres returned first; and a miDataworks template's
`bound_model_id` could not pin either precision.

THE RULE, per group of rows sharing an exact name (every other row is left exactly as it is, so
an already-unique name never changes under a client that uses it):

* one row KEEPS the name: a weights-origin carrier first (its name is its manifest's, which its
  lineage records), then the native precision (FP16, no GGUF label), then the lowest id;
* every other row becomes `name:TAG`, TAG being its GGUF label or its quantization (`:Q8`,
  `:Q4`, `:FP32`) — the scheme `ModelService.download_model` now uses for new registrations;
* if `name:TAG` is itself taken, `name:TAG-<id>`, which no other rule produces.

Then `uq_models_name` makes a second row with one name impossible.

Pure data + an index. Idempotent: a database with no duplicates renames nothing. The downgrade
drops the index and leaves the names: putting the duplicates back would restore the defect, which
is not something a downgrade owes anyone.

Revision ID: 027
Revises: 026
Create Date: 2026-10-18
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import sqlalchemy as sa
from alembic import op

revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None

#: (id, name, quantization, gguf_label, weights_origin)
Row = tuple[int, str, str, Optional[str], Optional[str]]


def _tag(row: Row) -> str:
    _id, _name, quantization, gguf_label, _origin = row
    return gguf_label or str(quantization)


def _keeper_order(row: Row) -> tuple[int, int, int]:
    row_id, _name, quantization, gguf_label, origin = row
    native = str(quantization) == "FP16" and not gguf_label
    return (0 if origin else 1, 0 if native else 1, row_id)


def renames(rows: Iterable[Row]) -> dict[int, str]:
    """`{row id: new name}` for every row that must give up a shared name (module docstring)."""
    rows = list(rows)
    taken = {row[1] for row in rows}
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[row[1]].append(row)
    out: dict[int, str] = {}
    for name in sorted(groups):
        group = groups[name]
        if len(group) < 2:
            continue
        for row in sorted(group, key=_keeper_order)[1:]:
            candidate = f"{name}:{_tag(row)}"
            if candidate in taken:
                candidate = f"{name}:{_tag(row)}-{row[0]}"
            taken.add(candidate)
            out[row[0]] = candidate
    return out


def upgrade() -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, name, CAST(quantization AS VARCHAR), gguf_label, weights_origin "
            "FROM models ORDER BY id"
        )
    ).fetchall()
    for row_id, new_name in sorted(renames(tuple(r) for r in rows).items()):
        bind.execute(
            sa.text("UPDATE models SET name = :name WHERE id = :id"),
            {"name": new_name, "id": row_id},
        )
    op.create_index("uq_models_name", "models", ["name"], unique=True)


def downgrade() -> None:
    op.drop_index("uq_models_name", table_name="models")
