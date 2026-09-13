"""Compare what the migrations build with what the ORM models declare.

Two independent views of one schema, because neither is enough alone:

* ``alembic_drift`` — Alembic's autogenerate comparison, rendered as stable
  signatures. It is exactly what ``alembic check`` sees, and it is blind to
  partial-index predicates, index methods, enum labels, CHECK constraints and
  triggers.
* ``snapshot`` / ``diff`` — a catalog snapshot read straight from ``pg_catalog``,
  which sees all of those. A database built by the migrations and one built by
  ``create_orm_schema`` must produce equal snapshots.

The functions take a synchronous connection; with asyncpg, call them through
``await conn.run_sync(fn)``. Plain Python rather than pg_dump, so it runs inside a
production pod::

    python -m millm.db.schema_parity snapshot --url postgresql+asyncpg://... > snapshot.json
    python -m millm.db.schema_parity drift    --url postgresql+asyncpg://...
    python -m millm.db.schema_parity diff a.json b.json

This mirrors ``backend/src/db/schema_parity.py`` in miStudio; the two repos do not
import each other.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from millm.db.alembic_support import COMPARE_OPTIONS, target_metadata

EXCLUDED_TABLES = frozenset({"alembic_version"})


# --------------------------------------------------------------------------- drift


def alembic_drift(conn: sa.engine.Connection, metadata: sa.MetaData | None = None) -> list[str]:
    """Differences between the database on ``conn`` and the models, as sorted signatures."""
    metadata = target_metadata() if metadata is None else metadata
    context = MigrationContext.configure(
        conn, opts={**COMPARE_OPTIONS, "target_metadata": metadata}
    )
    signatures: set[str] = set()
    for entry in compare_metadata(context, metadata):
        for op in entry if isinstance(entry, list) else [entry]:
            signatures.add(_signature(op))
    return sorted(signatures)


def _columns_key(constraint: Any) -> str:
    return ",".join(sorted(column.name for column in constraint.columns))


def _signature(op: tuple) -> str:
    """A stable, reviewable name for one autogenerate operation."""
    kind = op[0]
    if kind in ("add_table", "remove_table", "add_table_comment", "remove_table_comment"):
        return f"{kind}:{op[1].name}"
    if kind in ("add_column", "remove_column"):
        return f"{kind}:{op[2]}:{op[3].name}"
    if kind in ("add_index", "remove_index"):
        return f"{kind}:{op[1].table.name}:{op[1].name}"
    if kind in ("add_constraint", "remove_constraint"):
        constraint = op[1]
        return f"{kind}:{constraint.table.name}:{constraint.name or _columns_key(constraint)}"
    if kind in ("add_fk", "remove_fk"):
        fk = op[1]
        target = fk.elements[0].target_fullname if fk.elements else "?"
        return f"{kind}:{fk.parent.name}:{fk.name or _columns_key(fk)}->{target}"
    if kind.startswith("modify_"):
        return f"{kind}:{op[2]}:{op[3]}"
    return f"{kind}:{op!r}"


# ------------------------------------------------------------------------ snapshot

_PUBLIC = "public."

_TABLES = """
SELECT c.relname, c.relkind, pg_get_partkeydef(c.oid), pg_get_expr(c.relpartbound, c.oid),
       obj_description(c.oid, 'pg_class')
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
"""

_COLUMNS = """
SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull,
       pg_get_expr(d.adbin, d.adrelid), a.attidentity, a.attgenerated,
       col_description(c.oid, a.attnum)
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') AND a.attnum > 0 AND NOT a.attisdropped
"""

_CONSTRAINTS = """
SELECT c.relname, con.conname, con.contype, pg_get_constraintdef(con.oid)
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
"""

_INDEXES = "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'"

_ENUMS = """
SELECT t.typname, e.enumlabel
FROM pg_type t JOIN pg_enum e ON e.enumtypid = t.oid JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = 'public'
"""

_SEQUENCES = """
SELECT s.relname,
       (SELECT owner.relname || '.' || a.attname
          FROM pg_depend d
          JOIN pg_class owner ON owner.oid = d.refobjid
          JOIN pg_attribute a ON a.attrelid = d.refobjid AND a.attnum = d.refobjsubid
         WHERE d.objid = s.oid AND d.deptype IN ('a', 'i')
         LIMIT 1)
FROM pg_class s JOIN pg_namespace n ON n.oid = s.relnamespace
WHERE n.nspname = 'public' AND s.relkind = 'S'
"""

_TRIGGERS = """
SELECT c.relname, t.tgname, pg_get_triggerdef(t.oid)
FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND NOT t.tgisinternal
"""

_FUNCTIONS = """
SELECT p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')',
       md5(pg_get_functiondef(p.oid))
FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public' AND p.prokind IN ('f', 'p')
"""

_VIEWS = "SELECT viewname, definition FROM pg_views WHERE schemaname = 'public'"


def _strip_schema(text: str | None) -> str | None:
    return None if text is None else text.replace(_PUBLIC, "")


def snapshot(conn: sa.engine.Connection) -> dict[str, dict[str, Any]]:
    """Every schema object in ``public`` that the application depends on, keyed by name."""

    def rows(sql: str) -> list[tuple]:
        return [tuple(row) for row in conn.execute(sa.text(sql))]

    tables = {
        name: {"kind": kind, "partition_key": key, "partition_bound": bound, "comment": comment}
        for name, kind, key, bound, comment in rows(_TABLES)
        if name not in EXCLUDED_TABLES
    }
    columns = {
        f"{table}.{column}": {
            "type": col_type,
            "not_null": not_null,
            "default": _strip_schema(default),
            "identity": identity,
            "generated": generated,
            "comment": comment,
        }
        for table, column, col_type, not_null, default, identity, generated, comment in rows(
            _COLUMNS
        )
        if table not in EXCLUDED_TABLES
    }
    constraints = {
        f"{table}.{name}": {"type": contype, "definition": _strip_schema(definition)}
        for table, name, contype, definition in rows(_CONSTRAINTS)
        if table not in EXCLUDED_TABLES
    }
    indexes = {
        name: _strip_schema(definition)
        for table, name, definition in rows(_INDEXES)
        if table not in EXCLUDED_TABLES
    }
    enums: dict[str, list[str]] = {}
    for type_name, label in rows(_ENUMS):
        enums.setdefault(type_name, []).append(label)
    sequences = {
        name: owner
        for name, owner in rows(_SEQUENCES)
        if not (owner or "").startswith("alembic_version.")
    }
    return {
        "tables": tables,
        "columns": columns,
        "constraints": constraints,
        "indexes": indexes,
        # Label ORDER is ignored: create_all emits Python member order, while
        # `ALTER TYPE ... ADD VALUE` appends. Order only matters to ORDER BY / < on the enum.
        "enums": {name: sorted(labels) for name, labels in enums.items()},
        "sequences": sequences,
        "triggers": {f"{t}.{n}": _strip_schema(d) for t, n, d in rows(_TRIGGERS)},
        "functions": dict(rows(_FUNCTIONS)),
        "views": {name: _strip_schema(d) for name, d in rows(_VIEWS)},
    }


def diff(
    a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], a_name: str = "a", b_name: str = "b"
) -> list[str]:
    """Human-readable differences between two snapshots; empty when they match."""
    out: list[str] = []
    for section in sorted(set(a) | set(b)):
        left, right = a.get(section, {}), b.get(section, {})
        for key in sorted(set(left) - set(right)):
            out.append(f"{section}: only in {a_name}: {key} = {left[key]}")
        for key in sorted(set(right) - set(left)):
            out.append(f"{section}: only in {b_name}: {key} = {right[key]}")
        for key in sorted(set(left) & set(right)):
            if left[key] != right[key]:
                out.append(f"{section}: {key}: {a_name}={left[key]} {b_name}={right[key]}")
    return out


def create_orm_schema(conn: sa.engine.Connection, metadata: sa.MetaData | None = None) -> None:
    """Build the schema the models declare, creating native enum types first."""
    metadata = target_metadata() if metadata is None else metadata
    created: set[str] = set()
    for table in metadata.sorted_tables:
        for column in table.columns:
            enum_type = column.type
            if (
                isinstance(enum_type, sa.Enum)
                and enum_type.native_enum
                and enum_type.name
                and enum_type.name not in created
            ):
                enum_type.create(bind=conn, checkfirst=True)
                created.add(enum_type.name)
    metadata.create_all(bind=conn)


# ----------------------------------------------------------------------------- CLI


async def _run(url: str, fn: Any) -> Any:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            return await conn.run_sync(fn)
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m millm.db.schema_parity")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("snapshot", "print a pg_catalog schema snapshot as JSON"),
        ("drift", "print alembic drift signatures as JSON"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--url", default=os.environ.get("DATABASE_URL"))
    compare = commands.add_parser("diff", help="compare two snapshot files (exit 1 if they differ)")
    compare.add_argument("a")
    compare.add_argument("b")
    args = parser.parse_args(argv)

    if args.command == "diff":
        with open(args.a) as fa, open(args.b) as fb:
            differences = diff(json.load(fa), json.load(fb), args.a, args.b)
        print("\n".join(differences) if differences else "identical")
        return 1 if differences else 0

    if not args.url:
        parser.error("--url is required (or set DATABASE_URL)")
    fn = snapshot if args.command == "snapshot" else alembic_drift
    result = asyncio.run(_run(args.url, fn))
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
