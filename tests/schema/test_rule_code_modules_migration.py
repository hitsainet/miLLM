"""Migration 036 (039 S3): `rule_sets.code_modules`, on SQLite and PostgreSQL, against the ORM.

- the revision chain follows 035;
- up/down on SQLite through the migration module itself equals the ORM's column list;
- on PostgreSQL 16 through alembic: the column is JSONB, nullable, and down removes only it
  (skipped loudly without a server, required under `MILLM_REQUIRE_PG=1`).
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.schema.test_probe_sessions_migration import _exec, fresh_db  # noqa: F401
from tests.schema.test_weights_origin_migration import _alembic, _query

MIGRATION = (Path(__file__).resolve().parents[2] / "millm" / "db" / "migrations" / "versions"
             / "036_rule_code_modules.py")


def _module():
    spec = importlib.util.spec_from_file_location("m036", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(engine, step: str) -> None:
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            getattr(_module(), step)()


def _columns(engine) -> dict[str, str]:
    return {c["name"]: str(c["type"]) for c in sa.inspect(engine).get_columns("rule_sets")}


def test_the_migration_follows_035():
    module = _module()
    assert (module.revision, module.down_revision) == ("036", "035")


def test_sqlite_down_then_up_matches_the_orm(tmp_path):
    from millm.db.base import Base
    import millm.db.models  # noqa: F401 - every table

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    Base.metadata.create_all(engine)
    orm = _columns(engine)
    assert "code_modules" in orm
    _run(engine, "downgrade")
    assert "code_modules" not in _columns(engine)
    assert set(_columns(engine)) == set(orm) - {"code_modules"}
    _run(engine, "upgrade")
    assert _columns(engine) == orm
    engine.dispose()


def test_pg_036_up_and_down(fresh_db):  # noqa: F811
    _alembic(fresh_db, "upgrade", "035")
    query = ("SELECT column_name, data_type, is_nullable FROM information_schema.columns "
             "WHERE table_name = 'rule_sets' AND column_name = 'code_modules'")
    assert asyncio.run(_query(fresh_db, query)) == []
    asyncio.run(_exec(fresh_db, (
        "INSERT INTO rule_sets (id, name, behaviour_key, origin, digest, document) VALUES "
        "('rs_000000000001', 'n', 'k', 'authored', '" + "a" * 64 + "', '{}'::jsonb)")))
    _alembic(fresh_db, "upgrade", "036")
    assert [tuple(r) for r in asyncio.run(_query(fresh_db, query))] == [
        ("code_modules", "jsonb", "YES")]
    # A set imported before S3 keeps NULL: its code rules stay pending.
    assert asyncio.run(_query(fresh_db, "SELECT code_modules FROM rule_sets"))[0][0] is None
    _alembic(fresh_db, "downgrade", "035")
    assert asyncio.run(_query(fresh_db, query)) == []
    assert asyncio.run(_query(fresh_db, "SELECT count(*) FROM rule_sets"))[0][0] == 1
