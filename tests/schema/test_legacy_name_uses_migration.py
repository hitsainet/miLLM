"""Migration 040 (`legacy_name_uses`, 040 FR-040.10): up creates the table and exactly one `meta`
row (`observing_since`); down drops it. On PostgreSQL through alembic (skips loudly without a
server, fails under the require switch), and the same `upgrade()`/`downgrade()` on SQLite through
alembic's operations, which runs anywhere.
"""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.schema.test_weights_origin_migration import _admin, _alembic, _pg_available, _query
from tests.support.env_flags import flag
from tests.support.package_root import package_path

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = package_path("db", "migrations", "versions", "040_legacy_name_uses.py")
_COLUMNS = ("SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name = 'legacy_name_uses' ORDER BY column_name")


def _module():
    spec = importlib.util.spec_from_file_location("migration_040", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_chain():
    module = _module()
    assert (module.revision, module.down_revision) == ("040", "039")
    # 040 was the head when it was written; later migrations follow it (G19's 041, 2026-11-09). What must hold is a
    # linear chain: at most ONE migration follows 040 (two would fork the history into two heads).
    followers = sorted(p.name for p in MIGRATION.parent.glob("*.py")
                       if "down_revision = \"040\"" in p.read_text())
    assert len(followers) <= 1, f"more than one migration follows 040 (a fork): {followers}"


def _run(fn_name: str, engine: sa.Engine) -> None:
    module = _module()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(module, fn_name)()


def test_upgrade_and_downgrade_on_sqlite():
    engine = sa.create_engine("sqlite://")
    _run("upgrade", engine)
    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT kind, old_name, new_name, count FROM legacy_name_uses")).all()
        assert rows == [("meta", "observing_since", None, 0)]
        assert {c["name"] for c in sa.inspect(conn).get_columns("legacy_name_uses")} == {
            "kind", "old_name", "new_name", "first_seen_at", "last_seen_at", "count"}
        assert sa.inspect(conn).get_pk_constraint("legacy_name_uses")["constrained_columns"] == [
            "kind", "old_name"]
    _run("downgrade", engine)
    with engine.connect() as conn:
        assert "legacy_name_uses" not in sa.inspect(conn).get_table_names()


@pytest.fixture(scope="module")
def fresh_db():
    if not asyncio.run(_pg_available()):
        message = "PostgreSQL is not reachable"
        if flag("REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MISERVER_REQUIRE_PG=1 (or MILLM_) — this guard did NOT run")
        pytest.skip(f"{message} — a migration round trip needs a real server")
    name = f"legacy_names_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def test_upgrade_creates_table_and_one_meta_row(fresh_db):
    _alembic(fresh_db, "upgrade", "039")
    assert asyncio.run(_query(fresh_db, _COLUMNS)) == []
    _alembic(fresh_db, "upgrade", "040")
    assert [r[0] for r in asyncio.run(_query(fresh_db, _COLUMNS))] == [
        "count", "first_seen_at", "kind", "last_seen_at", "new_name", "old_name"]
    rows = asyncio.run(_query(fresh_db, "SELECT kind, old_name, new_name, count FROM legacy_name_uses"))
    assert rows == [("meta", "observing_since", None, 0)]


def test_downgrade_drops_the_table(fresh_db):
    _alembic(fresh_db, "upgrade", "040")
    _alembic(fresh_db, "downgrade", "039")
    assert asyncio.run(_query(fresh_db, _COLUMNS)) == []
    _alembic(fresh_db, "upgrade", "head")
    assert asyncio.run(_query(fresh_db, _COLUMNS)) != []
