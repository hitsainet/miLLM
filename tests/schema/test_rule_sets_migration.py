"""Migration 035 (039 S2): the four rule tables, on SQLite and PostgreSQL, against the ORM.

- up/down on SQLite through the migration module itself, and on PostgreSQL 16 through alembic
  (skipped loudly without a server, required under `MILLM_REQUIRE_PG=1`);
- the ORM's `rule_signal_results` column list: no `Text` column and no string wider than 64
  (FTDD §4.3) — a rule result carries a count, never a value;
- a deleted rule set keeps its results (no foreign key to `rule_sets`).
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.schema.test_probe_sessions_migration import _exec, fresh_db  # noqa: F401
from tests.schema.test_weights_origin_migration import _alembic, _query

MIGRATION = (Path(__file__).resolve().parents[2] / "millm" / "db" / "migrations" / "versions"
             / "035_rule_sets.py")
TABLES = ("rule_sets", "behaviour_rule_signals", "behaviour_rule_signal_history",
          "rule_signal_results")


def _module():
    spec = importlib.util.spec_from_file_location("m035", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(engine, step: str) -> None:
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            getattr(_module(), step)()


def _tables(engine) -> set[str]:
    return set(sa.inspect(engine).get_table_names())


def test_the_migration_follows_034():
    module = _module()
    assert (module.revision, module.down_revision) == ("035", "034")


def test_sqlite_down_then_up_matches_the_orm(tmp_path):
    from millm.db.base import Base
    import millm.db.models  # noqa: F401 - every table

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    Base.metadata.create_all(engine)
    orm = {t: {c["name"]: str(c["type"]) for c in sa.inspect(engine).get_columns(t)}
           for t in TABLES}
    orm_indexes = {t: {i["name"] for i in sa.inspect(engine).get_indexes(t)} for t in TABLES}
    _run(engine, "downgrade")
    assert not (_tables(engine) & set(TABLES))
    _run(engine, "upgrade")
    inspector = sa.inspect(engine)
    for table in TABLES:
        assert {c["name"]: str(c["type"]) for c in inspector.get_columns(table)} == orm[table]
        assert {i["name"] for i in inspector.get_indexes(table)} == orm_indexes[table]
    engine.dispose()


def test_a_rule_result_column_cannot_hold_text():
    """FTDD §4.3: no `Text`, no `String` wider than 64, in `rule_signal_results`."""
    from millm.db.models.rule_set import RuleSignalResult

    columns = RuleSignalResult.__table__.columns
    assert len(columns) >= 20, "the column list is unexpectedly short — wrong table?"
    for column in columns:
        assert not isinstance(column.type, sa.Text), f"{column.name} is Text"
        if isinstance(column.type, sa.String) and not isinstance(column.type, sa.Text):
            assert column.type.length is not None and column.type.length <= 64, column.name
    assert "rule_set_id" in columns and not columns["rule_set_id"].foreign_keys


def test_pg_035_up_and_down(fresh_db):  # noqa: F811
    _alembic(fresh_db, "upgrade", "034")
    present = {r[0] for r in asyncio.run(_query(
        fresh_db, "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))}
    assert not (present & set(TABLES))
    _alembic(fresh_db, "upgrade", "035")
    present = {r[0] for r in asyncio.run(_query(
        fresh_db, "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))}
    assert set(TABLES) <= present
    asyncio.run(_exec(fresh_db, (
        "INSERT INTO rule_sets (id, name, behaviour_key, origin, digest, document) VALUES "
        "('rs_000000000001', 'n', 'k', 'authored', '" + "a" * 64 + "', '{}'::jsonb)"
    )))
    asyncio.run(_exec(fresh_db, (
        "INSERT INTO rule_signal_results (result, result_id, event_time, outcome, rule_set_id, "
        "rule_set_digest, signal, form, \"window\", fires, count) VALUES ('{}'::jsonb, "
        "'00000000-0000-7000-8000-000000000001', now(), 'completed', 'rs_000000000001', '"
        + "a" * 64 + "', 'hit', 'pattern', 'prompt', true, 1)"
    )))
    # No foreign key: deleting the set keeps its result.
    asyncio.run(_exec(fresh_db, "DELETE FROM rule_sets"))
    assert asyncio.run(_query(fresh_db, "SELECT count(*) FROM rule_signal_results"))[0][0] == 1
    # The batch-line index is partial and unique.
    indexes = dict(asyncio.run(_query(
        fresh_db, "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = "
                  "'rule_signal_results'")))
    assert "WHERE (batch_id IS NOT NULL)" in indexes["uq_rule_signal_results_batch_line"]
    assert "UNIQUE" in indexes["uq_rule_signal_results_batch_line"]
    _alembic(fresh_db, "downgrade", "034")
    present = {r[0] for r in asyncio.run(_query(
        fresh_db, "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))}
    assert not (present & set(TABLES))
