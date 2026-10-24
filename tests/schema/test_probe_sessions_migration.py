"""Migration 029 (probe sessions, Feature 36) goes up, backfills `behaviour_key`, and comes down,
on PostgreSQL. The ORM declares the same tables; `test_schema_guards.py` compares the two."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from tests.schema.test_schema_guards import PG_BASE
from tests.schema.test_weights_origin_migration import _admin, _alembic, _pg_available, _query


def _tables() -> str:
    return (
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'probe_s%' "
        "OR table_name = 'probe_transcripts' ORDER BY table_name"
    )


def _column(table: str, column: str) -> str:
    return (
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_name = '{table}' AND column_name = '{column}'"
    )


async def _exec(db: str, sql: str) -> None:
    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text(sql))
    finally:
        await engine.dispose()


@pytest.fixture(scope="module")
def fresh_db():
    if not asyncio.run(_pg_available()):
        message = f"PostgreSQL is not reachable at {PG_BASE}"
        if os.environ.get("MILLM_REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MILLM_REQUIRE_PG=1 — this guard did NOT run")
        pytest.skip(f"{message} — a migration round trip needs a real server")
    name = f"probe_sessions_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def _probe(probe_id: str, definition: str) -> str:
    return (
        "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, rule, scope, "
        f"rung) VALUES ('{probe_id}', '{probe_id}', '{definition}'::jsonb, 'm', 8, 2, 0, 'mean', "
        "'all', 2)"
    )


def test_029_creates_the_tables_backfills_and_comes_down(fresh_db):
    _alembic(fresh_db, "upgrade", "028")
    asyncio.run(_exec(fresh_db, _probe("pr_with", '{"behaviour": {"key": "pii", "version": 1}}')))
    asyncio.run(_exec(fresh_db, _probe("pr_without", '{"name": "x"}')))
    asyncio.run(_exec(fresh_db, (
        "INSERT INTO probe_events (probe_id, request_id, scored, \"window\", provisional, "
        "threshold_revision, origin) VALUES ('pr_with', 'r1', true, 'all', false, 1, 'live')"
    )))
    assert asyncio.run(_query(fresh_db, _tables())) == []

    _alembic(fresh_db, "upgrade", "029")
    assert asyncio.run(_query(fresh_db, _tables())) == [
        ("probe_session_turns",), ("probe_sessions",), ("probe_transcripts",),
    ]
    # The one backfill: a projection of the definition, never a guess.
    assert asyncio.run(_query(
        fresh_db, "SELECT id, behaviour_key FROM probes ORDER BY id"
    )) == [("pr_with", "pii"), ("pr_without", None)]
    # Earlier events: NULL session columns ("not recorded"), not backfilled.
    assert asyncio.run(_query(fresh_db, "SELECT session_id, turn_index FROM probe_events")) == [
        (None, None)
    ]
    # A session delete detaches events (SET NULL) and cascades turns and transcripts.
    asyncio.run(_exec(fresh_db, "INSERT INTO probe_sessions (id, source) VALUES ('ps_1', 'header')"))
    asyncio.run(_exec(fresh_db, (
        "UPDATE probe_events SET session_id = 'ps_1', turn_index = 2"
    )))
    asyncio.run(_exec(fresh_db, (
        "INSERT INTO probe_session_turns (session_id, request_id, turn_index, prefix_hash, "
        "weak_hash) VALUES ('ps_1', 'r1', 2, 'p', 'w')"
    )))
    asyncio.run(_exec(fresh_db, (
        "INSERT INTO probe_transcripts (turn_id, session_id, user_text, reply_text) "
        "SELECT id, 'ps_1', 'u', 'r' FROM probe_session_turns"
    )))
    asyncio.run(_exec(fresh_db, "DELETE FROM probe_sessions WHERE id = 'ps_1'"))
    assert asyncio.run(_query(fresh_db, "SELECT session_id, turn_index FROM probe_events")) == [
        (None, 2)
    ]
    assert asyncio.run(_query(fresh_db, "SELECT count(*) FROM probe_session_turns")) == [(0,)]
    assert asyncio.run(_query(fresh_db, "SELECT count(*) FROM probe_transcripts")) == [(0,)]

    _alembic(fresh_db, "downgrade", "028")
    assert asyncio.run(_query(fresh_db, _tables())) == []
    assert asyncio.run(_query(fresh_db, _column("probe_events", "session_id"))) == []
    assert asyncio.run(_query(fresh_db, _column("probes", "behaviour_key"))) == []
    _alembic(fresh_db, "upgrade", "head")
    assert asyncio.run(_query(fresh_db, _column("probes", "behaviour_key"))) != []
