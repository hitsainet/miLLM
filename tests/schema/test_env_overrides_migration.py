"""Migration 028 (`env_overrides`, `env_override_changes`, Feature 35) must go up and come down,
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


def _columns(table: str) -> str:
    return (
        "SELECT column_name, data_type, is_nullable, character_maximum_length "
        f"FROM information_schema.columns WHERE table_name = '{table}' ORDER BY column_name"
    )


_INDEXES = (
    "SELECT indexname FROM pg_indexes WHERE tablename = 'env_override_changes' ORDER BY indexname"
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
    name = f"env_overrides_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def test_028_creates_both_tables_and_comes_down(fresh_db):
    _alembic(fresh_db, "upgrade", "027")
    assert asyncio.run(_query(fresh_db, _columns("env_overrides"))) == []

    _alembic(fresh_db, "upgrade", "028")
    assert asyncio.run(_query(fresh_db, _columns("env_overrides"))) == [
        ("key", "character varying", "NO", 64),
        ("reason", "character varying", "YES", 500),
        ("updated_at", "timestamp with time zone", "NO", None),
        ("updated_by", "character varying", "NO", 128),
        ("value", "json", "NO", None),
    ]
    assert asyncio.run(_query(fresh_db, _columns("env_override_changes"))) == [
        ("action", "character varying", "NO", 16),
        ("actor", "character varying", "NO", 128),
        ("approval_id", "character varying", "YES", 64),
        ("changed_at", "timestamp with time zone", "NO", None),
        ("id", "integer", "NO", None),
        ("key", "character varying", "NO", 64),
        ("new_value", "json", "YES", None),
        ("previous_value", "json", "YES", None),
        ("reason", "character varying", "YES", 500),
    ]
    assert asyncio.run(_query(fresh_db, _INDEXES)) == [
        ("env_override_changes_pkey",),
        ("ix_env_override_changes_changed_at",),
        ("ix_env_override_changes_key",),
    ]
    asyncio.run(_exec(fresh_db, (
        "INSERT INTO env_overrides (key, value, updated_by) VALUES ('MAX_PENDING_REQUESTS', '4', "
        "'operator')"
    )))
    asyncio.run(_exec(fresh_db, (
        "INSERT INTO env_override_changes (key, action, previous_value, new_value, actor) "
        "VALUES ('MAX_PENDING_REQUESTS', 'set', '10', '4', 'operator')"
    )))
    rows = asyncio.run(_query(
        fresh_db, "SELECT key, value::text, updated_at IS NOT NULL FROM env_overrides"))
    assert rows == [("MAX_PENDING_REQUESTS", "4", True)]
    rows = asyncio.run(_query(fresh_db, "SELECT id, changed_at IS NOT NULL FROM env_override_changes"))
    assert rows == [(1, True)], "the history id must be generated and the time defaulted"

    _alembic(fresh_db, "downgrade", "027")
    assert asyncio.run(_query(fresh_db, _columns("env_overrides"))) == []
    assert asyncio.run(_query(fresh_db, _columns("env_override_changes"))) == []
    _alembic(fresh_db, "upgrade", "head")
    assert asyncio.run(_query(fresh_db, _columns("env_overrides"))) != []
