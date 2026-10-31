"""Migration 025 (`app_settings`, Feature 34) must go up and come down, on PostgreSQL.

The table holds only encrypted envelopes (`millm/core/secret_box.py`); this checks the schema the
migration builds, and that a row written through it round-trips. The ORM declares the same table;
`test_schema_guards.py` compares the two.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from tests.schema.test_schema_guards import PG_BASE
from tests.schema.test_weights_origin_migration import _admin, _alembic, _pg_available, _query

_COLUMNS = (
    "SELECT column_name, data_type, is_nullable, character_maximum_length "
    "FROM information_schema.columns WHERE table_name = 'app_settings' ORDER BY column_name"
)
_PK = (
    "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid "
    "AND a.attnum = ANY(i.indkey) WHERE i.indrelid = 'app_settings'::regclass AND i.indisprimary"
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
    name = f"app_settings_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def test_025_creates_app_settings_and_comes_down(fresh_db):
    _alembic(fresh_db, "upgrade", "024")
    assert asyncio.run(_query(fresh_db, _COLUMNS)) == []

    _alembic(fresh_db, "upgrade", "025")
    assert asyncio.run(_query(fresh_db, _COLUMNS)) == [
        ("key", "character varying", "NO", 64),
        ("updated_at", "timestamp with time zone", "NO", None),
        ("value", "text", "NO", None),
        ("verified_at", "timestamp with time zone", "YES", None),
        ("verified_user", "character varying", "YES", 255),
    ]
    assert asyncio.run(_query(fresh_db, _PK)) == [("key",)]
    asyncio.run(_exec(fresh_db, "INSERT INTO app_settings (key, value) VALUES ('hf_token', 'v1:k:x')"))
    rows = asyncio.run(_query(fresh_db, "SELECT key, value, updated_at IS NOT NULL FROM app_settings"))
    assert rows == [("hf_token", "v1:k:x", True)]

    _alembic(fresh_db, "downgrade", "024")
    assert asyncio.run(_query(fresh_db, _COLUMNS)) == []
    _alembic(fresh_db, "upgrade", "head")
    assert asyncio.run(_query(fresh_db, _COLUMNS)) != []


def test_026_adds_last_check_error_and_comes_down(fresh_db):
    """A failed token check is stored with the setting (2026-10-18), so a restart keeps it."""
    # The module's database is shared, so this may start empty or at head: reach 024 either way.
    _alembic(fresh_db, "upgrade", "024")
    _alembic(fresh_db, "downgrade", "024")
    _alembic(fresh_db, "upgrade", "025")
    asyncio.run(_exec(fresh_db, "INSERT INTO app_settings (key, value) VALUES ('hf_token', 'v1:k:x')"))
    _alembic(fresh_db, "upgrade", "026")
    cols = {r[0]: r for r in asyncio.run(_query(fresh_db, _COLUMNS))}
    assert cols["last_check_error"] == ("last_check_error", "character varying", "YES", 32)
    rows = asyncio.run(_query(fresh_db, "SELECT key, last_check_error FROM app_settings"))
    assert rows == [("hf_token", None)], "an existing row must read as 'no failed check'"

    _alembic(fresh_db, "downgrade", "025")
    assert "last_check_error" not in {r[0] for r in asyncio.run(_query(fresh_db, _COLUMNS))}
    assert asyncio.run(_query(fresh_db, "SELECT key FROM app_settings")) == [("hf_token",)]
    _alembic(fresh_db, "upgrade", "head")
