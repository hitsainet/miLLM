"""Migration 021 (`models.weights_origin`) must go up, enforce its CHECK, and come down.

Runs on PostgreSQL because that is where the migration runs in production and where the CHECK is
enforced by the migration rather than by `create_all`. The ORM declares the same constraint;
`test_schema_guards.py` compares the two.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from tests.schema.test_schema_guards import PG_BASE

REPO = Path(__file__).resolve().parents[2]


async def _pg_available() -> bool:
    try:
        engine = create_async_engine(f"{PG_BASE}/postgres", isolation_level="AUTOCOMMIT")
        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception:
        return False


async def _admin(statement: str) -> None:
    engine = create_async_engine(f"{PG_BASE}/postgres", isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        await conn.execute(sa.text(statement))
    await engine.dispose()


async def _query(db: str, sql: str) -> list:
    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.connect() as conn:
            return list((await conn.execute(sa.text(sql))).all())
    finally:
        await engine.dispose()


async def _insert_model(db: str, origin: str | None) -> None:
    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO models (name, source, quantization, cache_path, status,"
                    " gguf_label, locked, trust_remote_code, created_at, updated_at,"
                    " local_path, weights_origin) VALUES (:name, 'local', 'FP16', 'c', 'ready',"
                    " '', false, false, now(), now(), :path, :origin)"
                ),
                {"name": f"b:r:{uuid.uuid4().hex[:6]}", "path": uuid.uuid4().hex, "origin": origin},
            )
    finally:
        await engine.dispose()


def _alembic(db: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_URL": f"{PG_BASE}/{db}"},
        timeout=600,
    )
    assert result.returncode == 0, f"alembic {' '.join(args)} failed:\n{result.stderr[-3000:]}"


@pytest.fixture(scope="module")
def fresh_db():
    if not asyncio.run(_pg_available()):
        message = f"PostgreSQL is not reachable at {PG_BASE}"
        if os.environ.get("MILLM_REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MILLM_REQUIRE_PG=1 — this guard did NOT run")
        pytest.skip(f"{message} — a migration round trip needs a real server")
    name = f"weights_origin_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


_COLUMN = (
    "SELECT is_nullable FROM information_schema.columns "
    "WHERE table_name = 'models' AND column_name = 'weights_origin'"
)
_CHECK = "SELECT conname FROM pg_constraint WHERE conname = 'ck_models_weights_origin'"


def test_021_goes_up_enforces_and_comes_down(fresh_db):
    _alembic(fresh_db, "upgrade", "head")
    assert asyncio.run(_query(fresh_db, _COLUMN)) == [("YES",)]
    assert len(asyncio.run(_query(fresh_db, _CHECK))) == 1

    for origin in (None, "checkpoint", "instantiated"):
        asyncio.run(_insert_model(fresh_db, origin))
    with pytest.raises(IntegrityError, match="ck_models_weights_origin"):
        asyncio.run(_insert_model(fresh_db, "adapter"))

    _alembic(fresh_db, "downgrade", "020")
    assert asyncio.run(_query(fresh_db, _COLUMN)) == []
    assert asyncio.run(_query(fresh_db, _CHECK)) == []
    rows = asyncio.run(_query(fresh_db, "SELECT count(*) FROM models"))
    assert rows == [(3,)], "the downgrade must keep the rows and drop only the column"

    _alembic(fresh_db, "upgrade", "head")
    assert asyncio.run(_query(fresh_db, _COLUMN)) == [("YES",)]
