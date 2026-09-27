"""Migration 016 must go down as well as up, on a real PostgreSQL server.

A downgrade nobody runs is a downgrade nobody knows is broken, and you find out during a rollback —
the one moment when a second failure is least affordable. This upgrades to head, downgrades to 015,
checks the probe tables are genuinely gone, and upgrades again.

⚠ It runs on PostgreSQL, not SQLite. 016 creates JSONB columns via
`sa.JSON().with_variant(postgresql.JSONB(), "postgresql")`; the SQLite unit suite exercises the
JSON side of that variant and would not notice a JSONB-specific mistake.
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
from sqlalchemy.ext.asyncio import create_async_engine

REPO = Path(__file__).resolve().parents[2]

# ⚠ IMPORTED, not redeclared. My first version repeated the default with the wrong password
# ("postgres" instead of "devpassword") and reported PostgreSQL unreachable while the sibling
# guard in this same directory was connecting to it happily. Two defaults for one server is the
# bug; one is the fix.
from tests.schema.test_schema_guards import PG_BASE  # noqa: E402

PROBE_TABLES = ("probes", "probe_events")


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


async def _tables(db: str) -> set[str]:
    engine = create_async_engine(f"{PG_BASE}/{db}")
    async with engine.connect() as conn:
        rows = await conn.execute(
            sa.text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        names = set(rows.scalars().all())
    await engine.dispose()
    return names


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
    name = f"probe_migration_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def test_016_goes_up_down_and_up_again(fresh_db):
    _alembic(fresh_db, "upgrade", "head")
    after_up = asyncio.run(_tables(fresh_db))
    for table in PROBE_TABLES:
        assert table in after_up, f"{table} missing after upgrade"

    _alembic(fresh_db, "downgrade", "015")
    after_down = asyncio.run(_tables(fresh_db))
    for table in PROBE_TABLES:
        assert table not in after_down, (
            f"{table} survived the downgrade — a rollback would leave a table the ORM at 015 "
            f"knows nothing about"
        )
    # The downgrade must not take anything else with it.
    assert "models" in after_down and "circuits" in after_down

    _alembic(fresh_db, "upgrade", "head")
    assert asyncio.run(_tables(fresh_db)) == after_up


def test_the_cascade_is_real_on_postgres(fresh_db):
    """⚠ SQLite ignored this until 2026-09-27, so assert it where it is actually enforced.

    Deleting a probe must take its events. If the FK were declared without ON DELETE CASCADE the
    delete would raise a foreign-key violation here — on SQLite it silently orphaned the rows.
    """
    _alembic(fresh_db, "upgrade", "head")

    async def exercise() -> int:
        engine = create_async_engine(f"{PG_BASE}/{fresh_db}")
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer,"
                    " rule, scope, rung) VALUES ('pr_x', 'x', '{}'::jsonb, 'hf/x', 8, 16, 1,"
                    " 'mean', 'all', 2)"
                )
            )
            await conn.execute(
                sa.text("INSERT INTO probe_events (probe_id, scored) VALUES ('pr_x', true)")
            )
            await conn.execute(sa.text("DELETE FROM probes WHERE id = 'pr_x'"))
            remaining = await conn.execute(
                sa.text("SELECT count(*) FROM probe_events WHERE probe_id = 'pr_x'")
            )
            count = int(remaining.scalar_one())
        await engine.dispose()
        return count

    assert asyncio.run(exercise()) == 0
