"""Migration 022 (`probe_events.served_model`, addendum D-36) must go up and come down.

On PostgreSQL, where the migration runs in production. The ORM declares the same column;
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

_COLUMN = (
    "SELECT is_nullable, character_maximum_length, column_default "
    "FROM information_schema.columns "
    "WHERE table_name = 'probe_events' AND column_name = 'served_model'"
)


async def _insert_event(db: str, served_model: str | None, *, with_column: bool) -> None:
    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer,"
                    " rule, scope, rung) VALUES ('pr_served', 'x', '{}'::jsonb, 'hf/x', 8, 16, 1,"
                    " 'mean', 'all', 2) ON CONFLICT DO NOTHING"
                )
            )
            if with_column:
                await conn.execute(
                    sa.text(
                        "INSERT INTO probe_events (probe_id, scored, served_model)"
                        " VALUES ('pr_served', true, :served)"
                    ),
                    {"served": served_model},
                )
            else:
                await conn.execute(
                    sa.text("INSERT INTO probe_events (probe_id, scored) VALUES ('pr_served', true)")
                )
    finally:
        await engine.dispose()


@pytest.fixture(scope="module")
def fresh_db():
    if not asyncio.run(_pg_available()):
        message = f"PostgreSQL is not reachable at {PG_BASE}"
        if os.environ.get("MILLM_REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MILLM_REQUIRE_PG=1 — this guard did NOT run")
        pytest.skip(f"{message} — a migration round trip needs a real server")
    name = f"probe_served_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def test_022_goes_up_nullable_with_no_default_and_comes_down(fresh_db):
    _alembic(fresh_db, "upgrade", "021")
    assert asyncio.run(_query(fresh_db, _COLUMN)) == []
    # A row written BEFORE the column exists: after the upgrade it must read NULL (not
    # recorded), never a backfilled name.
    asyncio.run(_insert_event(fresh_db, None, with_column=False))

    _alembic(fresh_db, "upgrade", "022")
    assert asyncio.run(_query(fresh_db, _COLUMN)) == [("YES", 255, None)]
    assert asyncio.run(_query(fresh_db, "SELECT served_model FROM probe_events")) == [(None,)]
    asyncio.run(_insert_event(fresh_db, "base:r1:adapter-200", with_column=True))
    rows = asyncio.run(_query(fresh_db, "SELECT served_model FROM probe_events ORDER BY id"))
    assert rows == [(None,), ("base:r1:adapter-200",)]

    _alembic(fresh_db, "downgrade", "021")
    assert asyncio.run(_query(fresh_db, _COLUMN)) == []
    assert asyncio.run(_query(fresh_db, "SELECT count(*) FROM probe_events")) == [(2,)], (
        "the downgrade must keep the events and drop only the column"
    )
