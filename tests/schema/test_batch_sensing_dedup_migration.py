"""Migration 023 (batch sensing dedup indexes, 026 F-1) goes up and comes down on PostgreSQL."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from tests.schema.test_weights_origin_migration import _admin, _alembic, _pg_available, _query

_INDEXES = (
    "SELECT indexname, indexdef FROM pg_indexes WHERE indexname IN "
    "('uq_sensing_events_batch_line', 'uq_circuit_edge_events_batch_line') ORDER BY indexname"
)


@pytest.fixture(scope="module")
def fresh_db():
    if not asyncio.run(_pg_available()):
        if os.environ.get("MILLM_REQUIRE_PG") == "1":
            pytest.fail("PostgreSQL is not reachable and MILLM_REQUIRE_PG=1")
        pytest.skip("PostgreSQL is not reachable — a migration round trip needs a real server")
    name = f"dedup_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def test_023_goes_up_partial_unique_and_comes_down(fresh_db):
    _alembic(fresh_db, "upgrade", "head")
    rows = asyncio.run(_query(fresh_db, _INDEXES))
    assert [r[0] for r in rows] == [
        "uq_circuit_edge_events_batch_line", "uq_sensing_events_batch_line",
    ]
    for _name, definition in rows:
        assert definition.startswith("CREATE UNIQUE INDEX")
        assert "WHERE (batch_id IS NOT NULL)" in definition

    _alembic(fresh_db, "downgrade", "022")
    assert asyncio.run(_query(fresh_db, _INDEXES)) == []
    _alembic(fresh_db, "upgrade", "head")
    assert len(asyncio.run(_query(fresh_db, _INDEXES))) == 2
