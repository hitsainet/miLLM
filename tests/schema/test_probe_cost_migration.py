"""Migration 030 (`probes.cost`, Feature 37) goes up and comes down on PostgreSQL."""

from __future__ import annotations

import asyncio

from tests.schema.test_probe_sessions_migration import _column, fresh_db  # noqa: F401
from tests.schema.test_weights_origin_migration import _alembic, _query


def test_030_adds_and_drops_the_cost_column(fresh_db):  # noqa: F811
    _alembic(fresh_db, "upgrade", "029")
    assert asyncio.run(_query(fresh_db, _column("probes", "cost"))) == []
    _alembic(fresh_db, "upgrade", "030")
    assert asyncio.run(_query(fresh_db, (
        "SELECT data_type, is_nullable FROM information_schema.columns "
        "WHERE table_name = 'probes' AND column_name = 'cost'"
    ))) == [("jsonb", "YES")]
    _alembic(fresh_db, "downgrade", "029")
    assert asyncio.run(_query(fresh_db, _column("probes", "cost"))) == []
    _alembic(fresh_db, "upgrade", "head")
