"""Migration 024 (one active attachment per SAE and layer) goes up and comes down on PostgreSQL.

Up: two SAEs can be active at once — the second attach that answered 500 on the RTX 3090 — while
one (SAE, layer) still cannot be active twice. Down: the single-SAE index returns and only the
most recent active row survives, as the old schema requires.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from tests.schema.test_schema_guards import PG_BASE
from tests.schema.test_weights_origin_migration import _admin, _alembic, _pg_available, _query

_INDEXES = (
    "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'sae_attachments' "
    "AND indexname IN ('idx_single_active_attachment', 'uq_active_attachment_sae_layer') "
    "ORDER BY indexname"
)
_ACTIVE = "SELECT sae_id, layer FROM sae_attachments WHERE is_active ORDER BY id"


@pytest.fixture(scope="module")
def fresh_db():
    if not asyncio.run(_pg_available()):
        if os.environ.get("MILLM_REQUIRE_PG") == "1":
            pytest.fail("PostgreSQL is not reachable and MILLM_REQUIRE_PG=1")
        pytest.skip("PostgreSQL is not reachable — a migration round trip needs a real server")
    name = f"attach_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


async def _exec(db: str, *statements: str) -> None:
    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.begin() as conn:
            for statement in statements:
                await conn.execute(sa.text(statement))
    finally:
        await engine.dispose()


_SEED = (
    "INSERT INTO models (id, name, source, quantization, cache_path, status, trust_remote_code)"
    " VALUES (1, 'm', 'huggingface', 'FP16', '/tmp/m', 'loaded', false)",
    "INSERT INTO saes (id, repository_id, name, d_in, d_sae, cache_path, status) VALUES"
    " ('sae-a', 'o/a', 'a', 16, 32, '/tmp/a', 'attached'),"
    " ('sae-b', 'o/b', 'b', 16, 32, '/tmp/b', 'attached')",
)


def _attach(sae_id: str, layer: int, at: str) -> str:
    return (
        "INSERT INTO sae_attachments (sae_id, model_id, layer, attached_at, is_active)"
        f" VALUES ('{sae_id}', 1, {layer}, '{at}', true)"
    )


def test_024_goes_up_admits_several_saes_and_comes_down(fresh_db):
    _alembic(fresh_db, "upgrade", "head")
    rows = asyncio.run(_query(fresh_db, _INDEXES))
    assert [r[0] for r in rows] == ["uq_active_attachment_sae_layer"]
    assert "(sae_id, layer)" in rows[0][1] and "WHERE (is_active = true)" in rows[0][1]

    asyncio.run(_exec(fresh_db, *_SEED, _attach("sae-a", 20, "2026-10-16 23:00"),
                      _attach("sae-b", 25, "2026-10-16 23:01")))
    with pytest.raises(IntegrityError):
        asyncio.run(_exec(fresh_db, _attach("sae-a", 20, "2026-10-16 23:02")))
    assert asyncio.run(_query(fresh_db, _ACTIVE)) == [("sae-a", 20), ("sae-b", 25)]

    _alembic(fresh_db, "downgrade", "023")
    assert [r[0] for r in asyncio.run(_query(fresh_db, _INDEXES))] == [
        "idx_single_active_attachment"
    ]
    assert asyncio.run(_query(fresh_db, _ACTIVE)) == [("sae-b", 25)]

    _alembic(fresh_db, "upgrade", "head")
    assert [r[0] for r in asyncio.run(_query(fresh_db, _INDEXES))] == [
        "uq_active_attachment_sae_layer"
    ]
