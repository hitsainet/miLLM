"""Migration 027 (one served name, one row) on PostgreSQL, with the live D56 rows.

Rows 7 (jevify Q8, bitsandbytes) and 13 (jevify FP16) shared `jevify-gemma4-e4b`. After 027 the
FP16 row keeps the name, the Q8 row is `jevify-gemma4-e4b:Q8`, every already-unique name is
untouched (an existing client's name must not move), and a second row with one name is refused by
`uq_models_name`.
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

_INSERT = (
    "INSERT INTO models (id, name, source, repo_id, quantization, cache_path, status, gguf_label,"
    " locked, trust_remote_code, created_at, updated_at) VALUES"
    " (:id, :name, 'huggingface', :repo, :quant, :cache, 'ready', :label, false, false, now(),"
    " now())"
)

#: (id, name, repo, quantization, gguf label)
_ROWS = (
    (7, "jevify-gemma4-e4b", "kushalpatil/jevify-gemma4-e4b", "Q8", ""),
    (13, "jevify-gemma4-e4b", "kushalpatil/jevify-gemma4-e4b", "FP16", ""),
    (1, "LFM2.5-1.2B-Instruct", "LiquidAI/LFM2.5-1.2B-Instruct", "FP16", ""),
    (20, "solo-q4", "o/solo-q4", "Q4", ""),  # unique and untagged: must NOT be renamed
    (43, "g-GGUF:Q4_K_M", "o/g-GGUF", "Q4", "Q4_K_M"),
)


async def _exec(db: str, sql: str, params: dict | None = None) -> None:
    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text(sql), params or {})
    finally:
        await engine.dispose()


@pytest.fixture
def fresh_db():
    if not asyncio.run(_pg_available()):
        message = f"PostgreSQL is not reachable at {PG_BASE}"
        if os.environ.get("MILLM_REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MILLM_REQUIRE_PG=1 — this guard did NOT run")
        pytest.skip(f"{message} — a migration round trip needs a real server")
    name = f"model_names_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def test_027_tags_the_duplicate_keeps_unique_names_and_enforces(fresh_db):
    _alembic(fresh_db, "upgrade", "026")
    for row_id, name, repo, quant, label in _ROWS:
        asyncio.run(_exec(fresh_db, _INSERT, {"id": row_id, "name": name, "repo": repo,
                                              "quant": quant, "cache": f"c{row_id}",
                                              "label": label}))

    _alembic(fresh_db, "upgrade", "027")

    names = dict(asyncio.run(_query(fresh_db, "SELECT id, name FROM models")))
    assert names == {
        7: "jevify-gemma4-e4b:Q8",
        13: "jevify-gemma4-e4b",
        1: "LFM2.5-1.2B-Instruct",
        20: "solo-q4",
        43: "g-GGUF:Q4_K_M",
    }
    with pytest.raises(IntegrityError):
        asyncio.run(_exec(fresh_db, _INSERT, {"id": 99, "name": "solo-q4", "repo": "x/solo-q4",
                                              "quant": "FP16", "cache": "c99", "label": ""}))

    _alembic(fresh_db, "downgrade", "026")
    indexes = asyncio.run(_query(
        fresh_db, "SELECT indexname FROM pg_indexes WHERE tablename = 'models'"))
    assert ("uq_models_name",) not in indexes
    _alembic(fresh_db, "upgrade", "027")  # idempotent: nothing left to rename
    assert dict(asyncio.run(_query(fresh_db, "SELECT id, name FROM models"))) == names
