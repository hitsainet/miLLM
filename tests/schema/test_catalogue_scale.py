"""The catalogue at scale, on PostgreSQL (039 FR-39.15, TD39-29; FTASKS 5.6).

10,000 imported probes over 2,000 generated behaviour keys (and 1,000 with no behaviour), written
with one bulk INSERT. Through the live app:

* `GET /api/probes` and `GET /api/probes/behaviours`, walked by their cursors, return every item
  exactly once, in a stable order (behaviour key — none last — then id);
* the number of SQL statements per page is CONSTANT (a `before_cursor_execute` listener counts
  them), so a per-row lookup on a page is red;
* `GET /api/probes/status` is the same size at 10 imported probes as at 10,000 with the same
  armed set — the catalogue is a count there, never a list.

PostgreSQL only (`NULLS LAST` ordering, the production planner), so it lives under `tests/schema/`
and runs in the CI `schema` job: skipped loudly without a server, REQUIRED under `MILLM_REQUIRE_PG=1`.

Control: a per-row lookup on a page (e.g. `repository.get` per summary) → the query-count test red.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from tests.schema.test_schema_guards import PG_BASE
from tests.schema.test_weights_origin_migration import _admin, _pg_available
from tests.support import behaviour_factory as bf

N_KEYED, N_KEYS, N_LOOSE = 9_000, 2_000, 1_000
#: Statements one page may run: the page query and, for behaviours, its rows query (plus the
#: session's own). Measured 2026-10-24; a per-row lookup adds one per item.
PAGE_QUERY_BOUND = 4


@pytest.fixture(scope="module")
def pg_db():
    if not asyncio.run(_pg_available()):
        message = f"PostgreSQL is not reachable at {PG_BASE}"
        if os.environ.get("MILLM_REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MILLM_REQUIRE_PG=1 — the scale test did NOT run")
        pytest.skip(f"{message} — the catalogue scale test needs a real server")
    name = f"scale_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield f"{PG_BASE}/{name}"
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def _rows(n_keyed, n_keys, n_loose):
    keys = sorted(bf.random_key() for _ in range(n_keys))
    definition = json.dumps({"name": "x"})
    out = []
    for i in range(n_keyed + n_loose):
        key = keys[i % n_keys] if i < n_keyed else None
        out.append({"id": f"pr_{i:012x}", "name": f"p-{i}", "definition": definition,
                    "hf_id": "tiny/llama", "d_model": 16, "n_layers": 2, "layer": i % 2,
                    "rule": "mean", "scope": "all", "rung": 2, "behaviour_key": key,
                    "behaviour_version": 1 if key else None})
    return out, keys


async def _setup(url, rows):
    from millm.db.base import Base
    import millm.db.models  # noqa: F401 - every table

    # NullPool: the setup runs in one event loop and the app in the TestClient's own; asyncpg
    # connections cannot cross loops, so none is kept between them.
    engine = create_async_engine(url, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(sa.text(
            "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, rule, "
            "scope, rung, behaviour_key, behaviour_version) VALUES (:id, :name, "
            "CAST(:definition AS jsonb), :hf_id, :d_model, :n_layers, :layer, :rule, :scope, "
            ":rung, :behaviour_key, :behaviour_version)"), rows)
    return engine


def _client(engine):
    from fastapi.testclient import TestClient

    from millm.api.dependencies import get_db
    from millm.main import create_app

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _db():
        async with factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = _db
    return TestClient(app)


class _Counter:
    def __init__(self, engine):
        self.n = 0
        sa.event.listen(engine.sync_engine, "before_cursor_execute", self._count)

    def _count(self, *_a, **_k):
        self.n += 1


def _walk(client, path, counter, limit):
    seen, per_page, cursor = [], [], None
    while True:
        before = counter.n
        params = {"limit": limit, **({"cursor": cursor} if cursor else {})}
        body = client.get(path, params=params).json()
        per_page.append(counter.n - before)
        seen.extend(body["data"])
        cursor = body["meta"]["next"]
        if cursor is None:
            return seen, per_page


def test_every_list_walks_the_whole_catalogue_once_with_constant_queries(pg_db):
    rows, keys = _rows(N_KEYED, N_KEYS, N_LOOSE)
    engine = asyncio.run(_setup(pg_db, rows))
    try:
        client = _client(engine)
        counter = _Counter(engine)
        probes, per_page = _walk(client, "/api/probes", counter, 500)
        assert len(probes) == len(rows) and len({p["id"] for p in probes}) == len(rows)
        key_of = {r["id"]: r["behaviour_key"] for r in rows}  # the projected column
        order = [(key_of[p["id"]] or "\uffff", p["id"]) for p in probes]
        assert order == sorted(order), "the order is not (behaviour key, nulls last; id)"
        assert len(set(per_page)) == 1, f"queries per page vary: {sorted(set(per_page))}"
        # ⚠ AND SMALL: a per-row lookup on a full page is constant across full pages too
        # (control S1-45 survived "constant" alone), so the count itself is bounded.
        assert max(per_page) <= PAGE_QUERY_BOUND, per_page

        behaviours, per_page = _walk(client, "/api/probes/behaviours", counter, 250)
        assert [b["key"] for b in behaviours] == keys
        assert sum(b["imported"] for b in behaviours) == N_KEYED
        assert len(set(per_page[:-1])) == 1, f"queries per page vary: {per_page}"
        assert max(per_page) <= PAGE_QUERY_BOUND, per_page
    finally:
        asyncio.run(engine.dispose())


def test_status_does_not_grow_with_the_catalogue(pg_db):
    """The response is the same size at 10 imported probes as at 10,000 — and no statement it runs
    reads the catalogue unbounded (control S1-46: loading every row and filtering in Python kept the
    response the same size, so the size alone could not see it)."""
    sizes = []
    for n_keyed, n_keys, n_loose in ((10, 5, 0), (N_KEYED, N_KEYS, N_LOOSE)):
        rows, _keys = _rows(n_keyed, n_keys, n_loose)
        engine = asyncio.run(_setup(pg_db, rows))
        statements: list[str] = []
        sa.event.listen(engine.sync_engine, "before_cursor_execute",
                        lambda _c, _cur, stmt, *_a: statements.append(" ".join(stmt.split())))
        try:
            status = _client(engine).get("/api/probes/status").json()["data"]
            unbounded = [s for s in statements
                         if " FROM probes" in s and " WHERE " not in s and "count(" not in s]
            assert unbounded == [], unbounded
            for volatile in ("catalogue", "imported_count", "events_recorded",
                             "last_retention_pass_at"):
                status.pop(volatile, None)
            sizes.append(len(json.dumps(status, sort_keys=True, default=str)))
        finally:
            asyncio.run(engine.dispose())
    assert sizes[0] == sizes[1], sizes


def test_five_hundred_rule_sets_walk_once_with_constant_queries(pg_db):
    """039 S2 (FTASKS 12.8): `GET /api/rule-sets` over 500 imported sets, by cursor."""
    from tests.support import rule_set_factory as rf

    engine = asyncio.run(_setup(pg_db, _rows(1, 1, 0)[0]))

    async def seed():
        async with engine.begin() as conn:
            await conn.execute(sa.text(
                "INSERT INTO rule_sets (id, name, behaviour_key, origin, digest, document) "
                "VALUES (:id, :name, :key, 'authored', :digest, CAST(:doc AS jsonb))"), [
                {"id": f"rs_{i:012x}", "name": f"s{i}", "key": f"b-{i % 37:08x}",
                 "digest": f"{i:064x}", "doc": json.dumps(rf.document(f"b-{i % 37:08x}"))}
                for i in range(500)])

    asyncio.run(seed())
    try:
        client = _client(engine)
        counter = _Counter(engine)
        sets, per_page = _walk(client, "/api/rule-sets", counter, 100)
        assert len(sets) == 500 and len({s["id"] for s in sets}) == 500
        order = [(s["behaviour_key"], s["id"]) for s in sets]
        assert order == sorted(order)
        assert len(set(per_page)) == 1 and max(per_page) <= PAGE_QUERY_BOUND, per_page
    finally:
        asyncio.run(engine.dispose())
