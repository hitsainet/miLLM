"""Review of 040 P0 (2026-11-08): the bounds and the shutdown path of the old-name store.

Controls L09, L12 and L15 of the review survived the P0 suite: the shutdown flush's 2 s bound, the
empty-flush heartbeat (`last_flush_at`, CF-5) and the health read's 1 s bound could each be deleted
with every test green. `test_a_cancelled_flush_keeps_its_hits` pins the review's code finding: the
lifespan cancels the flush loop before the last flush, and a flush cancelled mid-write must put its
hits back so the shutdown flush writes them.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

import pytest

from miserver.core import legacy_names
from miserver.services import legacy_names_store as store
from tests.unit.batch_fixtures import batch_db  # noqa: F401

OLD_HOST, NEW_HOST = "k8s-millm.hitsai.local", "k8s-miserver.hitsai.local"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("MISERVER_STRICT_RESULTS", "MILLM_STRICT_RESULTS"):
        monkeypatch.delenv(name, raising=False)
    legacy_names.drain_hits()
    yield
    legacy_names.drain_hits()


class _Stuck:
    """A session factory whose session never opens: a database that accepts and never answers."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    def __call__(self):
        return self

    async def __aenter__(self):
        self.entered.set()
        await asyncio.Event().wait()

    async def __aexit__(self, *exc):
        return False


async def test_shutdown_flush_is_bounded(monkeypatch):
    monkeypatch.setattr(store, "SHUTDOWN_FLUSH_TIMEOUT_S", 0.05)
    legacy_names.record("host", OLD_HOST, NEW_HOST)
    started = asyncio.get_running_loop().time()
    await asyncio.wait_for(store.shutdown_flush(_Stuck()), timeout=2.0)
    assert asyncio.get_running_loop().time() - started < 1.0, "a stuck database held the shutdown"


async def test_an_empty_flush_is_still_a_heartbeat(monkeypatch):
    monkeypatch.setattr(store, "_LAST_FLUSH_AT", None)

    def never_called():
        raise AssertionError("nothing to write: the database is not touched")

    assert await store.flush_once(never_called) == 0
    assert store._LAST_FLUSH_AT is not None, "an empty flush must stamp last_flush_at (CF-5)"


async def test_the_health_read_is_bounded(monkeypatch):
    monkeypatch.setattr(store, "READ_TIMEOUT_S", 0.05)

    async def stuck(_factory):
        await asyncio.Event().wait()

    monkeypatch.setattr(store, "_select_all", stuck)
    block, meta = await asyncio.wait_for(store.read_block(object()), timeout=2.0)
    assert meta["error"] == "TimeoutError" and block["in_use"] == []


async def test_a_cancelled_flush_keeps_its_hits(batch_db):  # noqa: F811
    """The lifespan order: cancel the loop, then `shutdown_flush`. A flush in flight when the loop is
    cancelled has already drained its hits; they must go back so the last flush writes them."""
    legacy_names.record("host", OLD_HOST, NEW_HOST)
    stuck = _Stuck()
    task = asyncio.create_task(store.flush_once(stuck))
    await asyncio.wait_for(stuck.entered.wait(), timeout=2.0)
    assert legacy_names.peek_hits() == {}, "precondition: the in-flight flush holds the hits"
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await store.shutdown_flush(batch_db)
    from sqlalchemy import select

    from miserver.db.models.legacy_name_use import LegacyNameUse

    async with batch_db() as session:
        rows = (await session.execute(select(LegacyNameUse))).scalars().all()
    assert [(r.kind, r.old_name, r.count) for r in rows] == [("host", OLD_HOST, 1)], (
        "a hit drained by a cancelled flush was lost at shutdown")
    assert all(isinstance(r.last_seen_at, datetime) for r in rows)
    _ = UTC
