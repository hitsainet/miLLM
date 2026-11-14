"""T07, T08: the persistent record of old-name use (040 FR-040.10; TID §3.2).

T07: a flush writes counts and the earliest/latest times; a database error keeps the hits for the
next flush and fails nothing; `last_seen` is read back from the row after a restart (memory
cleared); the `meta` row is `observing_since`.
T08 (wiring): the REAL lifespan starts a task named `legacy-names-flush` that flushes through the
patched session factory, and flushes exactly once more at shutdown.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from miserver.core import legacy_names
from miserver.db.models.legacy_name_use import LegacyNameUse
from miserver.services import legacy_names_store as store
from tests.unit.batch_fixtures import batch_db  # noqa: F401

OLD_HOST, NEW_HOST = "k8s-millm.hitsai.local", "k8s-miserver.hitsai.local"
T0 = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("MISERVER_STRICT_RESULTS", "MILLM_STRICT_RESULTS"):
        monkeypatch.delenv(name, raising=False)
    legacy_names.drain_hits()
    yield
    legacy_names.drain_hits()


def _hit_at(moment, monkeypatch, kind="host", old=OLD_HOST, new=NEW_HOST):
    monkeypatch.setattr(legacy_names, "_now", lambda: moment)
    legacy_names.record(kind, old, new)


async def _rows(factory):
    from sqlalchemy import select

    async with factory() as session:
        return {(r.kind, r.old_name): r for r in (await session.execute(select(LegacyNameUse))).scalars()}


async def test_flush_writes_counts_and_min_max_times(batch_db, monkeypatch):  # noqa: F811
    _hit_at(T0, monkeypatch)
    _hit_at(T0 + timedelta(minutes=5), monkeypatch)
    assert await store.flush_once(batch_db) == 1
    _hit_at(T0 + timedelta(minutes=9), monkeypatch)
    assert await store.flush_once(batch_db) == 1
    row = (await _rows(batch_db))[("host", OLD_HOST)]
    assert (row.new_name, row.count) == (NEW_HOST, 3)
    assert store._aware(row.first_seen_at) == T0
    assert store._aware(row.last_seen_at) == T0 + timedelta(minutes=9)
    assert legacy_names.peek_hits() == {}


async def test_flush_failure_keeps_hits_and_answers_the_request(monkeypatch):
    _hit_at(T0, monkeypatch)

    def broken():
        raise RuntimeError("database gone")

    assert await store.flush_once(broken) == 0  # never raises
    kept = legacy_names.peek_hits()
    assert kept[("host", OLD_HOST)].count == 1, "the hit waits for the next flush"


async def test_last_seen_survives_a_restart(batch_db, monkeypatch):  # noqa: F811
    seen = datetime.now(UTC) - timedelta(hours=2)
    _hit_at(seen, monkeypatch)
    monkeypatch.setattr(legacy_names, "_now", lambda: datetime.now(UTC))
    await store.flush_once(batch_db)
    legacy_names.drain_hits()  # a restart: memory is empty, the row is kept
    block, meta = await store.read_block(batch_db)
    assert meta["error"] is None
    assert block["in_use"] == [{"kind": "host", "old": OLD_HOST, "new": NEW_HOST, "last_seen": seen}]


async def test_meta_row_is_observing_since(batch_db):  # noqa: F811
    async with batch_db() as session:
        session.add(LegacyNameUse(kind="meta", old_name="observing_since", new_name=None,
                                  first_seen_at=T0, last_seen_at=T0, count=0))
        await session.commit()
    block, meta = await store.read_block(batch_db)
    assert meta["observing_since"] == T0
    assert block["in_use"] == [], "the meta row is not a name in use"


async def test_legacy_names_flush_is_started_by_lifespan(monkeypatch, batch_db):  # noqa: F811
    import miserver.db.base as db_base
    import miserver.main as main

    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    monkeypatch.setattr(main, "start_batch_api", AsyncMock())
    monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
    monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
    monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    monkeypatch.setattr(store, "FLUSH_INTERVAL_S", 0.01)
    calls: list = []
    flushed = asyncio.Event()

    async def spy(factory):
        calls.append(factory)
        flushed.set()
        return 0

    monkeypatch.setattr(store, "flush_once", spy)
    async with main.lifespan(MagicMock()):
        tasks = [t for t in asyncio.all_tasks() if t.get_name() == "legacy-names-flush"]
        assert len(tasks) == 1, "the lifespan must start exactly one flush task"
        await asyncio.wait_for(flushed.wait(), timeout=5)
        assert calls[0] is batch_db, "the loop flushes through the CURRENT session factory"
        during = len(calls)
        monkeypatch.setattr(store, "FLUSH_INTERVAL_S", 3600.0)
        await asyncio.sleep(0.05)
        during = len(calls)
    assert len(calls) == during + 1, "exactly one more flush at shutdown"
    assert calls[-1] is batch_db
    assert all(t.done() for t in tasks), "the flush loop outlived the lifespan"


async def test_the_loop_flush_is_bounded_and_keeps_its_hits(monkeypatch):
    """Review of 040 P0: a hung database must not hold the drained hits forever. The loop's flush
    times out, the timed-out flush puts its hits back, and the loop flushes again (the database is
    entered a second time, with the same hit); a final cancel also gives the hits back."""

    class Hung:
        def __init__(self) -> None:
            self.entries: list[dict] = []
            self.second = asyncio.Event()

        def __call__(self):
            return self

        async def __aenter__(self):
            self.entries.append(legacy_names.peek_hits())
            if len(self.entries) >= 2:
                self.second.set()
            await asyncio.Event().wait()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(store, "FLUSH_INTERVAL_S", 0.01)
    monkeypatch.setattr(store, "LOOP_FLUSH_TIMEOUT_S", 0.05)
    _hit_at(T0, monkeypatch)
    hung = Hung()
    loop = asyncio.create_task(store.flush_loop(lambda: hung))
    await asyncio.wait_for(hung.second.wait(), timeout=2.0)  # unbounded: never a second flush
    assert not loop.done(), "a timed-out flush must not end the loop"
    loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop
    assert legacy_names.peek_hits()[("host", OLD_HOST)].count == 1, "the hits came back, once"
