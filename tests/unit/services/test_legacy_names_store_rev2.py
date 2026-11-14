"""Rev2 review of 040 P0 (2026-11-08): two lines of the cancelled-flush fix that survived.

* S2: `LOOP_FLUSH_TIMEOUT_S` could be any value (the loop test patches it), e.g. an hour, and a hung
  database would then hold the drained hits that long. The bound must be shorter than the flush
  interval, and no shorter than the shutdown flush's.
* S4: the cancel handler could `return 0` instead of re-raising. The loop then survives its own
  cancellation: the lifespan's `await legacy_flush` waits on a loop that never ends, so shutdown
  hangs. The existing tests suppress `CancelledError`, so they cannot tell (one hung instead).
"""

from __future__ import annotations

import asyncio

import pytest

from miserver.core import legacy_names
from miserver.services import legacy_names_store as store
from tests.unit.batch_fixtures import batch_db  # noqa: F401

OLD_HOST, NEW_HOST = "k8s-millm.hitsai.local", "k8s-miserver.hitsai.local"


@pytest.fixture(autouse=True)
def _clean():
    legacy_names.drain_hits()
    yield
    legacy_names.drain_hits()


class _Stuck:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    def __call__(self):
        return self

    async def __aenter__(self):
        self.entered.set()
        await asyncio.Event().wait()

    async def __aexit__(self, *exc):
        return False


def test_the_loop_flush_bound_is_shorter_than_the_interval():
    assert 0 < store.SHUTDOWN_FLUSH_TIMEOUT_S <= store.LOOP_FLUSH_TIMEOUT_S < store.FLUSH_INTERVAL_S


async def test_a_cancelled_flush_re_raises_and_the_loop_ends(monkeypatch):
    legacy_names.record("host", OLD_HOST, NEW_HOST)
    stuck = _Stuck()
    task = asyncio.create_task(store.flush_once(stuck))
    await asyncio.wait_for(stuck.entered.wait(), timeout=5.0)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=5.0)
    assert done, "the cancelled flush did not finish"
    assert task.cancelled(), "a cancelled flush must re-raise CancelledError, not return"
    assert legacy_names.peek_hits()[("host", OLD_HOST)].count == 1

    # The loop, cancelled mid-flush, ends (the lifespan awaits it before the shutdown flush).
    monkeypatch.setattr(store, "FLUSH_INTERVAL_S", 0.0)
    stuck = _Stuck()
    loop = asyncio.create_task(store.flush_loop(lambda: stuck))
    await asyncio.wait_for(stuck.entered.wait(), timeout=5.0)
    loop.cancel()
    done, _ = await asyncio.wait({loop}, timeout=5.0)
    assert done and loop.cancelled(), "the flush loop survived its cancellation: shutdown would hang"


async def test_the_shutdown_flush_runs_only_after_the_loop_has_ended(monkeypatch, batch_db):  # noqa: F811
    """S5: the lifespan's ORDER is what makes the cancel fix work. Cancelling the loop first lets an
    in-flight flush give its hits back BEFORE the last flush drains; the other order (or both at
    once) loses them, since nothing flushes after the shutdown flush. Every test stayed green with
    the two swapped."""
    from unittest.mock import AsyncMock, MagicMock

    import miserver.db.base as db_base
    import miserver.main as main

    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    monkeypatch.setattr(main, "start_batch_api", AsyncMock())
    monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
    monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
    monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    monkeypatch.setattr(store, "FLUSH_INTERVAL_S", 3600.0)
    outside_loop: list[bool] = []

    async def spy(factory):
        if asyncio.current_task().get_name() != "legacy-names-flush":
            pending = [t for t in asyncio.all_tasks() if t.get_name() == "legacy-names-flush"]
            outside_loop.append(not pending)
        return 0

    monkeypatch.setattr(store, "flush_once", spy)
    async with main.lifespan(MagicMock()):
        assert [t for t in asyncio.all_tasks() if t.get_name() == "legacy-names-flush"]
    assert outside_loop == [True], (
        "the shutdown flush ran while the flush loop was still alive" if outside_loop else
        "no shutdown flush ran")
