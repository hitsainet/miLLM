"""Feature 26 task 5.10 / M11: the batch API is started by `lifespan`, by RUNNING it.

The lifespan is executed for real with the batch start/stop replaced by spies; the assertions are
on the call, its argument and its count — not on the source text. A second test runs
`start_batch_api` itself and asserts it reconciles BEFORE it starts the runner, then prunes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.unit.batch_fixtures import batch_db  # noqa: F401


async def test_lifespan_starts_the_batch_api_once_with_the_session_factory(monkeypatch, batch_db):
    import millm.db.base as db_base
    import millm.main as main

    # ⚠ The lifespan's startup resets run real UPDATEs; point them at the test's SQLite file, never
    # at whatever DATABASE_URL names on this machine.
    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    async_session_factory = batch_db
    start, stop = AsyncMock(), AsyncMock()
    monkeypatch.setattr(main, "start_batch_api", start)
    monkeypatch.setattr(main, "stop_batch_api", stop)
    monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
    monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)

    async with main.lifespan(MagicMock()):
        assert start.await_count == 1
        assert start.await_args.args == (async_session_factory,)
        assert stop.await_count == 0
    assert stop.await_count == 1


async def test_start_batch_api_reconciles_then_starts_then_prunes(monkeypatch, tmp_path):
    from millm.core.config import settings
    from millm.services.batch import reconcile as module

    monkeypatch.setattr(settings, "BATCH_FILES_DIR", str(tmp_path))
    order: list[str] = []
    runner = MagicMock()
    runner.start.side_effect = lambda: order.append("start")

    async def fake_reconcile(factory, r, store):
        assert r is runner and factory == "FACTORY"
        order.append("reconcile")
        return {}

    async def fake_prune(factory, store, now=None, **kw):
        order.append("prune")
        return {}

    monkeypatch.setattr(module, "reconcile_batches_on_startup", fake_reconcile)
    import millm.services.batch.retention as retention

    monkeypatch.setattr(retention, "prune_expired_files", fake_prune)

    async def never(*a, **k):
        import asyncio

        await asyncio.sleep(3600)

    monkeypatch.setattr(retention, "retention_loop", never)
    await module.start_batch_api("FACTORY", runner)
    assert order == ["reconcile", "start", "prune"]
    runner._retention.cancel()


def test_the_lifespan_names_are_the_real_functions():
    """A local stub named `start_batch_api` in main.py would satisfy a call-count test."""
    import millm.main as main
    from millm.services.batch import reconcile

    assert main.start_batch_api is reconcile.start_batch_api
    assert main.stop_batch_api is reconcile.stop_batch_api


async def test_a_failing_reconcile_never_stops_the_server_starting():
    from millm.services.batch.reconcile import reconcile_batches_on_startup

    def broken():
        raise RuntimeError("no database")

    found = await reconcile_batches_on_startup(broken, MagicMock(), MagicMock())
    assert all(v == [] for v in found.values())


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False), ("0", False),
                                            ("ture", True), ("", True)])
def test_batch_pack_default_fails_to_its_default(value, expected):
    from millm.core.config import Settings

    assert Settings(BATCH_PACK_DEFAULT=value).BATCH_PACK_DEFAULT is expected
