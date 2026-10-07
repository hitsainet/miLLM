"""Feature 26 task 9.2: the whole journey with the runner's REAL loop running.

Unit tests drive `runner.step()` by hand so their ordering is deterministic. Here `runner.start()`
runs the production loop as an asyncio task — wake-ups, polling and all — against the real app, a
real ModelService and a tiny real Llama: upload → create → validate → run → finalize → download,
then a cancel mid-run, then a restart mid-run (a new runner and the startup reconciliation).
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from millm.core.config import settings
from millm.services.model_lease import get_lease_registry
from tests.unit.batch_fixtures import (  # noqa: F401
    BatchHarness,
    batch_db,
    batch_dir,
    client_for,
    completion_line,
    harness,
)

pytestmark = pytest.mark.integration

N = 12


def _lines():
    return [completion_line(f"r{i}", prompt=" ".join(f"w{j}" for j in range(1, i % 5 + 2)))
            for i in range(N)]


async def _until(harness, batch_id, statuses, timeout=60.0):
    async def poll():
        while (await harness.batch(batch_id)).status not in statuses:
            await asyncio.sleep(0.05)
    await asyncio.wait_for(poll(), timeout)
    return await harness.batch(batch_id)


async def test_the_journey_end_to_end_under_the_real_loop(harness, monkeypatch):
    monkeypatch.setattr(settings, "BATCH_WAIT_POLL_S", 0.05)
    harness.runner.start()
    async with client_for(harness.app()) as client:
        created = await harness.create(client, _lines(), pack=False)
        batch_id = created.json()["id"]
        batch = await _until(harness, batch_id, {"completed"})
        got = (await client.get(f"/v1/batches/{batch_id}")).json()
        out = await harness.lines(client, batch.output_file_id)
    assert got["request_counts"] == {"total": N, "completed": N, "failed": 0}
    assert [line["custom_id"] for line in out] == [f"r{i}" for i in range(N)]
    assert get_lease_registry().current(1) is None, "the own lease outlived the batch"


async def test_cancel_mid_run_under_the_real_loop(harness, monkeypatch):
    from millm.services.batch.executors import RowExecutor

    monkeypatch.setattr(settings, "BATCH_WAIT_POLL_S", 0.05)
    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 1)
    real = RowExecutor.run_one
    gate = asyncio.Event()

    async def slow(self, row):
        if row.line_no == 4:
            gate.set()
        await asyncio.sleep(0.02)
        return await real(self, row)

    monkeypatch.setattr(RowExecutor, "run_one", slow)
    harness.runner.start()
    async with client_for(harness.app()) as client:
        batch_id = (await harness.create(client, _lines(), pack=False)).json()["id"]
        await asyncio.wait_for(gate.wait(), 30)
        await client.post(f"/v1/batches/{batch_id}/cancel")
        batch = await _until(harness, batch_id, {"cancelled"})
        out = await harness.lines(client, batch.output_file_id)
        err = await harness.lines(client, batch.error_file_id)
    ids = Counter(line["custom_id"] for line in out + err)
    assert ids == Counter({f"r{i}": 1 for i in range(N)})
    assert 4 <= len(out) < N
    assert {e["error"]["code"] for e in err} == {"batch_cancelled"}


async def test_restart_mid_run_resumes_with_no_row_repeated(harness, monkeypatch):
    from millm.services.batch.files import get_file_store
    from millm.services.batch.reconcile import reconcile_batches_on_startup

    monkeypatch.setattr(settings, "BATCH_WAIT_POLL_S", 0.05)
    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 2)
    async with client_for(harness.app()) as client:
        batch_id = (await harness.create(client, _lines(), pack=False)).json()["id"]
    await harness.settle_validation()
    for _ in range(2):
        assert await harness.runner.step()
    await harness.runner.stop()  # the pod goes away
    get_lease_registry().clear()  # ...taking every lease with it (X-01)
    fresh = BatchHarness(harness.factory)
    await reconcile_batches_on_startup(harness.factory, fresh.runner, get_file_store())
    fresh.runner.start()
    try:
        batch = await _until(harness, batch_id, {"completed"})
        async with client_for(fresh.app()) as client:
            out = await harness.lines(client, batch.output_file_id)
    finally:
        await fresh.runner.stop()
    assert Counter(line["custom_id"] for line in out) == Counter({f"r{i}": 1 for i in range(N)})
