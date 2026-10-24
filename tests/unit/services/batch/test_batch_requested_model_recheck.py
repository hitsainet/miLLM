"""A batch re-resolves each line's requested model when it runs (033 D-31, R1-C finding 4).

Lines are validated against the resident when the batch is created. The resident can change
before the batch's first chunk: an adapter unloaded after validation, before the batch takes its
lease. A line that named the adapter used to come back 200 from the bare base (labelled `tiny`,
honest, and not what was asked for). Now each line is re-checked inside the chunk's slot and a
line whose name no longer means the resident fails with exactly the synchronous route's
unknown-model answer (404 `model_not_found`) and is not run; a line naming the bare base runs.

Both the packed scoring path (which re-enters the slot WITHOUT a per-row name, so the runner's
re-check is its only guard) and the unpacked path are covered.

MUTATION CONTROLS (W3-B report, each verified red, restore verified):
  M5  the runner's re-check refuses nothing -> the packed case red; the unpacked case stays green
      because each unpacked row re-enters `_admit` with its own name (the second layer)
  M6  refused rows dropped from the chunk's results -> red (they are never recorded)
  M15 the re-check call removed from `run_chunk` -> the packed case red
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from millm.api.routes.openai.errors import model_not_found_error
from millm.services.adapter_service import AdapterService, get_adapter_state
from millm.services.model_lease import get_lease_registry
from millm.services.weight_barrier import WeightBarrier, set_weight_barrier
from tests.unit.adapter_fixtures import BASE_NAME, write_adapter
from tests.unit.adapter_service_fixtures import ADAPTER_NAME, identity
from tests.unit.batch_fixtures import (  # noqa: F401
    batch_db,
    batch_dir,
    client_for,
    completion_line,
    harness,
)


@pytest.fixture
def adapter(harness, tmp_path):  # noqa: F811
    get_adapter_state().take()
    barrier = WeightBarrier(inference=lambda: harness.inference)
    set_weight_barrier(barrier)
    root = tmp_path / "checkpoints"
    root.mkdir()
    from millm.ml.model_loader import LoadedModelState

    loaded = LoadedModelState().current
    loaded.quantization_method = "none"
    write_adapter(root / "r9" / "a200", loaded.model)
    service = AdapterService(
        inference=lambda: harness.inference,
        barrier=lambda: barrier,
        leases=get_lease_registry,
        row_lookup=AsyncMock(return_value=SimpleNamespace(
            name="tiny", repo_id=BASE_NAME, source=None, local_path=None)),
        names_lookup=AsyncMock(return_value=["tiny", BASE_NAME]),
        roots=lambda: [("primary", str(root))],
    )
    yield SimpleNamespace(service=service, rel="r9/a200")
    get_adapter_state().take()
    get_adapter_state().release_claim()
    set_weight_barrier(None)


@pytest.mark.parametrize("pack", [False, True])
async def test_a_line_naming_an_adapter_unloaded_after_validation_fails_404(
    harness, adapter, pack  # noqa: F811
):
    await adapter.service.load(identity(), adapter.rel)
    async with client_for(harness.app()) as client:
        created = await harness.create(client, [
            completion_line("a0", seed=7, model=ADAPTER_NAME),
            completion_line("b0", seed=7, model="tiny"),
            completion_line("a1", seed=7, model=ADAPTER_NAME),
        ], pack=pack)
    assert created.status_code == 200, created.text
    await harness.settle_validation()
    batch = await harness.batch(created.json()["id"])
    assert batch.status == "in_progress", (batch.status, batch.errors)
    # The resident changes after validation and before the batch's first lease.
    await adapter.service.unload(identity())

    await harness.drain()
    batch = await harness.batch(created.json()["id"])
    assert batch.status == "completed", (batch.status, batch.errors)
    async with client_for(harness.app()) as client:
        out = await harness.lines(client, batch.output_file_id)
        errors = await harness.lines(client, batch.error_file_id)

    assert [line["custom_id"] for line in out] == ["b0"]
    assert out[0]["response"]["status_code"] == 200
    assert out[0]["response"]["body"]["model"] == "tiny"

    expected = json.loads(model_not_found_error(ADAPTER_NAME, "tiny").body)
    by_id = {line["custom_id"]: line for line in errors}
    assert sorted(by_id) == ["a0", "a1"]
    for line in by_id.values():
        assert line["response"]["status_code"] == 404, line
        assert line["response"]["body"] == expected
    assert (batch.request_completed, batch.request_failed) == (1, 2)


async def test_lines_naming_the_bare_base_while_adapted_still_run(harness, adapter):  # noqa: F811
    await adapter.service.load(identity(), adapter.rel)
    async with client_for(harness.app()) as client:
        created = await harness.create(client, [completion_line("b0", model="tiny")], pack=False)
    await harness.drain()
    batch = await harness.batch(created.json()["id"])
    async with client_for(harness.app()) as client:
        out = await harness.lines(client, batch.output_file_id)
    assert out[0]["response"]["status_code"] == 200
    assert out[0]["response"]["body"]["model"] == ADAPTER_NAME


async def test_the_recheck_runs_with_the_chunks_slot_held(harness, adapter, monkeypatch):  # noqa: F811
    """D-31's point is WHERE the check runs: with the slot held, so nothing can change the
    resident between the check and the run. Here the adapter's unload is queued (an interactive
    waiter) ahead of the chunk, which waits behind it as background work: a check made before the
    chunk took its slot would pass, then the unload would land, and the packed scoring path (which
    re-enters the slot with no per-row name, so the runner's check is its only guard) would answer
    lines that named the adapter from the bare base.

    MUTATION CONTROL (R2-A): the re-check moved above `inference._admit(background=True)` in
    `run_chunk` -> red (the adapter lines come back 200 from the base). The earlier test above
    unloads before the chunk starts, so it cannot tell the two placements apart.
    """
    import asyncio

    inference = harness.inference
    await adapter.service.load(identity(), adapter.rel)
    async with client_for(harness.app()) as client:
        created = await harness.create(client, [
            completion_line("a0", seed=7, model=ADAPTER_NAME),
            completion_line("a1", seed=7, model=ADAPTER_NAME),
        ], pack=True)
    assert created.status_code == 200, created.text
    await harness.settle_validation()

    # Something in flight holds the one slot, so the unload queues behind it.
    release, holding = asyncio.Event(), asyncio.Event()

    async def in_flight() -> None:
        async with inference._admit():
            holding.set()
            await release.wait()

    holder_task = asyncio.create_task(in_flight())
    await holding.wait()
    # Since review R2-C, the unload re-checks the lease inside its pause, and a running batch
    # holds one, so in production this unload is refused (D-15) and the race cannot happen
    # that way. The chunk re-check is the second line of defence (e.g. an unload by a holder
    # of the batch's lease); bypass the lease here so the test still pins its placement.
    monkeypatch.setattr(adapter.service, "_refuse_if_leased", lambda *a, **k: None)
    unload = asyncio.create_task(adapter.service.unload(identity()))
    barrier = adapter.service._barrier()
    for _ in range(1000):
        if barrier._gate is not None:
            break
        await asyncio.sleep(0)
    assert barrier._gate is not None, "the unload never started taking the slot"

    original = inference._admit
    chunk_waiting = asyncio.Event()

    def spy(*args, **kwargs):
        if kwargs.get("background"):
            chunk_waiting.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, "_admit", spy)
    step = asyncio.create_task(harness.runner.step())
    await asyncio.wait_for(chunk_waiting.wait(), timeout=30)
    assert get_adapter_state().current is not None, "the adapter left before the chunk queued"
    release.set()
    await asyncio.wait_for(asyncio.gather(holder_task, unload, step), timeout=60)
    assert get_adapter_state().current is None
    monkeypatch.setattr(inference, "_admit", original)

    await harness.drain()
    batch = await harness.batch(created.json()["id"])
    async with client_for(harness.app()) as client:
        out = await harness.lines(client, batch.output_file_id)
        errors = await harness.lines(client, batch.error_file_id)
    assert out == [], out
    assert sorted(line["custom_id"] for line in errors) == ["a0", "a1"]
    assert all(line["response"]["status_code"] == 404 for line in errors)
