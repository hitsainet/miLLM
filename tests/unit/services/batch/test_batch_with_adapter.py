"""The Batch API while an adapter is applied (033 review R1-C; FR-33.4.2, addendum D-15).

A real BatchRunner, the real app, a real ModelService and lease registry and a tiny real Llama
(`BatchHarness`), plus a real `AdapterService` and the real barrier over the same model:

* every output line of a batch run while an adapter is applied reports the adapter's name in its
  body `model` and in `response.millm.headers["X-miLLM-Model"]`, whether the line named the
  adapter or the bare base (D12) — a batch line is labelled exactly as the synchronous route;
* a line's numbers differ from the base's while applied (the adapter really served it);
* a batch that has started holds the model lease, so an adapter cannot be loaded or unloaded
  under it (`MODEL_LEASED`, D-15) — the rows of one batch never straddle an adapter change.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from millm.core.errors import ModelLeasedError
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


async def _start(harness, lines):  # noqa: F811
    async with client_for(harness.app()) as client:
        created = await harness.create(client, lines, pack=False)
    assert created.status_code == 200, created.text
    await harness.settle_validation()
    return created.json()["id"]


async def _out(harness, batch_id):  # noqa: F811
    async with client_for(harness.app()) as client:
        batch = await harness.batch(batch_id)
        assert batch.status == "completed", (batch.status, batch.errors)
        return await harness.lines(client, batch.output_file_id)


async def test_batch_lines_report_the_adapter_whichever_name_they_used(harness, adapter):  # noqa: F811
    base_id = await _start(harness, [completion_line("b0", seed=7)])
    await harness.drain()
    base_line = (await _out(harness, base_id))[0]["response"]
    assert base_line["body"]["model"] == "tiny"

    await adapter.service.load(identity(), adapter.rel)
    batch_id = await _start(harness, [
        completion_line("a0", seed=7, model=ADAPTER_NAME),
        completion_line("a1", seed=7, model="tiny"),
    ])
    await harness.drain()
    out = await _out(harness, batch_id)
    assert [line["custom_id"] for line in out] == ["a0", "a1"]
    for line in out:
        response = line["response"]
        assert response["status_code"] == 200
        assert response["body"]["model"] == ADAPTER_NAME
        header = response["millm"]["headers"]["X-miLLM-Model"]
        assert header.startswith(f'"{ADAPTER_NAME}"'), header
        assert response["body"]["choices"] != base_line["body"]["choices"]


async def test_a_running_batch_holds_the_lease_against_adapter_changes(harness, adapter):  # noqa: F811
    batch_id = await _start(harness, [completion_line(f"c{i}") for i in range(4)])
    assert await harness.runner.step()  # acquires the batch's own lease and runs a chunk
    assert get_lease_registry().current(1) is not None
    with pytest.raises(ModelLeasedError):
        await adapter.service.load(identity(), adapter.rel)
    assert get_adapter_state().current is None
    await harness.drain()
    out = await _out(harness, batch_id)
    assert {line["response"]["body"]["model"] for line in out} == {"tiny"}
    # Once the batch released its own lease, the adapter loads.
    assert get_lease_registry().current(1) is None
    await adapter.service.load(identity(), adapter.rel)
    assert get_adapter_state().current is not None
