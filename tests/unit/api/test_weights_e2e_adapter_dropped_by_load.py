"""Flow (c), 033 and 032 together through the REAL app (review RX).

An adapter is applied (it names the resident by the manifest hash the save gave it, D-09), `/v1`
reports it (`origin=adapter`, D-33), then a 032 load REPLACES THE BASE and the adapter must go:
the in-place copy fires the base-replaced callbacks before its first write, the D-01 row reload
goes through the `LoadedModelState` chokepoints. Afterwards no hook is left (logits equal the
bare base exactly), the served name is the base's, and all three status surfaces agree.

The AdapterService is the production class with its collaborators pointed at this flow's live
objects (inference, barrier, lease registry, the real repository for row and name lookups, the
flow's checkpoint root). Harness and fakes: `tests/unit/weights_e2e_fixtures.py`.
"""

from __future__ import annotations

import pytest
import torch

from millm.services.adapter_service import AdapterService, addressable_names, get_adapter_service
from millm.services.model_lease import get_lease_registry
from tests.unit.adapter_fixtures import write_adapter
from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    OPS,
    ROW_ID,
    ROW_NAME,
    ROW_REPO,
    TRAINER,
    attach,
    chat,
    flow,
    holder,
    logits,
    manifest,
)

ADAPTER = "org/tiny:r1:adapter-7"


@pytest.fixture
def adapters(flow):
    repo = flow.models.repository

    async def names():
        return sorted(addressable_names(await repo.get_all()))

    service = AdapterService(
        inference=lambda: flow.inference, barrier=lambda: flow.barrier,
        leases=get_lease_registry, row_lookup=repo.get_by_id, names_lookup=names,
        roots=lambda: [("primary", str(flow.root))],
    )
    flow.app.dependency_overrides[get_adapter_service] = lambda: service
    return service


async def _apply(flow, client, *, base_name, base_hash=None) -> dict:
    write_adapter(flow.root / "adapters" / "a7", flow.model, base_name=base_name,
                  base_hash=base_hash,
                  manifest_overrides={"name": ADAPTER})
    r = await client.post("/api/weights/adapter/load", json={"path": "adapters/a7"},
                          headers=OPS)
    assert r.status_code == 200, r.text
    return r.json()["data"]


async def _three_surfaces(client) -> tuple[dict, dict, dict]:
    status = (await client.get("/api/weights/status")).json()["data"]
    adapter = (await client.get("/api/weights/adapter")).json()["data"]
    health = (await client.get("/api/health/detailed")).json()["weights"]
    return status, adapter, health


async def test_in_place_load_drops_the_adapter(flow, adapters):
    async with flow.client() as client:
        attachment_id = await attach(client)
        saved = await client.post("/api/weights/save", json={"manifest": manifest(step=1)},
                                  headers=holder(attachment_id))
        assert saved.status_code == 201, saved.text
        name1, hash1 = saved.json()["data"]["name"], saved.json()["data"]["manifest_sha256"]
        assert (await client.post("/api/weights/detach",
                                  headers=holder(attachment_id))).status_code == 200
        bare = logits(flow.model)

        applied = await _apply(flow, client, base_name=name1, base_hash=hash1)
        assert applied["resolved_model_name"] == ADAPTER
        assert not torch.equal(logits(flow.model), bare)
        r = await chat(client, ADAPTER)
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Model"].startswith(f'"{ADAPTER}";origin=adapter')
        status, adapter, _ = await _three_surfaces(client)
        assert adapter["applied"] is True and adapter["name"] == ADAPTER
        assert status["adapter"] == adapter
        assert adapter["base_name"] == name1 and adapter["manifest_base_sha256"] == hash1

        # The 032 in-place load replaces the base: the adapter is dropped before the copy.
        load = await client.post("/api/weights/load", json={"name": name1}, headers=TRAINER)
        assert load.status_code == 200, load.text
        assert load.json()["data"]["mode"] == "in_place"
        assert torch.equal(logits(flow.model), bare), "a LoRA hook survived the base change"
        status, adapter, health = await _three_surfaces(client)
        assert adapter["applied"] is False and adapter["name"] is None
        assert status["adapter"] == adapter
        assert status["resident_model"]["name"] == name1
        assert health["weights_state"] == status["weights_state"] == "clean"
        gone = await chat(client, ADAPTER)
        assert gone.status_code == 404
        r = await chat(client, name1)
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Model"] == f'"{name1}";origin=checkpoint;commits=0'


async def test_row_reload_drops_the_adapter(flow, adapters):
    async with flow.client() as client:
        bare = logits(flow.model)
        await _apply(flow, client, base_name=ROW_REPO)
        r = await chat(client, ROW_NAME)
        assert r.status_code == 200, r.text
        assert r.json()["model"] == ADAPTER

        load = await client.post("/api/weights/load", json={"model_id": ROW_ID}, headers=TRAINER)
        assert load.status_code == 200, load.text
        assert load.json()["data"]["mode"] == "full_reload"
        status, adapter, _ = await _three_surfaces(client)
        assert adapter["applied"] is False and status["adapter"] == adapter
        # The reloaded model is a NEW object read from the row's directory, with no hooks.
        assert flow.resident.model is not flow.model
        assert torch.equal(logits(flow.resident.model), bare)
        r = await chat(client, ROW_NAME)
        assert r.status_code == 200, r.text
        assert r.json()["model"] == ROW_NAME
        assert r.headers["X-miLLM-Model"] == f'"{ROW_NAME}";origin=model;commits=0'


async def test_save_while_adapted_writes_the_base_only(flow, adapters):
    """R-01.27: the adapter is applied at serve time and is not part of the shared parameter set,
    so neither the attach export nor a save carries it, and the saved values are the base's."""
    from safetensors.torch import load_file

    base = {n: p.detach().clone() for n, p in flow.model.named_parameters()}
    async with flow.client() as client:
        await _apply(flow, client, base_name=ROW_REPO)
        attach_body = (await client.post("/api/weights/attach", headers=TRAINER)).json()["data"]
        names = {p["name"] for p in attach_body["parameters"]}
        assert names == {n for n, _ in flow.model.named_parameters(remove_duplicate=False)}
        assert not any("lora" in n for n in names)
        saved = await client.post("/api/weights/save", json={"manifest": manifest(step=4)},
                                  headers=holder(attach_body["attachment_id"]))
        assert saved.status_code == 201, saved.text
        tensors = load_file(str(flow.root / "run" / "step-000004" / "model.safetensors"))
        assert not any("lora" in k for k in tensors)
        for name, value in tensors.items():
            assert torch.equal(value, base[name]), name
        # The adapter is still applied and still what /v1 reports.
        r = await chat(client, ADAPTER)
        assert r.status_code == 200 and r.json()["model"] == ADAPTER
