"""Flow (d): restart semantics, through the REAL app (review RX; PADR v1.6 §10, 031 FTID §7.7, D-01).

* A restart begins `clean`, `serving`, with no attachment and no counters, whatever the previous
  process left (here: a torn commit, a held slot, an attachment, a save, an overrun). The reset
  is `reset_weights_on_startup()`; that `lifespan` calls it, after the lease clear, is pinned by
  `tests/unit/test_startup_reset.py`. A real restart also starts with nothing resident.
* Clean by construction (D-01): a Feature-1 UNLOAD is allowed while `torn`, and with nothing
  resident the state is `clean` and the torn set's held slot is released.

Harness and fakes: `tests/unit/weights_e2e_fixtures.py`.
"""

from __future__ import annotations

import asyncio

import torch

from millm.core.config import settings
from millm.ml.model_loader import LoadedModelState
from millm.services import weight_attachment
from millm.services.weight_attachment import WeightService, set_weight_service
from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    ROW_ID,
    attach,
    cpu_export,
    flow,
    holder,
    manifest,
)


async def _until(predicate, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


async def _leave_a_mess(flow, client, monkeypatch) -> str:
    """Save, then a commit the watchdog tears, with the attachment still present."""
    monkeypatch.setattr(settings, "WEIGHTS_COMMIT_WATCHDOG_MS", 30)
    attachment_id = await attach(client)
    h = holder(attachment_id)
    assert (await client.post("/api/weights/save", json={"manifest": manifest(step=1)},
                              headers=h)).status_code == 201
    assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
    assert (await client.post("/api/weights/commit-begin", headers=h)).status_code == 200
    with torch.no_grad():
        next(flow.model.parameters()).add_(1.0)
    await _until(lambda: flow.barrier.snapshot().weights_state == "torn")
    status = (await client.get("/api/weights/status")).json()["data"]
    assert status["weights_state"] == "torn" and status["attached"] is True
    assert status["last_save"] is not None and status["commit_overran"] is True
    assert flow.barrier.holds_slot()
    return attachment_id


async def test_restart_begins_clean_with_no_attachment(flow, monkeypatch):
    async with flow.client() as client:
        old_id = await _leave_a_mess(flow, client, monkeypatch)

        # The restart: process memory is gone (nothing resident), and lifespan resets.
        LoadedModelState()._loaded = None
        weight_attachment.reset_weights_on_startup()
        # The fresh process builds its service on first use; this flow wires its fakes in.
        set_weight_service(WeightService(barrier=lambda: flow.barrier,
                                         inference=lambda: flow.inference, export=cpu_export))

        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["weights_state"] == "clean" and status["barrier"] == "serving"
        assert status["attached"] is False and status["attachment"] is None
        assert status["resident_model"] is None
        assert status["steps_since_load"] is None, "nothing resident: not recorded, never 0"
        assert status["last_save"] is None and status["commit"] is None
        assert status["commit_overran"] is False and status["last_overrun"] is None
        assert status["last_hold_timeout_at"] is None and status["held_requests"] is None
        assert not flow.barrier.holds_slot()
        health = (await client.get("/api/health/detailed")).json()["weights"]
        assert health == {"weights_state": "clean", "barrier": "serving", "attached": False,
                          "attachment_state": None, "error": None}
        # The previous process's attachment ID means nothing now.
        r = await client.post("/api/weights/heartbeat", headers=holder(old_id))
        assert r.status_code == 409 and r.json()["error"]["code"] == "NOT_ATTACHED"


async def test_unload_while_torn_is_clean_by_construction(flow, monkeypatch):
    async with flow.client() as client:
        attachment_id = await _leave_a_mess(flow, client, monkeypatch)
        assert (await client.post("/api/weights/detach",
                                  headers=holder(attachment_id))).status_code == 200
        assert flow.barrier.snapshot().weights_state == "torn"
        r = await client.post(f"/api/models/{ROW_ID}/unload")
        assert r.status_code == 200, r.text
        await _until(lambda: LoadedModelState().current is None)
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["weights_state"] == "clean" and status["barrier"] == "serving"
        assert status["resident_model"] is None and status["steps_since_load"] is None
        assert not flow.barrier.holds_slot(), "the torn set's held slot leaked past the unload"
