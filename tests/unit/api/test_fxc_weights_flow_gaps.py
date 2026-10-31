"""Features 031/032 requirement gaps left open by review RX (fix round FX-C), through the REAL app.

Harness and its fakes: `tests/unit/weights_e2e_fixtures.py` (real barrier, weight service,
checkpoint service, inference service over a tiny bfloat16 Llama, real model service).

* FR-32.2.4 — for the whole save snapshot a `commit-begin` and a weights load are refused
  (`409 PAUSE_IN_PROGRESS`, naming the save); the holder's quiesce is handed back afterwards.
* FR-32.6.3 — completions and embeddings by the resident's checkpoint served name are served by
  the resident: no load, the served name reported.
* FR-31.5.4 — a parameter whose address moved while attached tears the set at quiesce and logs
  `weights_address_moved` at error level, naming the parameter and the context.
* FR-31.7.5 — quiesce enters `_admit` as one interactive waiter: a full queue refuses it at the
  route with `503 QUEUE_FULL` and `Retry-After`, holding nothing.
* FR-31.17.1 / .3 — while attached and serving: an SAE attaches (setting `locked`), steers a
  served `/v1` forward and detaches; a probe armed on the resident scores exactly as it did
  unattached (not provisional, the base served name); the holder's bracket still works around
  both, and no address moves.

Mutation controls (each run against this file, restored, `git diff` verified clean):
  F1  `commit_begin`'s internal-owner refusal removed            -> FR-32.2.4 test red
  F2  `_begin_internal_locked`'s internal-owner refusal removed  -> FR-32.2.4 test red
  F3  resolver rule 1 (`name in resident_names(info)`) disabled   -> FR-32.6.3 tests red
  F4  `weights_address_moved` error line removed                 -> FR-31.5.4 test red
  F5  `_take_slot` no longer re-raises `_admit`'s refusal         -> FR-31.7.5 test red
  F6  (injected interaction) `attach_sae` refuses while attached  -> FR-31.17 SAE test red
  F7  (injected interaction) attachment makes verdicts provisional -> FR-31.17 probe test red
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from structlog.testing import capture_logs

import millm.services.weights_checkpoint_service as checkpoint_service
from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    ROW_NAME,
    ROW_REPO,
    addresses,
    attach,
    chat,
    flow,
    holder,
    manifest,
    params,
)


async def _until(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


async def _bracket(client, attachment_id) -> dict:
    """quiesce → commit-begin → commit-end → resume, each 200; returns commit-end's data."""
    h = holder(attachment_id)
    for path in ("quiesce", "commit-begin"):
        r = await client.post(f"/api/weights/{path}", headers=h)
        assert r.status_code == 200, (path, r.text)
    end = await client.post("/api/weights/commit-end", headers=h)
    assert end.status_code == 200, end.text
    r = await client.post("/api/weights/resume", headers=h)
    assert r.status_code == 200, r.text
    return end.json()["data"]


# ----------------------------------------------------------------------------- FR-32.2.4


async def test_commit_begin_and_load_are_refused_for_the_whole_snapshot(flow, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = checkpoint_service.snapshot_to_host

    def blocking_snapshot(model, into=None):
        entered.set()
        release.wait(timeout=30)
        return real(model, into)

    async with flow.client() as client:
        attachment_id = await attach(client)
        h = holder(attachment_id)
        first = await client.post("/api/weights/save", json={"manifest": manifest(step=1)},
                                  headers=h)
        assert first.status_code == 201, first.text
        name1 = first.json()["data"]["name"]
        assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
        before = params(flow.model)

        monkeypatch.setattr(checkpoint_service, "snapshot_to_host", blocking_snapshot)
        save = asyncio.create_task(client.post(
            "/api/weights/save", json={"manifest": manifest(step=2, parent=name1)}, headers=h))
        try:
            await _until(entered.is_set)
            begin = await client.post("/api/weights/commit-begin", headers=h)
            load = await client.post("/api/weights/load", json={"name": name1}, headers=h)
        finally:
            release.set()
        saved = await save

        assert begin.status_code == 409, begin.text
        assert begin.json()["error"]["code"] == "PAUSE_IN_PROGRESS"
        assert begin.json()["error"]["details"]["operation"] == "save"
        assert load.status_code == 409, load.text
        assert load.json()["error"]["code"] == "PAUSE_IN_PROGRESS"
        assert load.json()["error"]["details"]["operation"] == "save"

        assert saved.status_code == 201, saved.text
        assert all(torch.equal(p, before[n]) for n, p in flow.model.named_parameters())
        snap = flow.barrier.snapshot()
        # The holder's quiesce was adopted for the snapshot and handed back afterwards.
        assert (snap.weights_state, snap.barrier, snap.hold_owner) == ("clean", "quiesced",
                                                                       "holder")
        assert (await client.post("/api/weights/resume", headers=h)).status_code == 200


# ----------------------------------------------------------------------------- FR-32.6.3


async def _saved_name(client) -> str:
    attachment_id = await attach(client)
    r = await client.post("/api/weights/save", json={"manifest": manifest(step=1)},
                          headers=holder(attachment_id))
    assert r.status_code == 201, r.text
    assert (await client.post("/api/weights/detach",
                              headers=holder(attachment_id))).status_code == 200
    return r.json()["data"]["name"]


async def test_completions_by_the_checkpoint_served_name_are_served_by_the_resident(flow):
    async with flow.client() as client:
        name1 = await _saved_name(client)
        assert name1 == f"{ROW_REPO}:run:step-1"
        r = await client.post("/v1/completions", json={
            "model": name1, "prompt": "w1 w2 w3", "max_tokens": 3, "temperature": 0.0})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == name1
    assert r.headers["X-miLLM-Model"] == f'"{name1}";origin=checkpoint;commits=0'
    assert flow.loader.loads == [], "the resident was reloaded for its own served name"


async def test_embeddings_by_the_checkpoint_served_name_are_served_by_the_resident(flow):
    async with flow.client() as client:
        name1 = await _saved_name(client)
        r = await client.post("/v1/embeddings", json={"model": name1, "input": "w1 w2 w3"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == name1
    assert len(body["data"]) == 1 and len(body["data"][0]["embedding"]) == 16
    assert flow.loader.loads == [], "the resident was reloaded for its own served name"


# ----------------------------------------------------------------------------- FR-31.5.4


async def test_a_moved_address_tears_at_quiesce_and_logs_weights_address_moved(flow):
    async with flow.client() as client:
        attachment_id = await attach(client)
        moved = "model.layers.0.mlp.up_proj.weight"
        parameter = flow.model.get_parameter(moved)
        parameter.data = parameter.data.clone()  # a new storage: the exported handle is stale
        with capture_logs() as logs:
            r = await client.post("/api/weights/quiesce", headers=holder(attachment_id))
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "WEIGHTS_TORN"
    assert r.json()["error"]["details"]["reason"] == "address_moved"
    lines = [e for e in logs if e["event"] == "weights_address_moved"]
    assert len(lines) == 1
    line = lines[0]
    assert line["log_level"] == "error"
    assert line["parameters"] == [moved] and line["moved_count"] == 1
    assert line["context"] == "quiesce" and line["attachment_ref"]
    torn = [e for e in logs if e["event"] == "weights_torn"]
    assert [e["reason"] for e in torn] == ["address_moved"]
    assert flow.barrier.snapshot().weights_state == "torn"


# ----------------------------------------------------------------------------- FR-31.7.5


async def test_quiesce_on_a_full_queue_is_503_queue_full_and_holds_nothing(flow):
    queue = flow.inference.request_queue
    async with flow.client() as client:
        attachment_id = await attach(client)
        saved_max = queue._max_pending
        queue._max_pending = 0
        try:
            with capture_logs() as logs:
                r = await client.post("/api/weights/quiesce", headers=holder(attachment_id))
        finally:
            queue._max_pending = saved_max
        assert r.status_code == 503, r.text
        assert r.json()["error"]["code"] == "QUEUE_FULL"
        assert int(r.headers["Retry-After"]) >= 1
        audit = [e for e in logs if e["event"] == "weights_audit"]
        assert [(e["outcome"], e["code"]) for e in audit] == [("refused", "QUEUE_FULL")]
        snap = flow.barrier.snapshot()
        assert (snap.barrier, snap.hold_owner, snap.weights_state) == ("serving", None, "clean")
        assert not flow.barrier.holds_slot()
        # Nothing was left held: serving answers, and the holder can quiesce once there is room.
        assert (await chat(client)).status_code == 200
        assert (await client.post("/api/weights/quiesce",
                                  headers=holder(attachment_id))).status_code == 200
        assert (await client.post("/api/weights/resume",
                                  headers=holder(attachment_id))).status_code == 200


# ----------------------------------------------------------------------------- FR-31.17.1 / .3


def _sae_service(inference, row):
    from millm.ml.sae_config import SAEConfig
    from millm.ml.sae_hooker import SAEHooker
    from millm.ml.sae_wrapper import LoadedSAE
    from millm.services.sae_service import AttachedSAEState, CompatibilityResult, SAEService

    torch.manual_seed(11)
    d_in, d_sae = 16, 8
    config = SAEConfig(d_in=d_in, d_sae=d_sae, model_name=ROW_NAME,
                       hook_name="blocks.1.hook_resid_post", hook_layer=1, dtype="bfloat16")
    sae = LoadedSAE(W_enc=torch.randn(d_in, d_sae), b_enc=torch.zeros(d_sae),
                    W_dec=torch.randn(d_sae, d_in), b_dec=torch.zeros(d_in), config=config)
    svc = SAEService.__new__(SAEService)
    svc._sae_state = AttachedSAEState()
    svc._hooker = SAEHooker()
    svc._loader = MagicMock()
    svc._loader.load.return_value = sae
    svc._inference_service = inference
    svc.get_sae = AsyncMock(return_value=row)
    svc.check_compatibility = AsyncMock(
        return_value=CompatibilityResult(compatible=True, errors=[], warnings=[]))
    svc._reset_dynamo_for_hook_change = MagicMock()
    svc.repository = MagicMock()
    svc.repository.update_status = AsyncMock()
    svc.repository.create_attachment = AsyncMock()
    svc.repository.deactivate_attachment = AsyncMock()
    return svc


async def test_an_sae_attaches_steers_and_detaches_while_attached(flow, test_session):
    from millm.db.models.model import Model
    from millm.services.sae_service import AttachedSAEState

    AttachedSAEState().reset_for_tests()
    row = MagicMock(id="sae-1", cache_path="/nonexistent/sae-1", file_size_bytes=0)

    @contextlib.asynccontextmanager
    async def session_factory():
        yield test_session

    svc = _sae_service(flow.inference, row)
    fixed = addresses(flow.model)
    try:
        async with flow.client() as client:
            attachment_id = await attach(client)
            with patch("millm.db.base.async_session_factory", session_factory):
                result = await svc.attach_sae("sae-1", 1)
            assert result["status"] == "attached"
            model_row = await test_session.get(Model, 1)
            await test_session.refresh(model_row)
            assert model_row.locked is True, "FR-31.17.3: SAE attach sets locked"

            sae = svc._sae_state.attached_sae
            sae.set_steering(0, 4.0)
            sae.enable_steering(True)
            served = await chat(client)
            assert served.status_code == 200, served.text
            assert sae.steering_apply_count > 0, "the SAE did not steer the served forward"

            end = await _bracket(client, attachment_id)
            assert end["steps_since_load"] == 1 and end["weights_state"] == "clean"
            assert addresses(flow.model) == fixed and flow.weights.verify_addresses("t") == []

            with patch("millm.db.base.async_session_factory", session_factory):
                detached = await svc.detach_sae("sae-1")
            assert detached["status"] == "detached"
            await test_session.refresh(model_row)
            assert model_row.locked is False
            status = (await client.get("/api/weights/status")).json()["data"]
            assert status["attached"] is True and status["weights_state"] == "clean"
            assert (await chat(client)).status_code == 200
    finally:
        AttachedSAEState().reset_for_tests()


async def test_a_probe_scores_the_same_while_attached(flow):
    from millm.ml.probe_head import ProbeHead
    from millm.services.probe_runtime import ArmedProbe, ProbeRuntimeState

    ProbeRuntimeState.reset_for_tests()
    torch.manual_seed(3)
    probe = ArmedProbe(
        probe_id="pr_fxc", name="concept",
        head=ProbeHead(weight=torch.randn(16), bias=0.0, layer=1),
        rule="mean", scope="all", layer=1, rung=2, rung_language="detects on unseen tasks",
        threshold=0.0,
    )
    ids = torch.tensor([[0, 6, 7, 8, 9]])

    def score(request_id: str):
        def run():
            state = ProbeRuntimeState()
            state.begin_request(request_id)
            try:
                with torch.no_grad():
                    flow.model(ids)
            finally:
                context = state.end_request()
            return context.finish()[0]
        return run

    try:
        ProbeRuntimeState().arm(probe, flow.model)
        unattached = await flow.inference.run_model_work(score("unattached"))
        assert unattached.scored
        async with flow.client() as client:
            attachment_id = await attach(client)
            attached = await flow.inference.run_model_work(score("attached"))
            assert attached.scored
            assert attached.score == unattached.score
            assert attached.provisional is False and unattached.provisional is False
            assert attached.served_model == ROW_NAME == unattached.served_model
            assert (await chat(client)).status_code == 200
            end = await _bracket(client, attachment_id)
            assert end["weights_state"] == "clean"
            assert flow.weights.verify_addresses("t") == []
    finally:
        ProbeRuntimeState.reset_for_tests()
