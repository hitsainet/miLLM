"""Requirement-coverage gaps found by review RX's walk over Features 031 and 032 (BRD-01, PPRD
FR-31.x / FR-32.x, addendum decisions), each through the REAL app.

A test here either closes a CPU-testable gap that no other test pinned, or records an open
finding as a STRICT xfail: it turns red the moment the finding is fixed, so the marker must be
removed with the fix. Harness and fakes: `tests/unit/weights_e2e_fixtures.py`.
"""

from __future__ import annotations

import pytest

from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    attach,
    flow,
    holder,
)


async def test_heartbeat_returns_the_window_and_a_suggested_interval(flow, monkeypatch):
    """R-01.30 (R1 RX F4): the heartbeat returns the staleness window and a suggested interval,
    read from WEIGHTS_HEARTBEAT_INTERVAL_HINT_S (a non-default value, so a constant cannot pass)."""
    monkeypatch.setattr(settings, "WEIGHTS_HEARTBEAT_INTERVAL_HINT_S", 7.5)
    async with flow.client() as client:
        attachment_id = await attach(client)
        r = await client.post("/api/weights/heartbeat", headers=holder(attachment_id))
        assert r.status_code == 200, r.text
        body = r.json()["data"]
        assert body["timeout_s"] == 30.0
        assert body["interval_hint_s"] == 7.5


# --- 032 gaps closed (RX coverage walk) -------------------------------------------------------

import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

from structlog.testing import capture_logs  # noqa: E402

import millm.services.weights_checkpoint_service as checkpoint_service  # noqa: E402
from millm.core.config import settings  # noqa: E402
from tests.unit.weights_e2e_fixtures import TRAINER, manifest  # noqa: E402


async def _saved(flow, client, step: int = 1) -> dict:
    attachment_id = await attach(client)
    r = await client.post("/api/weights/save", json={"manifest": manifest(step=step)},
                          headers=holder(attachment_id))
    assert r.status_code == 201, r.text
    assert (await client.post("/api/weights/detach",
                              headers=holder(attachment_id))).status_code == 200
    return r.json()["data"]


async def test_save_writes_generation_config_beside_config(flow):
    """FR-32.1.5: `generation_config.json` is written when the model has one."""
    flow.model.generation_config.max_new_tokens = 7
    async with flow.client() as client:
        saved = await _saved(flow, client)
    step_dir = flow.root / saved["path"]
    written = json.loads((step_dir / "generation_config.json").read_text())
    assert written["max_new_tokens"] == 7
    assert (step_dir / "config.json").is_file() and (step_dir / "tokenizer.json").is_file()


async def test_a_manifest_that_does_not_name_its_directory_is_refused(flow):
    """FR-32.5.2: a step directory copied under another step number names the wrong directory."""
    async with flow.client() as client:
        saved = await _saved(flow, client, step=5)
        shutil.copytree(flow.root / saved["path"], flow.root / "run" / "step-000006")
        before = flow.barrier.snapshot().episode
        r = await client.post("/api/weights/load", json={"path": "run/step-000006"},
                              headers=TRAINER)
    assert r.status_code == 400, r.text
    error = r.json()["error"]
    assert error["code"] == "INVALID_MANIFEST"
    assert [e["json_path"] for e in error["details"]["errors"]] == ["$.step"]
    assert flow.barrier.snapshot().episode == before, "refused after taking the barrier"


async def test_a_hash_mismatch_is_refused_before_the_barrier(flow):
    """FR-32.5.3: the hash is checked with no barrier held (no hold episode is started)."""
    async with flow.client() as client:
        saved = await _saved(flow, client)
        weights = flow.root / saved["path"] / "model.safetensors"
        data = bytearray(weights.read_bytes())
        data[-1] ^= 0x01
        weights.write_bytes(bytes(data))
        before = flow.barrier.snapshot().episode
        r = await client.post("/api/weights/load", json={"name": saved["name"]}, headers=TRAINER)
    assert r.status_code == 422 and r.json()["error"]["code"] == "CHECKPOINT_HASH_MISMATCH"
    assert flow.barrier.snapshot().episode == before


async def test_the_load_audit_line_carries_the_hash_and_the_mode(flow):
    """FR-32.5.12 / R-01.18: one accepted `load` audit line with the checkpoint hash."""
    async with flow.client() as client:
        saved = await _saved(flow, client)
        with capture_logs() as logs:
            r = await client.post("/api/weights/load", json={"name": saved["name"]},
                                  headers=TRAINER)
    assert r.status_code == 200, r.text
    lines = [e for e in logs if e["event"] == "weights_audit" and e["operation"] == "weights_load"]
    assert len(lines) == 1
    assert lines[0]["outcome"] == "accepted" and lines[0]["mode"] == "in_place"
    assert lines[0]["checkpoint_sha256"] == saved["weights_sha256"]
    assert lines[0]["label"] == "forge" and lines[0]["weights_state"] == "clean"


async def test_a_slow_in_place_copy_is_not_torn_by_the_commit_watchdog(flow, monkeypatch):
    """FR-32.5.7: the commit watchdog governs a trainer's bracket, never a weights load."""
    monkeypatch.setattr(settings, "WEIGHTS_COMMIT_WATCHDOG_MS", 10)
    real = checkpoint_service.copy_in_place

    def slow(model, handle):
        time.sleep(0.1)
        return real(model, handle)

    monkeypatch.setattr(checkpoint_service, "copy_in_place", slow)
    async with flow.client() as client:
        saved = await _saved(flow, client)
        r = await client.post("/api/weights/load", json={"name": saved["name"]}, headers=TRAINER)
    assert r.status_code == 200, r.text
    assert r.json()["data"]["weights_state"] == "clean"
    assert flow.barrier.snapshot().weights_state == "clean"


async def test_a_steering_lock_does_not_refuse_a_weights_load(flow, test_session):
    """D-24: the steering `locked` flag is ignored by a weights load (in place and row reload)."""
    from millm.db.models.model import Model

    row = await test_session.get(Model, 1)
    row.locked = True
    await test_session.commit()
    async with flow.client() as client:
        saved = await _saved(flow, client)
        r = await client.post("/api/weights/load", json={"name": saved["name"]}, headers=TRAINER)
        assert r.status_code == 200, r.text
        r = await client.post("/api/weights/load", json={"model_id": 1}, headers=TRAINER)
        assert r.status_code == 200, r.text
        assert r.json()["data"]["mode"] == "full_reload"


# --- 031 gaps closed (RX coverage walk) -------------------------------------------------------

import asyncio  # noqa: E402

import torch  # noqa: E402

from millm.ml.model_loader import LoadedModelState  # noqa: E402
from tests.unit.weights_e2e_fixtures import OPS, chat, params  # noqa: E402


async def _until(predicate, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


async def test_serving_continues_while_the_attachment_is_stale(flow, monkeypatch):
    """BRD §5 criterion 3 (CPU half): a dead trainer leaves serving untouched; status says stale."""
    async with flow.client() as client:
        await attach(client)
        monkeypatch.setattr(settings, "WEIGHTS_HEARTBEAT_TIMEOUT_S", 0.0)
        r = await chat(client)
        assert r.status_code == 200, r.text
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["attachment"]["state"] == "stale" and status["weights_state"] == "clean"


async def test_a_hold_timeout_resumes_serving_with_the_weights_unchanged(flow, monkeypatch):
    """BRD §5 criterion 6 end to end: no commit-begin within the hold time → self-resume, the
    parameter set is unchanged, `last_hold_timeout_at` is recorded and /v1 answers."""
    monkeypatch.setattr(settings, "WEIGHTS_HOLD_TIMEOUT_S", 0.05)
    before = params(flow.model)
    async with flow.client() as client:
        attachment_id = await attach(client)
        assert (await client.post("/api/weights/quiesce",
                                  headers=holder(attachment_id))).status_code == 200
        await _until(lambda: flow.barrier.snapshot().barrier == "serving")
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["last_hold_timeout_at"] is not None and status["weights_state"] == "clean"
        assert all(torch.equal(p, before[n]) for n, p in flow.model.named_parameters())
        assert (await chat(client)).status_code == 200


async def test_attach_on_a_quantized_model_is_400_over_http(flow):
    """BRD §5 criterion 5 at the route: 400 NOT_ATTACHABLE, reason quantized."""
    LoadedModelState().current.quantization_method = "bitsandbytes"
    async with flow.client() as client:
        r = await client.post("/api/weights/attach", headers=TRAINER)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "NOT_ATTACHABLE"
    assert r.json()["error"]["details"]["reason"] == "quantized"


async def test_attach_on_a_gguf_engine_is_400_over_http(flow):
    """BRD §5 criterion 5 / 031 §11 item 4, the GGUF half at the route: 400 NOT_ATTACHABLE,
    reason gguf, and no attachment is left behind (acceptance walk 2026-10-14)."""
    from millm.ml.model_loader import ENGINE_LLAMACPP

    LoadedModelState().current.engine = ENGINE_LLAMACPP
    async with flow.client() as client:
        r = await client.post("/api/weights/attach", headers=TRAINER)
        status = (await client.get("/api/weights/status")).json()["data"]
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "NOT_ATTACHABLE"
    assert r.json()["error"]["details"]["reason"] == "gguf"
    assert status["attached"] is False


async def test_an_operator_token_may_attach_and_hold(flow):
    """FR-31.14.7: every gated 031 call but clear-stale accepts either role."""
    async with flow.client() as client:
        r = await client.post("/api/weights/attach", headers=OPS)
        assert r.status_code == 200, r.text
        h = {**OPS, "X-miLLM-Attachment": r.json()["data"]["attachment_id"]}
        for path in ("heartbeat", "quiesce", "commit-begin", "commit-end", "resume", "detach"):
            resp = await client.post(f"/api/weights/{path}", headers=h)
            assert resp.status_code == 200, (path, resp.text)


async def test_run_model_work_is_refused_while_torn(flow):
    """FR-31.6.2: the model-work path refuses a torn set and never runs the work."""
    from millm.core.errors import WeightsTornError

    ran = []
    flow.barrier.mark_torn("test")
    with pytest.raises(WeightsTornError):
        await flow.inference.run_model_work(lambda: ran.append(1))
    assert ran == []


async def test_a_detach_while_writing_logs_who_was_writing(flow):
    """R-01.22 / FR-31.11.2 (R1 RX F3): the torn line names the attaching identity."""
    async with flow.client() as client:
        attachment_id = await attach(client)
        h = holder(attachment_id)
        assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
        assert (await client.post("/api/weights/commit-begin", headers=h)).status_code == 200
        with capture_logs() as logs:
            assert (await client.post("/api/weights/detach", headers=h)).status_code == 200
    torn = [e for e in logs if e["event"] == "weights_torn"]
    assert len(torn) == 1 and torn[0]["reason"] == "detach"
    assert (torn[0]["role"], torn[0]["label"]) == ("trainer", "forge")
