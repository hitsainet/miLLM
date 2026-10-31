"""Flow (b), end to end through the REAL app (review RX; BRD-01 §5 criteria 7, 8, 10 on the CPU).

A commit the watchdog tears: `/v1` refuses with `503 WEIGHTS_TORN` (a request already waiting is
woken and refused, never answered), status reports `torn`, the state survives detach and
re-attach, a Feature-1 management load is refused naming `POST /api/weights/load` (D-01), and
`POST /api/weights/load {"model_id": N}` recovers `clean` and serving resumes. Also the other
recovery path: a late `commit-end` from the holder (criterion 8).

The watchdog runs on the real event loop with a 50 ms setting; nothing here fakes a clock.
Harness and fakes: `tests/unit/weights_e2e_fixtures.py`.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import torch

from millm.core.config import settings
from millm.db.models.model import QuantizationType
from tests.support.factories import make_model
from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    OPS,
    ROW_ID,
    ROW_NAME,
    TRAINER,
    attach,
    chat,
    flow,
    holder,
    params,
)

WATCHDOG_MS = 50


@pytest.fixture
def short_watchdog(monkeypatch):
    monkeypatch.setattr(settings, "WEIGHTS_COMMIT_WATCHDOG_MS", WATCHDOG_MS)


async def _until(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


async def _torn_commit(flow, client) -> str:
    """attach → quiesce → commit-begin → half a write → the watchdog fires."""
    attachment_id = await attach(client)
    h = holder(attachment_id)
    assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
    assert (await client.post("/api/weights/commit-begin", headers=h)).status_code == 200
    with torch.no_grad():
        next(flow.model.parameters()).add_(1.0)  # part of the parameter set, then "killed"
    return attachment_id


async def test_watchdog_tears_and_a_row_load_recovers(flow, short_watchdog, test_session):
    base_text = None
    forwards = []
    async with flow.client() as client:
        r = await chat(client)
        assert r.status_code == 200, r.text
        base_text = r.json()["choices"][0]["message"]["content"]
        attachment_id = await _torn_commit(flow, client)

        # A request that arrives while `writing` is HELD; the tear wakes it and refuses it.
        handle = flow.model.register_forward_hook(lambda *_: forwards.append(1))
        waiting = asyncio.create_task(chat(client))
        await _until(lambda: flow.barrier.snapshot().weights_state == "torn")
        held = await asyncio.wait_for(waiting, timeout=5)
        assert held.status_code == 503, held.text
        assert held.json()["error"]["code"] == "weights_torn"
        assert held.headers.get("Retry-After") == "30"
        r = await chat(client)
        assert r.status_code == 503 and r.json()["error"]["code"] == "weights_torn"
        handle.remove()
        assert forwards == [], "a torn or writing parameter set was served (criterion 10)"

        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["weights_state"] == "torn" and status["commit_overran"] is True
        assert status["commit"]["overran"] is True and status["commit"]["began_at"]
        health = (await client.get("/api/health/detailed")).json()
        assert health["weights"]["weights_state"] == "torn"

        # Survives detach and re-attach (R-01.23): the next process sees what it inherited.
        assert (await client.post("/api/weights/detach",
                                  headers=holder(attachment_id))).status_code == 200
        assert (await client.get("/api/weights/status")).json()["data"]["weights_state"] == "torn"
        again = await client.post("/api/weights/attach", headers=TRAINER)
        assert again.status_code == 200, again.text
        assert again.json()["data"]["weights_state"] == "torn"
        second = again.json()["data"]["attachment_id"]
        assert (await client.post("/api/weights/detach",
                                  headers=holder(second))).status_code == 200

        # D-01: a Feature-1 management load is refused, and says where recovery lives.
        test_session.add(make_model(id=2, name="other", repo_id="org/other",
                                    quantization=QuantizationType.FP16,
                                    cache_path=str(flow.base_dir)))
        await test_session.commit()
        refused = await client.post("/api/models/2/load", json={})
        assert refused.status_code == 503, refused.text
        assert refused.json()["error"]["code"] == "WEIGHTS_TORN"
        assert "/api/weights/load" in refused.json()["error"]["message"]
        assert flow.loader.loads == []

        # The recovery: reload the row inside the barrier.
        loaded = await client.post("/api/weights/load", json={"model_id": ROW_ID},
                                   headers=TRAINER)
        assert loaded.status_code == 200, loaded.text
        body = loaded.json()["data"]
        assert body["mode"] == "full_reload" and body["weights_state"] == "clean"
        assert body["name"] == ROW_NAME and body["weights_sha256"] is None
        assert body["steps_since_load"] == 0
        assert [load["model_id"] for load in flow.loader.loads] == [ROW_ID]

        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["weights_state"] == "clean" and status["barrier"] == "serving"
        assert status["commit"] is None and status["commit_overran"] is False
        assert status["attached"] is False
        r = await chat(client)
        assert r.status_code == 200, r.text
        # The base directory holds the original values, so the answer is the original one.
        assert r.json()["choices"][0]["message"]["content"] == base_text
        assert r.headers["X-miLLM-Model"] == f'"{ROW_NAME}";origin=model;commits=0'


async def test_late_commit_end_recovers_and_records_the_overrun(flow, short_watchdog):
    async with flow.client() as client:
        attachment_id = await _torn_commit(flow, client)
        await _until(lambda: flow.barrier.snapshot().weights_state == "torn")
        assert (await chat(client)).status_code == 503
        written = params(flow.model)
        late = await client.post("/api/weights/commit-end", headers=holder(attachment_id))
        assert late.status_code == 200, late.text
        assert late.json()["data"]["overran"] is True
        assert late.json()["data"]["weights_state"] == "clean"
        assert late.json()["data"]["steps_since_load"] == 1
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["commit_overran"] is True
        assert status["last_overrun"]["commit_ms"] >= WATCHDOG_MS
        # Still quiesced by the holder: resume, then the completed write is served.
        assert (await client.post("/api/weights/resume",
                                  headers=holder(attachment_id))).status_code == 200
        r = await chat(client)
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Model"] == f'"{ROW_NAME}";origin=model;commits=1'
        assert all(torch.equal(p, written[n]) for n, p in flow.model.named_parameters())


async def test_operator_in_place_load_recovers_a_stale_holders_torn_set(flow, short_watchdog,
                                                                        monkeypatch):
    """The trainer was killed mid-commit (criterion 7): its attachment goes stale, an operator
    loads the last checkpoint in place WITHOUT clearing it, and serving resumes."""
    from tests.unit.weights_e2e_fixtures import manifest

    async with flow.client() as client:
        attachment_id = await attach(client)
        saved = await client.post("/api/weights/save", json={"manifest": manifest(step=3)},
                                  headers=holder(attachment_id))
        assert saved.status_code == 201, saved.text
        name = saved.json()["data"]["name"]
        good = params(flow.model)
        h = holder(attachment_id)
        assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
        assert (await client.post("/api/weights/commit-begin", headers=h)).status_code == 200
        with torch.no_grad():
            next(flow.model.parameters()).add_(1.0)
        await _until(lambda: flow.barrier.snapshot().weights_state == "torn")
        # The trainer is gone: no heartbeat within the window.
        monkeypatch.setattr(settings, "WEIGHTS_HEARTBEAT_TIMEOUT_S", 0.0)
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["attachment"]["state"] == "stale" and status["weights_state"] == "torn"
        # A trainer token that is not the holder may not load over it…
        assert (await client.post("/api/weights/load", json={"name": name},
                                  headers=TRAINER)).status_code == 409
        # …an operator may (torn or stale), in place.
        r = await client.post("/api/weights/load", json={"name": name}, headers=OPS)
        assert r.status_code == 200, r.text
        assert r.json()["data"]["mode"] == "in_place"
        assert r.json()["data"]["weights_state"] == "clean"
        assert all(torch.equal(p, good[n]) for n, p in flow.model.named_parameters())
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["weights_state"] == "clean"
        r = await chat(client, name)
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Model"] == f'"{name}";origin=checkpoint;commits=0'
        cleared = await client.post("/api/weights/clear-stale", headers=OPS)
        assert cleared.status_code == 200, cleared.text


async def test_management_load_of_the_torn_resident_row_names_the_recovery(flow, short_watchdog):
    async with flow.client() as client:
        attachment_id = await _torn_commit(flow, client)
        await _until(lambda: flow.barrier.snapshot().weights_state == "torn")
        assert (await client.post("/api/weights/detach",
                                  headers=holder(attachment_id))).status_code == 200
        refused = await client.post(f"/api/models/{ROW_ID}/load", json={})
        assert refused.status_code == 503, refused.text
        assert refused.json()["error"]["code"] == "WEIGHTS_TORN"
        assert "/api/weights/load" in refused.json()["error"]["message"]
        assert flow.loader.loads == []


async def test_precheck_load_refuses_torn_before_the_row_status(flow, short_watchdog,
                                                                test_session):
    """RX F2 in `precheck_load` as well as `load_model`: while torn, a row the status check
    would call busy (still downloading) is refused with the torn recovery instead."""
    from millm.core.errors import WeightsTornError
    from millm.db.models.model import ModelStatus

    test_session.add(make_model(id=2, name="other", repo_id="org/other",
                                status=ModelStatus.DOWNLOADING,
                                quantization=QuantizationType.FP16,
                                cache_path=str(flow.base_dir)))
    await test_session.commit()
    async with flow.client() as client:
        attachment_id = await _torn_commit(flow, client)
        await _until(lambda: flow.barrier.snapshot().weights_state == "torn")
        assert (await client.post("/api/weights/detach",
                                  headers=holder(attachment_id))).status_code == 200
    with pytest.raises(WeightsTornError) as exc:
        await flow.models.precheck_load(2)
    assert "/api/weights/load" in exc.value.message
    with pytest.raises(WeightsTornError):
        await flow.models.load_model(2)
    assert flow.loader.loads == []


async def test_every_v1_surface_is_held_while_writing_and_refused_once_torn(flow, short_watchdog):
    """Criterion 10 / 031 §11 item 8 across every `/v1` generation surface, not only chat:
    `/v1/completions`, `/v1/embeddings` and a streamed chat arriving while `writing` are HELD,
    woken by the tear and refused with `503 weights_torn`; the model's forward runs zero times.
    (Acceptance walk 2026-10-14 found only chat driven through this path.)"""
    forwards = []
    async with flow.client() as client:
        await _torn_commit(flow, client)
        assert flow.barrier.snapshot().weights_state == "writing"
        handle = flow.model.register_forward_hook(lambda *_: forwards.append(1))
        try:
            completions = asyncio.create_task(client.post("/v1/completions", json={
                "model": ROW_NAME, "prompt": "w1 w2 w3", "max_tokens": 3, "temperature": 0.0}))
            embeddings = asyncio.create_task(
                client.post("/v1/embeddings", json={"model": ROW_NAME, "input": "w1 w2 w3"}))
            streamed = asyncio.create_task(chat(client, stream=True))
            await asyncio.sleep(0.01)
            assert not any(t.done() for t in (completions, embeddings, streamed)), (
                "a request was answered while the parameter set was being written")
            await _until(lambda: flow.barrier.snapshot().weights_state == "torn")
            for task in (completions, embeddings):
                r = await asyncio.wait_for(task, timeout=5)
                assert r.status_code == 503, r.text
                assert r.json()["error"]["code"] == "weights_torn"
            # A stream admitted to the queue while `writing` has committed its 200; it ends with
            # the refusal and `[DONE]` and carries no content (the committed-stream contract,
            # `test_weights_admission.py::TestStreams`).
            r = await asyncio.wait_for(streamed, timeout=5)
            assert r.status_code == 200, r.text
            events = [line.removeprefix("data: ") for line in r.text.splitlines() if line]
            assert events[-1] == "[DONE]"
            assert json.loads(events[-2])["error"]["code"] == "weights_torn"
            assert all("choices" not in json.loads(e) for e in events[:-1]), r.text
            # And a fresh request on each surface after the tear is refused up front.
            for r in (
                await client.post("/v1/completions", json={"model": ROW_NAME, "prompt": "w1"}),
                await client.post("/v1/embeddings", json={"model": ROW_NAME, "input": "w1"}),
            ):
                assert r.status_code == 503 and r.json()["error"]["code"] == "weights_torn"
        finally:
            handle.remove()
    assert forwards == [], "a torn or writing parameter set was served (criterion 10)"
