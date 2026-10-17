"""The requested name is re-checked once the slot is held (033 D-31), the header says when an
adapter supplied the name (D-33), and `GET /v1/models/{id}` answers names with `/` (D-34).

Real app, real routes, real `AdapterService`, real barrier and a real `InferenceService` over a
tiny real Llama (`build_env`); only the database-backed `ModelService` is a stand-in that answers
by exact row name.

D-31 (R1-C finding 4): the route resolves a name BEFORE admission. A chat naming the adapter, queued
behind an unload that is itself queued behind a running stream, used to be answered 200 by the bare
base (labelled honestly, but not what the client asked for). Now it is refused exactly as an
unknown name is: the same status, code, param and message as `model_not_found_error`.

MUTATION CONTROLS (W3-B report, each verified red, restore verified):
  M1  the name clause removed from `_admit.refused()`
  M2  the check compares the served name only (loses the bare base, D12)
  M2b `requested_model_name` returns None (no call site passes a name)
  M3  origin=adapter never set
  M4  `{model_id}` without `:path`
  M14 the in-slot re-check removed (the pre-queue check alone passes: the adapter is still
      applied when the request is parked behind the unload)
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from millm.api.routes.openai.errors import model_not_found_error
from millm.core.config import settings
from millm.core.errors import ModelNotFoundError
from millm.services.adapter_service import get_adapter_service
from millm.services.inference_service import (
    ServedModel,
    requested_model_not_found,
    served_model_header_value,
    snapshot_served_model,
)
from tests.unit.adapter_fixtures import BASE_NAME
from tests.unit.adapter_service_fixtures import (
    ADAPTER_NAME,
    OPERATOR_TOKEN,
    build_env,
    weights_tokens,
)

OPS = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
LOAD, UNLOAD = "/api/weights/adapter/load", "/api/weights/adapter/unload"
CHAT = {"messages": [{"role": "user", "content": "w1 w2 w3"}], "max_tokens": 2,
        "temperature": 0.0}


def _row(name: str = "tiny", model_id: int = 1):
    return SimpleNamespace(
        id=model_id, name=name, architecture="text-generation", gguf_files=None, revision="abc",
        quantization=SimpleNamespace(value="FP16"), gguf_label="", weights_origin=None,
        repo_id=BASE_NAME, source=None, local_path=None, loaded_at=datetime(2026, 10, 14),
        created_at=datetime(2026, 10, 14), status="loaded",
    )


class _Rows:
    def __init__(self, *rows):
        self.rows = {r.name: r for r in rows}
        self.by_id = {r.id: r for r in rows}
        self.find_model_by_name = AsyncMock(side_effect=lambda name: self.rows.get(name))
        self.load_model_and_wait = AsyncMock()
        self.get_locked_model = AsyncMock(return_value=None)
        self.get_available_models = AsyncMock(return_value=list(rows))

    async def get_model(self, model_id):
        if model_id not in self.by_id:
            raise ModelNotFoundError(f"Model with ID {model_id} not found")
        return self.by_id[model_id]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "WEIGHTS_TOKENS", weights_tokens())
    e = build_env(tmp_path)
    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app

    rows = _Rows(_row())
    app = create_app()
    app.dependency_overrides[get_adapter_service] = lambda: e.service
    app.dependency_overrides[get_inference_service] = lambda: e.inference
    app.dependency_overrides[get_model_service] = lambda: rows
    e.app, e.client, e.rows = app, TestClient(app), rows
    yield e
    e.close()


def _load(env) -> None:
    r = env.client.post(LOAD, json={"path": env.write()}, headers=OPS)
    assert r.status_code == 200, r.text


def _unload(env) -> None:
    r = env.client.post(UNLOAD, headers=OPS)
    assert r.status_code == 200, r.text


def _not_found_body(name: str, available: str) -> dict:
    return json.loads(model_not_found_error(name, available).body)


# ------------------------------------------------------------------------- D-31: the refusal


def test_the_refusal_is_the_unknown_model_refusal_exactly():
    """Same message as the route's `model_not_found_error`, in both of its forms, and the class
    the `/v1` handler maps to 404 `model_not_found` (`ERROR_STATUS_MAP`)."""
    from millm.api.routes.openai.errors import ERROR_STATUS_MAP

    for available in ("tiny", None):
        exc = requested_model_not_found("gone", available)
        expected = _not_found_body("gone", available)["error"]
        assert exc.message == expected["message"]
        assert exc.code.lower() == expected["code"] == "model_not_found"
        assert exc.details == {"param": "model"} and expected["param"] == "model"
        assert ERROR_STATUS_MAP[exc.code] == (404, expected["type"])


def test_the_check_reads_the_served_and_the_bare_base_name(env):
    _load(env)
    check = env.inference._requested_model_refusal
    assert check(ADAPTER_NAME) is None
    assert check("tiny") is None  # D12: the bare base keeps working while adapted
    assert check(None) is None  # work that requested no model is never refused here
    refusal = check("some-other")
    assert isinstance(refusal, ModelNotFoundError)
    assert refusal.message == _not_found_body("some-other", ADAPTER_NAME)["error"]["message"]
    _unload(env)
    assert isinstance(check(ADAPTER_NAME), ModelNotFoundError)
    assert check("tiny") is None


def test_the_bare_base_name_is_served_by_the_adapter_while_applied(env):
    _load(env)
    r = env.client.post("/v1/chat/completions", json={"model": "tiny", **CHAT})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == ADAPTER_NAME


async def _queued_behind_an_unload(env, requested: str, *, stream: bool = False):
    """R1-C's scenario: a stream naming the adapter holds the slot; an unload is queued behind
    it; then a request naming `requested` is queued behind the unload. Returns the three
    responses and whether the request was really sent while the adapter was still applied."""
    _load(env)
    model = env.model
    real_forward = model.forward
    started = threading.Event()

    def slow_forward(*args, **kwargs):
        started.set()
        threading.Event().wait(0.05)
        return real_forward(*args, **kwargs)

    model.forward = slow_forward
    transport = httpx.ASGITransport(app=env.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            running = asyncio.create_task(client.post(
                "/v1/chat/completions",
                json={"model": ADAPTER_NAME, "stream": True, **{**CHAT, "max_tokens": 6}},
            ))
            while not started.is_set():
                await asyncio.sleep(0.005)
            unload = asyncio.create_task(client.post(UNLOAD, headers=OPS))
            while env.barrier.snapshot().hold_owner != "internal:adapter/unload":
                await asyncio.sleep(0.005)
            applied_when_sent = get_adapter_service().status().get("applied") is True
            queued = asyncio.create_task(client.post(
                "/v1/chat/completions", json={"model": requested, "stream": stream, **CHAT},
            ))
            responses = await asyncio.gather(running, unload, queued)
    finally:
        model.forward = real_forward
    return responses, applied_when_sent


async def test_a_request_naming_the_adapter_queued_behind_its_unload_is_404_not_the_base(env):
    (running, unload, queued), applied_when_sent = await _queued_behind_an_unload(
        env, ADAPTER_NAME
    )
    assert applied_when_sent, "the request must have been resolved while the adapter applied"
    assert running.status_code == 200, running.text
    assert unload.status_code == 200, unload.text
    assert queued.status_code == 404, queued.text
    assert queued.json() == _not_found_body(ADAPTER_NAME, "tiny")
    assert "X-miLLM-Model" not in queued.headers


async def test_the_bare_base_name_queued_behind_the_unload_is_served_by_the_base(env):
    (_running, unload, queued), _ = await _queued_behind_an_unload(env, "tiny")
    assert unload.status_code == 200
    assert queued.status_code == 200, queued.text
    assert queued.json()["model"] == "tiny"


async def test_a_stream_naming_the_adapter_queued_behind_its_unload_ends_in_the_refusal(env):
    """A stream commits its 200 before it is admitted; the refusal arrives as its error event,
    with the unknown-model code, and no token was generated by the base."""
    (_running, _unload_r, queued), _ = await _queued_behind_an_unload(
        env, ADAPTER_NAME, stream=True
    )
    events = [json.loads(line[len("data: "):]) for line in queued.text.splitlines()
              if line.startswith("data: {")]
    errors = [e["error"] for e in events if "error" in e]
    assert errors and errors[0]["code"] == "model_not_found", queued.text
    assert not [e for e in events if e.get("choices")], "the base must not have answered"


# ------------------------------------------------------------------- D-33: origin=adapter


def test_the_header_says_origin_adapter_while_applied_and_model_after(env):
    base = env.client.post("/v1/chat/completions", json={"model": "tiny", **CHAT})
    assert base.headers["X-miLLM-Model"].startswith('"tiny";origin=model')
    _load(env)
    for requested in (ADAPTER_NAME, "tiny"):
        r = env.client.post("/v1/chat/completions", json={"model": requested, **CHAT})
        assert r.status_code == 200, r.text
        header = r.headers["X-miLLM-Model"]
        assert header.startswith(f'"{ADAPTER_NAME}";origin=adapter'), header
    stream = env.client.post("/v1/chat/completions",
                             json={"model": ADAPTER_NAME, "stream": True, **CHAT})
    terminal = [json.loads(line[len("data: "):]) for line in stream.text.splitlines()
                if line.startswith("data: {") and "millm_model" in line]
    assert terminal[0]["millm_model"].startswith(f'"{ADAPTER_NAME}";origin=adapter')
    _unload(env)
    after = env.client.post("/v1/chat/completions", json={"model": "tiny", **CHAT})
    assert after.headers["X-miLLM-Model"].startswith('"tiny";origin=model')


def test_origin_adapter_only_beside_the_adapters_own_name(monkeypatch):
    """An overlay name that is not the applied adapter's keeps the identity's origin: the origin
    describes the name it is printed beside."""
    import millm.services.adapter_service as adapter_service

    loaded = SimpleNamespace(model_name="tiny", weights_identity=None, model_id=1)
    monkeypatch.setattr(adapter_service, "applied_adapter_name", lambda: "x:adapter-1")
    monkeypatch.setattr("millm.services.resident_identity.served_name", lambda _l=None: "tiny")
    assert snapshot_served_model(loaded).origin == "model"
    monkeypatch.setattr(
        "millm.services.resident_identity.served_name", lambda _l=None: "x:adapter-1"
    )
    served = snapshot_served_model(loaded)
    assert served.origin == "adapter"
    assert served_model_header_value(served).startswith('"x:adapter-1";origin=adapter')
    assert served_model_header_value(ServedModel("n", "checkpoint", 2)) == '"n";origin=checkpoint;commits=2'


# ------------------------------------------------------------- D-34: names with a slash


def test_an_unknown_name_with_a_slash_is_the_openai_404_not_fastapis(env):
    r = env.client.get("/v1/models/org/nope")
    assert r.status_code == 404
    assert r.json() == _not_found_body("org/nope", None)


def test_a_plain_row_whose_name_has_a_slash_is_retrieved(env):
    env.rows.rows["org/other"] = _row("org/other", 2)
    env.rows.by_id[2] = env.rows.rows["org/other"]
    r = env.client.get("/v1/models/org/other")
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "org/other"
