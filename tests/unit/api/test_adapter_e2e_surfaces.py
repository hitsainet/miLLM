"""Adapter serving through EVERY `/v1` surface of the real app (033 review R1-C, FTASKS 11.6).

`test_adapter_serving_integration.py` drives load -> chat -> unload -> chat. This file covers the
rest of what a client can see while an adapter is applied, each through `create_app()`, the real
routes, the real `AdapterService`, the real barrier and a real `InferenceService` over a tiny real
Llama (only the database-backed `ModelService` is a stand-in that answers by exact row name):

* `/v1/completions`, `/v1/embeddings` and a streaming chat report the adapter's own name in the
  body, in `X-miLLM-Model` (non-streaming) and in every stream chunk plus the terminal chunk's
  `millm_model` (FR-33.4.1, FR-33.4.2); after unload each reports the base again and the
  completion's and embedding's numbers equal the bare base's exactly (FR-33.2.4).
* `GET /v1/models/{name}` answers for the adapter's name and for the bare base name (D12).
* An adapter loaded while a stream is mid-generation waits for the stream: no forward pass of that
  stream runs with the hooks installed, and every chunk is labelled with the base (FR-33.4.3).
  The load is shown to have been WAITING during the stream (the barrier's hold owner is the
  adapter load while the stream's forwards still run), so the property is not vacuous.
* Continuous batching refuses load over HTTP with nothing installed (D-13).
* A model unload with an adapter applied drops it: no hook left, the name back to the base, the
  status empty (FR-33.2.5).
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

from millm.core.config import settings
from millm.ml.model_loader import ModelLoader
from millm.services import resident_identity
from millm.services.adapter_service import get_adapter_service, get_adapter_state
from tests.unit.adapter_fixtures import BASE_NAME, param_fingerprint
from tests.unit.adapter_service_fixtures import (
    ADAPTER_NAME,
    OPERATOR_TOKEN,
    build_env,
    weights_tokens,
)

OPS = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
LOAD, UNLOAD, STATUS = "/api/weights/adapter/load", "/api/weights/adapter/unload", "/api/weights/adapter"
COMPLETION = {"prompt": "w1 w2 w3", "max_tokens": 1, "temperature": 0.0, "logprobs": 5}
CHAT = {"messages": [{"role": "user", "content": "w1 w2 w3"}], "max_tokens": 4,
        "temperature": 0.0}


class _Rows:
    """The ModelService surface the /v1 routes use, answering by exact row name."""

    def __init__(self, row):
        self.rows = {row.name: row}
        self.find_model_by_name = AsyncMock(side_effect=lambda name: self.rows.get(name))
        self.load_model_and_wait = AsyncMock()
        self.get_locked_model = AsyncMock(return_value=None)
        self.get_available_models = AsyncMock(return_value=[row])
        self.by_id = {row.id: row}

    async def get_model(self, model_id):
        from millm.core.errors import ModelNotFoundError

        if model_id not in self.by_id:
            raise ModelNotFoundError(f"Model with ID {model_id} not found")
        return self.by_id[model_id]


def _row():
    return SimpleNamespace(
        id=1, name="tiny", architecture="text-generation", gguf_files=None, revision="abc",
        quantization=SimpleNamespace(value="FP16"), gguf_label="", weights_origin=None,
        repo_id=BASE_NAME, source=None, local_path=None, loaded_at=datetime(2026, 10, 14),
        created_at=datetime(2026, 10, 14), status="loaded",
    )


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


def _hooks(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


def _load(env) -> None:
    r = env.client.post(LOAD, json={"path": env.write()}, headers=OPS)
    assert r.status_code == 200, r.text


def _unload(env) -> None:
    r = env.client.post(UNLOAD, headers=OPS)
    assert r.status_code == 200, r.text


def _stream_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: "):]))
    return events


# ------------------------------------------------------------------ every surface reports it


def test_completions_report_the_adapter_and_return_to_the_base_exactly(env):
    base = env.client.post("/v1/completions", json={"model": "tiny", **COMPLETION})
    assert base.status_code == 200, base.text
    assert base.json()["model"] == "tiny"
    assert base.headers["X-miLLM-Model"].startswith('"tiny"')

    _load(env)
    for requested in (ADAPTER_NAME, "tiny"):  # the adapter's name, and the bare base (D12)
        r = env.client.post("/v1/completions", json={"model": requested, **COMPLETION})
        assert r.status_code == 200, r.text
        assert r.json()["model"] == ADAPTER_NAME
        assert r.headers["X-miLLM-Model"].startswith(f'"{ADAPTER_NAME}"')
    adapted_lp = r.json()["choices"][0]["logprobs"]
    assert adapted_lp != base.json()["choices"][0]["logprobs"]  # the adapter changed the answer

    _unload(env)
    after = env.client.post("/v1/completions", json={"model": "tiny", **COMPLETION})
    assert after.json()["model"] == "tiny"
    assert after.headers["X-miLLM-Model"].startswith('"tiny"')
    assert after.json()["choices"] == base.json()["choices"]
    env.rows.load_model_and_wait.assert_not_called()


def test_embeddings_report_the_adapter_and_return_to_the_base_exactly(env):
    body = {"input": "w1 w2 w3"}
    base = env.client.post("/v1/embeddings", json={"model": "tiny", **body})
    assert base.status_code == 200, base.text
    assert base.json()["model"] == "tiny"

    _load(env)
    adapted = env.client.post("/v1/embeddings", json={"model": ADAPTER_NAME, **body})
    assert adapted.status_code == 200, adapted.text
    assert adapted.json()["model"] == ADAPTER_NAME
    assert adapted.headers["X-miLLM-Model"].startswith(f'"{ADAPTER_NAME}"')
    base_vec = base.json()["data"][0]["embedding"]
    assert adapted.json()["data"][0]["embedding"] != base_vec

    _unload(env)
    after = env.client.post("/v1/embeddings", json={"model": "tiny", **body})
    assert after.json()["model"] == "tiny"
    assert after.json()["data"][0]["embedding"] == base_vec


def test_every_stream_chunk_reports_the_adapter(env):
    _load(env)
    r = env.client.post("/v1/chat/completions",
                        json={"model": ADAPTER_NAME, "stream": True, **CHAT})
    assert r.status_code == 200, r.text
    events = _stream_events(r.text)
    assert events, r.text
    assert {e.get("model") for e in events} == {ADAPTER_NAME}
    terminal = [e for e in events if "millm_model" in e]
    assert len(terminal) == 1
    assert terminal[0]["millm_model"].startswith(f'"{ADAPTER_NAME}"')

    _unload(env)
    r = env.client.post("/v1/chat/completions", json={"model": "tiny", "stream": True, **CHAT})
    events = _stream_events(r.text)
    assert {e.get("model") for e in events} == {"tiny"}
    terminal = [e for e in events if "millm_model" in e]
    assert terminal[0]["millm_model"].startswith('"tiny"')


def test_model_retrieval_by_the_bare_base_name_reports_the_adapter(env):
    _load(env)
    r = env.client.get("/v1/models/tiny")
    assert r.status_code == 200, r.text
    assert r.json()["id"] == ADAPTER_NAME
    _unload(env)
    assert env.client.get("/v1/models/tiny").json()["id"] == "tiny"


def test_model_retrieval_answers_for_the_adapter_name_listed_in_v1_models(env):
    """D-34 (R1-C finding 2): the adapter's listed id contains `/`, and the retrieve route
    answers for it (`{model_id:path}`)."""
    assert "/" in ADAPTER_NAME, "the fixture must exercise a name with a slash"
    _load(env)
    listed = [m["id"] for m in env.client.get("/v1/models").json()["data"]]
    assert listed == [ADAPTER_NAME]
    r = env.client.get(f"/v1/models/{ADAPTER_NAME}")
    assert r.status_code == 200, r.text
    assert r.json()["id"] == ADAPTER_NAME


# ------------------------------------------------------------- a load lands between requests


async def test_a_load_during_a_stream_waits_for_it_and_never_mixes(env):
    """FR-33.4.3: the stream that was running when the load arrived runs entirely on the base
    and is labelled the base; the load installs only after it. The forward wrapper records, for
    every forward pass, whether adapter hooks were installed and who owned the barrier."""
    model = env.model
    real_forward = model.forward
    seen: list[tuple[int, object]] = []
    started = threading.Event()

    def slow_forward(*args, **kwargs):
        seen.append((_hooks(model), env.barrier.snapshot().hold_owner))
        started.set()
        threading.Event().wait(0.05)  # a forward long enough for the load to arrive mid-stream
        return real_forward(*args, **kwargs)

    model.forward = slow_forward
    rel = env.write()
    transport = httpx.ASGITransport(app=env.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        stream = asyncio.create_task(client.post(
            "/v1/chat/completions",
            json={"model": "tiny", "stream": True, **{**CHAT, "max_tokens": 6}},
        ))
        while not started.is_set():
            await asyncio.sleep(0.005)
        load = asyncio.create_task(client.post(LOAD, json={"path": rel}, headers=OPS))
        stream_response = await stream
        stream_seen = list(seen)
        load_response = await load
    model.forward = real_forward

    assert stream_response.status_code == 200, stream_response.text
    assert load_response.status_code == 200, load_response.text
    events = _stream_events(stream_response.text)
    assert {e.get("model") for e in events} == {"tiny"}
    assert [e for e in events if "millm_model" in e][0]["millm_model"].startswith('"tiny"')
    # Not one forward of the running stream saw an adapter hook.
    assert stream_seen and all(hooks == 0 for hooks, _ in stream_seen), stream_seen
    # And the load really was waiting while the stream ran (the property is not vacuous).
    assert any(owner == "internal:adapter/load" for _, owner in stream_seen), stream_seen
    # Once loaded, the next request is adapted and says so.
    assert _hooks(model) > 0
    r = env.client.post("/v1/chat/completions", json={"model": "tiny", **CHAT})
    assert r.json()["model"] == ADAPTER_NAME


# --------------------------------------------------------------------- refusals and the drop


def test_continuous_batching_refuses_over_http_with_nothing_installed(env, monkeypatch):
    rel = env.write()
    monkeypatch.setattr(env.inference, "cbm_enabled", lambda: True)
    r = env.client.post(LOAD, json={"path": rel}, headers=OPS)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "ADAPTER_NOT_SERVABLE"
    assert r.json()["error"]["details"]["reason"] == "cbm"
    assert _hooks(env.model) == 0
    assert resident_identity.served_name() == "tiny"


def test_continuous_batching_refuses_unload_and_the_adapter_stays(env, monkeypatch):
    _load(env)
    monkeypatch.setattr(env.inference, "cbm_enabled", lambda: True)
    r = env.client.post(UNLOAD, headers=OPS)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["details"]["reason"] == "cbm"
    assert get_adapter_state().current is not None
    assert resident_identity.served_name() == ADAPTER_NAME


def test_a_model_unload_with_an_adapter_applied_drops_it(env):
    _load(env)
    fingerprint = param_fingerprint(env.model)
    assert _hooks(env.model) > 0
    model = env.model
    assert ModelLoader().unload() is True  # the path every Feature-1 unload ends in
    assert _hooks(model) == 0
    assert get_adapter_state().current is None
    assert param_fingerprint(model) == fingerprint
    data = env.client.get(STATUS).json()["data"]
    assert data["applied"] is False and data["name"] is None
    # Unload with nothing applied after the drop is the ordinary 409.
    r = env.client.post(UNLOAD, headers=OPS)
    assert r.status_code == 409 and r.json()["error"]["code"] == "NO_ADAPTER_LOADED"


def test_load_response_names_what_v1_now_reports(env):
    """FR-33.1.11: the response's `resolved_model_name` is what `/v1` reports; unload's is the
    bare base. Every field the FPRD lists is present."""
    r = env.client.post(LOAD, json={"path": env.write()}, headers=OPS)
    data = r.json()["data"]
    assert data["resolved_model_name"] == ADAPTER_NAME
    adapter = data["adapter"]
    assert set(adapter) >= {"name", "manifest_sha256", "weights_sha256", "target_count", "rank",
                            "scaling"}
    assert adapter["scaling"] == pytest.approx(6.0 / 4)
    chat = env.client.post("/v1/chat/completions", json={"model": "tiny", **CHAT})
    assert chat.json()["model"] == data["resolved_model_name"]
    r = env.client.post(UNLOAD, headers=OPS)
    assert r.json()["data"]["resolved_model_name"] == "tiny"
    assert r.json()["data"]["unloaded"]["manifest_sha256"] == adapter["manifest_sha256"]
