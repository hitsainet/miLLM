"""Adapter serving end to end through the REAL app (033 FTASKS 11.1, 11.2, 5.2; acceptance 15 on
the CPU).

Load over `POST /api/weights/adapter/load`, generate over `/v1/chat/completions`, unload, generate
again — on a tiny real Llama, with the real barrier and a real InferenceService. Only the database
backed ModelService is a stand-in, answering by exact row name, so a request naming the adapter's
name can only be served by the resident path. At every step the base parameters' data pointers
and bytes are unchanged; after unload the logits equal the bare base exactly. Run with nothing
attached and with an attachment reported.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from millm.core.config import settings
from millm.services.weight_attachment import WeightService, set_weight_service
from tests.unit.adapter_fixtures import param_fingerprint
from tests.unit.adapter_service_fixtures import (
    ADAPTER_NAME,
    OPERATOR_TOKEN,
    build_env,
    weights_tokens,
)

OPS = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
CHAT = {"messages": [{"role": "user", "content": "w1 w2 w3"}], "max_tokens": 3,
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
    from datetime import datetime
    from types import SimpleNamespace

    from tests.unit.adapter_fixtures import BASE_NAME

    return SimpleNamespace(
        id=1, name="tiny", architecture="text-generation", gguf_files=None, revision="abc",
        quantization=SimpleNamespace(value="FP16"), gguf_label="", weights_origin=None,
        repo_id=BASE_NAME, source=None, local_path=None, loaded_at=datetime(2026, 10, 14),
        created_at=datetime(2026, 10, 14), status="loaded",
    )


@pytest.fixture(params=[False, True], ids=["nothing_attached", "attached"])
def env(request, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "WEIGHTS_TOKENS", weights_tokens())
    spy = None
    if request.param:
        spy = MagicMock(spec=WeightService)
        spy.is_attached.return_value = True
        set_weight_service(spy)
    e = build_env(tmp_path)
    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app
    from millm.services.adapter_service import get_adapter_service

    rows = _Rows(_row())
    app = create_app()
    app.dependency_overrides[get_adapter_service] = lambda: e.service
    app.dependency_overrides[get_inference_service] = lambda: e.inference
    app.dependency_overrides[get_model_service] = lambda: rows
    e.client, e.rows = TestClient(app), rows
    yield e
    e.close()
    set_weight_service(None)
    if spy is not None:
        assert spy.method_calls == []


def _chat(env, model):
    r = env.client.post("/v1/chat/completions", json={"model": model, **CHAT})
    assert r.status_code == 200, r.text
    return r


def test_load_generate_unload_generate(env):
    rel = env.write()
    base_logits, fingerprint = env.logits(), param_fingerprint(env.model)
    base = _chat(env, "tiny")
    assert base.json()["model"] == "tiny"
    base_text = base.json()["choices"][0]["message"]["content"]

    r = env.client.post("/api/weights/adapter/load", json={"path": rel}, headers=OPS)
    assert r.status_code == 200, r.text
    assert param_fingerprint(env.model) == fingerprint
    adapted_logits = env.logits()
    assert not torch.equal(adapted_logits, base_logits)

    # The adapter's own name is served without any row by that name, and without a load.
    adapted = _chat(env, ADAPTER_NAME)
    assert adapted.json()["model"] == ADAPTER_NAME
    assert adapted.headers["X-miLLM-Model"].startswith(f'"{ADAPTER_NAME}"')
    # 5.2: the bare base name still means the resident — served adapted, no reload.
    bare = _chat(env, "tiny")
    assert bare.json()["model"] == ADAPTER_NAME
    env.rows.load_model_and_wait.assert_not_called()
    ids = [m["id"] for m in env.client.get("/v1/models").json()["data"]]
    assert ids == [ADAPTER_NAME]
    assert param_fingerprint(env.model) == fingerprint

    r = env.client.post("/api/weights/adapter/unload", headers=OPS)
    assert r.status_code == 200, r.text
    after = _chat(env, "tiny")
    assert after.json()["model"] == "tiny"
    assert after.headers["X-miLLM-Model"].startswith('"tiny"')
    assert after.json()["choices"][0]["message"]["content"] == base_text
    assert torch.equal(env.logits(), base_logits)
    assert param_fingerprint(env.model) == fingerprint
    env.rows.load_model_and_wait.assert_not_called()
    # The adapter's name no longer means anything.
    gone = env.client.post("/v1/chat/completions", json={"model": ADAPTER_NAME, **CHAT})
    assert gone.status_code == 404
