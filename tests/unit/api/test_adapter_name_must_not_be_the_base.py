"""An adapter must never be served under the bare base's name (033 review R1-C; R-01.27 bullet 4).

R-01.27: "A completion served with an adapter never reports the bare base name." The served name
while applied is the manifest's own `name` (D11), and the vendored schema constrains it only to be
a string. Nothing at load compares it with the resident's names, so (measured through the real
app on a tiny Llama):

* a manifest whose `name` is the base row's name (`"tiny"`) loads, and every completion, the
  `X-miLLM-Model` header and `/v1/models` then report `"tiny"` while the adapter answers;
* an empty `name` loads too, and `resident_identity.served_name` skips the falsy overlay, so the
  same thing happens — the status says `applied: true, name: ""` and every response says the base.

A plain-PEFT manifest is written by hand or by `write_adapter_manifest --name`, so neither case
needs a malicious author. Proposed fix (millm/services/adapter_artifact.py or adapter_service.py):
refuse at load, before the pause, a `name` that is empty or not printable, or that equals the base
served name, the row name or its repo id (`ADAPTER_MANIFEST_INVALID`, reason `name`); and in
`write_adapter_manifest`, refuse such a `--name`.

Fixed in R1-B `b449554` (`adapter_artifact.require_adapter_name`, addendum D-32); this is now an
ordinary regression test. Refusing a name equal to ANY registered row still needs the service to
pass `registered_names` (tracked for W3).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from millm.core.config import settings
from millm.services.adapter_service import get_adapter_service
from tests.unit.adapter_fixtures import BASE_NAME
from tests.unit.adapter_service_fixtures import OPERATOR_TOKEN, build_env, weights_tokens

OPS = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
CHAT = {"messages": [{"role": "user", "content": "w1 w2"}], "max_tokens": 2, "temperature": 0.0}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "WEIGHTS_TOKENS", weights_tokens())
    e = build_env(tmp_path)
    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app

    row = SimpleNamespace(
        id=1, name="tiny", architecture="text-generation", gguf_files=None, revision="abc",
        quantization=SimpleNamespace(value="FP16"), gguf_label="", weights_origin=None,
        repo_id=BASE_NAME, source=None, local_path=None, loaded_at=datetime(2026, 10, 14),
        created_at=datetime(2026, 10, 14), status="loaded",
    )
    rows = SimpleNamespace(
        find_model_by_name=AsyncMock(side_effect=lambda n: row if n == "tiny" else None),
        get_model=AsyncMock(return_value=row), load_model_and_wait=AsyncMock(),
        get_locked_model=AsyncMock(return_value=None),
        get_available_models=AsyncMock(return_value=[row]),
    )
    app = create_app()
    app.dependency_overrides[get_adapter_service] = lambda: e.service
    app.dependency_overrides[get_inference_service] = lambda: e.inference
    app.dependency_overrides[get_model_service] = lambda: rows
    e.client = TestClient(app)
    yield e
    e.close()


@pytest.mark.parametrize("name", ["tiny", ""], ids=["base_row_name", "empty"])
def test_an_adapter_is_never_reported_under_the_bare_base_name(env, name):
    rel = env.write(manifest_overrides={"name": name})
    r = env.client.post("/api/weights/adapter/load", json={"path": rel}, headers=OPS)
    if r.status_code != 200:
        # The fix refuses it before anything is installed: that satisfies R-01.27.
        assert r.status_code == 400 and r.json()["error"]["code"] == "ADAPTER_MANIFEST_INVALID"
        return
    chat = env.client.post("/v1/chat/completions", json={"model": "tiny", **CHAT})
    assert chat.json()["model"] != "tiny"
    assert not chat.headers["X-miLLM-Model"].startswith('"tiny"')
