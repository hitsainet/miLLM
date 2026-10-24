"""R1-A: a token refusal on an adapter route audits under the SAME operation name the service
uses (`adapter/load`, `adapter/unload`), so one grep finds every attempt. `weights_auth` reads
`route.name`; before the fix it logged `load_adapter` / `unload_adapter`.

MUTATION CONTROLS (R1-A): C4 drop `name=OP_LOAD` from the load route -> this file red; m41
(`min_length=1` -> 0 on the request path) -> test_an_empty_path_is_a_422_before_the_service.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from millm.core.config import settings
from millm.services.adapter_service import get_adapter_service
from tests.unit.adapter_service_fixtures import build_env, weights_tokens


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "WEIGHTS_TOKENS", weights_tokens())
    e = build_env(tmp_path)
    from millm.main import create_app

    app = create_app()
    app.dependency_overrides[get_adapter_service] = lambda: e.service
    yield TestClient(app)
    e.close()


@pytest.mark.parametrize("path, operation", [
    ("/api/weights/adapter/load", "adapter/load"),
    ("/api/weights/adapter/unload", "adapter/unload"),
])
def test_token_refusal_audits_under_the_service_operation_name(client, path, operation):
    with patch("millm.services.weights_audit.logger") as log:
        r = client.post(path, json={"path": "x"} if path.endswith("load") and "unload" not in path
                        else None)
    assert r.status_code == 401
    calls = [c for c in log.warning.call_args_list if c.args and c.args[0] == "weights_audit"]
    assert len(calls) == 1
    assert calls[0].kwargs["operation"] == operation


def test_401_comes_before_body_validation(client):
    r = client.post("/api/weights/adapter/load", json={"path": "x", "force": True})
    assert r.status_code == 401


def test_an_empty_path_is_a_422_before_the_service(client):
    from tests.unit.adapter_service_fixtures import OPERATOR_TOKEN

    svc = client.app.dependency_overrides[get_adapter_service]()
    with patch.object(svc, "load") as load:
        r = client.post("/api/weights/adapter/load", json={"path": ""},
                        headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"})
    assert r.status_code == 422 and load.call_count == 0
