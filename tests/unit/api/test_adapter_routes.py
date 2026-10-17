"""The adapter routes against the REAL app (033 FTASKS 7.1-7.4, 8.1, 4.12; addendum D-07, D-15,
D-28).

`create_app()` with only `get_adapter_service` overridden — to a REAL `AdapterService` over a tiny
real Llama, the real barrier and the vendored schema (`adapter_service_fixtures`). Reachability is
read from `app.openapi()["paths"]`; each route's service call is asserted by payload AND call
count through a `wraps=` spy; a missing or invalid token is refused before the service is called.

MUTATION CONTROLS: M1/M2 (delete a route registration) -> TestReachability; M26 (router include)
-> test_weights_routers_registered + TestReachability.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from millm.core.config import settings
from millm.core.weights_auth import WeightsIdentity
from millm.services import resident_identity
from millm.services.adapter_service import get_adapter_service, get_adapter_state
from tests.unit.adapter_fixtures import target_paths
from tests.unit.adapter_service_fixtures import (
    ADAPTER_NAME,
    OPERATOR_TOKEN,
    TRAINER_TOKEN,
    build_env,
    weights_tokens,
)

OPS = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
FORGE = {"Authorization": f"Bearer {TRAINER_TOKEN}"}
LOAD, UNLOAD, STATUS = "/api/weights/adapter/load", "/api/weights/adapter/unload", "/api/weights/adapter"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "WEIGHTS_TOKENS", weights_tokens())
    e = build_env(tmp_path)
    from millm.main import create_app

    app = create_app()
    app.dependency_overrides[get_adapter_service] = lambda: e.service
    e.app, e.client = app, TestClient(app)
    yield e
    e.close()


def _hooks(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


class TestReachability:
    def test_the_three_routes_are_served(self, env):
        paths = env.app.openapi()["paths"]
        assert "post" in paths[LOAD]
        assert "post" in paths[UNLOAD]
        assert "get" in paths[STATUS]

    def test_load_calls_the_service_once_with_the_payload(self, env):
        rel = env.write()
        with patch.object(env.service, "load", wraps=env.service.load) as spy:
            r = env.client.post(LOAD, json={"path": rel}, headers=OPS)
        assert r.status_code == 200, r.text
        spy.assert_awaited_once_with(
            WeightsIdentity(label="ops", role="operator"), rel, lease_id=None)
        data = r.json()["data"]
        assert data["adapter"]["name"] == ADAPTER_NAME
        assert data["resolved_model_name"] == ADAPTER_NAME
        assert data["adapter"]["target_count"] == 6

    def test_unload_calls_the_service_once_with_the_payload(self, env):
        rel = env.write()
        env.client.post(LOAD, json={"path": rel}, headers=FORGE)
        with patch.object(env.service, "unload", wraps=env.service.unload) as spy:
            r = env.client.post(UNLOAD, headers={**FORGE, "X-miLLM-Lease": "abc"})
        assert r.status_code == 200, r.text
        spy.assert_awaited_once_with(WeightsIdentity(label="forge", role="trainer"), lease_id="abc")
        assert r.json()["data"]["resolved_model_name"] == "tiny"

    def test_status_calls_the_service_once_and_needs_no_token(self, env):
        with patch.object(env.service, "status", wraps=env.service.status) as spy:
            r = env.client.get(STATUS)
        assert r.status_code == 200
        spy.assert_called_once_with()
        assert r.json()["data"] == {
            "applied": False, "name": None, "base_name": None, "manifest_sha256": None,
            "manifest_base_name": None, "manifest_base_sha256": None, "weights_sha256": None,
            "rank": None, "target_count": None, "loaded_at": None, "base_steps_since_load": None,
        }

    def test_the_live_app_uses_the_process_service(self):
        """Without an override the route resolves the process singleton."""
        from millm.main import create_app

        app = create_app()
        assert not app.dependency_overrides
        r = TestClient(app).get(STATUS)
        assert r.status_code == 200 and r.json()["data"]["applied"] is False
        assert get_adapter_service().state is get_adapter_state()


class TestTokenFirst:
    @pytest.mark.parametrize("headers, reason", [
        ({}, "missing"),
        ({"Authorization": "Bearer nope"}, "invalid_token"),
    ])
    @pytest.mark.parametrize("path", [LOAD, UNLOAD])
    def test_refused_before_the_service(self, env, headers, reason, path):
        with patch.object(env.service, "load") as load, patch.object(env.service, "unload") as un:
            r = env.client.post(path, json={"path": "x"} if path == LOAD else None, headers=headers)
        assert r.status_code == 401
        assert r.json()["error"]["details"]["reason"] == reason
        assert load.call_count == 0 and un.call_count == 0

    @pytest.mark.parametrize("path", [LOAD, UNLOAD])
    def test_no_tokens_configured_refuses_and_says_how(self, env, monkeypatch, path):
        """R1-C finding 6: the guidance is SPECIFIC to adapters. The old substring check
        ("operator token") passed against the generic "a trainer or operator token"."""
        monkeypatch.setattr(settings, "WEIGHTS_TOKENS", "")
        with patch.object(env.service, "load") as load, patch.object(env.service, "unload") as un:
            r = env.client.post(path, json={"path": "x"} if path == LOAD else None, headers=OPS)
        assert r.status_code == 401
        assert r.json()["error"]["details"]["reason"] == "no_tokens_configured"
        guidance = (
            "Serving an adapter needs an operator token even with no trainer; see the "
            "configuration reference (WEIGHTS_TOKENS, manual/docs/reference/configuration.md)."
        )
        details = r.json()["error"]["details"]
        assert details["guidance"] == guidance
        technical = details["technical_message"]
        assert technical.startswith("No weights tokens are configured (WEIGHTS_TOKENS is empty)")
        assert technical.endswith(" " + guidance)
        # The crafted message must reach the envelope's top-level message too: a generic
        # ERROR_MESSAGES entry for WEIGHTS_UNAUTHORIZED used to replace it (W3-C finding).
        assert r.json()["error"]["message"] == technical
        assert load.call_count == 0 and un.call_count == 0

    def test_the_adapter_guidance_stays_off_other_weights_routes(self, monkeypatch):
        """Operation-aware, not a blanket change: a non-adapter gated route keeps the generic
        message (no adapter sentence)."""
        from millm.core.errors import WeightsUnauthorizedError

        generic = WeightsUnauthorizedError.for_reason("no_tokens_configured", "weights/save")
        assert "adapter" not in generic.message and "guidance" not in generic.details
        assert generic.message == WeightsUnauthorizedError.for_reason(
            "no_tokens_configured").message
        other = WeightsUnauthorizedError.for_reason("missing", "adapter/load")
        assert "adapter" not in other.message

    def test_extra_fields_are_refused(self, env):
        with patch.object(env.service, "load") as load:
            r = env.client.post(LOAD, json={"path": "x", "force": True}, headers=OPS)
        assert r.status_code == 422 and load.call_count == 0


class TestRefusalsOverHttp:
    def test_acceptance_16_a_mismatched_adapter_is_a_400_naming_the_module(self, env):
        """8.1: through the real app; pause entered zero times; no hook; name unchanged."""
        off = target_paths(env.model, ["down_proj"])[0]
        rel = env.write(shape_off_by_one=off)
        with patch.object(env.barrier, "serving_pause", wraps=env.barrier.serving_pause) as pause:
            r = env.client.post(LOAD, json={"path": rel}, headers=OPS)
        assert r.status_code == 400, r.text
        error = r.json()["error"]
        assert error["code"] == "ADAPTER_SHAPE_MISMATCH"
        first = error["details"]["first"]
        assert first["module"] == off and first["reason"] == "shape"
        module = env.model.get_submodule(off)
        assert f"({module.out_features}, 4)" in first["detail"]          # expected B
        assert f"({module.out_features + 1}, 4)" in first["detail"]      # what the file has
        assert pause.call_count == 0
        assert _hooks(env.model) == 0
        assert resident_identity.served_name() == "tiny"

    def test_a_foreign_lease_is_409_and_the_holder_passes(self, env):
        rel = env.write()
        grant = env.leases.grant(1, "tiny", "midataworks", "label run", 120)
        r = env.client.post(LOAD, json={"path": rel}, headers=OPS)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "MODEL_LEASED"
        assert r.json()["error"]["details"]["holder"] == "midataworks"
        r = env.client.post(LOAD, json={"path": rel},
                            headers={**OPS, "X-miLLM-Lease": grant.lease_id})
        assert r.status_code == 200, r.text

    def test_second_load_and_empty_unload_are_409(self, env):
        rel = env.write()
        assert env.client.post(LOAD, json={"path": rel}, headers=OPS).status_code == 200
        r = env.client.post(LOAD, json={"path": rel}, headers=OPS)
        assert r.status_code == 409 and r.json()["error"]["code"] == "ADAPTER_ALREADY_LOADED"
        assert env.client.post(UNLOAD, headers=OPS).status_code == 200
        r = env.client.post(UNLOAD, headers=OPS)
        assert r.status_code == 409 and r.json()["error"]["code"] == "NO_ADAPTER_LOADED"

    def test_status_while_applied(self, env):
        rel = env.write()
        env.client.post(LOAD, json={"path": rel}, headers=OPS)
        data = env.client.get(STATUS).json()["data"]
        assert data["applied"] is True and data["name"] == ADAPTER_NAME
        assert data["base_name"] == "tiny" and data["base_steps_since_load"] == 0
        assert data["rank"] == 4 and data["target_count"] == 6 and data["loaded_at"]
