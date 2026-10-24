"""Feature 29 task 5.8: `X-miLLM-Load-Policy` and `X-miLLM-Lease` on the three `/v1` routes.

The real app, the real ModelService (fake repository), the real loader. Model 1 ("m1") is
resident; requests name model 2 ("m2"). "No load started" is the background worker's call
count of 0 and `load_model_and_wait`'s await count of 0.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from millm.core.errors import ModelBusyError
from millm.db.models.model import ModelStatus
from tests.unit.lease_fixtures import (
    build_service,
    clear_resident,
    install_registry,
    resident_inference,
)

ROUTES = {
    "chat": ("/v1/chat/completions", lambda m: {"model": m, "messages": [
        {"role": "user", "content": "hi"}]}),
    "completions": ("/v1/completions", lambda m: {"model": m, "prompt": "hi"}),
    "embeddings": ("/v1/embeddings", lambda m: {"model": m, "input": "hi"}),
}


@pytest.fixture(autouse=True)
def _clean_loader():
    clear_resident()
    yield
    clear_resident()


@pytest.fixture
def env():
    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app

    registry, clock = install_registry()
    service, repo = build_service()
    service.load_model_and_wait = AsyncMock(wraps=service.load_model_and_wait)
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    app.dependency_overrides[get_inference_service] = resident_inference
    return TestClient(app), service, repo, registry


def _post(client, route, model="m2", headers=None, extra=None):
    path, body = ROUTES[route]
    payload = body(model)
    payload.update(extra or {})
    return client.post(path, json=payload, headers=headers or {})


@pytest.mark.parametrize("route", list(ROUTES))
class TestRefusePolicy:
    def test_refuse_non_resident_is_409_and_nothing_loads(self, env, route):
        client, service, repo, _ = env
        resp = _post(client, route, headers={"X-miLLM-Load-Policy": "refuse"})
        assert resp.status_code == 409, resp.text
        err = resp.json()["error"]
        assert err["code"] == "model_not_resident"
        assert err["type"] == "invalid_request_error"
        assert "'m2'" in err["message"] and "'m1'" in err["message"]
        assert service.load_model_and_wait.await_count == 0
        assert service._load_worker.call_count == 0
        assert repo.status_writes == []

    def test_refuse_reports_the_resident_models_lease(self, env, route):
        client, service, _, _ = env
        client.post("/api/models/1/lease",
                    json={"holder": "midataworks", "reason": "label run 7", "ttl_seconds": 120})
        resp = _post(client, route, headers={"X-miLLM-Load-Policy": "REFUSE"})
        assert resp.status_code == 409
        assert "midataworks" in resp.json()["error"]["message"]
        assert service.load_model_and_wait.await_count == 0

    def test_refuse_while_the_model_loads_is_503_with_retry_after(self, env, route):
        client, service, repo, _ = env
        repo.rows[2].status = ModelStatus.LOADING
        resp = _post(client, route, headers={"X-miLLM-Load-Policy": "refuse"})
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "model_loading"
        assert resp.headers["Retry-After"] == "15"
        assert service.load_model_and_wait.await_count == 0

    def test_refuse_while_the_slot_holds_this_model_is_503(self, env, route):
        client, service, _, _ = env
        service._loading_model_id = 2
        resp = _post(client, route, headers={"X-miLLM-Load-Policy": "refuse"})
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "model_loading"

    def test_absent_header_auto_loads_as_today(self, env, route):
        client, service, _, _ = env
        service.load_model_and_wait.side_effect = ModelBusyError(
            "stop here", details={"loading_model_id": 2}
        )
        resp = _post(client, route)
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "15"
        service.load_model_and_wait.assert_awaited_once_with(2, lease_id=None)

    def test_explicit_auto_auto_loads_and_passes_the_lease_header(self, env, route):
        client, service, _, _ = env
        service.load_model_and_wait.side_effect = ModelBusyError("stop", details={})
        _post(client, route, headers={"X-miLLM-Load-Policy": "auto", "X-miLLM-Lease": "abc"})
        service.load_model_and_wait.assert_awaited_once_with(2, lease_id="abc")

    def test_invalid_header_is_400_naming_it(self, env, route):
        client, service, _, _ = env
        resp = _post(client, route, headers={"X-miLLM-Load-Policy": "maybe"})
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["param"] == "X-miLLM-Load-Policy"
        assert service.load_model_and_wait.await_count == 0

    def test_refuse_for_the_resident_model_continues(self, env, route):
        """The resident model is not refused; the request reaches the inference service."""
        from millm.api.dependencies import get_inference_service

        client, service, _, _ = env
        inference = resident_inference()
        inference.active_circuit_rung = AsyncMock(return_value=None)
        reached = AsyncMock(side_effect=ModelBusyError("reached generation", details={}))
        for name in ("create_chat_completion", "create_text_completion", "create_embeddings"):
            setattr(inference, name, reached)
        client.app.dependency_overrides[get_inference_service] = lambda: inference
        resp = _post(client, route, model="m1", headers={"X-miLLM-Load-Policy": "refuse"})
        assert resp.status_code == 503, resp.text
        assert reached.await_count == 1
        assert service.load_model_and_wait.await_count == 0

    def test_foreign_lease_refuses_the_auto_load_as_model_leased(self, env, route):
        client, service, repo, _ = env
        repo.rows[1].locked = True  # both guards: the lease answer wins
        client.post("/api/models/1/lease",
                    json={"holder": "midataworks", "reason": "label run 7", "ttl_seconds": 120})
        resp = _post(client, route)
        assert resp.status_code == 409
        err = resp.json()["error"]
        assert err["code"] == "model_leased"
        assert "midataworks" in err["message"] and "2026-10-06T12:02:00" in err["message"]
        assert service._load_worker.call_count == 0
        assert repo.status_writes == []


class TestPreLoadRefusalsComeFirst:
    def test_embedding_only_model_is_refused_before_the_policy(self, env):
        client, service, repo, _ = env
        repo.rows[2].architecture = "feature-extraction"
        resp = _post(client, "chat", headers={"X-miLLM-Load-Policy": "refuse"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "model_not_generative"

    @pytest.mark.parametrize("route, extra", [
        ("chat", {"logprobs": True}), ("completions", {"logprobs": 1}),
    ])
    def test_gguf_refusal_comes_first(self, env, route, extra):
        client, service, repo, _ = env
        repo.rows[2].gguf_files = ["m2-Q4_K_M.gguf"]
        resp = _post(client, route, headers={"X-miLLM-Load-Policy": "refuse"}, extra=extra)
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["code"] != "model_not_resident"

    def test_unknown_model_is_404_first(self, env):
        client, _, _, _ = env
        resp = _post(client, "chat", model="nope", headers={"X-miLLM-Load-Policy": "refuse"})
        assert resp.status_code == 404


def _route_function(module):
    tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
    return [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
            and n.name.startswith("create_")]


@pytest.mark.parametrize("module_name", ["chat", "completions", "embeddings"])
def test_each_route_applies_the_policy_before_the_auto_load(module_name):
    """Control M13: the call, by AST, before `load_model_and_wait` in the same function."""
    import importlib

    module = importlib.import_module(f"millm.api.routes.openai.{module_name}")
    functions = _route_function(module)
    assert len(functions) == 1
    fn = functions[0]
    policy_lines = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and getattr(n.func, "id", None) == "apply_load_policy"]
    load_calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", None) == "load_model_and_wait"]
    assert len(policy_lines) == 1, f"{module_name}: apply_load_policy must be called once"
    assert len(load_calls) == 1
    assert policy_lines[0] < load_calls[0].lineno
    assert any(kw.arg == "lease_id" and getattr(kw.value, "id", None) == "x_millm_lease"
               for kw in load_calls[0].keywords)


def test_the_headers_are_declared_on_the_three_routes():
    from millm.main import create_app

    paths = create_app().openapi()["paths"]
    for path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
        names = {p["name"] for p in paths[path]["post"].get("parameters", []) if p["in"] == "header"}
        assert {"X-miLLM-Lease", "X-miLLM-Load-Policy"} <= names, path
