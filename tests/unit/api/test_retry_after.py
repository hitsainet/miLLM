"""Feature 29 task 6.6: `Retry-After` on every 503, from one policy, with a detector for a bypass.

Each producible 503 is asserted to carry ITS OWN code's value and to leave the middleware's
warning un-fired: a builder that forgot the header gets the distinct fallback (10) and the
`retry_after_defaulted` warning, and fails both assertions (controls M14, M15).

The middleware's logger is replaced by a spy rather than read through `capture_logs`, because
`setup_logging` caches loggers on first use and a capture would then see nothing.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from millm.core.errors import (
    InsufficientMemoryError,
    InvalidParameterError,
    ModelBusyError,
    ModelNotLoadedError,
)
from millm.services.cluster_hub_service import HubUnavailableError
from millm.services.request_queue import QueueFullError, RequestQueue
from tests.unit.lease_fixtures import (
    build_service,
    clear_resident,
    install_registry,
    resident_inference,
)


@pytest.fixture(autouse=True)
def _clean_loader():
    clear_resident()
    yield
    clear_resident()


@pytest.fixture
def warn_spy(monkeypatch):
    import millm.api.retry_after as module

    spy = MagicMock()
    monkeypatch.setattr(module, "logger", spy)
    return spy


def _defaulted(spy) -> list:
    return [c for c in spy.warning.call_args_list if c.args and c.args[0] == "retry_after_defaulted"]


def _app_with_raisers():
    """The real app plus synthetic routes that raise a given error on each API family, and a
    bare 503 that no builder made."""
    from millm.main import create_app

    app = create_app()
    errors = {
        "queue_full": lambda: QueueFullError("Request queue full (10 pending)."),
        "busy_load": lambda: ModelBusyError("loading", details={"loading_model_id": 2}),
        "busy_unload": lambda: ModelBusyError("unloading", details={"unloading": True}),
        "not_loaded": lambda: ModelNotLoadedError("no model"),
        "memory": lambda: InsufficientMemoryError("no card fits"),
        "hub": lambda: HubUnavailableError("hub down"),
    }

    async def raise_v1(name: str):
        raise errors[name]()

    async def raise_api(name: str):
        raise errors[name]()

    async def bare_503():
        return JSONResponse(status_code=503, content={"detail": "made by hand"})

    app.add_api_route("/v1/_test/raise/{name}", raise_v1, methods=["GET"])
    app.add_api_route("/api/_test/raise/{name}", raise_api, methods=["GET"])
    app.add_api_route("/api/_test/bare503", bare_503, methods=["GET"])
    return app


class TestEveryProducible503:
    @pytest.mark.parametrize("name, expected", [
        ("queue_full", "5"), ("busy_load", "15"), ("busy_unload", "5"),
        ("not_loaded", "30"), ("memory", "30"), ("hub", "1"),
    ])
    def test_v1_handler_sets_the_codes_own_value(self, name, expected, warn_spy):
        client = TestClient(_app_with_raisers())
        resp = client.get(f"/v1/_test/raise/{name}")
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == expected
        assert _defaulted(warn_spy) == []

    @pytest.mark.parametrize("name, expected", [("queue_full", "5"), ("hub", "1")])
    def test_management_handler_sets_it_on_its_503s(self, name, expected, warn_spy):
        client = TestClient(_app_with_raisers())
        resp = client.get(f"/api/_test/raise/{name}")
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == expected
        assert _defaulted(warn_spy) == []

    @pytest.mark.parametrize("name", ["busy_load", "memory"])
    def test_management_non_503_codes_carry_no_header(self, name, warn_spy):
        """MODEL_BUSY stays 409 and INSUFFICIENT_MEMORY 507 on management (FPRD §9)."""
        client = TestClient(_app_with_raisers())
        resp = client.get(f"/api/_test/raise/{name}")
        assert resp.status_code in (409, 507)
        assert "Retry-After" not in resp.headers

    def test_bare_503_gets_the_fallback_and_the_warning(self, warn_spy):
        """The detector: a 503 that bypassed the policy is visible (control M15)."""
        client = TestClient(_app_with_raisers())
        resp = client.get("/api/_test/bare503")
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "10"
        assert len(_defaulted(warn_spy)) == 1
        assert _defaulted(warn_spy)[0].kwargs["path"] == "/api/_test/bare503"

    def test_envelope_and_code_unchanged(self, warn_spy):
        client = TestClient(_app_with_raisers())
        body = client.get("/v1/_test/raise/busy_load").json()
        assert body == {"error": {"message": "loading", "type": "server_error",
                                  "param": None, "code": "model_busy"}}
        mgmt = client.get("/api/_test/raise/queue_full").json()
        assert mgmt["success"] is False and mgmt["error"]["code"] == "QUEUE_FULL"
        assert "retry_after" not in json.dumps(body) and "retry_after" not in json.dumps(mgmt)

    def test_a_non_503_gets_no_header(self, warn_spy):
        from millm.main import create_app

        app = create_app()

        async def bad():
            raise InvalidParameterError("bad", details={"param": "x"})

        app.add_api_route("/v1/_test/bad", bad, methods=["GET"])
        resp = TestClient(app).get("/v1/_test/bad")
        assert resp.status_code == 400 and "Retry-After" not in resp.headers


class TestTheRouteBuilders:
    @pytest.fixture
    def env(self):
        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.main import create_app

        install_registry()
        service, repo = build_service()
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: service
        app.dependency_overrides[get_inference_service] = resident_inference
        return TestClient(app), service

    def _chat(self, client, headers=None):
        return client.post("/v1/chat/completions", headers=headers or {},
                           json={"model": "m2", "messages": [{"role": "user", "content": "x"}]})

    @pytest.mark.parametrize("exc, expected", [
        (ModelBusyError("a load", details={"loading_model_id": 3}), "15"),
        (ModelBusyError("an unload", details={"model_id": 1, "unloading": True}), "5"),
    ])
    def test_model_busy_from_the_route(self, env, exc, expected, warn_spy):
        client, service = env
        service.load_model_and_wait = AsyncMock(side_effect=exc)
        resp = self._chat(client)
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "model_busy"
        assert resp.headers["Retry-After"] == expected
        assert _defaulted(warn_spy) == []

    def test_insufficient_memory_from_a_refused_load(self, env, warn_spy):
        client, service = env
        service.load_model_and_wait = AsyncMock(side_effect=InsufficientMemoryError("no fit"))
        resp = self._chat(client)
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "insufficient_memory"
        assert resp.headers["Retry-After"] == "30"
        assert _defaulted(warn_spy) == []

    def test_model_not_loaded_after_a_load(self, env, warn_spy):
        client, service = env
        service.load_model_and_wait = AsyncMock()  # "loaded", yet nothing is resident
        clear_resident()
        resp = self._chat(client)
        assert resp.status_code == 503
        err = resp.json()["error"]
        assert err["code"] == "model_not_loaded"
        assert "only after a model is loaded" in err["message"]
        assert resp.headers["Retry-After"] == "30"
        assert _defaulted(warn_spy) == []

    def test_model_loading_via_the_refuse_policy(self, env, warn_spy):
        client, service = env
        service._loading_model_id = 2
        resp = self._chat(client, headers={"X-miLLM-Load-Policy": "refuse"})
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "15"
        assert _defaulted(warn_spy) == []


class TestReadiness:
    def test_unready_probe_has_its_own_value(self, warn_spy):
        from millm.api.dependencies import get_model_loader
        from millm.main import create_app

        class Broken:
            @property
            def is_loaded(self):
                raise RuntimeError("loader gone")

        app = create_app()
        app.dependency_overrides[get_model_loader] = lambda: Broken()
        resp = TestClient(app).get("/api/health/ready")
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "5"
        assert _defaulted(warn_spy) == []

    def test_ready_probe_has_none(self):
        from millm.api.dependencies import get_model_loader
        from millm.main import create_app

        app = create_app()
        app.dependency_overrides[get_model_loader] = lambda: SimpleNamespace(
            is_loaded=False, model_name=None
        )
        resp = TestClient(app).get("/api/health/ready")
        assert resp.status_code == 200 and "Retry-After" not in resp.headers


class TestQueueFullBurst:
    """BRD-04 acceptance 15: eleven concurrent requests, MAX_PENDING_REQUESTS 10 → one 503
    queue_full with an integer Retry-After ≥ 1; the envelope and code unchanged."""

    async def _burst(self, monkeypatch, durations):
        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.main import create_app

        install_registry()
        service, _ = build_service()
        queue = RequestQueue(max_concurrent=1, max_pending=10)
        queue._durations.extend(durations)
        release = asyncio.Event()

        async def hold(request):
            async with queue.acquire():
                await release.wait()
            raise InvalidParameterError("released", details={"param": "prompt"})

        inference = resident_inference()
        inference.request_queue = queue
        inference._use_cbm = lambda: False
        inference.create_text_completion = hold
        monkeypatch.setattr("millm.api.dependencies.get_inference_service", lambda: inference)
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: service
        app.dependency_overrides[get_inference_service] = lambda: inference

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            async def one():
                return await client.post("/v1/completions", json={"model": "m1", "prompt": "x"})

            first_ten = [asyncio.create_task(one()) for _ in range(10)]
            for _ in range(1000):
                if queue.pending_count == 10:
                    break
                await asyncio.sleep(0.005)
            assert queue.pending_count == 10 and queue.holding_count == 1
            eleventh = await one()
            release.set()
            others = await asyncio.gather(*first_ten)
        return eleventh, others

    async def test_eleventh_request_is_503_with_the_default(self, monkeypatch, warn_spy):
        eleventh, others = await self._burst(monkeypatch, durations=[])
        assert eleventh.status_code == 503
        assert eleventh.json() == {"error": {
            "message": eleventh.json()["error"]["message"], "type": "server_error",
            "param": None, "code": "queue_full"}}
        assert eleventh.headers["Retry-After"] == "5"  # no estimate yet: the default
        assert all(r.status_code == 400 for r in others)
        assert _defaulted(warn_spy) == []

    async def test_eleventh_request_uses_the_estimate(self, monkeypatch, warn_spy):
        # median 2.0 s × (9 waiting + 1 holding) / 1 = 20 s
        eleventh, _ = await self._burst(monkeypatch, durations=[2.0, 2.0, 2.0])
        assert eleventh.status_code == 503
        assert eleventh.headers["Retry-After"] == "20"
        assert _defaulted(warn_spy) == []


class TestInStreamError:
    def test_a_503_code_carries_retry_after_in_the_event(self):
        from millm.services.inference_service import _stream_error_event

        event = _stream_error_event(ModelBusyError("unloading", details={"unloading": True}))
        body = json.loads(event.removeprefix("data: ").strip())
        assert body["error"]["retry_after"] == 5
        assert body["error"]["code"] == "model_busy"

    def test_a_4xx_code_carries_none(self):
        from millm.services.inference_service import _stream_error_event

        event = _stream_error_event(InvalidParameterError("bad"))
        assert "retry_after" not in event


class TestWiring:
    def test_the_middleware_is_in_the_live_app(self):
        """Control M15: the registration line."""
        from millm.api.retry_after import RetryAfterMiddleware
        from millm.main import create_app

        classes = [m.cls for m in create_app().user_middleware]
        assert classes.count(RetryAfterMiddleware) == 1
