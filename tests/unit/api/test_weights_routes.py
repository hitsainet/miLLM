"""The nine `/api/weights` routes of Feature 31 (FTASKS 6.1-6.8), through the REAL app.

The app is `create_app()` (routes, dependencies, error handlers; no lifespan) driven over ASGI on
the test's own event loop, so the barrier's holder task outlives one request as it does in
production. The barrier is the REAL `WeightBarrier` holding a real one-slot admission; the weight
service is the real `WeightService` with the GPU-facing collaborators faked (export, resident
model, clocks), exactly as in `tests/unit/services/test_weight_attachment.py`.

Every route's service call is asserted by payload AND call count through a `wraps=` spy, and
reachability is read from `app.openapi()["paths"]`, never `app.routes` (memory
`app-routes-is-not-a-route-list`). Removing the router include, or one route, turns this red.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from structlog.testing import capture_logs

import millm.services.weight_barrier as barrier_module
from millm.core.config import settings
from millm.core.weights_auth import WeightsIdentity
from millm.services.weight_attachment import set_weight_service
from millm.services.weight_barrier import WeightBarrier, set_weight_barrier
from tests.unit.services.test_weight_attachment import TIMEOUT, _build, _RealInference

TRAINER_TOKEN = "trainer-token-0123456789"
OPERATOR_TOKEN = "operator-token-9876543210"
TRAINER = WeightsIdentity("miforge", "trainer")
OPERATOR = WeightsIdentity("ops", "operator")
HEADER = "X-miLLM-Attachment"

#: The nine routes (031 FTDD §5.2): path -> method.
ROUTES = {
    "/api/weights/attach": "post",
    "/api/weights/detach": "post",
    "/api/weights/heartbeat": "post",
    "/api/weights/quiesce": "post",
    "/api/weights/resume": "post",
    "/api/weights/commit-begin": "post",
    "/api/weights/commit-end": "post",
    "/api/weights/clear-stale": "post",
    "/api/weights/status": "get",
}


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _auth(token: str = TRAINER_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def tokens_json() -> str:
    return json.dumps([
        {"label": "miforge", "role": "trainer", "sha256": _digest(TRAINER_TOKEN)},
        {"label": "ops", "role": "operator", "sha256": _digest(OPERATOR_TOKEN)},
    ])


class Env:
    def __init__(self, app: Any, service: Any, fakes: Any, barrier: WeightBarrier, inference: Any):
        self.app, self.service, self.fakes = app, service, fakes
        self.barrier, self.inference = barrier, inference

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False),
            base_url="http://test",
        )


@pytest.fixture
def env(monkeypatch):
    from millm.main import create_app

    monkeypatch.setattr(settings, "WEIGHTS_TOKENS", tokens_json())
    monkeypatch.setattr(barrier_module, "_synchronize_resident_devices", lambda: None)
    monkeypatch.setattr(settings, "WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S", 2.0)
    inference = _RealInference()
    barrier = WeightBarrier(inference=lambda: inference)
    service, fakes = _build(inference=inference)
    service._barrier = lambda: barrier
    fakes.barrier = barrier
    set_weight_barrier(barrier)
    set_weight_service(service)
    yield Env(create_app(), service, fakes, barrier, inference)
    barrier.reset("test teardown")


def _walk(value: Any):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _walk(v)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, str):
        yield value


async def _attach(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/weights/attach", headers=_auth())
    assert response.status_code == 200, response.text
    return response.json()["data"]["attachment_id"]


def _audits(logs: list[dict]) -> list[dict]:
    return [e for e in logs if e["event"] == "weights_audit"]


# --- reachability (6.8) ------------------------------------------------------------------------


class TestReachability:
    def test_the_nine_routes_are_served(self, env):
        paths = env.app.openapi()["paths"]
        for path, method in ROUTES.items():
            assert method in paths.get(path, {}), f"{method.upper()} {path} is not served"

    def test_the_holder_header_is_optional_so_a_missing_one_is_never_422(self, env):
        params = env.app.openapi()["paths"]["/api/weights/quiesce"]["post"]["parameters"]
        header = next(p for p in params if p["name"] == HEADER)
        assert header["in"] == "header" and header["required"] is False

    async def test_without_the_router_include_the_routes_are_gone(self, monkeypatch):
        """M17: the include is what serves them."""
        from fastapi import FastAPI

        from millm.api.routes.management.weights import router as weights_router
        from millm.main import create_app

        real = FastAPI.include_router

        def skip_weights(app, router, *args, **kwargs):
            if router is weights_router:
                return None
            return real(app, router, *args, **kwargs)

        monkeypatch.setattr(FastAPI, "include_router", skip_weights)
        app = create_app()
        paths = app.openapi()["paths"]
        assert not any(path in paths for path in ROUTES)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/api/weights/status")).status_code == 404


# --- each route: payload and call count (6.7) ------------------------------------------------


class TestEachRouteCallsTheServiceOnce:
    async def test_attach(self, env):
        with patch.object(env.service, "attach", wraps=env.service.attach) as spy:
            async with env.client() as client:
                response = await client.post("/api/weights/attach", headers=_auth())
        assert response.status_code == 200
        assert spy.call_count == 1 and spy.call_args.args == (TRAINER,)
        body = response.json()
        assert body["success"] is True and body["error"] is None
        assert set(body["data"]) == {
            "attachment_id", "model", "weights_state", "steps_since_load", "heartbeat",
            "producer", "storages", "parameters", "release", "config",
        }
        assert body["data"]["model"] == {"model_id": 7, "name": "tiny-model", "manifest_hash": None}

    @pytest.mark.parametrize(
        "path,method",
        [
            ("/api/weights/heartbeat", "heartbeat"),
            ("/api/weights/quiesce", "quiesce"),
            ("/api/weights/detach", "detach"),
        ],
    )
    async def test_holder_route(self, env, path, method):
        async with env.client() as client:
            attachment_id = await _attach(client)
            with patch.object(env.service, method, wraps=getattr(env.service, method)) as spy:
                response = await client.post(path, headers={**_auth(), HEADER: attachment_id})
            if method == "quiesce":
                await client.post("/api/weights/resume", headers={**_auth(), HEADER: attachment_id})
        assert response.status_code == 200, response.text
        assert spy.call_count == 1 and spy.call_args.args == (TRAINER, attachment_id)

    async def test_resume_commit_begin_commit_end(self, env):
        async with env.client() as client:
            attachment_id = await _attach(client)
            holder = {**_auth(), HEADER: attachment_id}
            assert (await client.post("/api/weights/quiesce", headers=holder)).status_code == 200
            for path, method in [
                ("/api/weights/commit-begin", "commit_begin"),
                ("/api/weights/commit-end", "commit_end"),
                ("/api/weights/resume", "resume"),
            ]:
                with patch.object(env.service, method, wraps=getattr(env.service, method)) as spy:
                    response = await client.post(path, headers=holder)
                assert response.status_code == 200, (path, response.text)
                assert spy.call_count == 1 and spy.call_args.args == (TRAINER, attachment_id)

    async def test_clear_stale(self, env):
        async with env.client() as client:
            await _attach(client)
            env.fakes.clock.advance(TIMEOUT + 1)
            with patch.object(env.service, "clear_stale", wraps=env.service.clear_stale) as spy:
                response = await client.post(
                    "/api/weights/clear-stale", headers=_auth(OPERATOR_TOKEN)
                )
        assert response.status_code == 200, response.text
        assert spy.call_count == 1 and spy.call_args.args == (OPERATOR,)
        data = response.json()["data"]
        assert data["cleared"]["role"] == "trainer" and data["cleared"]["label"] == "miforge"
        assert data["cleared"]["stale_since"] is not None
        assert (data["weights_state"], data["barrier"]) == ("clean", "serving")

    async def test_status(self, env):
        with patch.object(env.service, "status", wraps=env.service.status) as spy:
            async with env.client() as client:
                response = await client.get("/api/weights/status")
        assert response.status_code == 200
        assert spy.call_count == 1 and spy.call_args.args == ()


# --- the full bracket through HTTP (6.7) -------------------------------------------------------


class TestTheFullFlow:
    async def test_attach_quiesce_commit_resume_detach_with_audit_lines_in_order(self, env):
        with capture_logs() as logs:
            async with env.client() as client:
                attachment_id = await _attach(client)
                holder = {**_auth(), HEADER: attachment_id}
                q = (await client.post("/api/weights/quiesce", headers=holder)).json()["data"]
                assert set(q) == {"held_requests", "held_batch_chunks", "weights_state",
                                  "quiesced_at", "hold_deadline"}
                status = (await client.get("/api/weights/status")).json()["data"]
                assert status["barrier"] == "quiesced" and status["quiesced_at"] is not None
                assert status["held_requests"] == 0
                b = (await client.post("/api/weights/commit-begin", headers=holder)).json()["data"]
                assert b["weights_state"] == "writing"
                assert b["watchdog_ms"] == int(settings.WEIGHTS_COMMIT_WATCHDOG_MS)
                status = (await client.get("/api/weights/status")).json()["data"]
                assert status["weights_state"] == "writing"
                assert status["commit"]["began_at"] == b["began_at"]
                assert status["commit"]["overran"] is False
                e = (await client.post("/api/weights/commit-end", headers=holder)).json()["data"]
                assert e["weights_state"] == "clean" and e["steps_since_load"] == 1
                assert e["overran"] is False and e["commit_ms"] >= 0
                r = (await client.post("/api/weights/resume", headers=holder)).json()["data"]
                assert r == {"barrier": "serving", "weights_state": "clean"}
                d = (await client.post("/api/weights/detach", headers=holder)).json()["data"]
                assert d == {"weights_state": "clean", "barrier": "serving"}
        audits = _audits(logs)
        assert [(a["operation"], a["outcome"]) for a in audits] == [
            ("weights_attach", "accepted"),
            ("weights_quiesce", "accepted"),
            ("weights_commit_begin", "accepted"),
            ("weights_commit_end", "accepted"),
            ("weights_resume", "accepted"),
            ("weights_detach", "accepted"),
        ]
        ref = hashlib.sha256(attachment_id.encode()).hexdigest()[:8]
        assert all(a["attachment_ref"] == ref for a in audits)
        assert all(a["role"] == "trainer" and a["label"] == "miforge" for a in audits)
        begin = audits[2]
        assert begin["weights_state_before"] == "clean" and begin["weights_state"] == "writing"
        end = audits[3]
        assert end["weights_state_before"] == "writing" and end["steps"] == 1
        assert end["elapsed_ms"] == end["commit_ms"] == e["commit_ms"]
        text = str(logs)
        assert attachment_id not in text and TRAINER_TOKEN not in text

    async def test_a_watchdog_tear_reads_torn_and_a_late_commit_end_is_an_overrun(self, env):
        async with env.client() as client:
            attachment_id = await _attach(client)
            holder = {**_auth(), HEADER: attachment_id}
            await client.post("/api/weights/quiesce", headers=holder)
            await client.post("/api/weights/commit-begin", headers=holder)
            env.barrier._watchdog_deadline_mono = 0.0
            env.barrier._on_watchdog(env.barrier._watchdog_generation)
            status = (await client.get("/api/weights/status")).json()["data"]
            assert status["weights_state"] == "torn" and status["commit"]["overran"] is True
            e = (await client.post("/api/weights/commit-end", headers=holder)).json()["data"]
            assert e["overran"] is True and e["weights_state"] == "clean"
            status = (await client.get("/api/weights/status")).json()["data"]
            assert status["last_overrun"]["commit_ms"] == e["commit_ms"]
            await client.post("/api/weights/resume", headers=holder)


# --- refusals ----------------------------------------------------------------------------------


class TestRefusals:
    async def test_no_token_is_401_with_the_bearer_challenge_and_audited(self, env):
        with capture_logs() as logs:
            async with env.client() as client:
                response = await client.post("/api/weights/attach")
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        body = response.json()
        assert body["success"] is False and body["error"]["code"] == "WEIGHTS_UNAUTHORIZED"
        audit = _audits(logs)
        assert len(audit) == 1 and audit[0]["operation"] == "weights_attach"
        assert audit[0]["role"] is None and audit[0]["outcome"] == "refused"

    @pytest.mark.parametrize("path", [p for p, m in ROUTES.items() if m == "post"])
    async def test_every_gated_route_refuses_without_a_token(self, env, path):
        async with env.client() as client:
            response = await client.post(path)
        assert response.status_code == 401, path

    @pytest.mark.parametrize("path, operation", [
        *[(p, "weights_" + p.rsplit("/", 1)[1].replace("-", "_"))
          for p, m in ROUTES.items() if m == "post"],
        ("/api/weights/save", "weights_save"),
        ("/api/weights/load", "weights_load"),
    ])
    async def test_a_token_refusal_is_audited_under_the_operations_one_name(
        self, env, path, operation
    ):
        """Addendum D-42: one audit name per operation, `weights_<op>`, token refusals included.
        Read off the LIVE app, so a route's default name (032's `save_checkpoint`) is caught."""
        with capture_logs() as logs:
            async with env.client() as client:
                response = await client.post(path, json={})
        assert response.status_code == 401, response.text
        assert [a["operation"] for a in _audits(logs)] == [operation]

    async def test_a_trainer_token_on_clear_stale_is_403(self, env):
        async with env.client() as client:
            response = await client.post("/api/weights/clear-stale", headers=_auth())
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "WEIGHTS_FORBIDDEN"

    @pytest.mark.parametrize("path", ["/api/weights/quiesce", "/api/weights/heartbeat",
                                      "/api/weights/detach", "/api/weights/commit-end"])
    async def test_a_missing_header_with_nothing_attached_is_409_not_422(self, env, path):
        async with env.client() as client:
            response = await client.post(path, headers=_auth())
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "NOT_ATTACHED"

    async def test_a_missing_header_while_attached_is_403_and_audited_with_its_code(self, env):
        async with env.client() as client:
            await _attach(client)
            with patch.object(env.barrier, "quiesce", wraps=env.barrier.quiesce) as spy:
                with capture_logs() as logs:
                    response = await client.post("/api/weights/quiesce", headers=_auth())
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "NOT_ATTACHMENT_HOLDER"
        assert spy.call_count == 0
        audit = _audits(logs)
        assert len(audit) == 1
        assert (audit[0]["operation"], audit[0]["outcome"], audit[0]["code"]) == (
            "weights_quiesce", "refused", "NOT_ATTACHMENT_HOLDER")
        assert audit[0]["attachment_ref"] is None

    async def test_a_second_attach_is_409_naming_the_holder(self, env):
        async with env.client() as client:
            attachment_id = await _attach(client)
            response = await client.post("/api/weights/attach", headers=_auth(OPERATOR_TOKEN))
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "ALREADY_ATTACHED"
        assert error["details"]["label"] == "miforge"
        assert attachment_id not in json.dumps(response.json())

    async def test_commit_begin_while_serving_is_not_quiesced(self, env):
        async with env.client() as client:
            attachment_id = await _attach(client)
            response = await client.post(
                "/api/weights/commit-begin", headers={**_auth(), HEADER: attachment_id}
            )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "NOT_QUIESCED"

    async def test_clear_stale_of_an_active_attachment_is_409(self, env):
        async with env.client() as client:
            await _attach(client)
            response = await client.post("/api/weights/clear-stale", headers=_auth(OPERATOR_TOKEN))
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ATTACHMENT_NOT_STALE"


# --- status (6.5) ------------------------------------------------------------------------------


class TestStatus:
    FIELDS = {
        "attached", "attachment", "barrier", "quiesced_at", "held_requests", "weights_state",
        "resident_model", "steps_since_load", "last_save", "commit", "commit_overran",
        "last_hold_timeout_at", "last_overrun", "pause_unavailable", "adapter",
        "cuda_memory", "exports_since_start",
    }

    async def test_takes_no_token_and_pins_the_field_set(self, env):
        async with env.client() as client:
            response = await client.get("/api/weights/status")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"success", "data", "error"} and body["success"] is True
        assert set(body["data"]) == self.FIELDS
        data = body["data"]
        assert data["attached"] is False and data["attachment"] is None
        assert data["weights_state"] == "clean" and data["barrier"] == "serving"
        assert data["commit"] is None and data["last_save"] is None

    async def test_never_carries_the_attachment_id_or_a_token(self, env):
        """M14 / FR-31.16.2."""
        async with env.client() as client:
            attachment_id = await _attach(client)
            await client.post("/api/weights/quiesce", headers={**_auth(), HEADER: attachment_id})
            body = (await client.get("/api/weights/status")).json()
            await client.post("/api/weights/resume", headers={**_auth(), HEADER: attachment_id})
        strings = list(_walk(body))
        digest = hashlib.sha256(attachment_id.encode()).hexdigest()
        assert "attachment_id" not in strings
        assert all(attachment_id not in s and digest not in s for s in strings)
        assert all(TRAINER_TOKEN not in s for s in strings)
        assert body["data"]["attachment"]["state"] == "active"
        assert body["data"]["attachment"]["label"] == "miforge"

    async def test_steps_since_load_is_null_with_nothing_resident(self, env):
        """null = not recorded, never 0, when nothing is resident."""
        env.fakes.state.current = None
        async with env.client() as client:
            data = (await client.get("/api/weights/status")).json()["data"]
        assert data["steps_since_load"] is None and data["resident_model"] is None

    async def test_steps_since_load_is_reported_with_a_model_resident(self, env):
        async with env.client() as client:
            data = (await client.get("/api/weights/status")).json()["data"]
        assert data["steps_since_load"] == 0
        assert data["resident_model"] == {"model_id": 7, "name": "tiny-model",
                                          "manifest_hash": None}


class TestAutomaticTransitionLogs:
    """FTASKS 6.6: a late commit-end logs `weights_commit_overran` with its fields."""

    async def test_overrun_line(self, env):
        async with env.client() as client:
            attachment_id = await _attach(client)
            holder = {**_auth(), HEADER: attachment_id}
            await client.post("/api/weights/quiesce", headers=holder)
            await client.post("/api/weights/commit-begin", headers=holder)
            env.barrier._watchdog_deadline_mono = 0.0
            with capture_logs() as logs:
                env.barrier._on_watchdog(env.barrier._watchdog_generation)
                await client.post("/api/weights/commit-end", headers=holder)
            await client.post("/api/weights/resume", headers=holder)
        torn = [e for e in logs if e["event"] == "weights_torn"]
        assert len(torn) == 1 and torn[0]["label"] == "miforge" and "elapsed_ms" in torn[0]
        over = [e for e in logs if e["event"] == "weights_commit_overran"]
        assert len(over) == 1
        assert over[0]["label"] == "miforge" and over[0]["role"] == "trainer"
        assert over[0]["watchdog_ms"] == int(settings.WEIGHTS_COMMIT_WATCHDOG_MS)
        assert over[0]["commit_ms"] >= 0
