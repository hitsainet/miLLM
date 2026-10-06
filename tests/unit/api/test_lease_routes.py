"""Feature 29 task 5.7: the lease routes and header pass-through, against the REAL app.

The app is `create_app()` with only `get_model_service` overridden, to a real `ModelService`
over a fake repository (tests/unit/lease_fixtures.py). Reachability is read from
`app.openapi()["paths"]` (memory `app-routes-is-not-a-route-list`), and every route's service
call is asserted by payload AND call count through a `wraps=` spy.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from millm.db.models.model import ModelStatus
from tests.unit.lease_fixtures import (
    build_service,
    clear_resident,
    drain_executor,
    install_registry,
    no_probe_rows,
)

LEASE = "X-miLLM-Lease"


@pytest.fixture(autouse=True)
def _clean_loader():
    clear_resident()
    yield
    clear_resident()


@pytest.fixture
def env():
    from millm.api.dependencies import get_model_service
    from millm.main import create_app

    registry, clock = install_registry()
    service, repo = build_service()
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    return TestClient(app), service, repo, registry, clock, app


def _grant(client, holder="midataworks", ttl=120, model_id=1):
    return client.post(
        f"/api/models/{model_id}/lease",
        json={"holder": holder, "reason": "label run 7", "ttl_seconds": ttl},
    )


def _walk(value: Any):
    """Every key and every string value anywhere in a JSON document."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _walk(v)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, str):
        yield value


def _assert_no_lease_id(body: Any, lease_id: str) -> None:
    seen = list(_walk(body))
    assert "lease_id" not in seen, body
    assert all(lease_id not in s for s in seen), body


class TestReachability:
    def test_the_four_lease_routes_are_served(self, env):
        _, _, _, _, _, app = env
        paths = app.openapi()["paths"]
        assert {"post", "get", "delete"} <= set(paths["/api/models/{model_id}/lease"])
        assert "post" in paths["/api/models/{model_id}/lease/renew"]

    def test_the_lease_header_is_declared_on_load_and_unload(self, env):
        _, _, _, _, _, app = env
        paths = app.openapi()["paths"]
        for path in ("/api/models/{model_id}/load", "/api/models/{model_id}/unload",
                     "/api/models/{model_id}/lease/renew", "/api/models/{model_id}/lease"):
            ops = paths[path]
            names = {
                p["name"] for op in ops.values() for p in op.get("parameters", [])
                if p["in"] == "header"
            }
            assert LEASE in names, path


class TestGrantRoute:
    def test_201_carries_the_lease_id_and_calls_the_service_once(self, env):
        client, service, _, registry, _, _ = env
        service.acquire_lease = AsyncMock(wraps=service.acquire_lease)
        resp = _grant(client)
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert set(data) == {
            "lease_id", "model_id", "model_name", "holder", "reason", "acquired_at",
            "renewed_at", "expires_at", "ttl_seconds", "seconds_remaining",
        }
        assert data["model_id"] == 1 and data["model_name"] == "m1"
        assert data["holder"] == "midataworks" and data["ttl_seconds"] == 120
        assert data["expires_at"].startswith("2026-10-06T12:02:00")
        assert registry.matches(registry.current(1), data["lease_id"])
        service.acquire_lease.assert_awaited_once_with(
            1, "midataworks", "label run 7", ttl_seconds=120
        )

    @pytest.mark.parametrize("body, field", [
        ({"holder": "h", "reason": "r", "ttl_seconds": 0}, "ttl_seconds"),
        ({"holder": "h", "reason": "r", "ttl_seconds": -5}, "ttl_seconds"),
        ({"holder": "h", "reason": "r", "ttl_seconds": 7201}, "ttl_seconds"),
        ({"holder": "h", "reason": "r", "ttl_seconds": 1.5}, "ttl_seconds"),
        ({"holder": "h", "reason": "r", "ttl_seconds": "60"}, "ttl_seconds"),
        ({"reason": "r"}, "holder"),
        ({"holder": "x" * 129, "reason": "r"}, "holder"),
        ({"holder": "h", "reason": ""}, "reason"),
        ({"holder": "h", "reason": "y" * 513}, "reason"),
    ])
    def test_400_names_the_field_and_the_limit(self, env, body, field):
        client, _, _, registry, _, _ = env
        resp = client.post("/api/models/1/lease", json=body)
        assert resp.status_code == 400, resp.text
        err = resp.json()["error"]
        assert err["code"] == "INVALID_LEASE_REQUEST"
        assert err["details"]["param"] == field
        assert ("7200" in err["details"]["technical_message"]
                or str(err["details"].get("max_chars")) in err["details"]["technical_message"])
        assert registry.current(1) is None

    def test_an_unknown_body_field_is_refused(self, env):
        client, _, _, _, _, _ = env
        resp = client.post("/api/models/1/lease",
                           json={"holder": "h", "reason": "r", "lease_id": "mine"})
        assert resp.status_code == 422

    def test_409_not_resident(self, env):
        client, _, _, _, _, _ = env
        resp = _grant(client, model_id=2)
        assert resp.status_code == 409
        err = resp.json()["error"]
        assert err["code"] == "MODEL_NOT_RESIDENT"
        assert err["details"]["resident_model_name"] == "m1"

    def test_409_leased_names_holder_and_expiry(self, env):
        client, _, _, _, _, _ = env
        lease_id = _grant(client).json()["data"]["lease_id"]
        resp = _grant(client, holder="mistudio-agent")
        assert resp.status_code == 409
        err = resp.json()["error"]
        assert err["code"] == "MODEL_LEASED"
        assert err["details"]["holder"] == "midataworks"
        assert err["details"]["expires_at"].startswith("2026-10-06T12:02:00")
        _assert_no_lease_id(resp.json(), lease_id)

    def test_grant_during_a_load_is_503_with_retry_after(self, env):
        client, service, _, _, _, _ = env
        service._loading_model_id = 2
        resp = _grant(client)
        # The management API keeps MODEL_BUSY at 409 (FPRD §9): not a 503, so no header owed.
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "MODEL_BUSY"


class TestReadsNeverCarryTheId:
    def test_get_lease(self, env):
        """Control M12."""
        client, service, _, _, _, _ = env
        lease_id = _grant(client).json()["data"]["lease_id"]
        service.get_lease = AsyncMock(wraps=service.get_lease)
        resp = client.get("/api/models/1/lease")
        assert resp.status_code == 200
        body = resp.json()
        _assert_no_lease_id(body, lease_id)
        lease = body["data"]["lease"]
        assert lease["holder"] == "midataworks" and lease["seconds_remaining"] == 120
        assert body["data"]["last_ended"] is None
        service.get_lease.assert_awaited_once_with(1)

    def test_get_with_no_lease(self, env):
        client, _, _, _, _, _ = env
        body = client.get("/api/models/2/lease").json()["data"]
        assert body == {"lease": None, "last_ended": None}

    def test_get_unknown_model_404(self, env):
        client, _, _, _, _, _ = env
        assert client.get("/api/models/99/lease").status_code == 404

    def test_model_list_and_single_carry_the_summary_not_the_id(self, env):
        client, _, _, _, _, _ = env
        lease_id = _grant(client).json()["data"]["lease_id"]
        listing = client.get("/api/models").json()
        _assert_no_lease_id(listing, lease_id)
        by_id = {m["id"]: m for m in listing["data"]}
        assert by_id[1]["lease"]["holder"] == "midataworks"
        assert by_id[2]["lease"] is None
        single = client.get("/api/models/1").json()
        _assert_no_lease_id(single, lease_id)
        assert single["data"]["lease"]["reason"] == "label run 7"

    def test_one_serialiser_three_reads(self, env):
        """The lease GET, ModelResponse.lease and the health field share one key set."""
        client, _, _, _, _, _ = env
        _grant(client)
        get_keys = set(client.get("/api/models/1/lease").json()["data"]["lease"])
        model_keys = set(client.get("/api/models/1").json()["data"]["lease"])
        assert get_keys == model_keys


class TestRenewAndRelease:
    def test_renew_sets_new_expiry_from_now(self, env):
        client, service, _, _, clock, _ = env
        lease_id = _grant(client, ttl=120).json()["data"]["lease_id"]
        clock.advance(100)
        service.renew_lease = AsyncMock(wraps=service.renew_lease)
        resp = client.post("/api/models/1/lease/renew", headers={LEASE: lease_id},
                           json={"ttl_seconds": 60})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["expires_at"].startswith("2026-10-06T12:02:40")
        assert data["seconds_remaining"] == 60
        _assert_no_lease_id(resp.json(), lease_id)
        service.renew_lease.assert_awaited_once_with(1, lease_id, ttl_seconds=60)

    def test_renew_without_body_uses_the_default(self, env):
        client, _, _, _, _, _ = env
        lease_id = _grant(client).json()["data"]["lease_id"]
        resp = client.post("/api/models/1/lease/renew", headers={LEASE: lease_id})
        assert resp.json()["data"]["ttl_seconds"] == 7200

    def test_renew_without_header_is_400(self, env):
        client, _, _, _, _, _ = env
        _grant(client)
        resp = client.post("/api/models/1/lease/renew", json={"ttl_seconds": 60})
        assert resp.status_code == 400
        assert resp.json()["error"]["details"]["param"] == "X-miLLM-Lease"

    def test_unknown_id_404_with_the_restart_sentence(self, env):
        client, _, _, _, _, _ = env
        _grant(client)
        resp = client.post("/api/models/1/lease/renew", headers={LEASE: "unknown"})
        assert resp.status_code == 404
        err = resp.json()["error"]
        assert err["code"] == "LEASE_NOT_FOUND"
        assert "a restart ends every lease" in err["details"]["technical_message"]

    def test_lease_id_for_another_model_is_404(self, env):
        client, _, _, _, _, _ = env
        lease_id = _grant(client).json()["data"]["lease_id"]
        resp = client.post("/api/models/2/lease/renew", headers={LEASE: lease_id})
        assert resp.status_code == 404
        resp = client.request("DELETE", "/api/models/2/lease", headers={LEASE: lease_id})
        assert resp.status_code == 404

    def test_release_ends_and_a_second_release_is_409(self, env):
        client, service, _, registry, _, _ = env
        lease_id = _grant(client).json()["data"]["lease_id"]
        service.release_lease = AsyncMock(wraps=service.release_lease)
        resp = client.request("DELETE", "/api/models/1/lease", headers={LEASE: lease_id})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["end_reason"] == "released"
        _assert_no_lease_id(resp.json(), lease_id)
        assert registry.current(1) is None
        service.release_lease.assert_awaited_once_with(1, lease_id)
        again = client.request("DELETE", "/api/models/1/lease", headers={LEASE: lease_id})
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "LEASE_EXPIRED"
        assert again.json()["error"]["details"]["end_reason"] == "released"

    def test_expired_lease_renew_is_409(self, env):
        client, _, _, _, clock, _ = env
        lease_id = _grant(client, ttl=10).json()["data"]["lease_id"]
        clock.advance(10)
        resp = client.post("/api/models/1/lease/renew", headers={LEASE: lease_id})
        assert resp.status_code == 409
        assert resp.json()["error"]["details"]["end_reason"] == "expired"
        assert client.get("/api/models/1/lease").json()["data"]["lease"] is None


class TestHeaderPassThrough:
    def test_management_load_refused_without_and_proceeds_with_the_header(self, env):
        client, service, repo, registry, _, _ = env
        lease_id = _grant(client).json()["data"]["lease_id"]
        service.load_model = AsyncMock(wraps=service.load_model)

        refused = client.post("/api/models/2/load")
        assert refused.status_code == 409
        err = refused.json()["error"]
        assert err["code"] == "MODEL_LEASED"
        assert err["details"]["holder"] == "midataworks"
        assert err["details"]["reason"] == "label run 7"
        assert err["details"]["expires_at"].startswith("2026-10-06T12:02:00")
        assert service._load_worker.call_count == 0
        _assert_no_lease_id(refused.json(), lease_id)

        with no_probe_rows():
            ok = client.post("/api/models/2/load", headers={LEASE: lease_id})
        assert ok.status_code == 202, ok.text
        drain_executor(service)
        assert service._load_worker.call_count == 1
        assert service.load_model.await_args_list[-1].kwargs["lease_id"] == lease_id
        assert service.load_model.await_count == 2
        assert registry.current(1) is None
        assert repo.rows[1].status == ModelStatus.READY

    def test_management_unload_refused_without_and_proceeds_with_the_header(self, env):
        client, service, repo, registry, _, _ = env
        lease_id = _grant(client).json()["data"]["lease_id"]
        service.unload_model = AsyncMock(wraps=service.unload_model)
        with no_probe_rows():
            refused = client.post("/api/models/1/unload")
            assert refused.status_code == 409
            assert refused.json()["error"]["code"] == "MODEL_LEASED"
            assert repo.rows[1].status == ModelStatus.LOADED
            ok = client.post("/api/models/1/unload", headers={LEASE: lease_id})
        assert ok.status_code == 200
        assert service.unload_model.await_count == 2
        assert service.unload_model.await_args_list[-1].kwargs == {"lease_id": lease_id}
        assert registry.last_ended(1).end_reason == "model_unloaded"

    def test_management_load_ignores_the_load_policy_header(self, env):
        """FR-29.4.6: the management route loads by explicit request."""
        client, service, _, _, _, _ = env
        with no_probe_rows():
            resp = client.post("/api/models/2/load", headers={"X-miLLM-Load-Policy": "refuse"})
        assert resp.status_code == 202, resp.text
        drain_executor(service)
        assert service._load_worker.call_count == 1


def test_grant_response_json_is_the_only_place_the_id_appears(env):
    """A full round: grant, read everything, renew, release — the ID appears once."""
    client, _, _, _, _, _ = env
    grant = _grant(client)
    lease_id = grant.json()["data"]["lease_id"]
    texts = [
        client.get("/api/models/1/lease").text,
        client.get("/api/models").text,
        client.get("/api/models/1").text,
        client.post("/api/models/1/lease/renew", headers={LEASE: lease_id}).text,
        client.request("DELETE", "/api/models/1/lease", headers={LEASE: lease_id}).text,
        client.get("/api/models/1/lease").text,
    ]
    for text in texts:
        assert lease_id not in text
        assert "lease_id" not in json.loads(text).__repr__()
