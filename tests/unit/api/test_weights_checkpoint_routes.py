"""The 032 route surface, read from the LIVE app (032 FTASKS 11.1, 7.7-style reachability).

* The set: `app.openapi()["paths"]` restricted to the `weights-checkpoints` router is EXACTLY
  `{save, load, lineage}` (instantiate is deferred, so its absence is asserted too). Never
  `app.routes`: FastAPI wraps included routers (memory `app-routes-is-not-a-route-list`).
* Each handler calls its service ONCE with the parsed payload (a `wraps=` spy over the real
  method, so payload and call count are both asserted against code that really runs).
* The envelope `{success, data, error}` on success and on refusal; the token gate on save and
  load, and none on lineage.

Control M1: delete `app.include_router(weights_checkpoints_router)` -> the set test is red.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from millm.api.routes.management import weights_checkpoints as routes
from millm.core.config import settings
from millm.services.weights_checkpoint_service import LoadResult, SaveResult

EXPECTED = {"/api/weights/save", "/api/weights/load", "/api/weights/lineage"}
TOKEN = "trainer-token-for-tests"


def _paths(app) -> set[str]:
    return {
        path for path, ops in app.openapi()["paths"].items()
        if any("weights-checkpoints" in (op.get("tags") or []) for op in ops.values())
    }


def test_the_live_app_serves_exactly_the_three_routes():
    from millm.main import create_app

    paths = _paths(create_app())
    assert paths == EXPECTED
    assert "/api/weights/instantiate" not in paths  # FR-32.8 deferred


def test_the_guard_sees_a_registered_router():
    """Guard the guard: the tag filter finds routes when the router is included."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(routes.router)
    assert _paths(app) == EXPECTED


SAVED = SaveResult(name="org/m:run:step-5", path="run/step-000005", weights_sha256="a" * 64,
                   manifest_sha256="b" * 64, bytes=10, snapshot_ms=1.0, write_ms=2.0,
                   parent_check="match", latest="step-000005", identity_set=True)
LOADED = LoadResult(name="org/m:run:step-5", mode="in_place", weights_sha256="a" * 64,
                    manifest_sha256="b" * 64, verify_ms=1.0, write_ms=2.0,
                    weights_state="clean", steps_since_load=0)
LINEAGE = {"resident": {"name": "m", "origin": "model", "manifest_sha256": None,
                        "commits_since_identity": 0},
           "base_model": "org/m", "chain": [], "complete": True, "break": None,
           "weights_verified": False}


class FakeService:
    async def save(self, identity, attachment_id, fields):
        return SAVED

    async def load(self, identity, request, *, attachment_id=None, lease_id=None):
        return LOADED

    async def lineage(self):
        return LINEAGE


@pytest.fixture
def client(monkeypatch):
    from millm.main import create_app

    monkeypatch.setattr(settings, "WEIGHTS_TOKENS", json.dumps([{
        "label": "forge", "role": "trainer",
        "sha256": hashlib.sha256(TOKEN.encode()).hexdigest()}]))
    app = create_app()
    fake = FakeService()
    spies = {name: AsyncMock(wraps=getattr(fake, name)) for name in ("save", "load", "lineage")}
    for name, spy in spies.items():
        setattr(fake, name, spy)
    app.dependency_overrides[routes._service] = lambda: fake
    test_client = TestClient(app)  # no lifespan: the routes are under test, not startup
    test_client.spies = spies
    yield test_client


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_save_calls_the_service_once_with_the_parsed_payload(client):
    manifest = {"run_id": "run", "step": 5}
    response = client.post("/api/weights/save", json={"manifest": manifest},
                           headers={**AUTH, "X-miLLM-Attachment": "att-1"})
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True and body["data"]["name"] == SAVED.name
    assert body["data"]["identity_set"] is True
    spy = client.spies["save"]
    assert spy.await_count == 1
    identity, attachment_id, fields = spy.await_args.args
    assert (identity.label, identity.role, attachment_id, fields) == (
        "forge", "trainer", "att-1", manifest)


def test_load_calls_the_service_once_with_the_parsed_payload(client):
    response = client.post("/api/weights/load", json={"name": "org/m:run:step-5"},
                           headers={**AUTH, "X-miLLM-Attachment": "att-1",
                                    "X-miLLM-Lease": "lease-1"})
    assert response.status_code == 200
    assert response.json()["data"]["mode"] == "in_place"
    spy = client.spies["load"]
    assert spy.await_count == 1
    identity, request = spy.await_args.args
    assert identity.label == "forge"
    assert (request.name, request.path, request.model_id) == ("org/m:run:step-5", None, None)
    assert spy.await_args.kwargs == {"attachment_id": "att-1", "lease_id": "lease-1"}


def test_load_needs_exactly_one_target(client):
    response = client.post("/api/weights/load", json={"name": "a:b:step-1", "path": "b"},
                           headers=AUTH)
    assert response.status_code == 422
    assert client.spies["load"].await_count == 0


def test_lineage_is_ungated_and_keeps_the_break_key(client):
    response = client.get("/api/weights/lineage")
    assert response.status_code == 200
    data = response.json()["data"]
    assert "break" in data and data["break"] is None and data["weights_verified"] is False
    assert client.spies["lineage"].await_count == 1


@pytest.mark.parametrize("path", ["/api/weights/save", "/api/weights/load"])
def test_save_and_load_need_a_token(client, path):
    body = {"manifest": {}} if path.endswith("save") else {"name": "a:b:step-1"}
    response = client.post(path, json=body)
    assert response.status_code == 401
    assert response.json()["success"] is False
    assert response.json()["error"]["code"] == "WEIGHTS_UNAUTHORIZED"
    assert client.spies["save"].await_count == client.spies["load"].await_count == 0


def test_a_refusal_uses_the_management_envelope(client):
    from millm.core.errors import NotAttachedError

    client.spies["save"].side_effect = NotAttachedError("Nothing is attached.", {"label": "forge"})
    response = client.post("/api/weights/save", json={"manifest": {}}, headers=AUTH)
    assert response.status_code == 409
    body = response.json()
    assert body["success"] is False and body["data"] is None
    assert body["error"]["code"] == "NOT_ATTACHED"
    assert body["error"]["details"]["label"] == "forge"
