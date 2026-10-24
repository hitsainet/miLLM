"""
Feature 9 Task 1.2: active_profile on the detailed health response —
null when nothing is active, populated (id/name/source_kind/intensity/
sensing_enabled) when a profile is, and resilient to DB failures.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from millm.api.dependencies import get_inference_service, get_model_loader
from millm.api.routes.system.health import router


class _SessionCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *args):
        return False


def _client():
    app = FastAPI()
    app.include_router(router)
    loader = MagicMock(is_loaded=False)
    inference = MagicMock()
    inference.get_backend_info = MagicMock(return_value={})
    app.dependency_overrides[get_model_loader] = lambda: loader
    app.dependency_overrides[get_inference_service] = lambda: inference
    return TestClient(app)


def _async_return(value):
    async def _call(*args, **kwargs):
        return value

    return MagicMock(side_effect=_call)


def _get_detailed(active):
    client = _client()
    with patch("millm.db.base.async_session_factory",
               return_value=_SessionCtx()), \
         patch("millm.db.repositories.profile_repository.ProfileRepository") as MockRepo:
        MockRepo.return_value.get_active = _async_return(active)
        response = client.get("/api/health/detailed")
    assert response.status_code == 200
    return response.json()


class TestActiveProfileOnDetailedHealth:
    def test_null_when_no_active_profile(self):
        body = _get_detailed(None)
        assert body["active_profile"] is None

    def test_populated_for_active_cluster(self):
        active = MagicMock(
            id="prof_c1", source_kind="cluster", intensity=1.2,
            sensing_enabled=True,
        )
        active.name = "fear cluster"
        body = _get_detailed(active)
        assert body["active_profile"] == {
            "id": "prof_c1",
            "name": "fear cluster",
            "source_kind": "cluster",
            "intensity": 1.2,
            "sensing_enabled": True,
        }

    def test_manual_profile_defaults(self):
        active = MagicMock(id="prof_m1", source_kind=None, intensity=None,
                           sensing_enabled=False)
        active.name = "manual"
        body = _get_detailed(active)
        assert body["active_profile"]["source_kind"] == "manual"
        assert body["active_profile"]["intensity"] == 1.0

    def test_db_failure_degrades_to_null_not_500(self):
        client = _client()
        with patch("millm.db.base.async_session_factory",
                   side_effect=RuntimeError("db down")):
            response = client.get("/api/health/detailed")
        assert response.status_code == 200
        assert response.json()["active_profile"] is None

    def test_basic_health_untouched(self):
        """The gate's hot path must not gain a DB read."""
        client = _client()
        with patch("millm.db.base.async_session_factory") as factory:
            response = client.get("/api/health")
        assert response.status_code == 200
        factory.assert_not_called()
        assert "active_profile" not in response.json()
