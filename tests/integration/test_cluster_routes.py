"""
Route tests for /api/clusters (Feature 8, Task 4.3/4.4): envelope shapes,
kind discrimination, caps, hub params (repo_id with slash), intensity
routing (active vs {id} — declaration order), and error paths.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from millm.api.dependencies import get_cluster_hub_service, get_cluster_service
from millm.api.routes.management.clusters import router
from millm.api.schemas.cluster import (
    ClusterDefinitionV1,
    ClusterImportItem,
    ClusterImportResult,
    ClusterSummary,
    HubDefinitionRef,
    HubRepoInfo,
)


def make_summary(**overrides) -> ClusterSummary:
    base = dict(
        id="prof_c1",
        name="fear cluster",
        is_active=False,
        intensity=1.0,
        sensing_enabled=False,
        member_count=2,
        display_token="fear",
        bound=True,
        warnings=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return ClusterSummary(**base)


def make_definition_payload(name="fear cluster") -> dict:
    return {
        "kind": "mistudio.cluster-definition",
        "schema_version": "1",
        "name": name,
        "members": [{"feature_idx": 1, "strength": 1.0}],
    }


@pytest.fixture
def mock_service():
    svc = MagicMock()
    svc.list_clusters = AsyncMock(return_value=[make_summary()])
    svc.import_definition = AsyncMock(return_value=ClusterImportItem(
        name="fear cluster", status="imported", profile_id="prof_c1", warnings=[]
    ))
    svc.import_bundle = AsyncMock(return_value=ClusterImportResult(
        results=[], imported=2, blocked=0, errors=0
    ))
    svc.activate = AsyncMock(return_value={"profile_id": "prof_c1",
                                           "applied_steering": True, "feature_count": 2})
    svc.deactivate = AsyncMock(return_value={"profile_id": "prof_c1",
                                             "cleared_steering": True})
    svc.set_intensity = AsyncMock(return_value={"profile_id": "prof_c1",
                                                "intensity": 0.5, "reapplied": True})
    svc.export_definition = AsyncMock(return_value=make_definition_payload())
    return svc


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.search = AsyncMock(return_value=[HubRepoInfo(repo_id="org/pack", likes=1)])
    hub.list_definitions = AsyncMock(return_value=[HubDefinitionRef(file="a.cluster.json")])
    hub.fetch_definition = AsyncMock(return_value=(
        ClusterDefinitionV1.model_validate(make_definition_payload()),
        make_definition_payload(),   # raw payload (lossless storage)
        {"repo_id": "org/pack", "revision": "main", "path": "a.cluster.json"},
    ))
    return hub


@pytest.fixture
def client(mock_service, mock_hub):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_cluster_service] = lambda: mock_service
    app.dependency_overrides[get_cluster_hub_service] = lambda: mock_hub
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestListAndImport:
    async def test_list_envelope(self, client):
        async with client:
            r = await client.get("/api/clusters")
        body = r.json()
        assert body["success"] is True
        assert body["data"]["clusters"][0]["display_token"] == "fear"
        assert body["data"]["active_cluster_id"] is None

    async def test_import_definition_kind(self, client, mock_service):
        async with client:
            r = await client.post("/api/clusters/import",
                                  json=make_definition_payload())
        body = r.json()
        assert body["success"] is True
        assert body["data"]["imported"] == 1
        mock_service.import_definition.assert_awaited_once()

    async def test_import_bundle_kind(self, client, mock_service):
        payload = {"kind": "mistudio.cluster-bundle", "schema_version": "1",
                   "definitions": [make_definition_payload()]}
        async with client:
            r = await client.post("/api/clusters/import", json=payload)
        assert r.json()["success"] is True
        mock_service.import_bundle.assert_awaited_once()

    async def test_import_unknown_kind(self, client):
        async with client:
            r = await client.post("/api/clusters/import", json={"kind": "evil"})
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "UNKNOWN_KIND"

    async def test_import_size_cap(self, client):
        payload = {"kind": "mistudio.cluster-definition", "blob": "x" * 1_100_000}
        async with client:
            r = await client.post("/api/clusters/import", json=payload)
        assert r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

    async def test_import_contract_violation_reports_validation_error(self, client):
        payload = make_definition_payload()
        payload["members"] = []  # violates min_length=1
        async with client:
            r = await client.post("/api/clusters/import", json=payload)
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "VALIDATION_ERROR"

    async def test_import_on_conflict_param_validated(self, client):
        async with client:
            r = await client.post("/api/clusters/import?on_conflict=explode",
                                  json=make_definition_payload())
        assert r.status_code == 422


class TestHubRoutes:
    async def test_search_params(self, client, mock_hub):
        async with client:
            r = await client.get("/api/clusters/hub/search",
                                 params={"q": "fear", "base_model": "gemma", "limit": 10})
        assert r.json()["data"][0]["repo_id"] == "org/pack"
        mock_hub.search.assert_awaited_once_with(query="fear", base_model="gemma", limit=10)

    async def test_repo_id_with_slash(self, client, mock_hub):
        async with client:
            r = await client.get("/api/clusters/hub/org/pack/definitions")
        assert r.json()["data"][0]["file"] == "a.cluster.json"
        mock_hub.list_definitions.assert_awaited_once_with("org/pack", revision=None)

    async def test_hub_import(self, client, mock_hub, mock_service):
        async with client:
            r = await client.post("/api/clusters/hub/import",
                                  json={"repo_id": "org/pack",
                                        "filename": "a.cluster.json"})
        assert r.json()["success"] is True
        kwargs = mock_service.import_definition.await_args.kwargs
        assert kwargs["hub_ref"]["repo_id"] == "org/pack"


class TestIntensityRouting:
    async def test_active_intensity_no_active_cluster(self, client, mock_service):
        mock_service.list_clusters = AsyncMock(return_value=[make_summary(is_active=False)])
        async with client:
            r = await client.put("/api/clusters/active/intensity",
                                 json={"intensity": 0.5})
        assert r.json()["error"]["code"] == "NO_ACTIVE_CLUSTER"

    async def test_active_intensity_routes_to_active_row(self, client, mock_service):
        mock_service.list_clusters = AsyncMock(return_value=[
            make_summary(id="prof_on", is_active=True)
        ])
        async with client:
            r = await client.put("/api/clusters/active/intensity",
                                 json={"intensity": 1.5, "reapply": True})
        assert r.json()["success"] is True
        mock_service.set_intensity.assert_awaited_once_with("prof_on", 1.5, reapply=True)

    async def test_id_intensity_not_shadowed_by_active(self, client, mock_service):
        """'active' must match the literal route, other ids the parametrized one."""
        async with client:
            r = await client.put("/api/clusters/prof_c1/intensity",
                                 json={"intensity": 0.7})
        assert r.json()["success"] is True
        mock_service.set_intensity.assert_awaited_once_with("prof_c1", 0.7, reapply=True)

    async def test_intensity_bounds_422(self, client):
        async with client:
            r = await client.put("/api/clusters/prof_c1/intensity",
                                 json={"intensity": 2.5})
        assert r.status_code == 422


class TestExport:
    async def test_export_returns_raw_definition(self, client):
        async with client:
            r = await client.get("/api/clusters/prof_c1/export")
        body = r.json()
        # No envelope — the response IS the portable artifact
        assert body["kind"] == "mistudio.cluster-definition"
        assert "success" not in body
