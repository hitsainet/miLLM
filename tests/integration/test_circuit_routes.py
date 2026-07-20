"""Route tests for /api/circuits (Feature 13, tasks 5.3/5.4).

Envelope shapes, query params, the rung<2 activation gate surfaced as a
200+envelope refusal, slice-fallback disclosure in the activation response,
raw (non-enveloped) export, and the error paths.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from millm.api.dependencies import get_circuit_service
from millm.api.exception_handlers import millm_error_handler
from millm.api.routes.management.circuits import router
from millm.core.errors import (
    CircuitNotFoundError,
    MiLLMError,
    SAESetIncompleteError,
    UnvalidatedCircuitError,
)


def make_summary(**overrides) -> dict:
    base = dict(
        id="circ_1",
        name="fear→threat",
        description=None,
        rung=2,
        rung_language="causally validated (edge)",
        rung_next_step="run circuit-level faithfulness at promotion",
        validated=True,
        edge_count=1,
        layers=[10, 13],
        serveable=True,
        is_active=False,
        serving_mode=None,
        intensity=1.0,
        per_sae_warnings=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return base


def make_doc() -> dict:
    return {
        "kind": "mistudio.circuit-definition",
        "schema_version": "1",
        "name": "fear→threat",
        "saes": [{"layer": 10, "n_features": 8192}],
        "members": [{"layer": 10, "feature": {"feature_idx": 1, "strength": 10.0}}],
    }


@pytest.fixture
def mock_service():
    svc = MagicMock()
    svc.list_circuits = AsyncMock(return_value=[make_summary()])
    svc.repository = MagicMock()
    svc.repository.count = AsyncMock(return_value=1)
    svc.get_active = AsyncMock(return_value=None)
    svc.import_definition = AsyncMock(return_value=MagicMock())
    svc.summarize = MagicMock(return_value=make_summary())
    svc.activate = AsyncMock(
        return_value={
            **make_summary(is_active=True, serving_mode="full"),
            "bound_layers": [10, 13],
            "applied_per_layer": {10: {1: 40.0}, 13: {2: 30.0}},
            "hazards": [],
            "warnings": [],
            "acknowledged_unvalidated": False,
        }
    )
    svc.deactivate = AsyncMock(return_value=make_summary())
    svc.set_intensity = AsyncMock(return_value=make_summary(intensity=1.5))
    svc.delete = AsyncMock(return_value={"circuit_id": "circ_1", "deleted": True})
    svc.export_definition = AsyncMock(return_value=make_doc())
    return svc


@pytest.fixture
def client(mock_service):
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(MiLLMError, millm_error_handler)
    app.dependency_overrides[get_circuit_service] = lambda: mock_service
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestList:
    async def test_list_envelope(self, client):
        async with client:
            r = await client.get("/api/circuits")
        body = r.json()
        assert body["success"] is True
        assert body["data"]["circuits"][0]["id"] == "circ_1"
        assert body["data"]["total"] == 1
        assert body["data"]["active_circuit_id"] is None

    async def test_rung_language_is_server_rendered(self, client):
        """The client must never have to derive evidence phrasing."""
        async with client:
            r = await client.get("/api/circuits")
        row = r.json()["data"]["circuits"][0]
        assert row["rung_language"] == "causally validated (edge)"
        assert row["validated"] is True
        assert row["rung_next_step"]

    async def test_filters_passed_through(self, client, mock_service):
        async with client:
            await client.get("/api/circuits?min_rung=2&serveable=true&limit=10&offset=5")
        kwargs = mock_service.list_circuits.await_args.kwargs
        assert kwargs["min_rung"] == 2
        assert kwargs["serveable"] is True
        assert kwargs["limit"] == 10 and kwargs["offset"] == 5

    async def test_invalid_min_rung_rejected(self, client):
        async with client:
            r = await client.get("/api/circuits?min_rung=9")
        assert r.status_code == 422

    async def test_active_null_when_none_serving(self, client):
        async with client:
            r = await client.get("/api/circuits/active")
        assert r.json()["data"] is None


class TestImport:
    async def test_import_ok(self, client):
        async with client:
            r = await client.post("/api/circuits/import", json=make_doc())
        body = r.json()
        assert body["success"] is True
        assert body["data"]["name"] == "fear→threat"

    async def test_unknown_kind_refused_in_envelope(self, client):
        doc = make_doc()
        doc["kind"] = "mistudio.cluster-definition"
        async with client:
            r = await client.post("/api/circuits/import", json=doc)
        body = r.json()
        assert r.status_code == 200  # house style
        assert body["success"] is False
        assert body["error"]["code"] == "UNKNOWN_KIND"

    async def test_oversize_payload_refused(self, client):
        doc = make_doc()
        doc["narrative"] = "x" * 2_000_000
        async with client:
            r = await client.post("/api/circuits/import", json=doc)
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "PAYLOAD_TOO_LARGE"

    async def test_on_conflict_param_forwarded(self, client, mock_service):
        async with client:
            await client.post("/api/circuits/import?on_conflict=fail", json=make_doc())
        assert mock_service.import_definition.await_args.kwargs["on_conflict"] == "fail"

    async def test_bad_on_conflict_rejected(self, client):
        async with client:
            r = await client.post("/api/circuits/import?on_conflict=explode", json=make_doc())
        assert r.status_code == 422


class TestActivation:
    async def test_activate_full_serving(self, client):
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")
        data = r.json()["data"]
        assert data["serving_mode"] == "full"
        assert data["bound_layers"] == [10, 13]
        # int keys become strings for JSON
        assert data["applied_per_layer"]["10"]["1"] == 40.0

    async def test_unvalidated_refusal_is_200_envelope_with_rung(
        self, client, mock_service
    ):
        """A rung<2 circuit is refused in the envelope so the client can show
        the rung and re-send with the acknowledgement."""
        mock_service.activate.side_effect = UnvalidatedCircuitError(
            "Circuit 'x' is associated (rung 0), not causally validated.",
            details={"rung": 0, "rung_language": "associated"},
        )
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")
        body = r.json()
        assert r.status_code == 200
        assert body["success"] is False
        assert body["error"]["code"] == "UNVALIDATED_CIRCUIT"
        assert body["error"]["details"]["rung"] == 0
        assert body["error"]["details"]["rung_language"] == "associated"

    async def test_ack_param_forwarded(self, client, mock_service):
        async with client:
            await client.post(
                "/api/circuits/circ_1/activate?acknowledge_unvalidated=true"
            )
        assert mock_service.activate.await_args.kwargs["acknowledge_unvalidated"] is True

    async def test_slice_fallback_disclosed(self, client, mock_service):
        mock_service.activate.return_value = {
            **make_summary(is_active=True, serving_mode="slice_fallback"),
            "bound_layers": [10],
            "slice_layer": 10,
            "hazards": [],
            "warnings": ["Only L[10] of [10, 13] bound — serving the L10 slice"],
            "acknowledged_unvalidated": False,
        }
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")
        data = r.json()["data"]
        assert data["serving_mode"] == "slice_fallback"
        assert data["slice_layer"] == 10
        assert "PARTIAL" in data["warnings"][0] or "slice" in data["warnings"][0]

    async def test_sae_set_incomplete_maps_to_422(self, client, mock_service):
        mock_service.activate.side_effect = SAESetIncompleteError(
            [{"layer": 13, "sae_id": "sae-13", "reason": "unbound"}]
        )
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")
        assert r.status_code == 422
        body = r.json()
        assert body["error"]["code"] == "SAE_SET_INCOMPLETE"
        assert body["error"]["details"]["offenders"][0]["layer"] == 13

    async def test_unknown_circuit_404(self, client, mock_service):
        mock_service.activate.side_effect = CircuitNotFoundError("nope")
        async with client:
            r = await client.post("/api/circuits/ghost/activate")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "CIRCUIT_NOT_FOUND"


class TestIntensityAndLifecycle:
    async def test_set_active_intensity(self, client, mock_service):
        mock_service.get_active.return_value = make_summary(is_active=True)
        async with client:
            r = await client.put(
                "/api/circuits/active/intensity", json={"intensity": 1.5}
            )
        assert r.json()["data"]["intensity"] == 1.5

    async def test_no_active_circuit_refused_in_envelope(self, client):
        async with client:
            r = await client.put(
                "/api/circuits/active/intensity", json={"intensity": 1.5}
            )
        body = r.json()
        assert r.status_code == 200
        assert body["success"] is False
        assert body["error"]["code"] == "NO_ACTIVE_CIRCUIT"

    async def test_intensity_out_of_range_rejected(self, client):
        async with client:
            r = await client.put(
                "/api/circuits/active/intensity", json={"intensity": 9.0}
            )
        assert r.status_code == 422

    async def test_deactivate(self, client):
        async with client:
            r = await client.post("/api/circuits/circ_1/deactivate")
        assert r.json()["success"] is True

    async def test_delete(self, client):
        async with client:
            r = await client.delete("/api/circuits/circ_1")
        assert r.json()["data"]["deleted"] is True


class TestExport:
    async def test_export_is_raw_document_no_envelope(self, client):
        """The response IS the portable artifact — no {success,data} wrapper."""
        async with client:
            r = await client.get("/api/circuits/circ_1/export")
        body = r.json()
        assert body["kind"] == "mistudio.circuit-definition"
        assert "success" not in body and "data" not in body

    async def test_export_preserves_unknown_fields(self, client, mock_service):
        doc = make_doc()
        doc["future_field"] = {"kept": True}
        mock_service.export_definition.return_value = doc
        async with client:
            r = await client.get("/api/circuits/circ_1/export")
        assert r.json()["future_field"] == {"kept": True}
