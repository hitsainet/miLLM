"""Route tests for the multi-SAE attach/serve surface (Feature 12, task 4.3).

Covers the plural attachment status shape, attach-set idempotency + VRAM
warning, and that SAESetIncompleteError surfaces as a 422 envelope through the
registered MiLLMError handler.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from millm.api.exception_handlers import millm_error_handler
from millm.api.dependencies import get_sae_service
from millm.api.routes.management.saes import router
from millm.core.errors import MiLLMError, SAESetIncompleteError
from millm.services.sae_service import (
    AttachedEntryStatus,
    AttachmentStatusSet,
)


@pytest.fixture
def mock_service():
    svc = MagicMock()
    svc.get_attachment_status_set = MagicMock(
        return_value=AttachmentStatusSet(
            is_attached=True,
            count=2,
            entries=[
                AttachedEntryStatus(sae_id="sae-a", layer=10, memory_usage_mb=64),
                AttachedEntryStatus(sae_id="sae-b", layer=13, memory_usage_mb=64),
            ],
            total_memory_usage_mb=128,
        )
    )
    svc.attach_set = AsyncMock(
        return_value={
            "status": "attached",
            "entries": [
                {"sae_id": "sae-a", "layer": 10, "status": "attached", "memory_usage_mb": 64},
                {"sae_id": "sae-b", "layer": 13, "status": "attached", "memory_usage_mb": 64},
            ],
            "attached_count": 2,
            "total_memory_usage_mb": 128,
            "vram_envelope_mb": 200,
            "vram_warning": False,
        }
    )
    return svc


@pytest.fixture
def client(mock_service):
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(MiLLMError, millm_error_handler)
    app.dependency_overrides[get_sae_service] = lambda: mock_service
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestAttachmentsStatus:
    async def test_plural_status_shape(self, client):
        async with client:
            r = await client.get("/api/saes/attachments")
        body = r.json()
        assert body["success"] is True
        data = body["data"]
        assert data["is_attached"] is True and data["count"] == 2
        assert {e["sae_id"] for e in data["entries"]} == {"sae-a", "sae-b"}
        assert data["total_memory_usage_mb"] == 128
        assert data["vram_envelope_mb"] == 200
        assert data["vram_warning"] is False

    async def test_vram_warning_flag_when_over_envelope(self, client, mock_service):
        mock_service.get_attachment_status_set.return_value = AttachmentStatusSet(
            is_attached=True,
            count=4,
            entries=[
                AttachedEntryStatus(sae_id=f"s{i}", layer=i, memory_usage_mb=64)
                for i in range(4)
            ],
            total_memory_usage_mb=256,
        )
        async with client:
            r = await client.get("/api/saes/attachments")
        data = r.json()["data"]
        assert data["total_memory_usage_mb"] == 256
        assert data["vram_warning"] is True  # 256 > 200 envelope


class TestAttachSet:
    async def test_attach_set_envelope(self, client):
        async with client:
            r = await client.post(
                "/api/saes/attach-set",
                json={"saes": [
                    {"sae_id": "sae-a", "layer": 10},
                    {"sae_id": "sae-b", "layer": 13},
                ]},
            )
        body = r.json()
        assert body["success"] is True
        assert body["data"]["attached_count"] == 2
        assert body["data"]["total_memory_usage_mb"] == 128
        assert body["data"]["vram_warning"] is False

    async def test_attach_set_idempotent_status(self, client, mock_service):
        mock_service.attach_set.return_value = {
            "status": "attached",
            "entries": [{"sae_id": "sae-a", "layer": 10, "status": "already_attached"}],
            "attached_count": 1,
            "total_memory_usage_mb": 64,
            "vram_envelope_mb": 200,
            "vram_warning": False,
        }
        async with client:
            r = await client.post(
                "/api/saes/attach-set",
                json={"saes": [{"sae_id": "sae-a", "layer": 10}]},
            )
        assert r.json()["data"]["entries"][0]["status"] == "already_attached"

    async def test_empty_set_rejected_by_validation(self, client):
        async with client:
            r = await client.post("/api/saes/attach-set", json={"saes": []})
        assert r.status_code == 422  # pydantic min_length=1

    async def test_sae_set_incomplete_maps_to_422_envelope(self, client, mock_service):
        mock_service.attach_set.side_effect = SAESetIncompleteError(
            [{"feature_idx": 2, "layer": 13, "sae_id": "sae-b"}]
        )
        async with client:
            r = await client.post(
                "/api/saes/attach-set",
                json={"saes": [{"sae_id": "sae-b", "layer": 13}]},
            )
        assert r.status_code == 422
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "SAE_SET_INCOMPLETE"
        assert body["error"]["details"]["offenders"][0]["layer"] == 13
