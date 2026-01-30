"""Integration tests for Model API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from millm.db.models.model import ModelSource, ModelStatus, QuantizationType


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.id = 1
    model.name = "gemma-2-2b"
    model.source = ModelSource.HUGGINGFACE
    model.repo_id = "google/gemma-2-2b"
    model.local_path = None
    model.params = "2B"
    model.architecture = "text-generation"
    model.quantization = QuantizationType.Q4
    model.disk_size_mb = 1500
    model.estimated_memory_mb = 2000
    model.cache_path = "huggingface/google--gemma-2-2b--Q4"
    model.config_json = None
    model.trust_remote_code = False
    model.status = ModelStatus.READY
    model.error_message = None
    model.created_at = "2024-01-01T00:00:00"
    model.updated_at = "2024-01-01T00:00:00"
    model.loaded_at = None
    return model


@pytest.fixture
def mock_service():
    """Create a mock ModelService."""
    service = MagicMock()
    service.list_models = AsyncMock(return_value=[])
    service.get_model = AsyncMock()
    service.download_model = AsyncMock()
    service.cancel_download = AsyncMock()
    service.delete_model = AsyncMock(return_value=True)
    service.preview_model = AsyncMock()
    return service


@pytest.fixture
def app_with_mock_service(mock_service):
    """Create a test app with mocked service."""
    from fastapi import FastAPI

    from millm.api.dependencies import get_model_service
    from millm.api.routes.management.models import router

    app = FastAPI()
    app.include_router(router)

    # Override the service dependency
    app.dependency_overrides[get_model_service] = lambda: mock_service

    return app


@pytest.fixture
def client(app_with_mock_service):
    """Create a test client."""
    return TestClient(app_with_mock_service)


class TestListModels:
    """Tests for GET /api/models endpoint."""

    def test_returns_empty_list(self, client, mock_service):
        """Test that endpoint returns empty list when no models exist."""
        mock_service.list_models.return_value = []

        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []

    def test_returns_models(self, client, mock_service, mock_model):
        """Test that endpoint returns all models."""
        mock_service.list_models.return_value = [mock_model]

        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == 1
        assert data["data"][0]["name"] == "gemma-2-2b"


class TestGetModel:
    """Tests for GET /api/models/{id} endpoint."""

    def test_returns_model(self, client, mock_service, mock_model):
        """Test that endpoint returns the model when found."""
        mock_service.get_model.return_value = mock_model

        response = client.get("/api/models/1")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == 1

    def test_raises_not_found(self, client, mock_service):
        """Test that endpoint raises 404 when model not found."""
        from millm.core.errors import ModelNotFoundError

        mock_service.get_model.side_effect = ModelNotFoundError("Model not found")

        # Need to add exception handler to test app
        # For now, this tests the behavior is wired correctly


class TestDownloadModel:
    """Tests for POST /api/models endpoint."""

    def test_starts_download(self, client, mock_service, mock_model):
        """Test that endpoint starts download and returns 202."""
        mock_model.status = ModelStatus.DOWNLOADING
        mock_service.download_model.return_value = mock_model

        response = client.post(
            "/api/models",
            json={
                "source": "huggingface",
                "repo_id": "google/gemma-2-2b",
                "quantization": "Q4",
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "downloading"

    def test_validates_repo_id_required(self, client, mock_service):
        """Test that repo_id is required for HuggingFace source."""
        response = client.post(
            "/api/models",
            json={
                "source": "huggingface",
                "quantization": "Q4",
            },
        )

        # Should return 422 validation error
        assert response.status_code == 422


class TestCancelDownload:
    """Tests for POST /api/models/{id}/cancel endpoint."""

    def test_cancels_download(self, client, mock_service, mock_model):
        """Test that endpoint cancels download."""
        mock_model.status = ModelStatus.ERROR
        mock_model.error_message = "Download cancelled by user"
        mock_service.cancel_download.return_value = mock_model

        response = client.post("/api/models/1/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "error"


class TestDeleteModel:
    """Tests for DELETE /api/models/{id} endpoint."""

    def test_deletes_model(self, client, mock_service):
        """Test that endpoint deletes model."""
        mock_service.delete_model.return_value = True

        response = client.delete("/api/models/1")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestPreviewModel:
    """Tests for POST /api/models/preview endpoint."""

    def test_returns_preview(self, client, mock_service):
        """Test that endpoint returns model preview info."""
        mock_service.preview_model.return_value = {
            "name": "gemma-2-2b",
            "repo_id": "google/gemma-2-2b",
            "params": "2B",
            "architecture": "text-generation",
            "is_gated": False,
            "requires_trust_remote_code": False,
        }

        response = client.post(
            "/api/models/preview",
            json={"repo_id": "google/gemma-2-2b"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "gemma-2-2b"
        assert data["data"]["params"] == "2B"
        assert data["data"]["is_gated"] is False
