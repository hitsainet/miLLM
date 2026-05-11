"""
Integration tests for OpenAI models endpoint.

Tests the full endpoint behavior including:
- GET /v1/models returns empty list when no model loaded
- GET /v1/models/{id} returns 404 when model not found
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from millm.main import create_app


@pytest.fixture
def client():
    """Create test client with mocked ModelService (no real DB needed)."""
    from millm.api.dependencies import get_model_service

    mock_service = MagicMock()
    mock_service.get_locked_model = AsyncMock(return_value=None)
    mock_service.get_available_models = AsyncMock(return_value=[])
    mock_service.find_model_by_name = AsyncMock(return_value=None)

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: mock_service
    return TestClient(app)


class TestModelsListEndpoint:
    """Tests for GET /v1/models."""

    def test_returns_empty_list_when_no_model(self, client):
        """GET /v1/models returns empty list when no model is loaded."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_response_has_correct_structure(self, client):
        """Response has correct OpenAI structure."""
        response = client.get("/v1/models")
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert isinstance(data["data"], list)


class TestModelsGetEndpoint:
    """Tests for GET /v1/models/{model_id}."""

    def test_returns_404_for_unknown_model(self, client):
        """GET /v1/models/{id} returns 404 when model not found."""
        response = client.get("/v1/models/gpt-4")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_404_has_correct_error_format(self, client):
        """404 error has correct OpenAI format."""
        response = client.get("/v1/models/nonexistent-model")
        data = response.json()
        error = data["error"]
        assert "message" in error
        assert "type" in error
        assert error["type"] == "invalid_request_error"
        assert "nonexistent-model" in error["message"]

    def test_404_includes_code(self, client):
        """404 error includes model_not_found code."""
        response = client.get("/v1/models/gpt-5")
        data = response.json()
        assert data["error"].get("code") == "model_not_found"
