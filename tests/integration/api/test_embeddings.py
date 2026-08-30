"""
Integration tests for OpenAI embeddings endpoint.

Tests the full endpoint behavior including:
- No model loaded returns 503
- Invalid parameters return 400
"""

import pytest
from fastapi.testclient import TestClient

from millm.main import create_app



@pytest.fixture
def client():
    """Client whose ModelService reports the requested model as UNKNOWN.

    These tests post model "gpt-4", which this server does not have. Before
    on-demand loading they asserted 503 "no model loaded", because the endpoint
    rejected anything not already resident and never got as far as looking the
    name up properly — the mocked DB session returned a truthy stub for
    find_model_by_name, and nothing ever touched it.

    Now the endpoints LOAD what they are asked for, so an unknown name is a
    404 model_not_found and a known one would be loaded. Overriding the service
    explicitly makes that distinction real instead of an artifact of how deeply
    the session mock happened to be inspected.
    """
    from unittest.mock import AsyncMock, MagicMock

    from millm.api.dependencies import get_model_service

    svc = MagicMock()
    svc.find_model_by_name = AsyncMock(return_value=None)
    svc.get_locked_model = AsyncMock(return_value=None)
    svc.get_available_models = AsyncMock(return_value=[])
    svc.load_model_and_wait = AsyncMock()

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    return TestClient(app)


class TestEmbeddingsNoModel:
    """Tests when no model is loaded."""

    def test_returns_404_for_an_unknown_model(self, client):
        """POST /v1/embeddings returns 503 when no model is loaded."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": "Hello world",
            },
        )
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "model_not_found"


class TestEmbeddingsValidation:
    """Tests for request validation."""

    def test_missing_input(self, client):
        """Request without input returns 400."""
        response = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-ada-002"},
        )
        assert response.status_code == 400

    def test_missing_model(self, client):
        """Request without model returns 400."""
        response = client.post(
            "/v1/embeddings",
            json={"input": "Hello world"},
        )
        assert response.status_code == 400


class TestEmbeddingsWithValidParams:
    """Tests that valid parameters are accepted."""

    def test_accepts_string_input(self, client):
        """String input is accepted."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": "Hello world",
            },
        )
        # Should be 503 (no model) not 422 (validation)
        assert response.status_code == 404

    def test_accepts_list_input(self, client):
        """List input is accepted."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": ["Hello", "World"],
            },
        )
        assert response.status_code == 404

    def test_extra_fields_ignored(self, client):
        """Unknown fields are ignored."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": "Hello",
                "encoding_format": "float",  # Not supported
                "dimensions": 512,  # Not supported
            },
        )
        assert response.status_code == 404
