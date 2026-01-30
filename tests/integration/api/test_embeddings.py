"""
Integration tests for OpenAI embeddings endpoint.

Tests the full endpoint behavior including:
- No model loaded returns 503
- Invalid parameters return 422
"""

import pytest
from fastapi.testclient import TestClient

from millm.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestEmbeddingsNoModel:
    """Tests when no model is loaded."""

    def test_returns_503_when_no_model(self, client):
        """POST /v1/embeddings returns 503 when no model is loaded."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": "Hello world",
            },
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "model_not_loaded"


class TestEmbeddingsValidation:
    """Tests for request validation."""

    def test_missing_input(self, client):
        """Request without input returns 422."""
        response = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-ada-002"},
        )
        assert response.status_code == 422

    def test_missing_model(self, client):
        """Request without model returns 422."""
        response = client.post(
            "/v1/embeddings",
            json={"input": "Hello world"},
        )
        assert response.status_code == 422


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
        assert response.status_code == 503

    def test_accepts_list_input(self, client):
        """List input is accepted."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": ["Hello", "World"],
            },
        )
        assert response.status_code == 503

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
        assert response.status_code == 503
