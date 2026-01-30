"""
Integration tests for OpenAI text completions endpoint.

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


class TestCompletionsNoModel:
    """Tests when no model is loaded."""

    def test_returns_503_when_no_model(self, client):
        """POST /v1/completions returns 503 when no model is loaded."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5-turbo-instruct",
                "prompt": "Once upon a time",
            },
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "model_not_loaded"


class TestCompletionsValidation:
    """Tests for request validation."""

    def test_invalid_temperature(self, client):
        """Temperature out of range returns 422."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5-turbo-instruct",
                "prompt": "Test",
                "temperature": 3.0,
            },
        )
        assert response.status_code == 422

    def test_missing_prompt(self, client):
        """Request without prompt returns 422."""
        response = client.post(
            "/v1/completions",
            json={"model": "gpt-3.5-turbo-instruct"},
        )
        assert response.status_code == 422

    def test_missing_model(self, client):
        """Request without model returns 422."""
        response = client.post(
            "/v1/completions",
            json={"prompt": "Once upon a time"},
        )
        assert response.status_code == 422


class TestCompletionsWithValidParams:
    """Tests that valid parameters are accepted."""

    def test_accepts_string_prompt(self, client):
        """String prompt is accepted."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5-turbo-instruct",
                "prompt": "Once upon a time",
            },
        )
        # Should be 503 (no model) not 422 (validation)
        assert response.status_code == 503

    def test_accepts_list_prompt(self, client):
        """List prompt is accepted."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5-turbo-instruct",
                "prompt": ["Prompt 1", "Prompt 2"],
            },
        )
        assert response.status_code == 503

    def test_accepts_all_parameters(self, client):
        """All valid parameters are accepted."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5-turbo-instruct",
                "prompt": "Test",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": ["END"],
            },
        )
        assert response.status_code == 503
