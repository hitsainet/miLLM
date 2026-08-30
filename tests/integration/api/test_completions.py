"""
Integration tests for OpenAI text completions endpoint.

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


class TestCompletionsNoModel:
    """Tests when no model is loaded."""

    def test_returns_404_for_an_unknown_model(self, client):
        """POST /v1/completions returns 503 when no model is loaded."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5-turbo-instruct",
                "prompt": "Once upon a time",
            },
        )
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "model_not_found"


class TestCompletionsValidation:
    """Tests for request validation."""

    def test_invalid_temperature(self, client):
        """Temperature out of range returns 400."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5-turbo-instruct",
                "prompt": "Test",
                "temperature": 3.0,
            },
        )
        assert response.status_code == 400

    def test_missing_prompt(self, client):
        """Request without prompt returns 400."""
        response = client.post(
            "/v1/completions",
            json={"model": "gpt-3.5-turbo-instruct"},
        )
        assert response.status_code == 400

    def test_missing_model(self, client):
        """Request without model returns 400."""
        response = client.post(
            "/v1/completions",
            json={"prompt": "Once upon a time"},
        )
        assert response.status_code == 400


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
        assert response.status_code == 404

    def test_accepts_list_prompt(self, client):
        """List prompt is accepted."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5-turbo-instruct",
                "prompt": ["Prompt 1", "Prompt 2"],
            },
        )
        assert response.status_code == 404

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
        assert response.status_code == 404
