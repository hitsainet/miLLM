"""
Integration tests for OpenAI chat completions endpoint.

Tests the full endpoint behavior including:
- No model loaded returns 503
- Invalid parameters return 422
- Valid requests return proper responses
"""

import pytest
from fastapi.testclient import TestClient

from millm.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestChatCompletionsNoModel:
    """Tests when no model is loaded."""

    def test_returns_503_when_no_model(self, client):
        """POST /v1/chat/completions returns 503 when no model is loaded."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "model_not_loaded"

    def test_streaming_returns_503_when_no_model(self, client):
        """POST /v1/chat/completions with stream=true returns 503 when no model."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert response.status_code == 503


class TestChatCompletionsValidation:
    """Tests for request validation."""

    def test_invalid_temperature_too_low(self, client):
        """Negative temperature returns 422."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": -1.0,
            },
        )
        assert response.status_code == 422

    def test_invalid_temperature_too_high(self, client):
        """Temperature > 2 returns 422."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 3.0,
            },
        )
        assert response.status_code == 422

    def test_missing_messages(self, client):
        """Request without messages returns 422."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        assert response.status_code == 422

    def test_missing_model(self, client):
        """Request without model returns 422."""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 422

    def test_invalid_role(self, client):
        """Invalid message role returns 422."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "invalid", "content": "Hello"}],
            },
        )
        assert response.status_code == 422

    def test_too_many_stop_sequences(self, client):
        """More than 4 stop sequences returns 422."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": ["1", "2", "3", "4", "5"],
            },
        )
        assert response.status_code == 422

    def test_invalid_top_p(self, client):
        """top_p > 1 returns 422."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 1.5,
            },
        )
        assert response.status_code == 422


class TestChatCompletionsWithValidParams:
    """Tests that valid parameters are accepted (but may return 503 without model)."""

    def test_accepts_all_valid_parameters(self, client):
        """All valid parameters are accepted."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100,
                "stop": ["END"],
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
            },
        )
        # Should be 503 (no model) not 422 (validation)
        assert response.status_code == 503

    def test_accepts_temperature_zero(self, client):
        """Temperature=0 (greedy decoding) is accepted."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.0,
            },
        )
        assert response.status_code == 503

    def test_accepts_stop_string(self, client):
        """Single stop sequence as string is accepted."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": "END",
            },
        )
        assert response.status_code == 503

    def test_accepts_stop_list(self, client):
        """Stop sequences as list is accepted."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": ["END", "STOP"],
            },
        )
        assert response.status_code == 503

    def test_extra_fields_ignored(self, client):
        """Unknown fields are ignored, not rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "logprobs": True,  # Not supported
                "n": 5,  # Not supported
                "unknown_field": "value",
            },
        )
        # Should be 503 (no model) not 422 (validation)
        assert response.status_code == 503


class TestChatCompletionsErrorFormat:
    """Tests that error responses match OpenAI format."""

    def test_error_has_correct_structure(self, client):
        """Error response has correct OpenAI structure."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        data = response.json()

        # Must have error object
        assert "error" in data
        error = data["error"]

        # Must have required fields
        assert "message" in error
        assert "type" in error

        # May have optional fields
        # param and code can be None or missing

    def test_503_returns_server_error_type(self, client):
        """503 status returns server_error type."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        data = response.json()
        assert data["error"]["type"] == "server_error"
