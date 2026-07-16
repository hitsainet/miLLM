"""
Integration tests for OpenAI chat completions endpoint.

Tests the full endpoint behavior including:
- No model loaded returns 503
- Invalid parameters return 400
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
        """Negative temperature returns 400."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": -1.0,
            },
        )
        assert response.status_code == 400

    def test_invalid_temperature_too_high(self, client):
        """Temperature > 2 returns 400."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 3.0,
            },
        )
        assert response.status_code == 400

    def test_missing_messages(self, client):
        """Request without messages returns 400."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        assert response.status_code == 400

    def test_missing_model(self, client):
        """Request without model returns 400."""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 400

    def test_invalid_role(self, client):
        """Invalid message role returns 400."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "invalid", "content": "Hello"}],
            },
        )
        assert response.status_code == 400

    def test_too_many_stop_sequences(self, client):
        """More than 4 stop sequences returns 400."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": ["1", "2", "3", "4", "5"],
            },
        )
        assert response.status_code == 400

    def test_invalid_top_p(self, client):
        """top_p > 1 returns 400."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 1.5,
            },
        )
        assert response.status_code == 400


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


class TestSteeringIntensityDial:
    """Feature 10: the steering_intensity field at the HTTP boundary."""

    BODY = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    def test_invalid_dial_returns_openai_400(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={**self.BODY, "steering_intensity": 5.0},
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "steering_intensity" in (error.get("param") or "")
        assert "[0, 2]" in error["message"]

    def test_symbolic_garbage_returns_openai_400(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={**self.BODY, "steering_intensity": "loud"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    def test_valid_dial_reaches_model_gate(self, client):
        """A well-formed dial doesn't break routing — request proceeds to the
        normal no-model 503, not a validation failure."""
        for dial in ("off", "min", "max", 1.25):
            response = client.post(
                "/v1/chat/completions",
                json={**self.BODY, "steering_intensity": dial},
            )
            assert response.status_code == 503, dial

    def test_echo_header_non_streaming(self):
        """X-miLLM-Steering-Intensity echoes the effective lambda."""
        from unittest.mock import AsyncMock, MagicMock

        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.api.schemas.openai import (
            ChatCompletionChoice,
            ChatCompletionResponse,
            ChatMessage,
            Usage,
        )
        from millm.main import create_app

        app = create_app()
        inference = MagicMock()
        inference.get_loaded_model_info.return_value = MagicMock(name="gpt-4")
        inference.get_loaded_model_info.return_value.name = "gpt-4"
        inference.backend_name = "serial"
        inference.resolve_request_intensity = AsyncMock(return_value=1.4)
        inference.create_chat_completion = AsyncMock(
            return_value=ChatCompletionResponse(
                id="chatcmpl-1", created=1, model="gpt-4",
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hi"),
                    finish_reason="stop",
                )],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        )
        model_service = MagicMock()
        model_service.find_model_by_name = AsyncMock(return_value=MagicMock())
        app.dependency_overrides[get_inference_service] = lambda: inference
        app.dependency_overrides[get_model_service] = lambda: model_service

        with TestClient(app) as tc:
            response = tc.post(
                "/v1/chat/completions",
                json={**self.BODY, "steering_intensity": "max"},
            )
        assert response.status_code == 200
        assert response.headers["X-miLLM-Steering-Intensity"] == "1.4"

    def test_echo_header_streaming(self):
        """Streaming sends the echo header before the body."""
        from unittest.mock import AsyncMock, MagicMock

        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.main import create_app

        async def fake_stream(request):
            yield "data: [DONE]\n\n"

        app = create_app()
        inference = MagicMock()
        inference.get_loaded_model_info.return_value = MagicMock()
        inference.get_loaded_model_info.return_value.name = "gpt-4"
        inference.backend_name = "serial"
        inference.resolve_request_intensity = AsyncMock(return_value=0.0)
        inference.stream_chat_completion = fake_stream
        inference.request_queue = MagicMock(pending_count=0, max_pending=5)
        model_service = MagicMock()
        model_service.find_model_by_name = AsyncMock(return_value=MagicMock())
        app.dependency_overrides[get_inference_service] = lambda: inference
        app.dependency_overrides[get_model_service] = lambda: model_service

        with TestClient(app) as tc:
            response = tc.post(
                "/v1/chat/completions",
                json={**self.BODY, "steering_intensity": "off", "stream": True},
            )
        assert response.status_code == 200
        assert response.headers["X-miLLM-Steering-Intensity"] == "0"
        assert response.headers["X-miLLM-Backend"] == "serial"

    def test_no_dial_no_echo_header(self):
        """Without the field, the header is absent (and no resolution runs)."""
        from unittest.mock import AsyncMock, MagicMock

        from millm.api.dependencies import get_inference_service, get_model_service
        from millm.main import create_app

        app = create_app()
        inference = MagicMock()
        inference.get_loaded_model_info.return_value = None
        model_service = MagicMock()
        model_service.find_model_by_name = AsyncMock(return_value=MagicMock())
        app.dependency_overrides[get_inference_service] = lambda: inference
        app.dependency_overrides[get_model_service] = lambda: model_service

        with TestClient(app) as tc:
            response = tc.post("/v1/chat/completions", json=self.BODY)
        assert response.status_code == 503
        assert "X-miLLM-Steering-Intensity" not in response.headers
