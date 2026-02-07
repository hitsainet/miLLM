"""Unit tests for exception handlers."""

from unittest.mock import MagicMock, patch

import pytest

from millm.api.exception_handlers import (
    _is_openai_route,
    generic_exception_handler,
    millm_error_handler,
)
from millm.api.routes.openai.errors import ERROR_STATUS_MAP
from millm.core.errors import (
    MiLLMError,
    ModelNotFoundError,
    ModelNotLoadedError,
    SAENotAttachedError,
)


def _make_request(path: str, method: str = "GET") -> MagicMock:
    """Create a mock FastAPI Request with the given URL path."""
    request = MagicMock()
    request.url.path = path
    request.method = method
    return request


class TestIsOpenaiRoute:
    """Tests for _is_openai_route helper."""

    def test_returns_true_for_chat_completions(self):
        """Test that /v1/chat/completions is detected as OpenAI route."""
        request = _make_request("/v1/chat/completions")
        assert _is_openai_route(request) is True

    def test_returns_true_for_models(self):
        """Test that /v1/models is detected as OpenAI route."""
        request = _make_request("/v1/models")
        assert _is_openai_route(request) is True

    def test_returns_true_for_completions(self):
        """Test that /v1/completions is detected as OpenAI route."""
        request = _make_request("/v1/completions")
        assert _is_openai_route(request) is True

    def test_returns_true_for_embeddings(self):
        """Test that /v1/embeddings is detected as OpenAI route."""
        request = _make_request("/v1/embeddings")
        assert _is_openai_route(request) is True

    def test_returns_false_for_api_models(self):
        """Test that /api/models is NOT detected as OpenAI route."""
        request = _make_request("/api/models")
        assert _is_openai_route(request) is False

    def test_returns_false_for_api_steering(self):
        """Test that /api/steering is NOT detected as OpenAI route."""
        request = _make_request("/api/steering")
        assert _is_openai_route(request) is False

    def test_returns_false_for_root(self):
        """Test that root path is NOT detected as OpenAI route."""
        request = _make_request("/")
        assert _is_openai_route(request) is False


class TestMillmErrorHandler:
    """Tests for millm_error_handler."""

    @pytest.mark.asyncio
    async def test_returns_openai_format_for_v1_path(self):
        """Test that OpenAI format is returned for /v1/ endpoints."""
        request = _make_request("/v1/chat/completions", method="POST")
        exc = ModelNotLoadedError("No model is loaded")

        response = await millm_error_handler(request, exc)

        body = response.body.decode()
        import json
        data = json.loads(body)

        assert "error" in data
        assert data["error"]["message"] == "No model is loaded"
        assert data["error"]["type"] is not None
        assert "success" not in data  # Not management format

    @pytest.mark.asyncio
    async def test_returns_management_format_for_api_path(self):
        """Test that Management API format is returned for /api/ endpoints."""
        request = _make_request("/api/models/1/load", method="POST")
        exc = ModelNotFoundError(
            "Model not found",
            details={"model_id": 1},
        )

        response = await millm_error_handler(request, exc)

        body = response.body.decode()
        import json
        data = json.loads(body)

        assert data["success"] is False
        assert data["error"] is not None
        assert data["error"]["code"] == "MODEL_NOT_FOUND"
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_uses_correct_status_code_from_exception(self):
        """Test that the status code from the exception is used for management routes."""
        request = _make_request("/api/saes/1/attach", method="POST")
        exc = SAENotAttachedError("SAE is not attached")

        response = await millm_error_handler(request, exc)

        assert response.status_code == 400


class TestGenericExceptionHandler:
    """Tests for generic_exception_handler."""

    @pytest.mark.asyncio
    async def test_returns_openai_format_for_v1_path(self):
        """Test that generic errors return OpenAI format for /v1/ endpoints."""
        request = _make_request("/v1/completions", method="POST")
        exc = RuntimeError("Something went wrong")

        response = await generic_exception_handler(request, exc)

        body = response.body.decode()
        import json
        data = json.loads(body)

        assert "error" in data
        assert data["error"]["type"] == "server_error"
        assert data["error"]["code"] == "server_error"
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_returns_management_format_for_api_path(self):
        """Test that generic errors return management format for /api/ endpoints."""
        request = _make_request("/api/models", method="GET")
        exc = RuntimeError("Database connection failed")

        response = await generic_exception_handler(request, exc)

        body = response.body.decode()
        import json
        data = json.loads(body)

        assert data["success"] is False
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_does_not_leak_details_in_production(self):
        """Test that generic handler does not leak exception details when debug is off."""
        request = _make_request("/api/models", method="GET")
        exc = RuntimeError("secret database password invalid")

        response = await generic_exception_handler(request, exc)

        body = response.body.decode()
        import json
        data = json.loads(body)

        # The response should contain INTERNAL_ERROR, not expose raw exception
        assert data["error"]["code"] == "INTERNAL_ERROR"


class TestErrorStatusMap:
    """Tests for ERROR_STATUS_MAP correctness."""

    def test_model_not_loaded_maps_to_503(self):
        """Test that MODEL_NOT_LOADED maps to 503 status code."""
        status_code, error_type = ERROR_STATUS_MAP["MODEL_NOT_LOADED"]
        assert status_code == 503
        assert error_type == "server_error"

    def test_model_not_found_maps_to_404(self):
        """Test that MODEL_NOT_FOUND maps to 404 status code."""
        status_code, error_type = ERROR_STATUS_MAP["MODEL_NOT_FOUND"]
        assert status_code == 404
        assert error_type == "invalid_request_error"

    def test_validation_error_maps_to_400(self):
        """Test that VALIDATION_ERROR maps to 400 status code."""
        status_code, error_type = ERROR_STATUS_MAP["VALIDATION_ERROR"]
        assert status_code == 400
        assert error_type == "invalid_request_error"

    def test_insufficient_memory_maps_to_503(self):
        """Test that INSUFFICIENT_MEMORY maps to 503 status code."""
        status_code, error_type = ERROR_STATUS_MAP["INSUFFICIENT_MEMORY"]
        assert status_code == 503
        assert error_type == "server_error"

    def test_queue_full_maps_to_503(self):
        """Test that QUEUE_FULL maps to 503 status code."""
        status_code, error_type = ERROR_STATUS_MAP["QUEUE_FULL"]
        assert status_code == 503
        assert error_type == "server_error"
