"""
Unit tests for OpenAI error format helpers.

Tests error creation, status code mapping, and exception handling.
"""

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from millm.api.routes.openai.errors import (
    ERROR_STATUS_MAP,
    context_length_exceeded_error,
    create_openai_error,
    model_not_found_error,
    model_not_loaded_error,
    openai_exception_handler,
    rate_limit_error,
    server_error,
    validation_error,
)
from millm.core.errors import (
    MiLLMError,
    ModelNotFoundError,
    ModelNotLoadedError,
)


class TestCreateOpenAIError:
    """Tests for create_openai_error helper."""

    def test_basic_error(self):
        response = create_openai_error(
            message="Test error",
            error_type="server_error",
            status_code=500,
        )
        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

    def test_error_with_code_and_param(self):
        response = create_openai_error(
            message="Invalid temperature",
            error_type="invalid_request_error",
            code="invalid_parameter",
            param="temperature",
            status_code=400,
        )
        assert response.status_code == 400
        # Access body content for verification
        body = response.body.decode()
        assert "invalid_parameter" in body
        assert "temperature" in body

    def test_error_format_matches_openai(self):
        response = create_openai_error(
            message="Test",
            error_type="server_error",
            status_code=500,
        )
        body = response.body.decode()
        # Should contain the error structure
        assert '"error"' in body
        assert '"message"' in body
        assert '"type"' in body


class TestErrorStatusMap:
    """Tests for error code to status mapping."""

    def test_model_not_loaded_mapping(self):
        status, error_type = ERROR_STATUS_MAP["MODEL_NOT_LOADED"]
        assert status == 503
        assert error_type == "server_error"

    def test_model_not_found_mapping(self):
        status, error_type = ERROR_STATUS_MAP["MODEL_NOT_FOUND"]
        assert status == 404
        assert error_type == "invalid_request_error"

    def test_validation_error_mapping(self):
        status, error_type = ERROR_STATUS_MAP["VALIDATION_ERROR"]
        assert status == 400
        assert error_type == "invalid_request_error"

    def test_rate_limit_mapping(self):
        status, error_type = ERROR_STATUS_MAP["RATE_LIMIT_EXCEEDED"]
        assert status == 429
        assert error_type == "rate_limit_error"

    def test_authentication_mapping(self):
        status, error_type = ERROR_STATUS_MAP["AUTHENTICATION_ERROR"]
        assert status == 401
        assert error_type == "authentication_error"

    def test_queue_full_mapping(self):
        status, error_type = ERROR_STATUS_MAP["QUEUE_FULL"]
        assert status == 503
        assert error_type == "server_error"


class TestErrorHelpers:
    """Tests for convenience error helper functions."""

    def test_model_not_loaded_error(self):
        response = model_not_loaded_error()
        assert response.status_code == 503
        body = response.body.decode()
        assert "model_not_loaded" in body
        assert "No model is currently loaded" in body

    def test_model_not_found_error_basic(self):
        response = model_not_found_error("gpt-5")
        assert response.status_code == 404
        body = response.body.decode()
        assert "model_not_found" in body
        assert "gpt-5" in body

    def test_model_not_found_error_with_available(self):
        response = model_not_found_error("gpt-5", available_model="gpt-4")
        assert response.status_code == 404
        body = response.body.decode()
        assert "gpt-5" in body
        assert "gpt-4" in body

    def test_validation_error_basic(self):
        response = validation_error("Invalid temperature value")
        assert response.status_code == 400
        body = response.body.decode()
        assert "Invalid temperature value" in body
        assert "invalid_request_error" in body

    def test_validation_error_with_param(self):
        response = validation_error("Must be between 0 and 2", param="temperature")
        assert response.status_code == 400
        body = response.body.decode()
        assert "temperature" in body

    def test_context_length_exceeded_error(self):
        response = context_length_exceeded_error(max_length=4096, requested_length=5000)
        assert response.status_code == 400
        body = response.body.decode()
        assert "4096" in body
        assert "5000" in body
        assert "context_length_exceeded" in body

    def test_rate_limit_error_default(self):
        response = rate_limit_error()
        assert response.status_code == 429
        body = response.body.decode()
        assert "Rate limit exceeded" in body

    def test_rate_limit_error_custom(self):
        response = rate_limit_error("Too many requests per minute")
        assert response.status_code == 429
        body = response.body.decode()
        assert "Too many requests per minute" in body

    def test_server_error_default(self):
        response = server_error()
        assert response.status_code == 500
        body = response.body.decode()
        assert "Internal server error" in body

    def test_server_error_custom(self):
        response = server_error("GPU memory exhausted")
        assert response.status_code == 500
        body = response.body.decode()
        assert "GPU memory exhausted" in body


class TestExceptionHandler:
    """Tests for openai_exception_handler."""

    @pytest.mark.asyncio
    async def test_handles_millm_error(self):
        # Create a mock request
        class MockRequest:
            pass

        request = MockRequest()
        # Use specific error class - code is a class attribute
        error = ModelNotLoadedError("No model loaded")

        response = await openai_exception_handler(request, error)
        assert isinstance(response, JSONResponse)
        assert response.status_code == 503
        body = response.body.decode()
        assert "No model loaded" in body

    @pytest.mark.asyncio
    async def test_handles_base_millm_error(self):
        class MockRequest:
            pass

        request = MockRequest()
        # Base MiLLMError has code="INTERNAL_ERROR"
        error = MiLLMError("Unknown error")

        response = await openai_exception_handler(request, error)
        # Base error defaults to 500
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_error_code_lowercase_in_response(self):
        class MockRequest:
            pass

        request = MockRequest()
        # Use specific error class
        error = ModelNotFoundError("Model not found")

        response = await openai_exception_handler(request, error)
        body = response.body.decode()
        # Code should be lowercase in response
        assert "model_not_found" in body
