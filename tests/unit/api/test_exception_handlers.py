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
    ModelBusyError,
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


class TestTheLoggedStatusIsTheOneSent:
    """Hardware acceptance re-run, 2026-09-14: a /v1 request refused during an
    unload got 503 while the `api_error` log line said 409, because the handler
    logged the exception's management-API status before mapping it.

    MUTATION CONTROL (restored and sha256-verified; re-run 2026-09-15 against this final file):
      LOG-M1 the log line's status_code is exc.status_code again
             -> test_an_openai_route_logs_the_mapped_status (both parameters)

    This file was excluded from CI from 2026-04-09 (stale tests at the time). It passes and is
    run by CI again from 2026-09-15.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error", "sent"),
        [
            (lambda: ModelBusyError("unloading"), 503),
            (lambda: ModelNotLoadedError("No model is loaded"), 503),
        ],
        ids=["model_busy", "model_not_loaded"],
    )
    async def test_an_openai_route_logs_the_mapped_status(self, error, sent):
        with patch("millm.api.exception_handlers.logger") as logger:
            response = await millm_error_handler(_make_request("/v1/chat/completions", "POST"), error())
        assert response.status_code == sent
        logger.warning.assert_called_once()
        assert logger.warning.call_args.kwargs["status_code"] == sent

    @pytest.mark.asyncio
    async def test_a_management_route_logs_its_own_status(self):
        with patch("millm.api.exception_handlers.logger") as logger:
            response = await millm_error_handler(_make_request("/api/models/3/load", "POST"), ModelBusyError("busy"))
        assert response.status_code == 409
        logger.warning.assert_called_once()
        assert logger.warning.call_args.kwargs["status_code"] == 409


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


class TestLoadRefusalsAsTheCallerSeesThem:
    """Review round 2, 2026-09-14: what a refused load looks like on each surface.

    MUTATION CONTROLS (mutate.py; restored and sha256-verified):
      R2-M11 a generic INSUFFICIENT_MEMORY message is back in ERROR_MESSAGES
             -> test_an_insufficient_memory_refusal_reaches_the_admin_ui_as_written
      R2-M13 the UNSUPPORTED_QUANTIZATION row is removed from ERROR_STATUS_MAP
             -> test_an_unsupported_quantization_on_an_openai_route_is_the_callers_to_change
    """

    @pytest.mark.asyncio
    async def test_an_unsupported_quantization_on_an_openai_route_is_the_callers_to_change(self):
        """A request naming a Q2 transformers checkpoint auto-loads it and is
        refused. Without a row it went out as a 400 typed server_error, which an
        OpenAI client retries instead of naming another model."""
        import json

        from millm.core.errors import UnsupportedQuantizationError

        response = await millm_error_handler(
            _make_request("/v1/chat/completions", "POST"),
            UnsupportedQuantizationError("Q2 cannot be loaded as a transformers model"),
        )
        body = json.loads(response.body)
        assert response.status_code == 400
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "unsupported_quantization"

    @pytest.mark.asyncio
    async def test_an_insufficient_memory_refusal_reaches_the_admin_ui_as_written(self):
        """The Admin UI toasts `error.message`. A generic INSUFFICIENT_MEMORY
        sentence replaced every refusal's figures and fix with advice to unload
        'other models' on a server that holds one."""
        import json

        from millm.core.errors import InsufficientMemoryError

        message = "Not enough GPU memory. Need ~40000 MB; split across GPUs this can hold 31952 MB."
        response = await millm_error_handler(
            _make_request("/api/models/3/load", "POST"), InsufficientMemoryError(message)
        )
        body = json.loads(response.body)
        assert response.status_code == 507
        assert body["error"]["message"] == message

