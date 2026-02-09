"""
OpenAI error format helpers.

Provides utilities for creating OpenAI-compatible error responses.
All errors must match the OpenAI error response format exactly.

Error response format:
{
    "error": {
        "message": "Human-readable error message",
        "type": "error_type",
        "param": "parameter_name" | null,
        "code": "error_code" | null
    }
}
"""

from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from millm.core.errors import MiLLMError


def create_openai_error(
    message: str,
    error_type: str = "server_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
    status_code: int = 500,
) -> JSONResponse:
    """
    Create OpenAI-format error response.

    Args:
        message: Human-readable error message
        error_type: One of invalid_request_error, authentication_error,
                   rate_limit_error, server_error
        code: Machine-readable error code (e.g., "model_not_found")
        param: Parameter that caused the error (e.g., "model")
        status_code: HTTP status code

    Returns:
        JSONResponse with OpenAI error format
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


# Error code to (HTTP status, OpenAI error type) mapping
ERROR_STATUS_MAP: dict[str, tuple[int, str]] = {
    # Model errors
    "MODEL_NOT_LOADED": (503, "server_error"),
    "MODEL_NOT_FOUND": (404, "invalid_request_error"),
    "MODEL_ALREADY_LOADED": (400, "invalid_request_error"),
    "MODEL_LOADING": (503, "server_error"),
    # Validation errors
    "VALIDATION_ERROR": (400, "invalid_request_error"),
    "CONTEXT_LENGTH_EXCEEDED": (400, "invalid_request_error"),
    "INVALID_PARAMETER": (400, "invalid_request_error"),
    # Resource errors
    "INSUFFICIENT_MEMORY": (503, "server_error"),
    "RATE_LIMIT_EXCEEDED": (429, "rate_limit_error"),
    "QUEUE_FULL": (503, "server_error"),
    # Authentication (for future use)
    "AUTHENTICATION_ERROR": (401, "authentication_error"),
    "INVALID_API_KEY": (401, "authentication_error"),
    # Generic errors
    "SERVER_ERROR": (500, "server_error"),
    "INTERNAL_ERROR": (500, "server_error"),
}


async def openai_exception_handler(request: Request, exc: MiLLMError) -> JSONResponse:
    """
    Global exception handler for MiLLM errors on OpenAI endpoints.

    Converts MiLLMError exceptions to OpenAI-compatible error responses.
    Only handles requests to /v1/* endpoints.

    Register with FastAPI:
        app.add_exception_handler(MiLLMError, openai_exception_handler)

    Args:
        request: The FastAPI request object
        exc: The MiLLMError exception

    Returns:
        JSONResponse with OpenAI error format
    """
    # Get status code and error type from mapping
    status_code, error_type = ERROR_STATUS_MAP.get(exc.code, (500, "server_error"))

    return create_openai_error(
        message=exc.message,
        error_type=error_type,
        code=exc.code.lower() if exc.code else None,
        param=None,
        status_code=status_code,
    )


def model_not_loaded_error() -> JSONResponse:
    """Create error response for when no model is loaded."""
    return create_openai_error(
        message="No model is currently loaded. Load a model first using the Management API.",
        error_type="server_error",
        code="model_not_loaded",
        status_code=503,
    )


def model_not_found_error(model_id: str, available_model: Optional[str] = None) -> JSONResponse:
    """Create error response for model not found."""
    if available_model:
        message = f"The model '{model_id}' does not exist. Available: {available_model}"
    else:
        message = (
            f"The model '{model_id}' does not exist or has not been downloaded. "
            "Download it first using the Management API."
        )

    return create_openai_error(
        message=message,
        error_type="invalid_request_error",
        code="model_not_found",
        param="model",
        status_code=404,
    )


def model_locked_error(model_id: str, locked_model: str) -> JSONResponse:
    """Create error response when model is locked for steering."""
    return create_openai_error(
        message=f"The model '{model_id}' is not available. "
        f"Model '{locked_model}' is currently locked for steering.",
        error_type="invalid_request_error",
        code="model_locked",
        param="model",
        status_code=409,
    )


def validation_error(message: str, param: Optional[str] = None) -> JSONResponse:
    """Create error response for validation errors."""
    return create_openai_error(
        message=message,
        error_type="invalid_request_error",
        code="invalid_parameter",
        param=param,
        status_code=400,
    )


def context_length_exceeded_error(
    max_length: int, requested_length: int
) -> JSONResponse:
    """Create error response for context length exceeded."""
    return create_openai_error(
        message=f"This model's maximum context length is {max_length} tokens. "
        f"However, your messages resulted in {requested_length} tokens.",
        error_type="invalid_request_error",
        code="context_length_exceeded",
        status_code=400,
    )


def rate_limit_error(message: str = "Rate limit exceeded") -> JSONResponse:
    """Create error response for rate limiting."""
    return create_openai_error(
        message=message,
        error_type="rate_limit_error",
        code="rate_limit_exceeded",
        status_code=429,
    )


def server_error(message: str = "Internal server error") -> JSONResponse:
    """Create generic server error response."""
    return create_openai_error(
        message=message,
        error_type="server_error",
        code="server_error",
        status_code=500,
    )
