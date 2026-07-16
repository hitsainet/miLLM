"""
Exception handlers for FastAPI.

Routes errors to the correct format based on the request path:
- /v1/* endpoints -> OpenAI error format
- All other endpoints -> Management API format (ApiResponse)
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from millm.api.schemas.common import ApiResponse
from millm.api.routes.openai.errors import ERROR_STATUS_MAP, create_openai_error
from millm.core.errors import MiLLMError
from millm.core.error_messages import get_user_friendly_message
from millm.core.logging import get_logger

logger = get_logger(__name__)


def _is_openai_route(request: Request) -> bool:
    """Check if the request is for an OpenAI-compatible endpoint."""
    return request.url.path.startswith("/v1/")


async def openai_validation_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Convert FastAPI RequestValidationError to the response shape each API
    family promises.

    /v1/* clients speak the OpenAI protocol: invalid parameters are a 400
    with an {"error": {message, type, param, code}} envelope (EC-10.3) — not
    FastAPI's default 422 pydantic dump with errors.pydantic.dev URLs.
    Non-/v1 paths keep FastAPI's default 422 shape, which the Admin UI's
    envelope-first parser already handles.
    """
    from fastapi.exception_handlers import request_validation_exception_handler
    from fastapi.exceptions import RequestValidationError

    assert isinstance(exc, RequestValidationError)
    if not _is_openai_route(request):
        return await request_validation_exception_handler(request, exc)  # type: ignore[return-value]

    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = [str(part) for part in first.get("loc", []) if part != "body"]
    param = ".".join(loc) or None
    message = first.get("msg", "Invalid request")
    # pydantic prefixes custom validator messages with "Value error, "
    if message.startswith("Value error, "):
        message = message[len("Value error, "):]
    if param:
        message = f"Invalid value for '{param}': {message}"

    return create_openai_error(
        message=message,
        error_type="invalid_request_error",
        param=param,
        code="invalid_parameter",
        status_code=400,
    )


async def millm_error_handler(request: Request, exc: MiLLMError) -> JSONResponse:
    """
    Convert MiLLMError to the appropriate error response format.

    Routes to OpenAI error format for /v1/* endpoints,
    management API format for all others.

    Args:
        request: The FastAPI request object.
        exc: The MiLLMError exception.

    Returns:
        JSONResponse with the appropriate error format.
    """
    logger.warning(
        "api_error",
        error_code=exc.code,
        error_message=exc.message,
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method,
        details=exc.details,
    )

    # Use OpenAI format for /v1/* endpoints
    if _is_openai_route(request):
        status_code, error_type = ERROR_STATUS_MAP.get(
            exc.code, (exc.status_code, "server_error")
        )
        return create_openai_error(
            message=exc.message,
            error_type=error_type,
            code=exc.code.lower() if exc.code else None,
            status_code=status_code,
        )

    # Management API format for everything else
    user_message = get_user_friendly_message(exc.code, exc.message)

    enhanced_details = {
        **exc.details,
        "user_message": user_message,
        "technical_message": exc.message,
    }

    response = ApiResponse.fail(
        code=exc.code,
        message=user_message,
        details=enhanced_details,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions with the appropriate error format.

    Args:
        request: The FastAPI request object.
        exc: The exception.

    Returns:
        JSONResponse with the appropriate error format.
    """
    logger.error(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )

    # Use OpenAI format for /v1/* endpoints
    if _is_openai_route(request):
        return create_openai_error(
            message="An internal server error occurred.",
            error_type="server_error",
            code="server_error",
            status_code=500,
        )

    # Management API format for everything else
    user_message = get_user_friendly_message("INTERNAL_ERROR")

    response = ApiResponse.fail(
        code="INTERNAL_ERROR",
        message=user_message,
        details={
            "type": type(exc).__name__,
            # structlog's stdlib BoundLogger exposes isEnabledFor (camelCase);
            # is_enabled_for only resolves on the unconfigured lazy proxy, so
            # under the production logging config it raised AttributeError —
            # crashing the handler on every unhandled exception. Ask the
            # stdlib logger directly.
            "technical_message": str(exc)
            if logging.getLogger(__name__).isEnabledFor(logging.DEBUG)
            else "See server logs",
        },
    )

    return JSONResponse(
        status_code=500,
        content=response.model_dump(),
    )
