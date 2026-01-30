"""
Exception handlers for FastAPI.

Converts MiLLMError exceptions to standard API responses with user-friendly messages.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from millm.api.schemas.common import ApiResponse
from millm.core.errors import MiLLMError
from millm.core.error_messages import get_user_friendly_message
from millm.core.logging import get_logger

logger = get_logger(__name__)


async def millm_error_handler(request: Request, exc: MiLLMError) -> JSONResponse:
    """
    Convert MiLLMError to a standard API error response.

    Includes both technical error message and user-friendly message.

    Args:
        request: The FastAPI request object.
        exc: The MiLLMError exception.

    Returns:
        JSONResponse with the error in ApiResponse format.
    """
    # Get user-friendly message for this error code
    user_message = get_user_friendly_message(exc.code, exc.message)

    logger.warning(
        "api_error",
        error_code=exc.code,
        error_message=exc.message,
        user_message=user_message,
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method,
        details=exc.details,
    )

    # Include both technical and user-friendly messages in details
    enhanced_details = {
        **exc.details,
        "user_message": user_message,
        "technical_message": exc.message,
    }

    response = ApiResponse.fail(
        code=exc.code,
        message=user_message,  # Use user-friendly message as primary
        details=enhanced_details,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions with a generic error response.

    Args:
        request: The FastAPI request object.
        exc: The exception.

    Returns:
        JSONResponse with a generic error message.
    """
    logger.error(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )

    user_message = get_user_friendly_message("INTERNAL_ERROR")

    response = ApiResponse.fail(
        code="INTERNAL_ERROR",
        message=user_message,
        details={
            "type": type(exc).__name__,
            "technical_message": str(exc) if logger.isEnabledFor(10) else "See server logs",
        },
    )

    return JSONResponse(
        status_code=500,
        content=response.model_dump(),
    )
