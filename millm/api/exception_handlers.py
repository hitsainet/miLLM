"""
Exception handlers for FastAPI.

Routes errors to the correct format based on the request path:
- /v1/* endpoints -> OpenAI error format
- All other endpoints -> Management API format (ApiResponse)
"""

from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from millm.api.schemas.common import ApiResponse
from millm.api.routes.openai.errors import ERROR_STATUS_MAP, create_openai_error
from millm.core.backpressure import retry_after_for
from millm.core.errors import MiLLMError
from millm.core.error_messages import get_user_friendly_message
from millm.core.error_reference import (
    client_safe_details,
    is_server_fault,
    new_error_id,
    see_logs,
)
from millm.core.logging import get_logger

logger = get_logger(__name__)

#: Request fields that carry a credential. A 422 body must never echo their values: FastAPI's
#: default puts each failing value in `input`, and a model-level error's `input` is the WHOLE body —
#: so a malformed model download echoed its `hf_token` back (Feature 34, found while adding the
#: saved-token route).
SECRET_FIELDS = frozenset({"hf_token", "token", "access_token", "api_key", "authorization"})
REDACTED = "[redacted]"


def redact_secrets(value: Any) -> Any:
    """``value`` with every credential field's value replaced, at any depth."""
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in SECRET_FIELDS:
                out[key] = REDACTED
            else:
                out[key] = redact_secrets(item)
        loc = out.get("loc")
        if (
            isinstance(loc, (list, tuple))
            and loc
            and isinstance(loc[-1], str)
            and loc[-1].lower() in SECRET_FIELDS
            and "input" in out
        ):
            out["input"] = REDACTED
        return out
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return value


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
    from fastapi.exceptions import RequestValidationError

    if not isinstance(exc, RequestValidationError):
        # Registered for RequestValidationError only; anything else here is
        # a mis-registration — don't mask it as a 400 (and don't rely on an
        # assert that python -O strips).
        raise exc
    if not _is_openai_route(request):
        return JSONResponse(
            status_code=422,
            content={"detail": redact_secrets(jsonable_encoder(exc.errors()))},
        )

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
    if len(errors) > 1:
        # First-error bias matches OpenAI's API, but say more exist so
        # clients don't fix one field per round-trip blindly.
        message += f" (and {len(errors) - 1} more validation error"
        message += "s)" if len(errors) > 2 else ")"

    return create_openai_error(
        message=message,
        error_type="invalid_request_error",
        param=param,
        code="invalid_parameter",
        status_code=400,
    )


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _exception_class(exc: BaseException) -> str:
    return type(exc).__name__


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
    openai_route = _is_openai_route(request)
    if openai_route:
        status_code, error_type = ERROR_STATUS_MAP.get(
            exc.code, (exc.status_code, "server_error")
        )
    else:
        status_code = exc.status_code

    # The status logged is the one the caller receives. It logged exc.status_code,
    # the management API's status, so a /v1 request refused during an unload read
    # 409 in the log while the client got 503 (hardware acceptance re-run,
    # 2026-09-14).
    logger.warning(
        "api_error",
        error_code=exc.code,
        error_message=exc.message,
        status_code=status_code,
        path=request.url.path,
        method=request.method,
        details=exc.details,
    )

    # Feature 29: every 503 says when to retry, from the one policy (FR-29.6.1, FR-29.6.2).
    # A header only — the envelope and code are unchanged (FR-29.6.4).
    retry_after = retry_after_for(exc.code, exc.details) if status_code == 503 else None

    # Use OpenAI format for /v1/* endpoints
    if openai_route:
        # An error that knows its OpenAI type says so (GenerationOutOfMemoryError:
        # the request is too large for the card, not a server fault to retry).
        error_type = getattr(exc, "openai_error_type", None) or error_type
        # `param` names the field a refusal is about (Feature 25: every refusal names
        # its field). It was always None here, so a client could not tell WHICH field
        # a 400 meant without parsing prose.
        param = exc.details.get("param") if isinstance(exc.details, dict) else None
        from millm.api.routes.openai.errors import client_safe_message

        openai_response = create_openai_error(
            # A 500's text may wrap an internal exception: logged, never sent (CodeQL alert 9).
            message=client_safe_message(exc, status_code, event="v1_server_fault"),
            error_type=error_type,
            code=exc.code.lower() if exc.code else None,
            param=param if isinstance(param, str) else None,
            status_code=status_code,
            retry_after=retry_after,
        )
        if exc.headers:
            openai_response.headers.update(exc.headers)
        return openai_response

    # Management API format for everything else
    if is_server_fault(exc, status_code):
        # A 5xx fault's message and details may wrap an internal exception (llama.cpp's text,
        # a Hugging Face client error, an OSError path): logged with an error id, never sent.
        # The friendly sentence falls back to the generic one, not to `exc.message`.
        error_id = new_error_id()
        logger.error(
            "api_server_fault",
            error_id=error_id,
            error_code=exc.code,
            error_message=exc.message,
            details=exc.details,
            status_code=status_code,
            path=request.url.path,
            method=request.method,
            exc_info=exc,
        )
        user_message = get_user_friendly_message(exc.code)
        enhanced_details = {
            **client_safe_details(exc.details),
            "user_message": user_message,
            "technical_message": see_logs(error_id),
            "error_id": error_id,
        }
    else:
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
    # ⚠ JSON-SAFE (12.5a F4): a `details` value the response cannot serialise (an exception
    # object, as pydantic's `ctx.error` is) turned a 4xx refusal into a bare 500 at render time.
    # A refusal's exception is written as its class and message; a SERVER FAULT's as its class
    # only (its text may wrap an internal error: logged above, never sent). Everything else as
    # `jsonable_encoder` writes it.
    encode = _exception_class if is_server_fault(exc, status_code) else _exception_text
    content = jsonable_encoder(response.model_dump(), custom_encoder={BaseException: encode})
    # 039 FR-39.30: the MCP tool that resolves the refusal, as `error.next` — management routes
    # only (`/v1` keeps the OpenAI envelope exactly). Absent when there is none, so every refusal
    # that predates 039 keeps its bytes.
    next_tool = getattr(exc, "next_tool", None)
    if next_tool and isinstance(content.get("error"), dict):
        content["error"]["next"] = next_tool

    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    # A class's own headers (`WEIGHTS_UNAUTHORIZED` -> `WWW-Authenticate: Bearer`, RFC 6750 §3).
    headers.update(exc.headers or {})
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=headers or None,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions with the appropriate error format.

    The exception's text and traceback go to the server log under an ``error_id``; the client
    gets the stable code, a fixed sentence and that ``error_id`` — at EVERY log level. This
    handler returned ``str(exc)`` as ``details.technical_message`` whenever its logger was
    enabled for DEBUG (ENH-003 inventory, 2026-10-20), and the exception's class name always.

    Args:
        request: The FastAPI request object.
        exc: The exception.

    Returns:
        JSONResponse with the appropriate error format.
    """
    error_id = new_error_id()
    logger.error(
        "unhandled_exception",
        error_id=error_id,
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )

    # Use OpenAI format for /v1/* endpoints
    if _is_openai_route(request):
        return create_openai_error(
            message=f"An internal server error occurred (error id {error_id}).",
            error_type="server_error",
            code="server_error",
            status_code=500,
        )

    # Management API format for everything else
    user_message = get_user_friendly_message("INTERNAL_ERROR")

    response = ApiResponse.fail(
        code="INTERNAL_ERROR",
        message=user_message,
        details={"technical_message": see_logs(error_id), "error_id": error_id},
    )

    return JSONResponse(
        status_code=500,
        content=response.model_dump(),
    )
