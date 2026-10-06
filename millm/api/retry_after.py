"""
`RetryAfterMiddleware`: the safety net under the `Retry-After` policy (Feature 29, FR-29.6.2).

Every 503 is supposed to leave its builder with a `Retry-After` header computed by
`backpressure.retry_after_for`. This middleware sees every HTTP response start and, when a 503
arrives WITHOUT the header, adds `RETRY_AFTER_FALLBACK_S` and logs `retry_after_defaulted`.

The fallback is a DISTINCT value (10) on purpose: the per-code tests assert each code's own
number and that `retry_after_defaulted` never fired, so a builder that forgot the header fails
both assertions instead of passing as "a Retry-After was set".

Pure ASGI, not Starlette's `BaseHTTPMiddleware`: the latter wraps streaming responses, and the
chat route streams Server-Sent Events. Starlette runs the registered exception handlers inside
the user middleware stack, so a `MiLLMError` rendered by `millm_error_handler` passes through
here too.
"""

from __future__ import annotations

from typing import Any

from millm.core.config import settings
from millm.core.logging import get_logger

logger = get_logger(__name__)


class RetryAfterMiddleware:
    """Add a fallback `Retry-After` to any 503 that lacks one, and say so in the log."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def wrapped(message: dict) -> None:
            if message["type"] == "http.response.start" and message["status"] == 503:
                headers = list(message.get("headers", []))
                if not any(key.lower() == b"retry-after" for key, _ in headers):
                    headers.append(
                        (b"retry-after", str(settings.RETRY_AFTER_FALLBACK_S).encode("latin-1"))
                    )
                    logger.warning(
                        "retry_after_defaulted",
                        path=scope.get("path"),
                        value=settings.RETRY_AFTER_FALLBACK_S,
                        detail="a 503 reached the response without Retry-After; its builder "
                        "does not use backpressure.retry_after_for",
                    )
                    message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, wrapped)
