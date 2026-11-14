"""Requests that arrive under an old hostname (one from before the rename) are recorded (040 FR-040.11; FTDD §6.3).

A pure ASGI wrapper around the combined Socket.IO + FastAPI app (`miserver.main.create_combined_app`),
so admin-UI socket traffic on an old host counts too. It reads `Host` and `X-Forwarded-Host` from the
scope, records a hit for an observed alias, and passes EVERY scope through unchanged, `lifespan`
included (dropping it would silently skip every startup task). It awaits nothing but the app and
never logs: it runs on every request.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

from miserver.core import legacy_names

_HEADERS = (b"host", b"x-forwarded-host")


class LegacyHostMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            seen: set[str] = set()
            for name, value in scope.get("headers", ()):
                if name in _HEADERS:
                    for part in value.decode("latin-1").split(","):
                        alias = legacy_names.host_alias(part)
                        if alias is not None and alias.observed and alias.old not in seen:
                            seen.add(alias.old)
                            legacy_names.record("host", alias.old, alias.new)
        await self.app(scope, receive, send)
