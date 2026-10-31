"""Server-fault policy shared by every error handler: what a client may see, and the correlation id.

A server fault (an unexpected exception, or a `MiLLMError` answered with a 5xx that is not a
"retry later" or a capacity refusal) may wrap an internal exception: llama.cpp's text with file
paths, a database driver's message, a Hugging Face client error. That text is logged server-side
with an ``error_id`` and the client receives the stable code, a fixed sentence and the same
``error_id`` to quote — **at every log level**. The management handler used to return ``str(exc)``
whenever the root logger was enabled for DEBUG (ENH-003 inventory, 2026-10-20); the level of the
server's log must never change what the server sends.
"""

from __future__ import annotations

import uuid
from typing import Any

#: 503 says "retry later" and 507 says "not enough memory/disk for this": both are refusals miLLM
#: composes for the caller (figures, which card), not wrapped internals.
_REFUSAL_5XX = frozenset({503, 507})

#: `details` keys that carry a caught exception's text (`str(e)`, `f"{type(e).__name__}: {e}"`).
#: Removed from a server fault's details before they reach a client; the log keeps them.
EXCEPTION_TEXT_KEYS = frozenset(
    {"error", "errors", "circuit_error", "engine_message", "technical_message", "traceback"}
)


def new_error_id() -> str:
    """A fresh correlation id: in the server log line and in the client's response."""
    return uuid.uuid4().hex


def is_server_fault(exc: Any, status_code: int) -> bool:
    """True when `exc`'s own text must not reach the client.

    A `MiLLMError` subclass whose message is composed entirely by miLLM (configuration guidance,
    no wrapped exception) opts out with ``message_is_client_safe = True``.
    """
    if status_code < 500 or status_code in _REFUSAL_5XX:
        return False
    return not getattr(exc, "message_is_client_safe", False)


def client_safe_details(details: Any) -> dict[str, Any]:
    """A server fault's `details` with every exception-text field removed."""
    if not isinstance(details, dict):
        return {}
    return {k: v for k, v in details.items() if k not in EXCEPTION_TEXT_KEYS}


def see_logs(error_id: str) -> str:
    """The `technical_message` a client gets for a server fault."""
    return f"See server logs (error id {error_id})."
