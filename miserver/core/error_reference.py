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

import json
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

    A `MiLLMError` subclass whose message is composed entirely by miServer (configuration guidance,
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


def contract_errors(exc: Any, limit: int) -> list[dict[str, Any]]:
    """A pydantic ``ValidationError``'s errors as plain JSON with no ``input`` and no ``url``.

    The one form every import refusal sends (probe, circuit and cluster imports). Two traps:

    * ``exc.json()``, not ``exc.errors()`` (038 12.5a F4): a validator that raises ``ValueError``
      puts the exception OBJECT in ``ctx.error``, which a response cannot serialise; pydantic's
      own JSON writes it as its message.
    * ``include_input=False`` (038 12.5a R1, debt D-R1-a): a model-level validator's ``input`` is
      the WHOLE document (a probe's test-vector ``messages`` text, its weights) and these errors
      go back in the 4xx body. ``loc`` and ``msg`` name what is wrong; the caller already holds
      what it sent.
    """
    return json.loads(exc.json(include_url=False, include_input=False))[:limit]


def details_shape(details: Any) -> dict[str, Any]:
    """What a refusal's ``details`` may contribute to a log line: its keys and its numbers.

    Debt D-R1-b (038 controls R1): the ``api_error`` warning logged every ``details`` whole, so any
    refusal that carried user text (a document, a prompt, a client-chosen name) landed in the log.
    Top-level keys are named by the raise site and kept; a number, boolean or ``None`` is kept as
    its value; a string becomes ``"str[<length>]"``, a list ``"list[<length>]"`` and a mapping
    ``"dict[<n keys>]"`` — never their contents. The full details still reach the CALLER, who sent
    them; a server fault's details are logged in full under its ``error_id`` (``api_server_fault``).
    """
    if not isinstance(details, dict):
        return {}
    shape: dict[str, Any] = {}
    for key, value in details.items():
        if value is None or isinstance(value, (bool, int, float)):
            shape[str(key)] = value
        elif isinstance(value, str):
            shape[str(key)] = f"str[{len(value)}]"
        elif isinstance(value, (list, tuple)):
            shape[str(key)] = f"list[{len(value)}]"
        elif isinstance(value, dict):
            shape[str(key)] = f"dict[{len(value)}]"
        else:
            shape[str(key)] = type(value).__name__
    return shape
