"""The ONE place miLLM chooses a Hugging Face credential (Feature 34).

PRECEDENCE, for every Hub call in the server:

    1. the explicit per-request token (a download/preview form's optional field, an API ``hf_token``)
    2. the token saved on the Admin UI's Settings page (encrypted in ``app_settings``)
    3. the ``HF_TOKEN`` environment variable (the ``millm-secrets`` key)
    4. anonymous

A value that is empty, whitespace, or ``none`` in any case is ABSENT at every tier: a Secret created
with ``HF_TOKEN=none`` must fall through, not be sent as ``Bearer none`` (miStudio's ``_clean``).

CALL SITES PASS `hub_token(explicit)`, never `resolve_hf_token`. It returns the token, or ``False``
when the answer is anonymous. ``None`` would not mean anonymous to ``huggingface_hub``: it means "look
one up yourself", from ``HF_TOKEN`` (unfiltered — the literal ``none`` included) and from
``~/.cache/huggingface/token``. ``False`` sends no credential at all.

The saved token is held decrypted in process memory — as ``HF_TOKEN`` already is — so a Hub call on a
worker thread never opens a database session. `millm/services/hf_token.py` fills it at startup
(`load_hf_token_on_startup`, called from ``main.lifespan``) and on every save and clear. One replica,
one worker.

In ``core`` and importing nothing from ``millm.services`` or ``millm.ml``: both of those import it.

THE PLAINTEXT NEVER LEAVES THIS MODULE except as a Hub credential: routes see `mask()`, logs see
the source and the mask, errors carry neither.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

Source = Literal["request", "settings", "env", "none"]


@dataclass
class _Stored:
    token: str | None = None
    #: Why a stored row exists but is not usable: ``key_missing`` / ``key_mismatch`` / ``corrupt``.
    error: str | None = None
    verified_at: datetime | None = None
    verified_user: str | None = None
    #: The outcome of the last test of the stored token that FAILED (cleared on success or change).
    last_check_error: str | None = None


_lock = threading.Lock()
_stored = _Stored()


def clean(value: Any) -> str | None:
    """``value`` as a usable token, or None when it is absent, blank or ``none``."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "none":
        return None
    return stripped


def _env_token() -> str | None:
    token = clean(os.environ.get("HF_TOKEN"))
    if token is None:
        from millm.core.config import settings

        token = clean(settings.HF_TOKEN)
    return token


def _stored_token() -> str | None:
    with _lock:
        return _stored.token


def resolve_with_source(explicit: Any = None) -> tuple[str | None, Source]:
    """The credential a Hub call should carry, and which tier it came from."""
    token = clean(explicit)
    if token is not None:
        return token, "request"
    token = _stored_token()
    if token is not None:
        return token, "settings"
    token = _env_token()
    if token is not None:
        return token, "env"
    return None, "none"


def resolve_hf_token(explicit: Any = None) -> str | None:
    """The credential by the precedence in the module docstring, or None for anonymous."""
    return resolve_with_source(explicit)[0]


def hub_token(explicit: Any = None) -> str | Literal[False]:
    """What a Hub call passes as ``token=``: the resolved token, or ``False`` (send none)."""
    return resolve_hf_token(explicit) or False


def credential_tag(token: Any) -> str:
    """A short, non-reversible tag for cache keys: equal tokens share a cache, others never do."""
    if not token:
        return "anon"
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()[:12]


def mask(token: str | None) -> str | None:
    """``hf_…abcd``: enough to recognise a token, never enough to use it."""
    if not token:
        return None
    if len(token) < 12:
        return "…"
    prefix = token[:3] if token.startswith("hf_") else ""
    return f"{prefix}…{token[-4:]}"


def _set_memory(**fields: Any) -> None:
    with _lock:
        for name, value in fields.items():
            setattr(_stored, name, value)


def _reset_memory() -> None:
    """Forget the stored token in this process (tests and `clear`)."""
    with _lock:
        for field in dataclasses.fields(_Stored):
            setattr(_stored, field.name, None)


def snapshot() -> _Stored:
    """A copy of the saved-token state (the token included: callers must only mask it)."""
    with _lock:
        return dataclasses.replace(_stored)
