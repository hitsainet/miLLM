"""The Hugging Face token setting (Feature 34): the resolver's rule, and the saved token's store.

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
worker thread never opens a database session. It is loaded at startup (`load_hf_token_on_startup`,
called from ``main.lifespan``) and replaced by every save and clear. One replica, one worker.

THE PLAINTEXT NEVER LEAVES THIS MODULE except as a Hub credential: routes see `mask()`, logs see
the source and the mask, errors carry neither.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from millm.core import secret_box
from millm.core.errors import InvalidTokenError, MiLLMError

# The resolver itself lives in `millm/core/hf_credentials.py` so `millm/ml` can import it without
# importing the services package (which imports `millm/ml` back: a circular import at startup).
# Re-exported here: this module is the feature's one entry point for routes and tests.
from millm.core.hf_credentials import (  # noqa: F401
    Source,
    _env_token,
    _reset_memory,
    _set_memory,
    clean,
    credential_tag,
    hub_token,
    mask,
    resolve_hf_token,
    resolve_with_source,
    snapshot,
)

logger = structlog.get_logger()

#: The `app_settings` row name.
SETTING_KEY = "hf_token"
#: Longest token accepted on save. Hugging Face tokens are ~37 characters.
MAX_TOKEN_CHARS = 256
#: How long a Test of a TYPED token stays good for a Save of that same token. The Settings card's
#: natural order is type → Test → Save; without this the Save cleared the verification the Test had
#: just made, and the page read "never verified" for a token checked seconds earlier (2026-10-18,
#: found on the deployed pod). Short, because a check is a dated reading: a token can be revoked.
PENDING_CHECK_TTL = timedelta(minutes=10)


@dataclass(frozen=True)
class _PendingCheck:
    """The outcome of the last Test of a typed (not yet saved) token, keyed by its digest.

    Holds a SHA-256 of the token, never the token. A Save of the same token adopts the outcome;
    a Save of any other token, a Clear, or the TTL discards it."""

    digest: str
    at: datetime
    user: str | None = None
    error: str | None = None


_pending_lock = threading.Lock()
_pending: _PendingCheck | None = None


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _remember_check(
    token: str, *, at: datetime, user: str | None = None, error: str | None = None
) -> None:
    global _pending
    with _pending_lock:
        _pending = _PendingCheck(digest=_digest(token), at=at, user=user, error=error)


def _take_check(token: str) -> _PendingCheck | None:
    """The pending outcome for ``token`` if it is fresh, consuming whatever was pending."""
    global _pending
    with _pending_lock:
        pending, _pending = _pending, None
    if pending is None or pending.digest != _digest(token):
        return None
    if datetime.now(UTC) - pending.at > PENDING_CHECK_TTL:
        return None
    return pending


def _forget_check() -> None:
    global _pending
    with _pending_lock:
        _pending = None


class HubUnreachableError(MiLLMError):
    """The token test could not reach Hugging Face (network or Hub outage) — not a verdict on the token."""

    code = "HF_HUB_UNREACHABLE"
    status_code = 503


class HfTokenNotConfiguredError(MiLLMError):
    """A test was asked for with no token at any tier."""

    code = "HF_TOKEN_NOT_CONFIGURED"
    status_code = 404


class HfTokenInvalidValueError(MiLLMError):
    """A value that cannot be a token (empty, `none`, whitespace inside, too long, non-ASCII)."""

    code = "HF_TOKEN_INVALID_VALUE"
    status_code = 400


def validate_new_token(raw: Any) -> str:
    """The token to save, or HfTokenInvalidValueError. Messages never quote the value."""
    token = clean(raw)
    if token is None:
        raise HfTokenInvalidValueError(
            "Enter a token to save. To remove the saved token, use Clear.",
        )
    if len(token) > MAX_TOKEN_CHARS:
        raise HfTokenInvalidValueError(
            f"That is not a Hugging Face token: it is longer than {MAX_TOKEN_CHARS} characters.",
        )
    if not token.isascii() or any(ch.isspace() for ch in token) or not token.isprintable():
        raise HfTokenInvalidValueError(
            "That is not a Hugging Face token: it contains spaces or characters a token never has.",
        )
    return token


# ── Persistence ──────────────────────────────────────────────────────────────


async def _load_row(session: Any) -> Any:
    from millm.db.models.app_setting import AppSetting

    return await session.get(AppSetting, SETTING_KEY)


def _apply_row(row: Any) -> None:
    """Decrypt ``row`` into process memory. Never raises; an unreadable row is recorded, not used."""
    if row is None:
        _reset_memory()
        return
    token: str | None = None
    error: str | None = None
    try:
        token = clean(secret_box.decrypt(row.value, name=SETTING_KEY))
        if token is None:
            error = "corrupt"
    except secret_box.SettingsKeyMissingError:
        error = "key_missing"
    except secret_box.SecretDecryptError as exc:
        error = exc.reason
    _set_memory(
        token=token,
        error=error,
        verified_at=row.verified_at,
        verified_user=row.verified_user,
        last_check_error=getattr(row, "last_check_error", None),
    )
    if error is not None:
        logger.warning(
            "hf_token_stored_unreadable",
            reason=error,
            detail=(
                "a Hugging Face token is saved but cannot be decrypted, so it is NOT used; Hub "
                "calls fall back to HF_TOKEN or anonymous. Restore the key it was saved under, "
                "or save the token again."
            ),
        )


async def load_hf_token_on_startup(session_factory: Any) -> str | None:
    """Load the saved token into memory. Returns the source it leaves active. Never raises.

    Called from ``main.lifespan`` before the auto-load, so a model download at startup already
    carries the saved token."""
    try:
        async with session_factory() as session:
            row = await _load_row(session)
        _apply_row(row)
    except Exception as exc:  # noqa: BLE001 - a missing table (unmigrated DB) must not stop serving
        _reset_memory()
        logger.warning("hf_token_startup_load_failed", error_type=type(exc).__name__)
    _, source = resolve_with_source(None)
    stored_error = snapshot().error
    logger.info("hf_token_source", source=source, stored_error=stored_error)
    return source


async def save_token(session: Any, raw: Any) -> None:
    """Encrypt and store ``raw``; refuse (SettingsKeyMissingError) when no key is configured.

    Nothing is written before the encryption succeeds, so a refusal leaves no row.

    A save replaces the verification: it is cleared, unless THIS token was Tested (typed, unsaved)
    within `PENDING_CHECK_TTL`, in which case that Test's outcome is stored with it."""
    from millm.db.models.app_setting import AppSetting

    token = validate_new_token(raw)
    envelope = secret_box.encrypt(token, name=SETTING_KEY)
    pending = _take_check(token)
    verified_at: datetime | None = None
    verified_user: str | None = None
    check_error: str | None = None
    if pending is not None:
        check_error = pending.error
        if pending.error is None:
            verified_at, verified_user = pending.at, pending.user
    row = await _load_row(session)
    now = datetime.now(UTC)
    if row is None:
        row = AppSetting(key=SETTING_KEY, value=envelope, updated_at=now)
        session.add(row)
    else:
        row.value = envelope
        row.updated_at = now
    row.verified_at = verified_at
    row.verified_user = verified_user
    row.last_check_error = check_error
    await session.commit()
    _set_memory(
        token=token,
        error=None,
        verified_at=verified_at,
        verified_user=verified_user,
        last_check_error=check_error,
    )
    logger.info("hf_token_saved", masked=mask(token), verification_carried=pending is not None)


async def clear_token(session: Any) -> bool:
    """Delete the saved token. Returns whether one existed."""
    row = await _load_row(session)
    existed = row is not None
    if row is not None:
        await session.delete(row)
        await session.commit()
    _reset_memory()
    _forget_check()
    logger.info("hf_token_cleared", existed=existed)
    return existed


def status() -> dict[str, Any]:
    """What the Settings page shows. The plaintext is never part of it."""
    token, source = resolve_with_source(None)
    stored = snapshot()
    stored_present = stored.token is not None or stored.error is not None
    return {
        "configured": token is not None,
        "source": source,
        "masked": mask(token),
        "last_verified_at": stored.verified_at.isoformat() if stored.verified_at else None,
        "user": stored.verified_user,
        "stored": stored_present,
        "stored_error": stored.error,
        "last_check_error": stored.last_check_error,
        "encryption_available": secret_box.encryption_available(),
        "env_configured": _env_token() is not None,
    }


# ── Test (whoami) ────────────────────────────────────────────────────────────


def _whoami(token: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    return HfApi().whoami(token=token)


def _classify(exc: BaseException) -> str:
    """``invalid`` for a rejected token, ``network`` for anything that is not a verdict on it."""
    from huggingface_hub.errors import HfHubHTTPError

    if isinstance(exc, HfHubHTTPError):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in (401, 403):
            return "invalid"
    return "network"


async def check_token(session: Any, raw: Any = None) -> dict[str, Any]:
    """Ask Hugging Face who ``raw`` (or, when absent, the active token) belongs to.

    Testing the SAVED token — with no body, or with a typed token equal to it — records the outcome
    on its row: `verified_at` + `verified_user` on success, `last_check_error` on failure. A typed
    token that is not the saved one is remembered (by digest) for a Save of it that follows.
    Errors are named, never echo the token, and never carry the Hub's own message text."""
    candidate = clean(raw)
    if candidate is not None:
        token, source = candidate, "request"
    else:
        resolved, source = resolve_with_source(None)
        if resolved is None:
            raise HfTokenNotConfiguredError(
                "No Hugging Face token is configured: save one on this page or set HF_TOKEN."
            )
        token = resolved
    # The saved token's row is the record, whichever way the operator asked for the check.
    records_saved = source == "settings" or (
        source == "request" and snapshot().token is not None and snapshot().token == token
    )
    try:
        info = await asyncio.to_thread(_whoami, token)
    except Exception as exc:  # noqa: BLE001 - classified below; never re-raised raw
        kind = _classify(exc)
        logger.info("hf_token_test_failed", source=source, kind=kind, error_type=type(exc).__name__)
        reason = "invalid" if kind == "invalid" else "network"
        if records_saved:
            await _record_check(session, error=reason)
        elif source == "request":
            _remember_check(token, at=datetime.now(UTC), error=reason)
        if kind == "invalid":
            raise InvalidTokenError(
                "Hugging Face rejected the token: it is invalid, expired or revoked.",
                details={"source": source},
            ) from None
        raise HubUnreachableError(
            "Could not reach Hugging Face to check the token; the token itself was not judged. "
            "Retry shortly.",
            details={"source": source},
        ) from None
    user = info.get("name") if isinstance(info, dict) else None
    now = datetime.now(UTC)
    if records_saved:
        await _record_check(session, verified_at=now, user=user)
    elif source == "request":
        _remember_check(token, at=now, user=user)
    logger.info("hf_token_test_ok", source=source, user=user, recorded=records_saved)
    return {"ok": True, "user": user, "source": source, "verified_at": now.isoformat()}


async def _record_check(
    session: Any,
    *,
    verified_at: datetime | None = None,
    user: str | None = None,
    error: str | None = None,
) -> None:
    """Store a check of the SAVED token on its row and in memory.

    Success sets `verified_at`/`verified_user` and clears `last_check_error`; a failure sets
    `last_check_error` and keeps the last success (it is the LAST verified time, not "valid now")."""
    row = await _load_row(session)
    if error is None:
        if row is not None:
            row.verified_at = verified_at
            row.verified_user = user
            row.last_check_error = None
            await session.commit()
        _set_memory(verified_at=verified_at, verified_user=user, last_check_error=None)
    else:
        if row is not None:
            row.last_check_error = error
            await session.commit()
        _set_memory(last_check_error=error)
