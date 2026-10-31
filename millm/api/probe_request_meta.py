"""The probe-session request headers, parsed in ONE place, and the per-request carrier (Feature 36).

`X-miLLM-Session` names the conversation a request belongs to (D6, D8): an opaque string of 1–128
printable ASCII characters, stored as given. An invalid value is refused with 400
`SESSION_ID_INVALID` BEFORE anything else happens, and the refusal never echoes the value (it may
be whatever a client put there, including personal data the UI asks clients not to send).

`X-miLLM-Origin: judge` (phase 4, D14) marks a judge's own call: no probe scores it, no session is
opened and no text is recorded. Any other value is refused rather than ignored, because request
policy promises that nothing a client sends is dropped without a trace.

The chat and completions routes parse here, resolve the session (`millm.services.probe_sessions`)
and leave the result in `_PROBE_META`, which `InferenceService._probe_begin` reads. A ContextVar,
like `_PROBE_VERDICTS`: the inference service is a process singleton, so per-request state cannot
live on it, and the routes must decide the echo headers before a stream's first byte.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Optional

from millm.core.errors import OriginInvalidError, SessionIdInvalidError

SESSION_HEADER = "X-miLLM-Session"
ORIGIN_HEADER = "X-miLLM-Origin"
SESSION_ID_HEADER = "X-miLLM-Session-Id"
TRANSCRIPT_HEADER = "X-miLLM-Transcript"

#: The one origin a client may state. `live` is the absence of the header.
ORIGIN_JUDGE = "judge"
ORIGIN_LIVE = "live"

MAX_SESSION_ID_CHARS = 128


def parse_session_header(value: Optional[str]) -> Optional[str]:
    """The client's session id, or `None` when the header is absent.

    ⚠ The refusal message never contains the value.
    """
    if value is None:
        return None
    if not (1 <= len(value) <= MAX_SESSION_ID_CHARS) or any(
        not (0x20 <= ord(ch) <= 0x7E) for ch in value
    ):
        raise SessionIdInvalidError(
            f"{SESSION_HEADER} must be 1-{MAX_SESSION_ID_CHARS} printable ASCII characters "
            "with no control characters. Do not put personal data in a session id.",
            details={"header": SESSION_HEADER, "max_length": MAX_SESSION_ID_CHARS},
        )
    return value


def parse_origin_header(value: Optional[str]) -> str:
    """`judge` or `live`. Any other stated value is refused (400 `ORIGIN_INVALID`)."""
    if value is None:
        return ORIGIN_LIVE
    if value.strip().lower() == ORIGIN_JUDGE:
        return ORIGIN_JUDGE
    raise OriginInvalidError(
        f"{ORIGIN_HEADER} accepts only '{ORIGIN_JUDGE}'.",
        details={"header": ORIGIN_HEADER, "accepted": [ORIGIN_JUDGE]},
    )


@dataclass(frozen=True)
class SessionRef:
    """The session a request was resolved into, carried to the probe record. In memory only.

    ⚠ `pairs` and `user_text` are the conversation's text. They exist so the turn writer can hash
    the continuation and, when recording is on, store the newest user message. They are never
    logged and never serialised; `__repr__` leaves them out.
    """

    id: str
    client_session_id: Optional[str]
    source: str
    turn_index: int
    prefix_hash: str
    weak_hash: str
    #: `"on"` when this request's text will be recorded (decided at the route: "next request").
    transcripts: str = "off"
    #: Whether this request opened the session.
    new: bool = False
    pairs: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    user_text: Optional[str] = field(default=None, repr=False)
    #: Chat requests record text; a text completion never does (FR-36.7: chat turns).
    chat: bool = True


@dataclass(frozen=True)
class RequestMeta:
    """What the route decided about a request's probe bookkeeping."""

    session: Optional[SessionRef] = None
    origin: str = ORIGIN_LIVE


_PROBE_META: "contextvars.ContextVar[Optional[RequestMeta]]" = contextvars.ContextVar(
    "millm_probe_meta", default=None
)


def set_probe_meta(meta: Optional[RequestMeta]) -> None:
    _PROBE_META.set(meta)


def get_probe_meta() -> Optional[RequestMeta]:
    return _PROBE_META.get()


def reset_probe_meta() -> None:
    """Drop a meta left by an earlier request in a reused context (and every batch row)."""
    _PROBE_META.set(None)


def response_headers(meta: Optional[RequestMeta]) -> dict[str, str]:
    """`X-miLLM-Session-Id` and `X-miLLM-Transcript` for a resolved request, else nothing.

    Nothing at all when no session was resolved (nothing armed, a judge, a failed resolve), so an
    unarmed server's response is byte-identical to what it was before this feature.
    """
    if meta is None or meta.session is None:
        return {}
    session = meta.session
    return {
        SESSION_ID_HEADER: session.id,
        TRANSCRIPT_HEADER: "recorded" if session.transcripts == "on" and session.chat else "off",
    }


def _probes_armed() -> bool:
    from millm.services.probe_runtime import ProbeRuntimeState

    return ProbeRuntimeState().has_armed()


async def resolve_request_meta(
    messages: Optional[list],
    client_session_id: Optional[str],
    *,
    served_model: Optional[str] = None,
    origin: str = ORIGIN_LIVE,
) -> Optional[RequestMeta]:
    """Resolve this request's session and leave it in `_PROBE_META`. Never raises.

    Call AFTER `pre_generation` (which resets the context) and before the response is committed.
    `None` — and no response headers — when no probe is armed: sessions exist only while a probe
    is armed (D40), so an unarmed server is unchanged. `messages=None` is a text completion.
    """
    if origin == ORIGIN_JUDGE:
        # No session is resolved for a judge, armed or not; the marker reaches `_probe_begin`.
        meta = RequestMeta(session=None, origin=ORIGIN_JUDGE)
        set_probe_meta(meta)
        return meta
    try:
        if not _probes_armed():
            set_probe_meta(None)
            return None
    except Exception:
        set_probe_meta(None)
        return None
    from millm.services.probe_sessions import resolve_session

    session = await resolve_session(
        messages=messages, client_session_id=client_session_id, served_model=served_model
    )
    meta = RequestMeta(session=session)
    set_probe_meta(meta)
    return meta
