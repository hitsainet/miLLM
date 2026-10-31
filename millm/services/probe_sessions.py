"""Probe sessions: which conversation a request belongs to, and what each turn recorded (Feature 36).

THREE STEPS, ALL OBSERVERS
1. `resolve_session` (the route, before generation): the `X-miLLM-Session` header wins; without it
   a salted continuation fingerprint finds the conversation the request continues (D6, D43).
2. `ProbeSessionService.write_turn` (inside `ProbeEventService.record`, after generation): one turn
   row with the hashes the NEXT request will be matched against, and — only while the operator has
   recording on (D5) — the newest user message and the visible reply.
3. `run_retention_pass` / `probe_retention_loop` (the lifespan): age and count limits, in batches.

⚠ NONE OF THEM MAY RAISE INTO A REQUEST. A session is bookkeeping about a generation; a failure is
logged and COUNTED (`session_write_failures`, `session_resolve_failures` in `/api/probes/status`)
and the request is served exactly as before. Nothing goes quiet without a count.

⚠ TEXT NEVER LEAVES THROUGH HERE. Log lines carry ids and counts; the transcript is written to its
own table and served by one route only.

THE FINGERPRINT (FTDD §4.2)
Each message becomes `(role, visible text)`; the list is serialised canonically and hashed with
HMAC-SHA256 under a 32-byte per-installation salt stored once in `app_settings`. Per turn:

* `prefix_hash`       H(messages before the newest user message) — never looked up itself.
* `continuation_hash` H(messages + ("assistant", reply))         — the next request's prefix
                                                                    equals this when the client
                                                                    resent our reply verbatim.
* `weak_hash`         H(messages)                                 — the next prefix minus its
                                                                    trailing assistant message
                                                                    equals this when the client
                                                                    re-rendered the reply.

A weak match counts only when it identifies ONE session (TC-3): two conversations identical so far
must not be merged on a key that ignores the very message that might tell them apart.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

from millm.api.probe_request_meta import SessionRef
from millm.core.config import settings

logger = logging.getLogger(__name__)

SALT_KEY = "probe_session_salt"

#: Injectable clock (tests freeze it). Always timezone-aware UTC.
_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)  # noqa: E731

_salt_cache: Optional[bytes] = None

#: Failures counted rather than raised. Reported by `GET /api/probes/status`.
counters: dict[str, Any] = {
    "session_resolve_failures": 0,
    "session_write_failures": 0,
    "retention_failures": 0,
    "last_retention_pass_at": None,
}


def _forget_for_tests() -> None:
    global _salt_cache
    _salt_cache = None
    counters.update(
        session_resolve_failures=0, session_write_failures=0, retention_failures=0,
        last_retention_pass_at=None,
    )


# ── canonical form ───────────────────────────────────────────────────────────────


def _field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _text_of(content: Any) -> str:
    """A message's visible text. Parts: text joined by `\\n`; any other part `[<type>]`."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        out: list[str] = []
        for part in content:
            kind = _field(part, "type")
            if kind == "text":
                out.append(str(_field(part, "text") or ""))
            else:
                out.append(f"[{kind or 'part'}]")
        return "\n".join(out)
    return str(content)


def canonical_pairs(messages: Iterable[Any]) -> list[tuple[str, str]]:
    """`(role, text)` per message: role lower-cased; text with line endings normalised to `\\n`
    and outer whitespace stripped. Nothing else is normalised, so a one-character edit changes
    the hash. `reasoning_content`, `name` and every other key are dropped."""
    pairs: list[tuple[str, str]] = []
    for message in messages:
        role = str(_field(message, "role") or "").lower()
        text = _text_of(_field(message, "content")).replace("\r\n", "\n").replace("\r", "\n")
        pairs.append((role, text.strip()))
    return pairs


def fingerprint(salt: bytes, pairs: Sequence[tuple[str, str]]) -> str:
    data = json.dumps([list(p) for p in pairs], ensure_ascii=False, separators=(",", ":"))
    return hmac.new(salt, data.encode("utf-8"), hashlib.sha256).hexdigest()


def last_user_index(pairs: Sequence[tuple[str, str]]) -> Optional[int]:
    for i in range(len(pairs) - 1, -1, -1):
        if pairs[i][0] == "user":
            return i
    return None


def continuation_hash(salt: bytes, pairs: Sequence[tuple[str, str]], reply: str) -> str:
    reply_text = reply.replace("\r\n", "\n").replace("\r", "\n").strip()
    return fingerprint(salt, list(pairs) + [("assistant", reply_text)])


# ── the installation salt (D43) ──────────────────────────────────────────────────


async def installation_salt(db: Any) -> bytes:
    """32 random bytes, written once (insert-if-absent) and cached.

    ⚠ DELIBERATELY NOT `secret_box`-ENCRYPTED: that module refuses without
    `SETTINGS_ENCRYPTION_KEY` by design, and an install without the key would lose session
    grouping entirely. The salt defeats precomputed and cross-installation tables; a reader of
    this database can read any recorded transcript anyway, which is the larger exposure (FTDD
    §4.2). It is never returned by a route and never logged.
    """
    global _salt_cache
    if _salt_cache is not None:
        return _salt_cache
    from millm.db.models.app_setting import AppSetting

    row = await db.get(AppSetting, SALT_KEY)
    if row is None:
        db.add(AppSetting(key=SALT_KEY, value=secrets.token_hex(32)))
        try:
            await db.commit()
        except Exception:  # a concurrent first request won the insert: read theirs
            await db.rollback()
        row = await db.get(AppSetting, SALT_KEY)
    _salt_cache = bytes.fromhex(row.value)
    return _salt_cache


# ── resolution ───────────────────────────────────────────────────────────────────


def _aware(value: datetime) -> datetime:
    """SQLite returns naive datetimes for `timezone=True` columns; they are UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def is_open(last_seen_at: datetime, now: datetime) -> bool:
    """⚠ STRICT `<`: a session idle for exactly the limit is closed (FR-36.3, boundary pinned)."""
    idle = timedelta(minutes=int(settings.PROBE_SESSION_IDLE_MINUTES))
    return now - _aware(last_seen_at) < idle


def _new_id() -> str:
    return f"ps_{uuid.uuid4().hex[:12]}"


async def resolve_in(
    db: Any,
    *,
    messages: Optional[Sequence[Any]],
    client_session_id: Optional[str],
    served_model: Optional[str],
) -> SessionRef:
    """Find or open this request's session in `db`. Raises on a database fault (the caller
    counts it). `messages=None` is a text completion: header or a single-turn session."""
    from millm.db.repositories.probe_session_repository import ProbeSessionRepository

    repo = ProbeSessionRepository(db)
    salt = await installation_salt(db)
    now = _now()
    chat = messages is not None
    pairs = canonical_pairs(messages or [])
    last_user = last_user_index(pairs)
    prefix = pairs[:last_user] if last_user is not None else list(pairs)
    turn_index = sum(1 for role, _ in pairs if role == "user") if chat else 1
    turn_index = max(turn_index, 1)
    prefix_hash = fingerprint(salt, prefix)
    weak_hash = fingerprint(salt, pairs)
    user_text = pairs[last_user][1] if last_user is not None else None
    transcripts = "on" if str(settings.PROBE_TRANSCRIPTS).lower() == "on" else "off"

    def ref(session_id: str, source: str, new: bool) -> SessionRef:
        return SessionRef(
            id=session_id, client_session_id=client_session_id, source=source,
            turn_index=turn_index, prefix_hash=prefix_hash, weak_hash=weak_hash,
            transcripts=transcripts, new=new, pairs=tuple(pairs), user_text=user_text, chat=chat,
        )

    async def open_new(source: str, *, predecessor: Optional[str] = None,
                       ambiguous: bool = False) -> SessionRef:
        session_id = _new_id()
        await repo.create_session(
            id=session_id, client_session_id=client_session_id, source=source,
            predecessor_id=predecessor, link_ambiguous=ambiguous, served_model=served_model,
            opened_at=now, last_seen_at=now, n_turns=0,
        )
        return ref(session_id, source, True)

    # 1. The header wins (D6).
    if client_session_id is not None:
        existing = await repo.newest_by_client_id(client_session_id)
        if existing is None:
            return await open_new("header")
        if is_open(existing.last_seen_at, now):
            await repo.touch(existing.id, now)
            return ref(existing.id, "header", False)
        return await open_new("header", predecessor=existing.id)

    # 2. A text completion, or the first request of a conversation: nothing to continue.
    if not chat:
        return await open_new("single")
    if not prefix:
        return await open_new("fingerprint")

    # 3. The fingerprint: strong, then weak (unique only) — on open sessions, then closed ones.
    weak_prefix = prefix[:-1] if prefix and prefix[-1][0] == "assistant" else None
    weak_key = fingerprint(salt, weak_prefix) if weak_prefix is not None else None
    for want_open in (True, False):
        strong = await repo.sessions_matching("continuation_hash", prefix_hash)
        strong = [s for s in strong if is_open(s.last_seen_at, now) == want_open]
        candidates = strong
        if not candidates and weak_key is not None:
            weak = await repo.sessions_matching("weak_hash", weak_key)
            candidates = [s for s in weak if is_open(s.last_seen_at, now) == want_open]
        # Two or more hits: possibly two conversations identical so far (the weak key ignores the
        # reply altogether), so join neither (TC-3, D44).
        if len(candidates) > 1:
            return await open_new("fingerprint", ambiguous=True)
        if len(candidates) == 1:
            hit = candidates[0]
            if want_open:
                await repo.touch(hit.id, now)
                return ref(hit.id, "fingerprint", False)
            return await open_new("fingerprint", predecessor=hit.id)
    return await open_new("fingerprint")


async def resolve_session(
    *,
    messages: Optional[Sequence[Any]],
    client_session_id: Optional[str],
    served_model: Optional[str] = None,
) -> Optional[SessionRef]:
    """The route's entry point: its own database session, never raises (`None` on a fault)."""
    from millm.db import base as db_base

    try:
        async with db_base.async_session_factory() as db:
            return await resolve_in(
                db, messages=messages, client_session_id=client_session_id,
                served_model=served_model,
            )
    except Exception as exc:
        counters["session_resolve_failures"] += 1
        logger.warning("probe_session_resolve_failed error=%s", type(exc).__name__)
        return None


# ── the turn writer ──────────────────────────────────────────────────────────────


class ProbeSessionService:
    """Writes one turn (and its transcript while recording is on). Raises; the caller counts."""

    def __init__(self, db: Any) -> None:
        self.db = db

    async def write_turn(
        self,
        session: SessionRef,
        *,
        request_id: str,
        served_model: Optional[str],
        reply_text: Optional[str],
        incomplete: bool = False,
        n_choices: int = 1,
        origin: str = "live",
    ) -> dict[str, Any]:
        from millm.db.repositories.probe_session_repository import ProbeSessionRepository

        repo = ProbeSessionRepository(self.db)
        salt = await installation_salt(self.db)
        cont = (
            continuation_hash(salt, session.pairs, reply_text)
            if session.chat and reply_text is not None else None
        )
        record_text = session.transcripts == "on" and session.chat
        turn = await repo.add_turn(
            session_id=session.id,
            request_id=request_id,
            turn_index=session.turn_index,
            prefix_hash=session.prefix_hash,
            continuation_hash=cont,
            weak_hash=session.weak_hash,
            origin=origin,
            served_model=served_model,
            has_transcript=record_text,
            transcript_incomplete=bool(record_text and incomplete),
            created_at=_now(),
        )
        if record_text:
            await repo.add_transcript(
                turn_id=turn.id, session_id=session.id, user_text=session.user_text,
                reply_text=reply_text or "", n_choices=max(int(n_choices or 1), 1),
                created_at=_now(),
            )
        n_turns = await repo.bump(session.id, _now())
        return {
            "session_id": session.id,
            "turn_index": session.turn_index,
            "request_id": request_id,
            "n_turns": n_turns,
            "has_transcript": record_text,
        }


# ── retention (FR-36.10) ─────────────────────────────────────────────────────────


async def run_retention_pass(db: Any) -> dict[str, int]:
    """One pass of every age and count limit, in batches. Returns rows removed per table."""
    from millm.db.repositories.probe_repository import ProbeEventRepository
    from millm.db.repositories.probe_session_repository import ProbeSessionRepository

    repo = ProbeSessionRepository(db)
    now = _now()
    removed = {
        "events": await ProbeEventRepository(db).prune_aged(int(settings.PROBE_MAX_AGE_DAYS)),
        "transcripts": await repo.prune_transcripts(
            max_age=timedelta(days=int(settings.PROBE_TRANSCRIPT_MAX_AGE_DAYS)),
            max_turns=int(settings.PROBE_TRANSCRIPT_MAX_TURNS),
            now=now,
        ),
        "sessions": await repo.prune_sessions(
            max_age=timedelta(days=int(settings.PROBE_SESSION_MAX_AGE_DAYS)), now=now,
            idle=timedelta(minutes=int(settings.PROBE_SESSION_IDLE_MINUTES)),
        ),
    }
    # 039 S2 (TD39-17): rule results follow probe events — the age rule, then each signal's cap
    # per origin. Deleting a rule set never deletes its results; only these limits do.
    from millm.db.repositories.rule_set_repository import prune_rule_results

    removed["rule_results"] = await prune_rule_results(
        db, max_age_days=settings.PROBE_MAX_AGE_DAYS,
        live_cap=int(settings.PROBE_MAX_EVENTS_PER_PROBE),
        batch_cap=int(settings.PROBE_MAX_BATCH_EVENTS_PER_PROBE),
    )
    await db.commit()
    counters["last_retention_pass_at"] = now.isoformat()
    return removed


async def _one_pass() -> None:
    from millm.db import base as db_base

    try:
        async with db_base.async_session_factory() as db:
            removed = await run_retention_pass(db)
        if any(removed.values()):
            logger.info(
                "probe_retention_pass events=%s transcripts=%s sessions=%s",
                removed["events"], removed["transcripts"], removed["sessions"],
            )
    except Exception as exc:
        counters["retention_failures"] += 1
        logger.warning("probe_retention_pass_failed error=%s", type(exc).__name__)


async def probe_retention_loop() -> None:
    """Prune on a fixed interval, starting with one pass at startup. Never raises but on cancel.

    The interval is re-read each round, so a runtime change takes effect at the next pass.
    """
    while True:
        await _one_pass()
        await asyncio.sleep(max(float(settings.PROBE_RETENTION_INTERVAL_S), 1.0))
