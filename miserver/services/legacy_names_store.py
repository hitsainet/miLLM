"""The persistent record of old-name use, and the health block built from it (040 FR-040.10,
FR-040.12; FTDD §5.1, §6.2; TID §3.2).

`miserver.core.legacy_names` counts uses in memory on the request path; `flush_loop` (started by the
lifespan, `miserver/main.py`) writes them to `legacy_name_uses` once a minute, and once more at
shutdown, so `last_seen` survives a restart and P7 retires an alias on evidence. `read_block` builds
the suite's `legacy_names` block (BR-023: exactly four keys) and `legacy_names_meta` for
`GET /api/health/detailed`. Neither ever raises: a flush failure keeps its hits for the next flush,
a read failure is reported in `legacy_names_meta.error` (S001 reads a missing `error` key, or a
missing block, as an unmeasured day, CF-24).

Names, times and counts only. No value ever leaves `legacy_names.environment_view`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select

from miserver.core import legacy_names
from miserver.db.models.legacy_name_use import LegacyNameUse

logger = structlog.get_logger()

FLUSH_INTERVAL_S = 60.0
IN_USE_WINDOW = timedelta(hours=24)
READ_TIMEOUT_S = 1.0
SHUTDOWN_FLUSH_TIMEOUT_S = 2.0
#: A periodic flush gives up after this; a hung database must not hold the drained hits (review 040 P0).
LOOP_FLUSH_TIMEOUT_S = 10.0
META_KIND, META_NAME = "meta", "observing_since"

#: `legacy_names_meta.last_flush_at`: when this process last flushed (CF-4 "per-process freshness").
_LAST_FLUSH_AT: datetime | None = None


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(moment: datetime) -> datetime:
    """SQLite returns naive values for `DateTime(timezone=True)`; every stored time is UTC."""
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _stamp() -> None:
    global _LAST_FLUSH_AT
    _LAST_FLUSH_AT = _now()


async def flush_once(session_factory: Any) -> int:
    """Write every unflushed hit; returns the number of names written. Never raises: on a database
    error the hits go back into memory for the next flush."""
    hits = legacy_names.drain_hits()
    if not hits:
        _stamp()  # the heartbeat: nothing to write is still a flush (CF-5)
        return 0
    try:
        async with session_factory() as session, session.begin():
            for (kind, old), hit in hits.items():
                row = await session.get(LegacyNameUse, (kind, old))
                if row is None:
                    session.add(LegacyNameUse(kind=kind, old_name=old, new_name=hit.new,
                                              first_seen_at=hit.first, last_seen_at=hit.last,
                                              count=hit.count))
                else:
                    row.first_seen_at = min(_aware(row.first_seen_at), hit.first)
                    row.last_seen_at = max(_aware(row.last_seen_at), hit.last)
                    row.count = (row.count or 0) + hit.count
                    row.new_name = hit.new
        _stamp()
        return len(hits)
    except asyncio.CancelledError:
        legacy_names.restore_hits(hits)  # review fix: a cancelled flush gives its hits back
        raise
    except Exception as exc:  # noqa: BLE001 - a flush never fails anything; the hits are kept
        legacy_names.restore_hits(hits)
        logger.warning("legacy_names_flush_failed", error=type(exc).__name__)
        return 0


async def flush_loop(session_factory_getter: Callable[[], Any]) -> None:
    """Flush once a minute, forever (cancelled at shutdown). The getter is read at each flush, so a
    replaced `miserver.db.base.async_session_factory` (tests) is the one used."""
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_S)
        try:
            # Bounded: a timed-out flush is cancelled, and a cancelled flush restores its hits.
            await asyncio.wait_for(flush_once(session_factory_getter()), LOOP_FLUSH_TIMEOUT_S)
        except TimeoutError:
            logger.warning("legacy_names_flush_timeout")


async def shutdown_flush(session_factory: Any) -> None:
    """The last flush before exit, bounded so a stuck database cannot hold a rollout."""
    try:
        await asyncio.wait_for(flush_once(session_factory), SHUTDOWN_FLUSH_TIMEOUT_S)
    except TimeoutError:
        logger.warning("legacy_names_shutdown_flush_timeout")


async def _select_all(session_factory: Any) -> list[LegacyNameUse]:
    async with session_factory() as session:
        return list((await session.execute(select(LegacyNameUse))).scalars().all())


async def read_block(session_factory: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(legacy_names, legacy_names_meta)` as plain data for the health models. Never raises."""
    error: str | None = None
    try:
        rows = await asyncio.wait_for(_select_all(session_factory), READ_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - reported in meta.error, never the route failed
        rows, error = [], type(exc).__name__
        logger.warning("legacy_names_read_failed", error=error)
    now = _now()
    view = legacy_names.environment_view()
    for old, new in view.effective_old:  # set now: stamped, so the evidence survives a restart
        legacy_names.record("env", old, new)
    pending = legacy_names.peek_hits()

    last: dict[tuple[str, str], tuple[str, datetime]] = {}
    observing_since: datetime | None = None
    for row in rows:
        if row.kind == META_KIND:
            observing_since = _aware(row.first_seen_at) if row.old_name == META_NAME else observing_since
            continue
        last[(row.kind, row.old_name)] = (row.new_name or "", _aware(row.last_seen_at))
    for key, hit in pending.items():
        new, seen = last.get(key, (hit.new, hit.last))
        last[key] = (hit.new or new, max(seen, hit.last))
    for old, new in view.effective_old:
        last[("env", old)] = (new, now)

    window_start = now - IN_USE_WINDOW
    in_use = [{"kind": kind, "old": old, "new": new, "last_seen": seen}
              for (kind, old), (new, seen) in sorted(last.items())
              if kind in ("env", "host") and seen >= window_start]
    block = {
        "in_use": in_use,
        "also_set": [{"old": o, "new": n} for o, n in view.also_set],
        "conflicts": [{"old": o, "new": n} for o, n in view.conflicts],
        "retired_but_set": [o for o, _n in view.retired_but_set],
    }
    from miserver.core.config import settings

    meta = {
        "observing_since": observing_since,
        "last_flush_at": _LAST_FLUSH_AT,
        # A retired name is no longer read, so it is not watched (P7: `{"env": 0, "host": 0}`).
        "watching": {"env": sum(1 for a in legacy_names.ENV_ALIASES
                                if a.scope == "server" and a.old not in legacy_names.RETIRED_ENV),
                     "host": sum(1 for a in legacy_names.HOST_ALIASES if a.observed)},
        "in_use_window_hours": int(IN_USE_WINDOW.total_seconds() // 3600),
        "transition_hosts": sorted(a.old for a in legacy_names.HOST_ALIASES),
        "unobserved_hosts": sorted(a.old for a in legacy_names.HOST_ALIASES if not a.observed),
        "new_host_for": {a.old: a.new for a in legacy_names.HOST_ALIASES},
        "retire_on": settings.LEGACY_ALIASES_RETIRE_ON,
        "error": error,
    }
    return block, meta
