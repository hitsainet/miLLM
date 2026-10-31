"""Retention runs on its own, not only on write (FR-36.10, FR-36.17.3, D5's 7 days, D42's cap).

Wiring control (036 FTDD §10): removing `create_task(probe_retention_loop()…)` from `lifespan`
turns `test_probe_retention_is_started_by_lifespan` red — it RUNS the real lifespan.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from millm.core.config import settings
from millm.db.models.probe import ProbeEvent
from millm.db.models.probe_session import ProbeSession, ProbeSessionTurn, ProbeTranscript
from millm.db.repositories.probe_repository import ProbeRepository
from millm.services import probe_sessions
from millm.services.probe_service import ProbeService
from tests.unit.batch_fixtures import batch_db  # noqa: F401
from tests.unit.probe_fixtures import probe_definition
from tests.support.probe_event_rows import event_row

pytestmark = pytest.mark.asyncio

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    probe_sessions._forget_for_tests()
    monkeypatch.setattr(probe_sessions, "_now", lambda: NOW)
    yield
    probe_sessions._forget_for_tests()


async def _session(db, sid, *, last_seen, turns=(), text_ages=()):
    db.add(ProbeSession(id=sid, source="header", opened_at=last_seen, last_seen_at=last_seen))
    await db.commit()
    for i, age in enumerate(turns):
        turn = ProbeSessionTurn(session_id=sid, request_id=f"{sid}-{i}", turn_index=i + 1,
                                prefix_hash="p", weak_hash="w", has_transcript=i < len(text_ages),
                                created_at=NOW - age)
        db.add(turn)
        await db.commit()
        if i < len(text_ages):
            db.add(ProbeTranscript(turn_id=turn.id, session_id=sid, user_text="u",
                                   reply_text="r", created_at=NOW - text_ages[i]))
            await db.commit()


async def test_a_pass_removes_what_is_past_its_limits_with_no_new_traffic(test_session):
    db = test_session
    probe = await ProbeService(ProbeRepository(db)).import_definition(
        probe_definition(), raw_bytes=1, on_conflict="fail", origin="file")
    # Text 8 days old (past 7) and 6 days old (kept), in one live session.
    await _session(db, "ps_live", last_seen=NOW, turns=[timedelta(days=8), timedelta(days=6)],
                   text_ages=[timedelta(days=8), timedelta(days=6)])
    # A session not seen for 31 days (past PROBE_SESSION_MAX_AGE_DAYS = 30).
    await _session(db, "ps_aged", last_seen=NOW - timedelta(days=31), turns=[timedelta(days=31)])
    # Sessions with no turn: one idle (removed), one still open (a request may be generating).
    await _session(db, "ps_empty_idle", last_seen=NOW - timedelta(minutes=31))
    await _session(db, "ps_empty_open", last_seen=NOW - timedelta(minutes=5))
    db.add_all([
        ProbeEvent(**event_row(probe_id=probe.id, request_id="old", scored=True, window="all",
                   created_at=NOW - timedelta(days=31), session_id="ps_aged", turn_index=1)),
        ProbeEvent(**event_row(probe_id=probe.id, request_id="new", scored=True, window="all",
                   created_at=NOW - timedelta(days=1))),
    ])
    await db.commit()

    removed = await probe_sessions.run_retention_pass(db)

    texts = (await db.execute(select(ProbeTranscript))).scalars().all()
    assert [t.created_at.day for t in texts] == [(NOW - timedelta(days=6)).day]
    turns = {(t.session_id, t.turn_index): t.has_transcript
             for t in (await db.execute(select(ProbeSessionTurn))).scalars().all()}
    # The turn whose text expired stays (its verdicts are evidence) and says it has no text.
    assert turns == {("ps_live", 1): False, ("ps_live", 2): True}
    sessions = sorted(s.id for s in (await db.execute(select(ProbeSession))).scalars().all())
    assert sessions == ["ps_empty_open", "ps_live"]
    events = (await db.execute(select(ProbeEvent))).scalars().all()
    assert [e.request_id for e in events] == ["new"]
    assert removed == {"events": 1, "transcripts": 1, "sessions": 2, "rule_results": 0}  # 039 S2
    assert probe_sessions.counters["last_retention_pass_at"] == NOW.isoformat()


async def test_the_turn_cap_keeps_the_newest_text(test_session, monkeypatch):
    monkeypatch.setattr(settings, "PROBE_TRANSCRIPT_MAX_TURNS", 2)
    await _session(test_session, "ps_cap", last_seen=NOW, turns=[timedelta(hours=h) for h in (3, 2, 1)],
                   text_ages=[timedelta(hours=h) for h in (3, 2, 1)])
    await probe_sessions.run_retention_pass(test_session)
    kept = (await test_session.execute(select(ProbeTranscript))).scalars().all()
    assert sorted(round((NOW - t.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 3600)
                  for t in kept) == [1, 2]


async def test_a_failing_pass_is_counted_and_the_loop_goes_on(monkeypatch):
    calls = []

    async def broken(db):
        calls.append(1)
        raise RuntimeError("database gone")

    monkeypatch.setattr(probe_sessions, "run_retention_pass", broken)
    monkeypatch.setattr(settings, "PROBE_RETENTION_INTERVAL_S", 30)
    sleeps = []

    async def fast_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(probe_sessions.asyncio, "sleep", fast_sleep)
    with pytest.raises(asyncio.CancelledError):
        await probe_sessions.probe_retention_loop()
    assert len(calls) == 2 and sleeps == [30.0, 30.0]
    assert probe_sessions.counters["retention_failures"] == 2


async def test_probe_retention_is_started_by_lifespan(monkeypatch, batch_db):  # noqa: F811
    import millm.db.base as db_base
    import millm.main as main

    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    monkeypatch.setattr(main, "start_batch_api", AsyncMock())
    monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
    monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
    monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)
    # ⚠ The real `setup_logging` reconfigures structlog for the whole process (cached loggers), and
    # a later test in this process that captures logs with `structlog.testing` then sees nothing.
    # Logging is not under test here; the retention task is.
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    ran = asyncio.Event()
    passes = []

    async def spy(db):
        passes.append(db)
        ran.set()
        return {"events": 0, "transcripts": 0, "sessions": 0}

    monkeypatch.setattr(probe_sessions, "run_retention_pass", spy)
    async with main.lifespan(MagicMock()):
        await asyncio.wait_for(ran.wait(), timeout=5)
        assert len(passes) == 1  # the startup pass; the next waits for the interval
    tasks = [t for t in asyncio.all_tasks() if t.get_name() == "probe-retention"]
    assert all(t.done() for t in tasks), "the retention loop outlived the lifespan"
