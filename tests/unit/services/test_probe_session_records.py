"""`ProbeEventService.record` stamps the session on every event and writes the turn (FR-36.4).

Mutation control (036 FTDD §10): removing the `session_id` key from the rows `record` builds turns
`test_events_carry_session` red.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from millm.db.models.probe import ProbeEvent
from millm.db.models.probe_session import ProbeSessionTurn
from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services import probe_sessions
from millm.services.probe_event_service import ProbeEventService
from millm.services.probe_runtime import Verdict
from millm.services.probe_service import ProbeService
from millm.services.probe_sessions import resolve_in
from tests.unit.probe_fixtures import probe_definition

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _fresh():
    probe_sessions._forget_for_tests()
    yield
    probe_sessions._forget_for_tests()


def verdict(probe_id, window="all", **over):
    base = dict(probe_id=probe_id, name="p", rung=2, rung_language="x", scored=True, score=1.5,
                threshold=1.0, fires=True, n_scored_tokens=4, top_positions=[1], window=window)
    base.update(over)
    return Verdict(**base)


async def _probe(db):
    return await ProbeService(ProbeRepository(db)).import_definition(
        probe_definition(), raw_bytes=1, on_conflict="rename", origin="file"
    )


async def test_events_carry_session(test_session):
    probe = await _probe(test_session)
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
                {"role": "user", "content": "e"}]
    ref = await resolve_in(test_session, messages=messages, client_session_id="s1",
                           served_model="m")
    service = ProbeEventService(ProbeRepository(test_session), ProbeEventRepository(test_session))
    written = await service.record(
        "chatcmpl-1", [verdict(probe.id, "all"), verdict(probe.id, "prompt", fires=False)],
        session=ref, reply_text="f",
    )
    assert written == 2
    events = (await test_session.execute(select(ProbeEvent))).scalars().all()
    assert [(e.session_id, e.turn_index) for e in events] == [(ref.id, 3), (ref.id, 3)]
    turn = (await test_session.execute(select(ProbeSessionTurn))).scalar_one()
    assert (turn.request_id, turn.turn_index, turn.session_id) == ("chatcmpl-1", 3, ref.id)


async def test_without_a_session_events_are_unrecorded_not_guessed(test_session):
    probe = await _probe(test_session)
    service = ProbeEventService(ProbeRepository(test_session), ProbeEventRepository(test_session))
    await service.record("chatcmpl-2", [verdict(probe.id)])
    event = (await test_session.execute(select(ProbeEvent))).scalar_one()
    assert (event.session_id, event.turn_index) == (None, None)
    assert (await test_session.execute(select(ProbeSessionTurn))).scalars().all() == []


async def test_a_batch_row_never_gets_a_session(test_session):
    probe = await _probe(test_session)
    ref = await resolve_in(test_session, messages=[{"role": "user", "content": "x"}],
                           client_session_id="s1", served_model="m")
    service = ProbeEventService(ProbeRepository(test_session), ProbeEventRepository(test_session))
    await service.record("chatcmpl-3", [verdict(probe.id)], session=ref, batch_row=("b1", 1))
    event = (await test_session.execute(select(ProbeEvent))).scalar_one()
    assert event.session_id is None and event.origin == "batch"
    assert (await test_session.execute(select(ProbeSessionTurn))).scalars().all() == []


async def test_a_failed_turn_write_is_counted_and_the_events_stay(test_session, monkeypatch):
    probe = await _probe(test_session)
    ref = await resolve_in(test_session, messages=[{"role": "user", "content": "x"}],
                           client_session_id="s1", served_model="m")

    async def boom(self, *a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(probe_sessions.ProbeSessionService, "write_turn", boom)
    service = ProbeEventService(ProbeRepository(test_session), ProbeEventRepository(test_session))
    assert await service.record("chatcmpl-4", [verdict(probe.id)], session=ref) == 1
    assert probe_sessions.counters["session_write_failures"] == 1
    status = await service.status()
    assert status["session_write_failures"] == 1
    assert len((await test_session.execute(select(ProbeEvent))).scalars().all()) == 1


async def test_the_request_reset_drops_an_earlier_requests_session():
    """`pre_generation` → `reset_steering_memo` clears `_PROBE_META`. The batch runner calls it per
    row, so a row run in a context an interactive request used never inherits its session."""
    from millm.api.probe_request_meta import RequestMeta, SessionRef, get_probe_meta, set_probe_meta
    from millm.services.inference_service import reset_steering_memo

    set_probe_meta(RequestMeta(session=SessionRef(
        id="ps_x", client_session_id=None, source="header", turn_index=1, prefix_hash="p",
        weak_hash="w")))
    reset_steering_memo()
    assert get_probe_meta() is None


async def test_the_age_rule_runs_once_per_record_not_once_per_probe(test_session, monkeypatch):
    """037 hardware finding: N table-wide age DELETEs per request with N probes armed."""
    service = ProbeEventService(ProbeRepository(test_session), ProbeEventRepository(test_session))
    a = await _probe(test_session)
    doc = probe_definition(name="second")
    b = await ProbeService(ProbeRepository(test_session)).import_definition(
        doc, raw_bytes=1, on_conflict="fail", origin="file")
    calls = {"aged": 0, "cap": []}
    real_aged, real_cap = service.events.prune_aged, service.events.prune_to_cap

    async def aged(days):
        calls["aged"] += 1
        return await real_aged(days)

    async def cap(probe_id, cap, origin="live"):
        calls["cap"].append(probe_id)
        return await real_cap(probe_id, cap, origin)

    monkeypatch.setattr(service.events, "prune_aged", aged)
    monkeypatch.setattr(service.events, "prune_to_cap", cap)
    await service.record("chatcmpl-p", [verdict(a.id), verdict(b.id)])
    assert calls["aged"] == 1 and sorted(calls["cap"]) == sorted([a.id, b.id])
