"""`probe:results`: one message per request, every result in it (039 FR-39.14, TD39-28; FTASKS 4.6).

The REAL record path (`ProbeEventService.record` over real repositories) and the REAL emitter
(`progress_emitter.emit_probe_results`) with a recording socket server behind `_sio`.

Controls (039 controls record):
* restore a per-result `[:5]` slice                     → `test_all_27_results_of_one_request_arrive_in_one_message` red
* delete the `_emit_request` call in `_publish`          → same test red
* drop `behaviour` from the stored object's projection   → `test_each_result_names_its_behaviour` red
"""

from __future__ import annotations

import asyncio

import pytest

from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services.probe_event_service import ProbeEventService
from millm.services.probe_runtime import Verdict
from tests.support import behaviour_factory as bf

pytestmark = pytest.mark.asyncio


@pytest.fixture
def socket(monkeypatch):
    from millm.sockets.progress import progress_emitter

    emitted: list[tuple[str, dict]] = []

    class Sio:
        def emit(self, event, payload):
            emitted.append((event, payload))

            async def _done():
                return None

            return _done()

    monkeypatch.setattr(progress_emitter, "_sio", Sio())
    return emitted


async def _probes(session, n, key):
    return [await bf.import_probe(session, bf.definition(f"{key}-{i}", key=key, scale=0.1 + i))
            for i in range(n)]


def _verdicts(rows):
    return [Verdict(probe_id=r.id, name=r.name, rung=2, rung_language="detects on unseen tasks",
                    scored=True, score=0.1 * i, threshold=0.5, fires=False, n_scored_tokens=3,
                    window=w)
            for i, r in enumerate(rows) for w in ("all", "prompt", "response")]


async def _record(session, rows, request_id):
    from types import SimpleNamespace

    from millm.services.probe_result import build_probe_results
    from millm.services.probe_result_order import order_results
    from millm.services.probe_arming import armed_probe_from_row

    probes = [armed_probe_from_row(r) for r in rows]
    context = SimpleNamespace(request_id=request_id, probes=probes, session=None, trace=None,
                              interrupted=None)
    verdicts = _verdicts(rows)
    results = order_results(build_probe_results(verdicts, context), probes)
    service = ProbeEventService(ProbeRepository(session), ProbeEventRepository(session))
    await service.record(request_id, verdicts, results=results)
    return service


async def test_all_27_results_of_one_request_arrive_in_one_message(test_session, socket):
    rows = await _probes(test_session, 9, bf.random_key())
    await _record(test_session, rows, "req-27")
    await asyncio.sleep(0)
    messages = [p for e, p in socket if e == "probe:results"]
    assert len(messages) == 1
    assert messages[0]["request_id"] == "req-27"
    assert len(messages[0]["results"]) == 27
    assert {(r["probe_id"], r["window"]) for r in messages[0]["results"]} == {
        (row.id, w) for row in rows for w in ("all", "prompt", "response")}
    assert all("context_text" not in r for r in messages[0]["results"])
    assert not any(e == "probe:event" for e, _ in socket), "the per-result event is retired"


async def test_each_result_names_its_behaviour(test_session, socket):
    key = bf.random_key()
    rows = await _probes(test_session, 2, key)
    await _record(test_session, rows, "req-behaviour")
    (message,) = [p for e, p in socket if e == "probe:results"]
    assert {r["behaviour"]["key"] for r in message["results"]} == {key}


async def test_a_second_request_within_the_interval_is_dropped_whole_and_counted(
        test_session, socket):
    rows = await _probes(test_session, 2, bf.random_key())
    service = await _record(test_session, rows, "req-first")
    service.repository = ProbeRepository(test_session)
    verdicts = _verdicts(rows)
    await service.record("req-second", verdicts)  # within 0.1 s of the first
    messages = [p for e, p in socket if e == "probe:results"]
    assert [m["request_id"] for m in messages] == ["req-first"]
    assert service._ws_dropped == {"requests": 1, "results": 6}
    assert (await service.status())["socket_events_dropped"] == {"requests": 1, "results": 6}
