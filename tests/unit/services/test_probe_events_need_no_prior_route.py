"""Events are recorded from the FIRST served request, with no route having run first.

⚠ FOUND ON HARDWARE 2026-09-30, and it is the third defect of this exact shape in
`_probe_record` — the two before it are documented in that function's own comments.

`_probe_record` read the singleton straight off the dependency module::

    service = getattr(deps, "_probe_event_service", None)
    if service is None:
        return                      # ← silent. no log, no counter.

That global is populated by exactly ONE thing in the whole product: the `ProbeEventServiceDep` on
`GET /api/probes/status`. Nothing else in `millm/` resolves it — not arming, not import, not even
`GET /api/probes/events`.

So after every restart a probe could be armed, `hook_installed=True`, returning real verdicts in
`X-miLLM-Probe-Verdicts` — while every event was dropped on the floor. **A deploy restarts the
pod.** A caller driving `/v1/chat/completions` from the OpenAI API or Open WebUI, with no admin UI
polling status, never populates it at all, so the events are lost for the life of the process.

Observed exactly that way: `score=47.7113 verdict=?1` in the response header and
`events_recorded: 0` in the status payload; then `/status` was called once, the *identical* request
was replayed, and it recorded. The only thing that changed between the two was a GET.

⚠ THE SESSION IS THE SECOND HALF, and rebinding it is not optional. The singleton's repositories
belong to whichever REQUEST last resolved the dependency, and by the time inference finishes that
request is over. On hardware it still wrote, which is luck rather than a guarantee, and a closed
session would have failed into `probe_event_persist_failed` — logged, at least, but still no event.
`_probe_record` now owns a session for the write and rebinds the repositories to it, while REUSING
the instance, because the socket throttle and the dropped-event counter live on it.

The tests below assert the CHAIN from a cold module global to a persisted row. Restoring either
half of the old behaviour turns them red — verified as negative controls, both bit.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services.inference_service import InferenceService
from millm.services.probe_runtime import Verdict
from millm.services.probe_service import ProbeService
from tests.unit.probe_fixtures import probe_definition

pytestmark = pytest.mark.asyncio


def verdict(probe_id: str, **over) -> Verdict:
    base = dict(
        probe_id=probe_id, name="high-stakes", rung=2,
        rung_language="detects on unseen tasks", scored=True, score=47.7113,
        threshold=2.38749, fires=True, n_scored_tokens=134, top_positions=[39, 38, 45],
    )
    base.update(over)
    return Verdict(**base)


class _Context:
    """The minimum `_probe_record` reads off a request context."""

    def __init__(self, request_id="chatcmpl-cold", overhead_ms=5.94, n_passes=61):
        self.request_id = request_id
        self.overhead_ms = overhead_ms
        #: 61 = one prefill + 60 decode steps, the shape of the request that exposed the defect.
        self.n_passes = n_passes

    def finish(self):                       # only reached when verdicts are not passed in
        raise AssertionError("verdicts were passed; finish() must not be called again")


@pytest.fixture
async def cold(test_session, monkeypatch):
    """A process that has served no probe route: the module global is None.

    The session factory is redirected at the test session, so the write the fix performs in its
    OWN session lands somewhere this test can read.
    """
    import millm.api.dependencies as deps
    import millm.db.base as db_base

    monkeypatch.setattr(deps, "_probe_event_service", None, raising=False)

    @asynccontextmanager
    async def factory():
        yield test_session

    monkeypatch.setattr(db_base, "async_session_factory", factory)

    repo = ProbeRepository(test_session)
    events = ProbeEventRepository(test_session)
    probe = await ProbeService(repo).import_definition(probe_definition())

    service = MagicMock(spec=InferenceService)
    service.is_model_loaded = MagicMock(return_value=False)
    service._tokenizer = None
    return service, events, probe, deps


class TestAColdProcessStillRecords:
    async def test_the_first_request_records_without_any_route_having_run(self, cold):
        inference, events, probe, deps = cold
        assert deps._probe_event_service is None, "fixture precondition: the global is cold"

        await InferenceService._probe_record(
            inference, _Context(), [verdict(probe.id)], full_ids=[1, 2, 3]
        )

        assert await events.count(probe.id) == 1, (
            "a verdict was produced and no event was written — this is the defect: the event "
            "service is only created by GET /api/probes/status, so every event before the first "
            "status poll is dropped silently"
        )

    async def test_the_row_carries_the_verdict_it_was_given(self, cold):
        """⚠ PAYLOAD, not just presence. "an event exists" passes against a row recording the
        wrong score, and a probe monitor whose numbers are wrong is worse than one that is silent.
        """
        inference, events, probe, _deps = cold
        await InferenceService._probe_record(
            inference, _Context(), [verdict(probe.id)], full_ids=[1, 2, 3]
        )
        rows = await events.list_events(probe_id=probe.id)
        assert len(rows) == 1
        row = rows[0]
        assert row.verdict is True
        assert row.score == pytest.approx(47.7113, abs=1e-4)
        assert row.threshold == pytest.approx(2.38749, abs=1e-5)
        assert row.rung == 2, "the event must keep the rung that was true when observed"
        assert row.n_scored_tokens == 134

    async def test_it_publishes_the_service_so_status_can_report(self, cold):
        """The instance has to survive the call, or the throttle and the dropped-event counter
        reset on every request — which is the same as having no throttle.
        """
        inference, _events, probe, deps = cold
        await InferenceService._probe_record(
            inference, _Context(), [verdict(probe.id)], full_ids=[1, 2, 3]
        )
        assert deps._probe_event_service is not None, (
            "_probe_record built a service and threw it away; GET /api/probes/status would then "
            "report last_request_overhead_ms: null for traffic it has already scored"
        )

    async def test_the_overhead_reaches_the_service(self, cold):
        """The sibling defect in this same function, re-asserted from the cold path — the one
        that reported `last_request_overhead_ms: null` on every request ever served.
        """
        inference, _events, probe, deps = cold
        await InferenceService._probe_record(
            inference, _Context(overhead_ms=7.5), [verdict(probe.id)], full_ids=[1, 2, 3]
        )
        assert deps._probe_event_service._last_request_overhead_ms == pytest.approx(7.5)


class TestAWarmProcessRebindsRatherThanReusing:
    async def test_a_stale_session_is_replaced_not_trusted(self, cold):
        """⚠ The second half of the fix. A pre-existing singleton carries repositories bound to a
        FINISHED request's session; the write must not go through them.
        """
        inference, events, probe, deps = cold
        from millm.services.probe_event_service import ProbeEventService

        dead = MagicMock(name="closed-session-repo")
        stale = ProbeEventService(dead, dead)
        deps._probe_event_service = stale

        await InferenceService._probe_record(
            inference, _Context(), [verdict(probe.id)], full_ids=[1, 2, 3]
        )

        assert deps._probe_event_service is stale, "the instance must be reused, not replaced"
        assert stale.events is not dead, (
            "the write went through the repository bound to a finished request's session"
        )
        assert await events.count(probe.id) == 1


class TestItStillNeverRaises:
    async def test_a_broken_factory_is_swallowed(self, cold, monkeypatch):
        """`_probe_record` sits in a `finally` on three inference paths. It must never be able to
        fail a generation the user is waiting on.
        """
        inference, _events, probe, _deps = cold
        import millm.db.base as db_base

        def boom():
            raise RuntimeError("no database")

        monkeypatch.setattr(db_base, "async_session_factory", boom)
        await InferenceService._probe_record(
            inference, _Context(), [verdict(probe.id)], full_ids=[1, 2, 3]
        )   # must not raise

    async def test_no_context_is_a_no_op(self, cold):
        inference, _events, _probe, _deps = cold
        await InferenceService._probe_record(inference, None, None, full_ids=None)
