"""Recording verdicts, reporting status, and the one thing that must never leave the process.

⚠ `TestThePayloadNeverCarriesPromptText` ASSERTS ABSENCE, NOT PRESENCE.

This estate has already shipped a socket broadcast that leaked user prompt text while the suite
stayed 135/135 green, because the test checked what the payload *contained*. A test that asserts
the right fields are present passes perfectly well against a payload that also carries the user's
words. The only test that catches it is one that asserts the forbidden keys are gone.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services.probe_event_service import CONTEXT_PREFIX, ProbeEventService, strip_context
from millm.services.probe_runtime import Verdict
from millm.services.probe_service import ProbeService
from tests.unit.probe_fixtures import probe_definition

pytestmark = pytest.mark.asyncio


def verdict(**over) -> Verdict:
    base = dict(
        probe_id="pr_1", name="high-stakes", rung=2, rung_language="detects on unseen tasks",
        scored=True, score=2.31, threshold=1.07, fires=True,
        n_scored_tokens=12, top_positions=[4, 9],
    )
    base.update(over)
    return Verdict(**base)


@pytest.fixture
async def ctx(test_session):
    repo = ProbeRepository(test_session)
    events = ProbeEventRepository(test_session)
    probe = await ProbeService(repo).import_definition(probe_definition())
    return repo, events, ProbeEventService(repo, events), probe


class TestRecording:
    async def test_every_verdict_becomes_an_event(self, ctx):
        """A probe with no row for a request is indistinguishable from one never armed."""
        repo, events, service, probe = ctx
        written = await service.record(
            "chatcmpl-a", [verdict(probe_id=probe.id), verdict(probe_id=probe.id, name="b")]
        )
        assert written == 2
        assert await events.count(probe.id) == 2

    async def test_an_unscored_verdict_records_its_reason_and_no_numbers(self, ctx):
        repo, events, service, probe = ctx
        await service.record(
            "chatcmpl-a",
            [verdict(probe_id=probe.id, scored=False, not_scored_reason="continuous_batching",
                     score=None, threshold=None, fires=None)],
        )
        row = (await events.list_events(probe_id=probe.id))[0]
        assert row.scored is False
        assert row.not_scored_reason == "continuous_batching"
        assert row.score is None
        assert "not scored: continuous_batching" in row.summary

    async def test_the_request_id_links_the_event_to_its_response(self, ctx):
        repo, events, service, probe = ctx
        await service.record("chatcmpl-abc", [verdict(probe_id=probe.id)])
        assert (await events.list_events(request_id="chatcmpl-abc"))[0].probe_id == probe.id

    async def test_no_verdicts_writes_nothing(self, ctx):
        _repo, events, service, _probe = ctx
        assert await service.record("chatcmpl-a", []) == 0
        assert await events.count() == 0

    async def test_a_persistence_failure_does_not_raise(self, ctx):
        """A monitor must not take a request down. The verdict is already on the response."""
        _repo, events, service, probe = ctx
        with patch.object(events, "create_many", side_effect=RuntimeError("db gone")):
            assert await service.record("chatcmpl-a", [verdict(probe_id=probe.id)]) == 0

    async def test_retention_runs_on_every_write(self, ctx, monkeypatch):
        """Bounded by construction, like sensing: the table cannot grow without bound even if
        nobody ever calls the API."""
        from millm.core.config import settings

        monkeypatch.setattr(settings, "PROBE_MAX_EVENTS_PER_PROBE", 3)
        _repo, events, service, probe = ctx
        for i in range(8):
            await service.record(f"chatcmpl-{i}", [verdict(probe_id=probe.id)])
        assert await events.count(probe.id) == 3


class TestThePayloadNeverCarriesPromptText:
    def test_context_keys_are_ABSENT_from_the_stripped_payload(self):
        payload = {
            "probe_id": "pr_1",
            "score": 2.31,
            "context_text": "the user's actual words",
            "context_token_ids": [1, 2, 3],
        }
        slim = strip_context(payload)
        assert "context_text" not in slim
        assert "context_token_ids" not in slim
        assert not any(key.startswith(CONTEXT_PREFIX) for key in slim)
        # and the useful fields survive
        assert slim["score"] == 2.31

    def test_no_VALUE_in_the_payload_contains_the_prompt(self):
        """Belt and braces: a future field could carry the text under another name."""
        secret = "ESCALATE THE SBA FILING"
        slim = strip_context({"probe_id": "p", "context_text": secret, "summary": "score 2.3"})
        assert secret not in repr(slim)

    async def test_the_EMITTED_payload_carries_no_context(self, ctx):
        """The end-to-end assertion: what actually reaches the socket."""
        _repo, _events, service, probe = ctx
        emitter = MagicMock()
        with patch("millm.sockets.progress.progress_emitter", emitter):
            await service.record(
                "chatcmpl-a",
                [verdict(probe_id=probe.id)],
                contexts={probe.id: {"context_text": "user words", "context_token_ids": [7]}},
            )
        assert emitter.emit_probe_event.called
        sent = emitter.emit_probe_event.call_args[0][0]
        assert "context_text" not in sent and "context_token_ids" not in sent
        assert "user words" not in repr(sent)

    async def test_the_context_IS_still_stored_for_the_detail_route(self, ctx):
        """Stripped from the broadcast, kept in the row — a reviewer asks for one event."""
        _repo, events, service, probe = ctx
        await service.record(
            "chatcmpl-a",
            [verdict(probe_id=probe.id)],
            contexts={probe.id: {"context_text": "user words", "context_token_ids": [7]}},
        )
        assert (await events.list_events(probe_id=probe.id))[0].context_text == "user words"


def _install_hook(probe_id: str):
    """Put the probe in the LIVE registry, which is what `armed_count` now counts.

    ⚠ These tests used to set `armed=True` on the row alone and assert `armed_count == 1`. That
    pinned a defect rather than preventing one: the hook lives in `ProbeRuntimeState`, so a row
    saying armed with nothing installed is precisely the post-restart state where status reported a
    monitor that was not monitoring (observed live 2026-09-28). The stale case now has its own
    coverage in `test_probe_survives_no_restart.py`; here the probe is genuinely armed.
    """
    from unittest.mock import MagicMock

    from millm.services.probe_runtime import ProbeRuntimeState

    ProbeRuntimeState.reset_for_tests()
    armed = MagicMock()
    armed.probe_id = probe_id
    ProbeRuntimeState()._armed[probe_id] = armed


class TestStatus:
    async def test_it_reports_the_armed_probes_with_their_language(self, ctx):
        repo, _events, service, probe = ctx
        await repo.update(probe, armed=True)
        _install_hook(probe.id)
        try:
            status = await service.status()
            assert status["armed_count"] == 1
            assert status["armed_rows"] == 1
            assert status["stale_armed"] == []
            assert status["armed"][0]["hook_installed"] is True
            assert status["armed"][0]["rung_language"] == "detects on unseen tasks"
            assert status["armed"][0]["next_step"]
        finally:
            from millm.services.probe_runtime import ProbeRuntimeState

            ProbeRuntimeState.reset_for_tests()

    async def test_a_paused_probe_says_WHY(self, ctx):
        """⚠ "A probe never goes silently quiet" applies to status above all. An armed probe
        listed with no further comment IS that silence."""
        repo, _events, service, probe = ctx
        await repo.update(probe, armed=True, paused_reason="speculative_decoding")
        _install_hook(probe.id)
        try:
            status = await service.status()
            assert status["armed"][0]["paused_reason"] == "speculative_decoding"
            assert status["paused_reasons"] == ["speculative_decoding"]
        finally:
            from millm.services.probe_runtime import ProbeRuntimeState

            ProbeRuntimeState.reset_for_tests()

    async def test_the_overhead_field_is_present_even_when_nothing_has_run(self, ctx):
        """So an operator can tell "nothing is wrong" from "this field is missing"."""
        _repo, _events, service, _probe = ctx
        status = await service.status()
        assert "last_request_overhead_ms" in status
        assert status["overhead_warn_threshold_ms"] == 5.0

    async def test_dropped_socket_events_are_reported_not_hidden(self, ctx):
        _repo, _events, service, probe = ctx
        service._ws_dropped = 7
        assert (await service.status())["socket_events_dropped"] == 7

    async def test_overhead_above_the_threshold_warns(self, ctx, caplog):
        _repo, _events, service, _probe = ctx
        service.note_request_overhead(99.0)
        assert service._last_request_overhead_ms == 99.0


class TestTheRoutesAreReachable:
    """⚠ A capability is not shipped until a test fails when its wiring is removed.

    This repo has already shipped 16 MCP tools that were fully implemented, unit-tested and
    documented while never registered — every test passed by importing the module directly.
    """

    def test_the_event_routes_are_in_the_LIVE_app(self):
        """⚠ **THE WHOLE SURFACE IS ASSERTED AS A SET IN
        `tests/unit/api/test_probe_route_surface.py`.** This assertion was once the only one, and it
        was a `for expected in (…): assert expected in paths` membership loop over seven literals.
        A subset check cannot fail for a route missing from both the code and the list, and five
        were: `arm`, `parity` and the three `hub` paths. The arming service, the identity gate, the
        parity engine and the k-sparse slice therefore had no production caller at all. What remains
        here is the two routes this file's own subject depends on.
        """
        from fastapi import FastAPI

        from millm.api.routes import register_routes

        app = FastAPI()
        register_routes(app)
        # ⚠ app.openapi()['paths'], NOT app.routes — this FastAPI version wraps included routers
        # in objects with no `.path`, so introspecting app.routes reports an empty app that
        # serves fine.
        paths = set(app.openapi()["paths"])
        assert "/api/probes/events" in paths
        assert "/api/probes/events/{event_id}" in paths

    def test_the_router_is_included_not_merely_importable(self):
        import inspect

        from millm.api.routes import register_routes

        source = inspect.getsource(register_routes)
        assert "app.include_router(probes_router)" in source
