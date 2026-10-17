"""Admission while the resident weights are written or torn (031 FTASKS 4.0; FTID §7.5).

Everything here runs the REAL `InferenceService._admit`, `hold_admission` and `RequestQueue` over
a TINY REAL Llama: the barrier pauses serving by taking the real slot, a torn set is refused
before a request queues, a request already queued is woken and refused (never served by the
barrier's eventual release), and the model's forward runs ZERO times while `writing` or `torn`
(FPRD §11 acceptance 10).

Also: the abort plumbing in `RequestQueue` (no permit leaked, `_pending` restored), the quiescing
gate (031 review D9), the barrier's long hold kept out of the queue's duration window (D18), CBM
routing while attached (M23), the request-cache invalidator, the `commits=` formula, and the
FR-31.17 regression (chat, SAE-steered chat and probe scoring answer the same attached as not).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import millm.services.weight_attachment as attachment_module
import millm.services.weight_barrier as barrier_module
from millm.api.schemas.openai import ChatCompletionRequest
from millm.core.config import settings
from millm.core.errors import WeightsTornError
from millm.core.weights_auth import WeightsIdentity
from millm.services import inference_service
from millm.services.request_queue import AdmissionAborted, RequestQueue
from millm.services.weight_attachment import WeightService
from millm.services.weight_barrier import WeightBarrier
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

TRAINER = WeightsIdentity("forge", "trainer")
D = "c" * 64


async def _settle(turns: int = 10) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


class _Attached(WeightService):
    def is_attached(self) -> bool:
        return True


@pytest.fixture
def no_sync(monkeypatch):
    monkeypatch.setattr(barrier_module, "_synchronize_resident_devices", lambda: None)


@pytest.fixture
def svc(no_sync):
    model = word_model()
    service = make_service(model, word_tokenizer())
    service.test_model = model
    yield service
    clear_loaded()


@pytest.fixture
def barrier(svc, monkeypatch):
    """The process-wide barrier, pausing THIS service through its real `hold_admission`."""
    monkeypatch.setattr(settings, "WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S", 1.0)
    b = WeightBarrier(inference=lambda: svc)
    barrier_module.set_weight_barrier(b)
    yield b
    b.reset("test teardown")
    barrier_module.set_weight_barrier(None)


@pytest.fixture
def attached():
    attachment_module.set_weight_service(_Attached())
    yield
    attachment_module.set_weight_service(None)


def _forwards(model) -> list[int]:
    calls: list[int] = []
    model.register_forward_hook(lambda *_: calls.append(1))
    return calls


def _chat(**over) -> ChatCompletionRequest:
    body = {"model": "tiny", "messages": [{"role": "user", "content": "w1 w2 w3"}],
            "max_tokens": 4, "temperature": 0.0}
    body.update(over)
    return ChatCompletionRequest(**body)


# --- RequestQueue abort (FTASKS 4.1) -------------------------------------------------------


class TestQueueAbort:
    async def test_an_aborted_waiter_holds_nothing_and_restores_pending(self):
        queue = RequestQueue(max_concurrent=1, max_pending=5)
        abort = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with queue.acquire():
                await release.wait()

        async def waiter():
            async with queue.acquire(abort=abort):
                pytest.fail("an aborted waiter must not get the slot")

        held = asyncio.create_task(holder())
        await _settle()
        waiting = asyncio.create_task(waiter())
        await _settle()
        assert queue.pending_count == 2
        abort.set()
        with pytest.raises(AdmissionAborted):
            await asyncio.wait_for(waiting, 1)
        assert queue.pending_count == 1
        release.set()
        await held
        assert queue.pending_count == 0 and queue._semaphore._value == 1  # no permit leaked

    async def test_an_abort_already_set_refuses_at_once(self):
        queue = RequestQueue(max_concurrent=1, max_pending=5)
        abort = asyncio.Event()
        abort.set()
        with pytest.raises(AdmissionAborted):
            async with queue.acquire(abort=abort):
                pytest.fail("must not enter")
        assert queue.pending_count == 0 and queue._semaphore._value == 1

    async def test_a_background_waiter_is_aborted(self):
        queue = RequestQueue(max_concurrent=1, max_pending=5)
        abort = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with queue.acquire():
                await release.wait()

        async def chunk():
            async with queue.acquire_background(abort=abort):
                pytest.fail("an aborted chunk must not get the slot")

        held = asyncio.create_task(holder())
        await _settle()
        waiting = asyncio.create_task(chunk())
        await _settle()
        assert queue.background_waiting_count == 1
        abort.set()
        with pytest.raises(AdmissionAborted):
            await asyncio.wait_for(waiting, 1)
        assert queue.background_waiting_count == 0
        release.set()
        await held
        assert queue.occupied_count == 0 and queue._semaphore._value == 1

    async def test_a_slot_won_in_the_same_turn_as_the_abort_is_handed_back(self):
        """The abort and the freed permit land together: the waiter is refused AND the permit
        it was handed goes back, or the semaphore would be one short for ever."""
        queue = RequestQueue(max_concurrent=1, max_pending=5)
        abort = asyncio.Event()

        async def waiter():
            async with queue.acquire(abort=abort):
                pytest.fail("an aborted waiter must not get the slot")

        async with queue.acquire():
            waiting = asyncio.create_task(waiter())
            await _settle()
            abort.set()  # ...and the slot is released on this block's exit, before any turn
        with pytest.raises(AdmissionAborted):
            await asyncio.wait_for(waiting, 1)
        assert queue.pending_count == 0 and queue._semaphore._value == 1
        async with asyncio.timeout(1):
            async with queue.acquire():
                pass

    async def test_a_cancelled_waiter_returns_no_permit(self):
        queue = RequestQueue(max_concurrent=1, max_pending=5)
        abort = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with queue.acquire():
                await release.wait()

        async def waiter():
            async with queue.acquire(abort=abort):
                pass

        held = asyncio.create_task(holder())
        await _settle()
        waiting = asyncio.create_task(waiter())
        await _settle()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        release.set()
        await held
        assert queue.pending_count == 0 and queue._semaphore._value == 1

    async def test_a_hold_with_record_false_is_not_a_duration(self):
        queue = RequestQueue(max_concurrent=1, max_pending=5)
        async with queue.acquire(record=False):
            pass
        assert len(queue._durations) == 0
        async with queue.acquire():
            pass
        assert len(queue._durations) == 1


# --- _admit (FTASKS 4.2, 4.6) --------------------------------------------------------------


class TestAdmission:
    async def test_a_request_during_torn_is_refused_before_it_queues(self, svc, barrier):
        """M1."""
        barrier.mark_torn("address_moved")
        calls = []
        real = svc._request_queue.acquire
        svc._request_queue.acquire = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        with pytest.raises(WeightsTornError):
            async with svc._admit():
                pytest.fail("a torn set must not be served")
        assert calls == []
        assert svc.request_queue.pending_count == 0

    async def test_a_queued_waiter_is_woken_and_refused_when_the_set_turns_torn(self, svc, barrier):
        """M3: queued behind the barrier's writing hold; the abort wakes it (no hang)."""
        await barrier.quiesce(D, TRAINER)
        barrier.commit_begin(D)
        ran: list[int] = []

        async def request():
            async with svc._admit():
                ran.append(1)

        waiting = asyncio.create_task(request())
        await _settle()
        assert svc.request_queue.pending_count == 2  # the holder and this waiter
        barrier.mark_torn("watchdog")
        with pytest.raises(WeightsTornError):
            await asyncio.wait_for(waiting, 1)
        assert ran == []
        assert barrier.holds_slot()  # the torn set keeps the slot (FR-31.11.3)

    async def test_a_waiter_that_gets_the_slot_after_torn_is_refused_inside_it(
        self, svc, barrier, monkeypatch
    ):
        """M2: the post-acquire check, with the abort taken away to isolate it."""
        monkeypatch.setattr(WeightBarrier, "abort_event", property(lambda self: asyncio.Event()))
        release = asyncio.Event()
        ran: list[str] = []

        async def in_flight():
            async with svc._admit():
                ran.append("in_flight")
                await release.wait()

        async def waiter():
            async with svc._admit():
                ran.append("waiter")

        first = asyncio.create_task(in_flight())
        await _settle()
        second = asyncio.create_task(waiter())
        await _settle()
        barrier.mark_torn("address_moved")
        release.set()
        await first
        with pytest.raises(WeightsTornError):
            await asyncio.wait_for(second, 1)
        assert ran == ["in_flight"]

    async def test_the_reentrant_branch_refuses_torn(self, svc, barrier):
        async with svc._admit():
            barrier.mark_torn("address_moved")
            with pytest.raises(WeightsTornError):
                async with svc._admit():
                    pytest.fail("a batch row inside its chunk must not run on a torn set")

    async def test_a_waiting_batch_chunk_is_refused(self, svc, barrier):
        release = asyncio.Event()

        async def in_flight():
            async with svc._admit():
                await release.wait()

        async def chunk():
            async with svc._admit(background=True):
                pytest.fail("a chunk must not run on a torn set")

        first = asyncio.create_task(in_flight())
        await _settle()
        waiting = asyncio.create_task(chunk())
        await _settle()
        barrier.mark_torn("address_moved")
        with pytest.raises(WeightsTornError):
            await asyncio.wait_for(waiting, 1)
        release.set()
        await first

    async def test_no_forward_runs_while_writing_or_torn(self, svc, barrier):
        """Acceptance 10: the forward is called zero times during `writing` and `torn`."""
        forwards = _forwards(svc.test_model)
        await barrier.quiesce(D, TRAINER)
        barrier.commit_begin(D)
        request = asyncio.create_task(svc.create_chat_completion(_chat()))
        await _settle(50)
        assert forwards == [] and not request.done()
        barrier.mark_torn("watchdog")
        with pytest.raises(WeightsTornError):
            await asyncio.wait_for(request, 2)
        with pytest.raises(WeightsTornError):
            await svc.create_chat_completion(_chat())
        assert forwards == []

    async def test_serving_resumes_after_a_late_commit_end(self, svc, barrier):
        forwards = _forwards(svc.test_model)
        await barrier.quiesce(D, TRAINER)
        barrier.commit_begin(D)
        barrier.mark_torn("watchdog")
        barrier.commit_end(D)
        barrier.resume(D)
        await _settle()
        out = await svc.create_chat_completion(_chat())
        assert out.choices[0].message.content is not None and forwards

    async def test_the_holder_task_passes_a_torn_set_to_recover_it(self, svc, barrier):
        """The holder takes the slot through `_admit` even while torn (the recovery path)."""
        barrier.mark_torn("address_moved")
        async with barrier.internal_write("weights_load", TRAINER) as handle:
            assert barrier.holds_slot()
            handle.mark_first_write()
        await _settle()
        assert barrier.snapshot().weights_state == "clean"
        async with svc._admit():
            pass

    async def test_a_request_that_began_before_quiesce_finishes_on_the_old_weights(
        self, svc, barrier
    ):
        """Acceptance 2, unit half: quiesce returns only after the in-flight request ends."""
        weights = {"version": 0}
        seen: list[tuple[str, int]] = []
        release = asyncio.Event()

        async def in_flight():
            async with svc._admit():
                await release.wait()
                seen.append(("request", weights["version"]))

        async def trainer():
            await barrier.quiesce(D, TRAINER)
            seen.append(("quiesced", weights["version"]))
            barrier.commit_begin(D)
            weights["version"] = 1
            barrier.commit_end(D)
            barrier.resume(D)

        request = asyncio.create_task(in_flight())
        await _settle()
        step = asyncio.create_task(trainer())
        await _settle()
        assert seen == []
        release.set()
        await asyncio.wait_for(asyncio.gather(request, step), 2)
        assert seen == [("request", 0), ("quiesced", 0)]

    async def test_new_arrivals_park_while_the_barrier_takes_the_slot(self, svc, barrier):
        """031 review D9: a newcomer during quiesce waits OUTSIDE the queue, behind the barrier."""
        release = asyncio.Event()
        order: list[str] = []

        async def in_flight():
            async with svc._admit():
                await release.wait()
                order.append("in_flight")

        async def newcomer():
            async with svc._admit():
                order.append("newcomer")

        async def trainer():
            await barrier.quiesce(D, TRAINER)
            order.append("quiesced")

        first = asyncio.create_task(in_flight())
        await _settle()
        step = asyncio.create_task(trainer())
        await _settle()
        late = asyncio.create_task(newcomer())
        await _settle()
        assert svc.request_queue.pending_count == 2, "the newcomer must park, not queue"
        release.set()
        await asyncio.wait_for(asyncio.gather(first, step), 2)
        await _settle()
        assert not late.done()
        barrier.resume(D)
        await asyncio.wait_for(late, 2)
        assert order == ["in_flight", "quiesced", "newcomer"]

    async def test_the_barriers_hold_is_not_a_request_duration(self, svc, barrier):
        """031 review D18: the QUEUE_FULL wait estimate reads request durations only."""
        await barrier.quiesce(D, TRAINER)
        barrier.resume(D)
        await _settle()
        assert len(svc.request_queue._durations) == 0
        async with svc._admit():
            pass
        assert len(svc.request_queue._durations) == 1


class TestStreams:
    async def test_a_stream_is_refused_before_its_200(self, svc, barrier):
        barrier.mark_torn("address_moved")
        with pytest.raises(WeightsTornError):
            svc.check_stream_admission(_chat(stream=True))

    async def test_a_committed_stream_ends_with_the_refusal_and_done(self, svc, barrier):
        forwards = _forwards(svc.test_model)
        barrier.mark_torn("address_moved")
        events = [e async for e in svc.stream_chat_completion(_chat(stream=True))]
        assert events[-1] == "data: [DONE]\n\n"
        error = json.loads(events[-2].removeprefix("data: "))["error"]
        assert (error["type"], error["code"]) == ("server_error", "weights_torn")
        assert error["retry_after"] == settings.RETRY_AFTER_WEIGHTS_TORN_S
        assert forwards == []


class TestContinuousBatchingRouting:
    def _with_cbm(self, svc):
        svc._cbm_backend = SimpleNamespace(
            is_running=True, sampling_params_match=lambda *_: True
        )
        svc._cbm_force_serial_monitoring = False

    def test_not_routed_to_while_attached(self, svc, attached):
        """M23."""
        self._with_cbm(svc)
        assert svc._use_cbm_for_request() is False

    def test_routed_to_when_nothing_is_attached(self, svc):
        attachment_module.set_weight_service(None)
        self._with_cbm(svc)
        assert svc._use_cbm_for_request() is True


class TestCacheInvalidation:
    def test_the_draft_state_invalidator_is_registered_at_construction(self, svc):
        assert svc._cache_invalidators == [svc._invalidate_draft_state]

    def test_it_drops_a_cache_left_on_a_model(self, svc):
        svc.test_model._cache = object()
        draft = SimpleNamespace(_cache=object())
        svc._draft_model = draft
        assert svc.invalidate_request_caches() == 1
        assert vars(svc.test_model)["_cache"] is None and draft._cache is None
        assert svc._draft_model is draft  # the draft's weights are kept

    async def test_quiesce_runs_the_services_invalidators_once(self, svc, barrier):
        ran: list[int] = []
        svc.register_cache_invalidator(lambda: ran.append(1))
        await barrier.quiesce(D, TRAINER)
        assert ran == [1]
        async with barrier.serving_pause("save", TRAINER):
            assert ran == [1, 1]
        barrier.resume(D)


class TestServedCommits:
    """Addendum §7 item 13: `commits = steps_since_load - commits_at_identity`."""

    def _commits(self, steps, identity):
        barrier = WeightBarrier()
        barrier._steps_since_load = steps
        barrier_module.set_weight_barrier(barrier)
        try:
            return inference_service._commits_since(identity)
        finally:
            barrier_module.set_weight_barrier(None)

    def test_counts_commits_since_the_identity(self):
        assert self._commits(5, SimpleNamespace(commits_at_identity=2)) == 3
        assert self._commits(5, None) == 5

    def test_an_identity_older_than_the_counter_is_not_recorded(self):
        assert self._commits(0, SimpleNamespace(commits_at_identity=3)) is None


# --- FR-31.17: existing endpoints keep working while attached (FTASKS 4.7) ------------------


class TestAttachedRegression:
    async def test_chat_and_sae_steered_chat_answer_the_same_attached(self, svc):
        from millm.services.sae_service import AttachedSAEState
        from tests.unit.f28_fixtures import SAE_A, attach, no_cuda

        AttachedSAEState().reset_for_tests()
        try:
            def run():
                return svc.create_chat_completion(_chat(max_tokens=6))

            def steered():
                return svc.create_chat_completion(_chat(max_tokens=6, steering={
                    "features": [{"index": 1, "strength": 30.0}]}))

            with no_cuda():
                plain_free = (await run()).choices[0].message.content
                served = attach(svc.test_model, [(SAE_A, 0, 11)])  # SAE attach
                steered_free = (await steered()).choices[0].message.content
                attachment_module.set_weight_service(_Attached())
                plain_attached = (await run()).choices[0].message.content
                steered_attached = (await steered()).choices[0].message.content
            assert steered_free != plain_free, "precondition: the steering changes the output"
            assert (plain_attached, steered_attached) == (plain_free, steered_free)
            for handle in served.handles:
                handle.remove()
        finally:
            attachment_module.set_weight_service(None)
            AttachedSAEState().reset_for_tests()

    async def test_probe_scoring_answers_the_same_attached(self, svc, test_session, monkeypatch):
        from millm.api.schemas.probe_scoring import ProbeScoreRequest
        from millm.db.repositories.probe_repository import ProbeRepository
        from millm.services.probe_runtime import ProbeRuntimeState
        from millm.services.probe_scoring import ProbeScoringService
        from millm.services.probe_service import ProbeService
        from tests.unit.score_fixtures import tiny_definition, tiny_identity

        ProbeRuntimeState.reset_for_tests()
        monkeypatch.setattr(
            "millm.services.probe_arm_bridge.loaded_identity",
            AsyncMock(return_value=(tiny_identity(), svc.test_model, svc._tokenizer)),
        )
        repo = ProbeRepository(test_session)
        await ProbeService(repo).import_definition(tiny_definition())
        request = ProbeScoreRequest(windows=["all"], inputs=[{"token_ids": [2, 6, 7, 8]}])
        try:
            free = await ProbeScoringService(repo, svc).score(request, test_session)
            attachment_module.set_weight_service(_Attached())
            held = await ProbeScoringService(repo, svc).score(request, test_session)
        finally:
            attachment_module.set_weight_service(None)
            ProbeRuntimeState.reset_for_tests()
        assert free["results"][0]["verdicts"][0]["score"] is not None
        assert held["results"] == free["results"]
