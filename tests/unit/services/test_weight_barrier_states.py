"""The weight barrier's state model, one test per row of 031 FTDD §4.1, plus hold ownership.

Driven by a FAKE monotonic clock: a timer is fired by calling the barrier's callback with the
generation it was armed under, after moving the clock past (or short of) its deadline. That is
what lets "the hold timer never fires while writing" be asserted with a clock already past the
deadline (FPRD FR-31.9.4). `TestRealLoopWiring` then proves the same callbacks are reached by
`loop.call_later` on a real loop (FTASKS 3.8).

Hold ownership (031 review D5, X6, D10; addendum §2.1): resume, detach and the hold timer act
only on holder-owned holds; an internal op adopts a holder's clean quiesce and hands it back; an
internal write inherits a torn hold; a write never overwrites a concurrent `torn` with `clean`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

import millm.services.weight_barrier as barrier_module
from millm.core.config import settings
from millm.core.errors import (
    AlreadyQuiescedError,
    ModelBusyError,
    NoOpenCommitError,
    NotAttachmentHolderError,
    NotQuiescedError,
    PauseInProgressError,
    WeightsBarrierUnavailableError,
    WeightsNotCleanError,
)
from millm.core.weights_auth import WeightsIdentity
from millm.services.request_queue import QueueFullError
from millm.services.weight_barrier import WeightBarrier
from tests.unit.services.test_weight_barrier import FakeInference

TRAINER = WeightsIdentity("forge", "trainer")
OPS = WeightsIdentity("ops", "operator")
D = "a" * 64  # the holder's attachment digest
OTHER = "b" * 64


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeWall:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def synced(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(barrier_module, "_synchronize_resident_devices", lambda: calls.append(1))
    return calls


@pytest.fixture
def env(synced, monkeypatch):
    # Fail fast: a defect that queues for a second slot must not wait the 60 s default drain.
    monkeypatch.setattr(settings, "WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S", 1.0)
    inference = FakeInference()
    inference.request_queue = SimpleNamespace(
        max_concurrent=1, pending_count=4, holding_count=1, background_waiting_count=2
    )
    clock = FakeClock()
    barrier = WeightBarrier(inference=lambda: inference, clock=clock, wall=FakeWall())
    yield SimpleNamespace(barrier=barrier, inference=inference, clock=clock, synced=synced)
    barrier.reset("test teardown")


def _state(barrier: WeightBarrier) -> tuple[str, str, object]:
    snap = barrier.snapshot()
    return snap.weights_state, snap.barrier, snap.hold_owner


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _quiesced(env) -> None:
    await env.barrier.quiesce(D, TRAINER)


async def _writing(env) -> None:
    await _quiesced(env)
    env.barrier.commit_begin(D)


def _fire_watchdog(env) -> None:
    env.clock.advance(settings.WEIGHTS_COMMIT_WATCHDOG_MS / 1000.0 + 0.001)
    env.barrier._on_watchdog(env.barrier._watchdog_generation)


async def _torn(env) -> None:
    await _writing(env)
    _fire_watchdog(env)


# --- the state table, row by row ---------------------------------------------------------


class TestStateTable:
    async def test_row_quiesce_from_clean_serving(self, env):
        result = await env.barrier.quiesce(D, TRAINER)
        assert _state(env.barrier) == ("clean", "quiesced", "holder")
        assert env.barrier.holds_slot() and env.inference.slot.locked()
        # Hold timer armed; caches invalidated exactly once (M22); the cards synchronized (D7).
        assert env.barrier._hold_deadline_mono == env.clock.now + settings.WEIGHTS_HOLD_TIMEOUT_S
        assert env.inference.invalidated == 1 and env.synced == [1]
        assert (result.held_requests, result.held_batch_chunks) == (3, 2)
        assert result.weights_state == "clean" and result.state_before == "clean"
        assert result.hold_deadline == (FakeWall().now + timedelta(
            seconds=settings.WEIGHTS_HOLD_TIMEOUT_S)).isoformat()

    async def test_row_resume_releases_the_slot(self, env):
        await _quiesced(env)
        result = env.barrier.resume(D)
        await _settle()
        assert result.released is True
        assert _state(env.barrier) == ("clean", "serving", None)
        assert not env.barrier.holds_slot() and not env.inference.slot.locked()

    async def test_row_hold_timer_self_resumes_only_once_due(self, env):
        await _quiesced(env)
        generation = env.barrier._hold_generation
        env.clock.advance(settings.WEIGHTS_HOLD_TIMEOUT_S - 0.5)
        env.barrier._on_hold_timer(generation)  # early: rescheduled, nothing released
        assert env.barrier.holds_slot()
        env.clock.advance(1.0)
        with capture_logs() as logs:
            env.barrier._on_hold_timer(generation)
        await _settle()
        assert _state(env.barrier) == ("clean", "serving", None)
        assert not env.inference.slot.locked()
        assert env.barrier.snapshot().last_hold_timeout_at == FakeWall().now.isoformat()
        assert [e["label"] for e in logs if e["event"] == "weights_hold_timeout"] == ["forge"]

    async def test_row_commit_begin_suspends_the_hold_timer_and_arms_the_watchdog(self, env):
        await _quiesced(env)
        hold_generation = env.barrier._hold_generation
        result = env.barrier.commit_begin(D)
        assert _state(env.barrier) == ("writing", "quiesced", "holder")
        assert result.watchdog_ms == settings.WEIGHTS_COMMIT_WATCHDOG_MS
        assert env.barrier._watchdog_deadline_mono == pytest.approx(
            env.clock.now + settings.WEIGHTS_COMMIT_WATCHDOG_MS / 1000.0)
        # M4: the clock past the hold deadline, the OLD timer fired, and the CURRENT one too.
        env.clock.advance(settings.WEIGHTS_HOLD_TIMEOUT_S + 1)
        env.barrier._on_hold_timer(hold_generation)
        env.barrier._on_hold_timer(env.barrier._hold_generation)
        assert _state(env.barrier)[0:2] == ("writing", "quiesced")
        assert env.barrier.holds_slot()

    async def test_the_hold_deadline_is_cleared_while_writing(self, env):
        """Status reports no hold deadline during a commit: the timer is suspended, not merely
        ignored when it fires (M4, first half)."""
        await _quiesced(env)
        assert env.barrier.snapshot().hold_deadline is not None
        env.barrier.commit_begin(D)
        assert env.barrier.snapshot().hold_deadline is None
        assert env.barrier._hold_handle is None

    @pytest.mark.parametrize("state", ["writing", "torn"])
    async def test_the_timer_callback_itself_refuses_outside_clean(self, env, state):
        """M4, second half: even a timer whose generation was never voided does not resume a
        set that is not clean (FR-31.9.4, the clock moved past the deadline)."""
        await _quiesced(env)
        generation = env.barrier._hold_generation
        env.barrier._weights_state = state  # a path that forgot to suspend the timer
        env.clock.advance(settings.WEIGHTS_HOLD_TIMEOUT_S + 1)
        env.barrier._on_hold_timer(generation)
        assert env.barrier.holds_slot()
        assert env.barrier.snapshot().barrier == "quiesced"
        env.barrier._weights_state = "clean"

    async def test_row_commit_end_counts_a_step_and_rearms_the_hold_timer(self, env):
        await _writing(env)
        env.clock.advance(0.25)
        result = env.barrier.commit_end(D)
        assert _state(env.barrier) == ("clean", "quiesced", "holder")
        assert (result.steps_since_load, result.overran, result.commit_ms) == (1, False, 250.0)
        assert env.barrier._hold_deadline_mono == env.clock.now + settings.WEIGHTS_HOLD_TIMEOUT_S
        # The watchdog is void: firing it with the clock long past changes nothing.
        env.clock.advance(60)
        env.barrier._on_watchdog(env.barrier._watchdog_generation)
        assert env.barrier.snapshot().weights_state == "clean"

    async def test_row_watchdog_marks_torn_keeps_the_slot_and_refuses(self, env):
        await _writing(env)
        assert env.barrier.admission_refusal() is None  # writing: waiters park, not refused
        env.clock.advance(settings.WEIGHTS_COMMIT_WATCHDOG_MS / 1000.0 - 0.1)
        env.barrier._on_watchdog(env.barrier._watchdog_generation)  # not yet due
        assert env.barrier.snapshot().weights_state == "writing"
        env.clock.advance(0.2)
        with capture_logs() as logs:
            env.barrier._on_watchdog(env.barrier._watchdog_generation)
        assert _state(env.barrier) == ("torn", "quiesced", "holder")
        assert env.barrier.snapshot().commit_overran is True
        assert env.barrier.holds_slot()  # FR-31.11.3: the slot is kept
        refusal = env.barrier.admission_refusal()
        assert refusal is not None and refusal.code == "WEIGHTS_TORN"
        assert env.barrier._abort_flag is True
        torn = [e for e in logs if e["event"] == "weights_torn"]
        assert len(torn) == 1
        assert (torn[0]["reason"], torn[0]["label"], torn[0]["role"]) == (
            "watchdog", "forge", "trainer")
        assert torn[0]["elapsed_ms"] == pytest.approx(
            settings.WEIGHTS_COMMIT_WATCHDOG_MS + 100.0, abs=0.5)
        assert torn[0]["attachment_ref"] == D[:8]

    async def test_a_watchdog_after_the_holder_is_cleared_names_the_committer(self, env):
        """R1 RX F3: clear-stale (`tear_writing=False`) forgets the holder; the watchdog's torn
        line must still name who began the commit."""
        await _writing(env)
        env.barrier.on_holder_detached(D, tear_writing=False)
        assert env.barrier._holder_identity is None
        with capture_logs() as logs:
            _fire_watchdog(env)
        torn = [e for e in logs if e["event"] == "weights_torn"]
        assert [(e["reason"], e["role"], e["label"]) for e in torn] == [
            ("watchdog", "trainer", "forge")]

    async def test_a_closed_commit_leaves_no_committer_behind(self, env):
        """The committer's identity ends with its commit: a later tear with no holder (here a
        direct `mark_torn`) must not name a trainer that is long gone."""
        await _writing(env)
        env.barrier.commit_end(D)
        env.barrier.resume(D)
        env.barrier.on_holder_detached(D)
        await _settle()
        with capture_logs() as logs:
            env.barrier.mark_torn("address_moved")
        torn = [e for e in logs if e["event"] == "weights_torn"]
        assert [(e["role"], e["label"]) for e in torn] == [(None, None)]

    async def test_row_hold_timer_never_fires_while_torn(self, env):
        await _torn(env)
        env.clock.advance(3600)
        env.barrier._on_hold_timer(env.barrier._hold_generation)
        assert _state(env.barrier) == ("torn", "quiesced", "holder")
        assert env.barrier.holds_slot()

    async def test_row_detach_while_writing_is_torn(self, env):
        await _writing(env)
        assert env.barrier.on_holder_detached(D) == "torn"
        assert env.barrier.snapshot().weights_state == "torn"
        assert env.barrier.holds_slot()

    async def test_row_late_commit_end_from_the_owner_recovers(self, env):
        await _torn(env)
        env.clock.advance(1.5)
        result = env.barrier.commit_end(D)
        assert _state(env.barrier) == ("clean", "quiesced", "holder")
        assert result.overran is True and result.state_before == "torn"
        assert result.steps_since_load == 1  # M8
        assert env.barrier.admission_refusal() is None and env.barrier._abort_flag is False
        snap = env.barrier.snapshot()
        assert snap.last_overrun == {"at": FakeWall().now.isoformat(), "commit_ms": result.commit_ms}
        assert env.barrier._hold_deadline_mono == env.clock.now + settings.WEIGHTS_HOLD_TIMEOUT_S

    async def test_row_internal_write_from_torn_ends_clean_serving(self, env):
        await _torn(env)
        env.barrier._steps_since_load = 7
        async with env.barrier.internal_write("weights_load", OPS) as handle:
            assert env.barrier.snapshot().hold_owner == "internal:weights_load"
            handle.mark_first_write()
            assert env.barrier.snapshot().weights_state == "writing"
        await _settle()
        assert _state(env.barrier) == ("clean", "serving", None)
        assert env.barrier.snapshot().steps_since_load == 0
        assert env.barrier.admission_refusal() is None
        assert not env.barrier.holds_slot() and not env.inference.slot.locked()

    async def test_row_feature1_reload_of_an_unattached_torn_set(self, env):
        await _torn(env)
        env.barrier.on_holder_detached(D)  # the trainer is gone; the slot is still held (D10)
        assert env.barrier.holds_slot()
        env.barrier.on_parameter_set_replaced("model_load")
        await _settle()
        assert _state(env.barrier) == ("clean", "serving", None)
        assert not env.inference.slot.locked()
        assert env.barrier.admission_refusal() is None

    async def test_row_address_moved_from_any_state(self, env):
        env.barrier.mark_torn("address_moved", parameter="lm_head.weight")
        assert _state(env.barrier) == ("torn", "serving", None)
        assert env.barrier.admission_refusal() is not None

    def test_row_process_start(self):
        barrier = WeightBarrier()
        barrier._weights_state = "torn"
        barrier._steps_since_load = 4
        barrier.reset("startup")
        snap = barrier.snapshot()
        assert (snap.weights_state, snap.barrier, snap.hold_owner, snap.steps_since_load) == (
            "clean", "serving", None, 0)


# --- refusals ------------------------------------------------------------------------------


class TestRefusals:
    async def test_quiesce_refused_unless_clean(self, env):
        env.barrier._weights_state = "torn"
        with pytest.raises(WeightsNotCleanError):
            await env.barrier.quiesce(D, TRAINER)
        assert env.inference.entered == 0

    async def test_quiesce_twice(self, env):
        await _quiesced(env)
        with pytest.raises(AlreadyQuiescedError):
            await env.barrier.quiesce(D, TRAINER)

    async def test_quiesce_drain_timeout(self, env, monkeypatch):
        monkeypatch.setattr(settings, "WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S", 0.05)
        await env.inference.slot.acquire()
        try:
            with pytest.raises(ModelBusyError) as info:
                await env.barrier.quiesce(D, TRAINER)
        finally:
            env.inference.slot.release()
        assert info.value.details["reason"] == "drain_timeout"
        assert _state(env.barrier) == ("clean", "serving", None)
        await _settle()
        assert not env.inference.slot.locked()

    async def test_quiesce_on_a_full_queue(self, env):
        env.inference.refuse = QueueFullError("Request queue full (5 pending).")
        with pytest.raises(QueueFullError):
            await env.barrier.quiesce(D, TRAINER)
        assert _state(env.barrier) == ("clean", "serving", None)

    @pytest.mark.parametrize("cbm, slots, reason", [(True, 1, "cbm"), (False, 2, "concurrency")])
    async def test_single_slot_and_no_cbm_are_required(self, synced, cbm, slots, reason):
        inference = FakeInference(cbm=cbm, max_concurrent=slots)
        barrier = WeightBarrier(inference=lambda: inference)
        with pytest.raises(WeightsBarrierUnavailableError) as info:
            await barrier.quiesce(D, TRAINER)
        assert info.value.details["reason"] == reason
        with pytest.raises(WeightsBarrierUnavailableError):
            async with barrier.internal_write("weights_load", OPS):
                pytest.fail("the block must not run")
        assert inference.entered == 0

    async def test_resume_while_serving_is_a_noop(self, env):
        result = env.barrier.resume(D)
        assert result.released is False and result.barrier == "serving"

    async def test_resume_refused_unless_clean(self, env):
        await _writing(env)
        with pytest.raises(WeightsNotCleanError):
            env.barrier.resume(D)

    async def test_resume_by_another_attachment(self, env):
        await _quiesced(env)
        with pytest.raises(NotAttachmentHolderError):
            env.barrier.resume(OTHER)
        assert env.barrier.holds_slot()

    async def test_resume_while_quiesce_drains_is_refused(self, env):
        """Addendum D-39: a resume sent while the holder's quiesce is still draining is refused
        `NOT_QUIESCED`. A no-op success would let the quiesce complete afterwards and hold
        serving for the full hold timeout while the holder believes it resumed."""
        await env.inference.slot.acquire()  # in-flight work the quiesce must drain
        task = asyncio.ensure_future(env.barrier.quiesce(D, TRAINER))
        await _settle()
        assert env.barrier._quiescing
        with pytest.raises(NotQuiescedError) as info:
            env.barrier.resume(D)
        assert info.value.details == {
            "operation": "resume", "barrier": "serving", "reason": "quiesce_in_progress",
        }
        env.inference.slot.release()
        await task
        assert _state(env.barrier) == ("clean", "quiesced", "holder")
        # Once quiesce has returned, the same resume releases the slot.
        assert env.barrier.resume(D).released is True
        await _settle()
        assert _state(env.barrier) == ("clean", "serving", None)

    async def test_commit_begin_while_serving(self, env):
        with pytest.raises(NotQuiescedError):
            env.barrier.commit_begin(D)

    async def test_commit_begin_while_writing_or_torn(self, env):
        await _writing(env)
        with pytest.raises(WeightsNotCleanError):
            env.barrier.commit_begin(D)
        _fire_watchdog(env)
        with pytest.raises(WeightsNotCleanError):
            env.barrier.commit_begin(D)

    async def test_commit_begin_by_another_attachment(self, env):
        await _quiesced(env)
        with pytest.raises(NotAttachmentHolderError):
            env.barrier.commit_begin(OTHER)

    async def test_commit_end_with_no_open_commit(self, env):
        await _quiesced(env)
        with pytest.raises(NoOpenCommitError):
            env.barrier.commit_end(D)

    async def test_a_foreign_commit_end_is_refused_writing_and_torn(self, env):
        """M6: only the attachment that began the commit may end it (FR-31.12.2)."""
        await _writing(env)
        with pytest.raises(NotAttachmentHolderError):
            env.barrier.commit_end(OTHER)
        assert env.barrier.snapshot().weights_state == "writing"
        _fire_watchdog(env)
        with pytest.raises(NotAttachmentHolderError):
            env.barrier.commit_end(OTHER)
        assert env.barrier.snapshot().weights_state == "torn"

    async def test_torn_from_address_moved_has_no_commit_to_end(self, env):
        await _quiesced(env)
        env.barrier.mark_torn("address_moved")
        with pytest.raises(NoOpenCommitError):
            env.barrier.commit_end(D)


class TestCommitRacesTheWatchdog:
    async def test_commit_end_first_then_the_watchdog_is_void(self, env):
        await _writing(env)
        generation = env.barrier._watchdog_generation
        env.clock.advance(10)
        result = env.barrier.commit_end(D)
        env.barrier._on_watchdog(generation)
        assert result.overran is False
        assert env.barrier.snapshot().weights_state == "clean"

    async def test_watchdog_first_then_commit_end_is_an_overrun(self, env):
        await _writing(env)
        _fire_watchdog(env)
        result = env.barrier.commit_end(D)
        assert result.overran is True
        assert env.barrier.snapshot().weights_state == "clean"


# --- internal writes and pauses ------------------------------------------------------------


class TestInternalWrite:
    async def test_refused_during_an_open_commit(self, env):
        await _writing(env)
        with pytest.raises(WeightsNotCleanError):
            async with env.barrier.internal_write("weights_load", OPS):
                pytest.fail("the block must not run")
        assert _state(env.barrier) == ("writing", "quiesced", "holder")

    async def test_a_failure_before_the_first_write_changes_nothing(self, env):
        """3A.2: a refusal raised after the drain and before `mark_first_write()`."""
        with pytest.raises(RuntimeError, match="refused"):
            async with env.barrier.internal_write("weights_load", OPS):
                raise RuntimeError("refused")
        assert _state(env.barrier) == ("clean", "serving", None)
        assert env.barrier.admission_refusal() is None

    async def test_a_failure_after_the_first_write_is_torn(self, env):
        env.barrier._steps_since_load = 3
        with pytest.raises(RuntimeError, match="disk"):
            async with env.barrier.internal_write("weights_load", OPS) as handle:
                handle.mark_first_write()
                raise RuntimeError("disk")
        await _settle()
        assert _state(env.barrier) == ("torn", "serving", None)
        assert env.barrier.admission_refusal() is not None
        assert not env.inference.slot.locked()  # it releases the slot it took

    async def test_success_resets_steps_and_takes_and_releases_its_own_slot(self, env):
        env.barrier._steps_since_load = 3
        async with env.barrier.internal_write("weights_load", OPS) as handle:
            assert env.barrier.holds_slot()
            assert env.inference.invalidated == 1 and env.synced == [1]
            handle.mark_first_write()
        await _settle()
        assert _state(env.barrier) == ("clean", "serving", None)
        assert env.barrier.snapshot().steps_since_load == 0
        assert not env.inference.slot.locked()

    async def test_success_without_a_write_changes_nothing(self, env):
        env.barrier._steps_since_load = 3
        async with env.barrier.internal_write("weights_load", OPS):
            pass
        assert env.barrier.snapshot().steps_since_load == 3

    async def test_a_concurrent_torn_is_never_overwritten_with_clean(self, env):
        async with env.barrier.internal_write("weights_load", OPS) as handle:
            handle.mark_first_write()
            env.barrier.mark_torn("address_moved")
        assert env.barrier.snapshot().weights_state == "torn"

    async def test_parameter_set_replaced_is_a_noop_inside_it(self, env):
        """X5: a full reload inside the write fires the Feature-1 hook; the write transitions."""
        await _torn(env)
        async with env.barrier.internal_write("weights_load", OPS) as handle:
            handle.mark_first_write()
            env.barrier.on_parameter_set_replaced("model_load")
            assert _state(env.barrier) == ("writing", "quiesced", "internal:weights_load")
            assert env.barrier.holds_slot()
        assert env.barrier.snapshot().weights_state == "clean"

    async def test_holds_slot_while_a_torn_hold_has_no_holder(self, env):
        """X6: the barrier holds the slot from a torn episode whose holder detached."""
        await _torn(env)
        env.barrier.on_holder_detached(D)
        assert env.barrier.holds_slot()
        async with env.barrier.internal_write("weights_load", OPS) as handle:
            assert env.barrier.holds_slot()
            assert env.inference.entered == 1  # adopted; never queued for a second slot
            handle.mark_first_write()
        await _settle()
        assert not env.barrier.holds_slot()

    async def test_an_inherited_torn_hold_survives_an_early_failure(self, env):
        await _torn(env)
        with pytest.raises(RuntimeError):
            async with env.barrier.internal_write("weights_load", OPS):
                raise RuntimeError("refused before the first write")
        assert _state(env.barrier) == ("torn", "quiesced", "holder")
        assert env.barrier.holds_slot()
        # The commit is still the holder's, so its late commit-end still recovers.
        assert env.barrier.commit_end(D).overran is True

    async def test_a_late_commit_end_cannot_clear_a_set_an_internal_write_tore(self, env):
        await _torn(env)
        with pytest.raises(RuntimeError):
            async with env.barrier.internal_write("weights_load", OPS) as handle:
                handle.mark_first_write()
                raise RuntimeError("half-loaded")
        with pytest.raises(NoOpenCommitError):
            env.barrier.commit_end(D)
        assert env.barrier.snapshot().weights_state == "torn"

    async def test_adopting_a_clean_quiesce_ends_clean_quiesced(self, env):
        """I8: a write that reused the holder's clean quiesce hands it back."""
        await _quiesced(env)
        async with env.barrier.internal_write("weights_load", OPS) as handle:
            handle.mark_first_write()
        assert _state(env.barrier) == ("clean", "quiesced", "holder")
        assert env.barrier.holds_slot()
        assert env.barrier._hold_deadline_mono == env.clock.now + settings.WEIGHTS_HOLD_TIMEOUT_S

    async def test_mark_first_write_outside_its_block_is_refused(self, env):
        async with env.barrier.internal_write("weights_load", OPS) as handle:
            pass
        with pytest.raises(RuntimeError):
            handle._on_first_write(handle)


class TestServingPause:
    async def test_never_sets_writing(self, env):
        async with env.barrier.serving_pause("save", OPS):
            assert env.barrier.snapshot().weights_state == "clean"
        assert env.barrier.snapshot().weights_state == "clean"

    async def test_adopts_the_holders_quiesce_and_blocks_commit_begin(self, env):
        """3A.1: PAUSE_IN_PROGRESS while the pause holds, including the holder's own quiesce."""
        await _quiesced(env)
        async with env.barrier.serving_pause("save", OPS):
            assert env.barrier.snapshot().hold_owner == "internal:save"
            assert env.inference.entered == 1  # adopted, no second slot
            assert env.inference.invalidated == 2 and env.synced == [1, 1]
            with pytest.raises(PauseInProgressError) as info:
                env.barrier.commit_begin(D)
            assert info.value.details == {"operation": "save", "refused_operation": "commit_begin"}
        assert _state(env.barrier) == ("clean", "quiesced", "holder")
        env.barrier.commit_begin(D)  # after the snapshot it succeeds
        assert env.barrier.snapshot().weights_state == "writing"

    async def test_commit_begin_refused_during_a_pause_with_no_holder(self, env):
        async with env.barrier.serving_pause("adapter_load", OPS):
            with pytest.raises(PauseInProgressError):
                env.barrier.commit_begin(D)

    async def test_pause_refused_while_writing(self, env):
        await _writing(env)
        with pytest.raises(WeightsNotCleanError):
            async with env.barrier.serving_pause("save", OPS):
                pytest.fail("the block must not run")
        assert _state(env.barrier) == ("writing", "quiesced", "holder")

    async def test_a_quiesce_is_refused_during_a_pause(self, env):
        async with env.barrier.serving_pause("save", OPS):
            with pytest.raises(PauseInProgressError):
                await env.barrier.quiesce(D, TRAINER)


class TestHoldOwnership:
    async def test_resume_cannot_release_an_internal_pause(self, env):
        await _quiesced(env)
        async with env.barrier.serving_pause("save", OPS):
            with pytest.raises(PauseInProgressError):
                env.barrier.resume(D)
            assert env.barrier.holds_slot()
            assert env.barrier.snapshot().hold_owner == "internal:save"

    async def test_resume_without_a_holder_hold_leaves_a_pause_alone(self, env):
        async with env.barrier.serving_pause("adapter_load", OPS):
            assert env.barrier.resume(D).released is False
            assert env.barrier.holds_slot()

    async def test_hold_timer_cannot_fire_mid_internal_op(self, env):
        await _quiesced(env)
        armed = env.barrier._hold_generation
        async with env.barrier.serving_pause("save", OPS):
            env.clock.advance(settings.WEIGHTS_HOLD_TIMEOUT_S * 10)
            env.barrier._on_hold_timer(armed)
            env.barrier._on_hold_timer(env.barrier._hold_generation)
            assert env.barrier.holds_slot()
            assert env.barrier.snapshot().hold_owner == "internal:save"
        # Re-armed in full on exit: still held until a NEW deadline passes.
        assert env.barrier._hold_deadline_mono == env.clock.now + settings.WEIGHTS_HOLD_TIMEOUT_S
        env.barrier._on_hold_timer(env.barrier._hold_generation)
        assert env.barrier.holds_slot()

    async def test_an_adopted_hold_suspends_the_timer(self, env):
        await _quiesced(env)
        async with env.barrier.serving_pause("save", OPS):
            assert env.barrier.snapshot().hold_deadline is None
            assert env.barrier._hold_handle is None
        assert env.barrier.snapshot().hold_deadline is not None

    async def test_the_timer_callback_itself_ignores_an_internal_hold(self, env):
        """Even an un-voided generation cannot release a hold an internal op owns."""
        await _quiesced(env)
        generation = env.barrier._hold_generation
        env.barrier._hold_owner = "internal:save"  # a path that forgot to suspend the timer
        env.clock.advance(settings.WEIGHTS_HOLD_TIMEOUT_S + 1)
        env.barrier._on_hold_timer(generation)
        assert env.barrier.holds_slot()
        assert env.barrier.snapshot().hold_owner == "internal:save"
        env.barrier._hold_owner = "holder"

    async def test_a_late_commit_end_cannot_land_inside_an_internal_write(self, env):
        """The torn hold is adopted by a load; the old commit's late end must wait for it."""
        await _torn(env)
        async with env.barrier.internal_write("weights_load", OPS):
            with pytest.raises(PauseInProgressError):
                env.barrier.commit_end(D)
            assert env.barrier.snapshot().weights_state == "torn"
            assert env.barrier.snapshot().steps_since_load == 0

    async def test_detach_during_an_adopted_pause_releases_at_its_end(self, env):
        await _quiesced(env)
        async with env.barrier.serving_pause("save", OPS):
            assert env.barrier.on_holder_detached(D) == "clean"
            assert env.barrier.holds_slot()  # the pause keeps it
        await _settle()
        assert _state(env.barrier) == ("clean", "serving", None)
        assert not env.inference.slot.locked()

    async def test_parameter_set_replaced_never_releases_an_internal_pause(self, env):
        async with env.barrier.serving_pause("adapter_load", OPS):
            env.barrier.on_parameter_set_replaced("model_load")
            assert env.barrier.holds_slot()
            assert env.barrier.snapshot().hold_owner == "internal:adapter_load"

    async def test_a_second_internal_op_is_refused(self, env):
        async with env.barrier.internal_write("weights_load", OPS):
            with pytest.raises(PauseInProgressError):
                async with env.barrier.serving_pause("save", OPS):
                    pytest.fail("the block must not run")


class TestSynchronize:
    def test_synchronizes_each_card_of_a_transformers_model_once(self):
        from millm.ml.model_loader import ENGINE_TRANSFORMERS

        loaded = SimpleNamespace(engine=ENGINE_TRANSFORMERS, gpu_indices=[1, 0, 1])
        with patch("millm.ml.model_loader.LoadedModelState") as state, \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.synchronize") as sync:
            state.return_value.current = loaded
            barrier_module._synchronize_resident_devices()
        assert [c.args for c in sync.call_args_list] == [(0,), (1,)]

    def test_nothing_resident_synchronizes_nothing(self):
        with patch("millm.ml.model_loader.LoadedModelState") as state, \
                patch("torch.cuda.synchronize") as sync:
            state.return_value.current = None
            barrier_module._synchronize_resident_devices()
        assert sync.call_count == 0

    async def test_quiesce_synchronizes_before_it_returns(self, env):
        order: list[str] = []

        async def after(inference):
            order.append("synchronized")

        env.barrier._after_hold = after
        await env.barrier.quiesce(D, TRAINER)
        order.append("returned")
        assert order == ["synchronized", "returned"]


class TestRealLoopWiring:
    """FTASKS 3.8: the timers are reached through `loop.call_later` on a real loop."""

    async def test_the_watchdog_fires_within_50_ms_of_its_deadline(self, synced, monkeypatch):
        monkeypatch.setattr(settings, "WEIGHTS_COMMIT_WATCHDOG_MS", 50)
        inference = FakeInference()
        barrier = WeightBarrier(inference=lambda: inference)
        loop = asyncio.get_running_loop()
        await barrier.quiesce(D, TRAINER)
        began = loop.time()
        barrier.commit_begin(D)
        async with asyncio.timeout(2):
            while barrier.snapshot().weights_state != "torn":
                await asyncio.sleep(0.002)
        elapsed = loop.time() - began
        assert 0.05 <= elapsed <= 0.10, elapsed
        barrier.reset("test teardown")
        await _settle()

    async def test_the_hold_timer_releases_the_slot(self, synced, monkeypatch):
        monkeypatch.setattr(settings, "WEIGHTS_HOLD_TIMEOUT_S", 0.05)
        inference = FakeInference()
        barrier = WeightBarrier(inference=lambda: inference)
        await barrier.quiesce(D, TRAINER)
        async with asyncio.timeout(2):
            while inference.slot.locked():
                await asyncio.sleep(0.005)
        assert barrier.snapshot().barrier == "serving"
        assert barrier.snapshot().last_hold_timeout_at is not None

    async def test_a_release_from_another_thread_reaches_the_loop(self, synced):
        """A Feature-1 load completes in a worker thread: the release is handed to the loop."""
        inference = FakeInference()
        barrier = WeightBarrier(inference=lambda: inference)
        await barrier.quiesce(D, TRAINER)
        barrier.commit_begin(D)
        barrier.mark_torn("address_moved")
        barrier.on_holder_detached(D)
        await asyncio.to_thread(barrier.on_parameter_set_replaced, "model_load")
        async with asyncio.timeout(2):
            while inference.slot.locked():
                await asyncio.sleep(0.005)
        assert barrier.snapshot().weights_state == "clean"


async def test_a_failed_internal_write_over_an_adopted_quiesce_names_its_own_writer(env):
    """Review R2-A: a torn line says who was writing (R-01.22; FX-A's committer fallback). An
    internal write that adopted the holder's clean quiesce and failed after its first write was
    logged as the TRAINER's tear, because the adopted hold leaves `_holder_identity` set; the
    writer was the caller of `internal_write`.

    MUTATION CONTROL: `writer=identity` dropped from the internal write's `mark_torn` -> red."""
    await _quiesced(env)
    with capture_logs() as logs:
        with pytest.raises(RuntimeError, match="disk"):
            async with env.barrier.internal_write("weights_load", OPS) as handle:
                handle.mark_first_write()
                raise RuntimeError("disk")
    torn = [e for e in logs if e["event"] == "weights_torn"]
    assert [(e["reason"], e["role"], e["label"]) for e in torn] == [
        ("internal_write_failed", "operator", "ops")]
    assert env.barrier.snapshot().weights_state == "torn"
