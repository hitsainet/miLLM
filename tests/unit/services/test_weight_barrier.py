"""The step barrier's public surface and its working `serving_pause` (addendum §2.1, W0a).

`serving_pause` is exercised twice: against a recording fake holder (refusals, ownership,
cleanup), and against the REAL `InferenceService.hold_admission()`, `_admit` and `RequestQueue`,
to prove the pause waits for in-flight work, holds new work out, and that a caller admitting
inside the block deadlocks (the contract rule). The holder side, internal writes and hold
ownership are in `test_weight_barrier_states.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import millm.services.weight_barrier as barrier_module
from millm.core.config import settings
from millm.core.errors import (
    ModelBusyError,
    PauseInProgressError,
    WeightsBarrierUnavailableError,
    WeightsNotCleanError,
)
from millm.core.weights_auth import WeightsIdentity
from millm.services.request_queue import QueueFullError
from millm.services.weight_barrier import (
    BarrierSnapshot,
    PauseGuard,
    WeightBarrier,
    WriteHandle,
    get_weight_barrier,
    set_weight_barrier,
)

WHO = WeightsIdentity("ops", "operator")


class FakeInference:
    """The three things the barrier reads, plus a one-slot admission it can hold."""

    def __init__(self, cbm: bool = False, max_concurrent: int = 1) -> None:
        self.request_queue = SimpleNamespace(max_concurrent=max_concurrent)
        self._cbm = cbm
        self.entered = 0
        self.slot = asyncio.Semaphore(1)
        self.invalidated = 0
        self.refuse: Exception | None = None

    def cbm_enabled(self) -> bool:
        return self._cbm

    def invalidate_request_caches(self) -> None:
        self.invalidated += 1

    @contextlib.asynccontextmanager
    async def hold_admission(self):
        self.entered += 1
        if self.refuse is not None:
            raise self.refuse
        async with self.slot:
            yield


@pytest.fixture
def synced(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(barrier_module, "_synchronize_resident_devices",
                        lambda: calls.append(1))
    return calls


def _barrier(inference) -> WeightBarrier:
    return WeightBarrier(inference=lambda: inference)


class TestInitialState:
    def test_starts_clean_and_serving(self):
        snap = WeightBarrier().snapshot()
        assert snap == BarrierSnapshot(
            weights_state="clean", barrier="serving", hold_owner=None, episode=0,
            steps_since_load=0, commit_overran=False, last_save=None,
            last_hold_timeout_at=None, last_overrun=None, held_model_id=None)
        assert WeightBarrier().holds_slot() is False

    def test_the_singleton_and_its_reset(self):
        set_weight_barrier(None)
        first = get_weight_barrier()
        assert get_weight_barrier() is first
        mine = WeightBarrier()
        set_weight_barrier(mine)
        assert get_weight_barrier() is mine
        set_weight_barrier(None)
        assert get_weight_barrier() is not mine


class TestServingPause:
    async def test_holds_the_slot_for_the_block_and_releases_it(self, synced):
        inference = FakeInference()
        barrier = _barrier(inference)
        async with barrier.serving_pause("adapter_load", WHO) as guard:
            assert guard == PauseGuard(weights_state="clean", steps_since_load=0, episode=1)
            assert barrier.holds_slot() is True
            assert inference.slot.locked()
            snap = barrier.snapshot()
            assert (snap.barrier, snap.hold_owner, snap.episode) == (
                "quiesced", "internal:adapter_load", 1)
        assert barrier.holds_slot() is False
        assert not inference.slot.locked()
        snap = barrier.snapshot()
        assert (snap.barrier, snap.hold_owner, snap.weights_state) == ("serving", None, "clean")
        assert inference.entered == 1
        assert inference.invalidated == 1 and synced == [1]

    async def test_episodes_increase(self, synced):
        barrier = _barrier(FakeInference())
        async with barrier.serving_pause("a", WHO) as first:
            pass
        async with barrier.serving_pause("b", WHO) as second:
            pass
        assert (first.episode, second.episode) == (1, 2)

    @pytest.mark.parametrize("inference, reason", [
        (FakeInference(cbm=True), "cbm"),
        (FakeInference(max_concurrent=2), "concurrency"),
    ])
    async def test_refuses_when_the_barrier_cannot_hold(self, inference, reason, synced):
        barrier = _barrier(inference)
        with pytest.raises(WeightsBarrierUnavailableError) as info:
            async with barrier.serving_pause("save", WHO):
                pytest.fail("the block must not run")
        assert info.value.details == {"reason": reason, "operation": "save"}
        assert inference.entered == 0
        assert barrier.snapshot().hold_owner is None

    async def test_refuses_unless_clean(self, synced):
        inference = FakeInference()
        barrier = _barrier(inference)
        barrier._weights_state = "torn"
        with pytest.raises(WeightsNotCleanError) as info:
            async with barrier.serving_pause("save", WHO):
                pytest.fail("the block must not run")
        assert info.value.details["weights_state"] == "torn"
        assert inference.entered == 0

    async def test_a_second_pause_is_refused_not_queued(self, synced):
        barrier = _barrier(FakeInference())
        async with barrier.serving_pause("save", WHO):
            with pytest.raises(PauseInProgressError) as info:
                async with barrier.serving_pause("adapter_load", WHO):
                    pytest.fail("the second block must not run")
            assert info.value.details == {"operation": "save",
                                          "refused_operation": "adapter_load"}
            assert barrier.holds_slot() is True
        assert barrier.holds_slot() is False

    async def test_a_refusal_from_admission_propagates_and_holds_nothing(self, synced):
        inference = FakeInference()
        inference.refuse = QueueFullError("Request queue full (5 pending).")
        barrier = _barrier(inference)
        with pytest.raises(QueueFullError):
            async with barrier.serving_pause("save", WHO):
                pytest.fail("the block must not run")
        assert barrier.holds_slot() is False
        assert barrier.snapshot().hold_owner is None

    async def test_a_drain_timeout_is_model_busy_and_leaves_serving(self, synced, monkeypatch):
        monkeypatch.setattr(settings, "WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S", 0.05)
        inference = FakeInference()
        barrier = _barrier(inference)
        await inference.slot.acquire()  # a request that never finishes
        try:
            with pytest.raises(ModelBusyError) as info:
                async with barrier.serving_pause("save", WHO):
                    pytest.fail("the block must not run")
        finally:
            inference.slot.release()
        assert info.value.details["reason"] == "drain_timeout"
        assert barrier.holds_slot() is False
        snap = barrier.snapshot()
        assert (snap.barrier, snap.hold_owner) == ("serving", None)
        # The cancelled holder never took the slot afterwards.
        await asyncio.sleep(0)
        assert not inference.slot.locked()

    async def test_an_error_in_the_block_releases_the_slot(self, synced):
        inference = FakeInference()
        barrier = _barrier(inference)
        with pytest.raises(RuntimeError, match="boom"):
            async with barrier.serving_pause("adapter_load", WHO):
                raise RuntimeError("boom")
        assert barrier.holds_slot() is False and not inference.slot.locked()
        assert barrier.snapshot().hold_owner is None


def _real_inference():
    """A real InferenceService: the barrier holds the slot through its REAL `hold_admission()`
    (W0b), not an injected stand-in."""
    from millm.services.inference_service import InferenceService

    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        svc = InferenceService(model_service=None)
    assert "hold_admission" not in vars(svc)
    return svc


class TestAgainstTheRealQueue:
    async def test_waits_for_in_flight_work_and_holds_new_work_out(self, synced):
        svc = _real_inference()
        barrier = _barrier(svc)
        order: list[str] = []
        release_request = asyncio.Event()

        async def in_flight():
            async with svc._admit():
                order.append("request_start")
                await release_request.wait()
                order.append("request_end")

        async def newcomer():
            async with svc._admit():
                order.append("newcomer")

        request = asyncio.create_task(in_flight())
        await asyncio.sleep(0.01)

        async def pause():
            async with barrier.serving_pause("adapter_load", WHO):
                order.append("pause_held")
                late = asyncio.create_task(newcomer())
                await asyncio.sleep(0.02)
                order.append("pause_end")
            await late

        pausing = asyncio.create_task(pause())
        await asyncio.sleep(0.02)
        assert "pause_held" not in order  # still draining the in-flight request
        release_request.set()
        await asyncio.wait_for(asyncio.gather(request, pausing), timeout=2)
        assert order == ["request_start", "request_end", "pause_held", "pause_end", "newcomer"]

    async def test_admitting_inside_the_block_deadlocks(self, synced):
        """The contract rule: inside a pause the slot is the barrier's, not the caller's."""
        svc = _real_inference()
        barrier = _barrier(svc)
        async with barrier.serving_pause("save", WHO):
            with pytest.raises(asyncio.TimeoutError):
                async with asyncio.timeout(0.1):
                    async with svc._admit():
                        pytest.fail("a caller must not get a slot inside a pause")
        # Afterwards the slot is free again.
        async with asyncio.timeout(1):
            async with svc._admit():
                pass


class TestCountersAndChecks:
    def test_require_clean(self):
        barrier = WeightBarrier()
        barrier.require_clean("save")
        barrier._weights_state = "writing"
        with pytest.raises(WeightsNotCleanError) as info:
            barrier.require_clean("save")
        assert info.value.details == {"operation": "save", "weights_state": "writing"}

    def test_reset_steps_has_a_default_reason(self):
        barrier = WeightBarrier()
        barrier._steps_since_load = 4
        barrier.reset_steps_since_load()
        assert barrier.snapshot().steps_since_load == 0

    def test_record_save(self):
        barrier = WeightBarrier()
        at = datetime(2026, 10, 14, tzinfo=timezone.utc)
        barrier.record_save("a" * 64, "run/step-000001", at)
        assert barrier.snapshot().last_save == {
            "checkpoint_sha256": "a" * 64, "path": "run/step-000001", "at": at.isoformat()}

    def test_parameter_set_replaced_cleans_unless_an_internal_write_is_active(self):
        barrier = WeightBarrier()
        barrier._weights_state, barrier._steps_since_load = "torn", 3
        barrier._internal_write_active = True
        barrier.on_parameter_set_replaced("model_load")
        assert barrier.snapshot().weights_state == "torn"
        barrier._internal_write_active = False
        barrier.on_parameter_set_replaced("model_load")
        snap = barrier.snapshot()
        assert (snap.weights_state, snap.steps_since_load) == ("clean", 0)

    def test_write_handle(self):
        handle = WriteHandle(3)
        assert handle.first_write_marked is False
        handle.mark_first_write()
        assert handle.first_write_marked is True


class TestFinalSignatures:
    """W1-A fills these in; their signatures are fixed by the addendum."""

    @pytest.mark.parametrize("name, params", [
        ("serving_pause", ["operation", "identity"]),
        ("internal_write", ["operation", "identity"]),
        ("holds_slot", []),
        ("require_clean", ["operation"]),
        ("reset_steps_since_load", ["reason"]),
        ("record_save", ["checkpoint_sha256", "path", "at"]),
        ("on_parameter_set_replaced", ["reason"]),
        ("snapshot", []),
        ("quiesce", ["owner_digest", "identity"]),
        ("resume", ["owner_digest"]),
        ("commit_begin", ["owner_digest"]),
        ("commit_end", ["owner_digest"]),
    ])
    def test_signature(self, name, params):
        sig = inspect.signature(getattr(WeightBarrier, name))
        assert list(sig.parameters)[1:] == params
