"""The probe's own cost is MEASURED and REPORTED, not merely measurable.

⚠ Three things in this path were declared and wired to nothing, and the third is why the first
two went unnoticed for the whole feature:

1. `ProbeRequestContext.overhead_ms` was initialised to `0.0` and **written by nothing**.
2. `ProbeEventService.note_request_overhead` existed, warned above the threshold, and had **no
   production caller** — `_probe_record` called `record()` without `overhead_ms`.
3. So `GET /api/probes/status` reported `last_request_overhead_ms: null` on every request ever
   served, the above-threshold warning could never fire, and **SC-4 was unmeasurable from the
   product itself**.

Every piece had a unit test. `test_probe_events.py` calls `note_request_overhead(99.0)` directly
and asserts the warning; that test passes forever whether or not anything calls it. The gap was
found on hardware, by trying to read the number after scoring real traffic and getting `None`.

So these tests assert the CHAIN: the context accumulates, `record()` forwards, and the status
reports. None of them would pass against any of the three defects.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext


def _probe(probe_id: str = "pr_1", d_model: int = 8) -> ArmedProbe:
    return ArmedProbe(
        probe_id=probe_id,
        name="t",
        head=ProbeHead(weight=torch.randn(d_model)),
        rule="mean",
        scope="all",
        layer=1,
        rung=2,
        rung_language="detects on unseen tasks",
    )


class TestTheContextAccumulatesItsOwnCost:
    def test_observing_costs_measurable_time(self):
        context = ProbeRequestContext("req", [_probe()])
        assert context.overhead_ms == 0.0
        context.observe(1, torch.randn(1, 256, 8))
        assert context.overhead_ms > 0.0, (
            "scoring 256 tokens registered ZERO overhead — the field is initialised and never "
            "written, which is how it shipped"
        )

    def test_finishing_adds_the_rules_cost(self):
        context = ProbeRequestContext("req", [_probe()])
        context.observe(1, torch.randn(1, 64, 8))
        after_observe = context.overhead_ms
        context.finish()
        assert context.overhead_ms > after_observe

    def test_more_work_costs_more(self):
        """Specificity: a constant would satisfy every assertion above.

        ⚠ MEDIAN OF SEVERAL, AND A WIDE RATIO, because this test used to pass for the wrong
        reason. At `d_model=8` it compared a single wall-clock sample of 32 tokens against one of
        4096, and what made the larger one reliably bigger was the per-token PYTHON work — a
        4096-element `.tolist()` and a 4096-element mask list. Once those were replaced by
        tensors, `observe` at this width is dominated by fixed per-call dispatch, the two sizes
        came within noise of each other, and the test failed on correct code (0.0257 ms for 4096
        tokens against 0.0329 ms for 32).

        The property is still true and still worth asserting — the arithmetic is linear in tokens
        — but it has to be measured where the arithmetic is visible above the call overhead, and
        with more than one sample. `d_model=256` and a 256x token ratio put the work two orders
        above the dispatch cost.
        """
        wide = _probe(d_model=256)

        def median_ms(n_tokens: int) -> float:
            samples = []
            for _ in range(7):
                context = ProbeRequestContext("req", [wide])
                context.observe(1, torch.randn(1, n_tokens, 256))
                samples.append(context.overhead_ms)
            samples.sort()
            return samples[len(samples) // 2]

        small = median_ms(32)
        large = median_ms(8192)
        assert large > small, (
            f"8192 tokens cost {large:.4f} ms against 32 tokens' {small:.4f} ms — the reported "
            "overhead does not grow with the work, so it is not measuring the work"
        )

    def test_a_pass_with_no_probe_on_that_layer_is_not_charged(self):
        """The probe did no work, so it costs nothing — otherwise the number measures the model."""
        context = ProbeRequestContext("req", [_probe()])
        context.observe(7, torch.randn(1, 512, 8))  # no probe on layer 7
        assert context.overhead_ms == 0.0


class TestTheMeasurementReachesTheStatus:
    @pytest.mark.asyncio
    async def test_record_forwards_the_overhead(self):
        """⚠ The wiring, asserted with its PAYLOAD.

        `record()` being called is not enough: it was being called, without `overhead_ms`.
        """
        from millm.services.inference_service import InferenceService

        context = ProbeRequestContext("req", [_probe()])
        context.observe(1, torch.randn(1, 128, 8))
        measured = context.overhead_ms
        assert measured > 0

        service = MagicMock()
        service.record = AsyncMock()

        import millm.api.dependencies as deps

        original = getattr(deps, "_probe_event_service", None)
        deps._probe_event_service = service
        try:
            await InferenceService._probe_record(MagicMock(), context)
        finally:
            deps._probe_event_service = original

        service.record.assert_awaited_once()
        kwargs = service.record.await_args.kwargs
        assert "overhead_ms" in kwargs, (
            "record() was called WITHOUT overhead_ms — note_request_overhead then never runs and "
            "the status reports null forever"
        )
        # ⚠ `>=`, not `==`: `_probe_record` calls `finish()`, whose rule evaluation is ALSO
        # probe work and is correctly charged. My first version asserted equality and failed
        # against correct code — the forwarded number was larger because it included finish.
        assert kwargs["overhead_ms"] >= measured
        assert kwargs["overhead_ms"] > 0

    @pytest.mark.asyncio
    async def test_the_service_stores_what_it_is_given_and_status_reports_it(self):
        from millm.services.probe_event_service import ProbeEventService

        probes = MagicMock()
        probes.list = AsyncMock(return_value=[])
        probes.list_armed = AsyncMock(return_value=[])
        events = MagicMock()
        events.create_many = AsyncMock(return_value=[])
        events.count = AsyncMock(return_value=0)
        events.prune = AsyncMock(return_value=0)

        service = ProbeEventService(probes, events)
        assert (await service.status())["last_request_overhead_ms"] is None

        service.note_request_overhead(3.25)
        assert (await service.status())["last_request_overhead_ms"] == 3.25


class FakeCudaTensor(torch.Tensor):
    """A CPU tensor that claims to be on CUDA, so the CUDA branch runs without a GPU."""

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, False)

    @property
    def is_cuda(self):  # type: ignore[override]
        return True


class _FakeEvent:
    """Stands in for `torch.cuda.Event`: records when it is recorded, synchronised and read."""

    log: list[tuple[str, int]] = []
    count = 0

    def __init__(self, enable_timing: bool = False) -> None:
        assert enable_timing, "an event without timing cannot be read"
        type(self).count += 1
        self.id = type(self).count

    def record(self, stream=None) -> None:
        self.log.append(("record", self.id))

    def synchronize(self) -> None:
        self.log.append(("synchronize", self.id))

    def elapsed_time(self, other: "_FakeEvent") -> float:
        self.log.append(("elapsed", self.id))
        return 7.5   # GPU ms between the pair: larger than any host time in a unit test


@pytest.fixture
def fake_cuda(monkeypatch):
    import millm.services.probe_runtime as runtime

    _FakeEvent.log = []
    _FakeEvent.count = 0
    syncs: list[object] = []
    monkeypatch.setattr(runtime.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(runtime.torch.cuda, "current_stream", lambda device=None: object())
    monkeypatch.setattr(runtime.torch.cuda, "synchronize", lambda device=None: syncs.append(device))
    return syncs


class TestTheMeasurementDoesNotStallTheForward:
    """024 10.4 (2026-10-17): the timer no longer stalls the pass it times.

    History, both measured on the RTX 3090. A bare host clock charged the model's in-flight
    kernels to the probe (116 ms reported, 11.2 ms real). The fix synchronised before and after
    every pass — honest, and the stall it imposed was ~99% of the probe's cost: the same arithmetic
    added 0.204 ms per token with the two syncs and 0.002 ms without (LFM2.5-1.2B, L11).

    Now each CUDA pass records a start and an end `torch.cuda.Event` on the stream (bracketing
    exactly the probe's kernels in stream order) and the host enqueue time; nothing waits until
    `overhead_ms` is read, after the request's forwards are done.
    """

    def test_no_synchronize_on_the_hot_path(self, fake_cuda):
        context = ProbeRequestContext("req", [_probe()])
        for _ in range(3):
            context.observe(1, FakeCudaTensor(torch.randn(1, 4, 8)))
        assert fake_cuda == [], (
            "observe() synchronised the device — that stalls the forward pass it is timing "
            "(measured 0.204 vs 0.002 ms per token on the 3090)"
        )
        assert not any(kind in ("synchronize", "elapsed") for kind, _ in _FakeEvent.log), (
            "an event was waited on or read inside observe(); that is a host stall too"
        )
        assert [kind for kind, _ in _FakeEvent.log] == ["record"] * 6, "two events per pass"

    def test_timing_values_are_still_produced_when_read(self, fake_cuda):
        context = ProbeRequestContext("req", [_probe()])
        for _ in range(3):
            context.observe(1, FakeCudaTensor(torch.randn(1, 4, 8)))
        context.finish()
        overhead = context.overhead_ms
        # per pass max(host, GPU) = the fake's 7.5 ms GPU time, three passes, plus finish()
        assert 22.5 <= overhead < 22.5 + 50, overhead
        assert context.n_passes == 3
        waited = [i for kind, i in _FakeEvent.log if kind == "synchronize"]
        assert waited == [2, 4, 6], "each pass's END event is resolved, once, when read"
        assert context.overhead_ms == overhead, "resolving twice must not count twice"

    def test_the_host_time_wins_when_it_is_larger(self, fake_cuda, monkeypatch):
        """A launch-bound pass pays its host time; the GPU time must not hide it."""
        monkeypatch.setattr(_FakeEvent, "elapsed_time", lambda self, other: 0.0)
        context = ProbeRequestContext("req", [_probe()])
        context.observe(1, FakeCudaTensor(torch.randn(1, 4, 8)))
        assert context.overhead_ms > 0.0


class TestCpu:
    def test_a_cpu_tensor_does_not_synchronise(self):
        """Specificity: syncing unconditionally would raise without CUDA available."""
        import millm.services.probe_runtime as runtime

        called = []
        original = runtime.torch.cuda.synchronize
        runtime.torch.cuda.synchronize = lambda device=None: called.append(device)
        try:
            context = ProbeRequestContext("req", [_probe()])
            context.observe(1, torch.randn(1, 16, 8))
        finally:
            runtime.torch.cuda.synchronize = original
        assert not called
