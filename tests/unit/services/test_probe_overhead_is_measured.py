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
        """Specificity: a constant would satisfy every assertion above."""
        small = ProbeRequestContext("a", [_probe()])
        small.observe(1, torch.randn(1, 32, 8))
        large = ProbeRequestContext("b", [_probe()])
        large.observe(1, torch.randn(1, 4096, 8))
        assert large.overhead_ms > small.overhead_ms

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


class TestTheMeasurementDoesNotChargeTheModel:
    """⚠ The reported overhead was 10x the real cost.

    The hook fires DURING the forward pass, so the model's kernels are still in flight. The
    probe's first touch of the result (`.tolist()`) blocks until they finish, and a naive timer
    charges that wait to the probe.

    Measured on the node at 4k tokens with one probe: **116 ms reported, 11.2 ms actual**
    (wall-clock median 185.0 armed against 173.8 disarmed). An operator reading the reported
    number would conclude probes cost 62% of a request when they cost 6%, and the
    above-threshold warning would never stop firing.

    On CPU there is nothing to synchronise and the two agree, so this test asserts the CALL —
    the only thing a CPU test can check — and the real correction is recorded in the measurement
    above.
    """

    def test_a_cuda_tensor_is_synchronised_before_the_clock_starts(self, monkeypatch):
        import millm.services.probe_runtime as runtime

        calls: list[object] = []
        monkeypatch.setattr(
            runtime.torch.cuda, "synchronize", lambda device=None: calls.append(device)
        )

        class FakeCudaTensor(torch.Tensor):
            """A CPU tensor that claims to be on CUDA, so the branch runs without a GPU."""

            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data, False)

            @property
            def is_cuda(self):  # type: ignore[override]
                return True

        context = ProbeRequestContext("req", [_probe()])
        context.observe(1, FakeCudaTensor(torch.randn(1, 16, 8)))

        assert calls, (
            "the score path did not synchronise before timing — the reported overhead then "
            "includes the model's in-flight work, which measured 10x the probe's real cost"
        )

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
