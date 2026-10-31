"""Gate 5: cost-aware admission, with the count limit kept as the backstop (FR-37.2–37.4, 37.7).

The arm path is the real `ProbeArmingService.arm` (all four earlier gates run); only the measurer is
replaced, by a fake that returns DIFFERENT costs per probe — so a gate that merely counted probes
would admit or refuse at the wrong arm.

Mutation controls (037 FTDD §10):
* remove gate 5's `probe_cost.measure` call          → `test_arm_measures_cost` red
* `>` → `>=` at the budget                           → `test_budget_boundary` red
* measure on a fixed device instead of the layer's   → `test_split_costs_on_own_device` red
"""

from __future__ import annotations

import pytest
import torch

from millm.core.config import settings
from millm.core.errors import ProbeBudgetError, ProbeLimitError
from millm.services import probe_cost
from millm.services.probe_cost import ProbeCost
from millm.services.probe_event_service import ProbeEventService
from millm.services.probe_runtime import ProbeRuntimeState
from tests.unit.services.test_probe_arming import (  # noqa: F401 - fixtures
    TinyModel,
    clean_state,
    ctx,
    definition,
    loaded,
)

pytestmark = pytest.mark.asyncio


# These fixtures arm key-less probes, each a behaviour of its own; since ENH-006 D83 behaviours
# are not counted, so the 037 gates — the probe backstop and the cost budget — are what bind.


async def run_inline(fn):
    return fn()


def forward_on(layer: int):
    def _forward(input_ids, context):
        context.observe(layer, torch.full((1, input_ids.shape[1], 8), 0.5))

    return _forward


#: decode cost per probe NAME, and the layer's fixed part.
DENSE, SPARSE, FIXED = 0.01, 0.10, 0.02


@pytest.fixture
def fake_measure(monkeypatch):
    calls = []

    def measure(armed, *, device, d_model, k, dtype=torch.float32, first_on_layer=True):
        calls.append({"probe": armed.name, "device": str(device), "d_model": d_model, "k": k,
                      "first_on_layer": first_on_layer})
        own = SPARSE if armed.name.startswith("sparse") else DENSE
        return ProbeCost(decode_ms=own, prefill_ms_per_1k=own, layer_fixed_ms=FIXED,
                         layer_fixed_prefill_ms_per_1k=FIXED, layer=armed.layer,
                         device=str(device), k=k, measured_at="t")

    monkeypatch.setattr(probe_cost, "measure", measure)
    return calls


async def _import(service, name, layer=11):
    doc = definition(name=name)
    doc["read"]["layer"] = layer
    return await service.import_definition(doc, raw_bytes=1, on_conflict="fail", origin="file")


async def _arm(arming, probe, model=None):
    return await arming.arm(probe, executor=run_inline, model=model or TinyModel(),
                            loaded=loaded(), forward=forward_on(probe.layer))


async def test_arm_measures_cost(ctx, fake_measure):  # noqa: F811
    repo, service, arming = ctx
    probe = await _import(service, "dense-a")
    await _arm(arming, probe)
    assert fake_measure == [{"probe": "dense-a", "device": "cpu", "d_model": 8,
                             "k": settings.PROBE_COST_SAMPLES, "first_on_layer": True}]
    assert probe.cost["decode_ms"] == DENSE and probe.cost["device"] == "cpu"
    assert ProbeRuntimeState().get(probe.id).cost.decode_ms == DENSE


async def test_costs_differ_and_the_budget_refuses_with_the_figures(ctx, fake_measure):  # noqa: F811
    repo, service, arming = ctx
    # 0.25 × 0.9 = 0.225 per decode pass. dense 0.01 + fixed 0.02 = 0.03; + sparse 0.10 = 0.13;
    # another sparse would be 0.23 > 0.225 — refused, with only three armed (count limit is 8).
    await _arm(arming, await _import(service, "dense-a"))
    await _arm(arming, await _import(service, "sparse-a"))
    third = await _import(service, "sparse-b")
    with pytest.raises(ProbeBudgetError) as refused:
        await _arm(arming, third)
    details = refused.value.details
    assert details["projected"]["decode"] == pytest.approx(0.23)
    assert details["budget"] == {"decode_ms_per_pass": 0.25, "prefill_ms_per_1k": 0.5,
                                 "headroom": 0.9}
    assert details["candidate"]["decode"] == pytest.approx(SPARSE)
    assert details["candidate"]["first_on_layer"] is False
    assert "projected 0.230 ms per decode pass" in refused.value.message
    assert ProbeRuntimeState().get(third.id) is None and not third.armed
    # …while a cheaper probe still fits.
    await _arm(arming, await _import(service, "dense-b"))


async def test_budget_boundary(ctx, fake_measure, monkeypatch):  # noqa: F811
    """Exactly at headroom × budget is ADMITTED (strict `>`)."""
    monkeypatch.setattr(settings, "PROBE_BUDGET_HEADROOM", 1.0)
    monkeypatch.setattr(settings, "PROBE_MAX_OVERHEAD_MS_PER_PASS", 0.125)
    monkeypatch.setattr(settings, "PROBE_MAX_PREFILL_MS_PER_1K", 10.0)
    monkeypatch.setattr(probe_cost, "measure", lambda armed, **kw: ProbeCost(
        decode_ms=0.0625, prefill_ms_per_1k=0.0, layer_fixed_ms=0.0625,
        layer_fixed_prefill_ms_per_1k=0.0, layer=armed.layer, device="cpu", k=1, measured_at="t"))
    await _arm(arming=ctx[2], probe=await _import(ctx[1], "dense-a"))  # 0.125 == 0.125
    with pytest.raises(ProbeBudgetError):
        await _arm(ctx[2], await _import(ctx[1], "dense-b"))  # 0.1875 > 0.125


async def test_first_on_layer_adds_the_fixed_part_once(ctx, fake_measure):  # noqa: F811
    repo, service, arming = ctx
    await _arm(arming, await _import(service, "dense-a", layer=11))
    await _arm(arming, await _import(service, "dense-b", layer=11))
    assert ProbeRuntimeState().projected_cost()["decode"] == pytest.approx(DENSE * 2 + FIXED)
    await _arm(arming, await _import(service, "dense-c", layer=12))
    assert ProbeRuntimeState().projected_cost()["decode"] == pytest.approx(DENSE * 3 + FIXED * 2)
    assert [c["first_on_layer"] for c in fake_measure] == [True, False, True]


async def test_the_count_backstop_is_checked_first(ctx, fake_measure, monkeypatch):  # noqa: F811
    repo, service, arming = ctx
    monkeypatch.setattr(settings, "PROBE_MAX_ARMED", 1)
    await _arm(arming, await _import(service, "dense-a"))
    with pytest.raises(ProbeLimitError):
        await _arm(arming, await _import(service, "dense-b"))
    assert len(fake_measure) == 1  # the refused probe was never measured


async def test_a_failed_measurement_is_refused_not_armed_unmeasured(ctx, monkeypatch):  # noqa: F811
    repo, service, arming = ctx

    def boom(armed, **kw):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(probe_cost, "measure", boom)
    probe = await _import(service, "dense-a")
    with pytest.raises(ProbeBudgetError) as refused:
        await _arm(arming, probe)
    assert refused.value.details["reason"] == "measurement_failed"
    assert ProbeRuntimeState().get(probe.id) is None


async def test_split_costs_on_own_device(ctx, fake_measure, monkeypatch):  # noqa: F811
    """A model split across cards: the probe on layer 12 is measured on the device holding layer
    12, not on the first card or a default (FR-37.9). `meta` stands in for the second card."""
    import millm.ml.layer_resolution as resolution

    real = resolution.layer_device
    monkeypatch.setattr(resolution, "layer_device", lambda model, layer: (
        torch.device("meta") if layer == 12 else real(model, layer)))
    repo, service, arming = ctx
    await _arm(arming, await _import(service, "dense-a", layer=11))
    await _arm(arming, await _import(service, "dense-b", layer=12))
    assert [c["device"] for c in fake_measure] == ["cpu", "meta"]
    # Two (layer, device) pairs: two fixed parts.
    assert ProbeRuntimeState().projected_cost()["decode"] == pytest.approx(DENSE * 2 + FIXED * 2)


async def test_status_reports_the_sum_and_lowering_the_budget_disarms_nothing(
    ctx, fake_measure, monkeypatch,  # noqa: F811
):
    repo, service, arming = ctx
    a = await _arm(arming, await _import(service, "dense-a"))
    await _arm(arming, await _import(service, "sparse-a"))
    from millm.db.repositories.probe_repository import ProbeEventRepository

    status = await ProbeEventService(repo, ProbeEventRepository(repo.session)).status()
    assert status["projected"]["decode"] == pytest.approx(DENSE + SPARSE + FIXED)
    assert status["projected"]["over_budget"] is False
    assert {p["name"]: p["cost"]["decode_ms"] for p in status["armed"]} == {
        "dense-a": DENSE, "sparse-a": SPARSE}
    monkeypatch.setattr(settings, "PROBE_MAX_OVERHEAD_MS_PER_PASS", 0.05)
    status = await ProbeEventService(repo, ProbeEventRepository(repo.session)).status()
    assert status["projected"]["over_budget"] is True
    assert status["armed_count"] == 2 and ProbeRuntimeState().get(a.probe_id) is not None


async def test_disarm_clears_the_recorded_cost(ctx, fake_measure):  # noqa: F811
    repo, service, arming = ctx
    probe = await _import(service, "dense-a")
    await _arm(arming, probe)
    assert probe.cost is not None
    await arming.disarm(probe)
    assert probe.cost is None
