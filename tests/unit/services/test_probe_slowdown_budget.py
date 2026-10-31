"""ENH-006 D83 (operator, 2026-10-29): probe capacity is a SLOWDOWN BUDGET, not a count.

"There is not going to be a limit of using 2 probes … We need to be able to support as many as
technically practical." Probes may add at most `PROBE_SLOWDOWN_BUDGET_PCT` (5) per cent of the
resident model's own measured time; there is no behaviour count; the cost gate alone decides.

Mutation controls (`0xcc/reviews/findings_1029_controls.md`):
* re-enable the behaviour cap in `_count_gates`            → `test_ten_behaviours_arm_*` red
* absolute budget instead of relative (`_global_half`)      → `test_the_budget_scales_*` red
* `>` → `>=` in `over_budget`                               → `test_the_five_percent_boundary_*` red
* recommendation costed on the mean line (the old estimate) → `test_the_recommendation_admits_*` red
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from millm.core.config import settings
from millm.core.errors import ProbeBudgetError
from millm.services.probe_arming import ProbeArmingService
from millm.services.probe_cost import ProbeCost
from millm.services.probe_limit_recommendation import (
    recommend,
    reference_costs,
    settings_snapshot,
)
from millm.services.probe_limits import effective, slowdown_ms
from millm.services.probe_runtime import ProbeRuntimeState
from tests.unit.services.test_probe_arming import (  # noqa: F401 - fixtures
    TinyModel,
    clean_state,
    ctx,
    definition,
    loaded,
)
from tests.unit.services.test_probe_budget_gate import (  # noqa: F401 - fixtures
    _arm,
    _import,
    fake_measure,
)

FIXTURE = (Path(__file__).resolve().parents[2] / "fixtures" / "probe_capacity"
           / "measurements_2026-10-21.json")
MEASURED = {m["repo_id"]: SimpleNamespace(**m)
            for m in json.loads(FIXTURE.read_text())["measurements"]}
LFM = MEASURED["LiquidAI/LFM2.5-1.2B-Instruct"]        # 6.69 ms per generated token
LLAMA = MEASURED["meta-llama/Llama-3.1-8B-Instruct"]    # 22.68 ms per generated token


def _measurement(decode=20.0, prefill=100.0, lines=None, **over):
    dense = {"decode_own_ms": 0.02, "decode_fixed_ms": 0.05,
             "prefill_own_ms_per_1k": 0.05, "prefill_fixed_ms_per_1k": 0.05}
    return SimpleNamespace(**{
        "id": 77, "repo_id": "org/model", "quantization": "FP16",
        "device": "cuda:GPU-x", "decode_ms_per_token": decode, "prefill_ms_per_1k": prefill,
        "dense": dense, "sparse": None, "lines": lines, "measured_at": "2026-10-29T00:00:00+00:00",
        **over,
    })


def _row(**over):
    base = dict(max_armed=None, decode_budget_value=None, decode_budget_unit=None,
                prefill_budget_value=None, prefill_budget_unit=None, max_behaviours=None,
                source=None)
    base.update(over)
    return SimpleNamespace(**base)


def _plans(costs):
    """Arm plans as `_cost_gate` reads them: each a candidate `ArmedProbe` with its cost."""
    return [SimpleNamespace(armed=SimpleNamespace(probe_id=f"p{i}", cost=c, layer=c.layer),
                            probe=SimpleNamespace(id=f"p{i}"), first_on_layer=True)
            for i, c in enumerate(costs)]


def _gate():
    ProbeRuntimeState.reset_for_tests()
    return ProbeArmingService(None, ProbeRuntimeState())


def _cost(own, fixed, layer=0, prefill_own=0.0, prefill_fixed=0.0):
    return ProbeCost(decode_ms=own, prefill_ms_per_1k=prefill_own, layer_fixed_ms=fixed,
                     layer_fixed_prefill_ms_per_1k=prefill_fixed, layer=layer, device="cuda:GPU-x",
                     k=1, measured_at="t")


# ── the behaviour count no longer refuses ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ten_behaviours_arm_when_the_cost_fits(ctx, fake_measure):  # noqa: F811
    """Ten key-less probes are ten behaviours. Before D83 the third was refused (global 2); a
    stored per-model `max_behaviours` of 2 (the 8B's approval of 2026-10-29) binds nothing."""
    repo, service, arming = ctx
    limits = effective(_row(max_behaviours=2, source="operator"), None)
    assert limits.max_behaviours is None and "not enforced" in limits.sources["max_behaviours"]
    for i in range(10):
        probe = await _import(service, f"dense-{i}")
        await arming.arm(probe, executor=_run_inline, model=TinyModel(), loaded=loaded(),
                         forward=_forward_on(probe.layer), limits=limits)
    assert len(ProbeRuntimeState().armed()) == 10


async def _run_inline(fn):
    return fn()


def _forward_on(layer):
    import torch

    def _forward(input_ids, context):
        context.observe(layer, torch.full((1, input_ids.shape[1], 8), 0.5))

    return _forward


# ── the budget is relative to the model's own measured time ─────────────────────────────────


def test_the_global_budget_is_five_percent_of_the_models_own_time():
    eff = effective(None, LLAMA)
    assert settings.PROBE_SLOWDOWN_BUDGET_PCT == 5.0
    assert eff.decode_ms_per_pass == pytest.approx(0.05 * 22.68)
    assert eff.sources["decode"] == "global (5% of 22.68 ms)"


def test_the_budget_scales_with_the_models_measured_time():
    """Same probes, two models: the 8B (22.68 ms per token) admits them, LFM2.5 (6.69 ms) refuses.
    Twelve dense probes on one layer at the 8B's own line cost 0.0757 + 12 × 0.0236 = 0.359 ms per
    generated token: under 0.9 × 5% × 22.68 = 1.021, over 0.9 × 5% × 6.69 = 0.301."""
    line = LLAMA.dense
    costs = [_cost(line["decode_own_ms"], line["decode_fixed_ms"]) for _ in range(12)]
    big, small = effective(None, LLAMA), effective(None, LFM)
    _gate()._cost_gate(_plans(costs), big)
    with pytest.raises(ProbeBudgetError) as refused:
        _gate()._cost_gate(_plans(costs), small)
    assert "5% of the model's own time" in refused.value.message
    assert refused.value.details["budget_sources"]["decode"] == "global (5% of 6.69 ms)"


def test_the_five_percent_boundary_is_exact():
    """Exactly headroom × 5% of the step is ADMITTED; one nanosecond more is refused."""
    m = _measurement(decode=20.0, prefill=100.0)
    eff = effective(None, m)
    ceiling = slowdown_ms(5.0, 20.0, 0.25)[0] * eff.headroom
    assert eff.decode_ms_per_pass * eff.headroom == ceiling
    _gate()._cost_gate(_plans([_cost(ceiling / 2, ceiling / 2)]), eff)
    with pytest.raises(ProbeBudgetError):
        _gate()._cost_gate(_plans([_cost(ceiling / 2 + 1e-9, ceiling / 2)]), eff)


def test_a_per_model_percentage_and_an_operator_ms_override_both_resolve():
    m = _measurement(decode=20.0, prefill=100.0)
    pct = effective(_row(decode_budget_value=10.0, decode_budget_unit="pct",
                         prefill_budget_value=2.0, prefill_budget_unit="pct", source="operator"), m)
    assert (pct.decode_ms_per_pass, pct.prefill_ms_per_1k) == pytest.approx((2.0, 2.0))
    ms = effective(_row(decode_budget_value=0.6, decode_budget_unit="ms",
                        prefill_budget_value=1.2, prefill_budget_unit="ms", source="operator"), m)
    assert (ms.decode_ms_per_pass, ms.prefill_ms_per_1k) == (0.6, 1.2)
    assert ms.sources["decode"] == "model (ms)"


def test_a_pre_d83_approved_ms_budget_is_superseded():
    """The 8B's approval of 2026-10-29 (0.70 / 2.49 ms, `recommended-approved`) was sized from the
    behaviour count (2 sets × set cost): left binding it would re-impose the count D83 removed."""
    m = _measurement(decode=22.68, prefill=60.0)
    eff = effective(_row(decode_budget_value=0.70, decode_budget_unit="ms",
                         prefill_budget_value=2.49, prefill_budget_unit="ms",
                         max_behaviours=2, source="recommended-approved"), m)
    assert eff.decode_ms_per_pass == pytest.approx(0.05 * 22.68)
    assert eff.prefill_ms_per_1k == pytest.approx(0.05 * 60.0)
    assert "superseded" in eff.sources["decode"] and eff.max_behaviours is None


# ── an unmeasured model ─────────────────────────────────────────────────────────────────────


def test_an_unmeasured_model_falls_back_and_names_the_measure_action():
    """Decision: a model (or half) with no measured step time runs on the absolute fallback
    (0.25 / 0.5 ms) rather than refusing everything, and every refusal says how to get 5%."""
    eff = effective(None, None)
    assert (eff.decode_ms_per_pass, eff.prefill_ms_per_1k) == (0.25, 0.5)
    assert "not measured" in eff.sources["decode"]
    assert "/probe-capacity/measure" in eff.sources["decode"]
    with pytest.raises(ProbeBudgetError) as refused:
        _gate()._cost_gate(_plans([_cost(0.2, 0.1)]), eff)
    assert "probe-capacity/measure" in refused.value.details["next_action"]
    assert "not measured" in refused.value.message


def test_a_half_measured_model_uses_five_percent_where_it_can():
    """The 2026-10-21 seed rows measured decode only: decode is 5%, prefill the fallback."""
    eff = effective(None, LLAMA)
    assert eff.decode_ms_per_pass == pytest.approx(1.134)
    assert eff.prefill_ms_per_1k == 0.5 and "not measured" in eff.sources["prefill"]


# ── F-2: the recommendation and admission are one cost model ────────────────────────────────


def _lines():
    """Per-rule lines where `attention` and `max` carry a DEARER fixed part than `mean`: the
    case the old arithmetic (layers × the mean line's fixed part) under-stated."""
    return {
        "mean": {"decode_own_ms": 0.02, "decode_fixed_ms": 0.03,
                 "prefill_own_ms_per_1k": 0.04, "prefill_fixed_ms_per_1k": 0.03},
        "max": {"decode_own_ms": 0.021, "decode_fixed_ms": 0.05,
                "prefill_own_ms_per_1k": 0.041, "prefill_fixed_ms_per_1k": 0.05},
        "attention": {"decode_own_ms": 0.035, "decode_fixed_ms": 0.09,
                      "prefill_own_ms_per_1k": 0.07, "prefill_fixed_ms_per_1k": 0.09},
    }


@pytest.mark.parametrize("decode, prefill, binding", [
    (20.0, 60.0, "decode"), (40.0, 120.0, "decode"), (30.0, 20.0, "prefill"),
])
def test_the_recommendation_admits_the_sets_it_reports(decode, prefill, binding):
    """Approve the recommendation, then arm the K reference sets it says fit — in its own order AND
    with the cheapest probe first on every layer (the 038 window's arming order): admission
    admits them. K + 1 sets are refused. One cost model, not two (F-2)."""
    m = _measurement(decode=decode, prefill=prefill, lines=_lines())
    snap = settings_snapshot(settings)
    rec = recommend(SimpleNamespace(id=1), m, snap)
    k = rec["capacity"]["reference_sets"]
    assert k >= 1 and rec["capacity"]["binding"] == binding
    approved = effective(_row(decode_budget_value=rec["values"]["decode_budget_pct"],
                              decode_budget_unit="pct",
                              prefill_budget_value=rec["values"]["prefill_budget_pct"],
                              prefill_budget_unit="pct", source="recommended-approved"), m)
    # ⚠ The armed probes are priced INDEPENDENTLY of the recommendation's own builder, each at
    # its own rule's line — what admission is handed when each probe's arm-time cost equals its
    # line. Built with `reference_costs` they agreed by construction, and costing every rule on
    # the mean line (F-2's estimate) survived (controls record, C3).
    def armed_sets(n, dearest_first):
        out = []
        for layer in range(3 * n):
            probes = [_cost(line["decode_own_ms"], line["decode_fixed_ms"], layer,
                            line["prefill_own_ms_per_1k"], line["prefill_fixed_ms_per_1k"])
                      for line in _lines().values()]
            out.extend(sorted(probes, key=lambda c: c.layer_fixed_ms, reverse=dearest_first))
        return out

    assert len(reference_costs(m, snap, k)) == len(armed_sets(k, True)) == 9 * k
    _gate()._cost_gate(_plans(armed_sets(k, True)), approved)
    _gate()._cost_gate(_plans(armed_sets(k, False)), approved)  # the 038 window's arming order
    with pytest.raises(ProbeBudgetError):
        _gate()._cost_gate(_plans(armed_sets(k + 1, True)), approved)


@pytest.mark.parametrize("m, decode_n, prefill_n", [
    # 0.9 × 5% × 22.68 = 1.0206 ms → (1.0206 − 0.0757) / 0.0236 = 40.0; prefill (no step time in
    # these 2026-10-21 seed rows) runs on the 0.5 ms fallback: (0.45 − 0.0538) / 0.1063 = 3.7.
    (LLAMA, 40, 3),
    # 0.9 × 5% × 6.69 = 0.3011 → (0.3011 − 0.0496) / 0.0253 = 9.9; (0.45 − 0.0411) / 0.0569 = 7.2.
    (LFM, 9, 7),
], ids=["llama-3.1-8b", "lfm2.5-1.2b"])
def test_the_recommendation_counts_dense_probes_as_admission_would(m, decode_n, prefill_n):
    """`dense_probes` is the largest N admission admits on one layer, and agrees with the capacity
    view's per-half `implied` figures (the smaller of the two halves)."""
    from millm.services.probe_limits import implied

    rec = recommend(SimpleNamespace(id=9), m, settings_snapshot(settings))
    eff = effective(None, m)
    view = implied(m, eff)["dense"]
    assert (view["decode"]["fits_with_headroom"], view["prefill"]["fits_with_headroom"]) == (
        decode_n, prefill_n)
    assert rec["capacity"]["dense_probes"] == min(decode_n, prefill_n) == view["n"]
    assert rec["capacity"]["dense_binding"] == "prefill"
    assert rec["values"]["prefill_budget_ms"] is None  # not measured: the fallback applies
    line = m.dense
    one = _cost(line["decode_own_ms"], line["decode_fixed_ms"], 0,
                line["prefill_own_ms_per_1k"], line["prefill_fixed_ms_per_1k"])
    n = rec["capacity"]["dense_probes"]
    _gate()._cost_gate(_plans([one] * n), eff)
    with pytest.raises(ProbeBudgetError):
        _gate()._cost_gate(_plans([one] * (n + 1)), eff)


def test_the_recommendation_is_the_slowdown_budget_and_carries_no_count():
    rec = recommend(SimpleNamespace(id=9), _measurement(lines=_lines()), settings_snapshot(settings))
    assert rec["values"]["decode_budget_pct"] == rec["values"]["prefill_budget_pct"] == 5.0
    assert rec["values"]["max_behaviours"] is None
    assert any("at most 5% of this model's own time" in r for r in rec["reasons"])


def test_an_estimated_line_is_reported_as_an_upper_bound():
    rec = recommend(SimpleNamespace(id=1), LFM, settings_snapshot(settings))
    assert rec["estimated"] is True and rec["capacity"]["upper_bound"] is True
    assert any("upper bound" in r and "attention" in r for r in rec["reasons"])



# ── the per-pass warning follows the admitted budget ────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_overhead_warning_reads_the_budget_the_set_was_admitted_against(
        ctx, fake_measure, caplog):  # noqa: F811
    """A model admitted at 5% of its own time (1.134 ms per token on the 8B) must not warn at the
    absolute 0.25 ms on every request; the warning reads what `commit` admitted against."""
    import logging

    from millm.services.probe_event_service import ProbeEventService, _per_pass_threshold

    repo, service, arming = ctx
    limits = effective(None, LLAMA)
    probe = await _import(service, "dense-w")
    await arming.arm(probe, executor=_run_inline, model=TinyModel(), loaded=loaded(),
                     forward=_forward_on(probe.layer), limits=limits)
    assert ProbeRuntimeState().admitted_budget == limits.budget()
    assert _per_pass_threshold() == pytest.approx(0.05 * 22.68)
    events = ProbeEventService.__new__(ProbeEventService)
    with caplog.at_level(logging.WARNING):
        events.note_request_overhead(overhead_ms=0.5 * 10, n_passes=10)  # 0.5 ms per pass
    assert "probe_overhead_above_threshold" not in caplog.text
    with caplog.at_level(logging.WARNING):
        events.note_request_overhead(overhead_ms=1.5 * 10, n_passes=10)  # over 1.134
    assert "probe_overhead_above_threshold" in caplog.text
