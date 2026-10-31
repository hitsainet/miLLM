"""Each model's explained probe-limit recommendation (039 FR-39.8; rewritten for ENH-006 D83).

**D83 (operator, 2026-10-29): capacity is a slowdown budget, not a count.** "There is not going to be
a limit of using 2 probes … We need to be able to support as many as technically practical." Probes
may add at most `PROBE_SLOWDOWN_BUDGET_PCT` (default 5) per cent of the resident model's OWN measured
time — its decode milliseconds per generated token, and its prefill milliseconds per 1,000 prompt
tokens. There is no behaviour count any more; the cost gate alone decides how many probes fit.

So the recommendation is that percentage, for both halves, and an explanation of WHAT FITS under it:
how many reference sets (`PROBE_REFERENCE_SET_SHAPE`, default 9 probes over 3 layers, 3 each of the
`mean`, `max` and `attention` rules) and how many dense probes on one layer. Approving it writes the
two halves as `{value: <pct>, unit: "pct"}`, so the budget follows the model's next measurement.

⚠ ONE COST MODEL WITH ADMISSION (F-2, 038 window 2026-10-29). The old recommendation costed a set
with its own arithmetic — every probe's own part plus `layers × the MEAN line's fixed part` — and
sized an absolute budget to exactly that. Admission (`ProbeArmingService._cost_gate`) charges each
(layer, device) the fixed part of the FIRST probe armed on it, whatever its rule, through
`probe_cost.project`, and refuses on `probe_arming.over_budget`. Whenever the max or attention line's
fixed part exceeded the mean line's, the approved budget could not admit the set it was sized for —
and a measurement without per-rule lines (taken before 039) priced every attention probe at the
cheaper mean line. LFM2.5's recommended 0.36 / 0.70 ms then refused the eighth and ninth probes of
the very set it was computed for. Here the reference set is built as `ProbeCost`s with
`probe_limits.estimated_cost` (the preview's estimator), projected with `probe_cost.project` and
checked with `over_budget` against the budget `probe_limits.effective` would put in force — the
admission functions themselves, not a copy. Missing per-rule lines are still an estimate, and the
reasons say the figures are then an upper bound until the model is re-measured.

`recommend` stays PURE — no database, no clock, no settings singleton — so the approval route can
recompute it and compare digests (TD39-31).

⚠ NO BEHAVIOUR APPEARS HERE. The recommendation never uses which behaviour a set serves (FR-39.8).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Optional

#: The per-request wall cost the per-PASS budget does not see: about 2.5 – 3 ms per dense probe per
#: request in the 2026-10-21 hardware record (event rows, context windows, sessions), so a reader of
#: the recommendation is told what the fitting probes cost beyond the forward pass.
PER_PROBE_REQUEST_WALL_MS = 2.75
RULES = ("mean", "max", "attention")
#: Recorded in every recommendation and its digest: which arithmetic produced it.
COST_MODEL = "probe_cost.project + probe_arming.over_budget (D83 slowdown budget)"


def canonical_json(obj: Any) -> str:
    """The digest form (FTID §3.3 / §11): sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def settings_snapshot(settings: Any) -> dict[str, Any]:
    """The settings `recommend` reads, as plain data — and the part of the digest that says so."""
    return {
        "slowdown_budget_pct": float(settings.PROBE_SLOWDOWN_BUDGET_PCT),
        "fallback_decode_ms": float(settings.PROBE_MAX_OVERHEAD_MS_PER_PASS),
        "fallback_prefill_ms_per_1k": float(settings.PROBE_MAX_PREFILL_MS_PER_1K),
        "headroom": float(settings.PROBE_BUDGET_HEADROOM),
        "max_armed": int(settings.PROBE_MAX_ARMED),
        "reference_set": dict(settings.PROBE_REFERENCE_SET_SHAPE),
    }


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _devices(measurement: Any) -> list[str]:
    device = getattr(measurement, "device", None) or ""
    return sorted(d for d in str(device).split("+") if d)


def _has_line(measurement: Any, rule: str) -> bool:
    lines = getattr(measurement, "lines", None)
    return isinstance(lines, dict) and isinstance(lines.get(rule), dict)


def _shape(snapshot: dict[str, Any]) -> tuple[dict[str, int], int, int]:
    shape = snapshot["reference_set"]
    rules = {r: int(n) for r, n in (shape.get("rules") or {}).items() if int(n) > 0}
    n_layers = max(1, int(shape.get("layers") or 1))
    n_probes = int(shape.get("probes") or sum(rules.values()))
    return rules, n_layers, n_probes


def reference_costs(measurement: Any, snapshot: dict[str, Any], sets: int) -> Optional[list[Any]]:
    """`sets` reference sets as `ProbeCost`s, or None when the measurement has no usable line.

    Each set sits on its own `layers` layers, distinct from every other set's (which can only
    over-state the cost: no fixed part is shared across sets). A set's probes are dealt across its
    layers in turn (3 × mean/max/attention over 3 layers = one of each rule per layer, the shape of
    the 038 window's real set), and on each layer the probe with the LARGEST fixed part comes first —
    `project` charges a layer the fixed part of the first probe it meets, as admission charges the
    first probe armed there, so this is the dearest arming order.
    """
    from millm.services.probe_limits import estimated_cost

    rules, n_layers, _ = _shape(snapshot)
    device = (_devices(measurement) or ["reference"])[0]
    order = [rule for rule, count in sorted(rules.items()) for _ in range(count)]
    costs: list[Any] = []
    for s in range(int(sets)):
        per_layer: dict[int, list[Any]] = {}
        for i, rule in enumerate(order):
            layer = s * n_layers + (i % n_layers)
            cost = estimated_cost(measurement, rule, layer, device)
            if cost is None:
                return None
            per_layer.setdefault(layer, []).append(cost)
        for layer in sorted(per_layer):
            costs.extend(sorted(per_layer[layer], key=lambda c: -c.layer_fixed_ms
                                - c.layer_fixed_prefill_ms_per_1k))
    return costs


def dense_costs(measurement: Any, n: int) -> Optional[list[Any]]:
    """`n` dense (`mean`) probes on ONE layer: the cheapest arrangement."""
    from millm.services.probe_limits import estimated_cost

    device = (_devices(measurement) or ["reference"])[0]
    one = estimated_cost(measurement, "mean", 0, device)
    return None if one is None else [one] * int(n)


def budget_of(measurement: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    """The budget approving this recommendation puts in force — `probe_limits.slowdown_ms`, the
    function `probe_limits.effective` resolves a `pct` half with — and whether each half is
    measured (an unmeasured half falls back to the absolute default)."""
    from millm.services.probe_limits import slowdown_ms

    pct = float(snapshot["slowdown_budget_pct"])
    decode, d_measured = slowdown_ms(pct, getattr(measurement, "decode_ms_per_token", None),
                                     snapshot["fallback_decode_ms"])
    prefill, p_measured = slowdown_ms(pct, getattr(measurement, "prefill_ms_per_1k", None),
                                      snapshot["fallback_prefill_ms_per_1k"])
    return {"decode_ms_per_pass": decode, "prefill_ms_per_1k": prefill,
            "headroom": float(snapshot["headroom"]),
            "measured": {"decode": d_measured, "prefill": p_measured}}


def _fits(costs: list[Any], budget: dict[str, Any], max_armed: int) -> tuple[bool, Optional[str]]:
    """Admission's verdict on `costs` armed together: (fits, what refuses it)."""
    from millm.services.probe_arming import over_budget
    from millm.services.probe_cost import project

    if len(costs) > max_armed:
        return False, "max_armed"
    projected = project(costs)
    if not over_budget(projected, budget):
        return True, None
    if projected["decode"] > budget["decode_ms_per_pass"] * budget["headroom"]:
        return False, "decode"
    return False, "prefill"


def largest_fitting(build: Any, budget: dict[str, Any], max_armed: int) -> tuple[int, Optional[str]]:
    """The largest k with `build(k)` admitted, and what refuses k + 1. Terminates: `build(k)`
    holds at least k probes, and more than `max_armed` never fits."""
    k = 0
    while True:
        costs = build(k + 1)
        if costs is None:
            return k, None
        ok, why = _fits(costs, budget, max_armed)
        if not ok:
            return k, why
        k += 1


def recommend(model_row: Any, measurement: Any, snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The explained recommendation for one model from one measurement, or None without one."""
    if measurement is None:
        return None
    rules, n_layers, n_probes = _shape(snapshot)
    if reference_costs(measurement, snapshot, 1) is None:
        return None
    pct = float(snapshot["slowdown_budget_pct"])
    max_armed = int(snapshot["max_armed"])
    budget = budget_of(measurement, snapshot)
    headroom = budget["headroom"]
    estimated_rules = sorted(r for r in rules if not _has_line(measurement, r))

    sets, sets_binding = largest_fitting(
        lambda k: reference_costs(measurement, snapshot, k), budget, max_armed)
    dense, dense_binding = largest_fitting(lambda n: dense_costs(measurement, n), budget,
                                           max_armed)

    from millm.services.probe_cost import project

    one_set = project(reference_costs(measurement, snapshot, 1) or [])
    decode_step = getattr(measurement, "decode_ms_per_token", None)
    prefill_step = getattr(measurement, "prefill_ms_per_1k", None)
    values = {
        # D83: no behaviour count. Kept (null) so a reader of the v1.16 shape finds the key.
        "max_behaviours": None,
        "decode_budget_pct": pct,
        "prefill_budget_pct": pct,
        # What the percentages mean on this measurement — for reading; the approval writes `pct`.
        # None for a half whose step time is not measured (that half then falls back).
        "decode_budget_ms": round(budget["decode_ms_per_pass"], 6)
        if budget["measured"]["decode"] else None,
        "prefill_budget_ms": round(budget["prefill_ms_per_1k"], 6)
        if budget["measured"]["prefill"] else None,
    }

    devices = _devices(measurement)
    measured_at = _iso(getattr(measurement, "measured_at", None))
    reasons = [
        f"Measurement {measurement.id} of {measurement.repo_id} at {measurement.quantization}, "
        f"taken {measured_at} on {', '.join(devices) or 'an unlabelled device'}.",
        f"Probes may slow each token by at most {pct:g}% of this model's own time (D83).",
    ]
    for half, step, unit, fallback in (
        ("decode", decode_step, "ms per generated token", snapshot["fallback_decode_ms"]),
        ("prefill", prefill_step, "ms per 1,000 prompt tokens",
         snapshot["fallback_prefill_ms_per_1k"]),
    ):
        b = budget["decode_ms_per_pass" if half == "decode" else "prefill_ms_per_1k"]
        if budget["measured"][half]:
            reasons.append(f"{half.capitalize()}: {pct:g}% of the measured {float(step):g} {unit} "
                           f"is {b:.4f} ms; arming fills up to {headroom:g} of it "
                           f"({b * headroom:.4f} ms).")
        else:
            reasons.append(f"{half.capitalize()}: this model's {half} step time is not measured, "
                           f"so the absolute default {float(fallback):g} {unit} applies; measure "
                           "the model (POST /api/models/{id}/probe-capacity/measure) to use the "
                           f"{pct:g}% budget.")
    reasons.append(
        f"One reference set ({n_probes} probes over {n_layers} layers: "
        + ", ".join(f"{n} {r}" for r, n in sorted(rules.items()))
        + f") costs {one_set['decode']:.4f} ms per decode pass and {one_set['prefill']:.4f} ms per "
        "1,000 prompt tokens, costed as admission costs it (each probe's own part, each layer's "
        "shared part once)."
    )
    reasons.append(
        f"Fits: {sets} reference set(s) ({sets * n_probes} probes) and {dense} dense probes on one "
        "layer"
        + (f"; the {sets_binding} limit stops the next set" if sets_binding else "") + "."
    )
    if estimated_rules:
        reasons.append(
            "Estimated: this measurement has no line for " + ", ".join(estimated_rules)
            + "; those probes are costed on the mean line, which under-states an attention probe "
            "(038 window F-2), so these counts are an upper bound. Re-measure the model to replace "
            "the estimate. Admission always charges each probe's own cost measured at arm."
        )
    wall = PER_PROBE_REQUEST_WALL_MS * n_probes * sets
    reasons.append(
        f"Not in the per-pass budget: about {PER_PROBE_REQUEST_WALL_MS:g} ms per probe per request "
        f"of recording work (2026-10-21 hardware record), about {wall:.0f} ms per request for "
        f"{sets} armed set(s) of {n_probes}."
    )
    would_change = [
        "a new capacity measurement of this model (a different card, precision or quantization)",
        "the slowdown budget (PROBE_SLOWDOWN_BUDGET_PCT) or the admission headroom",
        "the reference set shape or the armed-probe backstop (PROBE_MAX_ARMED)",
    ]
    snapshot_for_digest = {k: snapshot[k] for k in sorted(snapshot)}
    digest = hashlib.sha256(canonical_json({
        "values": values, "measurement_id": measurement.id, "settings": snapshot_for_digest,
        "cost_model": COST_MODEL,
    }).encode("utf-8")).hexdigest()

    def share(half: str) -> Optional[float]:
        return pct if budget["measured"][half] else None

    return {
        "model_id": getattr(model_row, "id", None),
        "values": values,
        "budget": {k: budget[k] for k in ("decode_ms_per_pass", "prefill_ms_per_1k", "headroom")},
        "shares": {"decode_pct": share("decode"), "prefill_pct": share("prefill")},
        "capacity": {
            "reference_sets": sets,
            "reference_set_probes": sets * n_probes,
            "dense_probes": dense,
            "binding": sets_binding,
            "dense_binding": dense_binding,
            "upper_bound": bool(estimated_rules),
        },
        "reasons": reasons,
        # v1.16 keys, kept: the reference sets that fit (D83 has no target to aim at).
        "achieved_sets": sets,
        "target_sets": None,
        "reference_set": snapshot["reference_set"],
        "estimated": bool(estimated_rules),
        "measurement": {"id": measurement.id, "measured_at": measured_at, "devices": devices},
        "would_change": would_change,
        "per_request_wall_ms_estimate": round(wall, 2),
        "set_cost": {"decode_ms": round(one_set["decode"], 6),
                     "prefill_ms_per_1k": round(one_set["prefill"], 6)},
        "cost_model": COST_MODEL,
        "recommendation_digest": digest,
    }
