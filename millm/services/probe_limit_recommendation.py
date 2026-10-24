"""Each model's recommended probe limits, explained (039 FR-39.8, D13, TD39-1 – TD39-3, FTDD §6.3).

"An informed (explained) recommendation per model, but allow the operator to override that as easily
as just approving it" (D13). `recommend` is PURE — no database, no clock, no settings singleton:
its inputs are the model row, ONE measurement row and a settings snapshot, so the same inputs
always give the same values and the same digest, and the approval route can recompute it to check
that what the operator approved is what it writes (TD39-31).

The arithmetic (FTDD §6.3):

1. The reference set (`PROBE_REFERENCE_SET_SHAPE`, default 9 probes over 3 layers, 3 each of the
   `mean`, `max` and `attention` rules) is costed from the measurement's per-rule `lines`: each
   probe its rule's own part, plus one fixed part per layer. Its layers are assumed distinct from
   every other set's, which can only over-state the cost.
2. For K = `PROBE_RECOMMEND_TARGET_SETS` down to 1, K sets' cost ÷ the admission headroom is checked
   against two ceilings: decode ≤ `PROBE_RECOMMEND_MAX_DECODE_SHARE` of the model's measured decode
   step, prefill ≤ `PROBE_RECOMMEND_MAX_PREFILL_SHARE` of its prefill step. The largest K that
   passes both is recommended; never below 1, and when even one set breaks a ceiling the reasons
   say so. A step time that was never measured cannot bound anything: that half is not checked,
   and the reasons say so.
3. The budgets are K sets' cost ÷ headroom, rounded UP to 0.01 ms (FTID §3.3), so the approved
   budget always admits the K sets it was computed for.

A measurement taken before 039 has no `max` / `attention` line: those probes are costed on the
`mean` line (the `dense` column), the recommendation is marked `estimated`, and the reasons say
"attention cost estimated from the dense line".

⚠ NO BEHAVIOUR APPEARS HERE. The recommendation never uses which behaviour a set serves (FR-39.8).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any, Optional

#: The per-request wall cost the per-PASS budget does not see: about 2.5 – 3 ms per dense probe per
#: request in the 2026-10-21 hardware record (event rows, context windows, sessions), so a reader of
#: the recommendation is told what one set costs beyond the forward pass.
PER_PROBE_REQUEST_WALL_MS = 2.75
RULES = ("mean", "max", "attention")


def canonical_json(obj: Any) -> str:
    """The digest form (FTID §3.3 / §11): sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _ceil2(value: float) -> float:
    """Rounded UP to 0.01 ms (FTID §3.3), with a guard against float noise just above a cent."""
    return math.ceil(round(value * 100, 9)) / 100


def settings_snapshot(settings: Any) -> dict[str, Any]:
    """The settings `recommend` reads, as plain data — and the part of the digest that says so."""
    return {
        "target_sets": int(settings.PROBE_RECOMMEND_TARGET_SETS),
        "max_decode_share": float(settings.PROBE_RECOMMEND_MAX_DECODE_SHARE),
        "max_prefill_share": float(settings.PROBE_RECOMMEND_MAX_PREFILL_SHARE),
        "headroom": float(settings.PROBE_BUDGET_HEADROOM),
        "reference_set": dict(settings.PROBE_REFERENCE_SET_SHAPE),
    }


def _line(measurement: Any, rule: str) -> tuple[Optional[dict[str, Any]], bool]:
    """The cost line for `rule` and whether it was ESTIMATED from the mean (dense) line."""
    lines = getattr(measurement, "lines", None) or {}
    own = lines.get(rule) if isinstance(lines, dict) else None
    if isinstance(own, dict):
        return own, False
    mean = (lines.get("mean") if isinstance(lines, dict) else None) or getattr(
        measurement, "dense", None)
    return (mean if isinstance(mean, dict) else None), rule != "mean"


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _devices(measurement: Any) -> list[str]:
    device = getattr(measurement, "device", None) or ""
    return sorted(d for d in str(device).split("+") if d)


def recommend(model_row: Any, measurement: Any, snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The explained recommendation for one model from one measurement, or None without one."""
    if measurement is None:
        return None
    shape = snapshot["reference_set"]
    rules = {r: int(n) for r, n in (shape.get("rules") or {}).items() if int(n) > 0}
    n_layers = int(shape.get("layers") or 1)
    n_probes = int(shape.get("probes") or sum(rules.values()))
    headroom = float(snapshot["headroom"])

    estimated_rules: list[str] = []
    set_decode = 0.0
    set_prefill = 0.0
    fixed_decode = fixed_prefill = 0.0
    for rule, count in sorted(rules.items()):
        line, estimated = _line(measurement, rule)
        if line is None:
            return None
        if estimated:
            estimated_rules.append(rule)
        set_decode += count * float(line["decode_own_ms"])
        set_prefill += count * float(line["prefill_own_ms_per_1k"])
    mean_line, _ = _line(measurement, "mean")
    fixed_decode = float(mean_line["decode_fixed_ms"])
    fixed_prefill = float(mean_line["prefill_fixed_ms_per_1k"])
    set_decode += n_layers * fixed_decode
    set_prefill += n_layers * fixed_prefill

    decode_step = getattr(measurement, "decode_ms_per_token", None)
    prefill_step = getattr(measurement, "prefill_ms_per_1k", None)
    target = max(1, int(snapshot["target_sets"]))
    d_share = float(snapshot["max_decode_share"])
    p_share = float(snapshot["max_prefill_share"])

    def fits(k: int) -> tuple[bool, bool]:
        decode_ok = decode_step is None or k * set_decode / headroom <= d_share * decode_step
        prefill_ok = prefill_step is None or k * set_prefill / headroom <= p_share * prefill_step
        return decode_ok, prefill_ok

    achieved = 0
    for k in range(target, 0, -1):
        if all(fits(k)):
            achieved = k
            break
    sets = max(1, achieved)
    decode_budget = _ceil2(sets * set_decode / headroom)
    prefill_budget = _ceil2(sets * set_prefill / headroom)
    values = {"max_behaviours": sets, "decode_budget_ms": decode_budget,
              "prefill_budget_ms": prefill_budget}

    def pct(budget: float, step: Optional[float]) -> Optional[float]:
        return None if not step else round(100.0 * budget / float(step), 2)

    devices = _devices(measurement)
    measured_at = _iso(getattr(measurement, "measured_at", None))
    reasons = [
        f"Measurement {measurement.id} of {measurement.repo_id} at {measurement.quantization}, "
        f"taken {measured_at} on {', '.join(devices) or 'an unlabelled device'}.",
        f"One reference set ({n_probes} probes over {n_layers} layers: "
        + ", ".join(f"{n} {r}" for r, n in sorted(rules.items()))
        + f") costs {set_decode:.4f} ms per decode pass and {set_prefill:.4f} ms per 1,000 prompt "
        f"tokens ({n_layers} hooked layers at {fixed_decode:.4f} / {fixed_prefill:.4f} ms each, "
        "every layer counted as its own).",
    ]
    if decode_step:
        two = target * set_decode / headroom
        reasons.append(
            f"{target} set(s) would take {two:.4f} ms per decode pass after the {headroom:g} "
            f"headroom: {100.0 * two / float(decode_step):.1f}% of this model's "
            f"{float(decode_step):g} ms decode step, against a ceiling of {100 * d_share:g}%."
        )
    else:
        reasons.append("This model's decode step time is not measured, so the decode share is "
                       "not checked.")
    if prefill_step:
        two = target * set_prefill / headroom
        reasons.append(
            f"{target} set(s) would take {two:.4f} ms per 1,000 prompt tokens after headroom: "
            f"{100.0 * two / float(prefill_step):.1f}% of this model's {float(prefill_step):g} ms "
            f"prefill step, against a ceiling of {100 * p_share:g}%."
        )
    else:
        reasons.append("This model's prefill step time is not measured, so the prefill share is "
                       "not checked.")
    if achieved == 0:
        reasons.append(
            "Even one reference set breaks a ceiling; 1 behaviour is recommended anyway, because "
            "the operator chose to monitor this model — expect the cost to show in its latency."
        )
    reasons.append(
        f"Recommended: {sets} behaviour(s); budgets {decode_budget:.2f} ms per decode pass and "
        f"{prefill_budget:.2f} ms per 1,000 prompt tokens ({sets} set(s) ÷ {headroom:g} headroom, "
        "rounded up to 0.01 ms)."
    )
    if estimated_rules:
        reasons.append(
            "Estimated: this measurement has no line for "
            + ", ".join(sorted(estimated_rules))
            + "; attention cost estimated from the dense line. Re-measure the model to replace "
            "the estimate."
        )
    wall = PER_PROBE_REQUEST_WALL_MS * n_probes * sets
    reasons.append(
        f"Not in the per-pass budget: about {PER_PROBE_REQUEST_WALL_MS:g} ms per probe per request "
        f"of recording work (2026-10-21 hardware record), about {wall:.0f} ms per request for "
        f"{sets} armed set(s) of {n_probes}."
    )
    would_change = [
        "a new capacity measurement of this model (a different card, precision or quantization)",
        "a change to the reference set shape, the target number of sets or either share ceiling",
        "the admission headroom",
    ]
    snapshot_for_digest = {k: snapshot[k] for k in sorted(snapshot)}
    digest = hashlib.sha256(canonical_json({
        "values": values, "measurement_id": measurement.id, "settings": snapshot_for_digest,
    }).encode("utf-8")).hexdigest()
    return {
        "model_id": getattr(model_row, "id", None),
        "values": values,
        "shares": {"decode_pct": pct(decode_budget, decode_step),
                   "prefill_pct": pct(prefill_budget, prefill_step)},
        "reasons": reasons,
        "reference_set": shape,
        "target_sets": target,
        "achieved_sets": achieved,
        "estimated": bool(estimated_rules),
        "measurement": {"id": measurement.id, "measured_at": measured_at, "devices": devices},
        "would_change": would_change,
        "per_request_wall_ms_estimate": round(wall, 2),
        "set_cost": {"decode_ms": round(set_decode, 6), "prefill_ms_per_1k": round(set_prefill, 6)},
        "recommendation_digest": digest,
    }
