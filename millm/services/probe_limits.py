"""Per-model probe limits: what they are, what drives them, and what is in force (operator 2026-10-21).

THREE LAYERS, each reported separately so an operator can see why a number is what it is:

1. **Measured** (`probe_capacity_measurements`, newest for the model's Hub repo and precision): the
   model's own step time — decode ms per generated token, prefill ms per 1,000 prompt tokens — and
   the probe cost LINES (own cost per probe + shared cost per hooked layer, dense and k-sparse).
2. **Configured** (`model_probe_limits`): the operator's per-model count backstop and two budget
   halves, each in milliseconds (`ms`) or as a percentage of that model's measured step time (`pct`).
3. **Effective**: what admission uses for the RESIDENT model — the configured value, else the global
   default. A `pct` budget with no measured step time cannot be converted; it falls back to the
   global default and says so (`source`), never a guessed number.

**Implied limits** come from (1) and (3): the largest N of the cheapest arrangement (dense probes on
one layer: one shared part + N × own) that fits each budget half, with and without the admission
headroom — "what drives the limit" in one line per half.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from millm.core.config import settings
from millm.core.errors import ValidationError

UNITS = ("ms", "pct")


@dataclass(frozen=True)
class EffectiveLimits:
    """The limits admission applies, with where each came from.

    039 (D13): `max_behaviours` is THE count gate, in behaviours (probe sets); `max_armed` stays as
    the armed-PROBE backstop. `stale` says the approval this rests on was measured under another
    fingerprint (model, quantization, dtype or card) — the values STAY IN FORCE (F39-8) and every
    surface says so. `approval` is who approved what, when, on which measurement.
    """

    max_armed: int
    decode_ms_per_pass: float
    prefill_ms_per_1k: float
    headroom: float
    sources: dict[str, str] = field(default_factory=dict)
    max_behaviours: int = 2
    stale: bool = False
    approval: Optional[dict[str, Any]] = None

    def budget(self) -> dict[str, float]:
        """The shape `probe_arming` checks against (and status reports)."""
        return {
            "decode_ms_per_pass": self.decode_ms_per_pass,
            "prefill_ms_per_1k": self.prefill_ms_per_1k,
            "headroom": self.headroom,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def global_limits() -> EffectiveLimits:
    return EffectiveLimits(
        max_armed=int(settings.PROBE_MAX_ARMED),
        decode_ms_per_pass=float(settings.PROBE_MAX_OVERHEAD_MS_PER_PASS),
        prefill_ms_per_1k=float(settings.PROBE_MAX_PREFILL_MS_PER_1K),
        headroom=float(settings.PROBE_BUDGET_HEADROOM),
        sources={"max_armed": "global", "decode": "global", "prefill": "global",
                 "max_behaviours": "global"},
        max_behaviours=int(settings.PROBE_MAX_ARMED_BEHAVIOURS),
    )


def _resolve(value: Optional[float], unit: Optional[str], baseline: Optional[float],
             default: float, what: str) -> tuple[float, str]:
    if value is None or unit is None:
        return default, "global"
    if unit == "ms":
        return float(value), "model (ms)"
    if baseline is None:
        return default, f"global — {value:g}% set, but this model's {what} step time is not measured"
    return float(value) / 100.0 * float(baseline), f"model ({value:g}% of {baseline:g} ms)"


def effective(limits: Any, measurement: Any, *, stale: bool = False,
              approval: Optional[dict[str, Any]] = None) -> EffectiveLimits:
    """Configured row (or None) over the globals, converting `pct` through the measured step time."""
    g = global_limits()
    if limits is None:
        return g
    decode, decode_src = _resolve(
        limits.decode_budget_value, limits.decode_budget_unit,
        getattr(measurement, "decode_ms_per_token", None), g.decode_ms_per_pass, "decode",
    )
    prefill, prefill_src = _resolve(
        limits.prefill_budget_value, limits.prefill_budget_unit,
        getattr(measurement, "prefill_ms_per_1k", None), g.prefill_ms_per_1k, "prefill",
    )
    max_behaviours = getattr(limits, "max_behaviours", None)
    return EffectiveLimits(
        max_armed=int(limits.max_armed) if limits.max_armed is not None else g.max_armed,
        decode_ms_per_pass=decode,
        prefill_ms_per_1k=prefill,
        headroom=g.headroom,
        sources={
            "max_armed": "model" if limits.max_armed is not None else "global",
            "decode": decode_src,
            "prefill": prefill_src,
            # 039: `recommended-approved` / `operator` / `global` (FR-39.7's three words).
            "max_behaviours": (getattr(limits, "source", None) or "operator")
            if max_behaviours is not None else "global",
        },
        max_behaviours=int(max_behaviours) if max_behaviours is not None else g.max_behaviours,
        stale=stale,
        approval=approval,
    )


def implied_n(line: Optional[dict[str, Any]], budget: float, half: str,
              headroom: float = 1.0) -> Optional[int]:
    """Largest N of this kind on ONE layer (one shared part + N × own) within `budget × headroom`."""
    if not line:
        return None
    own = line.get(f"{half}_own_ms" if half == "decode" else "prefill_own_ms_per_1k")
    fixed = line.get(f"{half}_fixed_ms" if half == "decode" else "prefill_fixed_ms_per_1k")
    if own is None or fixed is None or own <= 0:
        return None
    return max(int(math.floor((budget * headroom - fixed) / own + 1e-9)), 0)


def implied(measurement: Any, eff: EffectiveLimits) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("dense", "sparse"):
        line = getattr(measurement, kind, None) if measurement is not None else None
        if not line:
            continue
        out[kind] = {
            half: {
                "fits_budget": implied_n(line, b, half),
                "fits_with_headroom": implied_n(line, b, half, eff.headroom),
            }
            for half, b in (("decode", eff.decode_ms_per_pass), ("prefill", eff.prefill_ms_per_1k))
        }
        dn = out[kind]["decode"]["fits_with_headroom"]
        pn = out[kind]["prefill"]["fits_with_headroom"]
        if dn is not None and pn is not None:
            out[kind]["binding"] = "prefill" if pn < dn else "decode"
            out[kind]["n"] = min(dn, pn, eff.max_armed)
    return out


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def measurement_dict(m: Any) -> Optional[dict[str, Any]]:
    if m is None:
        return None
    return {
        "id": m.id, "repo_id": m.repo_id, "quantization": m.quantization, "device": m.device,
        "decode_ms_per_token": m.decode_ms_per_token, "prefill_ms_per_1k": m.prefill_ms_per_1k,
        "dense": m.dense, "sparse": m.sparse, "method": m.method, "source": m.source,
        "measured_at": _iso(m.measured_at),
        # 039 (TD39-3): the per-rule lines; None for a measurement taken before 039.
        "lines": getattr(m, "lines", None),
    }


def limits_dict(row: Any) -> dict[str, Any]:
    def budget(value: Any, unit: Any) -> Optional[dict[str, Any]]:
        return None if value is None else {"value": value, "unit": unit}

    if row is None:
        return {"max_armed": None, "decode_budget": None, "prefill_budget": None, "reason": None,
                "max_behaviours": None, "source": None, "set_by": None, "approved_at": None,
                "approved_measurement_id": None, "fingerprint": None}
    return {
        "max_armed": row.max_armed,
        "decode_budget": budget(row.decode_budget_value, row.decode_budget_unit),
        "prefill_budget": budget(row.prefill_budget_value, row.prefill_budget_unit),
        "reason": row.reason,
        "updated_at": _iso(row.updated_at),
        # 039 (D13, TD39-31): the behaviour limit and who approved what.
        "max_behaviours": getattr(row, "max_behaviours", None),
        "source": getattr(row, "source", None),
        "set_by": getattr(row, "set_by", None),
        "approved_at": _iso(getattr(row, "approved_at", None)),
        "approved_measurement_id": getattr(row, "approved_measurement_id", None),
        "fingerprint": getattr(row, "fingerprint", None),
    }


# ── 039: fingerprints and staleness (TD39-15) ────────────────────────────────────────────────


def measurement_fingerprint(measurement: Any, *, dtype: Optional[str]) -> dict[str, Any]:
    """What a measurement was taken under: `{repo_id, quantization, dtype, devices[]}`.

    `dtype` comes from the resident model at approval (a measurement row records no dtype; the
    measurement route refuses unless the model is resident, so the resident dtype IS the
    measurement's). `None` when the model was not resident at approval: then only the other three
    fields can make it stale.
    """
    device = str(getattr(measurement, "device", "") or "")
    return {
        "repo_id": getattr(measurement, "repo_id", None),
        "quantization": getattr(measurement, "quantization", None),
        "dtype": dtype,
        "devices": sorted(d for d in device.split("+") if d),
    }


def resident_fingerprint(model_row: Any) -> Optional[dict[str, Any]]:
    """The fingerprint of `model_row` AS IT IS LOADED NOW, or None when it is not resident."""
    from millm.ml.model_loader import LoadedModelState

    current = LoadedModelState().current
    if current is None or getattr(current, "model_id", None) != getattr(model_row, "id", None):
        return None
    devices: set[str] = set()
    try:
        from millm.services.probe_cost import device_label

        for parameter in current.model.parameters():
            devices.add(device_label(parameter.device))
    except Exception:  # noqa: BLE001 - a stand-in model with no parameters (tests)
        devices = set()
    return {
        "repo_id": getattr(model_row, "repo_id", None),
        "quantization": _quant(model_row),
        "dtype": getattr(current, "dtype", None),
        "devices": sorted(devices),
    }


def is_stale(approved: Optional[dict[str, Any]], model_row: Any) -> bool:
    """⚠ FINGERPRINTS, NEVER MEASUREMENT IDS (control: compare ids instead → red).

    A newer measurement under the same fingerprint is not staleness, it is "a newer recommendation
    is available"; a reload at another dtype or on another card is staleness even with no new
    measurement at all. Fields the approval did not record (`dtype: None`) cannot make it stale.
    """
    if not approved:
        return False
    if approved.get("repo_id") != getattr(model_row, "repo_id", None):
        return True
    if approved.get("quantization") != _quant(model_row):
        return True
    current = resident_fingerprint(model_row)
    if current is None:
        return False
    if approved.get("dtype") is not None and approved.get("dtype") != current.get("dtype"):
        return True
    if approved.get("devices") and current.get("devices") and \
            approved.get("devices") != current.get("devices"):
        return True
    return False


# ── persistence ──────────────────────────────────────────────────────────────────────────────


def _quant(model_row: Any) -> str:
    q = getattr(model_row, "quantization", None)
    return str(getattr(q, "value", q) or "")


async def latest_measurement(db: Any, model_row: Any) -> Any:
    from sqlalchemy import select

    from millm.db.models.probe_capacity import ProbeCapacityMeasurement as M

    if model_row is None or not getattr(model_row, "repo_id", None):
        return None
    result = await db.execute(
        select(M).where(M.repo_id == model_row.repo_id, M.quantization == _quant(model_row))
        .order_by(M.measured_at.desc(), M.id.desc()).limit(1)
    )
    return result.scalar_one_or_none()


async def limits_row(db: Any, model_id: int) -> Any:
    from millm.db.models.probe_capacity import ModelProbeLimits

    return await db.get(ModelProbeLimits, model_id)


def approval_of(limits: Any, measurement: Any) -> Optional[dict[str, Any]]:
    """Who approved what, when, on which measurement — and whether a newer one exists."""
    if limits is None or getattr(limits, "source", None) is None:
        return None
    approved_id = getattr(limits, "approved_measurement_id", None)
    latest_id = getattr(measurement, "id", None)
    return {
        "source": limits.source,
        "set_by": getattr(limits, "set_by", None),
        "approved_at": _iso(getattr(limits, "approved_at", None)),
        "measurement_id": approved_id,
        # Same fingerprint, newer measurement: not stale, but worth a look (TD39-15).
        "newer_recommendation_available": bool(
            approved_id is not None and latest_id is not None and latest_id != approved_id
        ),
    }


async def effective_for_model(db: Any, model_id: Optional[int]) -> EffectiveLimits:
    """Never raises: a read failure is the global defaults (admission must not depend on it)."""
    if model_id is None:
        return global_limits()
    try:
        from millm.db.models.model import Model

        row = await db.get(Model, model_id)
        limits = await limits_row(db, model_id)
        measurement = await latest_measurement(db, row)
        return effective(
            limits, measurement,
            stale=is_stale(getattr(limits, "fingerprint", None), row),
            approval=approval_of(limits, measurement),
        )
    except Exception:  # noqa: BLE001 - fall back to the globals, which are always valid
        return global_limits()


def resident_model_id() -> Optional[int]:
    from millm.ml.model_loader import LoadedModelState

    current = LoadedModelState().current
    return getattr(current, "model_id", None) if current is not None else None


def _check_budget(name: str, budget: Any) -> tuple[Optional[float], Optional[str]]:
    if budget is None:
        return None, None
    value, unit = budget.get("value"), budget.get("unit")
    if unit not in UNITS:
        raise ValidationError(f"{name}.unit must be one of {', '.join(UNITS)}",
                              details={"field": f"{name}.unit"})
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name}.value must be a number", details={"field": f"{name}.value"})
    lo, hi = (0.01, 10.0) if unit == "ms" else (0.01, 100.0)
    if not lo <= value <= hi:
        raise ValidationError(f"{name}.value must be within {lo:g}–{hi:g} {unit}",
                              details={"field": f"{name}.value", "minimum": lo, "maximum": hi})
    return value, unit


SOURCES = ("recommended-approved", "operator")


def _check_count(name: str, value: Any) -> Optional[int]:
    if value is not None and not (isinstance(value, int) and not isinstance(value, bool)
                                  and 1 <= value <= 64):
        raise ValidationError(f"{name} must be an integer within 1–64",
                              details={"field": name, "minimum": 1, "maximum": 64})
    return value


async def set_limits(db: Any, model_id: int, body: dict[str, Any], *,
                     model_row: Any = None) -> Any:
    """Replace a model's limits, or approve its recommendation. Validates first; one transaction.

    Two forms (039 TD39-31):

    * **Approval** (`source: "recommended-approved"`): requires `recommendation_digest`. The server
      RECOMPUTES the recommendation from the newest measurement and refuses
      `RECOMMENDATION_CHANGED` (with the new values) when the digest differs; otherwise it writes
      EXACTLY the recomputed values — never the body's — as `{value, unit: "ms"}`, with the
      snapshot, the fingerprint and the measurement it rests on. `max_armed` (the probe backstop)
      keeps its stored value.
    * **Override** (anything else, recorded `source: "operator"`): the body's values, each NULL =
      the global default, as before 039 — plus `max_behaviours`.

    Either way ONE history row is appended in the same transaction (append-only, by this rule).
    """
    from millm.core.errors import RecommendationChangedError
    from millm.db.models.probe_capacity import ModelProbeLimits, ModelProbeLimitsHistory

    source = body.get("source") or "operator"
    if source not in SOURCES:
        raise ValidationError(f"source must be one of {', '.join(SOURCES)}",
                              details={"field": "source"})
    row = await db.get(ModelProbeLimits, model_id)
    old = limits_dict(row) if row is not None else None
    now = datetime.now().astimezone()
    if source == "recommended-approved":
        from millm.services.probe_limit_recommendation import recommend, settings_snapshot

        digest = body.get("recommendation_digest")
        if not digest:
            raise ValidationError(
                "Approving a recommendation needs its recommendation_digest",
                details={"field": "recommendation_digest"},
            )
        measurement = await latest_measurement(db, model_row)
        current = recommend(model_row, measurement, settings_snapshot(settings))
        if current is None:
            raise ValidationError(
                "This model has no capacity measurement, so there is no recommendation to approve",
                details={"field": "source"},
            )
        if current["recommendation_digest"] != digest:
            # ⚠ NOTHING IS WRITTEN: what the operator saw is not what would be written.
            raise RecommendationChangedError(
                "The recommendation changed since it was read; review the new values",
                details={"values": current["values"],
                         "recommendation_digest": current["recommendation_digest"]},
            )
        values = current["values"]
        if row is None:
            row = ModelProbeLimits(model_id=model_id)
            db.add(row)
        row.max_behaviours = int(values["max_behaviours"])
        row.decode_budget_value, row.decode_budget_unit = float(values["decode_budget_ms"]), "ms"
        row.prefill_budget_value, row.prefill_budget_unit = (
            float(values["prefill_budget_ms"]), "ms")
        resident = resident_fingerprint(model_row)
        row.fingerprint = measurement_fingerprint(
            measurement, dtype=resident.get("dtype") if resident else None)
        row.recommendation = current
        row.approved_measurement_id = measurement.id
        row.approved_at = now
    else:
        max_armed = _check_count("max_armed", body.get("max_armed"))
        max_behaviours = _check_count("max_behaviours", body.get("max_behaviours"))
        dv, du = _check_budget("decode_budget", body.get("decode_budget"))
        pv, pu = _check_budget("prefill_budget", body.get("prefill_budget"))
        if row is None:
            row = ModelProbeLimits(model_id=model_id)
            db.add(row)
        row.max_armed, row.decode_budget_value, row.decode_budget_unit = max_armed, dv, du
        row.prefill_budget_value, row.prefill_budget_unit = pv, pu
        row.max_behaviours = max_behaviours
        # An override rests on no measurement: nothing to be stale against.
        row.fingerprint = None
        row.recommendation = None
        row.approved_measurement_id = None
        row.approved_at = now
    row.source = source
    row.set_by = (body.get("set_by") or "admin-ui")[:64]
    row.reason = (body.get("reason") or None)
    row.updated_at = now
    db.add(ModelProbeLimitsHistory(
        model_id=model_id, changed_at=now, old=old, new=limits_dict(row), source=source,
        set_by=row.set_by, reason=row.reason,
    ))
    await db.commit()
    return row


async def capacity(db: Any, model_row: Any) -> dict[str, Any]:
    """The whole picture for one model: measured, configured, effective, implied, and — when it is
    the resident model — the armed set against each budget."""
    measurement = await latest_measurement(db, model_row)
    row = await limits_row(db, model_row.id)
    eff = effective(row, measurement,
                    stale=is_stale(getattr(row, "fingerprint", None), model_row),
                    approval=approval_of(row, measurement))
    out: dict[str, Any] = {
        "measured": measurement_dict(measurement),
        "configured": limits_dict(row),
        "effective": eff.as_dict(),
        "implied": implied(measurement, eff),
        "armed": None,
    }
    if resident_model_id() == model_row.id:
        from millm.services.probe_runtime import ProbeRuntimeState

        state = ProbeRuntimeState()
        projected = state.projected_cost()
        n = len(state.armed())
        out["armed"] = {
            "count": n,
            "projected": projected,
            "share": {
                "count": n / eff.max_armed if eff.max_armed else None,
                "decode": projected["decode"] / (eff.decode_ms_per_pass * eff.headroom),
                "prefill": projected["prefill"] / (eff.prefill_ms_per_1k * eff.headroom),
            },
        }
    return out


# ── 039: the set preview (FR-39.9, FTID §7.3) ──────────────────────────────────────────────


def _estimated_cost(measurement: Any, rule: str, layer: int, device: str) -> Any:
    """A cost for a probe never measured under the resident fingerprint, from the model's
    measured per-rule line (`lines[rule]`, else the `mean` / `dense` line). None without one."""
    from millm.services.probe_cost import ProbeCost

    lines = getattr(measurement, "lines", None) or {}
    line = (lines.get(rule) if isinstance(lines, dict) else None) or (
        lines.get("mean") if isinstance(lines, dict) else None) or getattr(measurement, "dense",
                                                                           None)
    if not isinstance(line, dict):
        return None
    try:
        return ProbeCost(
            decode_ms=float(line["decode_own_ms"]),
            prefill_ms_per_1k=float(line["prefill_own_ms_per_1k"]),
            layer_fixed_ms=float(line["decode_fixed_ms"]),
            layer_fixed_prefill_ms_per_1k=float(line["prefill_fixed_ms_per_1k"]),
            layer=int(layer), device=device, k=0,
            measured_at=_iso(getattr(measurement, "measured_at", None)) or "",
        )
    except (KeyError, TypeError, ValueError):
        return None


async def preview(db: Any, model_row: Any, repository: Any, *, behaviour: Optional[str],
                  version: Any, probe_ids: Optional[list[str]]) -> dict[str, Any]:
    """The projected cost of arming a behaviour's set (or named probes) NOW, without arming.

    ⚠ THE SAME PROJECTION AS THE GATE: `ProbeRuntimeState.projected_cost(extra=[...])` — the
    function `commit` refuses with — over the registry plus each member's cost. A member's cost is
    its `last_cost` when that was measured under the resident fingerprint (same model, precision,
    dtype and card), else an estimate from the model's measured line for its rule (labelled
    `estimated`); a member with neither is named `unmeasured` and costs nothing here.
    """
    from types import SimpleNamespace

    from millm.ml.layer_resolution import layer_device
    from millm.services import probe_arm_bridge, probe_selection
    from millm.services.probe_arming import cost_fingerprint, over_budget
    from millm.services.probe_cost import cost_from_dict, device_label
    from millm.services.probe_runtime import ProbeRuntimeState

    loaded, model, _tokenizer = await probe_arm_bridge.loaded_identity(db)
    if behaviour is not None:
        selection = await probe_selection.select_behaviour(
            repository, behaviour, version=version, probe_ids=probe_ids, loaded=loaded)
        members = list(selection.probes)
    else:
        members = [p for p in [await repository.get(pid) for pid in (probe_ids or [])] if p]
    measurement = await latest_measurement(db, model_row)
    row = await limits_row(db, model_row.id)
    eff = effective(row, measurement, stale=is_stale(getattr(row, "fingerprint", None), model_row),
                    approval=approval_of(row, measurement))
    state = ProbeRuntimeState()
    candidates, per_probe = [], []
    for probe in members:
        device = device_label(layer_device(model, probe.layer))
        stored = probe.last_cost if isinstance(getattr(probe, "last_cost", None), dict) else None
        cost, source = None, "unmeasured"
        if stored and stored.get("fingerprint") == cost_fingerprint(loaded, device):
            cost, source = cost_from_dict(stored), "measured"
        if cost is None:
            cost = _estimated_cost(measurement, probe.rule, probe.layer, device)
            source = "estimated" if cost is not None else "unmeasured"
        per_probe.append({"probe_id": probe.id, "layer": probe.layer, "rule": probe.rule,
                          "source": source, "armed": state.get(probe.id) is not None,
                          "cost": cost.as_dict() if cost is not None else None})
        if cost is not None and state.get(probe.id) is None:
            candidates.append(SimpleNamespace(probe_id=probe.id, cost=cost))
    projected = state.projected_cost(extra=candidates)
    budget = eff.budget()
    return {
        "behaviour": behaviour,
        "probes": per_probe,
        "unmeasured": [p["probe_id"] for p in per_probe if p["source"] == "unmeasured"],
        "projected": projected,
        "budget": budget,
        "fits": not over_budget(projected, budget),
        "stale": eff.stale,
    }
