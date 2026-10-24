"""Arming a probe: the four gates, in the order they are cheapest to fail (FR-24.4, 24.10, 24.11).

    1. the limit          — `PROBE_MAX_ARMED`, free
    2. identity           — can this probe read this model at all? free
    3. the evidence rung  — is the operator willing to monitor with this? free
    4. parity             — does this build reproduce miStudio's scores? runs the model

⚠ **THE ORDER IS A DECISION, NOT AN ACCIDENT.** Identity comes before the rung gate because there
is no point asking someone to acknowledge a weak probe that cannot run on the loaded model
anyway — the useful message is "wrong model", not "please confirm you accept rung 1". Parity comes
last because it is the only gate that costs a forward pass, and a probe that fails any earlier gate
would have spent it for nothing.

Only after all four does anything get hooked. A probe that is half-armed — persisted as armed with
no hook, or hooked with no row — is worse than one that refused, because status would describe a
monitor that is not monitoring.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import weakref
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

import torch

from millm.core.config import settings
from millm.core.errors import (
    BehaviourLimitError,
    ProbeBudgetError,
    ProbeDtypeMismatchError,
    ProbeLimitError,
    ProbeModelChangedError,
    ProbeModelMismatchError,
    ProbeParityFailedError,
    ProbeScopeUnverifiableError,
    UnvalidatedProbeError,
)
from millm.core.probe_evidence import (
    needs_arm_acknowledgement,
    probe_rung_language,
    probe_rung_next_step,
)
from millm.ml.probe_head import ProbeHead
from millm.services.probe_identity import LoadedIdentity, check_identity
from millm.services.probe_parity import NOT_COMPARABLE_SCOPE, ProbeParityEngine, model_summary
from millm.services.probe_scope import (
    RUNTIME_SCORABLE_SCOPES,
    SCOPES,
    WINDOWS,
    scope_is_runtime_scorable,
)
from millm.services import probe_cost
from millm.services.probe_runtime import (
    ArmedProbe,
    DecisionFacts,
    ProbeDescriptor,
    ProbeRuntimeState,
)

logger = logging.getLogger(__name__)


def _model_dtype(model: Any) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except Exception:  # noqa: BLE001 - a stand-in model with no parameters: score in float32
        return torch.float32


def _budget() -> dict[str, float]:
    return {
        "decode_ms_per_pass": float(settings.PROBE_MAX_OVERHEAD_MS_PER_PASS),
        "prefill_ms_per_1k": float(settings.PROBE_MAX_PREFILL_MS_PER_1K),
        "headroom": float(settings.PROBE_BUDGET_HEADROOM),
    }


def over_budget(projected: dict[str, float], budget: Optional[dict[str, float]] = None) -> bool:
    """⚠ STRICT `>`: a set that lands exactly on headroom × budget is admitted (FR-37.3)."""
    b = budget or _budget()
    return (
        projected["decode"] > b["decode_ms_per_pass"] * b["headroom"]
        or projected["prefill"] > b["prefill_ms_per_1k"] * b["headroom"]
    )


def head_from_definition(
    definition: dict[str, Any], *, dtype: torch.dtype = torch.float32
) -> ProbeHead:
    """Build the runtime head from a stored definition.

    float32 regardless of the model's dtype: the head is a handful of kilobytes, and scoring in
    half precision would introduce a difference from miStudio's float32 fit that parity would then
    have to absorb — spending the tolerance budget on a saving nobody needs.
    """
    head = definition["head"]
    query = head.get("attention_query")
    return ProbeHead(
        weight=torch.tensor(head["weights"], dtype=dtype),
        bias=float(head.get("bias", 0.0)),
        mean=torch.tensor(head["norm_mean"], dtype=dtype),
        std=torch.tensor(head["norm_std"], dtype=dtype),
        attention_query=torch.tensor(query, dtype=dtype) if query else None,
        layer=int(definition["read"]["layer"]),
    )


#: What a probe reports when the operator names no windows. All three by decision (2026-09-30):
#: `all` keeps continuity with every event recorded before this existed, and having it beside the
#: other two makes the dilution the feature exists to fix directly visible.
#: `last_user` joins the default (2026-10-04) ONLY for a probe that carries its own `last_user`
#: bar — it answers "is THIS message `<behaviour>`?" on a client that resends the conversation. An
#: older probe has no such bar, so the window would fire provisionally against the global one:
#: new alerts on traffic that was quiet yesterday (review round 1, M4).
BASE_DEFAULT_WINDOWS: tuple[str, ...] = ("all", "prompt", "response")


def resolve_windows(
    requested: Any, *, probe_scope: str, calibrated: Any = ()
) -> tuple[str, ...]:
    """The windows a probe will report, validated.

    ⚠ THIS IS A REPORTING CHOICE, NOT THE PROBE'S SCOPE. `scope` is identity — what the probe was
    trained on, what its threshold was cut under, the only thing parity can verify — and nothing
    here may change it. A probe whose CONTRACT scope is `prompt` or `response` is still refused at
    arm time on reproducibility grounds, exactly as before.

    Order is preserved and duplicates collapse, so a caller asking for the same window twice gets
    one verdict rather than two identical rows.
    """
    if requested is None:
        return BASE_DEFAULT_WINDOWS + (("last_user",) if "last_user" in set(calibrated) else ())
    seen: list[str] = []
    for name in requested:
        if name not in WINDOWS:
            raise ValueError(f"unknown window {name!r}; known: {', '.join(WINDOWS)}")
        if name not in seen:
            seen.append(name)
    if not seen:
        # An explicit empty list means "just my own scope" — the pre-feature behaviour, reachable
        # deliberately rather than only by omission.
        return (probe_scope,)
    return tuple(seen)


def parity_scored_nothing(details: Any) -> bool:
    """True when parity compared NOTHING — every vector failed before producing a number.

    ⚠ EXTRACTED SO IT CAN BE TESTED BY BEHAVIOUR. This lived inline in `arm`, and a control that
    replaced its `if` with `if False:` left the suite green: the test scraped the source for the
    message text, which `if False:` leaves perfectly intact. A guard matched by the wrong
    occurrence is the recurring failure in this estate, and the recorded remedy is this one —
    extract the decision, unit-test it, and assert the CALL by walking the AST.
    """
    vectors = (details or {}).get("vectors") or []
    if not vectors:
        return True
    return all(v.get("max_abs_diff") is None for v in vectors)


def parity_refusal_message(details: Any) -> str:
    """What to tell the operator when parity refuses.

    ⚠ "COULD NOT SCORE ANYTHING" IS NOT "THE NUMBERS DISAGREE". Reported 2026-10-01: five arm
    attempts refused with "does not reproduce the recorded scores" while every vector carried
    `max_abs_diff: null` and `scored_tokens: 0`. Nothing had been compared; the number the
    message was about did not exist. The two failures send an operator to different places — one
    to the model identity, the other to whatever stopped the vectors being scored — so they must
    not share wording.
    """
    if not parity_scored_nothing(details):
        return "This build does not reproduce the scores miStudio recorded for this probe"
    reasons = sorted({
        str(v.get("reason"))
        for v in ((details or {}).get("vectors") or [])
        if v.get("reason")
    })
    tail = f" Reported reason: {', '.join(reasons)}." if reasons else ""
    return (
        "Parity could not score any test vector, so nothing was compared — this is not a "
        "disagreement about the numbers." + tail
    )


def length_bands_from_definition(definition: Any) -> list[dict[str, Any]]:
    """`decision.length_bands` from the document, or `[]`.

    ⚠ A PROBE'S SCORE DRIFTS WITH INPUT LENGTH, so one constant threshold is miscalibrated at
    every length but the one it was cut at. miStudio measured a realised FPR of 5.4x its 1%
    budget on the longest quartile of `anthropic_hh_balanced`, and recall falling 0.500 -> 0.297
    on `mental_health_balanced` — the latter being the direction that took a live monitor here
    silent on turn four of a real conversation while the person's own sentence still carried the
    probe's two highest-scoring tokens.

    Tolerant of shape for the same reason as `window_thresholds_from_definition`: this is another
    repository's document, and a malformed block should cost the per-length bars, not the arming.
    Entries without a usable threshold are dropped — but a torn table is worse than none, so the
    caller gets `[]` rather than a table with a hole in it.
    """
    decision = (definition or {}).get("decision") or {}
    return bands_from_block(decision.get("length_bands"))


def bands_from_block(bands: Any) -> list[dict[str, Any]]:
    """A band table from another repository's document, or `[]` — whole or not at all.

    Shared by the global table and every window's own (2026-10-04), so the two cannot disagree
    about what a usable table is.

    ⚠ 039 DISCREPANCY X-1 (open, for the operator and miStudio): miStudio 040 S2's
    `decision-precedence-cases.json` treats a null-threshold band as "matches, bar unchanged" and a
    closed last band as legal (a length outside every band keeps the global bar); this parser
    refuses either table WHOLE. Two vendored cases (`thin-band.other-band`, `band-gap.match`) are
    strict xfails in `test_decision_precedence_cases.py` until the rule is decided.
    """
    if not isinstance(bands, list) or not bands:
        return []
    out: list[dict[str, Any]] = []
    for entry in bands:
        if not isinstance(entry, dict):
            return []
        value = entry.get("threshold")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return []
        lo = entry.get("min_tokens")
        if not isinstance(lo, int) or isinstance(lo, bool) or lo < 0:
            return []
        hi = entry.get("max_tokens")
        if hi is not None and (not isinstance(hi, int) or isinstance(hi, bool)):
            return []
        out.append({"min_tokens": lo, "max_tokens": hi, "threshold": float(value)})
    # The last band must be open-ended, or the longest inputs — the ones the drift hurts most —
    # fall outside the table and silently inherit nothing.
    if out[-1]["max_tokens"] is not None:
        return []
    return out


def window_thresholds_from_definition(definition: Any) -> dict[str, float]:
    """`{window: threshold}` from `decision.windows`, for windows that placed a bar.

    ⚠ ABSENT IS NOT ZERO AND NOT THE TOP-LEVEL THRESHOLD. A window with no entry, or an entry
    whose threshold is null, is one this producer never calibrated — the caller then falls back to
    the probe's own threshold and marks the verdict provisional. Defaulting to `0.0` here would
    make such a window fire on half its input, silently and confidently.

    Tolerant of shape: the definition is another repository's document, and a malformed `windows`
    block should cost the per-window thresholds, not the arming.
    """
    decision = (definition or {}).get("decision") or {}
    windows = decision.get("windows")
    if not isinstance(windows, dict):
        return {}
    out: dict[str, float] = {}
    for name, entry in windows.items():
        if not isinstance(entry, dict):
            continue
        value = entry.get("threshold")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(name)] = float(value)
    return out


def window_length_bands_from_definition(definition: Any) -> dict[str, list[dict[str, Any]]]:
    """`{window: bands}` from `decision.windows[w].length_bands`, for windows that carry a usable
    table AND a bar of their own — bands refine a window's own bar, so without one they apply to
    nothing (2026-10-04)."""
    decision = (definition or {}).get("decision") or {}
    windows = decision.get("windows")
    if not isinstance(windows, dict):
        return {}
    bars = window_thresholds_from_definition(definition)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, entry in windows.items():
        if not isinstance(entry, dict) or str(name) not in bars:
            continue
        bands = bands_from_block(entry.get("length_bands"))
        if bands:
            out[str(name)] = bands
    return out


def descriptor_from_row(probe: Any) -> ProbeDescriptor:
    """What every result of this probe says about its detector and decision (038, D38.18).

    Read from the row ONCE, at arm time. The digests are the stored ones (computed at import, or
    by migration `032`); a row without them (one written before 032 by a test, or a definition
    that cannot be digested) is digested here from its own definition, never guessed.
    """
    from millm.core.probe_labels import behaviour_of
    from millm.services.probe_digest import IDENTITY_ALGORITHM, digests_of
    from millm.services.probe_scoring import parity_status

    definition = probe.definition or {}
    detector = getattr(probe, "detector_digest", None)
    decision_d = getattr(probe, "decision_digest", None)
    source = getattr(probe, "detector_digest_source", None)
    if not detector or not decision_d:
        computed_det, computed_dec = digests_of(definition)
        if not detector and computed_det:
            detector, source = computed_det, "computed"
        decision_d = decision_d or computed_dec
    behaviour = behaviour_of(definition) or {}
    decision = definition.get("decision") or {}
    model = definition.get("model") or {}
    return ProbeDescriptor(
        behaviour_key=behaviour.get("key"),
        behaviour_version=behaviour.get("version"),
        detector_digest=detector,
        detector_digest_source=(source or "computed") if detector else "unknown",
        identity_algorithm=(getattr(probe, "identity_algorithm", None) or IDENTITY_ALGORITHM)
        if detector else None,
        hook=(definition.get("read") or {}).get("hook_point") or "resid_post",
        basis=getattr(probe, "basis", None) or "unknown",
        target="positive_class",
        template_sha256=getattr(probe, "template_sha256", None),
        model_revision=getattr(probe, "revision", None),
        precision=model.get("load_dtype"),
        parity_status=parity_status(probe)["status"],
        decision=DecisionFacts(
            decision_digest=decision_d,
            decision_revision=getattr(probe, "decision_revision", None),
            calibration_id=getattr(probe, "threshold_calibration_id", None),
            target_fpr=decision.get("target_fpr"),
        ),
    )


def armed_probe_from_row(
    probe: Any,
    *,
    encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    windows: Any = None,
) -> ArmedProbe:
    """Resolve a stored row into the runtime shape, once, at arm time."""
    definition = probe.definition
    return ArmedProbe(
        probe_id=probe.id,
        name=probe.name,
        head=head_from_definition(definition),
        rule=probe.rule,
        rule_params=(definition.get("aggregation") or {}).get("params") or {},
        scope=probe.scope,
        layer=probe.layer,
        rung=probe.rung,
        rung_language=probe_rung_language(probe.rung),
        threshold=probe.threshold,
        encoder=encoder,
        windows=resolve_windows(
            windows, probe_scope=probe.scope,
            calibrated=window_thresholds_from_definition(definition).keys(),
        ),
        window_thresholds=window_thresholds_from_definition(definition),
        length_bands=length_bands_from_definition(definition),
        window_length_bands=window_length_bands_from_definition(definition),
        # Read once here, and from then on the runtime object is the authority — a re-cut writes
        # the row first and refreshes the registry second, so between the two the row is ahead.
        threshold_revision=int(getattr(probe, "threshold_revision", 1) or 1),
        # 038: built here and never again per request — no surface reads the row (D38.18).
        descriptor=descriptor_from_row(probe),
    )


def identity_refusal(probe: Any, loaded: LoadedIdentity) -> Optional[Exception]:
    """The identity gate's refusal for `probe` against the loaded model, or None.

    ONE definition, shared by arming and by stateless scoring (FR-27.5f): a probe fitted on
    another model, or at another precision, is refused by both with the same error naming every
    mismatched field. Extracted from `ProbeArmingService.arm` rather than copied (Feature 27).
    """
    report = check_identity(probe.definition.get("model") or {}, loaded)
    if not report.ok and {m["field"] for m in report.mismatches} <= {"load_dtype", "quantization"}:
        # Same model, different activations: a precision or quantization difference. Its own
        # sentence, because "fitted on a different model" points at the wrong fix.
        said = "; ".join(
            f"{m['field']}: fitted at {m['expected']}, loaded at {m['actual']}"
            for m in report.mismatches
        )
        fields = {m["field"] for m in report.mismatches}
        if "quantization" in fields:
            remedy = ("Load the model at the quantization the probe was trained under, or "
                      "rebuild the probe in miStudio.")
        else:
            # Same quantization, different precision: one side did not follow the shared rule
            # (docs/schemas/native-dtype-cases.json) — reloading cannot fix that.
            remedy = ("The quantization matches, so one side did not load by the shared "
                      "precision rule: rebuild the probe in miStudio.")
        return ProbeDtypeMismatchError(
            f"This probe reads the right model at the wrong precision ({said}). {remedy}",
            details=report.as_details(),
        )
    if not report.ok:
        return ProbeModelMismatchError(
            "This probe was fitted on a different model than the one loaded",
            details=report.as_details(),
        )
    return None


def scope_refusal(probe: Any) -> Optional[Exception]:
    """Arming's refusal of a scope this runtime cannot score, or None. Shared with scoring
    (FR-27.5g)."""
    if not scope_is_runtime_scorable(probe.scope):
        return ProbeScopeUnverifiableError(
            f"This probe's scope is {probe.scope!r}, and this runtime can only score "
            f"scope 'all' probes. Two separate things are missing and neither is a problem "
            f"with your build: nothing here restricts scoring to a scope's positions yet, so "
            f"the probe would score the whole request; and a narrower scope's recorded "
            f"scores cannot be reproduced for comparison, because the definition records "
            f"which tokens miStudio scored but not which of them its role mask selected. "
            f"Re-export the probe with scope 'all' to arm it here.",
            details={
                "scope": probe.scope,
                "runtime_scorable_scopes": sorted(RUNTIME_SCORABLE_SCOPES),
                "reason": "scope_not_runtime_scorable",
            },
        )
    return None


#: The arm lock (039 TD39-9): one per event loop, serialising `arm`, `arm_set`, `disarm` and
#: `disarm_set` commits. ⚠ PER LOOP, not module-global: an `asyncio.Lock` binds to the first loop
#: that awaits it, and the test suite runs a loop per test. One process serves the API (one
#: replica, `Recreate`, one uvicorn worker), so an in-process lock is sufficient —
#: `test_single_worker_assumption.py` keeps that precondition true.
_arm_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _arm_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _arm_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _arm_locks[loop] = lock
    return lock


def behaviour_unit(armed: Any) -> str:
    """What the count gate counts for one armed probe: its behaviour key, or — a probe with no
    behaviour counts as its own (FR-39.7) — `probe:<id>`. Read from 038's `ProbeDescriptor`."""
    key = getattr(getattr(armed, "descriptor", None), "behaviour_key", None)
    return key if isinstance(key, str) and key else f"probe:{armed.probe_id}"


def armed_behaviour_units(state: ProbeRuntimeState) -> set[str]:
    """The distinct behaviours armed in THE REGISTRY (never the rows, FTID §7.1)."""
    return {behaviour_unit(p) for p in state.armed()}


def current_pin() -> Optional[tuple[Any, ...]]:
    """The resident model's pin (`probe_scoring.model_pin`), or None when nothing is loaded."""
    from millm.ml.model_loader import LoadedModelState
    from millm.services.probe_scoring import model_pin

    current = LoadedModelState().current
    return model_pin(current) if current is not None else None


def cost_fingerprint(loaded: Any, device: str) -> dict[str, Any]:
    """What a `last_cost` was measured under: `{repo_id, quantization, dtype, devices[]}`."""
    return {
        "repo_id": getattr(loaded, "hf_id", None),
        "quantization": getattr(loaded, "quantization", None),
        "dtype": getattr(loaded, "dtype", None),
        "devices": [device],
    }


@dataclasses.dataclass(frozen=True)
class ArmPlan:
    """One probe that passed every per-probe gate: the evidence `commit` needs (FTID §3.2)."""

    probe: Any
    armed: ArmedProbe
    parity: Any
    gate_ms: dict[str, float]
    first_on_layer: bool
    acknowledgement: Optional[dict[str, Any]] = None


def _outcome(probe: Any, gate: str, outcome: str, *, code: Optional[str] = None,
             cost: Optional[dict[str, Any]] = None,
             gate_ms: Optional[dict[str, float]] = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"probe_id": probe.id, "gate": gate, "outcome": outcome}
    if code is not None:
        entry["code"] = code
    if cost is not None:
        entry["cost"] = cost
    if gate_ms is not None:
        entry["gate_ms"] = gate_ms
    return entry


#: The gate a refusal class belongs to, for the per-probe outcome list of a set refusal.
_GATE_OF = {
    "PROBE_MODEL_MISMATCH": "identity", "PROBE_DTYPE_MISMATCH": "identity",
    "UNVALIDATED_PROBE": "rung", "PROBE_SCOPE_UNVERIFIABLE": "scope",
    "PROBE_PARITY_FAILED": "parity", "PROBE_BUDGET": "cost", "ADAPTER_APPLIED": "parity",
}


def _plan_cost(plan: ArmPlan) -> dict[str, Any]:
    cost = plan.armed.cost
    return {
        "decode": getattr(cost, "decode_ms", None),
        "prefill": getattr(cost, "prefill_ms_per_1k", None),
        "layer_fixed_decode": getattr(cost, "layer_fixed_ms", None),
        "layer_fixed_prefill": getattr(cost, "layer_fixed_prefill_ms_per_1k", None),
        "first_on_layer": plan.first_on_layer,
        "layer": plan.armed.layer,
        "device": getattr(cost, "device", None),
    }


class ProbeArmingService:
    """Runs the gates and, if they all pass, arms the probe — or a whole behaviour's set.

    039 (TD39-7, TD39-8) splits arming in two:

    * `check(probe)` runs the PER-PROBE gates in today's order (identity, rung, scope, parity, cost
      measurement) and writes only EVIDENCE: the parity report and `last_cost`. It touches no
      registry entry, no hook, and no `armed` / `paused_reason` / `arm_acknowledgement` / `cost`.
    * `commit(plans)` runs the SET gates once over every plan — the pin re-read, the behaviour
      count, the armed-probe backstop, the cost over the whole set — then installs every hook and
      writes every row in ONE transaction, removing every hook it installed if anything raises
      (D14: all or nothing).

    `arm(probe)` is `commit([check(probe)])`, so a single probe and a set share one path.
    """

    def __init__(self, repository: Any, state: Optional[ProbeRuntimeState] = None) -> None:
        self.repository = repository
        self.state = state or ProbeRuntimeState()

    # ── the public single-probe path (signature unchanged) ─────────────────────────────

    async def arm(self, probe: Any, **kwargs: Any) -> ArmedProbe:
        """`_arm` plus its lifecycle record (038 FR-38.20): `arm`, or `arm_refused` with the
        refusal's CODE. ⚠ The acknowledgement `reason` is never passed to the emitter."""
        from millm import telemetry

        try:
            armed = await self._arm(probe, **kwargs)
        except Exception as exc:
            telemetry.lifecycle_refused(probe, "arm_refused", exc)
            raise
        telemetry.emit_lifecycle(probe, "arm")
        return armed

    async def _arm(
        self,
        probe: Any,
        *,
        model: Any,
        loaded: LoadedIdentity,
        forward: Callable[[torch.Tensor, Any], None],
        executor: Callable[[Callable[[], Any]], Awaitable[Any]],
        tokenizer: Any = None,
        acknowledge_below_rung2: bool = False,
        reason: str = "",
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        windows: Any = None,
        by: str = "operator",
        limits: Any = None,
    ) -> ArmedProbe:
        # The RESIDENT model's effective limits (per-model, else global — `probe_limits`); the
        # route passes them. `None` is the globals, for callers that predate per-model limits.
        from millm.services.probe_limits import global_limits

        limits = limits or global_limits()
        # ⚠ THE PIN IS TAKEN BEFORE THE FIRST GATE (TD39-9): `commit` re-reads it, so a model
        # swapped while parity ran is refused `MODEL_CHANGED` instead of armed on the new model.
        pin = current_pin()
        # ── 1. the count gates, before anything costs a forward pass (cheapest first) ──
        # Advisory here; `commit` re-runs the same function under the lock, authoritatively.
        self._count_gates([probe.id], [_unit_of_row(probe)], limits)
        plan = await self.check(
            probe, model=model, loaded=loaded, forward=forward, executor=executor,
            tokenizer=tokenizer, acknowledge_below_rung2=acknowledge_below_rung2,
            encoder=encoder, windows=windows, by=by, reason=reason,
        )
        armed = await self.commit([plan], limits=limits, pin=pin, model=model)
        logger.info(
            "probe_armed id=%s name=%s layer=%s rung=%s parity_max_diff=%s",
            probe.id, probe.name, probe.layer, probe.rung, plan.parity.max_abs_diff,
        )
        return armed[0]

    # ── phase 1: per-probe gates, evidence only ────────────────────────────────────────

    async def check(
        self,
        probe: Any,
        *,
        model: Any,
        loaded: LoadedIdentity,
        forward: Callable[[torch.Tensor, Any], None],
        executor: Callable[[Callable[[], Any]], Awaitable[Any]],
        tokenizer: Any = None,
        acknowledge_below_rung2: bool = False,
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        windows: Any = None,
        by: str = "operator",
        reason: str = "",
    ) -> ArmPlan:
        """Identity, rung, scope, parity, cost measurement — in that order. Raises the refusal.

        ⚠ WRITES ONLY EVIDENCE (TD39-8): the parity report (as before) and `last_cost`. Nothing
        that says the probe is armed — the registry, `armed`, `paused_reason`,
        `arm_acknowledgement`, `cost` and the hooks are `commit`'s alone. A dry run is `check` with
        no `commit`, so a side effect here would be a dry run that arms (control: `check` sets
        `armed=True` → red).
        """
        # ── 2. identity ─────────────────────────────────────────────────────────────
        refusal = identity_refusal(probe, loaded)
        if refusal is not None:
            raise refusal

        # ── 3. the evidence rung ────────────────────────────────────────────────────
        if needs_arm_acknowledgement(probe.rung) and not acknowledge_below_rung2:
            raise UnvalidatedProbeError(
                f"This probe is rung {probe.rung} ({probe_rung_language(probe.rung)}); arming it "
                f"on live traffic requires an explicit acknowledgement",
                details={
                    "rung": probe.rung,
                    "rung_language": probe_rung_language(probe.rung),
                    "next_step": probe_rung_next_step(probe.rung),
                },
            )

        # ── 3.5 what this runtime can actually score ────────────────────────────────
        # ⚠ BEFORE PARITY, AND DELIBERATELY NOT A CONSEQUENCE OF IT. Nothing builds a scope mask
        # (`scored_mask` has no production caller and `ProbeRequestContext` gets `mask=None`), so
        # every armed probe scores every position. For `all` that is correct and is the only
        # reason this has never bitten. A `prompt`-scoped probe would score the model's own
        # output with weights that never saw one.
        #
        # Until 2026-09-28 the parity gate refused these scopes incidentally, because it cannot
        # replay them. Resting a safety property on another gate's side effect is how a later
        # "fix parity for prompt scope" would have quietly opened this one. Costs no forward pass,
        # so it belongs above parity regardless.
        refusal = scope_refusal(probe)
        if refusal is not None:
            raise refusal

        armed = armed_probe_from_row(probe, encoder=encoder, windows=windows)

        # ⚠ PARITY RUNS ON THE PROBE'S OWN SCOPE, NEVER THE OPERATOR'S WINDOWS, AND THAT NEEDS A
        # SEPARATE OBJECT.
        #
        # A test vector is a bare `token_ids` sequence: it has no prompt/response split, so the
        # parity context never calls `set_prompt_length`. Hand it a probe armed on
        # `['all','prompt','response']` and two of the three verdicts come back
        # `prompt_boundary_unknown` with zero scored tokens — permanently, by construction.
        #
        # Reproduced in the pod 2026-10-01:
        #     window=all       scored=True   n=12
        #     window=prompt    scored=False  reason=prompt_boundary_unknown  n=0
        #     window=response  scored=False  reason=prompt_boundary_unknown  n=0
        #
        # The engine selects the scope's verdict explicitly so those two cannot change a result,
        # but generating them at all is noise one selection bug away from refusing every arm —
        # and `POST /probes/{id}/parity` already passes `windows=[]` for exactly this reason.
        # That asymmetry between the two entry points was the defect: the same check reached two
        # different conclusions depending on which door you came through.
        for_parity = armed_probe_from_row(probe, encoder=encoder, windows=[])

        # ── 4. parity — the only gate that costs a forward pass ─────────────────────
        tolerance = max(
            float((probe.definition.get("test_vectors") or {}).get("tolerance", 0.0) or 0.0),
            settings.PROBE_PARITY_TOLERANCE,
        )
        # ⚠ THROUGH THE EXECUTOR, which the route sets to `InferenceService.run_model_work`: one
        # admission slot, every SAE suppressed in the thread that runs the forward (FR-27.6f-g,
        # T-73). Called directly, this forward ran beside a generation on the same model and read
        # a residual an earlier layer's steering had moved. REQUIRED, with no default, because a
        # default that skips the slot is the defect being fixed.
        started = time.perf_counter()
        parity = await executor(lambda: ProbeParityEngine(forward).run(
            for_parity, probe.definition, tolerance=tolerance, tokenizer=tokenizer,
            loaded_dtype=loaded.dtype,
            loaded_quantization=loaded.quantization,
            model=model_summary(loaded),
        ))
        parity_ms = (time.perf_counter() - started) * 1000.0
        await self.repository.update(probe, parity=parity.as_details())
        from millm import telemetry

        telemetry.emit_lifecycle(probe, "parity", outcome="ok" if parity.passed else "refused",
                                 code=None if parity.passed else "PROBE_PARITY_FAILED")
        if not parity.passed:
            details = parity.as_details()
            # ⚠ "NOTHING COULD BE COMPARED" IS NOT "THE NUMBERS DISAGREE".
            #
            # A `prompt` or `response` probe has no reproducible positions — the contract records
            # the token ids but not which ones miStudio's role mask selected — so every vector
            # comes back incomparable and nothing is scored. Reporting that as
            # PROBE_PARITY_FAILED tells the operator their build is wrong, and sends them to
            # debug a model, a precision and a hook point that are all fine. Reported
            # 2026-09-28 against an L6 probe exported with `scope: prompt`.
            unverifiable = [
                v for v in parity.vectors if v.reason == NOT_COMPARABLE_SCOPE
            ]
            if unverifiable and len(unverifiable) == len(parity.vectors):
                raise ProbeScopeUnverifiableError(
                    f"This probe's scope is {probe.scope!r}, and only 'all' can be verified "
                    f"against its recorded scores. miStudio scored it under a narrower role "
                    f"mask and the definition does not record which positions those were, so "
                    f"there is nothing to compare — this is not a disagreement about the "
                    f"numbers. Re-export the probe with scope 'all' to arm it here.",
                    details={**details, "scope": probe.scope},
                )
            # ⚠ "NOTHING COULD BE SCORED" IS NOT "THE NUMBERS DISAGREE", AND SAYING THE SECOND
            # SENDS AN OPERATOR HUNTING A MODEL MISMATCH THAT IS NOT THERE.
            #
            # Reported 2026-10-01: five arm attempts refused with "does not reproduce the
            # recorded scores" while every vector carried `max_abs_diff: null` and
            # `scored_tokens: 0`. Nothing had been compared at all. The number the message is
            # about did not exist, and the operator reasonably read it as "my build is wrong".
            raise ProbeParityFailedError(parity_refusal_message(details), details=details)

        # ── 5. cost — measured on the layer's own card, in the same slot (Feature 37) ──
        # ⚠ AFTER PARITY: a probe that cannot reproduce its scores is refused before its cost is
        # worth knowing. INSIDE THE EXECUTOR, like parity: the timings must not run beside a
        # generation, which would both disturb it and charge its kernels to the probe.
        first_on_layer = not any(
            p.layer == armed.layer and p.probe_id != armed.probe_id for p in self.state.armed()
        )
        started = time.perf_counter()
        try:
            from millm.ml.layer_resolution import layer_device

            device = layer_device(model, armed.layer)
            cost = await executor(lambda: probe_cost.measure(
                armed, device=device, d_model=int(probe.d_model),
                k=int(settings.PROBE_COST_SAMPLES), dtype=_model_dtype(model),
                first_on_layer=first_on_layer,
            ))
        except Exception as exc:  # noqa: BLE001 - never armed unmeasured (FPRD edge case)
            raise ProbeBudgetError(
                f"This probe's cost could not be measured on its layer's device, so it is not "
                f"armed: {exc}",
                details={"reason": "measurement_failed", "error": str(exc)[:300]},
            ) from exc
        cost_ms = (time.perf_counter() - started) * 1000.0
        # ⚠ 039 TD39-11: the measurement is EVIDENCE, kept on the row whatever happens next —
        # a set refused at commit, a dry run, a later disarm. The preview reads it back while its
        # fingerprint still matches the resident model.
        await self.repository.update(probe, last_cost={
            **cost.as_dict(), "fingerprint": cost_fingerprint(loaded, cost.device),
        })
        # The descriptor was built before parity ran; the stored report is now the arm-time
        # truth, so the results this probe produces carry it (038 FR-38.6, U11).
        from millm.services.probe_scoring import parity_status

        armed = dataclasses.replace(
            armed, cost=cost,
            descriptor=dataclasses.replace(
                armed.descriptor, parity_status=parity_status(probe)["status"]
            ) if armed.descriptor is not None else None,
        )
        acknowledgement = (
            {
                "by": by,
                "at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "rung": probe.rung,
            }
            if needs_arm_acknowledgement(probe.rung)
            else None
        )
        return ArmPlan(
            probe=probe, armed=armed, parity=parity,
            gate_ms={"parity": round(parity_ms, 3), "cost": round(cost_ms, 3)},
            first_on_layer=first_on_layer, acknowledgement=acknowledgement,
        )

    # ── the set gates (shared by commit, the advisory pre-check and the dry run) ─────────

    def _count_gates(self, probe_ids: list[str], units: list[str], limits: Any) -> None:
        """The behaviour count and the armed-probe backstop over the REGISTRY plus this set.

        ⚠ THE REGISTRY IS THE AUTHORITY, NEVER `repository.count_armed()` (rows), which counts
        probes and counts rows claiming `armed` after a restart (control: revert to
        `count_armed()` → red). Probes already armed in the registry are members, not additions.
        """
        armed_ids = {p.probe_id for p in self.state.armed()}
        armed_units = armed_behaviour_units(self.state)
        new_ids = [pid for pid in probe_ids if pid not in armed_ids]
        new_units = set(units) - armed_units
        limit = int(getattr(limits, "max_behaviours", settings.PROBE_MAX_ARMED_BEHAVIOURS))
        if new_units and len(armed_units | set(units)) > limit:
            raise BehaviourLimitError(
                f"{len(armed_units)} behaviour(s) already armed; the limit is {limit}"
                + (" (stale: measured under another model, precision or card)"
                   if getattr(limits, "stale", False) else ""),
                details={
                    "limit": "max_behaviours",
                    "value": limit,
                    "source": (getattr(limits, "sources", {}) or {}).get("max_behaviours",
                                                                         "global"),
                    "stale": bool(getattr(limits, "stale", False)),
                    "armed_behaviours": len(armed_units),
                    "size": len(new_ids),
                },
            )
        if new_ids and len(armed_ids) + len(new_ids) > limits.max_armed:
            raise ProbeLimitError(
                f"{limits.max_armed} probes is the most that may be armed; "
                f"{len(armed_ids)} are, and this adds {len(new_ids)}",
                details={"max_armed": limits.max_armed,
                         "source": limits.sources.get("max_armed", "global"),
                         "armed": len(armed_ids), "size": len(new_ids)},
            )

    def _cost_gate(self, plans: list[ArmPlan], limits: Any) -> dict[str, Any]:
        """The armed set plus EVERY plan, each (layer, device) fixed part charged once
        (`probe_cost.project` through `projected_cost(extra=[...])`). Returns the projection;
        raises `PROBE_BUDGET` naming every probe's own cost when it does not fit."""
        budget = limits.budget()
        projected = self.state.projected_cost(extra=[plan.armed for plan in plans])
        if over_budget(projected, budget):
            details: dict[str, Any] = {
                "budget": budget,
                "budget_sources": {k: limits.sources.get(k) for k in ("decode", "prefill")},
                "projected": projected,
                "stale": bool(getattr(limits, "stale", False)),
                "probes": [_outcome(plan.probe, "cost", "refused", cost=_plan_cost(plan))
                           for plan in plans],
            }
            if len(plans) == 1:
                # The single-probe shape every pre-039 consumer reads (037 FR-37.3).
                plan = plans[0]
                cost = plan.armed.cost
                share = probe_cost.project([cost]) if plan.first_on_layer else {
                    "decode": cost.decode_ms, "prefill": cost.prefill_ms_per_1k,
                }
                details["candidate"] = {
                    "decode": share["decode"], "prefill": share["prefill"],
                    "first_on_layer": plan.first_on_layer, "device": cost.device,
                }
            raise ProbeBudgetError(
                f"projected {projected['decode']:.3f} ms per decode pass and "
                f"{projected['prefill']:.3f} ms per 1,000 prompt tokens with "
                f"{'this probe' if len(plans) == 1 else f'these {len(plans)} probes'} (budget "
                f"{budget['decode_ms_per_pass']:g} and {budget['prefill_ms_per_1k']:g}, headroom "
                f"{budget['headroom']:g}); disarm a probe or raise the budget",
                details=details,
            )
        return projected

    # ── phase 2: the set, all or nothing ───────────────────────────────────────────────

    async def commit(
        self,
        plans: list[ArmPlan],
        *,
        limits: Any,
        pin: Optional[tuple[Any, ...]],
        model: Any,
    ) -> list[ArmedProbe]:
        """Arm every plan or none (D14). Under the arm lock, in FTDD §6.2's order."""
        async with _arm_lock():
            # ── 1. the pin: the model the gates checked is still the one loaded ────────
            now = current_pin()
            if now != pin:
                raise ProbeModelChangedError(
                    "The loaded model changed while the probes were being checked; nothing was "
                    "armed. Arm again on the model now loaded.",
                    details={"pin_before": _pin_text(pin), "pin_after": _pin_text(now),
                             "probe_ids": [plan.probe.id for plan in plans]},
                )
            # ── 2.–3. the behaviour count and the probe backstop ────────────────────────
            self._count_gates([plan.probe.id for plan in plans],
                              [behaviour_unit(plan.armed) for plan in plans], limits)
            # ── 4. the cost of the whole set ────────────────────────────────────────────
            self._cost_gate(plans, limits)
            # ── 5. hooks, then ONE transaction; undo every hook on any failure ──────────
            previous = {plan.probe.id: self.state.get(plan.probe.id) for plan in plans}
            installed: list[str] = []
            try:
                for plan in plans:
                    self.state.arm(plan.armed, model)
                    installed.append(plan.probe.id)
                await self.repository.update_many([
                    (plan.probe, {
                        "armed": True,
                        "paused_reason": None,
                        "arm_acknowledgement": plan.acknowledgement,
                        # ⚠ WHICH BAR THE OPERATOR ARMED AGAINST. The acknowledgement records a
                        # RUNG, which is right — consent is to the evidence, and a re-cut changes
                        # neither. But a bar can move under an armed probe, so without this the
                        # tile cannot say "armed under revision 2; now judging at revision 4".
                        "armed_threshold_revision":
                            int(getattr(plan.probe, "threshold_revision", 1) or 1),
                        # Feature 37: what it costs, on which card (by UUID). Cleared on disarm.
                        "cost": plan.armed.cost.as_dict(),
                    })
                    for plan in plans
                ])
            except BaseException:
                # ⚠ D14: NO HALF-ARMED SET. Every hook this call installed goes; a probe that
                # was already armed before this call gets its previous runtime shape back.
                for probe_id in reversed(installed):
                    before = previous.get(probe_id)
                    if before is None:
                        self.state.disarm(probe_id)
                    else:
                        self.state.arm(before, model)
                raise
        return [plan.armed for plan in plans]

    # ── sets ───────────────────────────────────────────────────────────────────────────

    async def check_all(
        self,
        probes: list[Any],
        *,
        forward_for: Callable[[Any], Any],
        encoder_for: Callable[[Any], Awaitable[Any]],
        **kwargs: Any,
    ) -> tuple[list[ArmPlan], list[dict[str, Any]], Optional[Exception]]:
        """`check` every probe; never stops at the first refusal, so a set refusal (and a dry run)
        names EVERY probe's outcome. Returns the plans, the per-probe outcomes and the first
        refusal (or None). `forward_for(probe)` and `encoder_for(probe)` build what one probe's
        parity needs (its layer's forward, its k-sparse encoder)."""
        plans: list[ArmPlan] = []
        outcomes: list[dict[str, Any]] = []
        first: Optional[Exception] = None
        for probe in probes:
            try:
                plan = await self.check(
                    probe, forward=forward_for(probe), encoder=await encoder_for(probe), **kwargs
                )
            except Exception as exc:  # noqa: BLE001 - every outcome is reported
                code = getattr(exc, "code", None)
                outcomes.append(_outcome(probe, _GATE_OF.get(code or "", "error"), "refused",
                                         code=code or type(exc).__name__))
                first = first or exc
                continue
            plans.append(plan)
            outcomes.append(_outcome(probe, "all", "passed", cost=_plan_cost(plan),
                                     gate_ms=plan.gate_ms))
        return plans, outcomes, first

    async def arm_set(
        self,
        selection: Any,
        *,
        limits: Any,
        model: Any,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Arm a behaviour's selection all or nothing (FR-39.2) — only the REMAINDER (FR-39.4):
        members already armed IN THE REGISTRY (`state.get`, never `row.armed`) are untouched.

        `dry_run` runs every gate for every probe — parity and cost included — and arms nothing;
        a set that would be refused answers with `would_refuse` instead of raising (FR-39.6), and
        its projection is the one `commit` would have refused with (one function, `_cost_gate`).
        """
        from millm import telemetry

        pin = current_pin()
        already = [p for p in selection.probes if self.state.get(p.id) is not None]
        remainder = [p for p in selection.probes if self.state.get(p.id) is None]
        units = [_unit_of_row(p) for p in remainder]
        result: dict[str, Any] = {
            "behaviour": {"key": selection.key, "version": selection.version},
            "already_armed": sorted(p.id for p in already),
            "dry_run": dry_run,
        }
        # Advisory count check before any forward pass (cheapest first); a dry run reports it.
        try:
            self._count_gates([p.id for p in remainder], units, limits)
        except (BehaviourLimitError, ProbeLimitError) as exc:
            if dry_run:
                plans, outcomes, first = await self.check_all(remainder, model=model, **kwargs)
                if isinstance(exc.details, dict):
                    exc.details = {**exc.details, "probes": outcomes}
                return {**result, "probes": outcomes, "armed": [],
                        "projected": self._projection_only(plans), "budget": limits.budget(),
                        "would_refuse": {"code": exc.code, "details": exc.details}}
            for probe in remainder:
                telemetry.lifecycle_refused(probe, "arm_refused", exc)
            raise
        plans, outcomes, first = await self.check_all(remainder, model=model, **kwargs)
        if dry_run:
            would: Optional[dict[str, Any]] = None
            if first is not None:
                would = {"code": getattr(first, "code", type(first).__name__),
                         "details": {**getattr(first, "details", {}), "probes": outcomes}}
            else:
                try:
                    self._count_gates([p.probe.id for p in plans],
                                      [behaviour_unit(p.armed) for p in plans], limits)
                    self._cost_gate(plans, limits)
                except (BehaviourLimitError, ProbeLimitError, ProbeBudgetError) as exc:
                    would = {"code": exc.code, "details": exc.details}
            return {**result, "probes": outcomes, "armed": [],
                    "projected": self._projection_only(plans), "budget": limits.budget(),
                    **({"would_refuse": would} if would is not None else {})}
        if first is not None:
            # ⚠ D14: one refusal refuses the set, and the refusal names every probe's outcome.
            if isinstance(getattr(first, "details", None), dict):
                first.details = {**first.details, "probes": outcomes,
                                 "behaviour": selection.key}
            for probe in remainder:
                telemetry.lifecycle_refused(probe, "arm_refused", first)
            raise first
        try:
            armed = await self.commit(plans, limits=limits, pin=pin, model=model) if plans else []
        except Exception as exc:
            if isinstance(getattr(exc, "details", None), dict) and "probes" not in exc.details:
                exc.details = {**exc.details, "probes": outcomes}
            for probe in remainder:
                telemetry.lifecycle_refused(probe, "arm_refused", exc)
            raise
        for plan in plans:
            telemetry.emit_lifecycle(plan.probe, "arm")
        logger.info("probe_set_armed behaviour=%s armed=%s already=%s", selection.key,
                    [a.probe_id for a in armed], result["already_armed"])
        return {**result, "probes": outcomes, "armed": sorted(a.probe_id for a in armed),
                "projected": self.state.projected_cost(), "budget": limits.budget(),
                "gate_ms": {plan.probe.id: plan.gate_ms for plan in plans}}

    def _projection_only(self, plans: list[ArmPlan]) -> dict[str, float]:
        return self.state.projected_cost(extra=[plan.armed for plan in plans])

    async def disarm(self, probe: Any, reason: str = "operator") -> bool:
        async with _arm_lock():
            removed = self.state.disarm(probe.id)
            await self.repository.update(probe, armed=False, paused_reason=reason, cost=None)
        logger.info("probe_disarmed id=%s reason=%s", probe.id, reason)
        from millm import telemetry

        # 038 FR-38.20. `reason` is free text (the operator's or a cause sentence): never passed.
        telemetry.emit_lifecycle(probe, "disarm")
        return removed

    async def disarm_set(self, selection: Any, reason: str = "operator") -> list[str]:
        """Disarm every member armed in the registry (FR-39.3). A layer hook another behaviour's
        probe still uses stays (`ProbeRuntimeState.disarm` keeps a shared layer's hook).
        ⚠ `last_cost` is never cleared: it is evidence, not arm state (TD39-11)."""
        from millm import telemetry

        async with _arm_lock():
            members = [p for p in selection.probes
                       if self.state.get(p.id) is not None or getattr(p, "armed", False)]
            for probe in members:
                self.state.disarm(probe.id)
            if members:
                # By id, as one statement: the selection's row objects may predate an arm that
                # committed since (see `ProbeRepository.set_by_ids`).
                await self.repository.set_by_ids(
                    [probe.id for probe in members], armed=False, paused_reason=reason, cost=None
                )
        for probe in members:
            telemetry.emit_lifecycle(probe, "disarm")
        logger.info("probe_set_disarmed behaviour=%s probes=%s", selection.key,
                    [p.id for p in members])
        return sorted(p.id for p in members)

    async def mark_all_disarmed(self, reason: str) -> int:
        """Record in the DATABASE that everything is disarmed, without touching the runtime.

        For callers that have already cleared the runtime — the hung-thread guard clears it
        synchronously, inside a generator, where awaiting a database write is not available at the
        moment the hooks must go. This closes the other half afterwards so the rows do not outlive
        the hooks claiming to be armed.
        """
        return await self.repository.disarm_all(reason)

    async def disarm_all(self, reason: str) -> int:
        """Disarm everything, in the runtime AND the database.

        ⚠ Both, always. Clearing only the runtime leaves rows claiming to be armed after a
        restart; clearing only the rows leaves hooks installed on a model nobody is tracking.
        """
        self.state.disarm_all(reason)
        return await self.repository.disarm_all(reason)


def _unit_of_row(probe: Any) -> str:
    """`behaviour_unit` for a ROW (before it has an `ArmedProbe`): the definition's key."""
    key = getattr(probe, "behaviour_key", None)
    return key if isinstance(key, str) and key else f"probe:{probe.id}"


def _pin_text(pin: Any) -> Optional[list[str]]:
    return None if pin is None else [str(part) for part in pin]


#: Why a probe stopped scoring, as recorded on its row. ⚠ `probes.paused_reason` is VARCHAR(64), and
#: the first unload reason was 70 characters: Postgres refused the write, the helper logged it, and
#: every row went on claiming to be armed. SQLite — the unit-test database — does not enforce the
#: length, so the suite was green. The helper now refuses an over-long reason before any write.
UNLOAD_REASON = "disarmed: the model was unloaded — re-arm after loading"
HANG_REASON = "disarmed: a generation thread hung"
#: What `main.disarm_probes_on_startup` writes (039 TD39-12; was an inline literal there, FTID
#: Discrepancy 9). ⚠ That path writes with a direct UPDATE, bypassing `mark_armed_rows_disarmed`'s
#: length guard, so `test_restart_reason.py` asserts this fits `paused_reason_limit()` itself.
RESTART_REASON = "disarmed by a restart — the hook did not survive the process"
#: The reasons a probe stopped WITHOUT the operator asking. "Armed before unload/restart" (FR-39.5)
#: is derived from these: a row whose `paused_reason` is one of them was armed when it stopped.
INVOLUNTARY_REASONS = frozenset({UNLOAD_REASON, HANG_REASON, RESTART_REASON})
#: The lifecycle record's `code` for each of those causes (038 FR-38.20).
LIFECYCLE_CODES = {UNLOAD_REASON: "model_unloaded", HANG_REASON: "thread_hung"}



def paused_reason_limit() -> int:
    """The column's declared length, read from the model rather than restated here."""
    from millm.db.models.probe import Probe

    return int(Probe.__table__.c.paused_reason.type.length)


async def mark_armed_rows_disarmed(session_factory: Any, paused_reason: str, *, event: str) -> list[str]:
    """Clear every `armed` probe row with `paused_reason`; return the ids. Never raises.

    ⚠ ONE WRITE FOR EVERY MOMENT THE HOOKS GO AWAY. `probes.armed` is a database column and the
    hook it describes lives in `ProbeRuntimeState`; anything that empties the second must clear
    the first, or status reports a monitor that is not monitoring. The same write
    `main.disarm_probes_on_startup` makes after a restart, for the two moments inside a running
    process: a model unload (`ModelService.unload_model`) and a hung generation thread. The
    hang guard read a `dependencies._probe_arming_service` global that was never defined, so its
    database half never ran (found 2026-10-04 with the unload defect).
    """
    from sqlalchemy import select, update as sa_update

    from millm.db.models.probe import Probe

    if len(paused_reason) > paused_reason_limit():
        # A programming error, not a runtime condition: raised so a test sees it, never swallowed.
        raise ValueError(
            f"paused_reason is {len(paused_reason)} characters; the column holds "
            f"{paused_reason_limit()}: {paused_reason!r}"
        )
    try:
        async with session_factory() as session:
            # The identity columns only (a lifecycle record needs nothing else), never the
            # definition JSON: this runs on every unload, with telemetry on or off.
            rows = (
                await session.execute(
                    select(Probe.id, Probe.behaviour_key, Probe.detector_digest,
                           Probe.threshold_revision, Probe.decision_revision,
                           Probe.decision_digest).where(Probe.armed.is_(True))
                )
            ).all()
            armed = [row.id for row in rows]
            if not armed:
                return []
            await session.execute(
                sa_update(Probe).where(Probe.armed.is_(True)).values(
                    armed=False, paused_reason=paused_reason, cost=None
                )
            )
            await session.commit()
            logger.info("%s count=%s probes=%s", event, len(armed), sorted(armed))
            # 038 FR-38.20: one `disarm` record per probe, with the CAUSE as a code
            # (`model_unloaded`, `thread_hung`) — never the `paused_reason` sentence.
            from millm import telemetry

            code = LIFECYCLE_CODES.get(paused_reason)
            for row in rows:
                telemetry.emit_lifecycle(row, "disarm", code=code)
            return sorted(armed)
    except Exception as e:  # noqa: BLE001 - bookkeeping must not break the caller
        logger.error(
            "%s_failed error=%s — armed rows may outlive their hooks; status cross-checks the "
            "live registry and reports them stale", event, e, exc_info=True,
        )
        return []
