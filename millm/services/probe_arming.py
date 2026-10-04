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

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import torch

from millm.core.config import settings
from millm.core.errors import (
    ProbeDtypeMismatchError,
    ProbeLimitError,
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
from millm.services.probe_parity import NOT_COMPARABLE_SCOPE, ProbeParityEngine
from millm.services.probe_scope import (
    RUNTIME_SCORABLE_SCOPES,
    SCOPES,
    WINDOWS,
    scope_is_runtime_scorable,
)
from millm.services.probe_runtime import ArmedProbe, ProbeRuntimeState

logger = logging.getLogger(__name__)


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
#: bar — it answers "is THIS message high-stakes?" on a client that resends the conversation. An
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
    )


class ProbeArmingService:
    """Runs the gates and, if they all pass, arms the probe."""

    def __init__(self, repository: Any, state: Optional[ProbeRuntimeState] = None) -> None:
        self.repository = repository
        self.state = state or ProbeRuntimeState()

    async def arm(
        self,
        probe: Any,
        *,
        model: Any,
        loaded: LoadedIdentity,
        forward: Callable[[torch.Tensor, Any], None],
        tokenizer: Any = None,
        acknowledge_below_rung2: bool = False,
        reason: str = "",
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        windows: Any = None,
        by: str = "operator",
    ) -> ArmedProbe:
        # ── 1. the limit ────────────────────────────────────────────────────────────
        if not probe.armed and await self.repository.count_armed() >= settings.PROBE_MAX_ARMED:
            raise ProbeLimitError(
                f"{settings.PROBE_MAX_ARMED} probes are already armed; disarm one first",
                details={"max_armed": settings.PROBE_MAX_ARMED},
            )

        # ── 2. identity ─────────────────────────────────────────────────────────────
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
            raise ProbeDtypeMismatchError(
                f"This probe reads the right model at the wrong precision ({said}). {remedy}",
                details=report.as_details(),
            )
        if not report.ok:
            raise ProbeModelMismatchError(
                "This probe was fitted on a different model than the one loaded",
                details=report.as_details(),
            )

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
        if not scope_is_runtime_scorable(probe.scope):
            raise ProbeScopeUnverifiableError(
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
        parity = ProbeParityEngine(forward).run(
            for_parity, probe.definition, tolerance=tolerance, tokenizer=tokenizer,
            loaded_dtype=loaded.dtype,
            loaded_quantization=loaded.quantization,
        )
        await self.repository.update(probe, parity=parity.as_details())
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

        # ── all gates passed ────────────────────────────────────────────────────────
        self.state.arm(armed, model)
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
        await self.repository.update(
            probe,
            armed=True,
            paused_reason=None,
            arm_acknowledgement=acknowledgement,
            # ⚠ WHICH BAR THE OPERATOR ARMED AGAINST. The acknowledgement records a RUNG, which
            # is right — consent is to the evidence, and a re-cut changes neither the rung nor
            # the evidence. But a bar can now move under an armed probe, so without this the
            # tile cannot say "armed under revision 2; now judging at revision 4" at all, and
            # consent given against one operating point carries silently to another.
            armed_threshold_revision=int(getattr(probe, "threshold_revision", 1) or 1),
        )
        logger.info(
            "probe_armed id=%s name=%s layer=%s rung=%s parity_max_diff=%s",
            probe.id,
            probe.name,
            probe.layer,
            probe.rung,
            parity.max_abs_diff,
        )
        return armed

    async def disarm(self, probe: Any, reason: str = "operator") -> bool:
        removed = self.state.disarm(probe.id)
        await self.repository.update(probe, armed=False, paused_reason=reason)
        logger.info("probe_disarmed id=%s reason=%s", probe.id, reason)
        return removed

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


#: Why a probe stopped scoring, as recorded on its row. ⚠ `probes.paused_reason` is VARCHAR(64), and
#: the first unload reason was 70 characters: Postgres refused the write, the helper logged it, and
#: every row went on claiming to be armed. SQLite — the unit-test database — does not enforce the
#: length, so the suite was green. The helper now refuses an over-long reason before any write.
UNLOAD_REASON = "disarmed: the model was unloaded — re-arm after loading"
HANG_REASON = "disarmed: a generation thread hung"


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
            armed = (
                await session.execute(select(Probe.id).where(Probe.armed.is_(True)))
            ).scalars().all()
            if not armed:
                return []
            await session.execute(
                sa_update(Probe).where(Probe.armed.is_(True)).values(
                    armed=False, paused_reason=paused_reason
                )
            )
            await session.commit()
            logger.info("%s count=%s probes=%s", event, len(armed), sorted(armed))
            return sorted(armed)
    except Exception as e:  # noqa: BLE001 - bookkeeping must not break the caller
        logger.error(
            "%s_failed error=%s — armed rows may outlive their hooks; status cross-checks the "
            "live registry and reports them stale", event, e, exc_info=True,
        )
        return []
