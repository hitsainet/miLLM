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


def armed_probe_from_row(
    probe: Any, *, encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
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

        armed = armed_probe_from_row(probe, encoder=encoder)

        # ── 4. parity — the only gate that costs a forward pass ─────────────────────
        tolerance = max(
            float((probe.definition.get("test_vectors") or {}).get("tolerance", 0.0) or 0.0),
            settings.PROBE_PARITY_TOLERANCE,
        )
        parity = ProbeParityEngine(forward).run(
            armed, probe.definition, tolerance=tolerance, tokenizer=tokenizer
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
            raise ProbeParityFailedError(
                "This build does not reproduce the scores miStudio recorded for this probe",
                details=details,
            )

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
