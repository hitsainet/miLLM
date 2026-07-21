"""The ONE derivation of a circuit's serving plan (Feature 18 / BR-002).

WHY THIS MODULE EXISTS.

Four call sites independently derived "which members does this circuit steer,
at what intensity, over which layers":

* ``CircuitService._serve_full`` — activation
* ``CircuitService.set_intensity`` — the operator dial
* ``InferenceService`` per-request dial — the OpenAI ``steering_intensity``
  extension
* ``InferenceService._steering_circuit_uncached`` — the echo predicate that
  decides whether a response carries a rung header

Three flattened members through the same static helper and one asked a
different question of the same document. That is four places that must agree
about an operator-visible claim, kept in agreement by nothing but care — and
Feature 14's review rounds found them disagreeing twice:

* **F14-R1-01** — the dial resolved intensity from the DB column while
  activation used the document's authored budget, so a circuit authored at 150
  dialled to λ=1.0 served 100. The authored value is the truth; the column is a
  cache.
* **F14-R2-01** — the dial snapshotted the layers in ``circuits.layers`` while
  the apply drove the layers its MEMBERS claim. Any layer in one and not the
  other was dialled and never restored: a per-request override leaking
  permanently into global state.

Both were fixed at their own call site. The shape that produced them was not.
This module makes the plan a single object derived once, so "the four sites
agree" becomes a property of the code rather than a promise about it.

WHAT IT DELIBERATELY DOES NOT DO.

It does not apply steering. ``SAEService.set_circuit_steering`` owns the apply,
the lock and the epoch, and is untouched by this feature. It does not combine
``budget`` with ``sign``: a NEGATIVE strength is already directional, and
``_directional_budget`` is the one place that rule is implemented. Carrying
both fields untouched is what keeps the sign from being applied twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from millm.core.logging import get_logger

logger = get_logger(__name__)

#: No serving intensity could be derived — the document declares no budget and
#: no circuit row was supplied. A distinguishable value rather than 0.0, which
#: means "serve nothing" and would make a missing basis look like a deliberate
#: off switch (R1-12).
UNSET_INTENSITY = float("nan")


@dataclass(frozen=True)
class ServingPlan:
    """Everything the four call sites need, derived once from one document.

    ``claimed_layers`` is defined AS the layers of ``members`` — not read from
    ``definition.layers()`` or the ``circuits.layers`` column. That identity is
    the structural fix for F14-R2-01: the layers a request snapshots and the
    layers its apply drives cannot drift, because they are the same set.
    """

    #: R2-12: a TUPLE, not a list. The dataclass is frozen, but a frozen
    #: dataclass holding a mutable list is only half frozen: appending to
    #: `members` broke the `claimed_layers == member layers` identity — the
    #: exact invariant F18 exists to make structural — while every field still
    #: reported its original value. Four consumers share this object.
    members: tuple[Any, ...]
    intensity: float
    claimed_layers: frozenset[int]
    attached_layers: frozenset[int]
    #: The registry entries this plan was derived from, for the CLAIMED layers
    #: only. Carried so a consumer never has to re-read the registry: a second
    #: read is a drift window, because a detach between the two means the
    #: snapshot the plan reports and the entries a request saves and restores
    #: disagree (R1-08).
    #:
    #: R2-03, recorded precisely: the TUPLE is frozen, the ENTRIES are live
    #: references. A detach after the plan is built leaves stale handles here,
    #: and mutating an entry mutates what this reports. Verified by execution.
    #:
    #: NOT deep-copied, deliberately. The entries carry `LoadedSAE` objects
    #: holding GPU tensors; copying them per request would be absurd, and the
    #: consumer needs the live SAE to read and restore its steering values.
    #: The dial's FIRST action is to copy the values it needs into plain dicts
    #: (`saved_layers`), so the exposure is the few statements between
    #: `plan_for` and that copy — strictly narrower than the two-read window
    #: this replaced, and narrower than the pre-F18 code, which re-read the
    #: registry even later.
    #:
    #: A consumer that holds a plan across an await MUST NOT assume these are
    #: still attached.
    claimed_entries: tuple[Any, ...] = ()

    @property
    def has_intensity(self) -> bool:
        """False when no serving intensity could be derived.

        A consumer that is about to APPLY must check this: `UNSET_INTENSITY` is
        NaN, so using it silently produces NaN steering values rather than
        raising — worse than either a crash or a zero.
        """
        return self.intensity == self.intensity  # NaN is the only self-inequality

    @property
    def unattached_layers(self) -> frozenset[int]:
        """Claimed but not currently attached.

        R1-15: this was documented as "the slice-fallback signal" and has ZERO
        production callers — the sixth declared-but-unwired mechanism in this
        arc. Investigated rather than wired: the slice-fallback decision uses
        `assess_compatibility`, whose verdicts already mean "attached AND
        compatible", which is strictly stronger than attachment alone. Wiring
        this there would replace a better signal with a worse one.

        KEPT, with the false claim removed, because it is the honest complement
        of `attached_layers` on a frozen plan and costs nothing. It is NOT a
        slice-fallback input, and a future caller should know that before
        reaching for it. Deliberately untested beyond its arithmetic: an
        assertion on an unused property passes for the wrong reason.
        """
        return self.claimed_layers - self.attached_layers

    @property
    def is_serveable(self) -> bool:
        """True when this circuit can actually steer right now.

        Requires members AND at least one claimed layer with an attached SAE.
        The echo predicate uses this to decide whether a response may carry a
        rung header: claiming evidence for a circuit that is not steering would
        attach an evidence claim to an intervention that never happened.
        """
        return bool(self.members) and bool(
            self.claimed_layers & self.attached_layers
        )


class CircuitSteeringEngine:
    """Derives a :class:`ServingPlan`. Holds no repository, loader or emitter.

    One optional argument, the attachment registry, because the only thing
    beyond the document that a plan depends on is which layers currently have
    an SAE. Everything else is a pure function of the definition — which is
    what makes the four call sites able to share it without dragging
    request-scoped dependency injection onto the inference hot path.
    """

    def __init__(self, state: Optional[Any] = None) -> None:
        self._state = state

    # ------------------------------------------------------------------
    # The flattening — moved verbatim from CircuitService._serving_members
    # ------------------------------------------------------------------

    @staticmethod
    def serving_members(definition: Any) -> list[Any]:
        """Flatten the circuit's members into the Feature 12 serving shape.

        A ``cluster_ref`` contributes its frozen ``expanded_members`` AND its
        own ``feature`` when both are present — taking only one silently
        dropped authored members from the intervention. Duplicates on a
        ``(layer, feature_idx)`` are collapsed FIRST-WINS because the serving
        path rejects a repeated key outright.

        ``budget`` and ``sign`` are carried UNTOUCHED. A negative strength is
        already directional; combining them here would apply the sign twice
        (EC-18.3).
        """
        from millm.api.schemas.circuit import CircuitMember

        out: list[CircuitMember] = []
        seen: set[tuple[int, int]] = set()
        for m in definition.members:
            ref = definition.sae_for_layer(m.layer)
            sae_id = ref.mistudio_sae_id if ref else None
            sources = list(m.expanded_members or [])
            if m.feature is not None:
                sources.append(m.feature)
            for feat in sources:
                key = (m.layer, feat.feature_idx)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    CircuitMember(
                        feature_idx=feat.feature_idx,
                        layer=m.layer,
                        budget=feat.strength,
                        sign=feat.sign,
                        sae_id=sae_id,
                        label=feat.label,
                    )
                )
        return out

    @staticmethod
    def serving_intensity(definition: Any, circuit: Any = None) -> float:
        """The intensity this circuit serves at.

        The DOCUMENT's authored budget wins over the DB column when both are
        present. F14-R1-01: a circuit authored at 150 and dialled to λ=1.0
        served 100 because the dial read the column. The column is a cache of
        the last applied value; the document is what the author wrote.

        Note the explicit ``is not None`` — a budget of 0.0 is a legitimate
        authored value meaning "off", and a truthiness check would silently
        fall through to the column.

        F18 R1-01, a DELIBERATE divergence from the pre-move expression
        (``definition.budget.intensity if definition.budget else
        circuit.intensity``), recorded because a "verbatim move" that quietly
        changes behaviour is how F17 lost an O(log n) fix for three rounds:

            budget.intensity is None   old -> None   new -> the DB column

        The schema declares ``intensity: float = Field(1.0, ge=0.0, le=2.0)``,
        so a null is unreachable through a parsed document and this only shows
        on a hand-built object. The new behaviour is nonetheless the correct
        one: returning None here would propagate a null into the apply, where
        it multiplies a budget. Falling back to the column degrades to the last
        known-good value instead.
        """
        budget = getattr(definition, "budget", None)
        if budget is not None and getattr(budget, "intensity", None) is not None:
            return budget.intensity
        # R1-12: with no document budget AND no circuit row there is no basis
        # at all. The pre-move expression raised AttributeError here; returning
        # a bare 0.0 would mean "serve nothing", turning a loud failure into a
        # silent no-op on the path that decides how hard to steer.
        #
        # But raising is wrong too: `plan_for(definition)` with no circuit is a
        # legitimate members-only derivation — ten tests and several callers do
        # exactly that, and my first attempt at this fix broke all of them.
        # What is NOT legitimate is USING an intensity that was never derived.
        #
        # So the absence is represented, not guessed: `UNSET_INTENSITY` is a
        # float (arithmetic and comparisons still work) that is distinguishable
        # from a real 0.0, and `ServingPlan.has_intensity` lets a consumer that
        # needs one check before relying on it.
        if circuit is None:
            return UNSET_INTENSITY
        intensity = getattr(circuit, "intensity", None)
        return UNSET_INTENSITY if intensity is None else intensity

    @staticmethod
    def claim_set(members: list[Any]) -> frozenset[int]:
        """The layers this circuit claims, DEFINED as the layers of its members.

        Not ``definition.layers()`` and not the ``circuits.layers`` column.
        F14-R2-01 was exactly the gap between those and this; making it an
        identity rather than an agreement is the structural fix.
        """
        return frozenset(m.layer for m in members)

    # ------------------------------------------------------------------
    # The plan
    # ------------------------------------------------------------------

    def _entries(self) -> list[Any]:
        """The registry entries, read ONCE. See `ServingPlan.claimed_entries`."""
        state = self._state
        if state is None:
            return []
        try:
            return list(state.entries())
        except Exception:
            logger.warning("circuit_attachment_registry_unreadable", exc_info=True)
            return []

    # R1-16: `attached_layers()` was DELETED here.
    #
    # R1-08 made `plan_for` read the registry ONCE and compute
    # `attached_layers` inline from those entries, which left this method with
    # zero callers — a mutation returning `frozenset()` from it survived the
    # whole suite, correctly, because nothing reached it. A fix orphaning a
    # method is how the next reader gets two ways to ask one question and picks
    # the one that is no longer maintained.
    #
    # `ServingPlan.attached_layers` is the answer; `_entries()` is the single
    # read it comes from.

    def plan_for(
        self,
        definition: Any,
        circuit: Any = None,
        intensity: Optional[float] = None,
    ) -> ServingPlan:
        """Derive the whole plan from ONE member list.

        ``intensity`` overrides the derived value for a per-request dial, where
        λ comes from the request rather than the document. The members and the
        claim set are still derived here, so a dialled request and an
        activation cannot disagree about WHO is being steered — only about how
        hard.
        """
        members = self.serving_members(definition)
        if intensity is not None and not math.isfinite(intensity):
            # R2-04: NaN and +inf both SURVIVE `max(lo, min(hi, x))` and
            # resolve to the CEILING — a garbage dial silently producing the
            # most aggressive intervention available, not a crash and not a
            # no-op. `_resolve_circuit_intensity` has rejected non-finite
            # values since F14 R3 for exactly this reason; R1-12 then
            # introduced NaN into the sibling path with no such guard.
            raise ValueError(f"serving intensity must be finite: {intensity}")
        if intensity is not None and intensity < 0:
            # R1-13: a negative λ passed straight through and only the
            # downstream clamp saved it, so `ServingPlan.intensity` could hold
            # a value the system will never serve — a plan that does not
            # describe what happens. The dial's own validation should catch
            # this first; refusing here means a plan is always truthful.
            raise ValueError(f"serving intensity must not be negative: {intensity}")
        resolved = (
            intensity
            if intensity is not None
            else self.serving_intensity(definition, circuit)
        )
        claimed = self.claim_set(members)
        # ONE registry read for the whole plan (R1-08).
        entries = self._entries()
        return ServingPlan(
            members=tuple(members),
            intensity=resolved,
            claimed_layers=claimed,
            attached_layers=frozenset(e.layer for e in entries),
            claimed_entries=tuple(e for e in entries if e.layer in claimed),
        )
