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

from dataclasses import dataclass
from typing import Any, Optional

from millm.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ServingPlan:
    """Everything the four call sites need, derived once from one document.

    ``claimed_layers`` is defined AS the layers of ``members`` — not read from
    ``definition.layers()`` or the ``circuits.layers`` column. That identity is
    the structural fix for F14-R2-01: the layers a request snapshots and the
    layers its apply drives cannot drift, because they are the same set.
    """

    members: list[Any]
    intensity: float
    claimed_layers: frozenset[int]
    attached_layers: frozenset[int]
    #: The registry entries this plan was derived from, for the CLAIMED layers
    #: only. Carried so a consumer never has to re-read the registry: a second
    #: read is a drift window, because a detach between the two means the
    #: snapshot the plan reports and the entries a request saves and restores
    #: disagree (R1-08).
    claimed_entries: tuple[Any, ...] = ()

    @property
    def unattached_layers(self) -> frozenset[int]:
        """Claimed but not currently attached — the slice-fallback signal."""
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
        return getattr(circuit, "intensity", 0.0) if circuit is not None else 0.0

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

    def attached_layers(self) -> frozenset[int]:
        """Layers with an SAE attached right now, or empty with no registry.

        F18 R1-02: a registry failure returns EMPTY, which makes a circuit that
        IS steering read as `is_serveable=False` and every claimed layer read as
        unattached. The direction is deliberately safe — the echo predicate then
        withholds a rung header rather than attaching an evidence claim to an
        intervention it cannot confirm, and under-claiming is the right way to
        fail on an evidence surface.

        But it was SILENT, and the same empty set also feeds
        `unattached_layers`, which drives slice-fallback decisions. A registry
        that cannot be read is an operational fault, not an empty registry, so
        it is logged at WARNING. `state is None` stays quiet: that is a
        deliberate construction (no registry supplied), not a failure.
        """
        return frozenset(e.layer for e in self._entries())

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
        resolved = (
            intensity
            if intensity is not None
            else self.serving_intensity(definition, circuit)
        )
        claimed = self.claim_set(members)
        # ONE registry read for the whole plan (R1-08).
        entries = self._entries()
        return ServingPlan(
            members=members,
            intensity=resolved,
            claimed_layers=claimed,
            attached_layers=frozenset(e.layer for e in entries),
            claimed_entries=tuple(e for e in entries if e.layer in claimed),
        )
