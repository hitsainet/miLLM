"""Circuit edge sensing — the matcher, its state, and the request context.

WHY THIS MODULE EXISTS (Feature 17 / BRD-MILLM-CIRCUITS-002 BR-001).

Feature 15 shipped edge sensing inside ``sae_wrapper.py``, where correctness
depended on N per-SAE position counters agreeing about an absolute coordinate
that no single component owned, and on a shared ring whose lifetime was managed
by whichever hook remembered to call it. Three review rounds produced eight
criticals, and **three of those eight share exactly that one root cause**:

* **R1-01** — a hook pruned the shared ring mid-pass, so the upstream layer
  destroyed the fires the downstream layer had not yet read. Cross-layer
  sensing went dark on ordinary traffic while status still reported "armed".
* **R1-03** — an early return skipped one SAE's offset advance, so its
  coordinates silently diverged from its siblings' for the rest of the request.
* **R3-01** — pruning was declared "request-level" in two consecutive rounds
  and wired in neither, the second time accompanied by a test named for the
  defect it failed to prevent.

Each was fixed individually. The shape that produced them was not. This module
moves position, ring lifetime and the event budget into ONE object owned by the
request, so those states become unrepresentable rather than test-guarded — the
difference between "correct because three comments keep being obeyed" and
"correct because there is nowhere else for the state to live".

It deliberately does not import ``sae_wrapper``: a module that cannot reach the
SAE cannot grow a second source of truth about where a request is (CTX-E1).
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

#: Floor for the per-pass fire budget. Below this even a tiny cap should still
#: tolerate an ordinarily busy pass.
_EDGE_FIRE_BUDGET_MIN = 2048

#: When a pass sheds, how many fired positions per COLUMN still feed the shared
#: ring. Bounds the upstream half, which is per-edge and was otherwise
#: unbounded — a shed pass at the contract's 200-edge maximum cost 544ms
#: because "cheap" only described the downstream half it skipped.
#:
#: Defined HERE, next to the matcher that enforces it, and imported by
#: sae_wrapper. It briefly existed as two independent literals in two files
#: with no shared source, which is how a shed threshold drifts silently.
_EDGE_SHED_POSITIONS_PER_COL = 64

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Specs and results
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class EdgeSpec:
    """One sensable edge, resolved against the SAEs actually attached.

    ``up_col``/``down_col`` are COLUMN OFFSETS into this SAE's armed member
    slice, not feature indices — the slice is what ``_W_enc_e`` selects, so the
    activation lookup must use the offset. ``up_feature_idx``/
    ``down_feature_idx`` keep the real indices for reporting.

    An edge whose endpoints live on different layers is sensed COOPERATIVELY:
    the upstream SAE records the fire into the circuit's ring and the
    downstream SAE matches against it. Both SAEs therefore hold the same
    EdgeSpec, and each uses only the half that belongs to its own layer.
    ``-1`` is the "not my half" sentinel; anything lower is a bug that would
    silently skip the edge rather than raise (F15 R2-07).
    """

    edge_key: str
    up_layer: int
    up_feature_idx: int
    up_col: int
    down_layer: int
    down_feature_idx: int
    down_col: int
    rung: int
    rung_language: str
    edge_type: Optional[str] = None


@dataclass
class SensedEdge:
    """One observed up→down firing within the lag window.

    An observation here is not causal evidence: it says the upstream member
    fired and the downstream partner then fired within ``token_lag`` tokens, in
    the authored direction. The rung carried on the row is the only statement
    about causality, and it comes from miStudio — never from having watched the
    edge fire.
    """

    edge_key: str
    up_layer: int
    up_feature_idx: int
    up_pos: int
    up_act: float
    down_layer: int
    down_feature_idx: int
    down_pos: int
    down_act: float
    token_lag: int
    phase: str
    rung: int
    rung_language: str
    edge_type: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────
# The ring
# ─────────────────────────────────────────────────────────────────────────


class EdgeFireRing:
    """Per-(request, circuit) record of upstream fires.

    A cross-layer edge cannot be detected inside one SAE: the upstream fire
    happens in layer L's hook and the downstream fire in layer M's, on
    different passes for decode tokens and different rows of the same pass for
    prefill. The ring is the shared state, keyed by ABSOLUTE token position so
    the two hooks agree on ordering regardless of which ran first.

    ONE RING PER (REQUEST, CIRCUIT), never per request (CTX-R1/R2). ``edge_key``
    is synthesised as ``{up_idx}@{up_layer}->{down_idx}@{down_layer}`` and is
    therefore NOT unique across circuits — two circuits can legitimately contain
    the same edge. A shared ring would let circuit A's upstream fire match
    circuit B's downstream fire and record an observation of an edge that fired
    in NEITHER. A fabricated observation on an evidence surface is categorically
    worse than a missed one, and Feature 19's concurrent circuits make it
    reachable rather than theoretical.
    """

    #: Per-edge upstream-fire retention. The ring cannot prune by position from
    #: inside a hook (see prune_before), so it bounds memory by count. Generous
    #: relative to any plausible lag window; the matcher filters by window.
    _MAX_FIRES_PER_EDGE = 512

    #: Consecutive passes with a missing reporter before the ring stops waiting
    #: for it (R3-01). A live layer reports on EVERY pass, including suppressed
    #: and quiet ones, so this only trips when a layer has genuinely stopped —
    #: never merely because a sibling is slow.
    _STALLED_REPORTER_PASSES = 64

    def __init__(self, max_lag: int):
        self._max_lag = max(1, int(max_lag))
        #: edge_key -> list of (abs_pos, activation), ascending by position.
        self._fires: dict[str, list[tuple[int, float]]] = {}
        #: layer -> position walked through, so the ring can prune to the
        #: SLOWEST layer without any hook knowing about its siblings.
        self._progress: dict[int, int] = {}
        self._last_pruned_at: int = 0
        #: Consecutive progress reports made while an expected reporter was
        #: missing (R3-01).
        self._unanswered: int = 0
        #: How many layers are expected to report progress for this circuit,
        #: or None until told. None means "unknown", and pruning then waits for
        #: a SECOND reporter — the old conservative behaviour — because a ring
        #: that assumed 1 would prune on the first report and destroy fires a
        #: sibling still needed (R1-01). The service calls `expect_layers` at
        #: begin, which is what lets a genuinely single-layer circuit prune.
        self._expected_layers: Optional[int] = None

    def expect_layers(self, count: int) -> None:
        """Tell the ring how many layers will report progress.

        Pruning must wait for the SLOWEST layer, and the ring cannot infer how
        many are coming. It previously approximated that with `len(_progress)
        < 2`, which meant a single-layer circuit never pruned at all — 512
        retained fires per edge instead of 4, measured. Dropping the guard
        instead resurrects R1-01, where the first layer to report prunes past
        fires the second still needs.
        """
        self._expected_layers = max(1, int(count))

    def record_up(self, edge_key: str, pos: int, act: float) -> None:
        fires = self._fires.setdefault(edge_key, [])
        fires.append((pos, float(act)))
        if len(fires) > self._MAX_FIRES_PER_EDGE:
            # Drop the OLDEST: match_down reports the newest antecedent, so
            # recent history is what matters.
            del fires[: len(fires) - self._MAX_FIRES_PER_EDGE]

    def match_down(self, edge_key: str, down_pos: int) -> Optional[tuple[int, float]]:
        """Newest upstream fire STRICTLY before ``down_pos``, within the lag.

        Strictly before: a same-position co-fire is co-activation, not an
        up→down sequence, and reporting it as one would overclaim direction.
        Newest-first because the closest antecedent is the most defensible
        attribution.

        ``fires`` is ascending, so bisect to the insertion point and walk back
        from there. Stepping over the tail one entry at a time was O(n) on the
        NORMAL cross-layer path — hooks run in layer order, so the upstream
        layer records its whole prefill before the downstream layer matches
        ascending (F15 R3).
        """
        fires = self._fires.get(edge_key)
        if not fires:
            return None
        # R1-08: this said `i = len(fires) - 1` — a linear backward walk — from
        # the F17 extraction until review round 1. The docstring above kept
        # describing the bisect while the code no longer did it, so reading the
        # function told you the opposite of what it ran. F15 R3's O(n)->O(log n)
        # fix was silently reverted by a change described as a pure move.
        #
        # Measured against a full 512-fire ring with a wide window, ASCENDING
        # (the normal cross-layer order the docstring names): 7.38ms/2000 vs
        # 0.29ms/2000 for a tail probe — 25.9x. The existing bound test only
        # ever probed the tail, where the linear walk terminates on iteration
        # one, so it could not fail.
        #
        # The `-inf` sentinel is load-bearing: it keeps the insertion point
        # strictly LEFT of any fire at `down_pos`, which is one of the two
        # guards enforcing strictly-before (see the pair-mutation control in
        # test_sensing_request_context.py).
        i = bisect.bisect_left(fires, (down_pos, float("-inf"))) - 1
        while i >= 0:
            pos, act = fires[i]
            if (down_pos - pos) > self._max_lag:
                break
            if pos < down_pos:
                return (pos, act)
            i -= 1
        return None

    def note_layer_progress(self, layer: int, through: int) -> None:
        """Record how far one layer has walked, and prune to the slowest.

        F15 R1 moved pruning out of the hooks and declared it request-level,
        then never wired a caller; R2 added service methods and ALSO never
        wired them. Third shape, and the one that works: the RING tracks each
        layer's progress, so it can prune to the slowest itself. No hook needs
        to know about its siblings — which is what made the previous two
        designs unwireable. Bounded by construction rather than by a caller
        remembering.
        """
        self._progress[layer] = through
        # R2-10: this used to `return` when only one layer had reported —
        # "a single layer: nothing to be slower than". But with one layer THAT
        # layer is the slowest, so pruning is both safe and correct, and
        # skipping it meant a single-layer circuit never pruned at all.
        # Measured on a 600-fire edge: one reporting layer retained 512 fires
        # (the per-edge hard cap), two retained 4. That is 128x the intended
        # memory on exactly the degraded path — and a single-layer circuit is a
        # legitimate configuration, not an edge case.
        #
        # Bounded either way, so this was never a leak; it is the difference
        # between the designed bound and the backstop.
        #
        # But simply dropping the guard resurrects R1-01: with two layers, the
        # FIRST to report would prune past fires the second still needs. The
        # ring cannot infer how many layers WILL report, so it is TOLD — see
        # `expect_layers`. Until every expected layer has reported, pruning
        # waits; a single-layer circuit therefore prunes immediately and a
        # multi-layer one still waits for its slowest member.
        expected = self._expected_layers
        if expected is None:
            # Unknown: fall back to the conservative rule. Never prune on a
            # single report, because a sibling may still be coming.
            if len(self._progress) < 2:
                return
        elif len(self._progress) < expected:
            # R3-01: waiting for the expected reporters is right, but waiting
            # FOREVER is not. `expect_layers` is fixed at begin, and a layer can
            # go dark AFTER that — an SAE detached mid-request, evicted, or
            # swapped. The ring then waits for a reporter that no longer exists
            # and never prunes: measured, 512 retained fires per edge after a
            # mid-request detach. R2-12's stall through a different door, and no
            # count computed at begin can close it.
            #
            # The wait is bounded by CONSECUTIVE UNANSWERED REPORTS, not by how
            # far the reporting layers have walked. Distance is the wrong
            # signal: a merely-slow sibling is exactly the case where one layer
            # races far ahead, so a distance bound prunes the history that
            # sibling still needs (R1-01, caught by two tests when tried).
            #
            # Counting reports is safe because a live layer reports on EVERY
            # pass, including suppressed and quiet ones (EC-17.1). A sibling
            # that has missed this many consecutive passes has stopped.
            self._unanswered += 1
            if self._unanswered < self._STALLED_REPORTER_PASSES:
                return
            logger.info(
                "edge_ring_pruning_without_all_reporters: expected=%s "
                "reporting=%s after %s passes — a layer stopped reporting",
                expected, len(self._progress), self._unanswered,
            )
        else:
            self._unanswered = 0
        slowest = min(self._progress.values())
        if slowest - self._last_pruned_at >= self._max_lag:
            self._last_pruned_at = slowest
            self.prune_before(slowest)

    def prune_before(self, pos: int) -> None:
        """Drop fires that can no longer match anything at or after ``pos``.

        MUST NOT be called from a hook. A hook cannot know whether a sibling
        layer still needs a fire — see note_layer_progress for the history.
        """
        cutoff = pos - self._max_lag
        for key, fires in list(self._fires.items()):
            kept = [f for f in fires if f[0] >= cutoff]
            if kept:
                self._fires[key] = kept
            else:
                del self._fires[key]

    def clear(self) -> None:
        self._fires.clear()
        self._progress.clear()
        self._last_pruned_at = 0


# ─────────────────────────────────────────────────────────────────────────
# The event budget
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class EventBudget:
    """Per-request observation budget, ATTRIBUTED PER CIRCUIT.

    F15's cap was per-SAE, so an N-layer circuit could emit N x cap events, and
    ``truncated`` was OR'd across layers onto every row — one saturated layer
    marked a whole request's observations truncated while other layers had not
    shed at all.

    Two rules the callers depend on:

    * ``try_spend`` returning False means the CALLER CONTINUES rather than
      returns. F15 R3-02 found the cap returning from the whole pass, which
      stopped a capped layer feeding the shared ring and blinded its uncapped
      siblings — the same starvation R2-03 had just fixed on the shed path.
    * Truncation is recorded against the LAYER THAT SHED, so the operator learns
      which layer lost data rather than being told the request did.
    """

    cap: int
    _spent: dict[str, int] = field(default_factory=dict)
    _truncated: dict[str, set[int]] = field(default_factory=dict)

    def try_spend(self, circuit_id: str, layer: int) -> bool:
        """Claim one observation slot. False ⇒ continue, do not return.

        ``layer`` must be the layer that is SHEDDING — the SAE's own
        ``cfg.layer`` — not an edge's ``down_layer``. A cross-layer edge's
        downstream endpoint may live on a layer this SAE does not own, and
        naming it here would accuse a layer that is not even armed. That is the
        R1-04 defect (status naming an uncontained layer) reached through a
        different path.
        """
        used = self._spent.get(circuit_id, 0)
        if used >= self.cap:
            self.note_truncated(circuit_id, layer)
            return False
        self._spent[circuit_id] = used + 1
        return True

    def note_truncated(self, circuit_id: str, layer: int) -> None:
        """Record that ``layer`` lost data for ``circuit_id``.

        Separate from ``try_spend`` because shedding truncates WITHOUT
        spending: a saturated pass drops events before the budget is consulted,
        so the budget would otherwise have no idea it happened. Before this the
        two truncation sources disagreed on exactly that case.
        """
        self._truncated.setdefault(circuit_id, set()).add(layer)

    def truncated_layers(self, circuit_id: str) -> list[int]:
        return sorted(self._truncated.get(circuit_id, ()))

    def spent(self, circuit_id: str) -> int:
        return self._spent.get(circuit_id, 0)


# ─────────────────────────────────────────────────────────────────────────
# The request context
# ─────────────────────────────────────────────────────────────────────────


class EdgeSensingRequestContext:
    """Owns absolute position, the per-circuit rings, and the event budget.

    NAMED ``EdgeSensingRequestContext``, not ``SensingRequestContext``:
    Feature 11 already owns that name at ``inference_service.py:110`` for its
    single-SAE cluster snapshot. Two same-named classes one import away, in a
    codebase where a mis-keyed layer lookup has already indexed the wrong SAE's
    feature space, is a realistic path to a real defect (task 2.2).

    Built for N circuits from the outset. Feature 19 lifts the single-active
    invariant, and designing this for one circuit and generalising later would
    repeat precisely the mistake this feature exists to correct.
    """

    def __init__(self, request_id: str, circuit_ids: frozenset[str], cap: int):
        self.request_id = request_id
        self.circuit_ids = circuit_ids
        # NOT a shared position counter — see the note where `advance()` used
        # to live. Position and phase are per-layer, on the SAE. These remain
        # only as the request's own bookkeeping for `close()`/logging.
        self.budget = EventBudget(cap=cap)
        self._rings: dict[str, EdgeFireRing] = {}
        self._closed = False

    def ring(self, circuit_id: str, max_lag: int) -> EdgeFireRing:
        """The ring for one circuit, created on first use."""
        r = self._rings.get(circuit_id)
        if r is None:
            r = EdgeFireRing(max_lag)
            self._rings[circuit_id] = r
        return r

    # `advance()` was DELETED (F17 task 5.2 mutation finding).
    #
    # The design called for ONE shared position counter per request, and it is
    # wrong. Every layer's hook sees the SAME tokens, so with a shared counter
    # the upstream layer advances to 12 and the downstream layer then senses
    # those same 12 tokens starting at 12 — the two layers' coordinates
    # diverge and no cross-layer edge can ever match. Verified by execution:
    # the characterization gate failed the moment it was wired, with
    # `ring._fires` holding upstream positions 2..11 against a downstream
    # layer that had begun at 12.
    #
    # Absolute position is shared BY CONSTRUCTION instead — every layer counts
    # the same tokens from 0, so their coordinates agree without any shared
    # counter, and F15 R1-03 (an early return skipping one SAE's advance) is
    # prevented by the single unconditional advance in
    # `LoadedSAE._advance_edge_position` rather than by centralising the count.
    #
    # A dead `advance()` was left here briefly with tests asserting that layer
    # 13 continues from layer 10's position. Those tests pinned the defect
    # rather than preventing it, which is the exact anti-pattern BR-005
    # forbids, so both are gone. What genuinely IS per-request and shared —
    # the rings, the budget, the pruning boundary — still lives here.

    def report_progress(
        self, layer: int, through: int,
        circuit_id: Optional[str] = None, max_lag: int = 1,
    ) -> None:
        """Tell every ring how far ``layer`` has walked, so pruning tracks the
        SLOWEST layer.

        Deliberately NOT part of ``advance``. Position must move before the
        guards (so no pass is skipped); progress must be reported after the
        match (so a layer's own advance cannot prune the ring out from under a
        sibling that has not read it yet). Folding the two together
        resurrects F15 R1-01 and takes cross-layer sensing dark — the
        characterization gate caught exactly that, twice: once on LoadedSAE
        and again here when ``advance`` was first wired in.

        ``through`` is the reporting layer's OWN absolute offset. It is passed
        in rather than read from ``self.position`` because position is
        per-layer: every layer's hook walks the same tokens, so a single shared
        counter would make the layers' coordinates diverge (see
        ``_advance_edge_position``). Reading ``self.position`` here reported 0
        forever once that was understood — caught by the gate, not by reading.

        Must be called on EVERY pass, including suppressed and batched ones,
        or ``_progress`` stays under the ring's len<2 guard and pruning never
        runs at all (EC-17.1).
        """
        if self._closed:
            return
        if circuit_id is not None:
            # Report to THIS layer's circuit ring, CREATING it if the request
            # has not matched anything yet. Rings are lazy, and a suppressed or
            # quiet pass never matches — so reporting only to already-existing
            # rings silently dropped the report on exactly the passes EC-17.1
            # is about. Caught by the gate: `_progress` stayed {} while the
            # layer had walked 3 tokens.
            try:
                self.ring(circuit_id, max_lag).note_layer_progress(layer, through)
            except Exception:
                logger.exception("edge_ring_progress_failed")
            return
        for ring in self._rings.values():
            try:
                ring.note_layer_progress(layer, through)
            except Exception:
                logger.exception("edge_ring_progress_failed")

    def close(self) -> None:
        """Release the boundary. Idempotent."""
        self._closed = True
        for ring in self._rings.values():
            ring.clear()
        self._rings.clear()

    @property
    def is_closed(self) -> bool:
        return self._closed


# The context-entry wrapper `match_edges(ctx, circuit_id, ...)` was DELETED.
#
# It existed to drive matching from the request context, reading `ctx.phase`
# and spending a per-circuit budget. With position and phase established as
# per-LAYER (see where `advance()` used to live), it had no caller in
# production or in tests — a second entry point to the same loop, kept alive
# only by having been written. `_match_edges_impl` below is the one matcher;
# `LoadedSAE._match_edges` is its one caller.


def _match_edges_impl(
    *,
    ring: EdgeFireRing,
    config: Any,
    phase: str,
    base: int,
    acts_cpu: Any,
    fired_cpu: Any,
    out: list[SensedEdge],
    n_cols: int,
    shed: bool,
    capped: bool,
    positions_per_col_when_shed: int = _EDGE_SHED_POSITIONS_PER_COL,
    try_spend: Optional[Any] = None,
    on_cap: Optional[Any] = None,
    on_truncated: Optional[Any] = None,
) -> None:
    """The single matching loop. ONE copy, two entry points.

    Callers differ only in where the cap lives, so that is the only thing
    parameterised:

    * ``try_spend(spec) -> bool`` — the context path, where the budget is
      per CIRCUIT and shared across its layers.
    * ``on_cap()`` — the LoadedSAE path, where the cap is per SAE and reaching
      it latches so later passes skip the downstream half.
    * ``on_truncated()`` — records data loss WITHOUT latching, for the shared
      circuit budget. A layer refused by a sibling's spending must stay
      eligible; latching it would starve it for the rest of the request
      (R2-03).

    Everything load-bearing — ordering, the shed cap, the strictly-before
    match, the fourteen event fields — exists exactly once. This function was
    briefly duplicated across two modules with the copies already diverging in
    cap semantics and truncation shape; the divergence is the reason it is not
    duplicated now.

    Both cap paths CONTINUE rather than return: the remaining upstream events
    in the pass still have to reach the ring, or a capped layer silently blinds
    its siblings (R2-03, R3-02).
    """
    fired_positions: list[list[int]] = [[] for _ in range(n_cols)]
    for col in range(n_cols):
        nz = fired_cpu[:, col].nonzero()
        if nz.numel():
            fired_positions[col] = nz.flatten().tolist()

    if shed:
        # Bound the UPSTREAM half too. Shedding originally bounded only the
        # downstream matching, but the upstream half is per-edge, so at the
        # contract's 200-edge maximum a shed pass still cost 544ms (F17 gate).
        # Keep the newest: match_down reports the nearest antecedent.
        for col in range(n_cols):
            if len(fired_positions[col]) > positions_per_col_when_shed:
                fired_positions[col] = fired_positions[col][
                    -positions_per_col_when_shed:
                ]

    events: list[tuple[int, int, bool]] = []
    for spec_i, spec in enumerate(config.edges):
        if 0 <= spec.up_col < n_cols:
            for local in fired_positions[spec.up_col]:
                events.append((local, spec_i, True))
        # When shedding OR capped, record upstream halves only — siblings
        # depend on them (R2-03, R3-02) — and skip the downstream matching.
        if not shed and not capped and 0 <= spec.down_col < n_cols:
            for local in fired_positions[spec.down_col]:
                events.append((local, spec_i, False))
    if not events:
        return
    # Upstream before downstream at equal positions, so an upstream fire at p
    # is visible to a downstream fire at p+1 within the same pass.
    #
    # R3-16: the `not e[2]` TIEBREAK itself is behaviourally INERT — verified by
    # running both orderings against a same-position co-fire, an intra-pass
    # up@0/down@1, and a co-fire followed by a later downstream fire: identical
    # results in all three. `match_down` requires STRICTLY before, so which of
    # two equal-position events is processed first cannot change any match.
    #
    # Kept because it makes the intent readable and costs nothing, and because
    # a future `match_down` that relaxed strictly-before would need it. NOT
    # given a test: an assertion on an inert line passes for the wrong reason.
    # The position ordering it rides on IS load-bearing and IS pinned, by the
    # characterization gate's intra-pass tests.
    events.sort(key=lambda e: (e[0], not e[2]))

    for local, spec_i, is_up in events:
        abs_pos = base + local
        spec = config.edges[spec_i]
        row_acts = acts_cpu[local]
        if is_up:
            ring.record_up(spec.edge_key, abs_pos, float(row_acts[spec.up_col]))
            continue
        match = ring.match_down(spec.edge_key, abs_pos)
        if match is None:
            continue
        up_pos, up_act = match
        # ORDER MATTERS (R2-03). This layer's OWN cap is checked first:
        # reaching it means the layer is genuinely done, so the latch is a
        # correct optimisation. The shared circuit budget is checked
        # second and never latches, because a layer refused by a SIBLING's
        # spending must stay eligible. Checking the budget first made the
        # per-SAE latch unreachable whenever the two caps were equal — the
        # common single-layer case.
        if on_cap is not None and len(out) >= config.max_events_per_request:
            # Per-SAE cap reached. Latch, then CONTINUE for the same reason.
            on_cap()
            continue
        if try_spend is not None and not try_spend(spec):
            # Budget exhausted for this circuit. CONTINUE — returning here
            # would stop this layer feeding the ring and blind its siblings.
            #
            # The refusal MUST be reported as truncation. Wiring the per-circuit
            # budget initially dropped events without setting `_edge_truncated`,
            # because `try_spend` refuses BEFORE the per-SAE latch below is
            # reached — so the drain reported a clean, complete result while
            # events were being discarded. That is the silent-dark failure this
            # feature exists to remove, reintroduced by the fix for it.
            # R2-03: report truncation WITHOUT latching. `on_cap` also sets
            # `_edge_done`, which makes every later pass skip the downstream
            # half — correct for the PER-SAE cap (this layer really is done)
            # and wrong for the shared circuit budget, which is a global
            # condition. Measured with a circuit cap of 4 across two layers:
            #
            #   pass 1 (only L10 busy): L10 4 events | L11 0 events, done=False
            #   pass 2 (L11 now fires): L11 0 events, done=TRUE   <- latched
            #   pass 3:                 L11 0 events              <- dark
            #
            # Layer 11 recorded NOTHING for the whole request because a SIBLING
            # spent the budget. That is the R2-03/R3-02 starvation this code
            # has already fixed twice, reintroduced through the budget path by
            # the R1-05 fix. The budget can legitimately free up — another
            # circuit's layers do not share it — so a refused layer must stay
            # eligible.
            if on_truncated is not None:
                on_truncated()
            continue
        out.append(
            SensedEdge(
                edge_key=spec.edge_key,
                up_layer=spec.up_layer,
                up_feature_idx=spec.up_feature_idx,
                up_pos=up_pos,
                up_act=up_act,
                down_layer=spec.down_layer,
                down_feature_idx=spec.down_feature_idx,
                down_pos=abs_pos,
                down_act=float(row_acts[spec.down_col]),
                token_lag=abs_pos - up_pos,
                phase=phase,
                rung=spec.rung,
                rung_language=spec.rung_language,
                edge_type=spec.edge_type,
            )
        )
