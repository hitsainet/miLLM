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

    def __init__(self, max_lag: int):
        self._max_lag = max(1, int(max_lag))
        #: edge_key -> list of (abs_pos, activation), ascending by position.
        self._fires: dict[str, list[tuple[int, float]]] = {}
        #: layer -> position walked through, so the ring can prune to the
        #: SLOWEST layer without any hook knowing about its siblings.
        self._progress: dict[int, int] = {}
        self._last_pruned_at: int = 0

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
        if len(self._progress) < 2:
            return  # a single layer: nothing to be slower than
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
        """Claim one observation slot. False ⇒ continue, do not return."""
        used = self._spent.get(circuit_id, 0)
        if used >= self.cap:
            self._truncated.setdefault(circuit_id, set()).add(layer)
            return False
        self._spent[circuit_id] = used + 1
        return True

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
        self.position: int = 0
        self.phase: str = "prefill"
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

    def advance(self, layer: int, seq_len: int) -> int:
        """Advance past ``seq_len`` tokens and report progress. Returns the
        position the pass STARTED at, or -1 once closed.

        Called UNCONDITIONALLY at the top of every pass, before any guard.
        F15's offset advance lived in a ``finally`` below three early returns,
        so a suppressed, unarmed or batched pass left one SAE behind its
        siblings — and F15 R3's own fix inherited the same shape, because
        ``note_layer_progress`` sat below the returns too, meaning a suppressed
        layer never reported progress and the ring never pruned (FPRD §15.6 /
        EC-17.1). Here there is no path that reaches sensing without advancing.
        """
        if self._closed:
            # A late write from a hung generate thread must never land in the
            # next request's accounting (CTX-L2, EC-17.5).
            logger.warning(
                "edge_sensing_write_after_close: request=%s layer=%s",
                self.request_id, layer,
            )
            return -1
        base = self.position
        self.position += int(seq_len)
        if self.phase == "prefill":
            self.phase = "decode"
        for ring in self._rings.values():
            ring.note_layer_progress(layer, self.position)
        return base

    def close(self) -> None:
        """Release the boundary. Idempotent."""
        self._closed = True
        for ring in self._rings.values():
            ring.clear()
        self._rings.clear()

    @property
    def is_closed(self) -> bool:
        return self._closed


def match_edges(
    ctx: EdgeSensingRequestContext,
    circuit_id: str,
    config: Any,
    base: int,
    seq_len: int,
    acts_cpu: Any,
    fired_cpu: Any,
    out: list[SensedEdge],
    *,
    shed: bool = False,
    capped: bool = False,
    positions_per_col_when_shed: int = 64,
) -> None:
    """Record upstream fires and match downstream ones, in position order.

    Ordering is load-bearing: within one prefill pass an upstream fire at p must
    be visible to a downstream fire at p+1, so events are sorted with upstream
    first at equal positions — which also keeps a same-position co-fire from
    matching, since ``match_down`` requires strictly before.
    """
    ring = ctx.ring(circuit_id, config.max_token_lag)
    phase = ctx.phase
    n_cols = fired_cpu.shape[-1] if fired_cpu.dim() > 1 else 0

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
        if not ctx.budget.try_spend(circuit_id, spec.down_layer):
            # Budget exhausted for this circuit. CONTINUE — returning here
            # would stop this layer feeding the ring and blind its siblings.
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
