"""Feature 17 tasks 2.6 / 2.7 — the request-scoped context.

Three of Feature 15's eight criticals share one root cause: N per-SAE counters
had to agree about an absolute coordinate no component owned, and the shared
ring's lifetime belonged to whichever hook remembered to call it. These tests
pin the properties that make those states unrepresentable rather than
test-guarded.
"""

import pytest

from millm.ml.edge_sensing import (
    EdgeFireRing,
    EdgeSensingRequestContext,
    EventBudget,
)


def ctx(cap=20, circuits=("c1",)):
    return EdgeSensingRequestContext("req-1", frozenset(circuits), cap=cap)


class TestPositionIsPerLayerNotShared:
    """The context deliberately owns NO position counter.

    The original design gave it one, shared by every layer, and that is wrong:
    every layer's hook sees the SAME tokens, so a shared counter makes the
    upstream layer advance to 12 and the downstream layer then sense those same
    12 tokens starting at 12. Their coordinates diverge and no cross-layer edge
    can match. Found by wiring it and watching the characterization gate fail.

    The tests that used to live here asserted `advance(13, 4) == 4` — layer 13
    continuing from layer 10's count. They pinned the defect instead of
    preventing it (BR-005), so they are gone along with `advance()` itself."""

    def test_the_context_owns_no_position_counter(self):
        c = ctx()
        assert not hasattr(c, "advance"), (
            "a shared position counter is back; it breaks cross-layer matching"
        )
        assert not hasattr(c, "position")

    def test_two_layers_reading_the_same_tokens_agree_on_coordinates(self):
        """The property the deleted counter was trying to provide, which holds
        BY CONSTRUCTION: each layer counts the same tokens from 0, so an
        upstream fire at absolute position 2 is at position 2 for every layer
        without anything being centralised."""
        c = ctx()
        ring = c.ring("c1", 8)
        # Layer 10 records an upstream fire at absolute position 2.
        ring.record_up("e", 2, 1.0)
        # Layer 13 walks the SAME tokens and matches at position 4 — BEFORE
        # either layer reports progress, which is the production order
        # (progress is reported after the match, never before; reporting first
        # is what prunes the ring out from under a sibling).
        assert ring.match_down("e", 4) == (2, 1.0), (
            "the layers disagree about where token 2 is"
        )
        c.report_progress(10, 12, circuit_id="c1", max_lag=8)
        c.report_progress(13, 12, circuit_id="c1", max_lag=8)


class TestSuppressedPassesStillReportProgress:
    def test_progress_is_reported_even_before_any_ring_exists(self):
        """FPRD §15.6 / EC-17.1. Rings are created lazily on first MATCH, and a
        suppressed or quiet pass never matches — so reporting only to
        already-existing rings dropped the report on exactly the passes this
        rule is about. `_progress` stayed {} while the layer had walked 3
        tokens; the ring's len<2 guard then meant it never pruned at all."""
        c = ctx()
        assert not c._rings, "precondition: no ring has been created yet"
        c.report_progress(10, 3, circuit_id="c1", max_lag=4)
        assert c.ring("c1", 4)._progress.get(10) == 3

    def test_progress_prunes_to_the_SLOWEST_layer(self):
        c = ctx()
        ring = c.ring("c1", 4)
        ring.record_up("e", 0, 1.0)
        c.report_progress(10, 100, circuit_id="c1", max_lag=4)
        c.report_progress(13, 100, circuit_id="c1", max_lag=4)
        assert ring.match_down("e", 101) is None

    def test_a_lagging_layer_holds_the_boundary_back(self):
        """The whole point of tracking per-layer progress: a fire the slow
        layer still needs must not be pruned by the fast one."""
        c = ctx()
        ring = c.ring("c1", 4)
        ring.record_up("e", 38, 1.0)
        c.report_progress(10, 5000, circuit_id="c1", max_lag=4)
        c.report_progress(13, 40, circuit_id="c1", max_lag=4)
        assert ring.match_down("e", 41) == (38, 1.0), (
            "pruned past a fire the lagging layer still needed"
        )

    def test_progress_after_close_is_a_no_op(self):
        """CTX-L2 / EC-17.5: a hung generate thread waking up later must never
        land in the next request's accounting. `advance` used to carry this
        guarantee; with it gone, `report_progress` is the write path that
        needs it."""
        c = ctx()
        ring = c.ring("c1", 4)
        c.close()
        c.report_progress(10, 500, circuit_id="c1", max_lag=4)
        assert ring._progress == {}, "a post-close write landed"


class TestRingIsolationPerCircuit:
    def test_two_circuits_with_the_SAME_edge_key_never_cross_match(self):
        """CTX-R2 / US-17.2. `edge_key` is synthesised from layers and feature
        indices, so it is NOT unique across circuits. With one shared ring,
        circuit A's upstream fire could match circuit B's downstream fire and
        record an observation of an edge that fired in NEITHER — a fabricated
        observation on an evidence surface."""
        c = ctx(circuits=("A", "B"))
        shared_key = "1@10->2@13"

        ring_a = c.ring("A", 8)
        ring_b = c.ring("B", 8)
        assert ring_a is not ring_b

        ring_a.record_up(shared_key, 0, 1.0)
        assert ring_b.match_down(shared_key, 1) is None, (
            "circuit B matched circuit A's upstream fire"
        )
        assert ring_a.match_down(shared_key, 1) == (0, 1.0)

    def test_the_same_circuit_gets_the_same_ring(self):
        c = ctx()
        assert c.ring("c1", 8) is c.ring("c1", 8)


class TestBudgetAttribution:
    def test_the_budget_is_per_circuit_not_per_request(self):
        """F15's cap was per-SAE, so an N-layer circuit emitted up to N x cap."""
        b = EventBudget(cap=2)
        assert [b.try_spend("A", 10) for _ in range(3)] == [True, True, False]
        # B is unaffected by A exhausting its budget.
        assert b.try_spend("B", 10) is True

    def test_truncation_names_the_layer_that_shed(self):
        """F15 OR'd `truncated` across layers onto every row, so one saturated
        layer marked a whole request's observations truncated."""
        b = EventBudget(cap=1)
        b.try_spend("A", 10)          # consumes the budget
        b.try_spend("A", 13)          # this layer sheds
        assert b.truncated_layers("A") == [13]
        assert b.truncated_layers("B") == []

    def test_a_refused_spend_does_not_stop_the_caller(self):
        """try_spend returning False means CONTINUE, never return — F15 R3-02
        found the cap returning from the whole pass, which stopped a capped
        layer feeding the shared ring and blinded its uncapped siblings."""
        b = EventBudget(cap=0)
        assert b.try_spend("A", 10) is False
        # The contract is that the caller keeps going; the budget itself must
        # stay usable rather than latching into an error state.
        assert b.try_spend("A", 11) is False
        assert sorted(b.truncated_layers("A")) == [10, 11]


class TestCloseAndWriteAfterClose:
    def test_close_is_idempotent(self):
        c = ctx()
        c.close()
        c.close()
        assert c.is_closed is True

    def test_a_closed_context_creates_no_new_rings(self):
        """CTX-L2 / EC-17.5. `close()` drops the rings; a late write must not
        resurrect one, or the next request inherits a live object from the
        previous one."""
        c = ctx()
        c.close()
        c.report_progress(10, 5, circuit_id="c1", max_lag=4)
        assert c._rings == {}, "a post-close write rebuilt a ring"

    def test_close_releases_the_rings(self):
        c = ctx()
        ring = c.ring("c1", 8)
        ring.record_up("e", 1, 1.0)
        c.close()
        assert ring.match_down("e", 2) is None


class TestStrictlyBeforeIsEnforcedByEachGuardIndEPENDENTLY:
    """F17 task 5.2. Mutating `bisect_left`→`bisect_right`, or the `-inf`
    sentinel, or `pos < down_pos` each left the whole suite GREEN — three
    mutations, three survivors on the invariant that matters most.

    None is a test gap in the ordinary sense. Mapping every combination shows
    the real structure — only the SENTINEL and the COMPARISON are load-bearing,
    and each alone is sufficient:

        sentinel + bisect broken   -> None          (comparison holds)
        bisect   + comparison      -> None          (sentinel holds)
        sentinel + comparison      -> (5, 1.0)      *** invariant broken ***
        all three                  -> (5, 1.0)

    `bisect_left` vs `bisect_right` is behaviourally INERT: the `-inf` sentinel
    makes both land at the same index. It is a performance choice (F15 R3's
    O(n)→O(log n) fix), not a correctness guard, and no behavioural test can
    pin it — the benchmark in test_edge_sensing_baseline.py is what protects it.

    So the two guards are mutually redundant by design, and no single-line
    mutation of them is observable. Rather than leave that as an unexplained
    pair of survivors, the pair mutation is tested directly below, and the
    boundary behaviour is pinned so the invariant is checked from the outside
    regardless of which mechanism enforces it."""

    def test_a_fire_at_exactly_down_pos_never_matches(self):
        """Same-position firing is co-activation, not an up→down sequence.
        Reporting it as one overclaims direction on an evidence surface."""
        c = ctx()
        ring = c.ring("c1", 8)
        ring.record_up("e", 5, 1.0)
        assert ring.match_down("e", 5) is None

    def test_a_fire_one_position_before_DOES_match(self):
        """The other side of the boundary: strictly-before must not be
        strengthened into something that drops legitimate adjacent fires."""
        c = ctx()
        ring = c.ring("c1", 8)
        ring.record_up("e", 5, 1.0)
        assert ring.match_down("e", 6) == (5, 1.0)

    def test_the_newest_STRICTLY_EARLIER_fire_wins_over_a_same_position_one(self):
        """The case that needs all three guards agreeing: a same-position fire
        sits at the bisect boundary next to a legitimate earlier one. The
        earlier fire must be returned, not the co-fire and not None."""
        c = ctx()
        ring = c.ring("c1", 8)
        ring.record_up("e", 3, 1.0)
        ring.record_up("e", 7, 2.0)
        assert ring.match_down("e", 7) == (3, 1.0), (
            "matched the same-position fire, or missed the real antecedent"
        )

    def test_several_fires_at_the_same_position_are_all_rejected(self):
        """Ties at the boundary are where bisect_left and bisect_right differ
        most. Every one of them is a co-fire and none may match."""
        c = ctx()
        ring = c.ring("c1", 8)
        for act in (1.0, 2.0, 3.0):
            ring.record_up("e", 4, act)
        assert ring.match_down("e", 4) is None
        # ...but an earlier fire behind the tie is still reachable.
        ring.record_up("e", 2, 9.0)
        ring._fires["e"].sort()
        assert ring.match_down("e", 4) == (2, 9.0)

    def test_the_invariant_survives_losing_EITHER_guard_alone(self):
        """The negative control for the two survivors, run in-process.

        Rebuilds `match_down`'s logic with each guard individually disabled and
        asserts the same-position fire is still rejected — which is what makes
        the surviving mutations safe rather than merely unobserved. With BOTH
        disabled the co-fire matches, so the redundancy is real and neither
        guard is decorative."""
        import bisect as _bisect

        fires = [(5, 1.0)]
        down_pos = 5
        max_lag = 8

        def match(sentinel_ok: bool, compare_ok: bool):
            key = (down_pos, float("-inf") if sentinel_ok else float("inf"))
            i = _bisect.bisect_left(fires, key) - 1
            while i >= 0:
                pos, act = fires[i]
                if (down_pos - pos) > max_lag:
                    break
                if (pos < down_pos) if compare_ok else (pos <= down_pos):
                    return (pos, act)
                i -= 1
            return None

        assert match(True, True) is None, "the shipped configuration"
        assert match(True, False) is None, "sentinel alone must hold"
        assert match(False, True) is None, "comparison alone must hold"
        assert match(False, False) == (5, 1.0), (
            "with both guards disabled a same-position co-fire must match — "
            "if it does not, this control proves nothing"
        )


class TestR2ASingleLayerCircuitPrunesToo:
    """F17 R2-10. `note_layer_progress` returned early when fewer than 2 layers
    had reported — "a single layer: nothing to be slower than". But with ONE
    layer, that layer IS the slowest, so pruning is safe and correct. Measured
    on a 600-fire edge: one reporting layer retained 512 (the per-edge hard
    cap), two retained 4. That is 128x the intended memory, and a single-layer
    circuit is a legitimate configuration.

    Naively dropping the guard resurrects R1-01: with two layers the FIRST to
    report prunes past fires the second still needs — six tests caught that
    immediately. The ring cannot infer how many layers are coming, so it is
    TOLD, and an untold ring keeps the old conservative rule."""

    def test_a_ring_told_it_has_one_layer_prunes_on_that_layer(self):
        from millm.ml.edge_sensing import EdgeFireRing

        ring = EdgeFireRing(4)
        ring.expect_layers(1)
        for pos in range(600):
            ring.record_up("e", pos, 1.0)
        ring.note_layer_progress(10, 600)
        assert len(ring._fires["e"]) < 50, (
            f"{len(ring._fires['e'])} fires retained — a single-layer circuit "
            "never pruned and fell back to the 512 hard cap"
        )

    def test_a_ring_told_it_has_TWO_layers_waits_for_both(self):
        """The R1-01 guarantee: the fast layer must not prune ahead of the
        slow one."""
        from millm.ml.edge_sensing import EdgeFireRing

        ring = EdgeFireRing(4)
        ring.expect_layers(2)
        ring.record_up("e", 38, 1.0)
        ring.note_layer_progress(10, 5000)          # fast layer races ahead
        assert ring.match_down("e", 41) == (38, 1.0), (
            "pruned on one report while a second layer was still expected"
        )

    def test_an_UNTOLD_ring_stays_conservative(self):
        """The default must not assume 1: a ring nobody told is a ring that may
        still have siblings coming, and guessing wrong destroys their data."""
        from millm.ml.edge_sensing import EdgeFireRing

        ring = EdgeFireRing(4)
        ring.record_up("e", 38, 1.0)
        ring.note_layer_progress(10, 5000)
        assert ring.match_down("e", 41) == (38, 1.0)

    def test_the_service_tells_the_ring_its_layer_count(self):
        """Declaring `expect_layers` without wiring it would be the exact
        anti-pattern this arc keeps producing."""
        from tests.unit.services.test_circuit_sensing_service import (
            circuit, definition, two_saes,
        )
        from millm.services.circuit_sensing_service import CircuitSensingService

        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("r", saes)
        ring = svc._ctx.ring("circ_1", svc._max_token_lag)
        assert ring._expected_layers == len(svc._armed_layers) == 2
