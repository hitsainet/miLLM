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


class TestPositionOwnership:
    def test_advance_returns_the_base_and_moves_the_position(self):
        c = ctx()
        assert c.advance(10, 5) == 0
        assert c.position == 5
        assert c.advance(10, 3) == 5
        assert c.position == 8

    def test_phase_flips_to_decode_exactly_once(self):
        c = ctx()
        assert c.phase == "prefill"
        c.advance(10, 4)
        assert c.phase == "decode"
        c.advance(10, 1)
        assert c.phase == "decode"

    def test_every_participating_layer_shares_ONE_counter(self):
        """The defect this replaces: two SAEs advancing independently meant the
        shared ring's absolute-position key silently diverged (F15 R1-03)."""
        c = ctx()
        c.advance(10, 4)
        base_for_layer_13 = c.advance(13, 4)
        assert base_for_layer_13 == 4, (
            "layer 13 started from its own counter instead of the request's"
        )


class TestSuppressedPassesStillReportProgress:
    def test_progress_is_reported_on_every_advance(self):
        """FPRD §15.6 / EC-17.1: F15 R3's own fix put note_layer_progress in a
        `finally` BELOW three early returns, so a suppressed layer never
        reported progress, `_progress` stayed under the len<2 guard, and the
        ring never pruned. `advance` is now called before any guard."""
        c = ctx()
        ring = c.ring("c1", 4)
        ring.record_up("e", 0, 1.0)

        # Two layers walk well past the fire; both advances report progress.
        c.advance(10, 100)
        c.advance(13, 0)   # a suppressed pass still advances (seq_len 0)
        # The slowest layer is now at 100, so pos 0 is far out of window.
        assert ring.match_down("e", 101) is None


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

    def test_advance_after_close_returns_minus_one(self):
        """CTX-L2 / EC-17.5: a hung generate thread waking up later must never
        land in the NEXT request's accounting."""
        c = ctx()
        c.close()
        assert c.advance(10, 5) == -1
        assert c.position == 0, "a post-close write moved the position"

    def test_close_releases_the_rings(self):
        c = ctx()
        ring = c.ring("c1", 8)
        ring.record_up("e", 1, 1.0)
        c.close()
        assert ring.match_down("e", 2) is None
