"""Feature 17 task 1.0 — CHARACTERIZATION GATE.

These tests pin the edge matcher's CURRENT behaviour, written and green BEFORE
any code moves. They are the parity contract for the extraction: after F17
moves this machinery into `millm/ml/edge_sensing.py`, every one of them must
still pass unchanged.

Why this gate exists (BRD-002 locked decision 3): this is the most defect-dense
code in the arc — eight criticals across three review rounds of Feature 15, plus
four more in Feature 16. Refactoring it without a behavioural net would be
moving code whose behaviour nobody has written down.

**From here on, editing a test in this file is a BEHAVIOUR CHANGE and requires
justification in the review record (CTX-V2).** A failure here after the move is
a regression, not a stale fixture.
"""

import pytest
import torch

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import (
    CircuitSensingConfig,
    EdgeFireRing,
    EdgeSpec,
    LoadedSAE,
)

D_IN = 8
D_SAE = 32


def real_sae() -> LoadedSAE:
    """Deterministic SAE: encoder column j responds 1:1 to input dim j % d_in."""
    W_enc = torch.zeros(D_IN, D_SAE)
    for j in range(D_SAE):
        W_enc[j % D_IN, j] = 1.0
    return LoadedSAE(
        W_enc=W_enc,
        b_enc=torch.zeros(D_SAE),
        W_dec=torch.zeros(D_SAE, D_IN),
        b_dec=torch.zeros(D_IN),
        config=SAEConfig(d_in=D_IN, d_sae=D_SAE, model_name="t",
                         hook_name="t", hook_layer=1),
        device="cpu",
    )


def hidden(*rows) -> torch.Tensor:
    out = torch.zeros(len(rows), D_IN)
    for i, row in enumerate(rows):
        for dim, val in row.items():
            out[i, dim] = val
    return out


def spec(up_col=0, down_col=1, up_layer=10, down_layer=10, rung=2,
         key="1@10->2@10"):
    return EdgeSpec(
        edge_key=key, up_layer=up_layer, up_feature_idx=1, up_col=up_col,
        down_layer=down_layer, down_feature_idx=2, down_col=down_col,
        rung=rung, rung_language="causally validated (edge)",
        edge_type="computed",
    )


def config(edges=None, max_lag=4, cap=20, layer=10):
    return CircuitSensingConfig(
        circuit_id="circ_1", layer=layer, member_indices=[1, 2],
        thresholds=torch.tensor([0.5, 0.5]), threshold_mode="epsilon_max",
        edges=edges if edges is not None else [spec()],
        max_token_lag=max_lag, context_tokens=8, max_events_per_request=cap,
    )


def armed(cfg=None, ring=None):
    sae = real_sae()
    cfg = cfg or config()
    sae.arm_edge_sensing(cfg, ring or EdgeFireRing(cfg.max_token_lag))
    sae.begin_edge_sensing_request("char")
    return sae


# ─────────────────────────────────────────────────────────────────────────
# 1.1 — the matcher's core contract
# ─────────────────────────────────────────────────────────────────────────

class TestOrderingContract:
    def test_strict_up_then_down_matches(self):
        sae = armed()
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert len(sae._sensed_edges) == 1
        ev = sae._sensed_edges[0]
        assert (ev.up_pos, ev.down_pos, ev.token_lag) == (0, 1, 1)

    def test_a_same_position_co_fire_does_NOT_match(self):
        """Simultaneous firing is co-activation, not a sequence. Reporting it
        as up→down would assert an ordering that was never observed."""
        sae = armed()
        sae._sense_edges(hidden({1: 2.0, 2: 2.0}))
        assert sae._sensed_edges == []

    def test_reversed_order_does_NOT_match(self):
        sae = armed()
        sae._sense_edges(hidden({2: 2.0}, {1: 2.0}))
        assert sae._sensed_edges == []

    def test_a_lone_upstream_fire_does_NOT_match(self):
        sae = armed()
        sae._sense_edges(hidden({1: 2.0}, {1: 2.0}))
        assert sae._sensed_edges == []


class TestLagBoundary:
    def test_exactly_at_the_window_matches(self):
        cfg = config(max_lag=3)
        sae = armed(cfg, EdgeFireRing(3))
        sae._sense_edges(hidden({1: 2.0}, {}, {}, {2: 2.0}))
        assert len(sae._sensed_edges) == 1
        assert sae._sensed_edges[0].token_lag == 3

    def test_one_past_the_window_does_NOT_match(self):
        cfg = config(max_lag=3)
        sae = armed(cfg, EdgeFireRing(3))
        sae._sense_edges(hidden({1: 2.0}, {}, {}, {}, {2: 2.0}))
        assert sae._sensed_edges == []


class TestAntecedentSelection:
    def test_the_newest_antecedent_wins(self):
        """The closest antecedent is the most defensible attribution."""
        sae = armed(config(max_lag=8), EdgeFireRing(8))
        sae._sense_edges(hidden({1: 2.0}, {1: 2.0}, {2: 2.0}))
        assert len(sae._sensed_edges) == 1
        assert sae._sensed_edges[0].up_pos == 1

    def test_the_read_is_NON_DESTRUCTIVE(self):
        """One upstream fire can father several downstream events. The FTDD
        originally specified a 'pop'; the implementation reads without
        removing, and that is the better evidence model — two downstream
        partners responding to one upstream fire is two real observations."""
        sae = armed(config(max_lag=8), EdgeFireRing(8))
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}, {2: 2.0}))
        assert len(sae._sensed_edges) == 2
        assert {e.up_pos for e in sae._sensed_edges} == {0}


class TestRingRetention:
    def test_the_ring_evicts_the_OLDEST_when_full(self):
        ring = EdgeFireRing(4)
        for pos in range(EdgeFireRing._MAX_FIRES_PER_EDGE + 50):
            ring.record_up("e", pos, 1.0)
        assert len(ring._fires["e"]) <= EdgeFireRing._MAX_FIRES_PER_EDGE
        newest = EdgeFireRing._MAX_FIRES_PER_EDGE + 49
        assert ring.match_down("e", newest + 1) == (newest, 1.0)

    def test_match_is_strictly_before(self):
        ring = EdgeFireRing(4)
        ring.record_up("e", 5, 1.0)
        assert ring.match_down("e", 5) is None
        assert ring.match_down("e", 6) == (5, 1.0)


class TestPhaseAccounting:
    def test_prefill_flips_to_decode_exactly_once(self):
        sae = armed(config(max_lag=16), EdgeFireRing(16))
        assert sae._edge_phase == "prefill"
        sae._sense_edges(hidden({}, {}, {}))
        assert sae._edge_phase == "decode"
        sae._sense_edges(hidden({}))
        assert sae._edge_phase == "decode"

    def test_positions_are_absolute_across_passes(self):
        sae = armed(config(max_lag=16), EdgeFireRing(16))
        sae._sense_edges(hidden({}, {1: 2.0}))
        sae._sense_edges(hidden({2: 2.0}))
        assert len(sae._sensed_edges) == 1
        ev = sae._sensed_edges[0]
        assert (ev.up_pos, ev.down_pos) == (1, 2)


# ─────────────────────────────────────────────────────────────────────────
# 1.2 — the twelve fixed criticals, characterized BY BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────

class TestFixedCriticalsStayFixed:
    """Each of these pins a defect that a review round found and fixed. The
    extraction must not resurrect any of them."""

    def test_F15R1_01_cross_layer_survives_a_noisy_upstream(self):
        """R1-01: pruning inside the per-position loop let the upstream layer
        wipe the ring before the downstream hook ran — cross-layer sensing
        went dark on any real traffic."""
        s = spec(up_layer=10, down_layer=13, key="1@10->2@13")
        ring = EdgeFireRing(3)

        up = real_sae()
        up.arm_edge_sensing(
            config(edges=[EdgeSpec(**{**s.__dict__, "up_col": 0, "down_col": -1})],
                   max_lag=3, layer=10), ring)
        up.begin_edge_sensing_request("c")
        rows = [{}] * 12
        rows[2] = {1: 2.0}
        for p in range(6, 12):
            rows[p] = {1: 2.0}
        up._sense_edges(hidden(*rows))

        down = real_sae()
        down.arm_edge_sensing(
            config(edges=[EdgeSpec(**{**s.__dict__, "up_col": -1, "down_col": 1})],
                   max_lag=3, layer=13), ring)
        down.begin_edge_sensing_request("c")
        drows = [{}] * 12
        drows[4] = {2: 2.0}
        down._sense_edges(hidden(*drows))

        assert len(down._sensed_edges) == 1
        assert down._sensed_edges[0].up_pos == 2

    def test_F15R1_03_offset_advances_on_EVERY_return_path(self):
        """R1-03: early returns skipped the offset advance, so one SAE fell
        behind its siblings and the SHARED ring's absolute-position key
        silently diverged."""
        # suppressed
        sae = armed()
        with sae.suppressed():
            sae._sense_edges(hidden({}, {}, {}))
        assert sae._edge_token_offset == 3

        # batched
        sae2 = armed()
        sae2._sense_edges(hidden({}, {}).unsqueeze(0).repeat(3, 1, 1))
        assert sae2._edge_token_offset == 2

        # raising
        sae3 = armed()
        sae3._match_edges = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        sae3._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae3._edge_token_offset == 2

    def test_F15R1_04_out_of_range_column_is_a_clean_arm_time_error(self):
        """R1-04: an IndexError inside the matcher was swallowed by the broad
        except, abandoning the ENTIRE pass rather than one bad spec."""
        sae = real_sae()
        bad = EdgeSpec(**{**spec().__dict__, "down_col": 99})
        with pytest.raises(ValueError, match="column out of range"):
            sae.arm_edge_sensing(config(edges=[bad]), EdgeFireRing(4))

    def test_F15R2_07_minus_one_is_the_only_valid_sentinel(self):
        """R2-07: validation checked only the upper bound, so -2 passed and the
        matcher silently skipped that half — armed, sensable, never firing."""
        sae = real_sae()
        with pytest.raises(ValueError, match="column out of range"):
            sae.arm_edge_sensing(
                config(edges=[EdgeSpec(**{**spec().__dict__, "down_col": -2})]),
                EdgeFireRing(4))
        # -1 IS legitimate: "not my half"
        ok = real_sae()
        ok.arm_edge_sensing(
            config(edges=[EdgeSpec(**{**spec().__dict__, "up_col": -1})]),
            EdgeFireRing(4))
        assert ok.is_edge_sensing_armed is True

    def test_F15R2_03_a_shedding_layer_still_feeds_its_siblings(self):
        """R2-03: load shedding returned before recording upstream fires, so a
        saturated upstream layer starved a quiet downstream sibling — and the
        truncated flag landed on the wrong layer."""
        s = spec(up_layer=10, down_layer=13, key="1@10->2@13")
        ring = EdgeFireRing(64)

        up = real_sae()
        up_cfg = config(
            edges=[EdgeSpec(**{**s.__dict__, "up_col": 0, "down_col": -1})],
            max_lag=64, layer=10)
        up_cfg.thresholds = torch.tensor([0.0001, 0.0001])
        up.arm_edge_sensing(up_cfg, ring)
        up.begin_edge_sensing_request("c")
        up._sense_edges(torch.rand(4096, D_IN) + 1.0)
        assert up._edge_truncated is True

        down = real_sae()
        down.arm_edge_sensing(
            config(edges=[EdgeSpec(**{**s.__dict__, "up_col": -1, "down_col": 1})],
                   max_lag=64, layer=13), ring)
        down.begin_edge_sensing_request("c")
        down._sense_edges(hidden(*([{}] * 4095 + [{2: 5.0}])))
        assert down._sensed_edges, "the shedding layer starved its sibling"

    def test_F15R3_02_a_capped_layer_still_feeds_its_siblings(self):
        """R3-02: the cap returned from the whole pass, so a capped layer
        stopped feeding the shared ring — R2-03's bug via the cap path."""
        ring = EdgeFireRing(64)
        sae = real_sae()
        sae.arm_edge_sensing(
            config(edges=[EdgeSpec(**{**spec().__dict__, "up_col": 0, "down_col": 1})],
                   max_lag=64, cap=1, layer=10), ring)
        sae.begin_edge_sensing_request("c")
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}, {1: 2.0}, {2: 2.0}))
        assert sae._edge_done is True

        before = len(ring._fires.get("1@10->2@10", []))
        sae._sense_edges(hidden({1: 2.0}))
        assert len(ring._fires.get("1@10->2@10", [])) > before, (
            "a capped layer stopped feeding the shared ring"
        )

    def test_F15R3_01_pruning_is_wired_and_respects_the_slowest_layer(self):
        """R3-01: pruning was declared 'request-level' in two consecutive
        rounds and wired in neither. The ring now tracks layer progress
        itself and prunes to the SLOWEST."""
        ring = EdgeFireRing(8)
        ring.record_up("e", 40, 1.0)
        ring.note_layer_progress(10, 5000)
        ring.note_layer_progress(13, 42)
        assert ring.match_down("e", 44) == (40, 1.0), (
            "pruned past a fire the lagging layer still needed"
        )

    def test_a_single_layer_alone_never_prunes(self):
        ring = EdgeFireRing(4)
        ring.record_up("e", 0, 1.0)
        ring.note_layer_progress(10, 1000)
        assert ring.match_down("e", 1) == (0, 1.0)

    def test_saturation_sheds_visibly_rather_than_silently(self):
        sae = real_sae()
        cfg = config()
        cfg.thresholds = torch.tensor([0.0001, 0.0001])
        sae.arm_edge_sensing(cfg, EdgeFireRing(4))
        sae.begin_edge_sensing_request("c")
        sae._sense_edges(torch.rand(4096, D_IN) + 1.0)
        assert sae._edge_truncated is True
        assert sae._edge_saturation_warned is True

    def test_a_pass_without_begin_senses_nothing(self):
        """Omitting the began-guard would buffer stale cross-request edges."""
        sae = real_sae()
        sae.arm_edge_sensing(config(), EdgeFireRing(4))
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._sensed_edges == []

    def test_collect_without_begin_returns_nothing(self):
        sae = real_sae()
        sae.arm_edge_sensing(config(), EdgeFireRing(4))
        assert sae.collect_sensed_edges() == ("", [], False)

    def test_the_rung_phrase_is_carried_verbatim(self):
        """The evidence guarantee: an observation never upgrades a rung."""
        s = spec(rung=0)
        s.rung_language = "associated"
        sae = armed(config(edges=[s]))
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        ev = sae._sensed_edges[0]
        assert ev.rung == 0 and ev.rung_language == "associated"
        assert "causal" not in ev.rung_language.lower()


# ─────────────────────────────────────────────────────────────────────────
# 3.2 — the advance/report split, pinned directly
#
# Task 3.2 collapsed a triplicated offset advance into one call above every
# guard. The first attempt also hoisted the ring PROGRESS report to the same
# place, which resurrected F15 R1-01: this layer's own advance pruned the ring
# before the downstream layer had read it, and cross-layer sensing went dark.
# test_F15R1_01 caught it, but only end-to-end. These pin the ordering itself,
# so the next refactor fails here — at the cause, not three layers away.
# ─────────────────────────────────────────────────────────────────────────


class TestPositionAdvancesBeforeGuardsProgressReportsAfterWork:

    def test_progress_is_NOT_reported_before_the_match_runs(self):
        """The load-bearing order. If progress were reported at advance time,
        the ring would prune to this layer's new position while an unread
        sibling still needed the older entries."""
        seen: list[str] = []
        sae = armed()
        ring = sae._edge_ring
        real_note = ring.note_layer_progress

        def spy(layer, pos):
            seen.append("progress")
            return real_note(layer, pos)

        ring.note_layer_progress = spy
        real_match = sae._match_edges

        def match_spy(*a, **kw):
            seen.append("match")
            return real_match(*a, **kw)

        sae._match_edges = match_spy
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))

        assert "match" in seen, "the matcher never ran; this test proves nothing"
        assert seen.index("match") < seen.index("progress"), (
            "progress was reported before matching — this is F15 R1-01, the "
            "defect that made cross-layer sensing go dark"
        )

    def test_a_suppressed_pass_still_reports_progress(self):
        """A suppressed layer that never reports leaves `_progress` under the
        ring's len<2 guard, so pruning never runs at all and the ring grows
        unbounded (EC-17.1)."""
        sae = armed()
        with sae.suppressed():
            sae._sense_edges(hidden({}, {}, {}))
        assert sae._edge_ring._progress.get(sae._edge_sensing.layer) == 3

    def test_position_advances_even_when_the_matcher_raises(self):
        """Position must survive a matcher failure, or this SAE silently
        desynchronises from its siblings on the SHARED absolute-position key."""
        sae = armed()

        def boom(*a, **kw):
            raise RuntimeError("matcher exploded")

        sae._match_edges = boom
        try:
            sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        except RuntimeError:
            pass
        assert sae._edge_token_offset == 2
