"""Feature 15 Task 2.5: edge detection core.

The property under test is DIRECTIONAL: an edge fires only when the upstream
member fires and the downstream partner then fires within the lag window.
A lone upstream fire (EC-15.1) and a reversed pair (EC-15.2) must both produce
nothing — those two are what separate an edge observation from plain
co-activation, and getting either wrong would silently overclaim direction.
"""

import pytest
import torch

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import (
    CircuitSensingConfig,
    EdgeFireRing,
    EdgeSpec,
    LoadedSAE,
    SensedEdge,
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
    """(seq, d_in) hidden states from per-position dim values."""
    out = torch.zeros(len(rows), D_IN)
    for i, row in enumerate(rows):
        for dim, val in row.items():
            out[i, dim] = val
    return out


def make_spec(
    up_col=0, down_col=1, up_layer=10, down_layer=10, rung=2, key="1@10->2@10"
):
    return EdgeSpec(
        edge_key=key,
        up_layer=up_layer,
        up_feature_idx=1,
        up_col=up_col,
        down_layer=down_layer,
        down_feature_idx=2,
        down_col=down_col,
        rung=rung,
        rung_language="causally validated (edge)",
        edge_type="computed",
    )


def make_config(edges=None, max_lag=4, cap=20, layer=10):
    return CircuitSensingConfig(
        circuit_id="circ_1",
        layer=layer,
        member_indices=[1, 2],
        thresholds=torch.tensor([0.5, 0.5]),
        threshold_mode="epsilon_max",
        edges=edges if edges is not None else [make_spec()],
        max_token_lag=max_lag,
        context_tokens=8,
        max_events_per_request=cap,
    )


class FakeSAE:
    """Exercises the real _match_edges against a controlled fire matrix."""

    def __init__(self, config, ring):
        from millm.ml.sae_wrapper import LoadedSAE

        self._edge_sensing = config
        self._edge_ring = ring
        self._sensed_edges = []
        self._edge_truncated = False
        self._edge_done = False
        self._edge_phase = "decode"
        self._match_edges = LoadedSAE._match_edges.__get__(self)


def run(fires, config=None, ring=None, base=0):
    """fires: list of per-position [bool, bool] over the 2 armed members."""
    config = config or make_config()
    ring = ring or EdgeFireRing(config.max_token_lag)
    sae = FakeSAE(config, ring)
    fired = torch.tensor(fires, dtype=torch.bool)
    acts = torch.where(fired, torch.tensor(2.0), torch.tensor(0.1))
    sae._match_edges(base, len(fires), acts, fired)
    return sae


class TestDirectionality:
    def test_up_then_down_within_the_window_fires(self):
        sae = run([[True, False], [False, True]])
        assert len(sae._sensed_edges) == 1
        ev = sae._sensed_edges[0]
        assert (ev.up_pos, ev.down_pos, ev.token_lag) == (0, 1, 1)
        assert ev.up_feature_idx == 1 and ev.down_feature_idx == 2

    def test_a_lone_upstream_fire_produces_nothing(self):
        """EC-15.1 — upstream alone is not an edge observation."""
        sae = run([[True, False], [True, False], [True, False]])
        assert sae._sensed_edges == []

    def test_a_lone_downstream_fire_produces_nothing(self):
        sae = run([[False, True], [False, True]])
        assert sae._sensed_edges == []

    def test_reversed_order_produces_nothing(self):
        """EC-15.2 — down THEN up is not the authored direction."""
        sae = run([[False, True], [True, False]])
        assert sae._sensed_edges == []

    def test_a_same_position_cofire_is_not_an_edge(self):
        """Simultaneous firing is co-activation, not a sequence. Reporting it
        as up->down would assert an ordering that was never observed."""
        sae = run([[True, True]])
        assert sae._sensed_edges == []


class TestLagWindow:
    def test_a_fire_beyond_the_window_does_not_match(self):
        cfg = make_config(max_lag=2)
        ring = EdgeFireRing(cfg.max_token_lag)
        rows = [[True, False]] + [[False, False]] * 3 + [[False, True]]
        sae = run(rows, config=cfg, ring=ring)
        assert sae._sensed_edges == []

    def test_a_fire_exactly_at_the_window_edge_matches(self):
        cfg = make_config(max_lag=3)
        ring = EdgeFireRing(cfg.max_token_lag)
        rows = [[True, False], [False, False], [False, False], [False, True]]
        sae = run(rows, config=cfg, ring=ring)
        assert len(sae._sensed_edges) == 1
        assert sae._sensed_edges[0].token_lag == 3

    def test_the_nearest_antecedent_wins(self):
        """Two upstream fires in window: the closest is the most defensible
        attribution, so it must be the one reported."""
        sae = run([[True, False], [True, False], [False, True]])
        assert len(sae._sensed_edges) == 1
        assert sae._sensed_edges[0].up_pos == 1


class TestCrossLayerCooperation:
    def test_two_saes_share_the_ring(self):
        """The upstream SAE owns layer 10, the downstream owns layer 13.
        Neither could detect the edge alone."""
        spec = make_spec(up_layer=10, down_layer=13, key="1@10->2@13")
        ring = EdgeFireRing(4)

        up_spec = EdgeSpec(**{**spec.__dict__, "up_col": 0, "down_col": -1})
        up_cfg = make_config(edges=[up_spec], layer=10)
        run([[True, False]], config=up_cfg, ring=ring, base=0)

        down_spec = EdgeSpec(**{**spec.__dict__, "up_col": -1, "down_col": 1})
        down_cfg = make_config(edges=[down_spec], layer=13)
        down = run([[False, False], [False, True]], config=down_cfg, ring=ring, base=0)

        assert len(down._sensed_edges) == 1
        ev = down._sensed_edges[0]
        assert ev.up_layer == 10 and ev.down_layer == 13
        assert ev.up_pos == 0 and ev.down_pos == 1

    def test_an_sae_owning_neither_end_records_nothing(self):
        spec = EdgeSpec(**{**make_spec().__dict__, "up_col": -1, "down_col": -1})
        sae = run([[True, True]], config=make_config(edges=[spec]))
        assert sae._sensed_edges == []


class TestAbsolutePositionsAcrossPasses:
    def test_a_prefill_fire_matches_a_decode_fire(self):
        """Positions are absolute across passes; a base offset must not reset
        the ring's view of ordering."""
        cfg = make_config(max_lag=8)
        ring = EdgeFireRing(cfg.max_token_lag)
        run([[False, False], [True, False]], config=cfg, ring=ring, base=0)
        sae = run([[False, True]], config=cfg, ring=ring, base=2)
        assert len(sae._sensed_edges) == 1
        ev = sae._sensed_edges[0]
        assert (ev.up_pos, ev.down_pos, ev.token_lag) == (1, 2, 1)


class TestCapAndTruncation:
    def test_the_cap_stops_collection_and_sets_truncated(self):
        cfg = make_config(cap=2, max_lag=64)
        ring = EdgeFireRing(cfg.max_token_lag)
        rows = []
        for _ in range(5):
            rows.append([True, False])
            rows.append([False, True])
        sae = run(rows, config=cfg, ring=ring)
        assert len(sae._sensed_edges) == 2
        assert sae._edge_truncated is True
        assert sae._edge_done is True


class TestRingHygiene:
    def test_prune_drops_fires_that_can_no_longer_match(self):
        ring = EdgeFireRing(2)
        ring.record_up("e", 0, 1.0)
        ring.prune_before(10)
        assert ring.match_down("e", 11) is None

    def test_prune_keeps_fires_still_in_window(self):
        ring = EdgeFireRing(4)
        ring.record_up("e", 8, 1.0)
        ring.prune_before(9)
        assert ring.match_down("e", 10) == (8, 1.0)

    def test_clear_empties_the_ring(self):
        ring = EdgeFireRing(4)
        ring.record_up("e", 1, 1.0)
        ring.clear()
        assert ring.match_down("e", 2) is None

    def test_match_is_strictly_before(self):
        ring = EdgeFireRing(4)
        ring.record_up("e", 5, 1.0)
        assert ring.match_down("e", 5) is None
        assert ring.match_down("e", 6) == (5, 1.0)


class TestReportedEvidence:
    def test_the_rung_language_is_carried_verbatim_not_recomposed(self):
        spec = make_spec(rung=0)
        spec.rung_language = "associated"
        sae = run([[True, False], [False, True]], config=make_config(edges=[spec]))
        ev = sae._sensed_edges[0]
        assert ev.rung == 0
        assert ev.rung_language == "associated"
        assert "causal" not in ev.rung_language.lower()

    def test_activations_are_reported_for_both_endpoints(self):
        sae = run([[True, False], [False, True]])
        ev = sae._sensed_edges[0]
        assert ev.up_act > 0.5 and ev.down_act > 0.5


class TestArmingAgainstARealSAE:
    def test_arming_builds_the_member_slice_and_cpu_mirror(self):
        sae = real_sae()
        cfg = make_config()
        sae.arm_edge_sensing(cfg, EdgeFireRing(4))
        assert sae.is_edge_sensing_armed is True
        assert sae._W_enc_e is not None and sae._W_enc_e.shape == (D_IN, 2)
        assert sae._edge_thresholds_cpu == [0.5, 0.5]

    def test_edge_arming_is_independent_of_cluster_sensing(self):
        """A deployment may run both; is_edge_sensing_armed is deliberately
        distinct from is_sensing_armed."""
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        assert sae.is_edge_sensing_armed is True
        assert sae.is_sensing_armed is False

    def test_an_out_of_range_member_is_refused_at_arm_time(self):
        """An out-of-range index_select on CUDA is a device-side assert that
        poisons the process. More exposed here than in F11: a mis-keyed layer
        lookup would index into the WRONG SAE's feature space."""
        sae = real_sae()
        cfg = make_config()
        cfg.member_indices = [1, D_SAE + 5]
        with pytest.raises(ValueError, match="out of range"):
            sae.arm_edge_sensing(cfg, EdgeFireRing(4))

    def test_disarm_releases_every_cache(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae.disarm_edge_sensing()
        assert sae.is_edge_sensing_armed is False
        assert sae._W_enc_e is None and sae._b_enc_e is None
        assert sae._edge_ring is None and sae._edge_thresholds_cpu == []

    def test_arming_is_idempotent(self):
        sae = real_sae()
        for _ in range(3):
            sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        assert sae._W_enc_e.shape == (D_IN, 2)


class TestBufferHygiene:
    def test_without_begin_a_pass_senses_nothing(self):
        """FTID pitfall 2: omitting _edge_began from the guard would buffer
        stale cross-request edges."""
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._sensed_edges == []

    def test_begin_then_sense_then_collect_closes_the_boundary(self):
        sae = real_sae()
        ring = EdgeFireRing(4)
        sae.arm_edge_sensing(make_config(), ring)
        sae.begin_edge_sensing_request("req-1")
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))

        request_id, edges, truncated = sae.collect_sensed_edges()
        assert request_id == "req-1"
        assert len(edges) == 1 and truncated is False
        assert sae._edge_began is False

        # A second collect on a closed boundary yields nothing.
        assert sae.collect_sensed_edges() == ("", [], False)

    def test_begin_resets_a_previous_request_buffer(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae.begin_edge_sensing_request("req-1")
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        sae.begin_edge_sensing_request("req-2")
        assert sae._sensed_edges == []
        assert sae._edge_token_offset == 0 and sae._edge_phase == "prefill"

    def test_suppressed_blocks_sensing(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae.begin_edge_sensing_request("req-1")
        with sae.suppressed():
            sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._sensed_edges == []

    def test_a_batched_pass_is_skipped_not_sensed_as_row_zero(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae.begin_edge_sensing_request("req-1")
        batched = hidden({1: 2.0}, {2: 2.0}).unsqueeze(0).repeat(3, 1, 1)
        sae._sense_edges(batched)
        assert sae._sensed_edges == []
        assert sae._edge_batch_warned is True


class TestOffsetAndPhaseAccounting:
    def test_offset_and_phase_advance_across_passes(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(max_lag=16), EdgeFireRing(16))
        sae.begin_edge_sensing_request("req-1")

        sae._sense_edges(hidden({1: 2.0}, {0: 0.0}, {0: 0.0}))   # prefill, 3 tok
        assert sae._edge_token_offset == 3
        assert sae._edge_phase == "decode"

        sae._sense_edges(hidden({2: 2.0}))                        # decode, 1 tok
        assert sae._edge_token_offset == 4
        assert len(sae._sensed_edges) == 1
        ev = sae._sensed_edges[0]
        assert (ev.up_pos, ev.down_pos) == (0, 3), "absolute positions across passes"

    def test_offset_still_advances_when_the_cap_short_circuits(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(cap=0), EdgeFireRing(4))
        sae.begin_edge_sensing_request("req-1")
        sae._edge_done = True
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._edge_token_offset == 2, "offset must advance in finally"

    def test_overhead_is_accumulated(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae.begin_edge_sensing_request("req-1")
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._edge_overhead_ms > 0.0


class TestDeviceMigration:
    def test_to_device_moves_the_edge_caches(self):
        """011 R3: a device move on an armed SAE left the member slices behind
        and every pass threw silently. The edge caches must move too."""
        sae = real_sae()
        cfg = make_config()
        sae.arm_edge_sensing(cfg, EdgeFireRing(4))
        sae.to_device("cpu")
        assert sae._W_enc_e is not None and sae._b_enc_e is not None
        assert str(sae._W_enc_e.device) == "cpu"
        assert str(cfg.thresholds.device) == "cpu"
