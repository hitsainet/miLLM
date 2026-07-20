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


class TestUpstreamNoiseDoesNotDestroyCrossLayerDetection:
    """R1 CRITICAL regression.

    prune_before was called per position INSIDE _match_edges, so the upstream
    layer's hook walked an entire prefill and pruned the ring down to
    (last_pos - max_lag) before the downstream layer's hook ever ran. Any
    ordinary traffic on the upstream layer therefore destroyed the fire the
    downstream needed — cross-layer sensing, the whole point of the feature,
    went silently dark while status still reported "armed".

    The original tests missed it because they used single-layer edges and
    quiet rows: _match_edges `continue`s on a row where nothing fired, so the
    prune never ran in the fixtures.
    """

    def test_a_busy_upstream_layer_still_detects_the_edge(self):
        spec = make_spec(up_layer=10, down_layer=13, key="1@10->2@13")
        ring = EdgeFireRing(3)

        # The upstream SAE walks the WHOLE prefill: the real fire is at
        # position 2, then a sibling member keeps firing at 6..11.
        up_spec = EdgeSpec(**{**spec.__dict__, "up_col": 0, "down_col": -1})
        up_rows = [[False, False]] * 12
        up_rows[2] = [True, False]
        for p in range(6, 12):
            up_rows[p] = [True, False]
        run(up_rows, config=make_config(edges=[up_spec], max_lag=3, layer=10),
            ring=ring, base=0)

        # THEN the downstream SAE walks the same prefill and fires at 4.
        down_spec = EdgeSpec(**{**spec.__dict__, "up_col": -1, "down_col": 1})
        down_rows = [[False, False]] * 12
        down_rows[4] = [False, True]
        down = run(down_rows,
                   config=make_config(edges=[down_spec], max_lag=3, layer=13),
                   ring=ring, base=0)

        assert len(down._sensed_edges) == 1, (
            "the upstream layer pruned away the pos-2 fire before the "
            "downstream hook ran"
        )
        ev = down._sensed_edges[0]
        assert (ev.up_pos, ev.down_pos, ev.token_lag) == (2, 4, 2)

    def test_no_hook_calls_prune_before(self):
        """Pruning is a REQUEST-level operation. A hook cannot know whether a
        sibling layer still needs a fire, so re-introducing a prune call into
        the matcher would silently restore the bug above."""
        import inspect

        src = inspect.getsource(LoadedSAE._match_edges)
        assert "prune_before" not in src

    def test_the_ring_bounds_its_own_growth(self):
        """Without positional pruning the ring must still not grow without
        bound over a long request."""
        ring = EdgeFireRing(4)
        for pos in range(5000):
            ring.record_up("e", pos, 1.0)
        assert len(ring._fires["e"]) <= EdgeFireRing._MAX_FIRES_PER_EDGE

    def test_bounding_keeps_the_newest_fires(self):
        """match_down reports the nearest antecedent, so recent history is
        what must survive."""
        ring = EdgeFireRing(4)
        for pos in range(EdgeFireRing._MAX_FIRES_PER_EDGE + 50):
            ring.record_up("e", pos, 1.0)
        newest = EdgeFireRing._MAX_FIRES_PER_EDGE + 49
        assert ring.match_down("e", newest + 1) == (newest, 1.0)


class TestPositionOffsetsStayInSync:
    """R1 CRITICAL: the early returns skipped the `finally` that advances
    _edge_token_offset, so a suppressed/unarmed/batched pass left one SAE's
    offset behind its siblings'. The ring is keyed on ABSOLUTE position and
    shared, so that silently shifts one layer's coordinates relative to
    another's — fabricating matches and losing real ones."""

    def test_a_suppressed_pass_still_advances_the_offset(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae.begin_edge_sensing_request("req-1")
        with sae.suppressed():
            sae._sense_edges(hidden({1: 2.0}, {2: 2.0}, {1: 0.0}))
        assert sae._edge_token_offset == 3

    def test_a_batched_pass_still_advances_the_offset(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae.begin_edge_sensing_request("req-1")
        batched = hidden({1: 2.0}, {2: 2.0}).unsqueeze(0).repeat(3, 1, 1)
        sae._sense_edges(batched)
        assert sae._edge_token_offset == 2

    def test_two_saes_stay_aligned_when_one_is_suppressed(self):
        """The failure this prevents: sibling coordinates drift apart."""
        up, down = real_sae(), real_sae()
        ring = EdgeFireRing(8)
        up.arm_edge_sensing(make_config(layer=10), ring)
        down.arm_edge_sensing(make_config(layer=13), ring)
        up.begin_edge_sensing_request("r")
        down.begin_edge_sensing_request("r")

        with up.suppressed():
            up._sense_edges(hidden({1: 0.0}, {1: 0.0}))
        down._sense_edges(hidden({1: 0.0}, {1: 0.0}))
        assert up._edge_token_offset == down._edge_token_offset == 2


class TestColumnValidation:
    def test_an_out_of_range_column_is_refused_at_arm_time(self):
        """R1 CRITICAL: an IndexError in the matcher was swallowed by the
        broad except, abandoning the ENTIRE pass — every edge, including
        upstream recording — rather than one bad spec."""
        sae = real_sae()
        bad = EdgeSpec(**{**make_spec().__dict__, "down_col": 99})
        with pytest.raises(ValueError, match="column out of range"):
            sae.arm_edge_sensing(make_config(edges=[bad]), EdgeFireRing(4))


class TestSaturationLoadShedding:
    """R1 CRITICAL: the per-request cap bounds OUTPUT, but the cost of finding
    fires is paid first — a miscalibrated threshold on a long prefill cost
    1430ms inside the forward hook against a 5ms budget."""

    def _saturating(self, seq=512):
        sae = real_sae()
        cfg = make_config()
        cfg.thresholds = torch.tensor([0.0001, 0.0001])
        cfg.max_events_per_request = 20
        sae.arm_edge_sensing(cfg, EdgeFireRing(4))
        sae.begin_edge_sensing_request("r")
        return sae, torch.rand(seq, D_IN) + 1.0

    def test_a_saturated_pass_is_shed_and_flagged(self):
        sae, hs = self._saturating(seq=4096)
        sae._sense_edges(hs)
        assert sae._edge_truncated is True, "shedding must be visible, not silent"
        assert sae._edge_saturation_warned is True

    def test_a_saturated_pass_stays_inside_the_latency_budget(self):
        import time

        sae, hs = self._saturating(seq=4096)
        t = time.perf_counter()
        sae._sense_edges(hs)
        ms = (time.perf_counter() - t) * 1000
        assert ms < 50.0, f"saturated pass cost {ms:.1f}ms in the forward hook"

    def test_an_ordinary_pass_is_not_shed(self):
        sae = real_sae()
        sae.arm_edge_sensing(make_config(max_lag=16), EdgeFireRing(16))
        sae.begin_edge_sensing_request("r")
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert len(sae._sensed_edges) == 1
        assert sae._edge_saturation_warned is False

    def test_warn_flags_reset_per_request(self):
        """R1: never reset, so a later independent violation went unlogged."""
        sae, hs = self._saturating(seq=4096)
        sae._sense_edges(hs)
        assert sae._edge_saturation_warned is True
        sae.begin_edge_sensing_request("r2")
        assert sae._edge_saturation_warned is False
        assert sae._edge_batch_warned is False


class TestCollectRequiresABoundary:
    def test_collect_without_begin_returns_nothing(self):
        """F11 parity: draining without an open boundary would surface stale
        edges attributed to an empty request_id."""
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        assert sae.collect_sensed_edges() == ("", [], False)


class TestMatchDownIsBounded:
    """R2: R1 traded a 286x latency blowout for a 15x one. match_down linear
    scanned up to 512 retained fires PER DOWNSTREAM FIRE — 78.5ms on a
    4096-token pass. `fires` is ascending, so the scan runs backward and breaks
    at the window edge."""

    def test_a_deep_ring_still_answers_in_constant_time(self):
        import time

        ring = EdgeFireRing(4)
        for pos in range(EdgeFireRing._MAX_FIRES_PER_EDGE):
            ring.record_up("e", pos, 1.0)
        last = EdgeFireRing._MAX_FIRES_PER_EDGE - 1

        t = time.perf_counter()
        for _ in range(2000):
            ring.match_down("e", last + 1)
        ms = (time.perf_counter() - t) * 1000
        assert ms < 50.0, f"2000 lookups against a full ring cost {ms:.1f}ms"

    def test_an_out_of_window_lookup_breaks_early(self):
        ring = EdgeFireRing(2)
        for pos in range(500):
            ring.record_up("e", pos, 1.0)
        assert ring.match_down("e", 10_000) is None

    def test_it_still_returns_the_nearest_antecedent(self):
        ring = EdgeFireRing(10)
        for pos in (1, 3, 7):
            ring.record_up("e", pos, float(pos))
        assert ring.match_down("e", 8) == (7, 7.0)


class TestShedStillFeedsSiblings:
    """R2 CRITICAL: R1's load shedding returned BEFORE recording upstream
    fires. Shedding is per-SAE per-pass, so a saturated UPSTREAM layer silently
    blinded a quiet downstream sibling that did not shed — and the truncated
    flag landed on the layer that shed, not the layer that lost data. The
    operator saw a clean, empty result: the silently-dark mode R1-01 existed to
    eliminate, reintroduced by the R1-02 fix."""

    def test_a_shedding_upstream_layer_still_records_for_its_sibling(self):
        ring = EdgeFireRing(64)
        spec = make_spec(up_layer=10, down_layer=13, key="1@10->2@13")

        # Upstream layer: saturated, so it sheds.
        up = real_sae()
        up_cfg = make_config(
            edges=[EdgeSpec(**{**spec.__dict__, "up_col": 0, "down_col": -1})],
            max_lag=64, layer=10,
        )
        up_cfg.thresholds = torch.tensor([0.0001, 0.0001])
        up.arm_edge_sensing(up_cfg, ring)
        up.begin_edge_sensing_request("r")
        up._sense_edges(torch.rand(4096, D_IN) + 1.0)
        assert up._edge_truncated is True, "the saturated layer must flag it"

        # Downstream layer: quiet, does NOT shed. It must still see the
        # upstream fires the shedding layer recorded.
        down = real_sae()
        down_cfg = make_config(
            edges=[EdgeSpec(**{**spec.__dict__, "up_col": -1, "down_col": 1})],
            max_lag=64, layer=13,
        )
        down.arm_edge_sensing(down_cfg, ring)
        down.begin_edge_sensing_request("r")
        rows = [{2: 0.0}] * 4095 + [{2: 5.0}]
        down._sense_edges(hidden(*rows))

        assert down._sensed_edges, (
            "the shedding upstream layer starved its sibling — R1's return "
            "recorded nothing into the shared ring"
        )


class TestColumnSentinelValidation:
    def test_minus_one_is_a_legitimate_not_my_half_sentinel(self):
        sae = real_sae()
        spec = EdgeSpec(**{**make_spec().__dict__, "up_col": -1})
        sae.arm_edge_sensing(make_config(edges=[spec]), EdgeFireRing(4))
        assert sae.is_edge_sensing_armed is True

    def test_a_column_below_minus_one_is_refused(self):
        """R2: validation checked only the upper bound, so -2 passed and the
        matcher's `0 <= col` guard silently skipped that half — the edge
        reported armed and sensable and simply never fired."""
        sae = real_sae()
        spec = EdgeSpec(**{**make_spec().__dict__, "down_col": -2})
        with pytest.raises(ValueError, match="column out of range"):
            sae.arm_edge_sensing(make_config(edges=[spec]), EdgeFireRing(4))


class TestSensingFailuresAreNotSilent:
    """R2 process finding: while fixing the shedding bug I introduced a
    NameError inside _sense_edges. The broad `except` swallowed it and turned a
    hard crash into silent non-detection — the suite stayed green on the ring
    tests and only an end-to-end assertion caught it. A pass that raises must
    be observable."""

    def test_a_raising_pass_is_logged_not_swallowed_silently(self, caplog):
        import logging

        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae.begin_edge_sensing_request("r")
        sae._match_edges = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        with caplog.at_level(logging.ERROR):
            sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert any(
            "edge_sensing_pass_failed" in r.message or "boom" in str(r)
            for r in caplog.records
        ), "a failing pass must leave a trace"

    def test_a_raising_pass_still_advances_the_offset(self):
        """Otherwise one failure desynchronises this SAE from its siblings for
        the rest of the request."""
        sae = real_sae()
        sae.arm_edge_sensing(make_config(), EdgeFireRing(4))
        sae.begin_edge_sensing_request("r")
        sae._match_edges = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        sae._sense_edges(hidden({1: 2.0}, {2: 2.0}))
        assert sae._edge_token_offset == 2
