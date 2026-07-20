"""Feature 15 Task 3.8: CircuitSensingService.

The interesting behaviour is REFUSAL. An edge is sensable only when both
endpoints resolve to an attached SAE with a usable threshold; every edge that
fails must be reported with a reason rather than silently dropped, because a
missing event is indistinguishable from "the edge never fired" — and that
silence would read as evidence of absence.
"""

from types import SimpleNamespace

import pytest
import torch

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.circuit_sensing_service import (
    CircuitSensingService,
    edge_key_for,
)

D_IN = 8
D_SAE = 32


def make_sae() -> LoadedSAE:
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


def node(layer, feature_idx, kind="feature"):
    return SimpleNamespace(layer=layer, feature_idx=feature_idx, kind=kind)


def edge(up_layer=10, up_idx=1, down_layer=13, down_idx=2, rung=2, type_="computed"):
    return SimpleNamespace(
        up=node(up_layer, up_idx),
        down=node(down_layer, down_idx),
        rung=rung,
        type=type_,
    )


def member(layer, feature_idx, max_activation=10.0):
    return SimpleNamespace(
        layer=layer,
        feature=SimpleNamespace(feature_idx=feature_idx, max_activation=max_activation),
        expanded_members=None,
    )


def definition(edges=None, members=None):
    return SimpleNamespace(
        edges=edges if edges is not None else [edge()],
        members=members
        if members is not None
        else [member(10, 1), member(13, 2)],
    )


def circuit(**overrides):
    base = dict(id="circ_1", name="fear→threat", circuit_meta={})
    base.update(overrides)
    return SimpleNamespace(**base)


def two_saes():
    return {10: make_sae(), 13: make_sae()}


class TestSensableEdgeResolution:
    def test_a_fully_resolved_edge_arms_both_layers(self):
        svc = CircuitSensingService()
        unsensable = svc.arm_for_circuit(circuit(), definition(), two_saes())
        assert unsensable == []
        assert svc.is_armed and svc._armed_layers == [10, 13]

    def test_each_sae_gets_only_its_own_half_of_the_edge(self):
        """A cross-layer edge is sensed cooperatively; a column offset that
        belonged to the other layer's slice would read the wrong feature."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)

        up_spec = svc._configs[10].edges[0]
        assert up_spec.up_col >= 0 and up_spec.down_col == -1
        down_spec = svc._configs[13].edges[0]
        assert down_spec.up_col == -1 and down_spec.down_col >= 0

    def test_the_rung_phrase_comes_from_the_ladder(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(), definition(edges=[edge(rung=0)]), two_saes()
        )
        spec = svc._configs[10].edges[0]
        assert spec.rung == 0
        assert spec.rung_language == "associated"
        assert "causal" not in spec.rung_language.lower()


class TestUnsensableEdges:
    def test_an_unattached_layer_is_reported_not_dropped(self):
        """EC-15.4: slice-fallback serves ONE layer, so nearly every
        cross-layer edge lands here."""
        svc = CircuitSensingService()
        unsensable = svc.arm_for_circuit(
            circuit(), definition(), {10: make_sae()}  # layer 13 missing
        )
        assert len(unsensable) == 1
        assert unsensable[0].reason == "layer_not_attached"
        assert "13" in unsensable[0].detail
        assert svc.is_armed is False, "nothing sensable ⇒ not armed"

    def test_a_cluster_supernode_endpoint_is_unsensable(self):
        """feature_idx is nullable and kind may be 'cluster' — there is no
        single activation to threshold. Neither EC-15.4 nor EC-15.6 covers it."""
        svc = CircuitSensingService()
        e = edge()
        e.up = node(10, None, kind="cluster")
        unsensable = svc.arm_for_circuit(circuit(), definition(edges=[e]), two_saes())
        assert len(unsensable) == 1
        assert unsensable[0].reason == "endpoint_not_a_feature"

    def test_a_member_without_max_activation_is_unsensable(self):
        """EC-15.6: no activation scale means theta is either 0 (fires on
        anything) or inf (never fires). Both make the edge unobservable."""
        svc = CircuitSensingService()
        unsensable = svc.arm_for_circuit(
            circuit(),
            definition(members=[member(10, 1, None), member(13, 2)]),
            two_saes(),
        )
        assert len(unsensable) == 1
        assert unsensable[0].reason == "no_activation_threshold"

    def test_a_zero_max_activation_is_treated_as_missing(self):
        """theta = eps*0 = 0 would fire on any positive activation — as
        degenerate as a missing stat (011 R3 #2)."""
        svc = CircuitSensingService()
        unsensable = svc.arm_for_circuit(
            circuit(),
            definition(members=[member(10, 1, 0.0), member(13, 2)]),
            two_saes(),
        )
        assert len(unsensable) == 1
        assert unsensable[0].reason == "no_activation_threshold"

    def test_sensable_and_unsensable_edges_coexist(self):
        svc = CircuitSensingService()
        good = edge(up_idx=1, down_idx=2)
        bad = edge(up_idx=3, down_idx=4, down_layer=99)
        unsensable = svc.arm_for_circuit(
            circuit(),
            definition(
                edges=[good, bad],
                members=[member(10, 1), member(13, 2), member(10, 3), member(99, 4)],
            ),
            two_saes(),
        )
        assert len(unsensable) == 1
        assert svc.is_armed is True
        assert svc.status()["sensable_edges"] == 1

    def test_unsensable_edges_are_surfaced_in_status(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(circuit(), definition(), {10: make_sae()})
        rows = svc.status()["unsensable_edges"]
        assert rows and rows[0]["reason"] == "layer_not_attached"
        assert "edge_key" in rows[0] and "detail" in rows[0]


class TestThresholds:
    def test_theta_is_epsilon_times_max_activation(self):
        svc = CircuitSensingService()
        assert svc._theta(10.0, 0.1, 0.0) == pytest.approx(1.0)

    def test_the_floor_wins_when_higher(self):
        svc = CircuitSensingService()
        assert svc._theta(1.0, 0.1, 0.5) == pytest.approx(0.5)

    def test_a_missing_stat_with_no_floor_is_infinite_not_zero(self):
        """Infinity means 'never fires'. Zero would mean 'always fires' —
        the wrong direction to fail in."""
        svc = CircuitSensingService()
        assert svc._theta(None, 0.1, 0.0) == float("inf")

    def test_a_missing_stat_with_a_positive_floor_uses_the_floor(self):
        svc = CircuitSensingService()
        assert svc._theta(None, 0.1, 0.25) == pytest.approx(0.25)


class TestOverrides:
    def test_an_authored_lag_override_is_honoured(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": 3}}),
            definition(),
            two_saes(),
        )
        assert svc._max_token_lag == 3

    def test_a_negative_override_degrades_to_the_default(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": -5}}),
            definition(),
            two_saes(),
        )
        assert svc._max_token_lag == 8

    def test_an_absurd_lag_is_capped(self):
        """A window wide enough to catch anything is a coincidence detector,
        not an attribution."""
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": 100_000}}),
            definition(),
            two_saes(),
        )
        assert svc._max_token_lag == 64

    def test_local_overrides_beat_document_overrides(self):
        svc = CircuitSensingService()
        svc.arm_for_circuit(
            circuit(
                circuit_meta={
                    "sensing": {"max_token_lag": 3},
                    "sensing_overrides": {"max_token_lag": 5},
                }
            ),
            definition(),
            two_saes(),
        )
        assert svc._max_token_lag == 5


class TestLifecycle:
    def test_begin_clears_the_shared_ring_once(self):
        """Clearing per-SAE would wipe fires a sibling already recorded."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc._ring.record_up("stale", 1, 1.0)

        assert svc.begin_request("req-1", saes) is True
        assert svc._ring.match_down("stale", 2) is None
        assert all(s._edge_began for s in saes.values())

    def test_collect_merges_every_layer_and_sums_overhead(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.begin_request("req-1", saes)
        for s in saes.values():
            s._edge_overhead_ms = 2.0

        request_id, edges, truncated = svc.collect_edges(saes)
        assert request_id == "req-1"
        assert svc._last_request_overhead_ms == pytest.approx(4.0), "summed, not assigned"

    def test_disarm_releases_every_layer(self):
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        svc.disarm(saes)
        assert svc.is_armed is False
        assert not any(s.is_edge_sensing_armed for s in saes.values())

    def test_status_reconciles_a_swallowed_disarm(self):
        """Reporting armed forever after a disarm that failed is the exact
        failure F11's reconciliation exists to prevent."""
        svc = CircuitSensingService()
        saes = two_saes()
        svc.arm_for_circuit(circuit(), definition(), saes)
        saes[13].disarm_edge_sensing()  # one layer lost behind the service's back

        assert svc.status(saes)["armed"] is False

    def test_begin_on_an_unarmed_service_is_a_noop(self):
        svc = CircuitSensingService()
        assert svc.begin_request("req-1", two_saes()) is False


class TestSummary:
    def test_the_summary_carries_the_rung_phrase_verbatim(self):
        svc = CircuitSensingService()
        ev = SimpleNamespace(
            up_feature_idx=1, up_layer=10, down_feature_idx=2, down_layer=13,
            token_lag=2, rung_language="associated",
        )
        text = svc.summarize(ev)
        assert "associated" in text
        assert "causal" not in text.lower()

    def test_a_rung_two_summary_names_the_ladder_phrase(self):
        svc = CircuitSensingService()
        ev = SimpleNamespace(
            up_feature_idx=1, up_layer=10, down_feature_idx=2, down_layer=13,
            token_lag=1, rung_language="causally validated (edge)",
        )
        assert "causally validated (edge)" in svc.summarize(ev)

    def test_the_summary_fits_the_column(self):
        svc = CircuitSensingService()
        ev = SimpleNamespace(
            up_feature_idx=1, up_layer=10, down_feature_idx=2, down_layer=13,
            token_lag=1, rung_language="x" * 500,
        )
        assert len(svc.summarize(ev)) <= 300


class TestEdgeKey:
    def test_the_key_is_stable_and_directional(self):
        assert edge_key_for(10, 1, 13, 2) == "1@10->2@13"
        assert edge_key_for(13, 2, 10, 1) != edge_key_for(10, 1, 13, 2)
