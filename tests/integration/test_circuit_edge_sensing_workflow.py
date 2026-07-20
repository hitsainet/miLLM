"""Feature 15 Task 6.0: edge sensing end-to-end.

Drives the REAL hook path — arm the service against real LoadedSAEs, run
hidden states through `_sense_edges` exactly as `sae_hooker.hook_fn` does, and
assert the observations that come out. The unit tests exercise `_match_edges`
directly; these prove the pieces are actually connected, which is where a
sensing feature fails in practice (it goes quietly dark rather than breaking).
"""

from types import SimpleNamespace

import pytest
import torch

import millm.api.dependencies as deps
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.circuit_sensing_service import CircuitSensingService

D_IN = 8
D_SAE = 32

#: Feature 1 reads input dim 1, feature 2 reads dim 2 (W_enc is one-hot on
#: j % D_IN), so a hidden state dials each member's activation directly.
UP_IDX, DOWN_IDX = 1, 2
UP_LAYER, DOWN_LAYER = 10, 13


@pytest.fixture(autouse=True)
def clean():
    deps._circuit_sensing_service = None
    yield
    deps._circuit_sensing_service = None


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


def hidden(*rows) -> torch.Tensor:
    out = torch.zeros(len(rows), D_IN)
    for i, row in enumerate(rows):
        for dim, val in row.items():
            out[i, dim] = val
    return out


def definition(rung=2):
    return SimpleNamespace(
        edges=[
            SimpleNamespace(
                up=SimpleNamespace(layer=UP_LAYER, feature_idx=UP_IDX, kind="feature"),
                down=SimpleNamespace(
                    layer=DOWN_LAYER, feature_idx=DOWN_IDX, kind="feature"
                ),
                rung=rung,
                type="computed",
            )
        ],
        members=[
            SimpleNamespace(
                layer=UP_LAYER,
                feature=SimpleNamespace(feature_idx=UP_IDX, max_activation=10.0),
                expanded_members=None,
            ),
            SimpleNamespace(
                layer=DOWN_LAYER,
                feature=SimpleNamespace(feature_idx=DOWN_IDX, max_activation=10.0),
                expanded_members=None,
            ),
        ],
    )


def circuit(**overrides):
    base = dict(id="circ_1", name="fear→threat", circuit_meta={}, sensing_enabled=True)
    base.update(overrides)
    return SimpleNamespace(**base)


def hook(sae, hidden_states):
    """Exactly what sae_hooker.hook_fn does for the edge-sensing branch."""
    if sae.is_edge_sensing_armed:
        with torch.no_grad():
            sae._sense_edges(hidden_states)


@pytest.fixture
def armed():
    """A circuit armed across two real SAEs on layers 10 and 13."""
    svc = CircuitSensingService()
    deps._circuit_sensing_service = svc
    saes = {UP_LAYER: make_sae(), DOWN_LAYER: make_sae()}
    unsensable = svc.arm_for_circuit(circuit(), definition(), saes)
    assert unsensable == []
    return svc, saes


class TestArmToObservation:
    def test_an_up_then_down_firing_is_observed(self, armed):
        """Task 6.1: arm → generate → an event with the right lag and rung."""
        svc, saes = armed
        assert svc.begin_request("req-1", saes) is True

        # Prefill: upstream fires at position 0 in layer 10's hook.
        hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}, {UP_IDX: 0.0}))
        # ...and layer 13 sees the same two positions, firing nothing.
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 0.0}, {DOWN_IDX: 0.0}))
        # Decode token 2: downstream fires in layer 13's hook.
        hook(saes[UP_LAYER], hidden({UP_IDX: 0.0}))
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 5.0}))

        request_id, edges, truncated = svc.collect_edges(saes)
        assert request_id == "req-1"
        assert len(edges) == 1
        ev = edges[0]
        assert (ev.up_layer, ev.up_feature_idx, ev.up_pos) == (UP_LAYER, UP_IDX, 0)
        assert (ev.down_layer, ev.down_feature_idx, ev.down_pos) == (
            DOWN_LAYER, DOWN_IDX, 2,
        )
        assert ev.token_lag == 2
        assert ev.rung == 2 and ev.rung_language == "causally validated (edge)"
        assert truncated is False

    def test_a_lone_upstream_firing_yields_no_event(self, armed):
        """EC-15.1 end-to-end."""
        svc, saes = armed
        svc.begin_request("req-1", saes)
        hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}, {UP_IDX: 5.0}))
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 0.0}, {DOWN_IDX: 0.0}))
        _, edges, _ = svc.collect_edges(saes)
        assert edges == []

    def test_a_reversed_firing_yields_no_event(self, armed):
        """EC-15.2 end-to-end: downstream first is not the authored direction."""
        svc, saes = armed
        svc.begin_request("req-1", saes)
        hook(saes[UP_LAYER], hidden({UP_IDX: 0.0}))
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 5.0}))
        hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}))
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 0.0}))
        _, edges, _ = svc.collect_edges(saes)
        assert edges == []

    def test_a_firing_beyond_the_lag_window_yields_no_event(self):
        svc = CircuitSensingService()
        deps._circuit_sensing_service = svc
        saes = {UP_LAYER: make_sae(), DOWN_LAYER: make_sae()}
        svc.arm_for_circuit(
            circuit(circuit_meta={"sensing": {"max_token_lag": 1}}),
            definition(),
            saes,
        )
        svc.begin_request("req-1", saes)
        hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}))
        for _ in range(3):
            hook(saes[UP_LAYER], hidden({UP_IDX: 0.0}))
            hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 0.0}))
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 5.0}))
        _, edges, _ = svc.collect_edges(saes)
        assert edges == []


class TestLifecycle:
    def test_disarm_stops_observation(self, armed):
        svc, saes = armed
        svc.disarm(saes)
        assert not any(s.is_edge_sensing_armed for s in saes.values())

        # A pass after disarm must be inert, not merely unrecorded.
        hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}))
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 5.0}))
        assert saes[DOWN_LAYER]._sensed_edges == []

    def test_a_new_request_does_not_inherit_the_previous_ring(self, armed):
        """The shared ring is cleared once per request BY THE SERVICE; a stale
        upstream fire must not match a downstream fire in the next request."""
        svc, saes = armed
        svc.begin_request("req-1", saes)
        hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}))
        svc.collect_edges(saes)

        svc.begin_request("req-2", saes)
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 5.0}))
        _, edges, _ = svc.collect_edges(saes)
        assert edges == [], "a fire from the previous request matched"

    def test_unsensable_edges_are_reported_when_a_layer_is_absent(self):
        """Task 6.3 / EC-15.4: slice-fallback serves one layer."""
        svc = CircuitSensingService()
        deps._circuit_sensing_service = svc
        unsensable = svc.arm_for_circuit(
            circuit(), definition(), {UP_LAYER: make_sae()}
        )
        assert len(unsensable) == 1
        assert unsensable[0].reason == "layer_not_attached"
        assert svc.status()["sensable_edges"] == 0


class TestSafety:
    def test_an_unarmed_circuit_adds_no_overhead(self, armed):
        """Task 6.4: the un-armed path must be inert."""
        svc, saes = armed
        svc.disarm(saes)
        for sae in saes.values():
            sae._edge_overhead_ms = 0.0
            hook(sae, hidden({UP_IDX: 5.0}))
            assert sae._edge_overhead_ms == 0.0

    def test_the_overhead_accumulator_is_populated_and_summed(self, armed):
        svc, saes = armed
        svc.begin_request("req-1", saes)
        hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}))
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 5.0}))
        svc.collect_edges(saes)
        assert svc._last_request_overhead_ms > 0.0

    def test_the_latency_budget_is_respected(self, armed):
        """NFR-1.5: a 128-token prefill must stay well inside the budget."""
        svc, saes = armed
        svc.begin_request("req-1", saes)
        rows = [{UP_IDX: 5.0 if i % 2 == 0 else 0.0} for i in range(128)]
        hook(saes[UP_LAYER], hidden(*rows))
        hook(saes[DOWN_LAYER], hidden(*[{DOWN_IDX: 5.0} for _ in range(128)]))
        svc.collect_edges(saes)
        assert svc._last_request_overhead_ms < 100.0, (
            f"edge sensing cost {svc._last_request_overhead_ms:.1f}ms on a "
            "128-token prefill"
        )

    def test_the_per_request_cap_truncates_rather_than_growing(self, armed):
        svc, saes = armed
        svc.begin_request("req-1", saes)
        for _ in range(60):
            hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}, {UP_IDX: 0.0}))
            hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 0.0}, {DOWN_IDX: 5.0}))
        _, edges, truncated = svc.collect_edges(saes)
        assert truncated is True
        assert len(edges) <= 20


class TestEvidenceHonesty:
    def test_a_rung_zero_edge_is_never_described_as_causal(self):
        """Task 6.5: the surfaced strings must not overclaim."""
        svc = CircuitSensingService()
        deps._circuit_sensing_service = svc
        saes = {UP_LAYER: make_sae(), DOWN_LAYER: make_sae()}
        svc.arm_for_circuit(circuit(), definition(rung=0), saes)
        svc.begin_request("req-1", saes)
        # Upstream at position 0, downstream at position 1 — a same-position
        # co-fire is correctly not an edge, so the two must be separated.
        hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}, {UP_IDX: 0.0}))
        hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 0.0}, {DOWN_IDX: 5.0}))

        _, edges, _ = svc.collect_edges(saes)
        assert len(edges) == 1
        ev = edges[0]
        assert ev.rung == 0
        assert ev.rung_language == "associated"
        assert "causal" not in ev.rung_language.lower()
        assert "causal" not in svc.summarize(ev).lower()

    def test_observing_an_edge_never_upgrades_its_rung(self, armed):
        """Watching an edge fire is co-activation evidence, not validation."""
        svc, saes = armed
        svc.begin_request("req-1", saes)
        for _ in range(5):
            hook(saes[UP_LAYER], hidden({UP_IDX: 5.0}, {UP_IDX: 0.0}))
            hook(saes[DOWN_LAYER], hidden({DOWN_IDX: 0.0}, {DOWN_IDX: 5.0}))
        _, edges, _ = svc.collect_edges(saes)
        assert edges, "expected observations"
        assert all(e.rung == 2 for e in edges), "rung must come from the document"
