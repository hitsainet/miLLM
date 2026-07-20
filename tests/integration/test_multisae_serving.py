"""Integration verification for multi-SAE circuit serving (Feature 12, task 6.0).

Uses REAL (CPU) LoadedSAE objects registered in the AttachedSAEState registry
and drives the full serving path — set_circuit_steering → each layer's real
set_steering_batch → get_steering_values — so the per-layer round-trip, the
SAE_SET_INCOMPLETE fail-closed path, per-layer detach, and the degenerate /
same-layer-conflict cases are exercised end to end (no GPU required).
"""

import pytest
import torch

from millm.api.schemas.circuit import CircuitMember
from millm.core.errors import SAESetIncompleteError
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.sae_service import AttachedSAEState, SAEService


def make_sae(d_in: int = 32, d_sae: int = 128) -> LoadedSAE:
    config = SAEConfig(
        d_in=d_in, d_sae=d_sae, model_name="test", hook_name="test", hook_layer=0
    )
    return LoadedSAE(
        W_enc=torch.randn(d_in, d_sae),
        b_enc=torch.zeros(d_sae),
        W_dec=torch.randn(d_sae, d_in),
        b_dec=torch.zeros(d_in),
        config=config,
        device="cpu",
    )


@pytest.fixture(autouse=True)
def clean_state():
    state = AttachedSAEState()
    state._entries.clear()
    yield
    state._entries.clear()


@pytest.fixture
def service():
    svc = SAEService.__new__(SAEService)
    svc._sae_state = AttachedSAEState()
    return svc


def _attach(state, sae, sae_id, layer):
    # No hook handle needed for the serving math path.
    state.set(sae, sae_id, layer, None)


class TestPerLayerRoundTrip:
    def test_each_layer_gets_its_own_members_at_lambda_scaled_values(self, service):
        s10, s13 = make_sae(), make_sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        _attach(service._sae_state, s13, "sae-13", 13)
        members = [
            CircuitMember(feature_idx=5, layer=10, budget=40.0, sign=1),
            CircuitMember(feature_idx=9, layer=13, budget=30.0, sign=-1),
        ]
        result = service.set_circuit_steering(members, intensity=1.5)

        # Real per-layer steering state reflects clamp(budget*sign*λ).
        assert s10.get_steering_values() == {5: 60.0}      # 40*1*1.5
        assert s13.get_steering_values() == {9: -45.0}     # 30*-1*1.5
        # No cross-contamination between layers.
        assert 9 not in s10.get_steering_values()
        assert 5 not in s13.get_steering_values()
        assert result.applied_per_layer == {10: {5: 60.0}, 13: {9: -45.0}}

    def test_single_layer_degenerate_case(self, service):
        s10 = make_sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1)]
        service.set_circuit_steering(members, intensity=1.0)
        assert s10.get_steering_values() == {1: 50.0}


class TestIncompleteAndConflict:
    def test_incomplete_set_422_nothing_applied(self, service):
        s10 = make_sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=40.0, sign=1),  # no SAE on 13
        ]
        with pytest.raises(SAESetIncompleteError):
            service.set_circuit_steering(members, intensity=1.0)
        # Fail-closed: layer 10 was NOT partially applied.
        assert s10.get_steering_values() == {}

    def test_same_layer_two_saes_is_ambiguous(self, service):
        _attach(service._sae_state, make_sae(), "sae-a", 10)
        _attach(service._sae_state, make_sae(), "sae-b", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1)]
        with pytest.raises(SAESetIncompleteError):
            service.set_circuit_steering(members, intensity=1.0)

    def test_out_of_bounds_index_422_not_valueerror(self, service):
        s10 = make_sae(d_sae=64)
        _attach(service._sae_state, s10, "sae-10", 10)
        members = [CircuitMember(feature_idx=999, layer=10, budget=50.0, sign=1)]
        with pytest.raises(SAESetIncompleteError):
            service.set_circuit_steering(members, intensity=1.0)
        assert s10.get_steering_values() == {}


class TestDetachClearsPerLayer:
    def test_clear_circuit_steering_clears_each_layer(self, service):
        s10, s13 = make_sae(), make_sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        _attach(service._sae_state, s13, "sae-13", 13)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=50.0, sign=1),
        ]
        service.set_circuit_steering(members, intensity=1.0)
        assert s10.get_steering_values() and s13.get_steering_values()

        cleared = service.clear_circuit_steering()
        assert sorted(cleared) == [10, 13]
        assert s10.get_steering_values() == {}
        assert s13.get_steering_values() == {}
