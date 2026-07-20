"""Unit tests for circuit serving (Feature 12, task 3.0).

Pins the multi-SAE serving contract: group-by-layer application through each
layer's own SAE, per-layer budgets under one global λ with the shared ±200
clamp (γ=0 ⇒ B=B_dir), SAE_SET_INCOMPLETE on any unresolved/ambiguous/out-of-
range member (nothing applied), cross-layer hazard labeling, config never
mutated by hazard detection, and per-layer clear.
"""

from unittest.mock import MagicMock

import pytest

from millm.api.schemas.circuit import CircuitMember
from millm.core.errors import SAESetIncompleteError
from millm.services.sae_service import AttachedSAEState, SAEService


@pytest.fixture(autouse=True)
def reset_registry():
    state = AttachedSAEState()
    state._entries.clear()
    yield
    state._entries.clear()


def _sae(d_sae: int = 8192):
    sae = MagicMock()
    sae.d_sae = d_sae
    applied: dict[int, float] = {}

    def _set_batch(steering: dict[int, float]) -> None:
        for idx in steering:
            if not 0 <= idx < d_sae:
                raise ValueError(f"idx {idx} out of range")
        applied.update(steering)

    def _clear(feature_idx=None):
        if feature_idx is None:
            applied.clear()
        else:
            applied.pop(feature_idx, None)

    sae.set_steering_batch.side_effect = _set_batch
    sae.clear_steering.side_effect = _clear
    sae.get_steering_values.side_effect = lambda: dict(applied)
    sae._applied = applied
    return sae


def _service() -> SAEService:
    svc = SAEService.__new__(SAEService)
    svc._sae_state = AttachedSAEState()
    return svc


def _attach(state, sae, sae_id, layer):
    state.set(sae, sae_id, layer, MagicMock())


class TestGroupByLayer:
    def test_each_member_applied_through_its_own_layer_sae(self):
        svc = _service()
        s10, s13 = _sae(), _sae()
        _attach(svc._sae_state, s10, "sae-10", 10)
        _attach(svc._sae_state, s13, "sae-13", 13)
        members = [
            CircuitMember(feature_idx=5, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=9, layer=13, budget=40.0, sign=-1),
        ]
        result = svc.set_circuit_steering(members, intensity=1.0)
        # L10 member on s10 only; L13 member on s13 only.
        assert s10._applied == {5: 50.0}
        assert s13._applied == {9: -40.0}
        assert result.applied_per_layer == {10: {5: 50.0}, 13: {9: -40.0}}

    def test_multiple_members_same_layer(self):
        svc = _service()
        s10 = _sae()
        _attach(svc._sae_state, s10, "sae-10", 10)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=30.0, sign=1),
            CircuitMember(feature_idx=2, layer=10, budget=20.0, sign=1),
        ]
        svc.set_circuit_steering(members, intensity=1.0)
        assert s10._applied == {1: 30.0, 2: 20.0}


class TestBudgetLambdaClamp:
    def test_lambda_scales_all_layers(self):
        svc = _service()
        s10, s13 = _sae(), _sae()
        _attach(svc._sae_state, s10, "sae-10", 10)
        _attach(svc._sae_state, s13, "sae-13", 13)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=30.0, sign=1),
        ]
        svc.set_circuit_steering(members, intensity=2.0)
        assert s10._applied == {1: 100.0}  # 50*1*2
        assert s13._applied == {2: 60.0}   # 30*1*2

    def test_effective_value_clamped_to_range_with_warning(self):
        svc = _service()
        s10 = _sae()
        _attach(svc._sae_state, s10, "sae-10", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=200.0, sign=1)]
        result = svc.set_circuit_steering(members, intensity=2.0)  # 400 → clamp 200
        assert s10._applied == {1: 200.0}
        assert len(result.clamp_warnings) == 1
        assert "clamped" in result.clamp_warnings[0]

    def test_gamma_zero_budget_is_bdir(self):
        """No re-derivation: budget is applied directly (× sign × λ)."""
        svc = _service()
        s10 = _sae()
        _attach(svc._sae_state, s10, "sae-10", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=17.5, sign=1)]
        svc.set_circuit_steering(members, intensity=1.0)
        assert s10._applied == {1: 17.5}

    def test_intensity_zero_disables(self):
        svc = _service()
        s10 = _sae()
        _attach(svc._sae_state, s10, "sae-10", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1)]
        result = svc.set_circuit_steering(members, intensity=0.0)
        # λ=0 clears the layer and leaves steering disabled (R2 fix) — no
        # zero-valued "active" features left behind.
        assert s10._applied == {}
        s10.enable_steering.assert_called_with(False)
        assert result.hazards == []  # λ=0 → no hazards


class TestSAESetIncomplete:
    def test_missing_layer_raises_with_offenders(self):
        svc = _service()
        _attach(svc._sae_state, _sae(), "sae-10", 10)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=40.0, sign=1, sae_id="sae-13"),
        ]
        with pytest.raises(SAESetIncompleteError) as ei:
            svc.set_circuit_steering(members, intensity=1.0)
        offenders = ei.value.offenders
        assert len(offenders) == 1
        assert offenders[0]["layer"] == 13 and offenders[0]["feature_idx"] == 2

    def test_nothing_applied_when_incomplete(self):
        svc = _service()
        s10 = _sae()
        _attach(svc._sae_state, s10, "sae-10", 10)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=40.0, sign=1),
        ]
        with pytest.raises(SAESetIncompleteError):
            svc.set_circuit_steering(members, intensity=1.0)
        assert s10._applied == {}  # fail-closed: NOTHING applied

    def test_ambiguous_layer_is_incomplete(self):
        """Two SAEs on the same layer → by_layer None → offender."""
        svc = _service()
        _attach(svc._sae_state, _sae(), "sae-a", 10)
        _attach(svc._sae_state, _sae(), "sae-b", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1)]
        with pytest.raises(SAESetIncompleteError):
            svc.set_circuit_steering(members, intensity=1.0)

    def test_out_of_bounds_index_is_incomplete_not_500(self):
        svc = _service()
        _attach(svc._sae_state, _sae(d_sae=100), "sae-10", 10)
        members = [CircuitMember(feature_idx=500, layer=10, budget=50.0, sign=1)]
        with pytest.raises(SAESetIncompleteError) as ei:
            svc.set_circuit_steering(members, intensity=1.0)
        assert ei.value.offenders[0]["reason"] == "index_out_of_bounds"


class TestHazards:
    def _two_layer(self, svc):
        _attach(svc._sae_state, _sae(), "sae-10", 10)
        _attach(svc._sae_state, _sae(), "sae-13", 13)

    def test_same_sign_is_compounding(self):
        svc = _service()
        self._two_layer(svc)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=50.0, sign=1),
        ]
        result = svc.set_circuit_steering(members, intensity=1.0)
        assert len(result.hazards) == 1
        assert result.hazards[0]["kind"] == "compounding"
        assert result.hazards[0]["label"] == "heuristic:co-steer-sign"

    def test_opposite_sign_is_cancellation(self):
        svc = _service()
        self._two_layer(svc)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=50.0, sign=-1),
        ]
        result = svc.set_circuit_steering(members, intensity=1.0)
        assert result.hazards[0]["kind"] == "cancellation"

    def test_validated_edge_quantifies_and_flips_on_negative_es(self):
        svc = _service()
        self._two_layer(svc)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=50.0, sign=1),  # same sign
        ]
        # A validated NEGATIVE-ES edge flips same-sign compounding → cancellation.
        edges = [{
            "up": {"layer": 10, "feature_idx": 1},
            "down": {"layer": 13, "feature_idx": 2},
            "effect_size": -0.42, "rung": 2,
        }]
        result = svc.set_circuit_steering(members, intensity=1.0, edges=edges)
        h = result.hazards[0]
        assert h["label"] == "validated:ES=-0.42"
        assert h["rung"] == 2
        assert h["kind"] == "cancellation"  # flipped by negative ES

    def test_config_not_mutated_by_hazards(self):
        """Hazard detection must not change what was applied."""
        svc = _service()
        s10 = _sae()
        s13 = _sae()
        _attach(svc._sae_state, s10, "sae-10", 10)
        _attach(svc._sae_state, s13, "sae-13", 13)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=50.0, sign=1),
        ]
        svc.set_circuit_steering(members, intensity=1.0)
        # Applied values are exactly the clamped budgets — hazards changed nothing.
        assert s10._applied == {1: 50.0}
        assert s13._applied == {2: 50.0}


class TestR1Fixes:
    def test_intensity_clamped_to_envelope(self):
        """λ is clamped to [CIRCUIT_INTENSITY_MIN, MAX]=[0,2]; a rogue negative
        never inverts the circuit and a huge value never over-drives."""
        service = _service()
        s10 = _sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1)]
        # Negative λ clamps to 0 → circuit OFF (cleared + disabled, not inverted).
        service.set_circuit_steering(members, intensity=-3.0)
        assert s10._applied == {}
        s10.enable_steering.assert_called_with(False)
        # λ above 2 clamps to 2.
        service.set_circuit_steering(members, intensity=9.0)
        assert s10._applied == {1: 100.0}  # 50*1*2

    def test_duplicate_member_rejected(self):
        service = _service()
        s10 = _sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        members = [
            CircuitMember(feature_idx=5, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=5, layer=10, budget=30.0, sign=-1),  # dup key
        ]
        with pytest.raises(SAESetIncompleteError) as ei:
            service.set_circuit_steering(members, intensity=1.0)
        reasons = {o.get("reason") for o in ei.value.offenders}
        assert "duplicate_member" in reasons
        assert s10._applied == {}  # nothing applied

    def test_stale_steering_cleared_before_new_serve(self):
        """A prior serve's features must not leak into a new serve on the same
        layer (R1: set_circuit_steering clears the layer first)."""
        service = _service()
        s10 = _sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        # First serve: features 1 and 2.
        service.set_circuit_steering(
            [
                CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1),
                CircuitMember(feature_idx=2, layer=10, budget=40.0, sign=1),
            ],
            intensity=1.0,
        )
        assert set(s10.get_steering_values()) == {1, 2}
        # Second serve touches only feature 5 — feature 1,2 must be gone.
        service.set_circuit_steering(
            [CircuitMember(feature_idx=5, layer=10, budget=30.0, sign=1)],
            intensity=1.0,
        )
        assert s10.get_steering_values() == {5: 30.0}


class TestR2Fixes:
    def test_negative_budget_not_double_negated(self):
        """Canonical sign rule: a negative budget is already directional; the
        sign field must NOT be multiplied in (would flip suppression→amplify)."""
        service = _service()
        s10 = _sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        # budget=-100, sign=-1 → served as -100 (suppression), NOT +100.
        members = [CircuitMember(feature_idx=1, layer=10, budget=-100.0, sign=-1)]
        service.set_circuit_steering(members, intensity=1.0)
        assert s10._applied == {1: -100.0}

    def test_positive_budget_takes_direction_from_sign(self):
        service = _service()
        s10 = _sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=80.0, sign=-1)]
        service.set_circuit_steering(members, intensity=1.0)
        assert s10._applied == {1: -80.0}

    def test_intensity_zero_disables_steering(self):
        """λ=0 clears and leaves steering DISABLED (not 'enabled' with zeros)."""
        service = _service()
        s10 = _sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1)]
        service.set_circuit_steering(members, intensity=0.0)
        s10.enable_steering.assert_called_with(False)

    def test_member_sae_id_disambiguates_same_layer(self):
        """Two SAEs on one layer: a member naming its sae_id serves through the
        exact one (get(sae_id,layer)), not the ambiguous by_layer path."""
        service = _service()
        sa, sb = _sae(), _sae()
        _attach(service._sae_state, sa, "sae-a", 10)
        _attach(service._sae_state, sb, "sae-b", 10)
        members = [CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1, sae_id="sae-b")]
        service.set_circuit_steering(members, intensity=1.0)
        assert sb._applied == {1: 50.0}
        assert sa._applied == {}  # sae-a untouched


class TestR3Fixes:
    def test_two_saes_one_layer_each_member_to_its_own_sae(self):
        """REGRESSION (R3): the resolution cache was keyed by LAYER, so the
        first member's SAE captured every later member on that layer — a
        wrong-basis serve. Cache is now keyed by (sae_id, layer)."""
        service = _service()
        sa, sb = _sae(), _sae()
        _attach(service._sae_state, sa, "sae-a", 10)
        _attach(service._sae_state, sb, "sae-b", 10)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1, sae_id="sae-a"),
            CircuitMember(feature_idx=2, layer=10, budget=60.0, sign=1, sae_id="sae-b"),
        ]
        service.set_circuit_steering(members, intensity=1.0)
        assert sa._applied == {1: 50.0}
        assert sb._applied == {2: 60.0}

    def test_empty_members_clears_previous_circuit(self):
        """REGRESSION (R3): an empty member list was a silent no-op that left
        the PREVIOUS circuit armed. It must now clear + disable."""
        service = _service()
        s10 = _sae()
        _attach(service._sae_state, s10, "sae-10", 10)
        service.set_circuit_steering(
            [CircuitMember(feature_idx=9, layer=10, budget=99.0, sign=1)], intensity=1.0
        )
        assert s10._applied == {9: 99.0}
        result = service.set_circuit_steering([], intensity=1.0)
        assert s10._applied == {}
        assert result.applied_per_layer == {}
        s10.enable_steering.assert_called_with(False)

    def test_unattached_declared_sae_id_records_substitution(self):
        """A member naming an SAE that is NOT attached falls back to the
        layer's unique SAE but the basis substitution is surfaced, not silent."""
        service = _service()
        s10 = _sae()
        _attach(service._sae_state, s10, "sae-real", 10)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1,
                          sae_id="sae-authored-elsewhere")
        ]
        result = service.set_circuit_steering(members, intensity=1.0)
        assert s10._applied == {1: 50.0}
        assert any("not attached" in w for w in result.clamp_warnings)
        assert any("different feature basis" in w for w in result.clamp_warnings)


class TestClearCircuitSteering:
    def test_clear_all_participating_layers(self):
        svc = _service()
        s10, s13 = _sae(), _sae()
        _attach(svc._sae_state, s10, "sae-10", 10)
        _attach(svc._sae_state, s13, "sae-13", 13)
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=50.0, sign=1),
            CircuitMember(feature_idx=2, layer=13, budget=50.0, sign=1),
        ]
        svc.set_circuit_steering(members, intensity=1.0)
        cleared = svc.clear_circuit_steering()
        assert sorted(cleared) == [10, 13]
        s10.clear_steering.assert_called()
        s13.clear_steering.assert_called()
