"""Per-request circuit dial (Feature 14, task 1.4).

One global λ scales EVERY layer of an active circuit together, each through its
own SAE; the override is saved and restored PER LAYER so it never leaks into
global state; a cluster-active or no-circuit deployment falls through to the
Feature 10 path unchanged.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.services.inference_service import InferenceService
from millm.services.sae_service import AttachedSAEState


@pytest.fixture(autouse=True)
def clean_registry():
    state = AttachedSAEState()
    state._entries.clear()
    yield
    state._entries.clear()


def make_sae(values: dict[int, float] | None = None, enabled: bool = True):
    sae = MagicMock()
    applied = dict(values or {})
    sae.d_sae = 8192
    sae.is_steering_enabled = enabled

    def _set_batch(s):
        applied.update(s)

    def _clear(feature_idx=None):
        applied.clear() if feature_idx is None else applied.pop(feature_idx, None)

    sae.set_steering_batch.side_effect = _set_batch
    sae.clear_steering.side_effect = _clear
    sae.get_steering_values.side_effect = lambda: dict(applied)
    sae.enable_steering.side_effect = lambda v: setattr(sae, "is_steering_enabled", v)
    sae._applied = applied
    return sae


def attach(sae, sae_id: str, layer: int):
    AttachedSAEState().set(sae, sae_id, layer, None)


def make_circuit(**overrides):
    base = dict(
        id="circ_1",
        name="fear→threat",
        layers=[10, 13],
        serving_mode="full",
        intensity=1.0,
        rung=2,
        circuit_meta={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def service_with_circuit(circuit):
    svc = InferenceService.__new__(InferenceService)
    svc._active_full_circuit = AsyncMock(
        return_value=circuit if (circuit and circuit.serving_mode == "full") else None
    )
    return svc


class TestAllLayersScaleTogether:
    async def test_one_dial_scales_every_layer(self):
        s10 = make_sae({1: 40.0})
        s13 = make_sae({2: 30.0})
        attach(s10, "sae-10", 10)
        attach(s13, "sae-13", 13)
        svc = service_with_circuit(make_circuit())

        saved = await svc._apply_request_circuit_steering(2.0)

        assert saved is not None and saved["circuit"] is True
        assert s10._applied == {1: 80.0}   # 40 × 2
        assert s13._applied == {2: 60.0}   # 30 × 2

    async def test_dial_is_absolute_not_a_multiplier_of_the_stored_dial(self):
        """A circuit already serving at λ=1.5 dialled to 2.0 must end at 2.0×
        the AUTHORED basis, not 3.0× (the stored λ is divided out first)."""
        s10 = make_sae({1: 60.0})  # authored 40 × stored λ 1.5
        attach(s10, "sae-10", 10)
        svc = service_with_circuit(make_circuit(layers=[10], intensity=1.5))

        await svc._apply_request_circuit_steering(2.0)
        assert s10._applied == {1: 80.0}   # 40 × 2, NOT 60 × 2

    async def test_lambda_zero_disables_every_layer(self):
        s10, s13 = make_sae({1: 40.0}), make_sae({2: 30.0})
        attach(s10, "sae-10", 10)
        attach(s13, "sae-13", 13)
        svc = service_with_circuit(make_circuit())

        saved = await svc._apply_request_circuit_steering(0.0)
        assert saved is not None
        assert s10.is_steering_enabled is False
        assert s13.is_steering_enabled is False

    async def test_values_are_clamped_per_member(self):
        s10 = make_sae({1: 150.0})
        attach(s10, "sae-10", 10)
        svc = service_with_circuit(make_circuit(layers=[10]))
        await svc._apply_request_circuit_steering(2.0)   # 300 → clamp 200
        assert s10._applied == {1: 200.0}


class TestFallthrough:
    async def test_no_active_circuit_falls_through(self):
        attach(make_sae({1: 1.0}), "sae-10", 10)
        svc = service_with_circuit(None)
        assert await svc._apply_request_circuit_steering(1.5) is None

    async def test_slice_fallback_circuit_is_not_dialled_here(self):
        """A slice is steered by a cluster PROFILE — the ordinary profile path
        owns it; dialling here would double-apply."""
        attach(make_sae({1: 1.0}), "sae-10", 10)
        svc = service_with_circuit(make_circuit(serving_mode="slice_fallback"))
        assert await svc._apply_request_circuit_steering(1.5) is None

    async def test_no_attached_layers_is_a_noop(self):
        svc = service_with_circuit(make_circuit())  # nothing attached
        assert await svc._apply_request_circuit_steering(1.5) is None

    async def test_no_live_values_is_a_noop(self):
        attach(make_sae({}), "sae-10", 10)
        svc = service_with_circuit(make_circuit(layers=[10]))
        assert await svc._apply_request_circuit_steering(1.5) is None

    async def test_absent_dial_is_a_noop(self):
        attach(make_sae({1: 1.0}), "sae-10", 10)
        svc = service_with_circuit(make_circuit(layers=[10]))
        assert await svc._apply_request_circuit_steering(None) is None


class TestPerLayerRestore:
    async def test_restore_returns_every_layer_to_its_prior_state(self):
        """Restoring only the first layer would leave the others dialled for
        every subsequent request — a per-request override leaking globally."""
        s10 = make_sae({1: 40.0})
        s13 = make_sae({2: 30.0})
        attach(s10, "sae-10", 10)
        attach(s13, "sae-13", 13)
        svc = service_with_circuit(make_circuit())

        saved = await svc._apply_request_circuit_steering(2.0)
        assert s10._applied == {1: 80.0} and s13._applied == {2: 60.0}

        svc._restore_request_profile(saved)
        assert s10._applied == {1: 40.0}
        assert s13._applied == {2: 30.0}

    async def test_restore_reinstates_disabled_state(self):
        s10 = make_sae({1: 40.0})
        attach(s10, "sae-10", 10)
        svc = service_with_circuit(make_circuit(layers=[10]))
        saved = await svc._apply_request_circuit_steering(0.0)
        assert s10.is_steering_enabled is False
        svc._restore_request_profile(saved)
        assert s10.is_steering_enabled is True

    async def test_restore_tolerates_a_layer_detached_mid_request(self):
        s10, s13 = make_sae({1: 40.0}), make_sae({2: 30.0})
        attach(s10, "sae-10", 10)
        attach(s13, "sae-13", 13)
        svc = service_with_circuit(make_circuit())
        saved = await svc._apply_request_circuit_steering(2.0)

        AttachedSAEState().clear(sae_id="sae-13", layer=13)
        svc._restore_request_profile(saved)  # must not raise
        assert s10._applied == {1: 40.0}


class TestIntensityResolution:
    def test_symbolic_resolves_against_the_config_envelope(self):
        svc = InferenceService.__new__(InferenceService)
        c = make_circuit()
        assert svc._resolve_circuit_intensity("off", c) == 0.0
        assert svc._resolve_circuit_intensity("min", c) == 0.0
        assert svc._resolve_circuit_intensity("max", c) == 2.0

    def test_symbolic_resolves_against_an_authored_range(self):
        svc = InferenceService.__new__(InferenceService)
        c = make_circuit(circuit_meta={"budget": {"intensity_range": [0.5, 1.5]}})
        assert svc._resolve_circuit_intensity("min", c) == 0.5
        assert svc._resolve_circuit_intensity("max", c) == 1.5

    def test_numeric_is_capped_at_the_ceiling(self):
        """/v1 must never exceed what an authenticated set_intensity accepts."""
        svc = InferenceService.__new__(InferenceService)
        c = make_circuit(circuit_meta={"budget": {"intensity_range": [0.5, 1.5]}})
        assert svc._resolve_circuit_intensity(9.0, c) == 1.5

    def test_dial_to_zero_always_allowed_below_an_authored_floor(self):
        svc = InferenceService.__new__(InferenceService)
        c = make_circuit(circuit_meta={"budget": {"intensity_range": [0.5, 1.5]}})
        assert svc._resolve_circuit_intensity(0.0, c) == 0.0

    def test_authored_range_cannot_smuggle_overdrive(self):
        """An authored [0, 9] must intersect with the config envelope."""
        svc = InferenceService.__new__(InferenceService)
        c = make_circuit(circuit_meta={"budget": {"intensity_range": [0.0, 9.0]}})
        assert svc._resolve_circuit_intensity("max", c) == 2.0

    def test_malformed_range_degrades_to_the_envelope(self):
        svc = InferenceService.__new__(InferenceService)
        c = make_circuit(circuit_meta={"budget": {"intensity_range": ["x", None]}})
        assert svc._resolve_circuit_intensity("max", c) == 2.0

    def test_inverted_range_is_normalised(self):
        svc = InferenceService.__new__(InferenceService)
        c = make_circuit(circuit_meta={"budget": {"intensity_range": [1.5, 0.5]}})
        assert svc._resolve_circuit_intensity("min", c) == 0.5
        assert svc._resolve_circuit_intensity("max", c) == 1.5


class TestRungEcho:
    async def test_rung_language_comes_from_the_ladder(self):
        svc = service_with_circuit(make_circuit(rung=2))
        assert await svc.active_circuit_rung() == (2, "causally validated (edge)")

    async def test_rung_below_two_is_never_described_as_causal(self):
        for rung, expected in ((0, "associated"), (1, "suggested (attribution-supported)")):
            svc = service_with_circuit(make_circuit(rung=rung))
            got = await svc.active_circuit_rung()
            assert got == (rung, expected)
            assert "causal" not in got[1].lower()

    async def test_no_circuit_no_header(self):
        svc = service_with_circuit(None)
        assert await svc.active_circuit_rung() is None
