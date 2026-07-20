"""Per-request circuit dial (Feature 14, task 1.4).

One global λ scales EVERY layer of an active circuit together, each through its
own SAE; the override is saved and restored PER LAYER so it never leaks into
global state; a cluster-active or no-circuit deployment falls through to the
Feature 10 path unchanged.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.services.inference_service import InferenceService, reset_steering_memo
from millm.services.sae_service import AttachedSAEState


@pytest.fixture(autouse=True)
def clean_registry():
    state = AttachedSAEState()
    state._entries.clear()
    reset_steering_memo()
    yield
    state._entries.clear()
    reset_steering_memo()


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


def make_meta(members=((10, 1, 40.0), (13, 2, 30.0)), intensity=1.0):
    """A REAL circuit-definition/v1 document — the dial re-derives the authored
    basis from it, so an empty stub would not exercise the real path."""
    layers = sorted({layer for layer, _, _ in members})
    return {
        "kind": "mistudio.circuit-definition",
        "schema_version": "1",
        "name": "fear→threat",
        "saes": [
            {"layer": layer, "n_features": 8192, "mistudio_sae_id": f"sae-{layer}"}
            for layer in layers
        ],
        "members": [
            {"layer": layer, "feature": {"feature_idx": idx, "strength": strength}}
            for layer, idx, strength in members
        ],
        "edges": [],
        "budget": {"layers": {}, "intensity": intensity, "intensity_range": [0.0, 2.0]},
    }


def make_circuit(**overrides):
    meta = overrides.pop("circuit_meta", None)
    base = dict(
        id="circ_1",
        name="fear→threat",
        layers=[10, 13],
        serving_mode="full",
        intensity=1.0,
        rung=2,
    )
    base.update(overrides)
    base["circuit_meta"] = meta if meta is not None else make_meta()
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
        """A circuit serving at λ=1.5 dialled to 2.0 must end at 2.0× the
        AUTHORED basis, not 3.0×. R1: rescaling the LIVE values could not
        recover the basis (clamping is lossy and the stored-λ column differs
        from the document's), so the dial re-derives from the definition."""
        s10 = make_sae({1: 60.0})  # authored 40 × document λ 1.5
        attach(s10, "sae-10", 10)
        svc = service_with_circuit(
            make_circuit(layers=[10], intensity=1.5,
                         circuit_meta=make_meta(members=((10, 1, 40.0),), intensity=1.5))
        )
        await svc._apply_request_circuit_steering(2.0)
        assert s10._applied == {1: 80.0}   # 40 × 2, NOT 60 × 2

    async def test_clamped_members_recover_their_authored_basis(self):
        """R1: authored 150 at λ=2 stores clamp(300)=200; dialling back to 1.0
        must give 150, which rescaling the clamped live value cannot do."""
        s10 = make_sae({1: 200.0})
        attach(s10, "sae-10", 10)
        svc = service_with_circuit(
            make_circuit(layers=[10], intensity=2.0,
                         circuit_meta=make_meta(members=((10, 1, 150.0),), intensity=2.0))
        )
        await svc._apply_request_circuit_steering(1.0)
        assert s10._applied == {1: 150.0}   # NOT 200/2*1 = 100

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
        svc = service_with_circuit(
            make_circuit(layers=[10],
                         circuit_meta=make_meta(members=((10, 1, 150.0),)))
        )
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
    def _steering(self, **kw):
        """A circuit that is genuinely steering — the rung echo now requires
        it, so the fixture must attach the SAEs the definition names."""
        attach(make_sae({1: 40.0}), "sae-10", 10)
        attach(make_sae({2: 30.0}), "sae-13", 13)
        return service_with_circuit(make_circuit(**kw))

    async def test_rung_language_comes_from_the_ladder(self):
        svc = self._steering(rung=2)
        assert await svc.active_circuit_rung() == (2, "causally validated (edge)")

    async def test_rung_below_two_is_never_described_as_causal(self):
        for rung, expected in ((0, "associated"), (1, "suggested (attribution-supported)")):
            AttachedSAEState()._entries.clear()
            reset_steering_memo()   # each iteration is a fresh "request"
            svc = self._steering(rung=rung)
            got = await svc.active_circuit_rung()
            assert got == (rung, expected)
            assert "causal" not in got[1].lower()

    async def test_no_circuit_no_header(self):
        svc = service_with_circuit(None)
        assert await svc.active_circuit_rung() is None

    async def test_rung_header_suppressed_when_nothing_is_actually_steering(self):
        """R2: R1 fixed the lambda echo's no-op rules and left the RUNG echo
        re-deriving its own, so a response could advertise
        'X-miLLM-Circuit-Rung: 2' while the dial no-opped. Both surfaces now
        ask the one _steering_circuit predicate."""
        svc = service_with_circuit(make_circuit(rung=2))  # nothing attached
        assert await svc.active_circuit_rung() is None

    async def test_rung_header_suppressed_for_an_unparseable_definition(self):
        attach(make_sae({1: 40.0}), "sae-10", 10)
        svc = service_with_circuit(make_circuit(circuit_meta={"garbage": True}))
        assert await svc.active_circuit_rung() is None


class TestSnapshotCoversEveryDialledLayer:
    async def test_a_layer_absent_from_the_db_column_is_still_restored(self):
        """R2 CRITICAL: the snapshot filtered on circuit.layers (the DB column)
        while the apply drove off the definition's member layers. Any layer in
        one and not the other was dialled and never restored — a per-request
        override leaking PERMANENTLY into global state."""
        s10, s13 = make_sae({1: 40.0}), make_sae({2: 30.0})
        attach(s10, "sae-10", 10)
        attach(s13, "sae-13", 13)
        # The row's column disagrees with the document: only L10 is listed.
        svc = service_with_circuit(make_circuit(layers=[10]))

        saved = await svc._apply_request_circuit_steering(2.0)
        assert s13._applied == {2: 60.0}, "L13 must be dialled (it is a member)"

        svc._restore_request_profile(saved)
        assert s10._applied == {1: 40.0}
        assert s13._applied == {2: 30.0}, "L13 leaked: dialled but not restored"

    async def test_lambda_zero_clears_rather_than_only_disabling(self):
        """R2: set_circuit_steering (the lambda>0 path) clears each SAE first,
        so disabling alone left stale values resident behind a false flag."""
        s10 = make_sae({1: 40.0})
        attach(s10, "sae-10", 10)
        svc = service_with_circuit(make_circuit(layers=[10]))
        await svc._apply_request_circuit_steering(0.0)
        assert s10.is_steering_enabled is False
        assert s10._applied == {}, "values stayed resident behind a disabled flag"


class TestDialInputValidation:
    def test_a_bool_is_not_a_dial_value(self):
        """R2: bool is an int subclass, so {"steering_intensity": true}
        silently dialled lambda=1.0 instead of being rejected."""
        svc = InferenceService.__new__(InferenceService)
        c = make_circuit()
        assert svc._resolve_circuit_intensity(True, c) is None
        assert svc._resolve_circuit_intensity(False, c) is None

    def test_numeric_respects_the_authored_floor(self):
        """R2: the numeric path capped at `hi` but ignored `lo`, so it could
        sit below a floor that "min" itself refuses to go below."""
        svc = InferenceService.__new__(InferenceService)
        c = make_circuit(circuit_meta={"budget": {"intensity_range": [0.5, 1.5]}})
        assert svc._resolve_circuit_intensity(0.1, c) == 0.5
        assert svc._resolve_circuit_intensity(0.0, c) == 0.0   # off still allowed


class TestSteeringMemoIsRequestScoped:
    """R3: R2 memoised the steering verdict on the InferenceService "which is
    request-scoped". It is not — get_inference_service() is @lru_cache'd and
    documents itself as a singleton, so the memo lived for the whole PROCESS.
    All four R3 reviewer perspectives independently found this."""

    async def test_a_reused_service_does_not_serve_a_stale_verdict(self):
        """The failure R2's memo caused: one service instance across two
        requests, the circuit deactivated in between, rung header forever."""
        attach(make_sae({1: 40.0}), "sae-10", 10)
        attach(make_sae({2: 30.0}), "sae-13", 13)
        svc = service_with_circuit(make_circuit(rung=2))

        reset_steering_memo()
        assert await svc.active_circuit_rung() == (2, "causally validated (edge)")

        # Operator deactivates. The SAME service object handles the next request.
        svc._active_full_circuit = AsyncMock(return_value=None)
        reset_steering_memo()
        assert await svc.active_circuit_rung() is None, (
            "stale memo advertised a deactivated circuit as causally validated"
        )

    async def test_a_reused_service_picks_up_a_newly_active_circuit(self):
        """The inverse, equally bad: a None cached before activation would
        suppress the rung disclosure while steering was live."""
        svc = service_with_circuit(None)
        reset_steering_memo()
        assert await svc.active_circuit_rung() is None

        attach(make_sae({1: 40.0}), "sae-10", 10)
        attach(make_sae({2: 30.0}), "sae-13", 13)
        circuit = make_circuit(rung=2)
        svc._active_full_circuit = AsyncMock(return_value=circuit)
        reset_steering_memo()
        assert await svc.active_circuit_rung() == (2, "causally validated (edge)")

    async def test_the_memo_still_collapses_repeat_lookups_within_one_request(self):
        """The perf win R2 was after must survive the correctness fix."""
        attach(make_sae({1: 40.0}), "sae-10", 10)
        attach(make_sae({2: 30.0}), "sae-13", 13)
        svc = service_with_circuit(make_circuit(rung=2))
        reset_steering_memo()

        await svc.active_circuit_rung()
        await svc.active_circuit_rung()
        await svc._resolve_active_circuit_intensity(1.0)
        assert svc._active_full_circuit.await_count == 1
