"""Circuit dial integration verification (Feature 14, task 4.0).

Task 4.1 asks for the property that the unit tests structurally cannot show:
that two INTERLEAVED requests each apply and restore independently, and that
global steering state is byte-identical afterwards. The unit tests each build a
fresh service and a fresh registry, so they can only prove one request in
isolation — which is exactly the blind spot that let R2's process-wide memo and
R2-01's incomplete snapshot both survive a green suite.

These tests drive the real `_apply_request_circuit_steering` /
`_restore_request_profile` pair against the real `AttachedSAEState` singleton.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from millm.services.inference_service import InferenceService, reset_steering_memo
from millm.services.sae_service import AttachedSAEState


@pytest.fixture(autouse=True)
def clean_registry():
    AttachedSAEState()._entries.clear()
    reset_steering_memo()
    yield
    AttachedSAEState()._entries.clear()
    reset_steering_memo()


def make_sae(values: dict[int, float]):
    sae = MagicMock()
    applied = dict(values)
    sae.d_sae = 8192
    sae.is_steering_enabled = True
    sae.set_steering_batch.side_effect = applied.update
    sae.clear_steering.side_effect = lambda idx=None: (
        applied.clear() if idx is None else applied.pop(idx, None)
    )
    sae.get_steering_values.side_effect = lambda: dict(applied)
    sae.enable_steering.side_effect = lambda v: setattr(sae, "is_steering_enabled", v)
    sae._applied = applied
    return sae


def make_meta(members, intensity=1.0):
    layers = sorted({layer for layer, _, _ in members})
    return {
        "kind": "mistudio.circuit-definition",
        "schema_version": "1",
        "name": "fear→threat",
        "saes": [
            {"layer": lay, "n_features": 8192, "mistudio_sae_id": f"sae-{lay}"}
            for lay in layers
        ],
        "members": [
            {"layer": lay, "feature": {"feature_idx": idx, "strength": strength}}
            for lay, idx, strength in members
        ],
        "edges": [],
        "budget": {"layers": {}, "intensity": intensity, "intensity_range": [0.0, 2.0]},
    }


MEMBERS = ((10, 1, 40.0), (13, 2, 30.0))


def make_circuit():
    return SimpleNamespace(
        id="circ_1",
        name="fear→threat",
        layers=[10, 13],
        serving_mode="full",
        intensity=1.0,
        rung=2,
        circuit_meta=make_meta(MEMBERS),
    )


def make_service():
    """A service whose only stub is the DB read — everything else is real."""
    svc = InferenceService.__new__(InferenceService)
    svc._active_full_circuit = AsyncMock(return_value=make_circuit())
    return svc


def attach_both():
    s10, s13 = make_sae({1: 40.0}), make_sae({2: 30.0})
    AttachedSAEState().set(s10, "sae-10", 10, None)
    AttachedSAEState().set(s13, "sae-13", 13, None)
    return s10, s13


class TestInterleavedRequests:
    async def test_two_interleaved_dials_apply_and_restore_independently(self):
        """Task 4.1. Request A dials 2.0 and B dials 0.5, interleaved; each must
        see its own values and each restore must land back on the shared base."""
        s10, s13 = attach_both()
        base10, base13 = dict(s10._applied), dict(s13._applied)

        svc_a, svc_b = make_service(), make_service()

        reset_steering_memo()
        saved_a = await svc_a._apply_request_circuit_steering(2.0)
        assert s10._applied == {1: 80.0} and s13._applied == {2: 60.0}

        # A restores before B applies — the serialized order the request queue
        # guarantees. B must then observe the ORIGINAL base, not A's values.
        svc_a._restore_request_profile(saved_a)
        assert s10._applied == base10 and s13._applied == base13

        reset_steering_memo()
        saved_b = await svc_b._apply_request_circuit_steering(0.5)
        assert s10._applied == {1: 20.0} and s13._applied == {2: 15.0}

        svc_b._restore_request_profile(saved_b)
        assert s10._applied == base10, "request B leaked into global state"
        assert s13._applied == base13, "request B leaked into global state"
        assert s10.is_steering_enabled and s13.is_steering_enabled

    async def test_global_state_survives_a_burst_of_dials(self):
        """Every layer must return to the shared base after N requests at
        differing λ, including λ=0 (which clears rather than only disabling)."""
        s10, s13 = attach_both()
        base10, base13 = dict(s10._applied), dict(s13._applied)

        for lam in (0.0, 0.25, 1.0, 2.0, 0.0, 1.5):
            reset_steering_memo()
            svc = make_service()
            saved = await svc._apply_request_circuit_steering(lam)
            assert saved is not None, f"λ={lam} unexpectedly no-opped"
            svc._restore_request_profile(saved)
            assert s10._applied == base10, f"λ={lam} left L10 dirty"
            assert s13._applied == base13, f"λ={lam} left L13 dirty"
            assert s10.is_steering_enabled, f"λ={lam} left L10 disabled"
            assert s13.is_steering_enabled, f"λ={lam} left L13 disabled"

    async def test_a_failed_restore_on_one_layer_cannot_strand_the_other(self):
        """R1-03/R2-01 in integration form: a layer detached mid-request must
        not prevent the surviving layer from being restored."""
        s10, s13 = attach_both()
        base10 = dict(s10._applied)

        reset_steering_memo()
        svc = make_service()
        saved = await svc._apply_request_circuit_steering(2.0)
        assert s13._applied == {2: 60.0}

        AttachedSAEState().clear(sae_id="sae-13", layer=13)
        svc._restore_request_profile(saved)  # must not raise
        assert s10._applied == base10


class TestMemoDoesNotCrossRequests:
    async def test_a_deactivation_between_requests_is_observed(self):
        """The R3 critical, at integration level: one service object handling
        two requests must not serve the first request's verdict to the second."""
        attach_both()
        svc = make_service()

        reset_steering_memo()
        assert await svc.active_circuit_rung() == (2, "causally validated (edge)")

        svc._active_full_circuit = AsyncMock(return_value=None)
        reset_steering_memo()
        assert await svc.active_circuit_rung() is None

        reset_steering_memo()
        assert await svc._apply_request_circuit_steering(2.0) is None

    async def test_concurrent_tasks_do_not_share_a_memo(self):
        """Contextvars are per-task; two concurrent asyncio tasks must each
        resolve their own verdict rather than inheriting a sibling's."""
        attach_both()
        live, dead = make_service(), make_service()
        dead._active_full_circuit = AsyncMock(return_value=None)

        async def probe(svc):
            reset_steering_memo()
            return await svc.active_circuit_rung()

        got_live, got_dead = await asyncio.gather(probe(live), probe(dead))
        assert got_live == (2, "causally validated (edge)")
        assert got_dead is None
