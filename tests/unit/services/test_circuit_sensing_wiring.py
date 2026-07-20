"""Feature 15 Task 3.2/3.5: inference wiring + routing.

These pin the exclusions rather than the happy path — an observation surface
fails by going quietly dark, so the tests that matter are the ones asserting it
refuses to sense when positions cannot be attributed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import millm.api.dependencies as deps
from millm.services.circuit_sensing_service import CircuitSensingService
from millm.services.inference_service import InferenceService
from millm.services.sae_service import AttachedSAEState


@pytest.fixture(autouse=True)
def clean():
    AttachedSAEState()._entries.clear()
    deps._circuit_sensing_service = None
    yield
    AttachedSAEState()._entries.clear()
    deps._circuit_sensing_service = None


def armed_service(armed=True):
    svc = CircuitSensingService()
    if armed:
        svc._circuit_id = "circ_1"
        svc._armed_layers = [10]
        svc._ring = MagicMock()
    deps._circuit_sensing_service = svc
    return svc


def attach(layer=10):
    sae = MagicMock()
    sae.is_edge_sensing_armed = True
    AttachedSAEState().set(sae, f"sae-{layer}", layer, None)
    return sae


def service(speculative=None):
    svc = InferenceService.__new__(InferenceService)
    svc._speculative_model_id = speculative
    # _use_cbm_for_request reads these before reaching the sensing gate.
    backend = MagicMock()
    backend.is_running = True
    svc._cbm_backend = backend
    svc._cbm_force_serial_monitoring = False
    svc._is_monitoring_enabled = lambda: False
    return svc


class TestBeginExclusions:
    def test_no_service_means_no_sensing(self):
        attach()
        assert service()._circuit_sensing_begin("req-1") is None

    def test_an_unarmed_service_means_no_sensing(self):
        armed_service(armed=False)
        attach()
        assert service()._circuit_sensing_begin("req-1") is None

    def test_speculative_decoding_is_excluded(self):
        """Verification passes advance the offset by a whole candidate block
        and rejected tokens re-run, so the ABSOLUTE positions the ring matches
        on diverge. Going unsensed beats mis-attributing."""
        armed_service()
        attach()
        assert service(speculative="draft-model")._circuit_sensing_begin("r") is None

    def test_no_attached_sae_means_no_sensing(self):
        armed_service()
        assert service()._circuit_sensing_begin("req-1") is None

    def test_a_begin_that_arms_nothing_returns_none(self):
        svc = armed_service()
        sae = attach()
        sae.is_edge_sensing_armed = False  # armed layer, unarmed SAE
        assert service()._circuit_sensing_begin("req-1") is None


class TestLayerResolution:
    def test_an_ambiguous_layer_is_excluded(self):
        """by_layer returns None when a layer has zero or >1 SAEs, so a caller
        can never silently pick the wrong basis."""
        AttachedSAEState().set(MagicMock(), "sae-a", 10, None)
        AttachedSAEState().set(MagicMock(), "sae-b", 10, None)
        assert service()._circuit_sensing_layer_saes() == {}

    def test_unambiguous_layers_resolve(self):
        attach(10)
        attach(13)
        assert sorted(service()._circuit_sensing_layer_saes()) == [10, 13]


class TestRoutingForcesSerial:
    def test_an_armed_circuit_forces_serial_routing(self):
        """CBM batch rows cannot be attributed to requests."""
        armed_service()
        attach()
        svc = service()
        assert svc._use_cbm_for_request(has_steering_override=False) is False

    def test_the_gate_asks_the_service_not_the_first_sae(self):
        """AttachedSAEState.attached_sae is only the FIRST entry, so a circuit
        armed on layers 10+13 would go undetected if 10 were absent. The gate
        must consult the service."""
        svc = armed_service()
        svc._armed_layers = [13]
        attach(13)
        infer = service()
        assert infer._use_cbm_for_request(has_steering_override=False) is False


class TestFlushIsBestEffort:
    async def test_no_layers_flushes_nothing(self):
        assert await service()._notify_circuit_sensing(None, None) is None

    async def test_a_failing_collect_never_raises(self):
        """A flush that raised would fail a chat request that already
        succeeded."""
        svc = armed_service()
        svc.collect_edges = MagicMock(side_effect=RuntimeError("boom"))
        await service()._notify_circuit_sensing({10: MagicMock()}, None)

    async def test_an_empty_result_skips_the_write(self):
        svc = armed_service()
        svc.collect_edges = MagicMock(return_value=("req-1", [], False))
        svc.record = MagicMock(side_effect=AssertionError("must not record"))
        await service()._notify_circuit_sensing({10: MagicMock()}, None)
