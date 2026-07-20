"""End-to-end circuit workflow (Feature 13, task 7.0 + F12 acceptance 8.1b).

Drives the REAL path with REAL (CPU) LoadedSAE objects and a fixture generated
by miStudio's OWN contract classes (tests/fixtures/real_circuit_definition.json):

  import → activate → each member applied through ITS OWN layer's SAE at the
  authored strength (F12 §9.1), SAE_SET_INCOMPLETE at activation (F12 §9.3),
  hazards surfaced at activation (F12 §9.4), slice-fallback on an incomplete
  SAE set, the rung<2 acknowledgement gate, and lossless re-export.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from millm.api.schemas.circuit import CircuitDefinitionV1
from millm.api.schemas.cluster import ClusterDefinitionV1
from millm.core.errors import SAESetIncompleteError, UnvalidatedCircuitError
from millm.db.repositories.circuit_repository import CircuitRepository
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.circuit_service import CircuitService
from millm.services.sae_service import AttachedSAEState, SAEService

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "real_circuit_definition.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def make_sae(d_in: int = 32, d_sae: int = 8192) -> LoadedSAE:
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
def clean_registry():
    state = AttachedSAEState()
    state._entries.clear()
    yield
    state._entries.clear()


@pytest.fixture
def sae_service():
    """A REAL SAEService (serving math), with only the repo/hooker stubbed."""
    svc = SAEService.__new__(SAEService)
    svc._sae_state = AttachedSAEState()
    return svc


@pytest.fixture
def cluster_service():
    svc = MagicMock()
    svc.import_definition = AsyncMock(return_value=MagicMock())
    svc.get_active_cluster = AsyncMock(return_value=None)
    svc.deactivate = AsyncMock()
    return svc


@pytest.fixture
def service(test_session, sae_service, cluster_service):
    return CircuitService(
        CircuitRepository(test_session),
        sae_service=sae_service,
        cluster_service=cluster_service,
    )


def attach(sae_id: str, layer: int, sae: LoadedSAE) -> None:
    AttachedSAEState().set(sae, sae_id, layer, None)


class TestFixtureIsReal:
    def test_fixture_was_generated_by_the_producer_contract(self):
        doc = load_fixture()
        assert doc["kind"] == "mistudio.circuit-definition"
        assert doc["schema_version"] == "1"
        # Producer-only fields that a hand-written fixture would miss.
        assert "provenance" in doc and "budget" in doc
        assert doc["budget"]["formula_id"] == "freq-budget/sim-alloc/per-layer@1"

    def test_fixture_validates_against_the_miLLM_mirror(self):
        d = CircuitDefinitionV1.model_validate(load_fixture())
        assert d.layers() == [10, 13]
        assert d.edges[0].rung == 2


class TestFullServing:
    """F12 §9.1 re-verified END-TO-END through circuit activation (8.1b)."""

    async def test_each_member_applied_through_its_own_layer_sae(self, service):
        s10, s13 = make_sae(), make_sae()
        attach("sae_L10", 10, s10)
        attach("sae_L13", 13, s13)

        circuit = await service.import_definition(load_fixture())
        assert circuit.serveable is True
        assert circuit.rung == 2  # causally validated → no ack needed

        result = await service.activate(circuit.id)

        assert result["serving_mode"] == "full"
        # Authored strengths, λ=1.0 from the budget: 42.5 @ L10, 31.0 @ L13.
        assert s10.get_steering_values() == {1234: 42.5}
        assert s13.get_steering_values() == {5678: 31.0}
        # No cross-layer contamination.
        assert 5678 not in s10.get_steering_values()
        assert 1234 not in s13.get_steering_values()
        assert s10.is_steering_enabled and s13.is_steering_enabled

    async def test_hazards_surface_at_activation(self, service):
        """F12 §9.4 — the fixture's rung-2 edge yields a QUANTIFIED hazard."""
        attach("sae_L10", 10, make_sae())
        attach("sae_L13", 13, make_sae())
        circuit = await service.import_definition(load_fixture())
        result = await service.activate(circuit.id)
        hazards = result["hazards"]
        assert hazards, "expected a cross-layer hazard for the 10→13 edge"
        top = hazards[0]
        assert top["label"].startswith("validated:ES=")
        assert top["rung"] == 2
        assert top["quantified_effect"] == pytest.approx(0.47)

    async def test_intensity_scales_every_layer_together(self, service):
        s10, s13 = make_sae(), make_sae()
        attach("sae_L10", 10, s10)
        attach("sae_L13", 13, s13)
        circuit = await service.import_definition(load_fixture())
        await service.activate(circuit.id)
        await service.set_intensity(circuit.id, 2.0)
        assert s10.get_steering_values() == {1234: 85.0}   # 42.5 × 2
        assert s13.get_steering_values() == {5678: 62.0}   # 31.0 × 2


class TestIncompleteSAESet:
    """F12 §9.3 re-verified at ACTIVATION (8.1b)."""

    async def test_no_layer_bound_raises_sae_set_incomplete(self, service):
        circuit = await service.import_definition(load_fixture())
        assert circuit.serveable is False
        with pytest.raises(SAESetIncompleteError) as ei:
            await service.activate(circuit.id)
        assert {o["layer"] for o in ei.value.offenders} == {10, 13}

    async def test_partial_set_degrades_to_slice_fallback(self, service, cluster_service):
        """One layer bound → serve its slice through the Feature 8 cluster path,
        NEVER the other layer through a mismatched SAE."""
        s10 = make_sae()
        attach("sae_L10", 10, s10)
        circuit = await service.import_definition(load_fixture())
        result = await service.activate(circuit.id)

        assert result["serving_mode"] == "slice_fallback"
        assert result["bound_layers"] == [10]
        assert result["slice_layer"] == 10
        cluster_service.import_definition.assert_awaited_once()
        call = cluster_service.import_definition.await_args
        slice_model = call[0][0]
        slice_doc = call.kwargs["raw_payload"]
        # The cluster importer takes a VALIDATED model (R1: passing the bare
        # dict crashed on `.name`, and the AsyncMock hid it).
        assert isinstance(slice_model, ClusterDefinitionV1)
        assert slice_model.kind == "mistudio.cluster-definition"
        assert slice_model.name.endswith("[L10 slice]")
        # The raw payload rides alongside for lossless storage.
        assert slice_doc["kind"] == "mistudio.cluster-definition"
        assert "partial_rendering=true" in slice_doc["provenance"]["source_note"]
        # And the multi-SAE serving path was NOT used.
        assert s10.get_steering_values() == {}

    async def test_wrong_feature_space_blocks_that_layer(self, service):
        attach("sae_L10", 10, make_sae(d_sae=4096))  # fixture declares 8192
        attach("sae_L13", 13, make_sae())
        circuit = await service.import_definition(load_fixture())
        verdicts = {v["layer"]: v["verdict"] for v in circuit.per_sae_warnings}
        assert verdicts[10] == "block"
        assert circuit.serveable is False


class TestEvidenceGate:
    async def test_rung_below_two_refused_then_allowed_with_ack(self, service):
        attach("sae_L10", 10, make_sae())
        attach("sae_L13", 13, make_sae())
        doc = load_fixture()
        doc["edges"][0]["rung"] = 0  # demote to MINED
        doc["name"] = "mined variant"
        circuit = await service.import_definition(doc)
        assert circuit.rung == 0

        with pytest.raises(UnvalidatedCircuitError):
            await service.activate(circuit.id)

        result = await service.activate(circuit.id, acknowledge_unvalidated=True)
        assert result["acknowledged_unvalidated"] is True
        assert result["serving_mode"] == "full"

    async def test_edgeless_circuit_is_mined_and_gated(self, service):
        attach("sae_L10", 10, make_sae())
        attach("sae_L13", 13, make_sae())
        doc = load_fixture()
        doc["edges"] = []
        doc["name"] = "edgeless"
        circuit = await service.import_definition(doc)
        assert circuit.rung == 0
        with pytest.raises(UnvalidatedCircuitError):
            await service.activate(circuit.id)


class TestRoundTrip:
    async def test_export_equals_the_imported_document_byte_for_byte(self, service):
        doc = load_fixture()
        circuit = await service.import_definition(doc)
        assert await service.export_definition(circuit.id) == doc

    async def test_unknown_producer_fields_survive(self, service):
        doc = load_fixture()
        doc["tier_2_5_position_data"] = {"mediating_heads": [3, 7]}
        doc["name"] = "with future fields"
        circuit = await service.import_definition(doc)
        exported = await service.export_definition(circuit.id)
        assert exported["tier_2_5_position_data"] == {"mediating_heads": [3, 7]}


class TestCoTenancy:
    async def test_active_cluster_on_a_target_layer_is_released(
        self, service, cluster_service
    ):
        """F12 R2/R3 inherited: a circuit must not silently clobber a cluster
        steering one of its layers."""
        attach("sae_L10", 10, make_sae())
        attach("sae_L13", 13, make_sae())
        cluster_service.get_active_cluster.return_value = MagicMock(
            id="prof_c1", name="fear cluster", layer=10
        )
        circuit = await service.import_definition(load_fixture())
        result = await service.activate(circuit.id)
        cluster_service.deactivate.assert_awaited_once_with("prof_c1")
        assert any("Deactivated cluster" in w for w in result["warnings"])

    async def test_cluster_on_an_untouched_layer_is_left_alone(
        self, service, cluster_service
    ):
        attach("sae_L10", 10, make_sae())
        attach("sae_L13", 13, make_sae())
        cluster_service.get_active_cluster.return_value = MagicMock(
            id="prof_c2", name="other", layer=99
        )
        circuit = await service.import_definition(load_fixture())
        await service.activate(circuit.id)
        cluster_service.deactivate.assert_not_awaited()
