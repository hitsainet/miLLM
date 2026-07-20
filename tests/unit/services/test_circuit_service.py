"""CircuitService tests (Feature 13, tasks 4.4/4.5).

Covers the per-referenced-SAE compatibility matrix, serveable logic, the
evidence-rung activation gate, the per-layer slice projection (must be a VALID
cluster-definition/v1 carrying the partial-rendering marker), full-serve
delegation to Feature 12, and lossless export.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from millm.api.schemas.circuit import CircuitDefinitionV1
from millm.core.errors import (
    CircuitNotFoundError,
    SAESetIncompleteError,
    UnvalidatedCircuitError,
    ValidationError,
)
from millm.db.repositories.circuit_repository import CircuitRepository
from millm.services.circuit_service import CircuitService
from millm.services.sae_service import AttachedSAEState


@pytest.fixture(autouse=True)
def clean_registry():
    state = AttachedSAEState()
    state._entries.clear()
    yield
    state._entries.clear()


def attach(sae_id: str, layer: int, d_sae: int = 8192):
    sae = MagicMock()
    sae.d_sae = d_sae
    AttachedSAEState().set(sae, sae_id, layer, None)
    return sae


def make_doc(**overrides) -> dict:
    doc = {
        "kind": "mistudio.circuit-definition",
        "schema_version": "1",
        "name": "fear→threat",
        "saes": [
            {"layer": 10, "n_features": 8192, "mistudio_sae_id": "sae-10"},
            {"layer": 13, "n_features": 8192, "mistudio_sae_id": "sae-13"},
        ],
        "members": [
            {"layer": 10, "feature": {"feature_idx": 1, "strength": 40.0}},
            {"layer": 13, "feature": {"feature_idx": 2, "strength": 30.0}},
        ],
        "edges": [
            {
                "up": {"layer": 10, "feature_idx": 1},
                "down": {"layer": 13, "feature_idx": 2},
                "rung": 2,
                "effect_size": 0.4,
            }
        ],
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def service(test_session):
    repo = CircuitRepository(test_session)
    sae_service = MagicMock()
    sae_service.set_circuit_steering.return_value = MagicMock(
        applied_per_layer={10: {1: 40.0}, 13: {2: 30.0}},
        hazards=[{"kind": "compounding"}],
        clamp_warnings=[],
    )
    cluster_service = MagicMock()
    cluster_service.import_definition = AsyncMock(return_value=MagicMock())
    return CircuitService(repo, sae_service=sae_service, cluster_service=cluster_service)


class TestCompatibilityMatrix:
    def test_all_bind_when_every_layer_attached(self, service):
        attach("sae-10", 10)
        attach("sae-13", 13)
        verdicts = service.assess_compatibility(
            CircuitDefinitionV1.model_validate(make_doc())
        )
        assert {v["verdict"] for v in verdicts} == {"bind"}

    def test_unbound_when_layer_missing(self, service):
        attach("sae-10", 10)
        verdicts = service.assess_compatibility(
            CircuitDefinitionV1.model_validate(make_doc())
        )
        by_layer = {v["layer"]: v for v in verdicts}
        assert by_layer[10]["verdict"] == "bind"
        assert by_layer[13]["verdict"] == "unbound"

    def test_block_on_feature_space_mismatch(self, service):
        attach("sae-10", 10, d_sae=4096)  # declares 8192
        attach("sae-13", 13)
        verdicts = service.assess_compatibility(
            CircuitDefinitionV1.model_validate(make_doc())
        )
        by_layer = {v["layer"]: v for v in verdicts}
        assert by_layer[10]["verdict"] == "block"
        assert "meaningless" in by_layer[10]["reason"]

    def test_warn_when_a_different_sae_id_is_attached(self, service):
        attach("some-other-sae", 10)
        attach("sae-13", 13)
        verdicts = service.assess_compatibility(
            CircuitDefinitionV1.model_validate(make_doc())
        )
        by_layer = {v["layer"]: v for v in verdicts}
        assert by_layer[10]["verdict"] == "warn"
        assert "different feature basis" in by_layer[10]["reason"]

    def test_ambiguous_layer_is_unbound(self, service):
        """Two SAEs on one layer → by_layer None → cannot bind."""
        attach("sae-a", 10)
        attach("sae-b", 10)
        attach("sae-13", 13)
        verdicts = service.assess_compatibility(
            CircuitDefinitionV1.model_validate(make_doc())
        )
        assert {v["layer"]: v["verdict"] for v in verdicts}[10] == "unbound"


class TestImport:
    async def test_import_stores_raw_document(self, service):
        doc = make_doc()
        doc["future_field"] = {"survives": True}
        circuit = await service.import_definition(doc)
        assert circuit.circuit_meta == doc
        assert circuit.circuit_meta["future_field"] == {"survives": True}

    async def test_import_computes_rung_and_layers(self, service):
        circuit = await service.import_definition(make_doc())
        assert circuit.rung == 2  # MIN over a single rung-2 edge
        assert circuit.layers == [10, 13]
        assert circuit.edge_count == 1

    async def test_rung_is_min_over_edges(self, service):
        doc = make_doc(
            edges=[
                {"up": {"layer": 10, "feature_idx": 1}, "down": {"layer": 13, "feature_idx": 2}, "rung": 3},
                {"up": {"layer": 10, "feature_idx": 1}, "down": {"layer": 13, "feature_idx": 3}, "rung": 0},
            ]
        )
        circuit = await service.import_definition(doc)
        assert circuit.rung == 0  # weakest edge governs

    async def test_edgeless_circuit_is_rung_zero(self, service):
        circuit = await service.import_definition(make_doc(edges=[]))
        assert circuit.rung == 0

    async def test_serveable_only_when_all_layers_bind(self, service):
        attach("sae-10", 10)  # L13 not attached
        circuit = await service.import_definition(make_doc())
        assert circuit.serveable is False

        AttachedSAEState()._entries.clear()
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit2 = await service.import_definition(make_doc(name="other"))
        assert circuit2.serveable is True

    async def test_unknown_kind_rejected(self, service):
        with pytest.raises(ValidationError, match="Unknown kind"):
            await service.import_definition(make_doc(kind="mistudio.cluster-definition"))

    async def test_oversize_payload_rejected(self, service):
        with pytest.raises(ValidationError, match="byte cap"):
            await service.import_definition(make_doc(), raw_bytes=2_000_000)

    async def test_name_deduped_on_conflict(self, service):
        await service.import_definition(make_doc())
        second = await service.import_definition(make_doc())
        assert second.name == "fear→threat (2)"

    async def test_name_conflict_can_fail_instead(self, service):
        await service.import_definition(make_doc())
        with pytest.raises(ValidationError, match="already exists"):
            await service.import_definition(make_doc(), on_conflict="fail")


class TestActivationEvidenceGate:
    async def test_rung_below_two_refused_without_ack(self, service):
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc(edges=[]))  # rung 0
        with pytest.raises(UnvalidatedCircuitError) as ei:
            await service.activate(circuit.id)
        assert ei.value.details["rung"] == 0
        assert ei.value.details["rung_language"] == "associated"
        # The refusal message must NOT call a rung-0 circuit causal.
        assert "not causally validated" in ei.value.message

    async def test_rung_below_two_allowed_with_ack(self, service):
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc(edges=[]))
        result = await service.activate(circuit.id, acknowledge_unvalidated=True)
        assert result["serving_mode"] == "full"
        assert result["acknowledged_unvalidated"] is True

    async def test_validated_circuit_needs_no_ack(self, service):
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc())  # rung 2
        result = await service.activate(circuit.id)
        assert result["serving_mode"] == "full"
        assert result["acknowledged_unvalidated"] is False

    async def test_unknown_circuit_404(self, service):
        with pytest.raises(CircuitNotFoundError):
            await service.activate("ghost")


class TestFullServing:
    async def test_delegates_to_feature_12_with_members_and_edges(self, service):
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc())
        result = await service.activate(circuit.id)

        service._sae_service.set_circuit_steering.assert_called_once()
        members, intensity = service._sae_service.set_circuit_steering.call_args[0]
        assert {(m.layer, m.feature_idx) for m in members} == {(10, 1), (13, 2)}
        assert {m.sae_id for m in members} == {"sae-10", "sae-13"}
        assert result["hazards"] == [{"kind": "compounding"}]
        assert result["serving_mode"] == "full"

    async def test_cluster_ref_members_are_expanded(self, service):
        attach("sae-10", 10)
        doc = make_doc(
            saes=[{"layer": 10, "n_features": 8192, "mistudio_sae_id": "sae-10"}],
            members=[
                {
                    "layer": 10,
                    "member_kind": "cluster_ref",
                    "expanded_members": [
                        {"feature_idx": 7, "strength": 10.0},
                        {"feature_idx": 8, "strength": 20.0},
                    ],
                }
            ],
            edges=[],
        )
        circuit = await service.import_definition(doc)
        await service.activate(circuit.id, acknowledge_unvalidated=True)
        members, _ = service._sae_service.set_circuit_steering.call_args[0]
        assert {m.feature_idx for m in members} == {7, 8}


class TestSliceFallback:
    async def test_incomplete_set_serves_a_slice(self, service):
        attach("sae-10", 10)  # L13 missing
        circuit = await service.import_definition(make_doc())
        result = await service.activate(circuit.id)
        assert result["serving_mode"] == "slice_fallback"
        assert result["bound_layers"] == [10]
        assert result["slice_layer"] == 10
        service._cluster_service.import_definition.assert_awaited_once()
        # Feature 12 serving must NOT have been used.
        service._sae_service.set_circuit_steering.assert_not_called()

    async def test_no_layer_bound_raises_sae_set_incomplete(self, service):
        circuit = await service.import_definition(make_doc())  # nothing attached
        with pytest.raises(SAESetIncompleteError) as ei:
            await service.activate(circuit.id)
        assert {o["layer"] for o in ei.value.offenders} == {10, 13}

    def test_slice_is_a_valid_cluster_definition(self, service):
        from millm.api.schemas.cluster import ClusterDefinitionV1

        definition = CircuitDefinitionV1.model_validate(make_doc())
        doc = service.to_layer_slice(definition, 10, circuit_rung_value=2)
        parsed = ClusterDefinitionV1.model_validate(doc)  # must validate as v1
        assert parsed.kind == "mistudio.cluster-definition"
        assert parsed.name == "fear→threat [L10 slice]"
        assert [m.feature_idx for m in parsed.members] == [1]

    def test_slice_carries_partial_rendering_marker(self, service):
        definition = CircuitDefinitionV1.model_validate(make_doc())
        doc = service.to_layer_slice(definition, 13, circuit_rung_value=1)
        note = doc["provenance"]["source_note"]
        assert "partial_rendering=true" in note
        assert "parent_rung=1" in note
        assert "a slice is NOT the circuit" in note
        assert doc["name"].endswith("[L13 slice]")

    def test_slice_with_no_members_on_that_layer_raises(self, service):
        definition = CircuitDefinitionV1.model_validate(make_doc())
        with pytest.raises(ValidationError, match="no serveable members"):
            service.to_layer_slice(definition, 99)

    def test_slice_carries_per_layer_budget_and_global_intensity(self, service):
        doc = make_doc(
            budget={
                "layers": {"10": {"B": 40.0, "formula_id": "freq-budget/sim-alloc@1"}},
                "intensity": 1.5,
                "intensity_range": [0.0, 2.0],
            }
        )
        definition = CircuitDefinitionV1.model_validate(doc)
        sl = service.to_layer_slice(definition, 10)
        assert sl["budget"]["B"] == 40.0
        assert sl["budget"]["intensity"] == 1.5


class TestExportAndLifecycle:
    async def test_export_is_lossless(self, service):
        doc = make_doc()
        doc["weird_producer_field"] = [1, 2, 3]
        circuit = await service.import_definition(doc)
        assert await service.export_definition(circuit.id) == doc

    async def test_deactivate_clears_steering(self, service):
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc())
        await service.activate(circuit.id)
        result = await service.deactivate(circuit.id)
        service._sae_service.clear_circuit_steering.assert_called()
        assert result["is_active"] is False
        assert result["serving_mode"] is None

    async def test_summarize_renders_rung_language_from_the_ladder(self, service):
        circuit = await service.import_definition(make_doc(edges=[]))
        row = service.summarize(circuit)
        assert row["rung"] == 0
        assert row["rung_language"] == "associated"
        assert row["validated"] is False
        assert "attribution" in row["rung_next_step"]

    async def test_delete_deactivates_first(self, service):
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc())
        await service.activate(circuit.id)
        await service.delete(circuit.id)
        with pytest.raises(CircuitNotFoundError):
            await service.get(circuit.id)

    async def test_set_intensity_reapplies_when_serving_full(self, service):
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc())
        await service.activate(circuit.id)
        service._sae_service.set_circuit_steering.reset_mock()
        result = await service.set_intensity(circuit.id, 1.75)
        assert result["reapplied"] is True
        _, intensity = service._sae_service.set_circuit_steering.call_args[0]
        assert intensity == 1.75

    async def test_set_intensity_does_not_reapply_when_inactive(self, service):
        circuit = await service.import_definition(make_doc())
        result = await service.set_intensity(circuit.id, 1.2)
        assert result["reapplied"] is False
        assert result["intensity"] == 1.2
