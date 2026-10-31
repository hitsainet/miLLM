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



def _slice_import_ok(status: str = "imported", profile_id: str = "prof_slice"):
    """A REAL ClusterImportItem — a bare MagicMock answers `.status` with a
    truthy Mock, which is exactly what hid the ignored-status bug (R2)."""
    from millm.api.schemas.cluster import ClusterImportItem

    return ClusterImportItem(
        name="slice", status=status, profile_id=profile_id, warnings=[]
    )

@pytest.fixture(autouse=True)
def clean_registry():
    state = AttachedSAEState()
    state.reset_for_tests()
    yield
    state.reset_for_tests()


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
    cluster_service.import_definition = AsyncMock(return_value=_slice_import_ok())
    cluster_service.get_active_cluster = AsyncMock(return_value=None)
    cluster_service.deactivate = AsyncMock()
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

    async def test_deactivate_stops_this_circuits_steering(self, service):
        """F19 changed the MECHANISM here deliberately, so this asserts the
        OUTCOME instead.

        It previously asserted `clear_circuit_steering.assert_called()` — a
        GLOBAL clear. That is correct when one circuit can serve and
        catastrophic when two can: it tears out every co-tenant's steering
        while their rows still read active. Deactivation now releases this
        circuit's OWNER contribution, and the co-tenant survival case is
        asserted in `test_sae_owner_provenance.py`.

        Asserting "the steering stopped" rather than "this particular function
        was called" is also what lets the mechanism change again without a
        test that only pins today's implementation.
        """
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc())
        await service.activate(circuit.id)

        # `_sae_service` is a Mock here, so the real apply never runs and no
        # owner registers. What IS observable is which release path was taken:
        # the activation must serve as an owner, and the deactivation must
        # release THAT owner rather than clearing globally.
        owner = f"circuit:{circuit.id}"
        _, kwargs = service._sae_service.set_circuit_steering.call_args
        assert kwargs.get("owner_id") == owner, (
            "the circuit did not serve as an owner, so nothing scopes its "
            "release and deactivating it would clear every co-tenant"
        )

        from millm.services.sae_service import AttachedSAEState

        # Register a real contribution so the release has something to drop.
        AttachedSAEState().apply_owner(owner, {})

        result = await service.deactivate(circuit.id)

        assert AttachedSAEState().owner_keys(owner) == {}, (
            "the circuit still owns steering after being deactivated"
        )
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


class TestR1Fixes:
    """Regressions for review round 1 — the slice-fallback path had NEVER
    executed (a raw dict was passed where a validated model was required, and
    an AsyncMock hid the crash)."""

    async def test_slice_fallback_passes_a_validated_model_not_a_dict(self, service):
        """The exact R1 bug: ClusterService.import_definition takes a
        ClusterDefinitionV1; a dict crashed on `.name`."""
        from millm.api.schemas.cluster import ClusterDefinitionV1

        attach("sae-10", 10)  # L13 missing → slice fallback
        circuit = await service.import_definition(make_doc())
        await service.activate(circuit.id)
        call = service._cluster_service.import_definition.await_args
        assert isinstance(call[0][0], ClusterDefinitionV1)
        assert call.kwargs["raw_payload"]["kind"] == "mistudio.cluster-definition"

    async def test_slice_survives_a_circuit_name_longer_than_the_cluster_cap(
        self, service
    ):
        """A circuit name may be 200 chars but a cluster name caps at 120 —
        the projection must still validate (truncate), not become impossible."""
        from millm.api.schemas.cluster import ClusterDefinitionV1

        long_name = "x" * 200
        definition = CircuitDefinitionV1.model_validate(make_doc(name=long_name))
        doc = service.to_layer_slice(definition, 10)
        parsed = ClusterDefinitionV1.model_validate(doc)  # must not raise
        assert parsed.name.endswith("[L10 slice]")
        assert len(parsed.name) <= 120

    def test_slice_carries_global_intensity_without_a_per_layer_budget(self, service):
        """Dropping λ when the layer has no per-layer entry silently served the
        slice at the cluster default (1.0) instead of the authored value."""
        doc = make_doc(budget={"layers": {"13": {"B": 30.0}}, "intensity": 1.5,
                               "intensity_range": [0.0, 2.0]})
        definition = CircuitDefinitionV1.model_validate(doc)
        sl = service.to_layer_slice(definition, 10)  # no '10' key in layers
        assert sl["budget"] is not None
        assert sl["budget"]["intensity"] == 1.5

    def test_cluster_ref_with_both_expansion_and_feature_keeps_both(self, service):
        """Taking only one branch silently dropped authored members."""
        doc = make_doc(
            saes=[{"layer": 10, "n_features": 8192, "mistudio_sae_id": "sae-10"}],
            members=[{
                "layer": 10, "member_kind": "cluster_ref",
                "expanded_members": [{"feature_idx": 7, "strength": 10.0}],
                "feature": {"feature_idx": 999, "strength": 5.0},
            }],
            edges=[],
        )
        definition = CircuitDefinitionV1.model_validate(doc)
        # F18: the derivation moved to the engine; these F13-R1 assertions
        # are unchanged, which is the parity claim.
        from millm.ml.circuit_steering import CircuitSteeringEngine

        members = CircuitSteeringEngine.serving_members(definition)
        assert {m.feature_idx for m in members} == {7, 999}
        sl = service.to_layer_slice(definition, 10)
        assert {m["feature_idx"] for m in sl["members"]} == {7, 999}

    def test_duplicate_feature_across_cluster_ref_and_feature_ref_is_deduped(
        self, service
    ):
        """The serving path rejects a repeated (layer, feature_idx) outright —
        a feature appearing both standalone and inside a referenced cluster is
        a natural authoring outcome and must not fail activation."""
        doc = make_doc(
            saes=[{"layer": 10, "n_features": 8192, "mistudio_sae_id": "sae-10"}],
            members=[
                {"layer": 10, "member_kind": "cluster_ref",
                 "expanded_members": [{"feature_idx": 1, "strength": 10.0}]},
                {"layer": 10, "feature": {"feature_idx": 1, "strength": 20.0}},
            ],
            edges=[],
        )
        definition = CircuitDefinitionV1.model_validate(doc)
        # F18: the derivation moved to the engine; these F13-R1 assertions
        # are unchanged, which is the parity claim.
        from millm.ml.circuit_steering import CircuitSteeringEngine

        members = CircuitSteeringEngine.serving_members(definition)
        assert [m.feature_idx for m in members] == [1]  # deduped, not doubled

    async def test_corrupt_stored_document_is_a_structured_error_not_a_500(
        self, service
    ):
        """circuit_meta is raw JSONB validated at IMPORT time; a later contract
        tightening or hand-edit must not escape as an opaque pydantic 500."""
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc())
        await service.repository.update(
            circuit.id, circuit_meta={"kind": "mistudio.circuit-definition",
                                      "schema_version": "1"}  # missing name/saes/members
        )
        with pytest.raises(ValidationError, match="no longer validates"):
            await service.activate(circuit.id)

    async def test_serveable_is_refreshed_at_activation(self, service):
        """serveable was a frozen import-time snapshot, so a circuit that became
        bindable kept reporting not-serveable while actively serving."""
        circuit = await service.import_definition(make_doc())  # nothing attached
        assert circuit.serveable is False
        attach("sae-10", 10)
        attach("sae-13", 13)
        await service.activate(circuit.id)
        refreshed = await service.repository.get(circuit.id)
        assert refreshed.serveable is True

    async def test_co_tenant_released_only_after_a_successful_serve(self, service):
        """Releasing before serving left the user with NOTHING steering when the
        serve then failed."""
        attach("sae-10", 10)
        attach("sae-13", 13)
        service._cluster_service.get_active_cluster = AsyncMock(
            return_value=MagicMock(id="prof_x", name="c", layer=10)
        )
        service._sae_service.set_circuit_steering.side_effect = RuntimeError("serve boom")
        circuit = await service.import_definition(make_doc())
        with pytest.raises(RuntimeError):
            await service.activate(circuit.id)
        # The cluster must still be running — the circuit never started.
        service._cluster_service.deactivate.assert_not_awaited()

    async def test_slice_fallback_releases_only_the_served_layer(self, service):
        """bound_layers may exceed what a slice actually serves; releasing the
        unused ones killed clusters for nothing."""
        attach("sae-10", 10)
        attach("sae-13", 13)
        doc = make_doc(
            saes=[
                {"layer": 10, "n_features": 8192, "mistudio_sae_id": "sae-10"},
                {"layer": 13, "n_features": 8192, "mistudio_sae_id": "sae-13"},
                {"layer": 20, "n_features": 8192, "mistudio_sae_id": "sae-20"},
            ],
            members=[
                {"layer": 10, "feature": {"feature_idx": 1, "strength": 10.0}},
                {"layer": 13, "feature": {"feature_idx": 2, "strength": 10.0}},
                {"layer": 20, "feature": {"feature_idx": 3, "strength": 10.0}},
            ],
            edges=[],
        )
        service._cluster_service.get_active_cluster = AsyncMock(
            return_value=MagicMock(id="prof_l13", name="on L13", layer=13)
        )
        circuit = await service.import_definition(doc)
        result = await service.activate(circuit.id, acknowledge_unvalidated=True)
        assert result["serving_mode"] == "slice_fallback"
        assert result["slice_layer"] == 10
        # The L13 cluster is untouched: the slice only serves L10.
        service._cluster_service.deactivate.assert_not_awaited()

    async def test_set_intensity_on_a_slice_says_it_was_not_applied(self, service):
        attach("sae-10", 10)
        circuit = await service.import_definition(make_doc())
        await service.activate(circuit.id)  # slice_fallback
        result = await service.set_intensity(circuit.id, 0.4)
        assert result["reapplied"] is False
        assert any("not applied" in w for w in result["warnings"])


class TestR2Fixes:
    """Review round 2 — the slice import's STATUS was ignored, so a failed
    slice still reported serving; and the evidence gate could be bypassed."""

    async def test_unbound_slice_import_does_not_report_serving(self, service):
        """ClusterService REPORTS its outcome (it does not raise): an
        incompatible slice returns status='imported_unbound' with activation
        explicitly skipped. Treating that as a serve marked the circuit active
        while the model ran completely unsteered."""
        attach("sae-10", 10)  # L13 missing → slice path
        service._cluster_service.import_definition = AsyncMock(
            return_value=_slice_import_ok(status="imported_unbound", profile_id="p1")
        )
        circuit = await service.import_definition(make_doc())
        with pytest.raises(SAESetIncompleteError) as ei:
            await service.activate(circuit.id)
        assert "slice_import_imported_unbound" in str(ei.value.offenders)
        refreshed = await service.repository.get(circuit.id)
        assert refreshed.is_active is False  # never claimed to serve

    async def test_errored_slice_import_does_not_report_serving(self, service):
        attach("sae-10", 10)
        service._cluster_service.import_definition = AsyncMock(
            return_value=_slice_import_ok(status="error", profile_id=None)
        )
        circuit = await service.import_definition(make_doc())
        with pytest.raises(SAESetIncompleteError):
            await service.activate(circuit.id)

    async def test_set_intensity_re_arm_requires_a_fresh_ack(self, service):
        """Re-applying steering is a fresh ARM — the gate must hold by
        construction, not just because activation checked it once (a restart
        can leave a stale active row)."""
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc(edges=[]))  # rung 0
        await service.activate(circuit.id, acknowledge_unvalidated=True)

        with pytest.raises(UnvalidatedCircuitError):
            await service.set_intensity(circuit.id, 1.5)

        result = await service.set_intensity(
            circuit.id, 1.5, acknowledge_unvalidated=True
        )
        assert result["reapplied"] is True

    async def test_validated_circuit_needs_no_ack_to_dial(self, service):
        attach("sae-10", 10)
        attach("sae-13", 13)
        circuit = await service.import_definition(make_doc())  # rung 2
        await service.activate(circuit.id)
        result = await service.set_intensity(circuit.id, 1.5)
        assert result["reapplied"] is True

    async def test_deactivate_tears_down_the_slice_cluster_profile(self, service):
        """In slice mode the CLUSTER profile is what steers — clearing only
        circuit steering reported success while the slice kept running."""
        attach("sae-10", 10)
        circuit = await service.import_definition(make_doc())
        await service.activate(circuit.id)  # slice_fallback
        service._cluster_service.deactivate.reset_mock()

        result = await service.deactivate(circuit.id)
        service._cluster_service.deactivate.assert_awaited_once_with("prof_slice")
        assert result["cleared_steering"] is True
        assert result["is_active"] is False

    async def test_slice_profile_id_is_persisted_for_teardown(self, service):
        attach("sae-10", 10)
        circuit = await service.import_definition(make_doc())
        await service.activate(circuit.id)
        refreshed = await service.repository.get(circuit.id)
        assert refreshed.provenance["slice_profile_id"] == "prof_slice"


class TestR3Fixes:
    """Review round 3 — the R2 fail-closed check was itself defeated, and the
    single-active invariant only held in one direction."""

    async def test_slice_import_that_imported_but_failed_to_activate(self, service):
        """ClusterService keeps status='imported' when ACTIVATION raised,
        recording the failure only as a warning — so checking status alone let
        the circuit claim to serve while the model ran unsteered."""
        attach("sae-10", 10)
        service._cluster_service.import_definition = AsyncMock(
            return_value=_slice_import_ok(status="imported", profile_id="p1")
        )
        # Simulate the real cluster behaviour: imported, but activation failed.
        service._cluster_service.import_definition.return_value.warnings = [
            "Imported but activation failed: SAE feature space mismatch"
        ]
        circuit = await service.import_definition(make_doc())
        with pytest.raises(SAESetIncompleteError) as ei:
            await service.activate(circuit.id)
        assert "slice_activation_failed" in str(ei.value.offenders)
        refreshed = await service.repository.get(circuit.id)
        assert refreshed.is_active is False

    def test_member_cap_counts_distinct_features_like_the_projection(self, service):
        """The cap must measure exactly what the projection emits: an overlap
        between a cluster_ref expansion and the member's own feature is ONE
        served member, not two."""
        overlapping = make_doc(
            saes=[{"layer": 10, "n_features": 8192, "mistudio_sae_id": "sae-10"}],
            members=[
                {
                    "layer": 10,
                    "member_kind": "cluster_ref",
                    "expanded_members": [
                        {"feature_idx": i, "strength": 1.0} for i in range(20)
                    ],
                    # Overlaps the expansion → still 20 DISTINCT features.
                    "feature": {"feature_idx": 0, "strength": 5.0},
                }
            ],
            edges=[],
        )
        definition = CircuitDefinitionV1.model_validate(overlapping)  # must not raise
        assert len(service.to_layer_slice(definition, 10)["members"]) == 20

    def test_member_cap_still_rejects_21_distinct_features(self, service):
        from pydantic import ValidationError as PydanticValidationError

        too_many = make_doc(
            saes=[{"layer": 10, "n_features": 8192}],
            members=[
                {"layer": 10, "feature": {"feature_idx": i, "strength": 1.0}}
                for i in range(21)
            ],
            edges=[],
        )
        with pytest.raises(PydanticValidationError):
            CircuitDefinitionV1.model_validate(too_many)

    def test_empty_budget_object_still_carries_the_dial(self, service):
        """A present-but-empty budget must not skip the λ fallback."""
        doc = make_doc(budget={"layers": {}, "intensity": 1.75})
        definition = CircuitDefinitionV1.model_validate(doc)
        sl = service.to_layer_slice(definition, 10, fallback_intensity=0.5)
        assert sl["budget"]["intensity"] == 1.75  # the document's λ wins

    @pytest.mark.parametrize("rung", [0, 1, 2, 3])
    def test_model_validated_property_delegates_to_the_ladder(self, rung):
        """Two implementations of the rung gate WILL drift — the property must
        delegate to the single ladder source."""
        from millm.core.circuit_evidence import is_validated
        from millm.db.models.circuit import Circuit

        c = Circuit(id="c", name="n", circuit_meta={}, layers=[], rung=rung)
        assert c.validated is is_validated(rung)
