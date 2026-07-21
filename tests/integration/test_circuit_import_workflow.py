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
from types import SimpleNamespace
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



def _slice_import_ok(status: str = "imported", profile_id: str = "prof_slice"):
    """A REAL ClusterImportItem — a bare MagicMock answers `.status` with a
    truthy Mock, which is exactly what hid the ignored-status bug (R2)."""
    from millm.api.schemas.cluster import ClusterImportItem

    return ClusterImportItem(
        name="slice", status=status, profile_id=profile_id, warnings=[]
    )

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
    state.reset_for_tests()
    yield
    state.reset_for_tests()


@pytest.fixture
def sae_service():
    """A REAL SAEService (serving math), with only the repo/hooker stubbed."""
    svc = SAEService.__new__(SAEService)
    svc._sae_state = AttachedSAEState()
    return svc


@pytest.fixture
def cluster_service():
    svc = MagicMock()
    svc.import_definition = AsyncMock(return_value=_slice_import_ok())
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


class TestF19CoTenantSurvivesTheLIFECYCLE:
    """Feature 19 task 3.4/6.4 — release is owner-scoped through the REAL
    `CircuitService.deactivate`, not just through `SAEService`.

    These exist because two mutations passed the entire suite:

        deactivate clears GLOBALLY again  -> SURVIVED
        rollback   clears GLOBALLY again  -> SURVIVED

    The unit tests covered the owner map, and the concurrent-serving tests
    covered `SAEService`, but nothing drove `CircuitService`'s own release
    paths — so reverting either to a blanket `clear_circuit_steering()` was
    invisible. That is the "declaring a mechanism is not wiring it" failure in
    its exact form: the mechanism was right and the caller was unpinned.
    """

    async def test_deactivating_one_circuit_leaves_a_CO_TENANTS_steering(
        self, service, sae_service
    ):
        attach("sae-10", 10, make_sae())
        attach("sae-13", 13, make_sae())

        circuit = await service.import_definition(load_fixture())
        await service.activate(circuit.id)

        state = AttachedSAEState()
        assert state.owner_keys(f"circuit:{circuit.id}"), (
            "the circuit did not serve as an owner"
        )

        # A co-tenant arrives on a layer the circuit does not touch.
        other_sae = make_sae()
        state.set(other_sae, "sae-20", 20, None)
        state.apply_owner("circuit:OTHER", {("sae-20", 20): {77: 25.0}})
        assert other_sae.get_steering_values() == {77: 25.0}

        await service.deactivate(circuit.id)

        assert other_sae.get_steering_values() == {77: 25.0}, (
            "deactivating one circuit cleared a co-tenant's steering — that "
            "circuit now serves nothing while its row still reads active"
        )
        assert state.owner_keys(f"circuit:{circuit.id}") == {}

    async def test_a_FAILED_activation_rolls_back_only_its_own_steering(
        self, service, sae_service, cluster_service
    ):
        """The ROLLBACK path, which the FTASKS names as the one most easily
        missed. A failed activation must not take a co-tenant down with it."""
        attach("sae-10", 10, make_sae())
        attach("sae-13", 13, make_sae())

        state = AttachedSAEState()
        other_sae = make_sae()
        state.set(other_sae, "sae-20", 20, None)
        state.apply_owner("circuit:OTHER", {("sae-20", 20): {77: 25.0}})

        circuit = await service.import_definition(load_fixture())

        # Make the post-serve step fail so the rollback runs.
        original = service.repository.update

        async def boom(*args, **kwargs):
            raise RuntimeError("activation exploded after the serve")

        service.repository.update = boom
        try:
            with pytest.raises(RuntimeError, match="exploded"):
                await service.activate(circuit.id)
        finally:
            service.repository.update = original

        assert other_sae.get_steering_values() == {77: 25.0}, (
            "the activation rollback cleared an unrelated circuit's steering"
        )


class TestF19TheClaimGate:
    """Feature 19 tasks 3.3/3.7/6.2 — contention refusal, atomicity, and the
    flag-off path.

    The gate's ORDERING is the feature: collision first and unconditionally,
    then contention, then the claim insert. A collision reachable through the
    override would let one author's strength silently overwrite another's.
    """

    async def _activate_first(self, service, monkeypatch):
        attach("sae-10", 10, make_sae())
        attach("sae-13", 13, make_sae())
        first = await service.import_definition(load_fixture())
        await service.activate(first.id)
        return first

    async def test_a_second_circuit_on_the_same_layers_is_REFUSED(
        self, service, monkeypatch
    ):
        from millm.core.errors import CircuitLayerContentionError

        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", True
        )
        first = await self._activate_first(service, monkeypatch)

        doc = load_fixture()
        doc["name"] = "second circuit"
        second = await service.import_definition(doc)

        with pytest.raises(CircuitLayerContentionError) as exc:
            await service.activate(second.id)

        err = exc.value
        assert err.code == "CIRCUIT_LAYER_CONTENTION"
        assert err.status_code == 200, "house style: refusal in the envelope"
        # Same features on the same layers — this is a COLLISION, and it must
        # be reported as one rather than as plain contention.
        assert err.details["overridable"] is False
        assert "cannot be overridden" in err.message

    async def test_the_refusal_is_ATOMIC(self, service, monkeypatch):
        """Nothing applied, incumbent untouched, no claim row left behind."""
        from millm.core.errors import CircuitLayerContentionError
        from millm.services.circuit_claim_registry import CircuitClaimRegistry

        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", True
        )
        first = await self._activate_first(service, monkeypatch)
        state = AttachedSAEState()
        incumbent_before = state.owner_keys(f"circuit:{first.id}")
        assert incumbent_before

        doc = load_fixture()
        doc["name"] = "second circuit"
        second = await service.import_definition(doc)

        with pytest.raises(CircuitLayerContentionError):
            await service.activate(second.id)

        assert state.owner_keys(f"circuit:{first.id}") == incumbent_before, (
            "the refused activation disturbed the incumbent's steering"
        )
        assert state.owner_keys(f"circuit:{second.id}") == {}, (
            "the refused circuit applied steering anyway"
        )

        registry = CircuitClaimRegistry(service.repository.session)
        owners = {c.circuit_id for c in await registry.live_claims()}
        assert second.id not in owners, "a refused activation left a claim row"

    async def test_the_incumbent_is_NAMED_in_the_refusal(
        self, service, monkeypatch
    ):
        """So the operator's next action is obvious: deactivate it, or edit
        one circuit's layers."""
        from millm.core.errors import CircuitLayerContentionError

        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", True
        )
        first = await self._activate_first(service, monkeypatch)

        doc = load_fixture()
        doc["name"] = "second circuit"
        second = await service.import_definition(doc)

        with pytest.raises(CircuitLayerContentionError) as exc:
            await service.activate(second.id)
        assert exc.value.details["incumbent"]["id"] == first.id

    async def test_disjoint_layers_activate_CLEANLY(self, service, monkeypatch):
        """The feature's whole point: no contention, no refusal, both serve."""
        from millm.services.circuit_claim_registry import CircuitClaimRegistry

        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", True
        )
        first = await self._activate_first(service, monkeypatch)

        # The fixture uses layers 10 and 13, so +10 shifts them to 20 and 23.
        attach("sae-20", 20, make_sae())
        attach("sae-23", 23, make_sae())
        doc = load_fixture()
        doc["name"] = "disjoint circuit"
        for sae in doc["saes"]:
            sae["layer"] += 10
        for member in doc["members"]:
            member["layer"] += 10
        for edge in doc.get("edges", []):
            for endpoint in ("up", "down"):
                if endpoint in edge:
                    edge[endpoint]["layer"] += 10
        second = await service.import_definition(doc)

        await service.activate(second.id)  # must not raise

        state = AttachedSAEState()
        assert state.owner_keys(f"circuit:{first.id}"), "the incumbent stopped"
        assert state.owner_keys(f"circuit:{second.id}"), "the newcomer did not serve"

        registry = CircuitClaimRegistry(service.repository.session)
        owners = {c.circuit_id for c in await registry.live_claims()}
        assert {first.id, second.id} <= owners

    async def test_the_FLAG_OFF_path_refuses_LOUDLY_naming_configuration(
        self, service, monkeypatch
    ):
        """CLAIM-M4. Flag-off must NOT fall back to the silent single-active
        disarm this feature replaces — that silent fallback IS the bug."""
        from millm.core.errors import CircuitLayerContentionError

        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", False
        )
        first = await self._activate_first(service, monkeypatch)
        state = AttachedSAEState()
        before = state.owner_keys(f"circuit:{first.id}")

        doc = load_fixture()
        doc["name"] = "second circuit"
        # Distinct features so this is CONTENTION, not a collision — the
        # flag-off branch is what must refuse it.
        for i, member in enumerate(doc["members"]):
            member["feature"]["feature_idx"] = 900 + i
        second = await service.import_definition(doc)

        with pytest.raises(CircuitLayerContentionError) as exc:
            await service.activate(second.id)

        assert "CIRCUIT_ALLOW_CONCURRENT" in exc.value.message, (
            "the refusal did not name configuration as the reason, so an "
            "operator cannot tell a policy refusal from a real conflict"
        )
        assert state.owner_keys(f"circuit:{first.id}") == before, (
            "the incumbent was silently disarmed — the exact bug F19 removes"
        )

    async def test_re_activating_the_SAME_circuit_is_idempotent(
        self, service, monkeypatch
    ):
        """EC-19.3. A circuit must not contend with its own incumbent claim."""
        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", True
        )
        first = await self._activate_first(service, monkeypatch)
        await service.activate(first.id)  # must not raise

        from millm.services.circuit_claim_registry import CircuitClaimRegistry

        registry = CircuitClaimRegistry(service.repository.session)
        live = [c for c in await registry.live_claims() if c.circuit_id == first.id]
        assert sorted(c.layer for c in live) == [10, 13], (
            "re-activation duplicated or dropped the circuit's claims"
        )

    async def test_contention_WITHOUT_the_override_is_refused(
        self, service, monkeypatch
    ):
        """The override branch itself, which was UNPINNED.

        A mutation removing `if not allow_layer_overlap:` SURVIVED the whole
        suite: contention was silently composed with nobody asking for it.
        That is the failure mode the entire feature exists to prevent — the
        close-out measured two steered layers at strength 5 destroying
        generation, and composition without an explicit act is how an operator
        gets there without being told.

        Distinct features, so this is CONTENTION (composable) rather than a
        collision (never composable) — the branch under test is the one the
        override can pass.
        """
        from millm.core.errors import CircuitLayerContentionError

        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", True
        )
        first = await self._activate_first(service, monkeypatch)

        doc = load_fixture()
        doc["name"] = "overlapping circuit"
        for i, member in enumerate(doc["members"]):
            member["feature"]["feature_idx"] = 900 + i
        second = await service.import_definition(doc)

        with pytest.raises(CircuitLayerContentionError) as exc:
            await service.activate(second.id)

        err = exc.value
        assert err.details["overridable"] is True
        assert err.details["override_param"] == "allow_layer_overlap"
        assert err.details["rung_header_suppressed_if_overridden"] is True

        from millm.services.sae_service import AttachedSAEState

        assert AttachedSAEState().owner_keys(f"circuit:{second.id}") == {}, (
            "contention was COMPOSED without anyone asking for it"
        )

    async def test_the_override_COMPOSES_and_says_so(self, service, monkeypatch):
        """The other side: with the explicit act, composition proceeds — and
        is reported, so the operator's response carries what they accepted."""
        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", True
        )
        first = await self._activate_first(service, monkeypatch)

        doc = load_fixture()
        doc["name"] = "overlapping circuit"
        for i, member in enumerate(doc["members"]):
            member["feature"]["feature_idx"] = 900 + i
        second = await service.import_definition(doc)

        result = await service.activate(second.id, allow_layer_overlap=True)

        assert sorted(result.get("composed_layers") or []) == [10, 13], (
            "the response did not report which layers are composed, so the "
            "operator cannot see what they accepted"
        )

        from millm.services.sae_service import AttachedSAEState

        state = AttachedSAEState()
        assert state.owner_keys(f"circuit:{first.id}"), "the incumbent stopped"
        assert state.owner_keys(f"circuit:{second.id}"), "the override did not serve"

        from millm.services.circuit_claim_registry import CircuitClaimRegistry

        registry = CircuitClaimRegistry(service.repository.session)
        composed = {c.circuit_id for c in await registry.live_claims() if c.composed}
        assert {first.id, second.id} <= composed, (
            "both sides of a composition must be marked, or the rung header "
            "cannot be suppressed for the incumbent"
        )


class TestF19R1TheClaimLifecycleIsCLOSED:
    """F19 R1-03/04. Two defects that a fully green suite could not see.

    R1-03 — `deactivate()` released the steering owner and left the DB claim
    row LIVE FOREVER. After activate(cA) → deactivate(cA), activating cB on
    cA's layers was refused naming cA as the incumbent; the obvious remedy —
    deactivate cA — is a NO-OP because cA is already inactive. The layer became
    permanently unclaimable and the only signal was a refusal naming a circuit
    plainly not running. Routine deactivation was a leak, not an edge case.

    R1-04 — `get_active()` used `scalar_one_or_none()`, which RAISES
    `MultipleResultsFound` on two active circuits: the exact state this feature
    exists to create. `_active_full_circuit` catches broadly and returns None,
    so every chat request would have served UNSTEERED while both rows read
    active. Verified by execution before the fix.
    """

    async def test_deactivating_RELEASES_the_layer_claims(
        self, service, monkeypatch
    ):
        from millm.services.circuit_claim_registry import CircuitClaimRegistry

        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", True
        )
        attach("sae-10", 10, make_sae())
        attach("sae-13", 13, make_sae())

        circuit = await service.import_definition(load_fixture())
        await service.activate(circuit.id)

        registry = CircuitClaimRegistry(service.repository.session)
        assert [c.layer for c in await registry.live_claims()] == [10, 13]

        await service.deactivate(circuit.id)

        assert await registry.live_claims() == [], (
            "deactivation left the layer claims live — those layers now refuse "
            "every future activation, for a circuit nobody can deactivate"
        )

    async def test_the_layers_are_REUSABLE_after_a_deactivation(
        self, service, monkeypatch
    ):
        """The operator-visible consequence, asserted end to end."""
        monkeypatch.setattr(
            "millm.core.config.settings.CIRCUIT_ALLOW_CONCURRENT", True
        )
        attach("sae-10", 10, make_sae())
        attach("sae-13", 13, make_sae())

        first = await service.import_definition(load_fixture())
        await service.activate(first.id)
        await service.deactivate(first.id)

        doc = load_fixture()
        doc["name"] = "second circuit"
        second = await service.import_definition(doc)

        # Must NOT raise: the first circuit is gone and its layers are free.
        await service.activate(second.id)

        from millm.services.sae_service import AttachedSAEState

        assert AttachedSAEState().owner_keys(f"circuit:{second.id}"), (
            "the second circuit could not take layers the first had released"
        )

    async def test_get_active_survives_TWO_active_circuits(self, service):
        """R1-04. `scalar_one_or_none()` raised here, and the caller's broad
        handler turned that into a silent unsteered serve."""
        from millm.db.models.circuit import Circuit

        session = service.repository.session
        for cid, layer in (("cA", 10), ("cB", 13)):
            session.add(
                Circuit(
                    id=cid, name=cid, circuit_meta={}, rung=2, edge_count=0,
                    layers=[layer], per_sae_warnings=[], serveable=True,
                    is_active=True, provenance={},
                )
            )
        await session.commit()

        active = await service.repository.get_active()
        assert active is not None, "get_active returned nothing with two active"

        every = await service.repository.list_active()
        assert {c.id for c in every} == {"cA", "cB"}, (
            "list_active did not report every serving circuit, so callers that "
            "must act on all of them (co-tenant release, status) see one"
        )


class TestF19R1AProfileReleasesEVERYCircuit:
    """F19 R1-05. `_release_active_circuit` read `get_active()` — SINGULAR.

    Under concurrency that either raised `MultipleResultsFound` into a broad
    handler (logged as one line, activation proceeding) or released an
    ARBITRARY one of the active circuits. Either way the profile then applied
    steering on top of a still-active, still-claimed circuit's layers: the
    silent co-tenant clobbering F19 exists to eliminate, arriving from the
    profile side instead.

    This path had ZERO test coverage — a mutation reverting it to
    `get_active()` survived the whole suite.
    """

    async def test_it_deactivates_ALL_active_circuits(self, test_session):
        from millm.db.models.circuit import Circuit
        from millm.services.profile_service import ProfileService

        for cid, layer in (("cA", 10), ("cB", 13)):
            test_session.add(
                Circuit(
                    id=cid, name=cid, circuit_meta={}, rung=2, edge_count=0,
                    layers=[layer], per_sae_warnings=[], serveable=True,
                    is_active=True, provenance={},
                )
            )
        await test_session.commit()

        svc = ProfileService.__new__(ProfileService)
        svc.repository = SimpleNamespace(session=test_session)

        warnings = await svc._release_active_circuit()

        assert len(warnings) == 2, (
            f"only {len(warnings)} circuit(s) released — a profile taking "
            "these layers left the other one active and steering them"
        )

        from millm.db.repositories.circuit_repository import CircuitRepository

        assert await CircuitRepository(test_session).list_active() == [], (
            "a circuit is still active after a profile took its layers"
        )

    async def test_it_releases_their_CLAIMS_too(self, test_session):
        """Otherwise the layers stay locked against every future activation —
        R1-03's defect reachable through the profile path."""
        from millm.db.models.circuit import Circuit
        from millm.services.circuit_claim_registry import CircuitClaimRegistry
        from millm.services.profile_service import ProfileService

        test_session.add(
            Circuit(
                id="cA", name="cA", circuit_meta={}, rung=2, edge_count=0,
                layers=[10], per_sae_warnings=[], serveable=True,
                is_active=True, provenance={},
            )
        )
        await test_session.commit()
        registry = CircuitClaimRegistry(test_session)
        await registry.claim("cA", {10})

        svc = ProfileService.__new__(ProfileService)
        svc.repository = SimpleNamespace(session=test_session)
        await svc._release_active_circuit()

        assert await registry.live_claims() == [], (
            "the profile took the layers but left the circuit's claims live"
        )

    async def test_no_active_circuits_is_a_clean_no_op(self, test_session):
        from millm.services.profile_service import ProfileService

        svc = ProfileService.__new__(ProfileService)
        svc.repository = SimpleNamespace(session=test_session)
        assert await svc._release_active_circuit() == []
