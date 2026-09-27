"""Importing a probe definition: the gates, their order, and what survives the round trip.

The gates run size → kind → schema, and the order matters. Posting a circuit definition to the
probe importer should say "unknown kind", not produce forty field errors about a document of a
different shape.
"""

from __future__ import annotations

import pytest

from millm.core.errors import ValidationError
from millm.db.repositories.probe_repository import ProbeRepository
from millm.services.probe_service import ProbeService
from tests.unit.probe_fixtures import acknowledged, probe_definition, sae_probe_definition

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def service(test_session):
    return ProbeService(ProbeRepository(test_session))


class TestTheProjection:
    async def test_the_typed_columns_are_filled_from_the_document(self, service):
        probe = await service.import_definition(probe_definition())
        assert probe.hf_id == "LiquidAI/LFM2.5-1.2B-Instruct"
        assert probe.d_model == 8 and probe.n_layers == 16
        assert probe.layer == 11 and probe.rule == "mean" and probe.streamable is True
        assert probe.scope == "all" and probe.basis == "residual"
        assert probe.threshold == 1.0 and probe.target_fpr == 0.01
        assert probe.rung == 2
        assert probe.template_sha256 == "a" * 64

    async def test_the_RAW_document_is_stored_so_re_export_is_lossless(self, service):
        """⚠ pydantic's parsed form would drop additive fields a newer miStudio emits.

        A document that loses fields on a round trip is not the document that was published.
        """
        doc = probe_definition()
        doc["a_field_this_build_has_never_heard_of"] = {"nested": [1, 2, 3]}
        probe = await service.import_definition(doc)
        assert probe.definition["a_field_this_build_has_never_heard_of"] == {"nested": [1, 2, 3]}

    async def test_the_origin_is_recorded_in_provenance(self, service):
        probe = await service.import_definition(probe_definition(), origin="hub")
        assert probe.provenance["origin"] == "hub"
        # and the document's own provenance survives beside it
        assert probe.provenance["mistudio_probe_id"] == "pm_fixture01"

    async def test_an_sae_probe_stores_its_sae_block(self, service):
        probe = await service.import_definition(sae_probe_definition())
        assert probe.basis == "sae_features"
        assert probe.sae_ref["hf_repo"] == "mistudio/sae-lfm2p5-1p2b-instruct-l11-jumprelu"
        assert probe.sae_ref["feature_indices"] == [3, 11, 29, 47]

    async def test_a_dense_probe_stores_no_sae_block(self, service):
        probe = await service.import_definition(probe_definition())
        assert probe.sae_ref is None

    async def test_an_imported_probe_is_not_armed(self, service):
        """Importing is not arming. Nothing should start scoring because a file was uploaded."""
        probe = await service.import_definition(probe_definition())
        assert probe.armed is False

    async def test_the_definitions_acknowledgement_is_projected(self, service):
        doc = probe_definition()
        doc["evidence"] = {
            "rung": 1, "rung_language": "detects on held-out data",
            "acknowledgement": acknowledged("exploratory"), "evaluations": [],
        }
        probe = await service.import_definition(doc)
        assert probe.definition_acknowledgement["reason"] == "exploratory"
        # ...and the ARM acknowledgement is separate and still empty.
        assert probe.arm_acknowledgement is None


class TestTheGates:
    async def test_an_oversized_payload_is_refused_before_it_is_parsed(self, service):
        with pytest.raises(ValidationError) as exc:
            await service.import_definition(probe_definition(), raw_bytes=3_000_000)
        assert exc.value.details["code"] == "PAYLOAD_TOO_LARGE"
        assert exc.value.details["max_bytes"] == 2_097_152

    async def test_a_payload_at_the_cap_is_accepted(self, service):
        probe = await service.import_definition(probe_definition(), raw_bytes=2_097_152)
        assert probe.id.startswith("pr_")

    async def test_another_kind_is_UNKNOWN_KIND_not_a_wall_of_field_errors(self, service):
        """The kind gate runs before the schema for exactly this reason."""
        doc = probe_definition(kind="mistudio.circuit-definition/v1")
        with pytest.raises(ValidationError) as exc:
            await service.import_definition(doc)
        assert exc.value.details["code"] == "UNKNOWN_KIND"
        assert "errors" not in exc.value.details

    async def test_a_malformed_document_reports_its_field_errors(self, service):
        doc = probe_definition()
        doc["head"]["norm_std"] = [0.0] * 8
        with pytest.raises(ValidationError) as exc:
            await service.import_definition(doc)
        assert "errors" in exc.value.details

    async def test_an_unknown_origin_is_refused(self, service):
        with pytest.raises(ValidationError, match="Unknown origin"):
            await service.import_definition(probe_definition(), origin="somewhere")

    async def test_a_low_rung_document_without_an_acknowledgement_is_refused_at_IMPORT(
        self, service
    ):
        """The contract gate bites here, not only at arming — an endpoint gate is bypassed by a
        hand-edited file."""
        doc = probe_definition()
        doc["evidence"] = {"rung": 0, "rung_language": "trained", "acknowledgement": None,
                           "evaluations": []}
        with pytest.raises(ValidationError):
            await service.import_definition(doc)


class TestOnConflict:
    async def test_rename_appends_a_counter(self, service):
        first = await service.import_definition(probe_definition())
        second = await service.import_definition(probe_definition())
        assert first.name == "high-stakes"
        assert second.name == "high-stakes (2)"

    async def test_rename_keeps_counting(self, service):
        for _ in range(3):
            await service.import_definition(probe_definition())
        names = {p.name for p in await service.repository.list()}
        assert names == {"high-stakes", "high-stakes (2)", "high-stakes (3)"}

    async def test_fail_refuses_rather_than_renaming(self, service):
        await service.import_definition(probe_definition())
        with pytest.raises(ValidationError, match="already exists"):
            await service.import_definition(probe_definition(), on_conflict="fail")

    async def test_replace_and_refuse_are_NOT_accepted(self, service):
        """⚠ The FPRD specified `rename|replace|refuse`; the repository's convention is
        `rename|fail`.

        `replace` is refused on merit, not only for consistency: overwriting a definition in place
        while the probe is armed would change the detector underneath a running monitor while every
        event kept the same probe_id.
        """
        for bad in ("replace", "refuse"):
            with pytest.raises(ValidationError, match="Unknown on_conflict"):
                await service.import_definition(probe_definition(), on_conflict=bad)

    async def test_a_differently_named_probe_does_not_collide(self, service):
        await service.import_definition(probe_definition())
        other = await service.import_definition(probe_definition(name="deception"))
        assert other.name == "deception"


class TestDelete:
    async def test_delete_removes_it(self, service):
        probe = await service.import_definition(probe_definition())
        assert await service.delete(probe.id) is True
        assert await service.repository.get(probe.id) is None

    async def test_deleting_a_missing_probe_is_false_not_an_error(self, service):
        assert await service.delete("pr_nope") is False

    async def test_an_ARMED_probe_cannot_be_deleted(self, service):
        """Deleting an armed probe would leave a hook installed against a row that no longer
        exists, and the next request would score into nothing."""
        probe = await service.import_definition(probe_definition())
        await service.repository.update(probe, armed=True)
        with pytest.raises(ValidationError, match="disarm it before deleting"):
            await service.delete(probe.id)
        assert await service.repository.get(probe.id) is not None
