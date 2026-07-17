"""
Regression tests for Feature 8 review-round-1 fixes: the shared activation
gate (bypass + wipe-before-validate), deactivate scoping, set_intensity
rollback + authored-range enforcement, lossless raw storage, activation-
failure warning persistence, and cluster-row edit/flat-export guards.
"""

from unittest.mock import MagicMock, patch

import pytest

from millm.api.schemas.cluster import ClusterDefinitionV1
from millm.core.errors import ValidationError
from millm.db.repositories.profile_repository import ProfileRepository
from millm.services.cluster_service import ClusterService
from millm.services.profile_service import ProfileService


def make_definition(**overrides) -> ClusterDefinitionV1:
    base = {
        "kind": "mistudio.cluster-definition",
        "schema_version": "1",
        "name": "fear cluster",
        "sae": {"layer": 12, "n_features": 16384},
        "members": [{"feature_idx": 100, "strength": 1.2}],
        "budget": {"intensity": 1.0, "intensity_range": [0.5, 1.5]},
    }
    base.update(overrides)
    return ClusterDefinitionV1.model_validate(base)


class FakeAttachedState:
    def __init__(self, d_sae=16384, sae_id="sae_local", layer=12, attached=True):
        self.attached_sae = MagicMock(d_sae=d_sae) if attached else None
        self.attached_sae_id = sae_id if attached else None
        self.attached_layer = layer if attached else None


def patched_state(**kw):
    import contextlib

    @contextlib.contextmanager
    def _both():
        state = FakeAttachedState(**kw)
        with patch("millm.services.cluster_service.AttachedSAEState",
                   return_value=state), \
             patch("millm.services.sae_service.AttachedSAEState",
                   return_value=state):
            yield state

    return _both()


@pytest.fixture
def mock_sae_service():
    svc = MagicMock()
    svc.get_attachment_status.return_value = MagicMock(
        is_attached=True, sae_id="sae_local", layer=12
    )
    return svc


@pytest.fixture
async def profile_service(test_session, mock_sae_service):
    return ProfileService(ProfileRepository(test_session), mock_sae_service)


@pytest.fixture
async def cluster_service(test_session, profile_service, mock_sae_service):
    return ClusterService(
        profile_service, profile_service.repository, mock_sae_service
    )


class TestSharedActivationGate:
    """The gate lives in ProfileService — the generic /api/profiles route can
    no longer bypass it, and validation runs BEFORE clear_steering."""

    async def test_generic_activate_path_enforces_cluster_gate(
        self, profile_service, cluster_service
    ):
        with patched_state():
            item = await cluster_service.import_definition(make_definition())
        # SAE swapped for a mismatched one; activate via the GENERIC service
        # path (what POST /api/profiles/{id}/activate calls)
        with patched_state(d_sae=4096):
            with pytest.raises(ValidationError, match="meaningless"):
                await profile_service.activate_profile(item.profile_id)

    async def test_failed_activation_never_clears_live_steering(
        self, profile_service, cluster_service, mock_sae_service
    ):
        with patched_state():
            item = await cluster_service.import_definition(
                make_definition(sae={"layer": 12},
                                members=[{"feature_idx": 99999, "strength": 1.0}])
            )
        with patched_state(d_sae=100):
            with pytest.raises(ValidationError):
                await profile_service.activate_profile(item.profile_id)
        # The wipe-before-validate bug would have called clear_steering here.
        mock_sae_service.clear_steering.assert_not_called()
        mock_sae_service.set_steering_batch.assert_not_called()

    async def test_deactivating_inactive_profile_keeps_live_steering(
        self, profile_service, mock_sae_service
    ):
        active = await profile_service.repository.create(
            profile_id="prof_a", name="active", steering={"1": 1.0}
        )
        inactive = await profile_service.repository.create(
            profile_id="prof_b", name="inactive", steering={"2": 1.0}
        )
        await profile_service.repository.set_active(active.id)

        await profile_service.deactivate_profile(inactive.id)
        mock_sae_service.clear_steering.assert_not_called()

        # Deactivating the ACTIVE one clears as before
        await profile_service.deactivate_profile(active.id)
        mock_sae_service.clear_steering.assert_called_once()

    async def test_cluster_steering_not_editable_via_update(
        self, profile_service, cluster_service
    ):
        with patched_state():
            item = await cluster_service.import_definition(make_definition())
        with pytest.raises(ValidationError, match="imported cluster"):
            await profile_service.update_profile(
                item.profile_id, steering={100: 50.0}
            )
        # Name/description edits stay allowed
        updated = await profile_service.update_profile(
            item.profile_id, description="renarrated"
        )
        assert updated.description == "renarrated"


class TestIntensityFixes:
    async def test_out_of_authored_range_rejected(self, cluster_service):
        with patched_state():
            item = await cluster_service.import_definition(make_definition())
            with pytest.raises(ValidationError, match="declared range"):
                await cluster_service.set_intensity(item.profile_id, 2.0)

    async def test_dial_off_always_allowed(self, cluster_service):
        with patched_state():
            item = await cluster_service.import_definition(make_definition())
            result = await cluster_service.set_intensity(item.profile_id, 0.0)
        assert result["intensity"] == 0.0

    async def test_failed_reapply_rolls_lambda_back(
        self, cluster_service, mock_sae_service
    ):
        with patched_state():
            item = await cluster_service.import_definition(make_definition())
            await cluster_service.activate(item.profile_id)
        # Reapply under a mismatched SAE → gate fails → lambda must roll back
        with patched_state(d_sae=4096):
            with pytest.raises(ValidationError):
                await cluster_service.set_intensity(item.profile_id, 1.5)
        row = await cluster_service.repository.get(item.profile_id)
        assert row.intensity == 1.0


class TestLosslessAndWarnings:
    async def test_raw_payload_with_unknown_fields_survives_roundtrip(
        self, cluster_service
    ):
        raw = make_definition().model_dump(mode="json")
        raw["future_field"] = {"from": "a newer producer"}
        raw["members"][0]["future_member_field"] = 42
        definition = ClusterDefinitionV1.model_validate(raw)  # extra ignored
        with patched_state():
            item = await cluster_service.import_definition(
                definition, raw_payload=raw
            )
        row = await cluster_service.repository.get(item.profile_id)
        assert row.cluster_meta["future_field"] == {"from": "a newer producer"}
        assert row.cluster_meta["members"][0]["future_member_field"] == 42
        # Round 2: the FULL round-trip — export must re-emit the unknown
        # fields too (re-validating through the mirror used to strip them).
        exported = await cluster_service.export_definition(item.profile_id)
        assert exported["future_field"] == {"from": "a newer producer"}
        assert exported["members"][0]["future_member_field"] == 42

    async def test_activation_failure_warning_is_persisted(self, cluster_service):
        d = make_definition(
            sae={"layer": 12},  # no declared n_features → import binds
            members=[{"feature_idx": 99999, "strength": 1.0}],
        )
        with patched_state(d_sae=100):
            item = await cluster_service.import_definition(d, activate=True)
        assert any("activation failed" in w for w in item.warnings)
        row = await cluster_service.repository.get(item.profile_id)
        assert any("activation failed" in w
                   for w in row.cluster_meta.get("warnings", []))


class TestUnboundActivateWarning:
    async def test_activate_on_unbound_import_warns_explicitly(
        self, cluster_service
    ):
        """009 R3: activate=true on an unbound import must not be silent."""
        d = make_definition(sae={"layer": 12, "n_features": 4096})
        with patched_state(d_sae=16384):  # mismatched -> unbound
            item = await cluster_service.import_definition(d, activate=True)
        assert item.status == "imported_unbound"
        assert any("Activation requested but skipped" in w
                   for w in item.warnings)
        row = await cluster_service.repository.get(item.profile_id)
        assert any("skipped" in w
                   for w in row.cluster_meta.get("warnings", []))


class TestSensingOverridesExportStrip:
    async def test_export_strips_local_sensing_overrides(
        self, cluster_service
    ):
        """Enh R1: UI-set min_k lives OUTSIDE the portable document."""
        raw = make_definition().model_dump(mode="json")
        definition = ClusterDefinitionV1.model_validate(raw)
        with patched_state():
            item = await cluster_service.import_definition(
                definition, raw_payload=raw)
        row = await cluster_service.repository.get(item.profile_id)
        row.cluster_meta = {**row.cluster_meta,
                            "sensing_overrides": {"min_k": 2}}
        exported = await cluster_service.export_definition(item.profile_id)
        assert "sensing_overrides" not in exported
