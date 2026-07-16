"""
Unit tests for ClusterService (Feature 8, Task 3.4/3.5): import mapping,
compatibility outcomes, name dedupe, the activation gate, lambda/clamp math,
intensity, and lossless export.
"""

from unittest.mock import MagicMock, patch

import pytest

from millm.api.schemas.cluster import ClusterBundleV1, ClusterDefinitionV1
from millm.core.errors import ValidationError
from millm.db.repositories.profile_repository import ProfileRepository
from millm.services.cluster_service import ClusterService
from millm.services.profile_service import ProfileService


def make_definition(**overrides) -> ClusterDefinitionV1:
    base = {
        "kind": "mistudio.cluster-definition",
        "schema_version": "1",
        "name": "fear cluster",
        "narrative": "Steers toward fear.",
        "display_token": "fear",
        "model": {"hf_id": "google/gemma-2-2b"},
        "sae": {"mistudio_sae_id": "sae_a", "layer": 12, "n_features": 16384},
        "members": [
            {"feature_idx": 100, "strength": 1.2, "sign": 1, "max_activation": 3.1},
            {"feature_idx": 200, "strength": 0.8, "sign": -1},
        ],
        "budget": {"B": 2.4, "intensity": 1.25, "intensity_range": [0.5, 1.5]},
    }
    base.update(overrides)
    return ClusterDefinitionV1.model_validate(base)


class FakeAttachedState:
    """Stands in for the AttachedSAEState singleton."""

    def __init__(self, d_sae=16384, sae_id="sae_local", layer=12, attached=True):
        self.attached_sae = MagicMock(d_sae=d_sae) if attached else None
        self.attached_sae_id = sae_id if attached else None
        self.attached_layer = layer if attached else None


@pytest.fixture
def mock_sae_service():
    svc = MagicMock()
    svc.get_attachment_status.return_value = MagicMock(
        is_attached=True, sae_id="sae_local", layer=12
    )
    return svc


@pytest.fixture
async def service(test_session, mock_sae_service):
    repo = ProfileRepository(test_session)
    profile_service = ProfileService(repo, mock_sae_service)
    return ClusterService(profile_service, repo, mock_sae_service)


def patched_state(**kw):
    return patch(
        "millm.services.cluster_service.AttachedSAEState",
        return_value=FakeAttachedState(**kw),
    )


class TestImportMapping:
    async def test_sign_folds_into_steering_at_lambda1_basis(self, service):
        with patched_state():
            item = await service.import_definition(make_definition())
        assert item.status == "imported"
        profile = await service.repository.get(item.profile_id)
        assert profile.steering == {"100": 1.2, "200": -0.8}
        assert profile.source_kind == "cluster"
        assert profile.intensity == 1.25  # from budget.intensity, NOT baked in
        assert profile.description == "Steers toward fear."
        assert profile.cluster_meta["display_token"] == "fear"
        assert profile.sae_id == "sae_local"

    async def test_unbound_when_no_sae_attached(self, service):
        with patched_state(attached=False):
            item = await service.import_definition(make_definition())
        assert item.status == "imported_unbound"
        assert any("No SAE attached" in w for w in item.warnings)
        profile = await service.repository.get(item.profile_id)
        assert profile.sae_id is None

    async def test_feature_space_mismatch_imports_unbound_with_warning(self, service):
        with patched_state(d_sae=4096):
            item = await service.import_definition(make_definition())
        assert item.status == "imported_unbound"
        assert any("Feature-space mismatch" in w for w in item.warnings)

    async def test_layer_mismatch_warn_binds(self, service):
        with patched_state(layer=6):
            item = await service.import_definition(make_definition())
        assert item.status == "imported"
        assert any("Layer mismatch" in w for w in item.warnings)

    async def test_range_warning_when_lambda_max_exceeds_gate(self, service):
        d = make_definition(members=[
            {"feature_idx": 1, "strength": 150.0, "sign": 1},
        ])
        # 150 * λ_max 1.5 = 225 > 200
        with patched_state():
            item = await service.import_definition(d)
        assert any("clamp at apply time" in w for w in item.warnings)

    async def test_name_dedupe_rename(self, service):
        with patched_state():
            first = await service.import_definition(make_definition())
            second = await service.import_definition(make_definition())
        assert first.name == "fear cluster"
        assert second.name == "fear cluster (2)"

    async def test_name_conflict_fail_mode(self, service):
        with patched_state():
            await service.import_definition(make_definition())
            item = await service.import_definition(
                make_definition(), on_conflict="fail"
            )
        assert item.status == "error"
        assert "already exists" in item.error

    async def test_bundle_per_item_isolation(self, service):
        bundle = ClusterBundleV1.model_validate({
            "kind": "mistudio.cluster-bundle",
            "schema_version": "1",
            "definitions": [
                make_definition(name="a").model_dump(mode="json"),
                make_definition(name="b").model_dump(mode="json"),
            ],
        })
        with patched_state():
            # Force the second import to explode inside the service
            original = service.import_definition
            calls = {"n": 0}

            async def flaky(definition, **kw):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("boom")
                return await original(definition, **kw)

            service.import_definition = flaky
            result = await service.import_bundle(bundle)
        assert result.imported == 1
        assert result.errors == 1
        assert [r.status for r in result.results] == ["imported", "error"]


class TestActivationGate:
    async def test_activation_scales_by_lambda_and_clamps(
        self, service, mock_sae_service
    ):
        d = make_definition(members=[
            {"feature_idx": 10, "strength": 100.0, "sign": 1},
            {"feature_idx": 20, "strength": 180.0, "sign": 1},
        ])
        with patched_state():
            item = await service.import_definition(d)
            await service.set_intensity(item.profile_id, 1.5, reapply=False)
            await service.activate(item.profile_id)
        applied = mock_sae_service.set_steering_batch.call_args[0][0]
        assert applied[10] == 150.0        # 100 × 1.5
        assert applied[20] == 200.0        # 180 × 1.5 = 270 → clamped

    async def test_activation_blocked_on_declared_feature_space_mismatch(self, service):
        with patched_state():
            item = await service.import_definition(make_definition())
        # SAE swapped for a smaller one after import
        with patched_state(d_sae=4096):
            with pytest.raises(ValidationError, match="meaningless"):
                await service.activate(item.profile_id)

    async def test_activation_blocked_on_out_of_bounds_member(self, service):
        d = make_definition(
            sae={"layer": 12},  # no declared n_features — bounds check is backstop
            members=[{"feature_idx": 99999, "strength": 1.0}],
        )
        with patched_state(d_sae=16384):
            item = await service.import_definition(d)
            with pytest.raises(ValidationError, match="out of range"):
                await service.activate(item.profile_id)

    async def test_unbound_import_binds_on_successful_activation(self, service):
        with patched_state(attached=False):
            item = await service.import_definition(make_definition(sae={"layer": 12}))
        profile = await service.repository.get(item.profile_id)
        assert profile.sae_id is None
        with patched_state():
            await service.activate(item.profile_id)
        profile = await service.repository.get(item.profile_id)
        assert profile.sae_id == "sae_local"

    async def test_non_cluster_profile_refused(self, service):
        manual = await service.repository.create(
            profile_id="prof_manual1", name="manual", steering={"1": 1.0}
        )
        with pytest.raises(ValidationError, match="not an imported cluster"):
            await service.activate(manual.id)


class TestIntensityAndExport:
    async def test_set_intensity_reapplies_when_active(self, service, mock_sae_service):
        with patched_state():
            item = await service.import_definition(make_definition())
            await service.activate(item.profile_id)
            mock_sae_service.set_steering_batch.reset_mock()
            result = await service.set_intensity(item.profile_id, 0.5)
        assert result["reapplied"] is True
        applied = mock_sae_service.set_steering_batch.call_args[0][0]
        assert applied[100] == pytest.approx(0.6)   # 1.2 × 0.5

    async def test_set_intensity_no_reapply_when_inactive(self, service, mock_sae_service):
        with patched_state():
            item = await service.import_definition(make_definition())
            result = await service.set_intensity(item.profile_id, 0.5)
        assert result["reapplied"] is False
        mock_sae_service.set_steering_batch.assert_not_called()

    async def test_export_is_lossless(self, service):
        original = make_definition()
        with patched_state():
            item = await service.import_definition(original)
            exported = await service.export_definition(item.profile_id)
        assert exported.model_dump(mode="json") == original.model_dump(mode="json")

    async def test_list_clusters_excludes_manual(self, service):
        await service.repository.create(
            profile_id="prof_manual2", name="manual2", steering={}
        )
        with patched_state():
            await service.import_definition(make_definition())
            clusters = await service.list_clusters()
        assert len(clusters) == 1
        assert clusters[0].display_token == "fear"
        assert clusters[0].member_count == 2
        assert clusters[0].bound is True
