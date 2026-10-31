"""
Integration workflow for Feature 8 (Task 6.0): a REAL miStudio-exported
definition (tests/fixtures/mistudio_export.cluster.json, captured from a
production miStudio instance on 2026-07-16) imports, activates with the
lambda-clamped steering applied, survives the single-active invariant across
manual<->cluster switches, and re-exports losslessly.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.support.attached_state import FakeAttachedState
from tests.support.attached_state import patched_state as _patched_state

#: The real cluster fixture in this module declares a 32768-wide SAE, so the
#: default attached width must match it for a BOUND import.
FIXTURE_D_SAE = 32768


def patched_state(**kw):
    kw.setdefault('d_sae', FIXTURE_D_SAE)
    return _patched_state(**kw)


from millm.api.schemas.cluster import ClusterBundleV1, ClusterDefinitionV1
from millm.core.steering_range import STEERING_RANGE
from millm.db.repositories.profile_repository import ProfileRepository
from millm.services.cluster_service import ClusterService
from millm.services.profile_service import ProfileService

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mistudio_export.cluster.json"





@pytest.fixture
def applied_steering():
    """Captures every set_steering_batch the SAE service receives."""
    return []


@pytest.fixture
async def service(test_session, applied_steering):
    sae_service = MagicMock()
    sae_service.get_attachment_status.return_value = MagicMock(
        is_attached=True, sae_id="sae_local", layer=12
    )
    sae_service.set_steering_batch.side_effect = (
        lambda steering: applied_steering.append(steering)
    )
    repo = ProfileRepository(test_session)
    return ClusterService(ProfileService(repo, sae_service), repo, sae_service)


@pytest.fixture
def real_definition() -> ClusterDefinitionV1:
    return ClusterDefinitionV1.model_validate(json.loads(FIXTURE.read_text()))


class TestRealFixtureWorkflow:
    async def test_import_activate_applies_authored_strengths(
        self, service, real_definition, applied_steering
    ):
        with patched_state():
            item = await service.import_definition(real_definition)
            assert item.status == "imported"
            await service.activate(item.profile_id)

        expected = {
            m.feature_idx: float(m.sign) * float(m.strength)
            for m in real_definition.members
        }
        lam = real_definition.budget.intensity if real_definition.budget else 1.0
        applied = applied_steering[-1]
        assert len(applied) == len(real_definition.members)
        for idx, base in expected.items():
            effective = max(-STEERING_RANGE, min(STEERING_RANGE, base * lam))
            assert applied[idx] == pytest.approx(effective), f"member {idx}"

    async def test_reexport_equals_original(self, service, real_definition):
        raw = json.loads(FIXTURE.read_text())
        with patched_state():
            item = await service.import_definition(real_definition, raw_payload=raw)
            exported = await service.export_definition(item.profile_id)
        # export re-emits the RAW document byte-semantically
        assert exported == raw

    async def test_single_active_invariant_across_manual_and_cluster(
        self, service, real_definition
    ):
        manual = await service.repository.create(
            profile_id="prof_manualX", name="manual", steering={"5": 1.0}
        )
        await service.repository.set_active(manual.id)

        with patched_state():
            item = await service.import_definition(real_definition)
            await service.activate(item.profile_id)

        manual_row = await service.repository.get(manual.id)
        cluster_row = await service.repository.get(item.profile_id)
        assert manual_row.is_active is False   # deactivated by the invariant
        assert cluster_row.is_active is True

    async def test_bundle_of_real_and_broken_definition(self, service, real_definition):
        broken = real_definition.model_dump(mode="json")
        broken["name"] = "will explode"
        bundle = ClusterBundleV1.model_validate({
            "kind": "mistudio.cluster-bundle",
            "schema_version": "1",
            "definitions": [real_definition.model_dump(mode="json"), broken],
        })

        original = service.import_definition
        calls = {"n": 0}

        async def flaky(definition, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated storage failure")
            return await original(definition, **kw)

        service.import_definition = flaky
        with patched_state():
            result = await service.import_bundle(bundle)
        assert result.imported == 1
        assert result.errors == 1

    async def test_unbound_fixture_blocks_until_compatible_sae(
        self, service, real_definition
    ):
        with patched_state(attached=False):
            item = await service.import_definition(real_definition)
        assert item.status == "imported_unbound"

        # Wrong feature space attached -> activation blocked
        declared = real_definition.sae.n_features
        if declared:
            with patched_state(d_sae=declared // 2):
                from millm.core.errors import ValidationError
                with pytest.raises(ValidationError):
                    await service.activate(item.profile_id)

        # Compatible SAE attached -> binds + activates
        with patched_state(d_sae=declared or 32768):
            await service.activate(item.profile_id)
        row = await service.repository.get(item.profile_id)
        assert row.sae_id == "sae_local"
        assert row.is_active is True


class TestGenericRouteGateHTTP:
    """Round-3: the round-1 bypass fix verified at the HTTP layer — the
    generic profiles route must surface the cluster gate as a 422 envelope."""

    async def test_profiles_activate_route_blocks_mismatched_cluster(
        self, service, real_definition
    ):
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from millm.api.dependencies import get_profile_service
        from millm.api.exception_handlers import millm_error_handler
        from millm.api.routes.management.profiles import router
        from millm.core.errors import MiLLMError

        with patched_state():
            item = await service.import_definition(real_definition)

        app = FastAPI()
        app.include_router(router)
        app.add_exception_handler(MiLLMError, millm_error_handler)
        app.dependency_overrides[get_profile_service] = (
            lambda: service.profile_service
        )

        with patched_state(d_sae=64):   # mismatched SAE
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post(f"/api/profiles/{item.profile_id}/activate", json={})
        assert r.status_code == 422
        body = r.json()
        assert "meaningless" in body["error"]["message"]
