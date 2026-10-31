"""031's weights status embeds 033's adapter block (addendum D-28), and never fails over it.

MUTATION CONTROLS (W3): drop `adapter=_adapter_status()` from `WeightService.status()` ->
TestTheEmbed; let `_adapter_status` re-raise -> TestTheEmbedNeverRaises; fill
`manifest_base_name` from the resident instead of the manifest -> TestTheEmbed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from millm.api.schemas.weights_adapters import AdapterStatus
from millm.services import adapter_service
from millm.services.weight_attachment import WeightService
from tests.unit.adapter_fixtures import BASE_NAME
from tests.unit.adapter_service_fixtures import ADAPTER_NAME, build_env, identity


@pytest.fixture
def env(tmp_path):
    e = build_env(tmp_path)
    yield e
    e.close()


class TestTheEmbed:
    def test_nothing_applied_is_an_unapplied_block_not_null(self, env):
        status = WeightService().status()
        assert status.adapter == AdapterStatus(applied=False)
        assert status.adapter.model_dump() == adapter_service.get_adapter_service().status()

    async def test_an_applied_adapter_is_the_same_block_the_adapter_route_returns(self, env):
        result = await env.service.load(identity(), env.write())
        block = WeightService().status().adapter
        assert block is not None and block.applied is True
        assert block.name == ADAPTER_NAME
        assert block.manifest_sha256 == result["adapter"]["manifest_sha256"]
        # base_name is the RESIDENT's served name; the manifest's base is separate.
        assert block.base_name == "tiny"
        assert block.manifest_base_name == BASE_NAME
        assert block.manifest_base_sha256 is None
        assert block == AdapterStatus.model_validate(adapter_service.get_adapter_service().status())

    async def test_the_route_carries_it(self, env):
        from httpx import ASGITransport, AsyncClient

        from millm.main import create_app

        await env.service.load(identity(), env.write())
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            data = (await client.get("/api/weights/status")).json()["data"]
        assert data["adapter"]["applied"] is True
        assert data["adapter"]["name"] == ADAPTER_NAME
        assert data["adapter"]["manifest_base_name"] == BASE_NAME


class TestTheEmbedNeverRaises:
    def test_a_failing_adapter_service_reports_null(self, env):
        with patch.object(adapter_service, "get_adapter_service",
                          side_effect=RuntimeError("boom")):
            status = WeightService().status()
        assert status.adapter is None
        assert status.weights_state == "clean" and status.barrier == "serving"

    def test_a_block_that_fails_validation_reports_null(self, env):
        service = adapter_service.get_adapter_service()
        with patch.object(service, "status", return_value={"applied": True, "leak": "x"}):
            assert WeightService().status().adapter is None
