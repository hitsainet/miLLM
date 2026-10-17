"""`POST /api/profiles/import` refusals answer with their codes instead of raising.

A local `from millm.api.schemas.common import ApiResponse` inside `import_profile` made the name
local to the whole function, so the cluster-document branch above it raised `UnboundLocalError`
(ruff F823) and answered 500 instead of `IS_CLUSTER_DOCUMENT`. Nothing tested either branch.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from millm.api.routes.management.profiles import ProfileImportRequest, import_profile


async def test_a_cluster_document_is_redirected_not_raised():
    service = AsyncMock()
    request = ProfileImportRequest(kind="mistudio.cluster-definition/v1", name="humor")

    response = await import_profile(request, service)

    assert response.success is False
    assert response.error is not None and response.error.code == "IS_CLUSTER_DOCUMENT"
    assert "/api/clusters/import" in response.error.message
    service.create_profile.assert_not_called()


async def test_invalid_steering_keys_are_refused_with_their_names():
    service = AsyncMock()
    request = ProfileImportRequest(name="p", steering={"12": 1.0, "abc": 2.0})

    response = await import_profile(request, service)

    assert response.success is False
    assert response.error is not None and response.error.code == "VALIDATION_ERROR"
    assert response.error.details == {"invalid_keys": ["abc"]}
    service.create_profile.assert_not_called()


@pytest.mark.parametrize("kind", [None, "something-else"])
async def test_a_profile_document_is_created(kind):
    class _Created(Exception):
        pass

    service = AsyncMock()
    service.create_profile.side_effect = _Created  # stop at the call; its payload is the point
    request = ProfileImportRequest(kind=kind, name="p", steering={"12": 1.5})

    with pytest.raises(_Created):
        await import_profile(request, service)

    service.create_profile.assert_awaited_once()
    assert service.create_profile.await_args.kwargs["steering"] == {12: 1.5}
