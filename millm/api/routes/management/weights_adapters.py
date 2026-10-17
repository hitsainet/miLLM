"""Adapter serving (Feature 33): adapter load, unload and status.

Load and unload need a `trainer` or `operator` token (`WeightsIdentityDep`, addendum D-07): with
no trainer deployed, an operator token must be configured in `WEIGHTS_TOKENS` to serve adapters.
They honour the model lease (`X-miLLM-Lease`, D-15). Status needs no token (D-28).

Importing `adapter_service` here is what registers its base-replaced drop and its served-name
overlay in the live app; `tests/unit/api/test_adapter_routes.py` pins both against the registry.
"""

from typing import Annotated, Optional

from fastapi import APIRouter, Body, Depends, Header

from millm.api.schemas.common import ApiResponse
from millm.api.schemas.weights_adapters import (
    AdapterLoadRequest,
    AdapterLoadResponse,
    AdapterStatus,
    AdapterUnloadRequest,
    AdapterUnloadResponse,
)
from millm.core.weights_auth import WeightsIdentityDep
from millm.services.adapter_service import (
    OP_LOAD,
    OP_UNLOAD,
    AdapterService,
    get_adapter_service,
)

router = APIRouter(prefix="/api/weights", tags=["weights-adapters"])

AdapterServiceDep = Annotated[AdapterService, Depends(get_adapter_service)]
LeaseHeader = Annotated[
    Optional[str],
    Header(alias="X-miLLM-Lease", description="The lease ID, when the caller holds the lease"),
]


@router.post(
    "/adapter/load",
    response_model=ApiResponse[AdapterLoadResponse],
    summary="Apply a LoRA adapter to the resident base",
    # The route name is the audit `operation` of a token refusal (weights_auth reads
    # `route.name`), so it matches the service's own audit lines: one name per operation.
    name=OP_LOAD,
)
async def load_adapter(
    body: AdapterLoadRequest,
    identity: WeightsIdentityDep,
    service: AdapterServiceDep,
    lease: LeaseHeader = None,
) -> ApiResponse[AdapterLoadResponse]:
    result = await service.load(identity, body.path, lease_id=lease)
    return ApiResponse.ok(AdapterLoadResponse.model_validate(result))


@router.post(
    "/adapter/unload",
    response_model=ApiResponse[AdapterUnloadResponse],
    summary="Remove the applied adapter",
    name=OP_UNLOAD,
)
async def unload_adapter(
    identity: WeightsIdentityDep,
    service: AdapterServiceDep,
    lease: LeaseHeader = None,
    # Declared, never read: it carries nothing, but `extra="forbid"` makes a stray field a 422.
    body: Annotated[Optional[AdapterUnloadRequest], Body()] = None,  # noqa: ARG001
) -> ApiResponse[AdapterUnloadResponse]:
    result = await service.unload(identity, lease_id=lease)
    return ApiResponse.ok(AdapterUnloadResponse.model_validate(result))


@router.get(
    "/adapter",
    response_model=ApiResponse[AdapterStatus],
    summary="The applied adapter, if any (no token)",
)
async def adapter_status(service: AdapterServiceDep) -> ApiResponse[AdapterStatus]:
    return ApiResponse.ok(AdapterStatus.model_validate(service.status()))
