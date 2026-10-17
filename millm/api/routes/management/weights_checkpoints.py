"""Checkpoints (Feature 32): save, load and lineage.

Three routes under `/api/weights` (addendum scope: instantiate, FR-32.8, is DEFERRED and has no
route). Each handler parses, calls the service ONCE and wraps the result in the management
envelope `{success, data, error}`; refusals are `MiLLMError`s the registered handler renders.

| Route | Gate |
|---|---|
| `POST /api/weights/save` | a `trainer`/`operator` token AND the holding attachment (`X-miLLM-Attachment`) |
| `POST /api/weights/load` | a `trainer`/`operator` token; `X-miLLM-Attachment` recognises the holder (R-01.31, review 3.3); `X-miLLM-Lease` lifts a lease |
| `GET /api/weights/lineage` | none (FR-32.7.7) |

Registered in `millm/api/routes/__init__.py`; the set is pinned by
`tests/unit/api/test_weights_checkpoint_routes.py` over `app.openapi()["paths"]`.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header

from millm.api.dependencies import ModelServiceDep
from millm.api.schemas.common import ApiResponse
from millm.api.schemas.weights_checkpoints import (
    LineageResponse,
    LoadRequest,
    LoadResponse,
    SaveRequest,
    SaveResponse,
)
from millm.core.weights_auth import WeightsIdentityDep
from millm.services.weights_checkpoint_service import (
    WeightsCheckpointService,
    get_weights_checkpoint_service,
)

router = APIRouter(prefix="/api/weights", tags=["weights-checkpoints"])


def _service(model_service: ModelServiceDep) -> WeightsCheckpointService:
    return get_weights_checkpoint_service(model_service)


CheckpointServiceDep = Annotated[WeightsCheckpointService, Depends(_service)]

#: The attachment ID, as on every holder call (031 FTDD §5.1). Never logged.
AttachmentHeader = Annotated[
    str | None,
    Header(alias="X-miLLM-Attachment", description="The attachment ID returned by attach"),
]
LeaseHeader = Annotated[
    str | None,
    Header(alias="X-miLLM-Lease", description="The lease ID returned by the grant"),
]


@router.post(
    "/save",
    name="weights_save",
    response_model=ApiResponse[SaveResponse],
    status_code=201,
    summary="Save the resident weights as a checkpoint",
)
async def save_checkpoint(
    request: SaveRequest,
    identity: WeightsIdentityDep,
    service: CheckpointServiceDep,
    x_millm_attachment: AttachmentHeader = None,
) -> ApiResponse[SaveResponse]:
    """Snapshot the clean resident weights under a serving pause and publish
    `<root>/<run_id>/step-<NNNNNN>/` atomically (FR-32.1–FR-32.3). Holder only."""
    result = await service.save(identity, x_millm_attachment, request.manifest)
    return ApiResponse.ok(SaveResponse(**result.to_dict()))


@router.post(
    "/load",
    name="weights_load",
    response_model=ApiResponse[LoadResponse],
    summary="Load a verified checkpoint into the resident model",
)
async def load_checkpoint(
    request: LoadRequest,
    identity: WeightsIdentityDep,
    service: CheckpointServiceDep,
    x_millm_attachment: AttachmentHeader = None,
    x_millm_lease: LeaseHeader = None,
) -> ApiResponse[LoadResponse]:
    """Verify `weights_sha256`, then copy in place (addresses kept) or fully reload, inside the
    barrier (FR-32.5). `{"model_id": N}` reloads a plain row: the recovery from `torn` (D-01)."""
    result = await service.load(
        identity, request, attachment_id=x_millm_attachment, lease_id=x_millm_lease
    )
    return ApiResponse.ok(LoadResponse(**result.to_dict()))


@router.get(
    "/lineage",
    name="weights_lineage",
    response_model=ApiResponse[LineageResponse],
    summary="The resident model's parent chain, read from manifests",
)
async def lineage(service: CheckpointServiceDep) -> ApiResponse[Any]:
    """Walk `parent` back to `base_model` (FR-32.7). Weights are never hashed here."""
    body = await service.lineage()
    return ApiResponse.ok(LineageResponse.model_validate(body))
