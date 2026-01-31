"""
SAE management API endpoints.

Provides endpoints for downloading, attaching, steering, and managing SAEs.
"""

from typing import Annotated

from fastapi import APIRouter, Path

from millm.api.dependencies import SAEServiceDep
from millm.api.schemas.common import ApiResponse
from millm.api.schemas.sae import (
    AttachResponse,
    AttachSAERequest,
    CompatibilityResult,
    DeleteResponse,
    DetachResponse,
    DownloadResponse,
    DownloadSAERequest,
    MonitoringRequest,
    PreviewSAERequest,
    PreviewSAEResponse,
    SAEFileInfo,
    SAEListResponse,
    SAEMetadata,
    SteeringBatchRequest,
    SteeringRequest,
    SteeringStatus,
)
from millm.api.schemas.sae import AttachmentStatus as AttachmentStatusSchema

router = APIRouter(prefix="/api/saes", tags=["saes"])


# Type alias for SAE ID path parameter
SAEId = Annotated[str, Path(description="SAE ID", max_length=100)]


@router.get(
    "",
    response_model=ApiResponse[SAEListResponse],
    summary="List all SAEs",
    description="Returns a list of all downloaded SAEs and current attachment status.",
)
async def list_saes(
    service: SAEServiceDep,
) -> ApiResponse[SAEListResponse]:
    """
    List all downloaded SAEs.

    Returns all SAEs in the database and their current status,
    along with the current attachment status.
    """
    saes = await service.list_saes()
    attachment = service.get_attachment_status()

    response = SAEListResponse(
        saes=[SAEMetadata.from_sae(s) for s in saes],
        total=len(saes),
        attachment=AttachmentStatusSchema(
            is_attached=attachment.is_attached,
            sae_id=attachment.sae_id,
            layer=attachment.layer,
            memory_usage_mb=attachment.memory_usage_mb,
            steering_enabled=attachment.steering_enabled,
            monitoring_enabled=attachment.monitoring_enabled,
        ),
    )
    return ApiResponse.ok(response)


@router.post(
    "/download",
    response_model=ApiResponse[DownloadResponse],
    status_code=202,
    summary="Download an SAE",
    description="Start downloading an SAE from HuggingFace.",
)
async def download_sae(
    request: DownloadSAERequest,
    service: SAEServiceDep,
) -> ApiResponse[DownloadResponse]:
    """
    Start downloading an SAE.

    Downloads the SAE from HuggingFace in the background.
    Progress updates are sent via WebSocket.
    """
    sae_id = await service.start_download(
        repository_id=request.repository_id,
        revision=request.revision,
    )
    return ApiResponse.ok(DownloadResponse(
        sae_id=sae_id,
        status="downloading",
        message=f"Download started for {request.repository_id}",
    ))


@router.post(
    "/preview",
    response_model=ApiResponse[PreviewSAEResponse],
    summary="Preview SAE repository",
    description="List SAE files in a HuggingFace repository without downloading.",
)
async def preview_sae_repository(
    request: PreviewSAERequest,
    service: SAEServiceDep,
) -> ApiResponse[PreviewSAEResponse]:
    """
    Preview SAE repository contents.

    Lists all SAE files available in the repository, including
    layer information and file sizes.
    """
    result = await service.preview_repository(
        repository_id=request.repository_id,
        revision=request.revision,
        token=request.hf_token,
    )
    return ApiResponse.ok(PreviewSAEResponse(
        repository_id=result["repository_id"],
        revision=result["revision"],
        model_id=result.get("model_id"),
        files=[
            SAEFileInfo(
                path=f["path"],
                size_bytes=f["size_bytes"],
                layer=f.get("layer"),
                width=f.get("width"),
            )
            for f in result["files"]
        ],
        total_files=result["total_files"],
    ))


@router.get(
    "/attachment",
    response_model=ApiResponse[AttachmentStatusSchema],
    summary="Get attachment status",
    description="Get the current SAE attachment status.",
)
async def get_attachment_status(
    service: SAEServiceDep,
) -> ApiResponse[AttachmentStatusSchema]:
    """
    Get current SAE attachment status.

    Returns information about the currently attached SAE,
    or indicates no SAE is attached.
    """
    status = service.get_attachment_status()
    return ApiResponse.ok(AttachmentStatusSchema(
        is_attached=status.is_attached,
        sae_id=status.sae_id,
        layer=status.layer,
        memory_usage_mb=status.memory_usage_mb,
        steering_enabled=status.steering_enabled,
        monitoring_enabled=status.monitoring_enabled,
    ))


@router.get(
    "/{sae_id}",
    response_model=ApiResponse[SAEMetadata],
    summary="Get SAE details",
    description="Returns details for a single SAE.",
)
async def get_sae(
    sae_id: SAEId,
    service: SAEServiceDep,
) -> ApiResponse[SAEMetadata]:
    """
    Get details for a single SAE.

    Args:
        sae_id: The SAE's unique identifier.
    """
    sae = await service.get_sae(sae_id)
    return ApiResponse.ok(SAEMetadata.from_sae(sae))


@router.delete(
    "/{sae_id}",
    response_model=ApiResponse[DeleteResponse],
    summary="Delete an SAE",
    description="Delete an SAE from disk and database.",
)
async def delete_sae(
    sae_id: SAEId,
    service: SAEServiceDep,
) -> ApiResponse[DeleteResponse]:
    """
    Delete an SAE.

    Removes the SAE from the database and deletes cached files.
    Cannot delete an SAE that is currently attached.
    """
    result = await service.delete_sae(sae_id)
    return ApiResponse.ok(DeleteResponse(
        status=result["status"],
        sae_id=result["sae_id"],
        freed_disk_mb=result["freed_disk_mb"],
    ))


@router.get(
    "/{sae_id}/compatibility",
    response_model=ApiResponse[CompatibilityResult],
    summary="Check SAE compatibility",
    description="Check if an SAE is compatible with the currently loaded model.",
)
async def check_compatibility(
    sae_id: SAEId,
    layer: int,
    service: SAEServiceDep,
) -> ApiResponse[CompatibilityResult]:
    """
    Check SAE compatibility with loaded model.

    Validates dimensions, layer range, and trained model match.
    """
    result = await service.check_compatibility(sae_id, layer)
    return ApiResponse.ok(CompatibilityResult(
        compatible=result.compatible,
        errors=result.errors,
        warnings=result.warnings,
    ))


@router.post(
    "/{sae_id}/attach",
    response_model=ApiResponse[AttachResponse],
    summary="Attach an SAE",
    description="Attach an SAE to the currently loaded model.",
)
async def attach_sae(
    sae_id: SAEId,
    request: AttachSAERequest,
    service: SAEServiceDep,
) -> ApiResponse[AttachResponse]:
    """
    Attach an SAE to the model.

    Loads the SAE into GPU memory and installs a forward hook
    at the specified layer.
    """
    result = await service.attach_sae(sae_id, request.layer)
    return ApiResponse.ok(AttachResponse(
        status=result["status"],
        sae_id=result["sae_id"],
        layer=result["layer"],
        memory_usage_mb=result["memory_usage_mb"],
        warnings=result.get("warnings", []),
    ))


@router.post(
    "/{sae_id}/detach",
    response_model=ApiResponse[DetachResponse],
    summary="Detach an SAE",
    description="Detach an SAE from the model.",
)
async def detach_sae(
    sae_id: SAEId,
    service: SAEServiceDep,
) -> ApiResponse[DetachResponse]:
    """
    Detach an SAE from the model.

    Removes the forward hook and frees GPU memory.
    """
    result = await service.detach_sae(sae_id)
    return ApiResponse.ok(DetachResponse(
        status=result["status"],
        sae_id=result["sae_id"],
        memory_freed_mb=result["memory_freed_mb"],
    ))


# =============================================================================
# Steering endpoints
# =============================================================================


@router.get(
    "/steering",
    response_model=ApiResponse[SteeringStatus],
    summary="Get steering status",
    description="Get current steering configuration.",
)
async def get_steering(
    service: SAEServiceDep,
) -> ApiResponse[SteeringStatus]:
    """
    Get current steering status.

    Returns whether steering is enabled and current steering values.
    """
    attachment = service.get_attachment_status()
    values = service.get_steering_values() if attachment.is_attached else {}

    return ApiResponse.ok(SteeringStatus(
        enabled=attachment.steering_enabled,
        values=values,
    ))


@router.post(
    "/steering",
    response_model=ApiResponse[SteeringStatus],
    summary="Set steering value",
    description="Set steering value for a single feature.",
)
async def set_steering(
    request: SteeringRequest,
    service: SAEServiceDep,
) -> ApiResponse[SteeringStatus]:
    """
    Set steering value for a feature.

    Positive values amplify the feature, negative values suppress it.
    """
    service.set_steering(request.feature_idx, request.value)
    service.enable_steering(True)

    return ApiResponse.ok(SteeringStatus(
        enabled=True,
        values=service.get_steering_values(),
    ))


@router.post(
    "/steering/batch",
    response_model=ApiResponse[SteeringStatus],
    summary="Set multiple steering values",
    description="Set steering values for multiple features at once.",
)
async def set_steering_batch(
    request: SteeringBatchRequest,
    service: SAEServiceDep,
) -> ApiResponse[SteeringStatus]:
    """
    Set steering values for multiple features.

    More efficient than setting individually when configuring many features.
    """
    service.set_steering_batch(request.steering)
    service.enable_steering(True)

    return ApiResponse.ok(SteeringStatus(
        enabled=True,
        values=service.get_steering_values(),
    ))


@router.post(
    "/steering/enable",
    response_model=ApiResponse[SteeringStatus],
    summary="Enable/disable steering",
    description="Enable or disable steering without clearing values.",
)
async def toggle_steering(
    enabled: bool,
    service: SAEServiceDep,
) -> ApiResponse[SteeringStatus]:
    """
    Enable or disable steering.

    When disabled, steering values are preserved but not applied.
    """
    service.enable_steering(enabled)

    attachment = service.get_attachment_status()
    values = service.get_steering_values() if attachment.is_attached else {}

    return ApiResponse.ok(SteeringStatus(
        enabled=enabled,
        values=values,
    ))


@router.delete(
    "/steering/{feature_idx}",
    response_model=ApiResponse[SteeringStatus],
    summary="Clear single feature steering",
    description="Clear steering for a single feature.",
)
async def clear_feature_steering(
    feature_idx: int,
    service: SAEServiceDep,
) -> ApiResponse[SteeringStatus]:
    """
    Clear steering for a single feature.

    Removes steering configuration for the specified feature index.
    """
    service.clear_steering(feature_idx)

    attachment = service.get_attachment_status()
    values = service.get_steering_values() if attachment.is_attached else {}

    return ApiResponse.ok(SteeringStatus(
        enabled=attachment.steering_enabled,
        values=values,
    ))


@router.delete(
    "/steering",
    response_model=ApiResponse[SteeringStatus],
    summary="Clear all steering",
    description="Clear all steering values.",
)
async def clear_steering(
    service: SAEServiceDep,
) -> ApiResponse[SteeringStatus]:
    """
    Clear all steering values.

    Removes all feature steering configurations.
    """
    service.clear_steering()

    return ApiResponse.ok(SteeringStatus(
        enabled=False,
        values={},
    ))


# =============================================================================
# Monitoring endpoints
# =============================================================================


@router.post(
    "/monitoring",
    response_model=ApiResponse[None],
    summary="Configure monitoring",
    description="Enable or disable feature activation monitoring.",
)
async def configure_monitoring(
    request: MonitoringRequest,
    service: SAEServiceDep,
) -> ApiResponse[None]:
    """
    Configure feature monitoring.

    When enabled, feature activations are captured during forward passes.
    Optionally specify specific features to monitor for efficiency.
    """
    service.enable_monitoring(request.enabled, request.features)
    return ApiResponse.ok(None)
