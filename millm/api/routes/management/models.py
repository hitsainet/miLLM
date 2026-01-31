"""
Model management API endpoints.

Provides endpoints for downloading, loading, unloading, and managing LLM models.
"""

from typing import Annotated

from fastapi import APIRouter, Path

from millm.api.dependencies import ModelServiceDep
from millm.api.schemas.common import ApiResponse
from millm.api.schemas.model import (
    ModelDownloadRequest,
    ModelPreviewRequest,
    ModelPreviewResponse,
    ModelResponse,
    SizeEstimate,
)

router = APIRouter(prefix="/api/models", tags=["models"])


# Type alias for model ID path parameter
ModelId = Annotated[int, Path(description="Model ID", ge=1)]


@router.get(
    "",
    response_model=ApiResponse[list[ModelResponse]],
    summary="List all models",
    description="Returns a list of all downloaded models.",
)
async def list_models(
    service: ModelServiceDep,
) -> ApiResponse[list[ModelResponse]]:
    """
    List all downloaded models.

    Returns all models in the database, including their current status.
    Download progress is included for models currently downloading.
    """
    models = await service.list_models()
    responses = []
    for m in models:
        response = ModelResponse.from_model(m)
        # Inject download progress for downloading models
        progress = service.get_download_progress(m.id)
        if progress is not None:
            response.download_progress = progress
        responses.append(response)
    return ApiResponse.ok(responses)


@router.post(
    "",
    response_model=ApiResponse[ModelResponse],
    status_code=202,
    summary="Download a model",
    description="Start downloading a model from HuggingFace or register a local model.",
)
async def download_model(
    request: ModelDownloadRequest,
    service: ModelServiceDep,
) -> ApiResponse[ModelResponse]:
    """
    Start downloading a model.

    For HuggingFace models, this starts a background download and returns immediately
    with status: downloading. Progress updates are sent via WebSocket.

    For local models, this validates the path and registers the model immediately.
    """
    model = await service.download_model(request)
    return ApiResponse.ok(ModelResponse.from_model(model))


@router.get(
    "/{model_id}",
    response_model=ApiResponse[ModelResponse],
    summary="Get model details",
    description="Returns details for a single model.",
)
async def get_model(
    model_id: ModelId,
    service: ModelServiceDep,
) -> ApiResponse[ModelResponse]:
    """
    Get details for a single model.

    Args:
        model_id: The model's unique identifier.
    """
    model = await service.get_model(model_id)
    response = ModelResponse.from_model(model)
    # Inject download progress for downloading models
    progress = service.get_download_progress(model_id)
    if progress is not None:
        response.download_progress = progress
    return ApiResponse.ok(response)


@router.delete(
    "/{model_id}",
    response_model=ApiResponse[None],
    summary="Delete a model",
    description="Delete a model from disk and database.",
)
async def delete_model(
    model_id: ModelId,
    service: ModelServiceDep,
) -> ApiResponse[None]:
    """
    Delete a model.

    Removes the model from the database and deletes cached files.
    Cannot delete a model that is currently loaded.
    """
    await service.delete_model(model_id)
    return ApiResponse.ok(None)


@router.post(
    "/{model_id}/load",
    response_model=ApiResponse[ModelResponse],
    status_code=202,
    summary="Load a model",
    description="Load a model into GPU memory.",
)
async def load_model(
    model_id: ModelId,
    service: ModelServiceDep,
) -> ApiResponse[ModelResponse]:
    """
    Load a model into GPU memory.

    If another model is already loaded, it will be unloaded first.
    Progress updates are sent via WebSocket.
    """
    model = await service.load_model(model_id)
    return ApiResponse.ok(ModelResponse.from_model(model))


@router.post(
    "/{model_id}/unload",
    response_model=ApiResponse[ModelResponse],
    summary="Unload a model",
    description="Unload a model from GPU memory.",
)
async def unload_model(
    model_id: ModelId,
    service: ModelServiceDep,
) -> ApiResponse[ModelResponse]:
    """
    Unload a model from GPU memory.

    Waits for any pending inference requests to complete before unloading.
    """
    model = await service.unload_model(model_id)
    return ApiResponse.ok(ModelResponse.from_model(model))


@router.post(
    "/{model_id}/cancel",
    response_model=ApiResponse[ModelResponse],
    summary="Cancel download",
    description="Cancel an in-progress model download.",
)
async def cancel_download(
    model_id: ModelId,
    service: ModelServiceDep,
) -> ApiResponse[ModelResponse]:
    """
    Cancel an in-progress download.

    Only works for models with status: downloading.
    The model's status will be changed to error.
    """
    model = await service.cancel_download(model_id)
    return ApiResponse.ok(ModelResponse.from_model(model))


@router.post(
    "/preview",
    response_model=ApiResponse[ModelPreviewResponse],
    summary="Preview a model",
    description="Get model information from HuggingFace without downloading.",
)
async def preview_model(
    request: ModelPreviewRequest,
    service: ModelServiceDep,
) -> ApiResponse[ModelPreviewResponse]:
    """
    Preview a model before downloading.

    Returns model metadata including size estimates for different quantization levels.
    """
    info = await service.preview_model(request)

    # Calculate estimated sizes based on params
    estimated_sizes = None
    params_str = info.get("params", "")
    if params_str:
        # Parse params like "2.5B", "9B", "3B" to estimate memory
        try:
            multiplier = 1.0
            if params_str.endswith("B"):
                params_num = float(params_str[:-1])
            elif params_str.endswith("M"):
                params_num = float(params_str[:-1]) / 1000
            else:
                params_num = float(params_str)

            # Estimate: ~0.5 bytes/param for Q4, ~1 byte for Q8, ~2 bytes for FP16
            base_gb = params_num  # params in billions
            estimated_sizes = {
                "Q4": SizeEstimate(
                    disk_mb=int(base_gb * 500),  # ~0.5 GB per billion params
                    memory_mb=int(base_gb * 600),  # slightly more for runtime
                ),
                "Q8": SizeEstimate(
                    disk_mb=int(base_gb * 1000),  # ~1 GB per billion params
                    memory_mb=int(base_gb * 1200),
                ),
                "FP16": SizeEstimate(
                    disk_mb=int(base_gb * 2000),  # ~2 GB per billion params
                    memory_mb=int(base_gb * 2400),
                ),
            }
        except (ValueError, TypeError):
            pass

    # Build preview response
    preview = ModelPreviewResponse(
        name=info.get("name", ""),
        params=info.get("params"),
        architecture=info.get("architecture"),
        is_gated=info.get("is_gated", False),
        requires_trust_remote_code=info.get("requires_trust_remote_code", False),
        estimated_sizes=estimated_sizes,
    )

    return ApiResponse.ok(preview)
