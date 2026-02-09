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
    Runtime properties (num_parameters, device, etc.) included for loaded models.
    """
    models = await service.list_models()
    loaded_info = service.get_loaded_model_info()

    responses = []
    for m in models:
        response = ModelResponse.from_model(m)
        # Inject download progress for downloading models
        progress = service.get_download_progress(m.id)
        if progress is not None:
            response.download_progress = progress
        # Inject runtime properties for loaded model
        if loaded_info and loaded_info["model_id"] == m.id:
            response.num_parameters = loaded_info["num_parameters"]
            response.memory_footprint = loaded_info["memory_footprint"]
            response.device = loaded_info["device"]
            response.dtype = loaded_info["dtype"]
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
    "/{model_id}/lock",
    response_model=ApiResponse[ModelResponse],
    summary="Lock a model",
    description="Lock a loaded model for steering (prevents auto-unload by inference requests).",
)
async def lock_model(
    model_id: ModelId,
    service: ModelServiceDep,
) -> ApiResponse[ModelResponse]:
    """
    Lock a model for steering.

    The model must be in LOADED state. Only one model can be locked at a time.
    While locked, the model is the only one returned by /v1/models and cannot
    be auto-unloaded by inference requests targeting other models.
    """
    model = await service.lock_model(model_id)
    return ApiResponse.ok(ModelResponse.from_model(model))


@router.post(
    "/{model_id}/unlock",
    response_model=ApiResponse[ModelResponse],
    summary="Unlock a model",
    description="Unlock a model to allow auto-unload by inference requests.",
)
async def unlock_model(
    model_id: ModelId,
    service: ModelServiceDep,
) -> ApiResponse[ModelResponse]:
    """
    Unlock a model.

    After unlocking, all downloaded models are visible via /v1/models and
    inference requests can auto-load any available model.
    """
    model = await service.unlock_model(model_id)
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
            if params_str.endswith("B"):
                params_num = float(params_str[:-1])
            elif params_str.endswith("M"):
                params_num = float(params_str[:-1]) / 1000
            elif params_str.endswith("T"):
                params_num = float(params_str[:-1]) * 1000
            else:
                params_num = float(params_str)

            # Estimates include ~20% overhead for inference runtime
            base_gb = params_num  # params in billions
            estimated_sizes = {
                "FP32": SizeEstimate(
                    disk_mb=int(base_gb * 4000),  # ~4 bytes/param
                    memory_mb=int(base_gb * 4800),
                ),
                "FP16": SizeEstimate(
                    disk_mb=int(base_gb * 2000),  # ~2 bytes/param
                    memory_mb=int(base_gb * 2400),
                ),
                "Q8": SizeEstimate(
                    disk_mb=int(base_gb * 1000),  # ~1 byte/param
                    memory_mb=int(base_gb * 1200),
                ),
                "Q4": SizeEstimate(
                    disk_mb=int(base_gb * 500),  # ~0.5 bytes/param
                    memory_mb=int(base_gb * 600),
                ),
                "Q2": SizeEstimate(
                    disk_mb=int(base_gb * 250),  # ~0.25 bytes/param
                    memory_mb=int(base_gb * 300),
                ),
            }
        except (ValueError, TypeError):
            pass

    # Build preview response with all available metadata
    preview = ModelPreviewResponse(
        name=info.get("name", ""),
        params=info.get("params"),
        architecture=info.get("architecture"),
        is_gated=info.get("is_gated", False),
        requires_trust_remote_code=info.get("requires_trust_remote_code", False),
        estimated_sizes=estimated_sizes,
        downloads=info.get("downloads", 0),
        likes=info.get("likes", 0),
        tags=info.get("tags"),
        pipeline_tag=info.get("pipeline_tag"),
        model_type=info.get("model_type"),
        architectures=info.get("architectures"),
        license=info.get("license"),
        language=info.get("language"),
    )

    return ApiResponse.ok(preview)
