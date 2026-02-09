"""
OpenAI-compatible models endpoint.

GET /v1/models - List available models
GET /v1/models/{model_id} - Get model details

Behavior:
- When no model is locked: returns ALL available models (READY, LOADED, LOADING)
- When a model is locked for steering: returns only the locked model
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.routes.openai.errors import model_not_found_error
from millm.api.schemas.openai import ModelListResponse, ModelObject, OpenAIErrorResponse
from millm.core.logging import get_logger
from millm.services.inference_service import InferenceService

router = APIRouter()
logger = get_logger(__name__)


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    service: ModelServiceDep,
    inference: InferenceService = Depends(get_inference_service),
) -> ModelListResponse:
    """
    List available models.

    When no model is locked: returns all available models (READY, LOADED, LOADING).
    When a model is locked for steering: returns only the locked model.
    """
    data: list[ModelObject] = []

    locked = await service.get_locked_model()

    if locked:
        # Locked mode: only return the locked model
        created = int(locked.loaded_at.timestamp()) if locked.loaded_at else int(locked.created_at.timestamp())
        data.append(
            ModelObject(
                id=locked.name,
                created=created,
                owned_by=locked.repo_id or "miLLM",
            )
        )
    else:
        # Unlocked mode: return all available models
        models = await service.get_available_models()
        for m in models:
            created = int(m.loaded_at.timestamp()) if m.loaded_at else int(m.created_at.timestamp())
            data.append(
                ModelObject(
                    id=m.name,
                    created=created,
                    owned_by=m.repo_id or "miLLM",
                )
            )

    logger.debug("models_list_request", model_count=len(data))
    return ModelListResponse(data=data)


@router.get(
    "/models/{model_id}",
    response_model=ModelObject,
    responses={
        404: {"model": OpenAIErrorResponse, "description": "Model not found"},
    },
)
async def get_model(
    model_id: str,
    service: ModelServiceDep,
    inference: InferenceService = Depends(get_inference_service),
) -> ModelObject | JSONResponse:
    """
    Get details for a specific model.

    Returns 404 if model not found in available models.
    """
    # Check if model exists in database by name
    model = await service.find_model_by_name(model_id)
    if not model:
        return model_not_found_error(model_id)

    # If locked to a different model, hide this one
    locked = await service.get_locked_model()
    if locked and locked.id != model.id:
        return model_not_found_error(model_id)

    logger.debug("model_get_request", model_id=model_id)

    created = int(model.loaded_at.timestamp()) if model.loaded_at else int(model.created_at.timestamp())
    return ModelObject(
        id=model.name,
        created=created,
        owned_by=model.repo_id or "miLLM",
    )
