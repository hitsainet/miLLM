"""
OpenAI-compatible models endpoint.

GET /v1/models - List available models
GET /v1/models/{model_id} - Get model details
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from millm.api.dependencies import get_inference_service
from millm.api.routes.openai.errors import model_not_found_error
from millm.api.schemas.openai import ModelListResponse, ModelObject, OpenAIErrorResponse
from millm.core.logging import get_logger
from millm.services.inference_service import InferenceService

router = APIRouter()
logger = get_logger(__name__)


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    inference: InferenceService = Depends(get_inference_service),
) -> ModelListResponse:
    """
    List available models.

    Returns models in OpenAI format. If no model is loaded,
    returns an empty list (not an error).
    """
    data: list[ModelObject] = []

    if inference.is_model_loaded():
        model_info = inference.get_loaded_model_info()
        if model_info:
            data.append(
                ModelObject(
                    id=model_info.name,
                    created=int(model_info.loaded_at.timestamp()),
                    owned_by="miLLM",
                )
            )
            logger.debug("models_list_request", model_count=1)
    else:
        logger.debug("models_list_request", model_count=0)

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
    inference: InferenceService = Depends(get_inference_service),
) -> ModelObject | JSONResponse:
    """
    Get details for a specific model.

    Returns 404 if model not found or model_id doesn't match loaded model.
    """
    if not inference.is_model_loaded():
        return model_not_found_error(model_id)

    model_info = inference.get_loaded_model_info()
    if not model_info or model_info.name != model_id:
        return model_not_found_error(model_id)

    logger.debug("model_get_request", model_id=model_id)

    return ModelObject(
        id=model_info.name,
        created=int(model_info.loaded_at.timestamp()),
        owned_by="miLLM",
    )
