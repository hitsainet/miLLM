"""
OpenAI-compatible embeddings endpoint.

POST /v1/embeddings - Create embeddings

Requires a model to already be loaded via the Management API.
"""

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.routes.openai.errors import (
    model_not_found_error,
    model_not_loaded_error,
)
from millm.api.schemas.openai import (
    EmbeddingRequest,
    EmbeddingResponse,
    OpenAIErrorResponse,
)
from millm.core.logging import get_logger
from millm.services.inference_service import InferenceService

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    responses={
        503: {"model": OpenAIErrorResponse, "description": "No model loaded"},
    },
)
async def create_embeddings(
    request: EmbeddingRequest,
    service: ModelServiceDep,
    response: Response,
    inference: InferenceService = Depends(get_inference_service),
) -> EmbeddingResponse | JSONResponse:
    """
    Create embeddings for input text.

    Returns vector embeddings using the model's last hidden layer
    with mean pooling. Auto-loads the requested model if not already loaded.
    """
    # Check if requested model exists in database
    model = await service.find_model_by_name(request.model)
    if not model:
        return model_not_found_error(request.model)

    # Require model to already be loaded — no auto-load
    model_info = inference.get_loaded_model_info()
    if not model_info:
        return model_not_loaded_error()
    if model_info.name != request.model:
        return model_not_found_error(request.model, model_info.name)

    input_count = len(request.input) if isinstance(request.input, list) else 1
    logger.info(
        "embedding_request",
        model=request.model,
        input_count=input_count,
    )

    response.headers["X-miLLM-Backend"] = inference.backend_name
    return await inference.create_embeddings(request)
