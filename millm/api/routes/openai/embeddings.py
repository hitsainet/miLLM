"""
OpenAI-compatible embeddings endpoint.

POST /v1/embeddings - Create embeddings
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from millm.api.dependencies import get_inference_service
from millm.api.routes.openai.errors import model_not_loaded_error
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
    inference: InferenceService = Depends(get_inference_service),
) -> EmbeddingResponse | JSONResponse:
    """
    Create embeddings for input text.

    Returns vector embeddings using the model's last hidden layer
    with mean pooling.
    """
    # Check if model is loaded
    if not inference.is_model_loaded():
        return model_not_loaded_error()

    input_count = len(request.input) if isinstance(request.input, list) else 1
    logger.info(
        "embedding_request",
        model=request.model,
        input_count=input_count,
    )

    response = await inference.create_embeddings(request)
    return response
