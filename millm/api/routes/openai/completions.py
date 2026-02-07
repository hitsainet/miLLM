"""
OpenAI-compatible text completions endpoint.

POST /v1/completions - Create text completion (legacy endpoint)
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from millm.api.dependencies import get_inference_service
from millm.api.routes.openai.errors import model_not_found_error, model_not_loaded_error, validation_error
from millm.api.schemas.openai import (
    OpenAIErrorResponse,
    TextCompletionRequest,
    TextCompletionResponse,
)
from millm.core.logging import get_logger
from millm.services.inference_service import InferenceService

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/completions",
    response_model=TextCompletionResponse,
    responses={
        400: {"model": OpenAIErrorResponse, "description": "Streaming not supported"},
        503: {"model": OpenAIErrorResponse, "description": "No model loaded"},
    },
)
async def create_completion(
    request: TextCompletionRequest,
    inference: InferenceService = Depends(get_inference_service),
) -> TextCompletionResponse | JSONResponse:
    """
    Create a text completion.

    Accepts a prompt and returns a completion.
    This is the legacy completions endpoint (not chat).
    """
    # Check if model is loaded
    if not inference.is_model_loaded():
        return model_not_loaded_error()

    # Validate model name matches loaded model
    model_info = inference.get_loaded_model_info()
    if model_info and request.model != model_info.name:
        return model_not_found_error(request.model, model_info.name)

    logger.info(
        "text_completion_request",
        model=request.model,
        stream=request.stream,
    )

    if request.stream:
        return validation_error(
            "Streaming is not supported for the /v1/completions endpoint. "
            "Use /v1/chat/completions with stream=true instead.",
            param="stream",
        )

    response = await inference.create_text_completion(request)
    return response
