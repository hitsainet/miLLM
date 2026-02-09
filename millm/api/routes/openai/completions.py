"""
OpenAI-compatible text completions endpoint.

POST /v1/completions - Create text completion (legacy endpoint)

Auto-loads requested model if not already loaded (Ollama-like behavior).
"""

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.routes.openai.errors import (
    model_locked_error,
    model_not_found_error,
    server_error,
    validation_error,
)
from millm.api.schemas.openai import (
    OpenAIErrorResponse,
    TextCompletionRequest,
    TextCompletionResponse,
)
from millm.core.errors import ModelLockedError
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
    service: ModelServiceDep,
    inference: InferenceService = Depends(get_inference_service),
) -> TextCompletionResponse | JSONResponse:
    """
    Create a text completion.

    Accepts a prompt and returns a completion.
    This is the legacy completions endpoint (not chat).
    Auto-loads the requested model if not already loaded.
    """
    # Check if requested model exists in database
    model = await service.find_model_by_name(request.model)
    if not model:
        return model_not_found_error(request.model)

    # Ensure the model is loaded (auto-load if needed)
    model_info = inference.get_loaded_model_info()
    if not model_info or model_info.name != request.model:
        try:
            await service.load_model_and_wait(model.id)
        except ModelLockedError:
            locked = await service.get_locked_model()
            locked_name = locked.name if locked else "unknown"
            return model_locked_error(request.model, locked_name)
        except asyncio.TimeoutError:
            return server_error(f"Model '{request.model}' took too long to load")
        except Exception as e:
            logger.error("auto_load_failed", model=request.model, error=str(e))
            return server_error(f"Failed to load model '{request.model}': {str(e)}")

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
