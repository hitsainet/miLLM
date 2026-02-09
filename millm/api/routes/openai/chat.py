"""
OpenAI-compatible chat completions endpoint.

POST /v1/chat/completions - Create chat completion

Supports both streaming and non-streaming responses.
Auto-loads requested model if not already loaded (Ollama-like behavior).
"""

import asyncio
from typing import Union

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.routes.openai.errors import (
    model_locked_error,
    model_not_found_error,
    server_error,
)
from millm.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIErrorResponse,
)
from millm.core.errors import ModelLockedError
from millm.core.logging import get_logger
from millm.services.inference_service import InferenceService

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        503: {"model": OpenAIErrorResponse, "description": "No model loaded"},
    },
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    service: ModelServiceDep,
    inference: InferenceService = Depends(get_inference_service),
) -> Union[ChatCompletionResponse, StreamingResponse, JSONResponse]:
    """
    Create a chat completion.

    Accepts messages in OpenAI format and returns a completion.
    Supports both streaming (stream=true) and non-streaming responses.
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

    # Apply profile override if specified
    if request.profile:
        try:
            from millm.services.sae_service import AttachedSAEState
            from millm.db.base import async_session_factory
            from millm.db.repositories.profile_repository import ProfileRepository

            async with async_session_factory() as session:
                repo = ProfileRepository(session)
                profile = await repo.get_by_name(request.profile)
                if profile and profile.steering:
                    sae_state = AttachedSAEState()
                    sae = sae_state.attached_sae
                    if sae:
                        sae.set_steering_batch(profile.get_steering_dict())
                        sae.enable_steering(True)
                        logger.info(
                            "profile_applied",
                            profile=request.profile,
                            features=len(profile.steering),
                        )
        except Exception as e:
            logger.warning("profile_apply_failed", profile=request.profile, error=str(e))

    logger.info(
        "chat_completion_request",
        model=request.model,
        message_count=len(request.messages),
        stream=request.stream,
    )

    # Handle streaming vs non-streaming
    if request.stream:
        return StreamingResponse(
            inference.stream_chat_completion(request),
            media_type="text/event-stream",
        )
    else:
        response = await inference.create_chat_completion(request)
        return response
