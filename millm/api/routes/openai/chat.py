"""
OpenAI-compatible chat completions endpoint.

POST /v1/chat/completions - Create chat completion

Supports both streaming and non-streaming responses.
"""

from typing import Union

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from millm.api.dependencies import get_inference_service
from millm.api.routes.openai.errors import model_not_loaded_error
from millm.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIErrorResponse,
)
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
    inference: InferenceService = Depends(get_inference_service),
) -> Union[ChatCompletionResponse, StreamingResponse, JSONResponse]:
    """
    Create a chat completion.

    Accepts messages in OpenAI format and returns a completion.
    Supports both streaming (stream=true) and non-streaming responses.
    """
    # Check if model is loaded
    if not inference.is_model_loaded():
        return model_not_loaded_error()

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
