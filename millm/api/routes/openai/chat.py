"""
OpenAI-compatible chat completions endpoint.

POST /v1/chat/completions - Create chat completion

Supports both streaming and non-streaming responses.
Requires a model to already be loaded via the Management API.
"""

from typing import Union

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse, StreamingResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.routes.openai.errors import (
    model_not_found_error,
    model_not_loaded_error,
)
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
    service: ModelServiceDep,
    response: Response,
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

    # Require model to already be loaded — no auto-load
    model_info = inference.get_loaded_model_info()
    if not model_info:
        return model_not_loaded_error()
    if model_info.name != request.model:
        return model_not_found_error(request.model, model_info.name)

    # Profile override (request.profile) is applied inside the inference service's
    # request-queue semaphore to prevent concurrent requests from racing on the
    # global SAE steering state.  The previous steering is restored after each
    # generation completes.  See InferenceService._apply_request_profile.

    logger.info(
        "chat_completion_request",
        model=request.model,
        message_count=len(request.messages),
        stream=request.stream,
    )

    # X-miLLM-Backend header lets clients distinguish which inference path served
    # the request (serial queue vs continuous batching) for latency debugging.
    backend = inference.backend_name

    # Handle streaming vs non-streaming
    if request.stream:
        # Check queue capacity before committing to a 200 streaming response.
        # If we let QueueFullError propagate from inside the generator, the HTTP
        # status is already 200 and the client sees a malformed stream instead of
        # a proper error response.
        from millm.services.request_queue import QueueFullError
        queue = inference.request_queue
        if queue.pending_count >= queue.max_pending:
            raise QueueFullError(
                f"Request queue is full ({queue.pending_count}/{queue.max_pending} pending). "
                "Please retry shortly."
            )
        return StreamingResponse(
            inference.stream_chat_completion(request),
            media_type="text/event-stream",
            headers={"X-miLLM-Backend": backend},
        )
    else:
        # FastAPI's injected Response lets us set custom headers on the
        # auto-serialised Pydantic response without wrapping it manually.
        response.headers["X-miLLM-Backend"] = backend
        return await inference.create_chat_completion(request)
