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
from millm.services.inference_service import (
    InferenceService,
    circuit_apply_failed,
    reset_steering_memo,
)

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
    # generation completes.  See InferenceService._apply_request_steering.

    logger.info(
        "chat_completion_request",
        model=request.model,
        message_count=len(request.messages),
        stream=request.stream,
    )

    # X-miLLM-Backend header lets clients distinguish which inference path served
    # the request (serial queue vs continuous batching) for latency debugging.
    backend = inference.backend_name

    # X-miLLM-Steering-Intensity echoes the resolved lambda back to dial
    # clients (Feature 10) — emitted only when the field was present.  Resolved
    # here (not stashed at apply time) because streaming headers are sent
    # before the generator body runs.  Best-effort by design: the helper
    # returns None (no header) when nothing can apply (no SAE, unknown
    # profile, DB hiccup), and a concurrent profile switch while the request
    # queues can still skew a symbolic echo — documented in the API reference.
    # X-miLLM-Circuit-Rung tells a dial client WHAT it is steering with
    # (Feature 14). The phrase comes from the evidence ladder, never composed
    # here, so the header can never describe a rung<2 circuit as causal.
    # Drop any memoised steering verdict from a previous request sharing this
    # context — the memo must never outlive the request that set it.
    reset_steering_memo()

    echo_circuit_rung = None
    try:
        rung_info = await inference.active_circuit_rung()
        if rung_info is not None:
            # Structured (RFC 8941): the rung stays trivially parseable as an
            # int and the phrase is a quoted-string, so punctuation in the
            # ladder vocabulary can never break a naive parser.
            echo_circuit_rung = f'{rung_info[0]}; language="{rung_info[1]}"'

    except Exception:  # observability must never fail a chat request
        echo_circuit_rung = None

    echo_intensity = None
    if request.steering_intensity is not None:
        # For streaming, the echo resolution doubles as the pre-commit 404
        # check for a named profile (one profile read, not two).
        effective = await inference.resolve_request_intensity(
            request, ensure_named_profile=bool(request.stream)
        )
        if effective is not None:
            echo_intensity = f"{effective:g}"

    # Handle streaming vs non-streaming
    if request.stream:
        # Check queue capacity before committing to a 200 streaming response.
        # If we let QueueFullError propagate from inside the generator, the HTTP
        # status is already 200 and the client sees a malformed stream instead of
        # a proper error response.
        from millm.services.request_queue import QueueFullError

        # Same pre-commit rule for a bad profile name: apply-time validation
        # runs inside the generator (headers already sent), so check the 404
        # case here while we can still return a proper error response. When
        # a dial is present the echo resolution above already verified it.
        if request.profile and request.steering_intensity is None:
            await inference.ensure_profile_exists(request.profile)

        queue = inference.request_queue
        if queue.pending_count >= queue.max_pending:
            raise QueueFullError(
                f"Request queue is full ({queue.pending_count}/{queue.max_pending} pending). "
                "Please retry shortly."
            )
        stream_headers = {"X-miLLM-Backend": backend}
        if echo_intensity is not None:
            stream_headers["X-miLLM-Steering-Intensity"] = echo_intensity
        if echo_circuit_rung is not None:
            stream_headers["X-miLLM-Circuit-Rung"] = echo_circuit_rung
        return StreamingResponse(
            inference.stream_chat_completion(request),
            media_type="text/event-stream",
            headers=stream_headers,
        )
    else:
        # FastAPI's injected Response lets us set custom headers on the
        # auto-serialised Pydantic response without wrapping it manually.
        response.headers["X-miLLM-Backend"] = backend

        # X-miLLM-Batch advertises the batched-generation extension. Both
        # request schemas are extra="ignore", so a server predating
        # `extra_messages` ACCEPTS the field and silently returns a single
        # choice — a client cannot tell that from a batch of one. This header
        # is the capability probe that makes the difference observable; a
        # client that does not see it must fall back to serial requests.
        response.headers["X-miLLM-Batch"] = str(
            len(request.extra_messages) + 1 if request.extra_messages else 1
        )
        if echo_intensity is not None:
            response.headers["X-miLLM-Steering-Intensity"] = echo_intensity
        # F18 R3-01: generate FIRST, then decide whether the rung header is
        # still true. The dial applies inside generation and can fail; setting
        # the header beforehand made the response advertise causal-validated
        # evidence for an intervention that did not run. `circuit_apply_failed`
        # is the request-scoped record of that outcome.
        #
        # Only the non-streaming branch can do this. The streaming branch must
        # commit its headers before the first byte, so its header is a
        # best-effort statement of intent — recorded as known debt in the F18
        # review notes rather than papered over.
        result = await inference.create_chat_completion(request)
        if echo_circuit_rung is not None and not circuit_apply_failed():
            response.headers["X-miLLM-Circuit-Rung"] = echo_circuit_rung
        return result
