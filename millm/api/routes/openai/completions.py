"""
OpenAI-compatible text completions endpoint.

POST /v1/completions - Create text completion (legacy endpoint)

Requires a model to already be loaded via the Management API.
"""

import asyncio
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.request_policy import (
    IGNORED_FIELDS_HEADER,
    Endpoint,
    apply_request_policy,
    ignored_fields_header,
)
from millm.api.routes.openai.errors import (
    create_openai_error,
    embedding_model_error,
    is_embedding_only,
    load_refused_error,
    model_busy_error,
    model_locked_error,
    model_not_found_error,
    model_not_loaded_error,
    server_error,
    validation_error,
)
from millm.api.schemas.openai import (
    OpenAIErrorResponse,
    TextCompletionRequest,
    TextCompletionResponse,
)
from millm.core.errors import (
    MiLLMError,
    ModelBusyError,
    ModelLockedError,
)
from millm.core.logging import get_logger
from millm.api.routes.openai.chat import seed_header
from millm.services.inference_service import (
    InferenceService,
    get_request_outcome,
    reset_request_outcome,
)
from millm.services.system_fingerprint import build_system_fingerprint

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
    response: Response,
    http_request: Request,
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

    # An embedding model cannot generate text. Refuse before loading it —
    # loading costs time and VRAM to reach an answer guaranteed to be nonsense.
    if is_embedding_only(getattr(model, "architecture", None)):
        return embedding_model_error(request.model, model.architecture)

    # Feature 25 request policy, before anything that could load a model (see chat.py).
    policy = apply_request_policy(request, Endpoint.COMPLETIONS, model, http_request.headers)
    ignored_header = ignored_fields_header(policy)

    # Streaming has never been implemented on /v1/completions at all, on any
    # engine. Refused here rather than after the auto-load below: the answer
    # depends only on `request.stream`, so loading a model first spends a full
    # swap — evicting whatever is resident and any SAEs attached to it — to
    # reach a 400 that was decided by the request body.
    if request.stream:
        return validation_error(
            "Streaming is not supported for the /v1/completions endpoint. "
            "Use /v1/chat/completions with stream=true instead.",
            param="stream",
        )

    # Scoring mode needs a per-token distribution, which the llama.cpp engine does not expose here.
    # Decided by the request and the row, so refused BEFORE the auto-load below would evict the
    # resident model and its SAEs to reach the same 400 (review round 1, M2).
    if request.wants_scores() and getattr(model, "gguf_files", None):
        return validation_error(
            "Scoring mode (logprobs / allowed_token_ids) needs the transformers engine; "
            f"'{request.model}' is a GGUF model served by llama.cpp.",
            param="logprobs",
        )

    # Load the requested model on demand.
    #
    # An OpenAI client — Open WebUI included — selects a model by naming it in
    # the request body. Rejecting anything that is not already loaded makes
    # model selection a no-op: the picker changes, the request 404s, and the
    # user has to go and load the model by hand somewhere else.
    #
    # load_model_and_wait() was written for exactly this ("Used by the
    # OpenAI-compatible endpoints for auto-load on demand") and had NO callers.
    # It returns immediately when the model is already loaded, so the common
    # path costs nothing.
    #
    # A model LOCKED for steering is the one case where the model must not
    # change: swapping the weights out from under an attached SAE would leave
    # the steering vectors pointing at a different model. load_model_and_wait
    # raises ModelLockedError for that, and only that.
    model_info = inference.get_loaded_model_info()
    if not model_info or model_info.name != request.model:
        try:
            await service.load_model_and_wait(model.id)
        except ModelLockedError as exc:
            locked_name = (exc.details or {}).get("locked_model_name")
            if not locked_name:
                locked = await service.get_locked_model()
                locked_name = locked.name if locked else "unknown"
            return model_locked_error(request.model, locked_name)
        except ModelBusyError as exc:
            return model_busy_error(
                f"{exc}. Retry once it finishes before requesting '{request.model}'."
            )
        except asyncio.TimeoutError:
            return server_error(
                f"Timed out loading '{request.model}'. The model may still be "
                f"loading; retry shortly."
            )
        except MiLLMError as exc:
            return load_refused_error(request.model, exc)

        # Confirm the switch actually happened rather than assuming it did.
        model_info = inference.get_loaded_model_info()
        if not model_info:
            return model_not_loaded_error()
        if model_info.name != request.model:
            return model_not_found_error(request.model, model_info.name)

    logger.info(
        "text_completion_request",
        model=request.model,
        stream=request.stream,
    )

    response.headers["X-miLLM-Backend"] = inference.backend_name
    reset_request_outcome()
    result = await inference.create_text_completion(request)
    if ignored_header:
        response.headers[IGNORED_FIELDS_HEADER] = ignored_header
    if request.seed is not None:
        response.headers["X-miLLM-Seed"] = seed_header(
            request.seed,
            get_request_outcome().get("seed_scope") or inference.seed_scope_for(request),
        )
    result.system_fingerprint = build_system_fingerprint(model, inference.loaded_model())
    return result
