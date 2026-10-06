"""
OpenAI-compatible text completions endpoint.

POST /v1/completions - Create text completion (legacy endpoint)

Requires a model to already be loaded via the Management API.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import JSONResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.request_policy import (
    IGNORED_FIELDS_HEADER,
    Endpoint,
    apply_request_policy,
    ignored_fields_header,
)
from millm.api.routes.openai.load_policy import apply_load_policy, parse_load_policy
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
from millm.api.routes.openai.chat import build_probe_verdicts_header, seed_header
from millm.services.inference_service import (
    InferenceService,
    get_probe_verdicts,
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
    x_millm_lease: Annotated[str | None, Header(alias="X-miLLM-Lease")] = None,
    x_millm_load_policy: Annotated[str | None, Header(alias="X-miLLM-Load-Policy")] = None,
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

    # Scoring on a GGUF row is refused by the request policy above (the `logprobs` /
    # `allowed_token_ids` cells for llama.cpp), before the auto-load. The route kept its own copy
    # of that check until a test proved the table gives the same answer (Feature 25, FTID I9).

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
    # Feature 29: `X-miLLM-Load-Policy: refuse` promises this request never causes a swap.
    # After the pre-load refusals above, before the auto-load below (FR-29.4.5).
    load_policy = parse_load_policy(x_millm_load_policy)
    policy_refusal = await apply_load_policy(load_policy, model, inference, service)
    if policy_refusal is not None:
        return policy_refusal

    model_info = inference.get_loaded_model_info()
    if not model_info or model_info.name != request.model:
        try:
            # The holder's lease ID lifts a lease refusal on the swap; a foreign lease answers
            # 409 model_leased through load_refused_error below (never model_locked).
            await service.load_model_and_wait(model.id, lease_id=x_millm_lease)
        except ModelLockedError as exc:
            locked_name = (exc.details or {}).get("locked_model_name")
            if not locked_name:
                locked = await service.get_locked_model()
                locked_name = locked.name if locked else "unknown"
            return model_locked_error(request.model, locked_name)
        except ModelBusyError as exc:
            return model_busy_error(
                f"{exc}. Retry once it finishes before requesting '{request.model}'.",
                exc.details,
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

    # Feature 27: every `return_sae_activations` refusal, after the model is resident and before
    # any slot or generation (FR-27.1i, FR-27.2). A refused request never generated anything.
    from millm.services.request_activations import refuse_before_generation

    refuse_before_generation(request, inference, chat=False)

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
    # Probe verdicts (FR-24.7, FR-27.8g). ⚠ THIS ROUTE NEVER SENT THEM: only the chat route read
    # the verdicts, so FR-24.7's promise of the header "on both" was false here, and FR-27.8d's
    # `continuous_batching` reason on a text completion was invisible to every caller. Read AFTER
    # generation, as the chat route does — the verdict does not exist before. "" (nothing armed)
    # sets no header, so an unarmed server's response is unchanged.
    probe_header = build_probe_verdicts_header(get_probe_verdicts())
    if probe_header:
        response.headers["X-miLLM-Probe-Verdicts"] = probe_header
    if request.wants_scores() and request.return_sae_activations is not None:
        # X-09: scoring is always unsteered, and a scoring response carrying activations says so
        # in the header as well as in `read_point: "unsteered"` (FTDD §5.3; 028 owns the header
        # on every other response).
        response.headers["X-miLLM-Steering"] = "none"
    return result
