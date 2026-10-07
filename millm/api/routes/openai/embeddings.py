"""
OpenAI-compatible embeddings endpoint.

POST /v1/embeddings - Create embeddings

Loads the requested model on demand, like chat and completions, after every refusal that can be
decided from the request and the model row — so a refused request never evicts the resident model.
"""

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import JSONResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.request_policy import (
    IGNORED_FIELDS_HEADER,
    STRICT_HEADER,
    Endpoint,
    PolicyResult,
    engine_of,
    model_name_of,
    evaluate,
    ignored_fields_header,
    parse_strict,
    report_unused,
)
from millm.api.routes.openai.load_policy import apply_load_policy, parse_load_policy
from millm.api.routes.openai.errors import (
    create_openai_error,
    load_refused_error,
    model_busy_error,
    model_locked_error,
    model_not_found_error,
    model_not_loaded_error,
    server_error,
)
from millm.api.schemas.openai import (
    EmbeddingRequest,
    EmbeddingResponse,
    OpenAIErrorResponse,
)
from millm.core.errors import (
    MiLLMError,
    ModelBusyError,
    ModelLockedError,
)
from millm.core.logging import get_logger
from millm.api.provenance import PreGeneration, post_generation
from millm.services.inference_service import InferenceService

router = APIRouter()
logger = get_logger(__name__)


def validate_embeddings(request: EmbeddingRequest, model_row: Any, *, strict: bool) -> PolicyResult:
    """Every embeddings refusal decidable from the request and the ROW (026 FTDD §5).

    Feature 25's policy, which since Feature 30 also refuses `dimensions` and a GGUF row's
    non-mean `pooling`; schema validation has already applied the input caps.
    """
    return evaluate(request, Endpoint.EMBEDDINGS, engine_of(model_row), strict=strict,
        model_name=model_name_of(model_row),
    )


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
    http_request: Request,
    inference: InferenceService = Depends(get_inference_service),
    x_millm_lease: Annotated[str | None, Header(alias="X-miLLM-Lease")] = None,
    x_millm_load_policy: Annotated[str | None, Header(alias="X-miLLM-Load-Policy")] = None,
) -> EmbeddingResponse | JSONResponse:
    """
    Create embeddings for input text.

    Returns vectors pooled from the model's last hidden layer by `pooling` (`mean` default,
    `last`, `cls`), L2-normalised when `normalize` is true. Auto-loads the requested model if not
    already loaded. Never steered.
    """
    # Check if requested model exists in database
    model = await service.find_model_by_name(request.model)
    if not model:
        return model_not_found_error(request.model)

    # Refusals decidable from the request and the row run BEFORE the auto-load below, so none
    # of them evicts the resident model. Schema validation has already refused empty input and
    # a list over EMBEDDINGS_MAX_INPUTS (both 400, param `input`) and a `pooling` outside
    # mean/last/cls. The request policy (Feature 25's table) now refuses `dimensions` on every
    # model — none declares truncated-embedding support (T-91) — and, on a GGUF row, `pooling`
    # other than `mean`, since llama.cpp fixes pooling at load. A GGUF model otherwise embeds
    # (llama.cpp loaded with embedding=True and MEAN pooling). Inputs over the model's length
    # limit are refused by the service, which must tokenise them with the loaded tokenizer.
    # One copy, two callers: this route and the batch validator (Feature 26).
    policy = validate_embeddings(
        request, model, strict=parse_strict(http_request.headers.get(STRICT_HEADER))
    )
    report_unused(policy, Endpoint.EMBEDDINGS, http_request.headers)
    ignored_header = ignored_fields_header(policy)

    # Load on demand, same as chat and completions. Open WebUI calls this for
    # RAG with its own embedding model selected, which is a DIFFERENT model from
    # the chat one — so refusing anything not already loaded broke retrieval
    # even when the chat model was up.
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
            return server_error(f"Timed out loading '{request.model}'.")
        except MiLLMError as exc:
            return load_refused_error(request.model, exc)

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

    result = await inference.create_embeddings(request)
    response.headers.update(
        post_generation(
            request, inference, endpoint="embeddings", pre=PreGeneration(),
            ignored_header=ignored_header,
        )
    )
    return result
