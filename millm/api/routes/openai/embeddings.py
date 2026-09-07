"""
OpenAI-compatible embeddings endpoint.

POST /v1/embeddings - Create embeddings

Requires a model to already be loaded via the Management API.
"""

import asyncio
from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.routes.openai.errors import (
    create_openai_error,
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
    service: ModelServiceDep,
    response: Response,
    inference: InferenceService = Depends(get_inference_service),
) -> EmbeddingResponse | JSONResponse:
    """
    Create embeddings for input text.

    Returns vector embeddings using the model's last hidden layer
    with mean pooling. Auto-loads the requested model if not already loaded.
    """
    # Check if requested model exists in database
    model = await service.find_model_by_name(request.model)
    if not model:
        return model_not_found_error(request.model)

    # Embeddings are impossible on the llama.cpp engine, not merely
    # unimplemented: llama.cpp needs embedding=True at CONSTRUCTION and pools
    # internally, so there is no hidden_states[-1] to mean-pool.
    # `InferenceService.create_embeddings` already refuses — but it refuses
    # AFTER the auto-load below has evicted the resident transformers model and
    # any SAEs attached to it, and spent minutes and tens of GB bringing up a
    # model that could never have answered. `gguf_files` on the row is set at
    # DOWNLOAD time, so the answer is knowable with nothing resident. Same
    # signal and same reason as the GGUF guard in completions.py and the text
    # completion guard in completions.py.
 
    # Load on demand, same as chat and completions. Open WebUI calls this for
    # RAG with its own embedding model selected, which is a DIFFERENT model from
    # the chat one — so refusing anything not already loaded broke retrieval
    # even when the chat model was up. The docstring above has always claimed
    # this behaviour; only the code disagreed.
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
        except ModelBusyError:
            return server_error(
                f"Another model load is already in progress; retry before "
                f"requesting '{request.model}'."
            )
        except asyncio.TimeoutError:
            return server_error(f"Timed out loading '{request.model}'.")
        except MiLLMError as exc:
            return server_error(f"Could not load '{request.model}': {exc}")

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

    response.headers["X-miLLM-Backend"] = inference.backend_name
    return await inference.create_embeddings(request)
