"""
OpenAI-compatible models endpoint.

GET /v1/models - List available models
GET /v1/models/{model_id} - Get model details (`{model_id:path}`: names may contain `/`)

Behavior:
- When no model is locked: returns ALL available models (READY, LOADED, LOADING)
- When a model is locked for steering: returns only the locked model
- The resident model is listed under its SERVED name (032 FR-32.6.2): the manifest name after a
  checkpoint load, the adapter name while one is applied, the row name otherwise. A checkpoint
  (weights-origin) row that is not resident is not listed: it is not loadable on demand.
"""

from typing import Any, Optional


from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from millm.api.dependencies import ModelServiceDep, get_inference_service
from millm.api.routes.openai.errors import model_not_found_error
from millm.api.routes.openai.model_resolution import (
    is_resident_row,
    resolve_requested_model,
    served_name_of,
    weights_origin_of,
)
from millm.api.schemas.openai import ModelListResponse, ModelObject, OpenAIErrorResponse
from millm.core.logging import get_logger
from millm.services.inference_service import InferenceService

router = APIRouter()
logger = get_logger(__name__)


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    service: ModelServiceDep,
    inference: InferenceService = Depends(get_inference_service),
) -> ModelListResponse:
    """
    List available models.

    When no model is locked: returns all available models (READY, LOADED, LOADING).
    When a model is locked for steering: returns only the locked model.
    """
    data: list[ModelObject] = []

    info = inference.get_loaded_model_info()
    served = served_name_of(info)

    def listed_name(row: Any) -> Optional[str]:
        """The id `row` is listed under, or None to leave it out."""
        # The resident is recognised by ROW ID: by name, a second row sharing the resident's
        # name (two precisions of one repo before migration 027) was listed under the
        # resident's served name too.
        if is_resident_row(row, info):
            return served or row.name
        # A weights-origin carrier (migration 021) loads only through POST /api/weights/load, so
        # an idle one is never listed: judged against the in-process resident above, never the
        # row's status, which can lag a load or an unload.
        if weights_origin_of(row):
            return None
        return row.name

    locked = await service.get_locked_model()

    if locked:
        # Locked mode: only return the locked model
        created = int(locked.loaded_at.timestamp()) if locked.loaded_at else int(locked.created_at.timestamp())
        name = listed_name(locked)
        if name is not None:
            data.append(
                ModelObject(
                    id=name,
                    created=created,
                    owned_by=locked.repo_id or "miLLM",
                )
            )
    else:
        # Unlocked mode: return all available models
        models = await service.get_available_models()
        for m in models:
            name = listed_name(m)
            if name is None:
                continue
            created = int(m.loaded_at.timestamp()) if m.loaded_at else int(m.created_at.timestamp())
            data.append(
                ModelObject(
                    id=name,
                    created=created,
                    owned_by=m.repo_id or "miLLM",
                )
            )

    logger.debug("models_list_request", model_count=len(data))
    return ModelListResponse(data=data)


# ⚠ `:path`, not a plain segment (addendum D-34): every Hub-based adapter and checkpoint name
# contains `/` (`meta-llama/Llama-3.1-8B-Instruct:r9:adapter-200`), `/v1/models` lists exactly
# those ids, and a plain `{model_id}` answered them with FastAPI's bare 404, not the OpenAI
# envelope.
@router.get(
    "/models/{model_id:path}",
    response_model=ModelObject,
    responses={
        404: {"model": OpenAIErrorResponse, "description": "Model not found"},
    },
)
async def get_model(
    model_id: str,
    service: ModelServiceDep,
    inference: InferenceService = Depends(get_inference_service),
) -> ModelObject | JSONResponse:
    """
    Get details for a specific model.

    Returns 404 if model not found in available models.
    """
    # The same resolver as the generation routes (032 FR-32.6.3): the resident answers to its
    # served and base names; an unknown name or a checkpoint row that is not resident is a 404.
    resolution = await resolve_requested_model(model_id, service, inference)
    if resolution.refusal is not None:
        return resolution.refusal
    model = resolution.row

    # If locked to a different model, hide this one
    locked = await service.get_locked_model()
    if locked and locked.id != model.id:
        return model_not_found_error(model_id)

    logger.debug("model_get_request", model_id=model_id)

    created = int(model.loaded_at.timestamp()) if model.loaded_at else int(model.created_at.timestamp())
    return ModelObject(
        id=(resolution.served_name or model.name) if resolution.resident else model.name,
        created=created,
        owned_by=model.repo_id or "miLLM",
    )
