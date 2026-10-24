"""`/api/models/{model_id}/probe-capacity` and `/probe-limits` — per-model probe limits (operator
2026-10-21: "I want to know the actual limits, what drives them, and to be able to adjust on a per
model basis accordingly").

* `GET  …/probe-capacity` — measured step time and probe cost lines, the configured per-model limits,
  the EFFECTIVE limits admission uses, the limits the cost lines imply, and (for the resident model)
  how close the armed set is to each.
* `PUT  …/probe-limits` — set this model's `max_armed` and its two budget halves (`{value, unit}` with
  unit `ms` or `pct` of the measured step time); `null` = the global default.
* `POST …/probe-capacity/measure` — measure the RESIDENT model's step time and dense cost line on its
  card(s), by UUID. ⚠ GPU work: run it in an approved window.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field

from millm.api.dependencies import DbSession, InferenceServiceDep, ModelServiceDep, ProbeRepo
from millm.api.schemas.common import ApiResponse
from millm.services import probe_limits

router = APIRouter(prefix="/api/models", tags=["models"])


class BudgetSetting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: Literal["ms", "pct"]


class ProbeLimitsRequest(BaseModel):
    """⚠ `extra="forbid"`: a misspelt field must refuse, not silently leave the default in force.

    039 (D13, TD39-31): `source: "recommended-approved"` APPROVES the recommendation whose
    `recommendation_digest` it names — the server recomputes it, refuses `RECOMMENDATION_CHANGED`
    on a mismatch, and writes exactly the recomputed values (any value fields sent are ignored).
    Anything else is an override, recorded `operator`.
    """

    model_config = ConfigDict(extra="forbid")

    max_armed: Optional[int] = Field(None, ge=1, le=64)
    max_behaviours: Optional[int] = Field(None, ge=1, le=64)
    decode_budget: Optional[BudgetSetting] = None
    prefill_budget: Optional[BudgetSetting] = None
    source: Optional[Literal["recommended-approved", "operator"]] = None
    recommendation_digest: Optional[str] = Field(None, pattern=r"^[0-9a-f]{64}$")
    set_by: Optional[str] = Field(None, max_length=64)
    reason: Optional[str] = Field(None, max_length=500)


@router.get("/{model_id}/probe-capacity", response_model=ApiResponse)
async def get_probe_capacity(
    model_id: int,
    db: DbSession,
    service: ModelServiceDep,
    repository: ProbeRepo,
    behaviour: Annotated[Optional[str], Query(max_length=64)] = None,
    version: Annotated[Optional[int], Query(ge=1)] = None,
    probe_ids: Annotated[Optional[list[str]], Query()] = None,
) -> ApiResponse:
    """The capacity picture; with `behaviour` (and `version`) or `probe_ids`, also the PREVIEW of
    arming that set now (039 FR-39.9) — the same projection the arm gate computes."""
    model = await service.get_model(model_id)
    data = await probe_limits.capacity(db, model)
    if behaviour is not None or probe_ids:
        from millm.services import probe_selection

        data["preview"] = await probe_limits.preview(
            db, model, repository, behaviour=behaviour,
            version=version if version is not None else probe_selection.UNSET,
            probe_ids=probe_ids,
        )
    return ApiResponse.ok(data)


@router.get("/{model_id}/probe-limits/recommendation", response_model=ApiResponse)
async def get_probe_limit_recommendation(model_id: int, db: DbSession,
                                         service: ModelServiceDep) -> ApiResponse:
    """This model's explained recommendation (039 FR-39.8, D13): values, reasons in words and
    numbers, the measurement it rests on, and the digest an approval must name."""
    from millm.core.config import settings
    from millm.services.probe_limit_recommendation import recommend, settings_snapshot

    model = await service.get_model(model_id)
    measurement = await probe_limits.latest_measurement(db, model)
    recommendation = recommend(model, measurement, settings_snapshot(settings))
    if recommendation is None:
        return ApiResponse.ok({
            "model_id": model_id, "values": None, "recommendation_digest": None,
            "reasons": ["This model has no capacity measurement; measure it first "
                        "(POST /api/models/{id}/probe-capacity/measure, a GPU action)."],
        })
    return ApiResponse.ok(recommendation)


@router.put("/{model_id}/probe-limits", response_model=ApiResponse)
async def put_probe_limits(model_id: int, request: ProbeLimitsRequest, db: DbSession,
                           service: ModelServiceDep) -> ApiResponse:
    model = await service.get_model(model_id)
    body: dict[str, Any] = request.model_dump()
    await probe_limits.set_limits(db, model_id, body, model_row=model)
    return ApiResponse.ok(await probe_limits.capacity(db, model))


@router.post("/{model_id}/probe-capacity/measure", response_model=ApiResponse)
async def measure_probe_capacity(model_id: int, db: DbSession, service: ModelServiceDep,
                                 inference: InferenceServiceDep) -> ApiResponse:
    from millm.services.probe_capacity_measure import measure_and_record

    model = await service.get_model(model_id)
    await measure_and_record(db, inference, model)
    return ApiResponse.ok(await probe_limits.capacity(db, model))
