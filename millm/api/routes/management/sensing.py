"""
Sensing management API endpoints (Feature 11).

Status (armed state + overhead), bounded event history, per-cluster
enable/disable (persists the column AND live-arms/disarms when that cluster
is active), and event clearing.
"""

from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Path, Query

from millm.api.schemas.common import ApiResponse
from millm.api.schemas.sensing import (
    SensingEventListResponse,
    SensingEventResponse,
    SensingStatusResponse,
    SensingToggleResult,
)
from millm.core.errors import ProfileNotFoundError, ValidationError
from millm.core.logging import get_logger

router = APIRouter(prefix="/api/sensing", tags=["sensing"])
logger = get_logger(__name__)

ProfileId = Annotated[str, Path(description="Cluster profile ID")]


def _sensing_service():
    from millm.api.dependencies import get_sensing_service

    return get_sensing_service()


@router.get(
    "/status",
    response_model=ApiResponse[SensingStatusResponse],
    summary="Sensing runtime status",
)
async def sensing_status() -> ApiResponse[SensingStatusResponse]:
    """Armed state, threshold mode, overhead accumulator, retention limits."""
    service = _sensing_service()
    return ApiResponse.ok(SensingStatusResponse(**service.status()))


@router.get(
    "/events",
    response_model=ApiResponse[SensingEventListResponse],
    summary="List co-activation events (newest first)",
)
async def list_events(
    profile_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    since: Optional[datetime] = Query(default=None),
) -> ApiResponse[SensingEventListResponse]:
    from millm.db.base import async_session_factory
    from millm.db.repositories.sensing_repository import SensingRepository

    async with async_session_factory() as session:
        repo = SensingRepository(session)
        events = await repo.list_events(
            profile_id=profile_id, limit=limit, since=since
        )
        total = await repo.count(profile_id=profile_id)
    return ApiResponse.ok(SensingEventListResponse(
        events=[SensingEventResponse.model_validate(e) for e in events],
        total=total,
    ))


@router.get(
    "/events/{event_id}",
    response_model=ApiResponse[SensingEventResponse],
    summary="Event detail (includes context)",
)
async def get_event(event_id: int) -> ApiResponse[SensingEventResponse]:
    from millm.db.base import async_session_factory
    from millm.db.repositories.sensing_repository import SensingRepository

    async with async_session_factory() as session:
        repo = SensingRepository(session)
        event = await repo.get(event_id)
    if event is None:
        raise ValidationError(
            f"Sensing event {event_id} not found",
            details={"event_id": event_id},
        )
    return ApiResponse.ok(SensingEventResponse.model_validate(event))


@router.delete(
    "/events",
    response_model=ApiResponse[dict],
    summary="Clear events (all, or one cluster's)",
)
async def clear_events(
    profile_id: Optional[str] = Query(default=None),
) -> ApiResponse[dict]:
    from millm.db.base import async_session_factory
    from millm.db.repositories.sensing_repository import SensingRepository

    async with async_session_factory() as session:
        repo = SensingRepository(session)
        deleted = await repo.clear(profile_id=profile_id)
        await session.commit()
    return ApiResponse.ok({"deleted": deleted})


async def _toggle(profile_id: str, enabled: bool) -> SensingToggleResult:
    """Persist the sensing_enabled column and live-arm/disarm when the
    toggled cluster is the active one."""
    from millm.db.base import async_session_factory
    from millm.db.repositories.profile_repository import ProfileRepository
    from millm.services.sae_service import AttachedSAEState

    service = _sensing_service()
    async with async_session_factory() as session:
        repo = ProfileRepository(session)
        profile = await repo.get(profile_id)
        if profile is None:
            raise ProfileNotFoundError(
                f"Profile '{profile_id}' not found",
                details={"profile_id": profile_id},
            )
        if getattr(profile, "source_kind", None) != "cluster":
            raise ValidationError(
                "Sensing applies to imported clusters only",
                details={"profile_id": profile_id,
                         "source_kind": profile.source_kind},
            )
        profile.sensing_enabled = enabled
        is_active = bool(profile.is_active)
        await session.commit()

    sae = AttachedSAEState().attached_sae
    if is_active:
        if enabled and sae is not None:
            service.arm_for_profile(profile, sae)
        else:
            service.disarm(sae)

    return SensingToggleResult(
        profile_id=profile_id,
        sensing_enabled=enabled,
        armed=service.is_armed,
    )


@router.post(
    "/{profile_id}/enable",
    response_model=ApiResponse[SensingToggleResult],
    summary="Enable sensing for a cluster",
)
async def enable_sensing(profile_id: ProfileId) -> ApiResponse[SensingToggleResult]:
    return ApiResponse.ok(await _toggle(profile_id, True))


@router.post(
    "/{profile_id}/disable",
    response_model=ApiResponse[SensingToggleResult],
    summary="Disable sensing for a cluster",
)
async def disable_sensing(profile_id: ProfileId) -> ApiResponse[SensingToggleResult]:
    return ApiResponse.ok(await _toggle(profile_id, False))
