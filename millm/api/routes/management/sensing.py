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
    SensingConfigRequest,
    SensingConfigResult,
    SensingEventListResponse,
    SensingEventResponse,
    SensingStatusResponse,
    SensingToggleResult,
)
from millm.core.errors import (
    ProfileNotFoundError,
    SensingEventNotFoundError,
    ValidationError,
)

# Read-path retention runs at most this often (011 R3: pruning on EVERY
# list request made reads DB writers racing the flush-path prune).
_PRUNE_ON_READ_INTERVAL_S = 600.0
_last_read_prune: list[float] = [0.0]
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
    """Armed state, threshold mode, overhead, retention — PLUS the
    persistent per-cluster intent (sensing_enabled columns), reported
    distinctly from the runtime armed state (FTID pitfall 8): enabled but
    not armed answers 'why no events?' from this one endpoint."""
    from millm.db.base import async_session_factory
    from millm.db.models.profile import Profile
    from sqlalchemy import select

    service = _sensing_service()
    status = service.status()
    async with async_session_factory() as session:
        result = await session.execute(
            select(Profile.id, Profile.name, Profile.is_active)
            .where(Profile.sensing_enabled == True)  # noqa: E712
        )
        status["enabled_clusters"] = [
            {"id": row.id, "name": row.name, "is_active": row.is_active}
            for row in result
        ]
    return ApiResponse.ok(SensingStatusResponse(**status))


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

    import time

    from millm.core.config import settings

    async with async_session_factory() as session:
        repo = SensingRepository(session)
        # Retention on READ as well as on flush (FPRD SEN-P2): the age cap
        # is the documented privacy control and must hold for idle clusters
        # that never flush again. Throttled — reads must not become writers
        # on every request (011 R3).
        now = time.monotonic()
        if now - _last_read_prune[0] >= _PRUNE_ON_READ_INTERVAL_S:
            _last_read_prune[0] = now
            pruned = await repo.prune_aged(settings.SENSING_MAX_AGE_DAYS)
            if pruned:
                await session.commit()
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
        # 404, not 422: a pruned event is EXPECTED under retention — clients
        # must be able to branch on not-found (011 R1).
        raise SensingEventNotFoundError(
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
            try:
                service.arm_for_profile(profile, sae)
            except ValueError as exc:
                # Column stays enabled (persistent intent) but arming
                # refused — tell the caller exactly why (mismatched SAE,
                # unusable thresholds) instead of a silent no-arm.
                raise ValidationError(
                    f"Sensing enabled but could not arm: {exc}",
                    details={"profile_id": profile_id},
                ) from exc
        else:
            service.disarm(sae)

    return SensingToggleResult(
        profile_id=profile_id,
        sensing_enabled=enabled,
        armed=service.is_armed,
    )


@router.put(
    "/{profile_id}/config",
    response_model=ApiResponse[SensingConfigResult],
    summary="Adjust sensing runtime overrides (quorum etc.) for a cluster",
)
async def set_sensing_config(
    profile_id: ProfileId, request: SensingConfigRequest
) -> ApiResponse[SensingConfigResult]:
    """Persist miLLM-local sensing overrides (stored OUTSIDE the portable
    document — export stays lossless) and live re-arm when this cluster is
    the armed one. min_k=null clears the override back to the default
    (ALL sensable members)."""
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
                details={"profile_id": profile_id},
            )
        member_count = len((profile.cluster_meta or {}).get("members", []))
        # Validate against the SENSABLE ceiling, not the raw member count:
        # members with infinite thresholds can never fire, so a quorum above
        # the sensable count makes events permanently unreachable while
        # status looks healthy (R1 find).
        try:
            probe = service.build_config(profile)
            sensable = int(sum(1 for theta in probe.thresholds.tolist()
                               if theta != float("inf")))
        except ValueError:
            sensable = member_count
        if request.min_k is not None and not 1 <= request.min_k <= sensable:
            raise ValidationError(
                f"min_k must be between 1 and {sensable} — this cluster has "
                f"{member_count} members but only {sensable} carry usable "
                f"thresholds and can fire",
                details={"min_k": request.min_k,
                         "member_count": member_count,
                         "sensable_count": sensable},
            )
        meta = dict(profile.cluster_meta or {})
        local = dict(meta.get("sensing_overrides", {}) or {})
        if request.min_k is None:
            local.pop("min_k", None)
        else:
            local["min_k"] = int(request.min_k)
        if local:
            meta["sensing_overrides"] = local
        else:
            meta.pop("sensing_overrides", None)
        profile.cluster_meta = meta
        is_active_enabled = bool(profile.is_active) and bool(
            profile.sensing_enabled)
        await session.commit()

    # Live re-arm so the new quorum applies immediately
    effective_min_k = None
    if is_active_enabled and service.armed_profile_id == profile_id:
        sae = AttachedSAEState().attached_sae
        if sae is not None:
            try:
                service.arm_for_profile(profile, sae)
            except ValueError as exc:
                raise ValidationError(
                    f"Override saved but re-arm failed: {exc}",
                    details={"profile_id": profile_id},
                ) from exc
    try:
        effective_min_k = service.build_config(profile).min_k
    except ValueError:
        effective_min_k = None

    return ApiResponse.ok(SensingConfigResult(
        profile_id=profile_id,
        min_k=request.min_k,
        effective_min_k=effective_min_k,
        armed=service.is_armed,
    ))


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
