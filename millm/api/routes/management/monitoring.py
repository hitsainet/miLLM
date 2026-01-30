"""
Feature monitoring API endpoints.

Provides endpoints for configuring monitoring, retrieving activation history,
and viewing feature statistics.
"""

from typing import Annotated, Optional

from fastapi import APIRouter, Path, Query

from millm.api.dependencies import MonitoringServiceDep
from millm.api.schemas.common import ApiResponse
from millm.api.schemas.monitoring import (
    ActivationHistoryResponse,
    ActivationRecord,
    ClearResponse,
    ConfigureMonitoringRequest,
    FeatureStatistics,
    MonitoringState,
    StatisticsResponse,
    ToggleMonitoringRequest,
    TopFeaturesRequest,
    TopFeaturesResponse,
)

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


# Type alias for feature index path parameter
FeatureIdx = Annotated[int, Path(description="Feature index", ge=0)]


@router.get(
    "",
    response_model=ApiResponse[MonitoringState],
    summary="Get monitoring status",
    description="Get current monitoring configuration and state.",
)
async def get_monitoring_state(
    service: MonitoringServiceDep,
) -> ApiResponse[MonitoringState]:
    """
    Get current monitoring state.

    Returns whether monitoring is enabled, which features are monitored,
    and history buffer status.
    """
    state = service.get_state()
    return ApiResponse.ok(MonitoringState(
        enabled=state["enabled"],
        sae_attached=state["sae_attached"],
        sae_id=state["sae_id"],
        monitored_features=state["monitored_features"],
        history_size=state["history_size"],
        history_count=state["history_count"],
    ))


@router.post(
    "/configure",
    response_model=ApiResponse[MonitoringState],
    summary="Configure monitoring",
    description="Configure monitoring parameters including history size and features.",
)
async def configure_monitoring(
    request: ConfigureMonitoringRequest,
    service: MonitoringServiceDep,
) -> ApiResponse[MonitoringState]:
    """
    Configure monitoring parameters.

    Allows setting which features to monitor, history buffer size,
    and enabling/disabling monitoring.
    """
    service.configure(
        enabled=request.enabled,
        features=request.features,
        history_size=request.history_size,
    )

    state = service.get_state()
    return ApiResponse.ok(MonitoringState(
        enabled=state["enabled"],
        sae_attached=state["sae_attached"],
        sae_id=state["sae_id"],
        monitored_features=state["monitored_features"],
        history_size=state["history_size"],
        history_count=state["history_count"],
    ))


@router.post(
    "/enable",
    response_model=ApiResponse[MonitoringState],
    summary="Enable/disable monitoring",
    description="Enable or disable monitoring without changing other settings.",
)
async def toggle_monitoring(
    request: ToggleMonitoringRequest,
    service: MonitoringServiceDep,
) -> ApiResponse[MonitoringState]:
    """
    Enable or disable monitoring.

    When disabled, no activations are captured. Configuration is preserved.
    """
    service.set_enabled(request.enabled)

    state = service.get_state()
    return ApiResponse.ok(MonitoringState(
        enabled=state["enabled"],
        sae_attached=state["sae_attached"],
        sae_id=state["sae_id"],
        monitored_features=state["monitored_features"],
        history_size=state["history_size"],
        history_count=state["history_count"],
    ))


@router.get(
    "/history",
    response_model=ApiResponse[ActivationHistoryResponse],
    summary="Get activation history",
    description="Get recent activation records from the history buffer.",
)
async def get_activation_history(
    service: MonitoringServiceDep,
    limit: int = Query(default=50, ge=1, le=1000, description="Max records to return"),
    request_id: Optional[str] = Query(default=None, description="Filter by request ID"),
) -> ApiResponse[ActivationHistoryResponse]:
    """
    Get activation history.

    Returns recent activation records, optionally filtered by request ID.
    Records are returned newest first.
    """
    records = service.get_history(limit=limit, request_id=request_id)

    return ApiResponse.ok(ActivationHistoryResponse(
        records=[
            ActivationRecord(
                timestamp=r["timestamp"],
                request_id=r["request_id"],
                token_position=r["token_position"],
                activations=r["activations"],
                top_k=r["top_k"],
            )
            for r in records
        ],
        total=len(records),
    ))


@router.delete(
    "/history",
    response_model=ApiResponse[ClearResponse],
    summary="Clear activation history",
    description="Clear all activation records from history buffer.",
)
async def clear_history(
    service: MonitoringServiceDep,
) -> ApiResponse[ClearResponse]:
    """
    Clear activation history.

    Removes all records from the history buffer.
    """
    count = service.clear_history()
    return ApiResponse.ok(ClearResponse(
        cleared=count,
        message=f"Cleared {count} activation records",
    ))


@router.get(
    "/statistics",
    response_model=ApiResponse[StatisticsResponse],
    summary="Get feature statistics",
    description="Get running statistics for monitored features.",
)
async def get_statistics(
    service: MonitoringServiceDep,
    features: Optional[str] = Query(
        default=None,
        description="Comma-separated feature indices to get stats for",
    ),
) -> ApiResponse[StatisticsResponse]:
    """
    Get feature statistics.

    Returns running statistics (mean, std, min, max, active_ratio)
    for monitored features.
    """
    # Parse feature indices if provided
    feature_indices = None
    if features:
        feature_indices = [int(f.strip()) for f in features.split(",")]

    stats = service.get_statistics(feature_indices=feature_indices)

    return ApiResponse.ok(StatisticsResponse(
        features=[
            FeatureStatistics(
                feature_idx=f["feature_idx"],
                count=f["count"],
                mean=f["mean"],
                std=f["std"],
                min=f["min"],
                max=f["max"],
                active_ratio=f["active_ratio"],
            )
            for f in stats["features"]
        ],
        total_activations=stats["total_activations"],
        since=stats["since"],
    ))


@router.delete(
    "/statistics",
    response_model=ApiResponse[ClearResponse],
    summary="Reset feature statistics",
    description="Reset all feature statistics.",
)
async def reset_statistics(
    service: MonitoringServiceDep,
) -> ApiResponse[ClearResponse]:
    """
    Reset feature statistics.

    Clears all running statistics and resets the start time.
    """
    count = service.reset_statistics()
    return ApiResponse.ok(ClearResponse(
        cleared=count,
        message=f"Reset statistics for {count} features",
    ))


@router.post(
    "/statistics/top",
    response_model=ApiResponse[TopFeaturesResponse],
    summary="Get top features",
    description="Get top features ranked by a metric.",
)
async def get_top_features(
    request: TopFeaturesRequest,
    service: MonitoringServiceDep,
) -> ApiResponse[TopFeaturesResponse]:
    """
    Get top features by metric.

    Returns the top K features ranked by mean, max, active_ratio, or count.
    """
    features = service.get_top_features(k=request.k, metric=request.metric)

    return ApiResponse.ok(TopFeaturesResponse(
        features=[
            FeatureStatistics(
                feature_idx=f["feature_idx"],
                count=f["count"],
                mean=f["mean"],
                std=f["std"],
                min=f["min"],
                max=f["max"],
                active_ratio=f["active_ratio"],
            )
            for f in features
        ],
        metric=request.metric,
        k=request.k,
    ))
