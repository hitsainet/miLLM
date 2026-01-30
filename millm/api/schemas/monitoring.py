"""
Pydantic schemas for feature monitoring API endpoints.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ConfigureMonitoringRequest(BaseModel):
    """Request schema for configuring monitoring."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable monitoring",
    )
    features: list[int] | None = Field(
        default=None,
        description="Specific features to monitor (None = all)",
    )
    history_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Max entries in history buffer",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[int] | None) -> list[int] | None:
        """Validate all feature indices are non-negative."""
        if v is not None:
            for idx in v:
                if idx < 0:
                    raise ValueError(f"Feature index {idx} must be non-negative")
        return v


class ToggleMonitoringRequest(BaseModel):
    """Request schema for enabling/disabling monitoring."""

    enabled: bool = Field(..., description="Whether to enable monitoring")


class FeatureStatistics(BaseModel):
    """Statistics for a single feature."""

    feature_idx: int = Field(..., description="Feature index")
    count: int = Field(..., description="Number of observations")
    mean: float = Field(..., description="Mean activation value")
    std: float = Field(..., description="Standard deviation")
    min: float = Field(..., description="Minimum activation")
    max: float = Field(..., description="Maximum activation")
    active_ratio: float = Field(
        ...,
        description="Ratio of non-zero activations",
    )


class MonitoringState(BaseModel):
    """Response schema for monitoring state."""

    enabled: bool = Field(..., description="Whether monitoring is enabled")
    sae_attached: bool = Field(..., description="Whether an SAE is attached")
    sae_id: str | None = Field(default=None, description="Attached SAE ID")
    monitored_features: list[int] | None = Field(
        default=None,
        description="Features being monitored (None = all)",
    )
    history_size: int = Field(..., description="Max history entries")
    history_count: int = Field(..., description="Current history count")


class ActivationRecord(BaseModel):
    """Record of feature activations from a single forward pass."""

    timestamp: datetime = Field(..., description="When activations were captured")
    request_id: str | None = Field(
        default=None,
        description="Associated inference request ID",
    )
    token_position: int = Field(..., description="Token position in sequence")
    activations: dict[int, float] = Field(
        ...,
        description="Feature activations (feature_idx → value)",
    )
    top_k: list[tuple[int, float]] = Field(
        default_factory=list,
        description="Top-K most active features [(idx, value), ...]",
    )


class ActivationHistoryResponse(BaseModel):
    """Response schema for activation history."""

    records: list[ActivationRecord] = Field(
        ...,
        description="Activation records (newest first)",
    )
    total: int = Field(..., description="Total records returned")


class StatisticsResponse(BaseModel):
    """Response schema for feature statistics."""

    features: list[FeatureStatistics] = Field(
        ...,
        description="Statistics per feature",
    )
    total_activations: int = Field(
        ...,
        description="Total forward passes recorded",
    )
    since: datetime | None = Field(
        default=None,
        description="Statistics since this time",
    )


class ClearResponse(BaseModel):
    """Response schema for clear operations."""

    cleared: int = Field(..., description="Number of items cleared")
    message: str = Field(..., description="Status message")


class TopFeaturesRequest(BaseModel):
    """Request schema for getting top features."""

    k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top features to return",
    )
    metric: str = Field(
        default="mean",
        description="Metric to rank by (mean, max, active_ratio)",
    )

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate metric is supported."""
        valid = {"mean", "max", "active_ratio", "count"}
        if v not in valid:
            raise ValueError(f"Metric must be one of: {valid}")
        return v


class TopFeaturesResponse(BaseModel):
    """Response schema for top features."""

    features: list[FeatureStatistics] = Field(
        ...,
        description="Top features by metric",
    )
    metric: str = Field(..., description="Metric used for ranking")
    k: int = Field(..., description="Number of features requested")
