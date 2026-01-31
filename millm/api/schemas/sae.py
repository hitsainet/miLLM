"""
Pydantic schemas for SAE API endpoints.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from millm.db.models.sae import SAEStatus


class DownloadSAERequest(BaseModel):
    """Request schema for downloading an SAE."""

    repository_id: str = Field(
        ...,
        pattern=r"^[\w-]+/[\w.-]+$",
        max_length=255,
        description="HuggingFace repository ID (e.g., 'jbloom/gemma-2-2b-res-jb')",
        examples=["jbloom/gemma-2-2b-res-jb", "ckkissane/tinystories-1M-saes"],
    )
    revision: str = Field(
        default="main",
        max_length=100,
        description="Git revision (branch, tag, or commit hash)",
    )


class AttachSAERequest(BaseModel):
    """Request schema for attaching an SAE to a model."""

    layer: int = Field(
        ...,
        ge=0,
        description="Layer index to attach the SAE to (0-indexed)",
    )

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, v: int) -> int:
        """Validate layer is non-negative."""
        if v < 0:
            raise ValueError("layer must be non-negative")
        return v


class SteeringRequest(BaseModel):
    """Request schema for setting steering values."""

    feature_idx: int = Field(
        ...,
        ge=0,
        description="Feature index to steer",
    )
    value: float = Field(
        ...,
        description="Steering strength (positive=amplify, negative=suppress)",
    )


class SteeringBatchRequest(BaseModel):
    """Request schema for setting multiple steering values."""

    steering: dict[int, float] = Field(
        ...,
        description="Dictionary mapping feature indices to steering values",
    )

    @field_validator("steering")
    @classmethod
    def validate_steering(cls, v: dict[int, float]) -> dict[int, float]:
        """Validate all indices are non-negative."""
        for idx in v.keys():
            if idx < 0:
                raise ValueError(f"Feature index {idx} must be non-negative")
        return v


class MonitoringRequest(BaseModel):
    """Request schema for enabling monitoring."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable monitoring",
    )
    features: list[int] | None = Field(
        default=None,
        description="Specific features to monitor (None = all)",
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


class SAEMetadata(BaseModel):
    """Response schema for SAE metadata."""

    id: str = Field(..., description="Unique SAE identifier")
    repository_id: str = Field(..., description="HuggingFace repository ID")
    revision: str = Field(..., description="Git revision")
    name: str = Field(..., description="Display name")
    format: str = Field(..., description="SAE format (e.g., 'saelens')")
    d_in: int = Field(..., description="Input dimension (model hidden size)")
    d_sae: int = Field(..., description="SAE feature dimension")
    trained_on: str | None = Field(
        default=None,
        description="Model the SAE was trained on",
    )
    trained_layer: int | None = Field(
        default=None,
        description="Layer the SAE was trained for",
    )
    file_size_bytes: int | None = Field(
        default=None,
        description="Size on disk in bytes",
    )
    status: SAEStatus = Field(..., description="Current SAE status")
    error_message: str | None = Field(
        default=None,
        description="Error message if status is ERROR",
    )
    created_at: datetime = Field(..., description="When the SAE was added")
    updated_at: datetime = Field(..., description="When the SAE was last updated")

    model_config = {"from_attributes": True}

    @classmethod
    def from_sae(cls, sae: Any) -> "SAEMetadata":
        """
        Create a response from an ORM model.

        Args:
            sae: The SAE ORM instance.

        Returns:
            SAEMetadata populated from the ORM model.
        """
        return cls.model_validate(sae)


class AttachmentStatus(BaseModel):
    """Response schema for SAE attachment status."""

    is_attached: bool = Field(..., description="Whether an SAE is attached")
    sae_id: str | None = Field(
        default=None,
        description="ID of attached SAE",
    )
    layer: int | None = Field(
        default=None,
        description="Layer where SAE is attached",
    )
    memory_usage_mb: int | None = Field(
        default=None,
        description="GPU memory used by SAE in MB",
    )
    steering_enabled: bool = Field(
        default=False,
        description="Whether steering is enabled",
    )
    monitoring_enabled: bool = Field(
        default=False,
        description="Whether monitoring is enabled",
    )


class CompatibilityResult(BaseModel):
    """Response schema for SAE-model compatibility check."""

    compatible: bool = Field(..., description="Whether SAE is compatible with model")
    errors: list[str] = Field(
        default_factory=list,
        description="Compatibility errors (if any)",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Compatibility warnings",
    )


class SAEListResponse(BaseModel):
    """Response schema for SAE list endpoint."""

    saes: list[SAEMetadata] = Field(
        ...,
        description="List of SAEs",
    )
    total: int = Field(..., description="Total number of SAEs")
    attachment: AttachmentStatus = Field(
        ...,
        description="Current attachment status",
    )


class AttachResponse(BaseModel):
    """Response schema for SAE attach operation."""

    status: str = Field(..., description="Operation status")
    sae_id: str = Field(..., description="SAE ID")
    layer: int = Field(..., description="Attached layer")
    memory_usage_mb: int = Field(..., description="GPU memory used in MB")
    warnings: list[str] = Field(
        default_factory=list,
        description="Compatibility warnings",
    )


class DetachResponse(BaseModel):
    """Response schema for SAE detach operation."""

    status: str = Field(..., description="Operation status")
    sae_id: str = Field(..., description="SAE ID")
    memory_freed_mb: int = Field(..., description="GPU memory freed in MB")


class DeleteResponse(BaseModel):
    """Response schema for SAE delete operation."""

    status: str = Field(..., description="Operation status")
    sae_id: str = Field(..., description="SAE ID")
    freed_disk_mb: float = Field(..., description="Disk space freed in MB")


class SteeringStatus(BaseModel):
    """Response schema for steering status."""

    enabled: bool = Field(..., description="Whether steering is enabled")
    values: dict[int, float] = Field(
        default_factory=dict,
        description="Current steering values (feature_idx -> value)",
    )


class DownloadResponse(BaseModel):
    """Response schema for SAE download initiation."""

    sae_id: str = Field(..., description="SAE ID")
    status: str = Field(
        default="downloading",
        description="Download status",
    )
    message: str = Field(
        default="Download started",
        description="Status message",
    )


class PreviewSAERequest(BaseModel):
    """Request schema for previewing an SAE repository."""

    repository_id: str = Field(
        ...,
        pattern=r"^[\w-]+/[\w.-]+$",
        max_length=255,
        description="HuggingFace repository ID (e.g., 'google/gemma-scope-2b-pt-res')",
        examples=["google/gemma-scope-2b-pt-res", "jbloom/gemma-2-2b-res-jb"],
    )
    revision: str = Field(
        default="main",
        max_length=100,
        description="Git revision (branch, tag, or commit hash)",
    )
    hf_token: str | None = Field(
        default=None,
        max_length=255,
        description="HuggingFace access token for gated repositories",
    )


class SAEFileInfo(BaseModel):
    """Information about a single SAE file in a repository."""

    path: str = Field(..., description="File path within the repository")
    size_bytes: int = Field(default=0, description="File size in bytes")
    layer: int | None = Field(default=None, description="Layer number extracted from path")
    width: str | None = Field(default=None, description="SAE width (e.g., '16k')")


class PreviewSAEResponse(BaseModel):
    """Response schema for SAE repository preview."""

    repository_id: str = Field(..., description="HuggingFace repository ID")
    revision: str = Field(..., description="Git revision")
    model_id: str | None = Field(
        default=None,
        description="Model ID extracted from repository name",
    )
    files: list[SAEFileInfo] = Field(
        default_factory=list,
        description="List of SAE files in the repository",
    )
    total_files: int = Field(..., description="Total number of SAE files found")
