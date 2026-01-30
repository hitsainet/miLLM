"""
Pydantic schemas for Model API endpoints.
"""

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from millm.db.models.model import ModelSource, ModelStatus, QuantizationType


class ModelDownloadRequest(BaseModel):
    """Request schema for downloading a model."""

    source: ModelSource = Field(
        ...,
        description="Source of the model (huggingface or local)",
    )
    repo_id: str | None = Field(
        default=None,
        pattern=r"^[\w-]+/[\w.-]+$",
        max_length=255,
        description="HuggingFace repository ID (e.g., 'google/gemma-2-2b')",
        examples=["google/gemma-2-2b", "meta-llama/Llama-3.2-3B"],
    )
    local_path: str | None = Field(
        default=None,
        max_length=500,
        description="Path to local model directory",
    )
    quantization: QuantizationType = Field(
        default=QuantizationType.Q4,
        description="Quantization level for the model",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code (required for some models)",
    )
    hf_token: Annotated[str | None, Field(exclude=True)] = Field(
        default=None,
        description="HuggingFace access token for gated models (never logged)",
    )
    custom_name: str | None = Field(
        default=None,
        max_length=100,
        description="Optional custom display name for the model",
    )

    @model_validator(mode="after")
    def validate_source_fields(self) -> "ModelDownloadRequest":
        """Validate that required fields are present based on source."""
        if self.source == ModelSource.HUGGINGFACE and not self.repo_id:
            raise ValueError("repo_id is required for HuggingFace source")
        if self.source == ModelSource.LOCAL and not self.local_path:
            raise ValueError("local_path is required for local source")
        return self

    @field_validator("local_path")
    @classmethod
    def validate_local_path(cls, v: str | None) -> str | None:
        """Validate that local_path is absolute."""
        if v is None:
            return v
        if not v.startswith("/"):
            raise ValueError("local_path must be an absolute path")
        return v


class ModelPreviewRequest(BaseModel):
    """Request schema for previewing a model before download."""

    repo_id: str = Field(
        ...,
        pattern=r"^[\w-]+/[\w.-]+$",
        description="HuggingFace repository ID",
    )
    hf_token: Annotated[str | None, Field(exclude=True)] = Field(
        default=None,
        description="HuggingFace access token for gated models",
    )


class SizeEstimate(BaseModel):
    """Estimated size for a model at different quantization levels."""

    disk_mb: int = Field(..., description="Estimated disk size in MB")
    memory_mb: int = Field(..., description="Estimated VRAM requirement in MB")


class ModelPreviewResponse(BaseModel):
    """Response schema for model preview."""

    name: str = Field(..., description="Model name")
    params: str = Field(..., description="Parameter count (e.g., '2.5B')")
    architecture: str = Field(..., description="Model architecture")
    requires_trust_remote_code: bool = Field(
        ...,
        description="Whether the model requires trust_remote_code",
    )
    is_gated: bool = Field(
        ...,
        description="Whether the model is gated and requires authentication",
    )
    estimated_sizes: dict[str, SizeEstimate] = Field(
        ...,
        description="Estimated sizes for each quantization level (Q4, Q8, FP16)",
    )


class ModelResponse(BaseModel):
    """Response schema for a single model."""

    id: int = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Display name for the model")
    source: ModelSource = Field(..., description="Source of the model")
    repo_id: str | None = Field(
        default=None,
        description="HuggingFace repository ID",
    )
    local_path: str | None = Field(
        default=None,
        description="Path to local model directory",
    )
    params: str | None = Field(
        default=None,
        description="Parameter count (e.g., '2.5B')",
    )
    architecture: str | None = Field(
        default=None,
        description="Model architecture",
    )
    quantization: QuantizationType = Field(
        ...,
        description="Quantization level",
    )
    disk_size_mb: int | None = Field(
        default=None,
        description="Size on disk in MB",
    )
    estimated_memory_mb: int | None = Field(
        default=None,
        description="Estimated VRAM requirement in MB",
    )
    status: ModelStatus = Field(..., description="Current model status")
    error_message: str | None = Field(
        default=None,
        description="Error message if status is ERROR",
    )
    created_at: datetime = Field(..., description="When the model was added")
    updated_at: datetime = Field(..., description="When the model was last updated")
    loaded_at: datetime | None = Field(
        default=None,
        description="When the model was loaded into memory",
    )

    model_config = {"from_attributes": True}

    @classmethod
    def from_model(cls, model: Any) -> "ModelResponse":
        """
        Create a response from an ORM model.

        Args:
            model: The Model ORM instance.

        Returns:
            ModelResponse populated from the ORM model.
        """
        return cls.model_validate(model)


class ModelListResponse(BaseModel):
    """Response schema for model list endpoint."""

    models: list[ModelResponse] = Field(
        ...,
        description="List of models",
    )
    total: int = Field(..., description="Total number of models")
    loaded_model_id: int | None = Field(
        default=None,
        description="ID of the currently loaded model (if any)",
    )
