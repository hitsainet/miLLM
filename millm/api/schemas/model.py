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
    revision: str | None = Field(
        default=None,
        max_length=100,
        description="Git revision (branch, tag, or commit hash) to download",
    )
    gguf_files: list[str] | None = Field(
        default=None,
        max_length=64,
        description=(
            "Exact repo-relative paths of the ONE quantization to download. "
            "Omit for an ordinary model, which downloads the whole repo. Pass "
            "EVERY file of a split quantization — a subset yields a directory "
            "that looks complete and a model that cannot load."
        ),
        examples=[["Qwen2.5-7B-Instruct-Q4_K_M.gguf"]],
    )
    gguf_label: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "The quantization's label (Q4_K_M, IQ4_XS). Distinguishes downloads "
            "that share a coarse QuantizationType — Q4_K_M, Q4_K_S and Q4_0 are "
            "all 'Q4' — which would otherwise collide on one cache directory."
        ),
    )

    @field_validator("gguf_files")
    @classmethod
    def validate_gguf_files(cls, v: list[str] | None) -> list[str] | None:
        """Reject traversal sequences and non-GGUF paths.

        These become `allow_patterns` for snapshot_download. A '..' component
        could widen the pattern to match files the caller never chose. Repo
        files are remote, so this is input integrity rather than a local
        filesystem risk — the same reasoning as the SAE download path, whose
        validator this mirrors.

        An EMPTY list is rejected rather than accepted: an empty allow list
        matches nothing, and the download would report success over an empty
        directory. Absent means "whole repo"; present means "these files".
        """
        if v is None:
            return v
        if not v:
            raise ValueError(
                "gguf_files may not be empty — omit it to download the whole repository"
            )
        from pathlib import PurePosixPath

        for path in v:
            if path.startswith("/") or ".." in PurePosixPath(path).parts:
                raise ValueError(
                    "gguf_files entries must be relative paths with no '..' components"
                )
            if not path.lower().endswith(".gguf"):
                raise ValueError(f"gguf_files entries must be .gguf files, got: {path}")
        return v

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
        """Validate that local_path is an absolute path outside system directories."""
        if v is None:
            return v
        if not v.startswith("/"):
            raise ValueError("local_path must be an absolute path")
        # Normalise to collapse any .. segments before the prefix check.
        from pathlib import Path
        try:
            normalised = str(Path(v).resolve())
        except (ValueError, OSError):
            raise ValueError("local_path contains invalid characters")
        # Block directories that can never contain model files and whose
        # existence could be probed via error messages.
        _BLOCKED = ("/proc", "/sys", "/dev", "/run", "/boot", "/etc", "/var/run", "/tmp")
        for blocked in _BLOCKED:
            if normalised == blocked or normalised.startswith(blocked + "/"):
                raise ValueError(
                    f"local_path may not point to system directory: {blocked}"
                )
        return v


class ModelPreviewRequest(BaseModel):
    """Request schema for previewing a model before download."""

    repo_id: str = Field(
        ...,
        pattern=r"^[\w-]+/[\w.-]+$",
        description="HuggingFace repository ID",
    )
    revision: str | None = Field(
        default=None,
        max_length=100,
        description="Git revision (branch, tag, or commit hash) to inspect",
    )
    hf_token: Annotated[str | None, Field(exclude=True)] = Field(
        default=None,
        description="HuggingFace access token for gated models",
    )


class SizeEstimate(BaseModel):
    """Estimated size for a model at different quantization levels."""

    disk_mb: int = Field(..., description="Estimated disk size in MB")
    memory_mb: int = Field(..., description="Estimated VRAM requirement in MB")


class GGUFFileInfo(BaseModel):
    """One `.gguf` file within a quantization."""

    path: str = Field(..., description="Repo-relative path")
    size_bytes: int = Field(..., description="Exact size, from HuggingFace file metadata")


class GGUFQuantInfo(BaseModel):
    """One selectable GGUF quantization.

    The unit here is a QUANTIZATION, not a file. On large models a quant is
    split into numbered parts inside its own directory, and downloading one part
    produces a directory that looks populated and a model that cannot load. Every
    quant on a 7B repo happens to be a single file, which is why this distinction
    is easy to miss and expensive to get wrong.

    Sizes are MEASURED, never estimated from a parameter count — for a
    mixed-precision quant a bytes-per-parameter figure means nothing.
    """

    label: str = Field(..., description="Quantization label, e.g. Q4_K_M, IQ4_XS, F16")
    files: list[GGUFFileInfo] = Field(..., description="Every file this quant needs")
    total_size_bytes: int = Field(..., description="Sum of the files' true sizes")
    is_split: bool = Field(
        default=False, description="Whether this quant is split across several files"
    )
    quant_parsed: bool = Field(
        default=True,
        description=(
            "False when the label could not be read from the filename and is the "
            "filename itself. The file is still selectable; it is just not named "
            "by a recognised quantization token."
        ),
    )


class ModelPreviewResponse(BaseModel):
    """Response schema for model preview."""

    name: str = Field(..., description="Model name")
    params: str | None = Field(default=None, description="Parameter count (e.g., '2.5B')")
    architecture: str | None = Field(default=None, description="Model architecture / pipeline tag")
    requires_trust_remote_code: bool = Field(
        default=False,
        description="Whether the model requires trust_remote_code",
    )
    is_gated: bool = Field(
        default=False,
        description="Whether the model is gated and requires authentication",
    )
    estimated_sizes: dict[str, SizeEstimate] | None = Field(
        default=None,
        description="Estimated sizes for each quantization level",
    )
    downloads: int = Field(default=0, description="Total downloads from HuggingFace")
    likes: int = Field(default=0, description="Total likes on HuggingFace")
    tags: list[str] | None = Field(default=None, description="Model tags from HuggingFace")
    pipeline_tag: str | None = Field(default=None, description="Pipeline tag (e.g., text-generation)")
    model_type: str | None = Field(default=None, description="Model type from config (e.g., llama)")
    architectures: list[str] | None = Field(
        default=None, description="Model architectures from config (e.g., ['LlamaForCausalLM'])"
    )
    license: str | None = Field(default=None, description="Model license")
    language: str | list[str] | None = Field(default=None, description="Model language(s)")
    revision: str | None = Field(
        default=None,
        description="Resolved commit the listing and sizes were measured against",
    )
    gguf_quants: list[GGUFQuantInfo] | None = Field(
        default=None,
        description=(
            "GGUF quantizations offered by this repo, smallest first. None or "
            "empty for an ordinary safetensors repo — which is the signal the UI "
            "branches on."
        ),
    )
    gguf_companions: list[GGUFFileInfo] | None = Field(
        default=None,
        description=(
            "Sidecar .gguf files that are NOT servable models — chiefly the "
            "multimodal projector (mmproj) a VLM needs to see images. Reported "
            "so the information is not lost: a quantization downloaded without "
            "its projector loads as a silently text-only model. Which projector "
            "precision to pair with a given quant is a serving decision."
        ),
    )
    gguf_architecture: str | None = Field(
        default=None, description="Architecture from HuggingFace's GGUF metadata, when indexed"
    )
    gguf_context_length: int | None = Field(
        default=None, description="Context length from HuggingFace's GGUF metadata, when indexed"
    )
    gguf_total_params: int | None = Field(
        default=None, description="Parameter count from HuggingFace's GGUF metadata, when indexed"
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
    locked: bool = Field(
        default=False,
        description="Whether the model is locked for steering (prevents auto-unload)",
    )
    download_progress: int | None = Field(
        default=None,
        description="Download progress percentage (0-100) when status is DOWNLOADING",
    )
    created_at: datetime = Field(..., description="When the model was added")
    updated_at: datetime = Field(..., description="When the model was last updated")
    loaded_at: datetime | None = Field(
        default=None,
        description="When the model was loaded into memory",
    )
    # Runtime properties (only populated when model is loaded)
    num_parameters: int | None = Field(
        default=None,
        description="Number of model parameters (only available when loaded)",
    )
    memory_footprint: int | None = Field(
        default=None,
        description="Actual memory usage in bytes (only available when loaded)",
    )
    device: str | None = Field(
        default=None,
        description="Device the model is loaded on (only available when loaded)",
    )
    dtype: str | None = Field(
        default=None,
        description="Data type of model weights (only available when loaded)",
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
