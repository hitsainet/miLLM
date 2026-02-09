"""
Model ORM class for downloaded/local LLM models.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base


class ModelStatus(str, enum.Enum):
    """Status of a model in the system."""

    DOWNLOADING = "downloading"
    READY = "ready"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class ModelSource(str, enum.Enum):
    """Source of the model."""

    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class QuantizationType(str, enum.Enum):
    """Quantization level for the model."""

    FP32 = "FP32"
    FP16 = "FP16"
    Q8 = "Q8"
    Q4 = "Q4"
    Q2 = "Q2"


class Model(Base):
    """
    ORM model representing a downloaded or local LLM model.

    Attributes:
        id: Primary key
        name: Display name for the model
        source: Where the model came from (huggingface/local)
        repo_id: HuggingFace repository ID (e.g., "google/gemma-2-2b")
        local_path: Path to local model directory
        params: Parameter count string (e.g., "2.5B", "350M")
        architecture: Model architecture (e.g., "causal-lm")
        quantization: Quantization level (Q4, Q8, FP16)
        disk_size_mb: Size on disk in megabytes
        estimated_memory_mb: Estimated VRAM needed
        cache_path: Path to cached model files
        config_json: Model configuration from config.json
        trust_remote_code: Whether model requires trust_remote_code
        status: Current status of the model
        error_message: Error message if status is ERROR
        created_at: When the model was added
        updated_at: When the model was last updated
        loaded_at: When the model was loaded into memory
    """

    __tablename__ = "models"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Core fields
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[ModelSource] = mapped_column(
        Enum(
            ModelSource,
            name="modelsource",
            create_constraint=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
    )
    repo_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    local_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Model metadata
    params: Mapped[str | None] = mapped_column(String(50), nullable=True)
    architecture: Mapped[str | None] = mapped_column(String(100), nullable=True)
    quantization: Mapped[QuantizationType] = mapped_column(
        Enum(
            QuantizationType,
            name="quantizationtype",
            create_constraint=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
    )

    # Storage
    disk_size_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_memory_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Configuration
    config_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    trust_remote_code: Mapped[bool] = mapped_column(Boolean, default=False)

    # State
    status: Mapped[ModelStatus] = mapped_column(
        Enum(
            ModelStatus,
            name="modelstatus",
            create_constraint=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        default=ModelStatus.READY,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lock state (for steering)
    locked: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    loaded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Constraints
    __table_args__ = (
        UniqueConstraint("repo_id", "quantization", name="uq_repo_quantization"),
        UniqueConstraint("local_path", name="uq_local_path"),
    )

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name='{self.name}', status={self.status.value})>"
