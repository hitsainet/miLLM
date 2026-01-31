"""
SAE ORM classes for Sparse Autoencoder management.

Tracks downloaded SAEs and their attachment to models.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    BigInteger,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from millm.db.base import Base


class SAEStatus(str, enum.Enum):
    """Status of an SAE in the system."""

    DOWNLOADING = "downloading"
    CACHED = "cached"
    ATTACHED = "attached"
    ERROR = "error"


class SAE(Base):
    """
    ORM model representing a downloaded SAE (Sparse Autoencoder).

    Status flow:
    - downloading: Download in progress
    - cached: Downloaded and ready to attach
    - attached: Currently attached to model
    - error: Download or validation failed

    Attributes:
        id: Primary key (generated from repository_id)
        repository_id: HuggingFace repository ID
        revision: Git revision (branch, tag, commit)
        name: Display name for the SAE
        format: SAE format (e.g., "saelens")
        d_in: Input dimension (must match model hidden_size)
        d_sae: SAE feature dimension
        trained_on: Model the SAE was trained on
        trained_layer: Layer the SAE was trained for
        file_size_bytes: Size on disk in bytes
        cache_path: Path to cached SAE files
        status: Current status of the SAE
        error_message: Error message if status is ERROR
        created_at: When the SAE was added
        updated_at: When the SAE was last updated
    """

    __tablename__ = "saes"

    # Primary key (generated from repo_id, e.g., "jbloom--gemma-2-2b-res-jb")
    id: Mapped[str] = mapped_column(String(100), primary_key=True)

    # Source information
    repository_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    revision: Mapped[str] = mapped_column(String(100), default="main", nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(50), default="saelens", nullable=False)

    # Dimensions
    d_in: Mapped[int] = mapped_column(Integer, nullable=False)
    d_sae: Mapped[int] = mapped_column(Integer, nullable=False)

    # Training metadata
    trained_on: Mapped[str | None] = mapped_column(String(255), nullable=True)
    trained_layer: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Storage
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    cache_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # State
    status: Mapped[SAEStatus] = mapped_column(
        Enum(
            SAEStatus,
            name="saestatus",
            create_constraint=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        default=SAEStatus.CACHED,
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

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

    # Relationships
    attachments: Mapped[list["SAEAttachment"]] = relationship(
        "SAEAttachment",
        back_populates="sae",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<SAE(id='{self.id}', name='{self.name}', status={self.status.value})>"


class SAEAttachment(Base):
    """
    ORM model tracking SAE-model attachments.

    Only one active attachment is allowed at a time (v1.0).
    This is enforced via a partial unique index.

    Attributes:
        id: Primary key
        sae_id: Reference to the attached SAE
        model_id: Reference to the model it's attached to
        layer: Layer where the SAE is attached
        attached_at: When the SAE was attached
        detached_at: When the SAE was detached (null if active)
        memory_usage_mb: GPU memory used by this attachment
        is_active: Whether this is the current active attachment
    """

    __tablename__ = "sae_attachments"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign keys
    sae_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("saes.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Attachment details
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    attached_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    detached_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    memory_usage_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    sae: Mapped["SAE"] = relationship("SAE", back_populates="attachments")

    # Indexes and constraints
    __table_args__ = (
        # Partial unique index: only one active attachment allowed
        # PostgreSQL-specific: postgresql_where for partial unique index
        Index(
            "idx_single_active_attachment",
            "is_active",
            unique=True,
            postgresql_where=(is_active == True),  # noqa: E712
        ),
        # Index for looking up attachments by SAE
        Index("idx_sae_attachments_sae_id", "sae_id"),
    )

    def __repr__(self) -> str:
        return f"<SAEAttachment(id={self.id}, sae_id='{self.sae_id}', layer={self.layer}, active={self.is_active})>"
