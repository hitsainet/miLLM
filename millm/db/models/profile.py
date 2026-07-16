"""
Profile database model.

Stores steering configuration profiles for quick recall and sharing.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from millm.db.base import Base


class Profile(Base):
    """
    Steering configuration profile.

    Stores steering values along with metadata about the model and SAE
    the profile was created with. Only one profile can be active at a time,
    enforced via a partial unique index on is_active.

    Attributes:
        id: Unique profile identifier (UUID).
        name: Display name (unique).
        description: Optional description text.
        model_id: ID of the model this profile was created with.
        sae_id: ID of the SAE this profile was created with.
        layer: Layer the SAE was attached to.
        steering: JSONB dict mapping feature indices (as strings) to values.
        is_active: Whether this is the currently active profile.
        created_at: When the profile was created.
        updated_at: When the profile was last updated.
    """

    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    model_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    sae_id: Mapped[str | None] = mapped_column(
        # Matches SAE.id (String(100)) — 50 truncated real HF-derived ids
        # and 500'd every BOUND cluster import (found by the 010 live E2E)
        String(100),
        nullable=True,
    )
    layer: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    steering: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    is_active: Mapped[bool] = mapped_column(
        default=False,
    )
    # Feature 8 (Cluster Import): 'manual' rows are user-authored steering
    # configs; 'cluster' rows are imported mistudio.cluster-definition/v1
    # documents materialized as profiles.
    source_kind: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="manual",
        server_default="manual",
        index=False,  # covered by idx_profiles_source_kind (migration 007)
    )
    # Full original definition (members verbatim, budget, narrative, refs,
    # provenance, import warnings, optional hub_ref) — lossless re-export.
    cluster_meta: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    # Current lambda dial; steering values are stored at lambda=1 basis and
    # scaled by this at activation (never baked in).
    intensity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        server_default="1.0",
    )
    # Feature 11 (Co-Activation Sensing) opt-in.
    sensing_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="false",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Feature 11: sensing events cascade with their profile. The ORM loads
    # and deletes children on session.delete (works on SQLite, where the FK
    # pragma is off by default); postgres ALSO has the migration's FK
    # ondelete=CASCADE covering bulk/non-ORM deletes. Bounded load: retention
    # caps events per cluster.
    sensing_events = relationship(
        "SensingEvent",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # Partial unique index ensures only one profile can be active at a time
    __table_args__ = (
        Index(
            "idx_active_profile",
            "is_active",
            unique=True,
            postgresql_where=(is_active == True),  # noqa: E712
            # SQLite (tests) supports partial indexes too; without this the
            # index degrades to a FULL unique index on is_active and any two
            # inactive rows collide.
            sqlite_where=(is_active == True),  # noqa: E712
        ),
    )

    @property
    def is_cluster(self) -> bool:
        """True when this row is an imported cluster definition (Feature 8)."""
        return self.source_kind == "cluster"

    @property
    def feature_count(self) -> int:
        """Number of features in the steering configuration."""
        return len(self.steering) if self.steering else 0

    def get_steering_dict(self) -> dict[int, float]:
        """
        Get steering as int keys (for SAE use).

        JSONB stores keys as strings, so we convert them back to integers.

        Returns:
            Dict mapping feature indices to values.
        """
        if not self.steering:
            return {}
        return {int(k): float(v) for k, v in self.steering.items()}

    def __repr__(self) -> str:
        return (
            f"Profile(id={self.id!r}, name={self.name!r}, "
            f"features={self.feature_count}, active={self.is_active})"
        )
