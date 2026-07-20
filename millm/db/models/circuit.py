"""
Circuit database model (Feature 13: Circuit Import).

A circuit is a multi-layer graph over several SAEs — unlike a cluster it is not
a ``profiles`` row. ``circuit_meta`` stores the full original
``mistudio.circuit-definition/v1`` document verbatim so re-export is lossless.
At most one circuit may be active at a time (partial unique index, mirroring
``idx_active_profile``).
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base


class Circuit(Base):
    """
    An imported ``mistudio.circuit-definition/v1`` document.

    Attributes:
        id: Unique circuit identifier.
        name: Display name (unique).
        description: Optional narrative.
        circuit_meta: The RAW imported document (lossless re-export).
        rung: Circuit evidence rung = MIN over edges (0..3); empty edges ⇒ 0.
        edge_count: Number of edges (cached for list display).
        layers: Layers the circuit references, e.g. ``[10, 13]``.
        per_sae_warnings: Per-referenced-SAE compatibility verdicts from import.
        serveable: True only when EVERY referenced SAE binds (full multi-SAE
            serving); otherwise activation degrades to per-layer slices.
        is_active: Whether this circuit is currently serving.
        serving_mode: ``"full"`` | ``"slice_fallback"`` | None.
        intensity: Current global lambda dial.
        provenance: Import origin (file/hub, timestamps).
    """

    __tablename__ = "circuits"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    circuit_meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    rung: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    edge_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    layers: Mapped[list[int]] = mapped_column(JSONB, nullable=False, default=list)
    per_sae_warnings: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB, nullable=True
    )

    serveable: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    serving_mode: Mapped[str | None] = mapped_column(String(20), nullable=True)
    intensity: Mapped[float] = mapped_column(
        Float, default=1.0, server_default="1.0", nullable=False
    )
    provenance: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "uq_circuits_active",
            "is_active",
            unique=True,
            postgresql_where=(is_active == True),  # noqa: E712
            sqlite_where=(is_active == True),  # noqa: E712
        ),
        Index("idx_circuits_rung", "rung"),
    )

    @property
    def validated(self) -> bool:
        """True when the circuit's evidence rung is CAUSALLY_VALIDATED or above.

        Below this, the runtime must never describe the circuit as "causal"
        and activation requires an explicit unvalidated acknowledgement.
        """
        return self.rung >= 2

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Circuit id={self.id!r} name={self.name!r} rung={self.rung} "
            f"layers={self.layers} serveable={self.serveable} active={self.is_active}>"
        )
