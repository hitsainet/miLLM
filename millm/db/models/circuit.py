"""
Circuit database model (Feature 13: Circuit Import).

A circuit is a multi-layer graph over several SAEs — unlike a cluster it is not
a ``profiles`` row. ``circuit_meta`` stores the full original
``mistudio.circuit-definition/v1`` document verbatim so re-export is lossless.
Several circuits may be active at once (Feature 19), provided their claim sets
are disjoint — see ``circuit_layer_claim.py``.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

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

    #: Persistent operator INTENT for edge sensing (Feature 15), reported
    #: distinctly from runtime ``armed``: a circuit can be enabled but unarmed
    #: because it is not active, or because its SAE set is not attached.
    sensing_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )

    # Mirrors Profile.sensing_events: the ORM cascade covers SQLite (FK pragma
    # off by default); postgres ALSO has the migration's FK ondelete=CASCADE
    # for bulk/non-ORM deletes. Bounded load: retention caps events per circuit.
    edge_sensing_events = relationship(
        "CircuitEdgeSensingEvent",
        cascade="all, delete-orphan",
        lazy="select",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
        nullable=False,
    )

    # Feature 19: `uq_circuits_active` is GONE. Several circuits may be active
    # at once, provided their claim sets are disjoint — the constraint moved
    # from "one circuit" to "one circuit PER LAYER"
    # (`circuit_layer_claims.uq_circuit_layer_claim_live`), which is the unit
    # contention actually has.
    #
    # Dropping it in migration 013 is not sufficient on its own: the index also
    # lives in `Base.metadata`, so every test database built by
    # `create_all` would keep enforcing single-active and make concurrent
    # serving untestable. That is exactly how this was caught — the first
    # registry test could not insert a second active circuit.
    __table_args__ = (Index("idx_circuits_rung", "rung"),)

    @property
    def validated(self) -> bool:
        """True when the circuit's evidence rung is CAUSALLY_VALIDATED or above.

        Below this, the runtime must never describe the circuit as "causal"
        and activation requires an explicit unvalidated acknowledgement.

        Delegates to the evidence ladder rather than hardcoding the threshold —
        two implementations of the same gate WILL drift, and this one is the
        feature's central honesty invariant.
        """
        from millm.core.circuit_evidence import is_validated

        return is_validated(self.rung)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Circuit id={self.id!r} name={self.name!r} rung={self.rung} "
            f"layers={self.layers} serveable={self.serveable} active={self.is_active}>"
        )
