"""Probe monitors and their events (Feature 24).

The whole imported document lives in ``Probe.definition``. Every typed column beside it is a
PROJECTION of that document, kept for queries and for the arm-time identity check — never a second
source of truth. When the two could disagree, the definition wins, and nothing writes a projected
column without writing the definition in the same transaction.

⚠ ``ProbeEvent.rung`` is denormalised at write time, exactly as ``CircuitEdgeSensingEvent``
denormalises ``edge_rung``. A probe's rung rises when more evidence arrives; an event must keep
describing the evidence that was true WHEN IT WAS OBSERVED. Joining to the probe's current rung
would retroactively upgrade what a month-old observation claimed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import false as sa_false, true as sa_true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from millm.db.base import Base

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite test DBs).
JSONVariant = JSON().with_variant(JSONB(), "postgresql")


class Probe(Base):
    """One imported probe definition, and its runtime state."""

    __tablename__ = "probes"

    id: Mapped[str] = mapped_column(String(24), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)

    #: The full `mistudio.probe-definition/v1` document, additive fields included.
    definition: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False)

    # --- identity: all of these are compared at arm time (FR-24.3) --------------------
    hf_id: Mapped[str] = mapped_column(String(255), nullable=False)
    #: What the definition says it was fitted on. May be a commit SHA or a branch name; when it is
    #: not a 40-hex SHA the arm-time check records REVISION_UNVERIFIED and allows.
    revision: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    d_model: Mapped[int] = mapped_column(Integer, nullable=False)
    n_layers: Mapped[int] = mapped_column(Integer, nullable=False)
    template_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # --- read point and readout -------------------------------------------------------
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    rule: Mapped[str] = mapped_column(String(32), nullable=False)
    streamable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_false()
    )
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    basis: Mapped[str] = mapped_column(
        String(32), nullable=False, default="residual", server_default="residual"
    )
    #: The definition's `sae` block for a k-sparse probe; NULL for a dense one.
    sae_ref: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)

    # --- decision ----------------------------------------------------------------------
    #: ⚠ NULL means no threshold was placed — the probe ranks but does not decide. It is NOT a
    #: threshold of 0, and a runtime that treats it as one turns a probe that has said nothing
    #: into one that fires on half its input.
    threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_fpr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- evidence -----------------------------------------------------------------------
    rung: Mapped[int] = mapped_column(Integer, nullable=False)
    #: The acknowledgement carried IN the definition (miStudio's operator).
    definition_acknowledgement: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONVariant, nullable=True
    )
    #: The acknowledgement given HERE when arming below rung 2. Two separate consents on purpose:
    #: the person exporting a weak probe and the person arming it on live traffic are not
    #: necessarily the same person, and only the second one is choosing to monitor with it.
    arm_acknowledgement: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONVariant, nullable=True
    )

    # --- runtime -------------------------------------------------------------------------
    #: The last parity report: max deviation, per-vector, and tokenization drift.
    parity: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    armed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_false()
    )
    #: Why an armed probe is not currently scoring (`speculative_decoding`, `model_changed`, ...).
    #: Armed-but-paused is a real state and it must be visible, not inferred from silence.
    paused_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    provenance: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    events: Mapped[list["ProbeEvent"]] = relationship(
        "ProbeEvent", back_populates="probe", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_probes_name", "name", unique=True),
        Index("ix_probes_armed", "armed"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "armed" if self.armed else "idle"
        return f"<Probe {self.id} {self.name!r} L{self.layer} {self.rule} rung={self.rung} {state}>"


class ProbeEvent(Base):
    """One request, as one armed probe saw it — scored or explicitly not."""

    __tablename__ = "probe_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    probe_id: Mapped[str] = mapped_column(
        String(24), ForeignKey("probes.id", ondelete="CASCADE"), nullable=False
    )
    #: The `/v1` completion id (`chatcmpl-…`). The only link between a response and its verdict.
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    #: ⚠ `scored=False` ALWAYS carries `not_scored_reason`. An unscored request that does not say
    #: why is the failure this feature exists to prevent (BR-006).
    scored: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=sa_true()
    )
    not_scored_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    verdict: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    #: The rung AS OF THIS OBSERVATION. See the module docstring.
    rung: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    top_positions: Mapped[Optional[list[Any]]] = mapped_column(JSONVariant, nullable=True)
    n_scored_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    #: ⚠ NEVER emitted on the socket. Fetched on demand from the event detail route only.
    context_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context_token_ids: Mapped[Optional[list[int]]] = mapped_column(JSONVariant, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    probe: Mapped["Probe"] = relationship("Probe", back_populates="events")

    __table_args__ = (
        Index("ix_probe_events_probe_created", "probe_id", "created_at"),
        Index("ix_probe_events_request_id", "request_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if not self.scored:
            return f"<ProbeEvent {self.id} {self.probe_id} not_scored={self.not_scored_reason!r}>"
        return f"<ProbeEvent {self.id} {self.probe_id} score={self.score} verdict={self.verdict}>"
