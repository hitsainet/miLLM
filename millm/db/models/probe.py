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

    #: WHICH CUT OF THE BAR THIS IS. 1 is the one the producer's training run placed; every
    #: recalibration increments it.
    #:
    #: ⚠ IT EXISTS BECAUSE A SCORE AND A BAR FAIL DIFFERENTLY. `probe_events` records the
    #: threshold each verdict was judged against, so two verdicts under different bars are
    #: already distinguishable BY NUMBER — until two cuts happen to land on the same number, or
    #: a reader wants to know whether a `provisional` marker disappeared because a window got
    #: its own bar or because it never needed one. The revision answers both, and it is the key
    #: `threshold_history` is joined on.
    threshold_revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    #: The producer's identity for the calibration this bar was cut from, when it sent one.
    #: NULL is "not stated", which is different from "never re-cut" — that is revision 1.
    threshold_calibration_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    #: EVERY BAR THIS PROBE HAS SERVED, append-only, oldest first.
    #:
    #: ⚠ WITHOUT IT AN EVENT STAMPED `revision 3` IS UNANSWERABLE once the probe reaches 7: the
    #: row carries only the current bar. Seeded at import with revision 1 from the definition's
    #: own `decision`, so the history is complete from the first event rather than from the
    #: first re-cut.
    threshold_history: Mapped[Optional[list[Any]]] = mapped_column(JSONVariant, nullable=True)

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
    #: WHICH BAR WAS IN FORCE WHEN THIS PROBE WAS ARMED.
    #:
    #: ⚠ THE ACKNOWLEDGEMENT RECORDS A RUNG, NOT A BAR, AND THAT IS RIGHT — the operator consented
    #: to monitoring with a probe whose evidence is rung N, and a re-cut changes neither the rung
    #: nor the evidence. But a bar can now move under an armed probe, so consent given against
    #: one operating point can silently carry to another. Annotating is the proportionate answer:
    #: invalidating the consent would force a disarm/re-arm, which overwrites the acknowledgement
    #: anyway and resets it to NULL above rung 1 — losing the record in order to protect it.
    #:
    #: NULL for a probe armed before this column existed. Defaulting it to 1 would claim a fact
    #: nobody recorded, which is the `chat_format NOT NULL DEFAULT 'auto'` mistake.
    armed_threshold_revision: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

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

    #: WHICH SLICE OF THE REQUEST THIS VERDICT READ — `all`, `prompt` or `response`. One probe now
    #: reports several, because averaging the person's words together with the model's answer
    #: answers neither question: a long reply drags the mean down and the same conversation
    #: scores differently depending on how much the model said.
    window: Mapped[str] = mapped_column(String(16), nullable=False, server_default="all")
    #: The threshold was calibrated under the probe's own scope. This verdict was not read under
    #: it, so the number fires against a bar that was never cut for this slice.
    provisional: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sa_false())
    #: WHICH CUT OF THE BAR JUDGED THIS VERDICT. Denormalised per the module docstring, and taken
    #: from the ARMED probe rather than the row — see `probe_runtime._verdict_for`.
    #:
    #: ⚠ THE DEFAULT OF 1 IS TRUE OF EVERY ROW IT LANDS ON: no probe had ever been re-cut when
    #: this column was added, so every existing event really was judged under revision 1.
    threshold_revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )

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
