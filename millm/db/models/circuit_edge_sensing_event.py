"""Circuit edge sensing event (Feature 15).

One row = one observation that an edge's UPSTREAM member fired and its
DOWNSTREAM partner then fired within the lag window, on live traffic.

This is deliberately weaker than a causal claim and the schema is shaped to
keep it that way: ``edge_rung`` / ``edge_rung_language`` are denormalised at
write time so an event keeps describing the evidence that was true when it was
OBSERVED. Re-deriving the phrase from the circuit's current rung would let a
later re-validation retroactively upgrade months-old observations — which is
precisely the overclaim the evidence ladder exists to prevent.

An observation here is not causal evidence: it is co-activation in the
authored direction within a lag window. The rung carried on the row is the
only statement about causality, and it comes from miStudio — never from
having watched the edge fire.
"""

from typing import Any

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
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite test DBs).
JSONVariant = JSON().with_variant(JSONB(), "postgresql")


class CircuitEdgeSensingEvent(Base):
    """A single observed up→down edge firing."""

    __tablename__ = "circuit_edge_sensing_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    circuit_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("circuits.id", ondelete="CASCADE"),
        nullable=False,
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    phase: Mapped[str] = mapped_column(String(10), nullable=False)

    #: "{up_idx}@{up_layer}->{down_idx}@{down_layer}" — v1 edges carry no id.
    edge_key: Mapped[str] = mapped_column(String(128), nullable=False)

    up_layer: Mapped[int] = mapped_column(Integer, nullable=False)
    up_feature_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    up_pos: Mapped[int] = mapped_column(Integer, nullable=False)
    up_act: Mapped[float] = mapped_column(Float, nullable=False)

    down_layer: Mapped[int] = mapped_column(Integer, nullable=False)
    down_feature_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    down_pos: Mapped[int] = mapped_column(Integer, nullable=False)
    down_act: Mapped[float] = mapped_column(Float, nullable=False)

    #: down_pos - up_pos; >= 1 (strict ordering), <= the configured window.
    token_lag: Mapped[int] = mapped_column(Integer, nullable=False)

    #: Evidence AS OF OBSERVATION — never re-derived. See module docstring.
    edge_rung: Mapped[int] = mapped_column(Integer, nullable=False)
    edge_rung_language: Mapped[str] = mapped_column(String(64), nullable=False)
    edge_type: Mapped[str | None] = mapped_column(String(32), nullable=True)

    ambient_fired_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    context_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_token_ids: Mapped[list[int] | None] = mapped_column(
        JSONVariant, nullable=True
    )
    #: {before, span, after}; span covers up_pos..down_pos inclusive.
    context_parts: Mapped[dict[str, str] | None] = mapped_column(
        JSONVariant, nullable=True
    )

    summary: Mapped[str] = mapped_column(String(300), nullable=False)
    truncated: Mapped[bool] = mapped_column(
        Boolean, server_default="false", nullable=False
    )
    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_circuit_edge_events_circuit_created", "circuit_id", "created_at"),
        Index("idx_circuit_edge_events_request", "request_id"),
        Index("idx_circuit_edge_events_edge", "circuit_id", "edge_key"),
    )

    def to_dict(self, include_context: bool = True) -> dict[str, Any]:
        """Serialise for the API and the WS payload.

        ``include_context=False`` omits the decoded text — the WS broadcast
        carries no user prompt content.
        """
        data: dict[str, Any] = {
            "id": self.id,
            "circuit_id": self.circuit_id,
            "request_id": self.request_id,
            "phase": self.phase,
            "edge_key": self.edge_key,
            "up": {
                "layer": self.up_layer,
                "feature_idx": self.up_feature_idx,
                "pos": self.up_pos,
                "act": self.up_act,
            },
            "down": {
                "layer": self.down_layer,
                "feature_idx": self.down_feature_idx,
                "pos": self.down_pos,
                "act": self.down_act,
            },
            "token_lag": self.token_lag,
            "edge_rung": self.edge_rung,
            "edge_rung_language": self.edge_rung_language,
            "edge_type": self.edge_type,
            "ambient_fired_count": self.ambient_fired_count,
            "summary": self.summary,
            "truncated": self.truncated,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_context:
            data["context_text"] = self.context_text
            data["context_token_ids"] = self.context_token_ids
            data["context_parts"] = self.context_parts
        return data
