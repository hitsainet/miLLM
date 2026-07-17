"""
SensingEvent database model (Feature 11: Co-Activation Sensing).

One row per detected cluster co-activation event: a debounced token span in
one request where >= min_k members of the armed cluster fired together.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
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
from sqlalchemy.types import JSON

from millm.db.base import Base

# JSONB on postgres, plain JSON elsewhere (SQLite test databases)
JSONVariant = JSON().with_variant(JSONB(), "postgresql")


class SensingEvent(Base):
    """
    A persisted co-activation event.

    Attributes:
        id: Autoincrement primary key.
        profile_id: Owning cluster profile (CASCADE on delete).
        request_id: Generation request the event occurred in.
        phase: 'prefill' or 'decode' — which pass produced the span.
        pos_start / pos_end: Absolute token positions of the debounced span
            (the token being READ at each position — attribution convention).
        fired_members: list of [feature_idx, peak_activation] pairs.
        fired_count: Number of members that fired (>= min_k).
        score: max(act_i / theta_i) over fired members.
        ambient_fired_count: Best-effort full-SAE fired count when
            un-compacted monitoring co-ran; NULL otherwise (never estimated).
        context_text: Decoded ±K token window (None when K=0).
        context_token_ids: Token ids of the window (None when K=0).
        summary: Human-readable one-liner (<= 300 chars).
        truncated: True when the request hit the per-request event cap.
        created_at: Insertion time (retention pruning key).
    """

    __tablename__ = "sensing_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    phase: Mapped[str] = mapped_column(String(10), nullable=False)
    pos_start: Mapped[int] = mapped_column(Integer, nullable=False)
    pos_end: Mapped[int] = mapped_column(Integer, nullable=False)
    fired_members: Mapped[list[Any]] = mapped_column(JSONVariant, nullable=False)
    fired_count: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    ambient_fired_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_token_ids: Mapped[list[int] | None] = mapped_column(
        JSONVariant, nullable=True
    )
    # {before, span, after} decoded segments — the span is the fired
    # position(s); lets the UI highlight the prime token (goal item 1)
    context_parts: Mapped[dict | None] = mapped_column(JSONVariant, nullable=True)
    summary: Mapped[str] = mapped_column(String(300), nullable=False)
    truncated: Mapped[bool] = mapped_column(
        Boolean, server_default="false", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_sensing_events_profile_created", "profile_id", "created_at"),
        Index("idx_sensing_events_request", "request_id"),
    )

    def to_dict(self, include_context: bool = True) -> dict[str, Any]:
        """API-shaped dict; context is excluded from WS payloads (size +
        user content — the UI fetches detail via REST)."""
        data: dict[str, Any] = {
            "id": self.id,
            "profile_id": self.profile_id,
            "request_id": self.request_id,
            "phase": self.phase,
            "pos_start": self.pos_start,
            "pos_end": self.pos_end,
            "fired_members": self.fired_members,
            "fired_count": self.fired_count,
            "score": self.score,
            "ambient_fired_count": self.ambient_fired_count,
            "summary": self.summary,
            "truncated": self.truncated,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
        }
        if include_context:
            data["context_text"] = self.context_text
            data["context_token_ids"] = self.context_token_ids
            data["context_parts"] = self.context_parts
        return data
