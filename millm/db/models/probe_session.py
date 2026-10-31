"""Probe sessions, their turns and the opt-in transcripts (Feature 36, migration 029).

A SESSION groups the requests of one conversation, so an operator can follow what each armed
probe said turn by turn. It is named by the client (`X-miLLM-Session`) or found by a salted
continuation fingerprint of the conversation (`millm.services.probe_sessions`).

⚠ THE TRANSCRIPT IS PERSONAL DATA. `probe_transcripts` holds the newest user message and the
reply of one turn, and exists only while the operator has turned recording on (D5, off by
default). It is never selected by a list query, never emitted on the socket and never logged;
the one route that serves it is `GET /api/probes/sessions/{id}/transcript`. Retention removes it
after `PROBE_TRANSCRIPT_MAX_AGE_DAYS` (7) or past `PROBE_TRANSCRIPT_MAX_TURNS`, and a session
delete removes it at once (cascade).

The turn's three hashes are HMAC-SHA256 under a per-installation salt (D43). They are of user
text, so they are pruned with their turns and never served.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import false as sa_false
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base

#: How a session was found: the client named it, a continuation matched, or a request that
#: carries no conversation (a text completion with no header).
SESSION_SOURCES = ("header", "fingerprint", "single")


class ProbeSession(Base):
    """One conversation, as the probes saw it."""

    __tablename__ = "probe_sessions"

    id: Mapped[str] = mapped_column(String(24), primary_key=True)
    #: The client's own id (`X-miLLM-Session`), stored as given (D8). NOT unique: a successor
    #: opened after the idle end shares it with its predecessor.
    client_session_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    #: The session this one continues after it went idle (FR-36.3). SET NULL when it is pruned.
    predecessor_id: Mapped[Optional[str]] = mapped_column(
        String(24), ForeignKey("probe_sessions.id", ondelete="SET NULL"), nullable=True
    )
    #: The fingerprint matched more than one session, so this one joined none (D44).
    link_ambiguous: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_false()
    )
    served_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    n_turns: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    __table_args__ = (
        Index("ix_probe_sessions_client_session_id", "client_session_id"),
        Index("ix_probe_sessions_last_seen_at", "last_seen_at"),
    )


class ProbeSessionTurn(Base):
    """One request of a session: its place in the conversation and the hashes that find it."""

    __tablename__ = "probe_session_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(24), ForeignKey("probe_sessions.id", ondelete="CASCADE"), nullable=False
    )
    #: The `/v1` completion id. Joins the turn to its `probe_events` rows.
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: The number of `user` messages in the request (1 for a text completion).
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    #: H(messages before the newest user message) — what a continuation is looked up BY.
    prefix_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    #: H(messages + the reply) — what the NEXT request's prefix matches exactly.
    continuation_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    #: H(messages) — the weak key, for a client that re-renders the reply it sends back.
    weak_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    origin: Mapped[str] = mapped_column(String(8), nullable=False, server_default="live")
    served_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    has_transcript: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_false()
    )
    #: The request failed before it finished: the transcript holds whatever reply existed.
    transcript_incomplete: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_false()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("uq_probe_session_turns_request_id", "request_id", unique=True),
        Index("ix_probe_session_turns_session_turn", "session_id", "turn_index"),
        Index("ix_probe_session_turns_prefix_hash", "prefix_hash"),
        Index("ix_probe_session_turns_continuation_hash", "continuation_hash"),
        Index("ix_probe_session_turns_weak_hash", "weak_hash"),
    )


class ProbeTranscript(Base):
    """⚠ USER TEXT. The newest user message and the reply of one turn (D5). See the module doc."""

    __tablename__ = "probe_transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("probe_session_turns.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(
        String(24), ForeignKey("probe_sessions.id", ondelete="CASCADE"), nullable=False
    )
    user_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reply_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    #: How many choices the request asked for; only the first is recorded (`n > 1`).
    n_choices: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("uq_probe_transcripts_turn_id", "turn_id", unique=True),
        Index("ix_probe_transcripts_session_id", "session_id"),
        Index("ix_probe_transcripts_created_at", "created_at"),
    )
