"""Rule sets, their attachments to behaviours, and rule results (039 S2; FTDD §4.3, migration 035;
S3's `code_modules`, migration 036).

- `rule_sets`: one imported miDataworks rule-set document, never replaced in place (a new version
  is a new import with a new id and digest, as probes are). `digest` is the producer's identity
  (RFC 8785 + SHA-256), unique: an identical re-import returns the existing id.
- `behaviour_rule_signals`: one row per behaviour key with any attachment, created on first write.
  No foreign key on the id list: the attach route checks every id, and deleting a set removes it
  from the list with a history row.
- `behaviour_rule_signal_history`: append-only, one row per change.
- `rule_signal_results`: one row per recorded `kind: "rule"` result. ⚠ NO `Text` COLUMN and no
  string wider than 64: a rule result carries a count, never a value, and the schema test reads
  this class's column list to hold that (`tests/schema/test_rule_sets_migration.py`). No foreign
  key to `rule_sets` either: a deleted set's results keep its id and digest (FPRD §5).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base
from millm.db.models.probe_capacity import JSONVariant


class RuleSet(Base):
    __tablename__ = "rule_sets"

    id: Mapped[str] = mapped_column(String(24), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    behaviour_key: Mapped[str] = mapped_column(String(64), nullable=False)
    #: `authored` / `derived` (derived = the document carries provenance). Never read by the
    #: evaluator (FR-39.21).
    origin: Mapped[str] = mapped_column(String(16), nullable=False)
    digest: Mapped[str] = mapped_column(String(64), nullable=False)
    #: The whole document as received, dataset-building fields included (OQ-39.2 (a)).
    document: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False)
    provenance: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    #: `passed` / `vectors_failed` (the startup self-check, FR-39.20).
    vector_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="passed", server_default="passed"
    )
    vector_report: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONVariant, nullable=True)
    #: 039 S3 (migration 036): `{code_sha256: canonical UTF-8 text}` of each code-rule document the
    #: import carried, verified against its hash. NULL: none carried (its code rules stay pending).
    code_modules: Mapped[Optional[dict[str, str]]] = mapped_column(JSONVariant, nullable=True)
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    checked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("uq_rule_sets_digest", "digest", unique=True),
        Index("ix_rule_sets_behaviour_key", "behaviour_key", "id"),
    )


class BehaviourRuleSignals(Base):
    __tablename__ = "behaviour_rule_signals"

    behaviour_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    #: The attached rule-set ids, in attach order.
    rule_set_ids: Mapped[list[str]] = mapped_column(JSONVariant, nullable=False)
    #: `{rule_set_id: [expression name, ...]}` narrowing a set to named signals; NULL = all.
    signals: Mapped[Optional[dict[str, list[str]]]] = mapped_column(JSONVariant, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class BehaviourRuleSignalHistory(Base):
    __tablename__ = "behaviour_rule_signal_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    behaviour_key: Mapped[str] = mapped_column(String(64), nullable=False)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    #: `attach` / `detach` / `clear` / `rule_set_deleted`.
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    rule_set_ids: Mapped[list[str]] = mapped_column(JSONVariant, nullable=False)
    signals: Mapped[Optional[dict[str, list[str]]]] = mapped_column(JSONVariant, nullable=True)
    by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_behaviour_rule_signal_history_key", "behaviour_key", "changed_at"),
    )


class RuleSignalResult(Base):
    __tablename__ = "rule_signal_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: The recorded `mi.probe.result/v1` object (`kind: "rule"`), verbatim, WITHOUT
    #: `source_event_id`; every column below is a projection of it.
    result: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False)
    result_id: Mapped[str] = mapped_column(String(36), nullable=False)
    trace_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    span_id: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    outcome: Mapped[str] = mapped_column(String(12), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(12), nullable=False, default="final", server_default="final"
    )
    persistence: Mapped[str] = mapped_column(
        String(14), nullable=False, default="persisted", server_default="persisted"
    )
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(
        String(24), ForeignKey("probe_sessions.id", ondelete="SET NULL"), nullable=True
    )
    turn_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    #: `live` / `batch` (stateless writes nothing).
    origin: Mapped[str] = mapped_column(
        String(8), nullable=False, default="live", server_default="live"
    )
    batch_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    batch_line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    behaviour_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    rule_set_id: Mapped[str] = mapped_column(String(24), nullable=False)
    rule_set_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    signal: Mapped[str] = mapped_column(String(64), nullable=False)
    form: Mapped[str] = mapped_column(String(24), nullable=False)
    window: Mapped[str] = mapped_column(String(16), nullable=False)
    #: NULL when the outcome is not `completed`.
    fires: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    #: How many matches — never a value.
    count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("uq_rule_signal_results_result_id", "result_id", unique=True),
        Index("ix_rule_signal_results_request_id", "request_id"),
        Index("ix_rule_signal_results_behaviour_created", "behaviour_key", "created_at"),
        Index("ix_rule_signal_results_signal_origin_created",
              "rule_set_id", "signal", "origin", "created_at"),
        Index("ix_rule_signal_results_session_turn", "session_id", "turn_index"),
        Index(
            "uq_rule_signal_results_batch_line",
            "rule_set_id", "signal", "batch_id", "batch_line", "window",
            unique=True,
            postgresql_where=text("batch_id IS NOT NULL"),
            sqlite_where=text("batch_id IS NOT NULL"),
        ),
    )
