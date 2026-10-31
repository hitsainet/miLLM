"""Agent requests that wait for the operator (ENH-008 D1; migration 037).

An AI agent's call to one of miLLM's gated routes (``millm.api.approval_gate``) is not run. It is
stored here as ``pending`` and answered ``202``. The operator approves or rejects it in the admin
UI's Approvals page; approving runs the stored request in-process, exactly once
(``millm.services.approval_service``).

Lifecycle: ``pending`` -> ``executing`` -> ``executed`` | ``failed``; or ``pending`` ->
``rejected`` | ``expired``.

``secret_payload`` holds what must reach the executor but must never be shown: the lease id an
agent sent with a model load or unload. No route returns it, and it is cleared (set to NULL) when
the request is decided, whatever the decision.

The tuples below are the CHECK constraints. Migration 037 writes the same lists out literally (038 adds
``telemetry_write``), and ``tests/schema/test_approvals_migration.py`` pins them equal.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base

#: Every status a request can be in.
APPROVAL_STATUSES: tuple[str, ...] = (
    "pending", "executing", "executed", "failed", "rejected", "expired",
)

#: The statuses that end a request: nothing changes it after one of these.
DECIDED_STATUSES: tuple[str, ...] = ("executed", "failed", "rejected", "expired")

#: Every gated action (BUILD_SPEC §3.1). ``millm.api.approval_gate.APPROVAL_ACTIONS`` describes
#: each one, and a test holds the two equal.
APPROVAL_ACTION_NAMES: tuple[str, ...] = (
    "model_load", "model_unload", "model_delete",
    "probe_limits_write", "capacity_measure", "code_rule_import",
    # ENH-006 D76 (migration 038): an agent's change to a telemetry setting.
    "telemetry_write",
)

#: Longest reason stored with a decision (the UI sends at most 1000 characters).
REASON_MAX_CHARS = 1000


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"
    __table_args__ = (
        CheckConstraint(_in("status", APPROVAL_STATUSES), name="ck_approval_requests_status"),
        CheckConstraint(_in("action", APPROVAL_ACTION_NAMES), name="ck_approval_requests_action"),
        Index("ix_approval_requests_status_created_at", "status", "created_at"),
    )

    #: ``apr_<32 hex>``.
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    #: ``module.qualname`` of the gated route function the executor runs.
    executor_key: Mapped[str] = mapped_column(String(200), nullable=False)
    #: The facts the operator reads on the card.
    summary: Mapped[Any] = mapped_column(JSON, nullable=False)
    #: Every value the route was called with (path, query, body), as JSON. Shown in full.
    payload: Mapped[Any] = mapped_column(JSON, nullable=False)
    #: Never returned by any route; cleared at decision.
    secret_payload: Mapped[Any] = mapped_column(JSON, nullable=True)
    #: sha256 (hex) of the canonical ``payload``; checked again before the request runs.
    request_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    #: ``agent:<name>`` from the ``X-miLLM-Agent`` header.
    requested_by: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[Any] = mapped_column(JSON, nullable=True)
    error: Mapped[Any] = mapped_column(JSON, nullable=True)
    #: ``operator`` for a decision, ``system`` for an expiry or a restart.
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(REASON_MAX_CHARS), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: ENH-006 D79 (migration 039): ``bat_<32 hex>`` when the operator approved this request with
    #: "Approve all settings changes"; the same value on every request of that one approval.
    #: NULL for a request decided on its own.
    batch_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
