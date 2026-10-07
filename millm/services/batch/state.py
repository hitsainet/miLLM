"""Batch state: the status set, the ONE transition table, and the OpenAI object serialisers.

Feature 26 (FR-26.3.1, FR-26.3.5). Three rules this module exists to hold in one place:

* **The status set is OpenAI's eight values and nothing else.** `BatchStatus` is the only list;
  the database CHECK constraint is built from it, so a ninth value cannot be written by any path.
* **Transitions come from one function.** `transition()` holds the FR-26.3.5 table and stamps the
  OpenAI timestamp for the status it enters. Every status write in the runner, the validator, the
  routes and the startup reconciliation goes through it, so a path that skips a state (or a
  timestamp) is a refusal, not a silent write.
* **Served endpoints come from the live app**, never a hand-kept list (`served_batch_endpoints`):
  `/api/probes/score` is accepted the moment Feature 27's route is registered (FR-26.1.4).

`BATCH_ROW` lives here, not in `runner.py`, so `inference_service` can read it without importing
the runner (FTID §2: avoid an import cycle).
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


from millm.core.batch_values import (  # noqa: E402  re-exported: one definition
    TERMINAL,
    BatchStatus,
    FilePurpose,
    FileStatus,
    RowKind,
    RowState,
    WaitingReason,
)


#: FR-26.3.5, verbatim. Nothing else may move a batch.
TRANSITIONS: dict[BatchStatus, frozenset[BatchStatus]] = {
    BatchStatus.VALIDATING: frozenset(
        {BatchStatus.IN_PROGRESS, BatchStatus.FAILED, BatchStatus.CANCELLING}
    ),
    BatchStatus.IN_PROGRESS: frozenset(
        {BatchStatus.FINALIZING, BatchStatus.CANCELLING, BatchStatus.EXPIRED, BatchStatus.FAILED}
    ),
    BatchStatus.FINALIZING: frozenset({BatchStatus.COMPLETED, BatchStatus.FAILED}),
    BatchStatus.CANCELLING: frozenset({BatchStatus.CANCELLED}),
    BatchStatus.COMPLETED: frozenset(),
    BatchStatus.FAILED: frozenset(),
    BatchStatus.CANCELLED: frozenset(),
    BatchStatus.EXPIRED: frozenset(),
}

#: The OpenAI timestamp each status stamps when entered.
_TIMESTAMP_FOR: dict[BatchStatus, str] = {
    BatchStatus.IN_PROGRESS: "in_progress_at",
    BatchStatus.FINALIZING: "finalizing_at",
    BatchStatus.COMPLETED: "completed_at",
    BatchStatus.FAILED: "failed_at",
    BatchStatus.CANCELLING: "cancelling_at",
    BatchStatus.CANCELLED: "cancelled_at",
    BatchStatus.EXPIRED: "expired_at",
}


class IllegalTransitionError(RuntimeError):
    """A status write the FR-26.3.5 table does not allow. A defect, never a user error."""


#: The (batch_id, line_no) of the row whose service call is running, or None (FTID §3). Read by
#: `_use_cbm_for_request` (no row runs on the continuous batching manager) and by
#: `_probe_record` (batch-marked probe events, T-70).
BATCH_ROW: "contextvars.ContextVar[Optional[tuple[str, int]]]" = contextvars.ContextVar(
    "millm_batch_row", default=None
)


def origin_fields_for(batch_row: Optional[tuple[str, int]]) -> dict[str, Any]:
    """`origin`/`batch_id`/`batch_line` for an event row: `live`, or the batch row it came from
    (T-70 for probe events; FTASKS 0.4 extends it to sensing and circuit-edge sensing)."""
    if batch_row is None:
        return {"origin": "live", "batch_id": None, "batch_line": None}
    return {"origin": "batch", "batch_id": batch_row[0], "batch_line": int(batch_row[1])}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def can_transition(current: str, to: str) -> bool:
    return BatchStatus(to) in TRANSITIONS[BatchStatus(current)]


def transition(batch: Any, to: BatchStatus, now: Optional[datetime] = None) -> None:
    """Move `batch` to `to` and stamp its OpenAI timestamp. Raises on a move the table forbids.

    Leaving `in_progress` clears `waiting_reason`: the field describes why a RUNNING batch is not
    running a row, and a terminal batch reporting `lease_unavailable` would be a stale claim.
    """
    current = BatchStatus(batch.status)
    if to not in TRANSITIONS[current]:
        raise IllegalTransitionError(f"batch {batch.id}: {current.value} -> {to.value} is not allowed")
    batch.status = to.value
    stamp = _TIMESTAMP_FOR.get(to)
    if stamp is not None:
        setattr(batch, stamp, now or utcnow())
    if to is not BatchStatus.IN_PROGRESS:
        batch.waiting_reason = None


# --------------------------------------------------------------------------- serialisers


def epoch(value: Optional[datetime]) -> Optional[int]:
    """OpenAI timestamps are Unix seconds. SQLite drops tzinfo, so a naive value is UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def file_object(row: Any) -> dict[str, Any]:
    """OpenAI's file object (FR-26.1.1). `storage_path` is never returned (FTDD §8)."""
    return {
        "id": row.id,
        "object": "file",
        "bytes": int(row.bytes),
        "created_at": epoch(row.created_at),
        "expires_at": epoch(row.expires_at),
        "filename": row.filename,
        "purpose": row.purpose,
        "status": row.status,
        "status_details": None,
    }


def batch_object(row: Any) -> dict[str, Any]:
    """OpenAI's batch object (FR-26.1.2) plus the one `millm` extension object (FTDD §5)."""
    window = row.completion_window
    return {
        "id": row.id,
        "object": "batch",
        "endpoint": row.endpoint,
        "model": row.model_name,
        "errors": row.errors,
        "input_file_id": row.input_file_id,
        "completion_window": window,
        "status": row.status,
        "output_file_id": row.output_file_id,
        "error_file_id": row.error_file_id,
        "created_at": epoch(row.created_at),
        "in_progress_at": epoch(row.in_progress_at),
        "expires_at": epoch(row.expires_at),
        "finalizing_at": epoch(row.finalizing_at),
        "completed_at": epoch(row.completed_at),
        "failed_at": epoch(row.failed_at),
        "expired_at": epoch(row.expired_at),
        "cancelling_at": epoch(row.cancelling_at),
        "cancelled_at": epoch(row.cancelled_at),
        "request_counts": {
            "total": int(row.request_total or 0),
            "completed": int(row.request_completed or 0),
            "failed": int(row.request_failed or 0),
        },
        "metadata": row.batch_metadata,
        "millm": {
            "pack": bool(row.pack),
            "waiting_reason": row.waiting_reason,
            "lease_mode": row.lease_mode,
            # `24h` is OpenAI's only value; anything else is this server's extension (T-65).
            "completion_window_extension": window != "24h",
            "output_expires_after": int(row.output_expires_after_s),
        },
    }


def openai_list(data: list[dict[str, Any]], has_more: bool) -> dict[str, Any]:
    """OpenAI's list shape (`first_id`, `last_id`, `has_more`), shared by files and batches."""
    return {
        "object": "list",
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": has_more,
    }


def error_line(
    custom_id: Optional[str],
    line_no: int,
    status_code: Optional[int],
    code: str,
    message: str,
    body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """ONE shape for validation, row and unrun errors (FR-26.2.5, FR-26.6.7, FTID §11).

    `line` is the 1-based input line number; `response` is null for a line that never reached a
    service (invalid, cancelled, expired) and carries the synchronous status and body otherwise.
    """
    import uuid

    return {
        "id": f"batch_req_{uuid.uuid4().hex[:24]}",
        "custom_id": custom_id,
        "line": line_no,
        "response": None
        if status_code is None
        else {
            "status_code": status_code,
            "request_id": f"req_{uuid.uuid4().hex[:24]}",
            "body": body,
        },
        "error": {"code": code, "message": message},
    }


def served_batch_endpoints(app: Any) -> frozenset[str]:
    """The batch endpoints this app SERVES, read from `app.openapi()["paths"]` (FR-26.1.4).

    Intersected with the four FR-26.1.4 names, so a route that is not one of them can never be a
    batch endpoint, and one of them that is not registered is refused. `app.routes` is not a route
    list under FastAPI's router wrapper; the OpenAPI document is.
    """
    paths = app.openapi().get("paths", {})
    served = {path for path, ops in paths.items() if "post" in ops}
    return frozenset(SUPPORTED_ENDPOINTS & served)


#: The FR-26.1.4 names. Membership here is necessary, not sufficient: see `served_batch_endpoints`.
SUPPORTED_ENDPOINTS: frozenset[str] = frozenset(
    {"/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/api/probes/score"}
)


def status_values() -> Iterable[str]:
    return (s.value for s in BatchStatus)
