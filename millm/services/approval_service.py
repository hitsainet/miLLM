"""miLLM's approval queue (ENH-008 D1; BUILD_SPEC §3.1).

Pattern copied from miForge's ``services/approval_service.py`` (not imported; the apps stay
standalone). Changed for miLLM: async; the executor runs as an ``asyncio`` task, so a long one
(capacity measurement) leaves the row ``executing`` while the approve call returns; a
``secret_payload`` that is never returned and is cleared at decision; Socket.IO events.

Lifecycle: ``pending`` -> ``executing`` -> ``executed`` | ``failed``; or ``pending`` ->
``rejected`` | ``expired``.

**Exactly once.** Approving claims the row with ``UPDATE approval_requests SET
status='executing' WHERE id = :id AND status = 'pending'`` and commits BEFORE the executor runs.
Of two concurrent approves exactly one sees a row count of 1; the other is refused
``approval_not_pending``. So the route function runs once, whatever the timing.

**Expiry is lazy**: evaluated on every list, get, approve and reject, so no timer is needed and an
expired request can never run.

**A restart** ends any executor in flight: ``main.STALE_STATE_RESETS`` marks every ``executing``
row ``failed`` at startup ("interrupted by a restart; ask again").
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import sqlalchemy as sa
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession

from millm.core.config import settings
from millm.core.errors import (
    ApprovalExecutorMissingError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalNotPendingError,
    ApprovalTamperedError,
    MiLLMError,
)
from millm.core.logging import get_logger
from millm.db.models.approval import APPROVAL_STATUSES, ApprovalRequest

logger = get_logger(__name__)

HINT = "Waiting for the operator to approve this in miLLM's Approvals page."

#: How long ``approve`` waits for the executor before answering with the row still
#: ``executing``. Most actions finish well inside it (a load only STARTS the load); capacity
#: measurement may not, and finishes in the background.
APPROVE_WAIT_S = 10.0

#: The status filter meaning "every decided or running request" (the History section).
DECIDED_FILTER = "decided"

#: Executors still running, by approval id (tests and shutdown await them).
RUNNING: dict[str, asyncio.Task[None]] = {}


def now() -> datetime:
    """The current time, aware. Module-level so a test can move the clock."""
    return datetime.now(UTC)


def _aware(moment: datetime) -> datetime:
    """SQLite returns naive datetimes for timezone columns; they were written as UTC."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def new_id() -> str:
    return f"apr_{uuid.uuid4().hex}"


def digest(payload: Any) -> str:
    """sha256 (hex) of the canonical JSON of a stored payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _emit_created(approval: ApprovalRequest) -> None:
    from millm.sockets.progress import progress_emitter

    progress_emitter.emit_approval_created({"id": approval.id, "action": approval.action})


def _emit_resolved(approval_id: str, status: str) -> None:
    from millm.sockets.progress import progress_emitter

    progress_emitter.emit_approval_resolved({"id": approval_id, "status": status})


async def create_pending(
    session: AsyncSession,
    *,
    action: str,
    executor_key: str,
    payload: dict[str, Any],
    secret_payload: dict[str, Any] | None,
    requested_by: str,
    summary: dict[str, Any],
) -> ApprovalRequest:
    """Store an agent's request. Commits, so the 202 the agent receives names a real row."""
    stored = jsonable_encoder(payload)
    created = now()
    approval = ApprovalRequest(
        id=new_id(),
        action=action,
        executor_key=executor_key,
        summary=jsonable_encoder(summary),
        payload=stored,
        secret_payload=secret_payload,
        request_digest=digest(stored),
        requested_by=requested_by,
        status="pending",
        created_at=created,
        expires_at=created + timedelta(hours=int(settings.APPROVAL_TTL_HOURS)),
    )
    session.add(approval)
    await session.commit()
    logger.info("approval_requested", approval_id=approval.id, action=action,
                requested_by=requested_by)
    _emit_created(approval)
    return approval


def pending_body(approval: ApprovalRequest) -> dict[str, Any]:
    """The ``data`` of the ``202`` an agent receives."""
    return {
        "approval_id": approval.id,
        "status": approval.status,
        "action": approval.action,
        "summary": approval.summary,
        "request_digest": "sha256:" + approval.request_digest,
        "expires_at": _aware(approval.expires_at).isoformat(),
        "hint": HINT,
    }


def _iso(moment: datetime | None) -> str | None:
    return None if moment is None else _aware(moment).isoformat()


def as_payload(approval: ApprovalRequest) -> dict[str, Any]:
    """One request on the wire. ⚠ ``secret_payload`` is NEVER included (it holds the lease id)."""
    from millm.api.approval_gate import APPROVAL_ACTIONS

    known = APPROVAL_ACTIONS.get(approval.action)
    return {
        "approval_id": approval.id,
        "action": approval.action,
        "action_label": known.label if known is not None else approval.action,
        # ENH-006 D79/D81: which "Approve all" batch may approve it (None: one at a time).
        "settings_change": bool(known is not None and known.settings_change),
        "batch_kind": known.batch_kind if known is not None else None,
        "status": approval.status,
        "summary": approval.summary,
        "payload": approval.payload,
        "request_digest": "sha256:" + approval.request_digest,
        "requested_by": approval.requested_by,
        "decided_by": approval.decided_by,
        "reason": approval.reason,
        "result": approval.result,
        "error": approval.error,
        "created_at": _iso(approval.created_at),
        "decided_at": _iso(approval.decided_at),
        "expires_at": _iso(approval.expires_at),
        "batch_id": approval.batch_id,
    }


async def expire_due(session: AsyncSession) -> int:
    """Mark every pending request past its expiry ``expired``. Returns how many."""
    moment = now()
    due = list((await session.execute(
        sa.select(ApprovalRequest.id).where(ApprovalRequest.status == "pending",
                                            ApprovalRequest.expires_at <= moment)
    )).scalars())
    if not due:
        return 0
    result = await session.execute(
        sa.update(ApprovalRequest)
        .where(ApprovalRequest.id.in_(due), ApprovalRequest.status == "pending")
        .values(status="expired", decided_by="system", decided_at=moment, secret_payload=None)
        .execution_options(synchronize_session=False)
    )
    await session.commit()
    for approval_id in due:
        _emit_resolved(approval_id, "expired")
    return int(getattr(result, "rowcount", 0) or 0)


async def listing(session: AsyncSession, status: str | None = None, limit: int = 200,
                  offset: int = 0) -> tuple[list[ApprovalRequest], int]:
    """``(requests, pending count)`` after expiring what is due.

    ``status`` is one status, or ``decided`` for every request that is no longer waiting (the
    History section: newest decision first). Otherwise newest request first.
    """
    if status is not None and status not in APPROVAL_STATUSES and status != DECIDED_FILTER:
        from millm.core.errors import InvalidParameterError

        raise InvalidParameterError(
            f"status must be one of {', '.join((*APPROVAL_STATUSES, DECIDED_FILTER))}",
            details={"param": "status"})
    await expire_due(session)
    query = sa.select(ApprovalRequest)
    if status == DECIDED_FILTER:
        query = query.where(ApprovalRequest.status != "pending").order_by(
            sa.func.coalesce(ApprovalRequest.decided_at, ApprovalRequest.created_at).desc(),
            ApprovalRequest.id.desc())
    else:
        if status is not None:
            query = query.where(ApprovalRequest.status == status)
        query = query.order_by(ApprovalRequest.created_at.desc(), ApprovalRequest.id.desc())
    rows = list((await session.execute(query.offset(offset).limit(limit))).scalars())
    pending = int((await session.execute(
        sa.select(sa.func.count()).select_from(ApprovalRequest)
        .where(ApprovalRequest.status == "pending")
    )).scalar_one())
    return rows, pending


async def _load(session: AsyncSession, approval_id: str) -> ApprovalRequest:
    approval = await session.get(ApprovalRequest, approval_id, populate_existing=True)
    if approval is None:
        raise ApprovalNotFoundError(f"No approval request {approval_id!r}.",
                                    details={"approval_id": approval_id})
    return approval


async def get(session: AsyncSession, approval_id: str) -> ApprovalRequest:
    await expire_due(session)
    return await _load(session, approval_id)


def _not_pending(approval: ApprovalRequest) -> MiLLMError:
    if approval.status == "expired":
        return ApprovalExpiredError(
            f"Approval request {approval.id} expired at {_iso(approval.expires_at)} and will "
            f"never run. The agent must ask again.",
            details={"approval_id": approval.id, "status": approval.status})
    return ApprovalNotPendingError(
        f"Approval request {approval.id} is {approval.status}; only a pending request can be "
        f"decided.",
        details={"approval_id": approval.id, "status": approval.status})


async def _claim(session: AsyncSession, approval_id: str, to_status: str,
                 **values: Any) -> bool:
    """Move a row from ``pending`` to ``to_status`` iff it is still pending. Commits.

    ⚠ THE EXACTLY-ONCE GUARANTEE IS THIS ``WHERE status = 'pending'`` AND ITS ROW COUNT.
    """
    result = await session.execute(
        sa.update(ApprovalRequest)
        .where(ApprovalRequest.id == approval_id, ApprovalRequest.status == "pending")
        .values(status=to_status, **values)
        .execution_options(synchronize_session=False)
    )
    await session.commit()
    return int(getattr(result, "rowcount", 0) or 0) == 1


async def approve(session: AsyncSession, approval_id: str, *, app: Any,
                  decided_by: str = "operator",
                  wait_s: float | None = None,
                  batch_id: str | None = None) -> ApprovalRequest:
    """Run the stored request exactly once and record the outcome.

    In order: lazy expiry; (1) the digest check, a mismatch is ``failed``/``approval_tampered``;
    (2) the claim, ``pending`` -> ``executing``, committed; (3) the executor, as an ``asyncio``
    task under the requester's identity; (4) ``executed`` with the result, or ``failed`` with the
    error. Waits up to ``APPROVE_WAIT_S`` for (4) and answers with the row as it then stands.

    ``batch_id`` is set only by :func:`approve_batch` and is recorded on the row with the decision.
    """
    from millm.api.approval_gate import EXECUTORS

    await expire_due(session)
    approval = await _load(session, approval_id)
    if approval.status != "pending":
        raise _not_pending(approval)
    executor = EXECUTORS.get(approval.executor_key)
    if executor is None:
        raise ApprovalExecutorMissingError(
            f"Approval request {approval_id} names {approval.executor_key!r}, which this server "
            f"no longer serves. Reject it.",
            details={"approval_id": approval_id})
    payload = dict(approval.payload or {})
    secret = dict(approval.secret_payload or {})
    if digest(payload) != approval.request_digest:
        error = {"code": "approval_tampered",
                 "message": "The stored request no longer matches the digest taken when it was "
                            "made; it will not run.",
                 "details": {}}
        if await _claim(session, approval_id, "failed", error=error, decided_by=decided_by,
                        decided_at=now(), secret_payload=None, batch_id=batch_id):
            _emit_resolved(approval_id, "failed")
        raise ApprovalTamperedError(str(error["message"]), details={"approval_id": approval_id})

    # ---- THE CLAIM. Exactly one caller gets past this line for a given row. ----------------
    if not await _claim(session, approval_id, "executing", decided_by=decided_by,
                        decided_at=now(), batch_id=batch_id):
        raise _not_pending(await _load(session, approval_id))

    task = asyncio.create_task(_execute(approval_id, approval.requested_by, executor, payload,
                                        secret, app))
    RUNNING[approval_id] = task
    task.add_done_callback(lambda _t: RUNNING.pop(approval_id, None))
    await asyncio.wait({task}, timeout=APPROVE_WAIT_S if wait_s is None else wait_s)
    return await _load(session, approval_id)


def new_batch_id() -> str:
    return f"bat_{uuid.uuid4().hex}"


#: The most ids one "Approve all settings changes" may name.
BATCH_MAX = 200


#: What a refusal calls an id of the wrong kind, and what the batch is called, per kind.
_KIND_WORDS = {
    "settings": ("not_a_settings_change", "Approve all settings changes",
                 "waiting settings changes"),
    "code_rules": ("not_a_code_rule", "Approve all code rules",
                   "waiting imports of operator code"),
}


async def approve_batch(session: AsyncSession, approval_ids: list[str], *, app: Any,
                        decided_by: str = "operator",
                        kind: str = "settings") -> tuple[str, list[dict[str, Any]]]:
    """Approve every named waiting request of ONE batch kind with one decision: settings
    changes (ENH-006 D79) or code-rule imports (D81). Kinds are never mixed.

    1. **All or nothing on the ids.** Every id must name a ``pending`` request whose action is of
       ``kind`` (``approval_gate.batch_actions(kind)``). Otherwise ``approval_batch_refused``
       (409) names each bad id and why, and NOTHING is approved.
    2. **Each through the single approve.** Each id then goes through :func:`approve`, the same
       claim, digest check and executor as approving it on its own, in the order given, so each
       runs exactly once and is recorded on its own row (``decided_by``, ``decided_at``, status,
       result), with the batch's ``batch_id``.
    3. **One failure does not stop the rest.** An item that fails while running (or that another
       caller decided between the check and its turn) is reported in its result; the others still
       run.

    Returns ``(batch_id, [{approval_id, status, error}])`` in the order given.
    """
    from millm.api.approval_gate import batch_actions
    from millm.core.errors import InvalidParameterError

    if kind not in _KIND_WORDS:
        raise InvalidParameterError(f"kind must be one of {', '.join(_KIND_WORDS)}.",
                                    details={"param": "kind"})
    wrong_kind, batch_name, batch_scope = _KIND_WORDS[kind]
    ids = list(dict.fromkeys(approval_ids))
    if not ids:
        raise InvalidParameterError("approval_ids must name at least one request.",
                                    details={"param": "approval_ids"})
    if len(ids) > BATCH_MAX:
        raise InvalidParameterError(f"approval_ids may name at most {BATCH_MAX} requests.",
                                    details={"param": "approval_ids"})
    await expire_due(session)
    allowed = batch_actions(kind)
    found = {r.id: r for r in (await session.execute(
        sa.select(ApprovalRequest).where(ApprovalRequest.id.in_(ids))
        .execution_options(populate_existing=True))).scalars()}
    refused: list[dict[str, Any]] = []
    for approval_id in ids:
        row = found.get(approval_id)
        if row is None:
            refused.append({"approval_id": approval_id, "reason": "not_found"})
        elif row.status != "pending":
            refused.append({"approval_id": approval_id, "reason": f"not_pending ({row.status})"})
        elif row.action not in allowed:
            refused.append({"approval_id": approval_id,
                            "reason": f"{wrong_kind} ({row.action})"})
    if refused:
        from millm.core.errors import ApprovalBatchRefusedError

        raise ApprovalBatchRefusedError(
            "Nothing was approved: "
            + "; ".join(f"{r['approval_id']} is {r['reason']}" for r in refused)
            + f". \"{batch_name}\" approves only {batch_scope}.",
            details={"refused": refused, "kind": kind})

    batch_id = new_batch_id()
    results: list[dict[str, Any]] = []
    for approval_id in ids:
        try:
            row = await approve(session, approval_id, app=app, decided_by=decided_by,
                                batch_id=batch_id)
            results.append({"approval_id": approval_id, "status": row.status,
                             "error": row.error})
        except MiLLMError as exc:
            # Decided by someone else since the check, expired, tampered: reported, never fatal.
            results.append({"approval_id": approval_id, "status": "not_run",
                            "error": {"code": exc.code, "message": exc.message,
                                      "details": jsonable_encoder(exc.details)}})
    logger.info("approval_batch_decided", batch_id=batch_id, kind=kind, count=len(ids),
                statuses=[r["status"] for r in results])
    return batch_id, results


async def _execute(approval_id: str, requested_by: str, executor: Any, payload: dict[str, Any],
                   secret: dict[str, Any], app: Any) -> None:
    """Run the stored request through the ORIGINAL route function and record the outcome."""
    from millm.api.approval_gate import (
        BODY_KEY,
        Actor,
        DependencyResolver,
        executing_as,
        synthetic_request,
    )
    from millm.db import base

    status = "executed"
    result: Any = None
    error: Any = None
    body = payload.get(BODY_KEY)
    request = synthetic_request(app, body.encode("utf-8") if isinstance(body, str) else b"")
    resolver = DependencyResolver(app, request)
    try:
        kwargs: dict[str, Any] = {}
        for name, annotation in executor.value_params.items():
            source = secret if name in executor.secret_params else payload
            kwargs[name] = TypeAdapter(annotation).validate_python(source.get(name))
        if executor.request_param is not None:
            kwargs[executor.request_param] = request
        for name, dependency in executor.dependency_params.items():
            kwargs[name] = await resolver.resolve(dependency)
        with executing_as(Actor(origin="agent", agent=requested_by, approval_id=approval_id)):
            outcome = await executor.fn(**kwargs)
        result = _result_of(outcome)
    except MiLLMError as exc:
        status = "failed"
        safe = exc.status_code < 500 or exc.message_is_client_safe
        error = {"code": exc.code,
                 "message": exc.message if safe else "The approved request failed on the "
                                                     "server. Check the server log.",
                 "details": jsonable_encoder(exc.details) if safe else {}}
    except HTTPException as exc:
        status = "failed"
        error = {"code": f"http_{exc.status_code}", "message": str(exc.detail), "details": {}}
    except Exception as exc:  # noqa: BLE001 - recorded, never raised out of a background task
        status = "failed"
        logger.error("approval_execution_failed", approval_id=approval_id,
                     error_type=type(exc).__name__, exc_info=True)
        error = {"code": "execution_failed",
                 "message": "The approved request failed unexpectedly. Check the server log "
                            "for the correlated entry.",
                 "details": {"error_type": type(exc).__name__}}
    finally:
        try:
            await resolver.aclose()
        except Exception:  # noqa: BLE001 - closing a session must not lose the outcome
            logger.warning("approval_resolver_close_failed", approval_id=approval_id)
    async with base.async_session_factory() as session:
        await session.execute(
            sa.update(ApprovalRequest)
            .where(ApprovalRequest.id == approval_id, ApprovalRequest.status == "executing")
            .values(status=status, result=result, error=error, secret_payload=None)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    logger.info("approval_decided", approval_id=approval_id, status=status)
    _emit_resolved(approval_id, status)


def _result_of(outcome: Any) -> Any:
    """What History shows as the result: the route's ``data`` (never a lease id: the model
    serialiser carries the lease's holder and expiry only)."""
    data = getattr(outcome, "data", outcome)
    encoded = jsonable_encoder(data)
    return {"value": None} if encoded is None else encoded


async def reject(session: AsyncSession, approval_id: str, *, reason: str | None = None,
                 decided_by: str = "operator") -> ApprovalRequest:
    """Reject a pending request. It will never run."""
    await expire_due(session)
    approval = await _load(session, approval_id)
    if approval.status != "pending":
        raise _not_pending(approval)
    if not await _claim(session, approval_id, "rejected", decided_by=decided_by,
                        decided_at=now(), reason=reason, secret_payload=None):
        raise _not_pending(await _load(session, approval_id))
    _emit_resolved(approval_id, "rejected")
    return await _load(session, approval_id)


async def wait_all() -> None:
    """Await every executor still running (tests; shutdown)."""
    while RUNNING:
        await asyncio.gather(*list(RUNNING.values()), return_exceptions=True)
