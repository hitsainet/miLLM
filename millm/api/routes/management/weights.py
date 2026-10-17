"""Weight attachment and the step barrier (Feature 31): attach, detach, heartbeat, quiesce,
resume, commit-begin, commit-end, clear-stale and status (031 FTDD §5.2, FTID §5.1).

Every gated route has the same shape: ONE identity dependency (`WeightsIdentityDep`, or
`OperatorIdentityDep` for clear-stale), ONE call into `WeightService`, ONE audit line. Refusals
raised by the dependency (401/403) are audited there; refusals raised by the service are audited
here by `_audited` and re-raised to the management error handler.

The holder routes read `X-miLLM-Attachment`, declared `str | None`: a missing header is the
service's `NOT_ATTACHED` (409) or `NOT_ATTACHMENT_HOLDER` (403), never FastAPI's 422 (031
review I11). The attachment ID is never in a path or a query, because paths reach access logs.

`GET /status` takes no token and no slot, and never returns the attachment ID or a token
(FR-31.16.2).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Annotated, Any, Iterator, Optional

from fastapi import APIRouter, Header

from millm.api.schemas.common import ApiResponse
from millm.api.schemas.weights import (
    AttachResponse,
    ClearStaleResponse,
    CommitBeginResponse,
    CommitEndResponse,
    DetachResponse,
    HeartbeatResponse,
    QuiesceResponse,
    ResumeResponse,
    WeightsStatus,
)
from millm.core.errors import MiLLMError
from millm.core.weights_auth import OperatorIdentityDep, WeightsIdentity, WeightsIdentityDep
from millm.services.weight_attachment import get_weight_service
from millm.services.weight_barrier import get_weight_barrier
from millm.services.weights_audit import weights_audit

router = APIRouter(prefix="/api/weights", tags=["weights"])

#: The attachment ID, carried as a header and never in a path (031 FTDD §5.1).
AttachmentHeader = Annotated[
    Optional[str],
    Header(alias="X-miLLM-Attachment", description="The attachment ID the attach call returned"),
]


@contextlib.contextmanager
def _audited(operation: str, identity: WeightsIdentity, ref: Optional[str]) -> Iterator[None]:
    """Audit a refusal the service raised, with its code and reason, then re-raise it
    (FR-31.15.3, FTID §5.1)."""
    try:
        yield
    except MiLLMError as exc:
        reason = exc.details.get("reason") if isinstance(exc.details, dict) else None
        weights_audit(
            operation,
            "refused",
            identity,
            code=exc.code,
            reason=reason if isinstance(reason, str) else None,
            attachment_ref=ref,
        )
        raise
    except Exception as exc:
        # An unexpected exception (a 500: a torch error from `_share_cuda_` on a real card, a
        # bug) still owes the call its one audit line (FR-31.15), as 033's routes write it. Only
        # the exception TYPE: its message is not ours and may carry a value.
        weights_audit(
            operation,
            "refused",
            identity,
            code="INTERNAL_ERROR",
            reason=type(exc).__name__,
            attachment_ref=ref,
        )
        raise
    except asyncio.CancelledError:
        # A cancelled call (client gone, shutdown) is audited too, then the cancellation goes on.
        weights_audit(
            operation, "refused", identity, code="CANCELLED", reason="cancelled",
            attachment_ref=ref,
        )
        raise


def _holder_ref(identity: WeightsIdentity, attachment_id: Optional[str]) -> Optional[str]:
    """The 8-hex ref of the attachment when this caller holds it, for the audit line; never
    raises (the service call that follows makes the decision)."""
    attachment = get_weight_service().holder_of(identity, attachment_id)
    return attachment.ref if attachment is not None else None


def _state_now() -> str:
    return get_weight_barrier().snapshot().weights_state


@router.post(
    "/attach",
    response_model=ApiResponse[AttachResponse],
    name="weights_attach",
    summary="Attach to the resident weights",
    description=(
        "Returns a CUDA IPC handle per parameter and the attachment ID, once. Needs a trainer "
        "or operator token. One attachment at a time."
    ),
)
async def attach(identity: WeightsIdentityDep) -> ApiResponse[AttachResponse]:
    before = _state_now()
    with _audited("weights_attach", identity, None):
        result = await get_weight_service().attach(identity)
    weights_audit(
        "weights_attach", "accepted", identity, attachment_ref=result.ref, state_before=before
    )
    return ApiResponse.ok(AttachResponse.model_validate(result.to_dict()))


@router.post(
    "/detach",
    response_model=ApiResponse[DetachResponse],
    name="weights_detach",
    summary="Detach (the weight values are kept)",
)
async def detach(
    identity: WeightsIdentityDep, x_millm_attachment: AttachmentHeader = None
) -> ApiResponse[DetachResponse]:
    before, ref = _state_now(), _holder_ref(identity, x_millm_attachment)
    with _audited("weights_detach", identity, ref):
        body = await get_weight_service().detach(identity, x_millm_attachment)
    weights_audit("weights_detach", "accepted", identity, attachment_ref=ref, state_before=before)
    return ApiResponse.ok(DetachResponse.model_validate(body))


@router.post(
    "/heartbeat",
    response_model=ApiResponse[HeartbeatResponse],
    name="weights_heartbeat",
    summary="Prove the attached process is alive",
)
async def heartbeat(
    identity: WeightsIdentityDep, x_millm_attachment: AttachmentHeader = None
) -> ApiResponse[HeartbeatResponse]:
    before, ref = _state_now(), _holder_ref(identity, x_millm_attachment)
    with _audited("weights_heartbeat", identity, ref):
        body = get_weight_service().heartbeat(identity, x_millm_attachment)
    weights_audit(
        "weights_heartbeat", "accepted", identity, attachment_ref=ref, state_before=before
    )
    return ApiResponse.ok(HeartbeatResponse.model_validate(body))


@router.post(
    "/quiesce",
    response_model=ApiResponse[QuiesceResponse],
    name="weights_quiesce",
    summary="Drain in-flight requests and hold new ones",
)
async def quiesce(
    identity: WeightsIdentityDep, x_millm_attachment: AttachmentHeader = None
) -> ApiResponse[QuiesceResponse]:
    ref = _holder_ref(identity, x_millm_attachment)
    with _audited("weights_quiesce", identity, ref):
        result = await get_weight_service().quiesce(identity, x_millm_attachment)
    weights_audit(
        "weights_quiesce",
        "accepted",
        identity,
        attachment_ref=ref,
        state_before=result.state_before,
    )
    return ApiResponse.ok(
        QuiesceResponse(
            held_requests=result.held_requests,
            held_batch_chunks=result.held_batch_chunks,
            weights_state=result.weights_state,
            quiesced_at=result.quiesced_at,
            hold_deadline=result.hold_deadline,
        )
    )


@router.post(
    "/resume",
    response_model=ApiResponse[ResumeResponse],
    name="weights_resume",
    summary="Release held requests",
)
async def resume(
    identity: WeightsIdentityDep, x_millm_attachment: AttachmentHeader = None
) -> ApiResponse[ResumeResponse]:
    ref = _holder_ref(identity, x_millm_attachment)
    with _audited("weights_resume", identity, ref):
        result = get_weight_service().resume(identity, x_millm_attachment)
    weights_audit(
        "weights_resume",
        "accepted",
        identity,
        attachment_ref=ref,
        state_before=result.state_before,
    )
    return ApiResponse.ok(
        ResumeResponse(barrier=result.barrier, weights_state=result.weights_state)
    )


@router.post(
    "/commit-begin",
    response_model=ApiResponse[CommitBeginResponse],
    name="weights_commit_begin",
    summary="Open a commit: the weights are writing",
)
async def commit_begin(
    identity: WeightsIdentityDep, x_millm_attachment: AttachmentHeader = None
) -> ApiResponse[CommitBeginResponse]:
    ref = _holder_ref(identity, x_millm_attachment)
    with _audited("weights_commit_begin", identity, ref):
        result = get_weight_service().commit_begin(identity, x_millm_attachment)
    weights_audit(
        "weights_commit_begin",
        "accepted",
        identity,
        attachment_ref=ref,
        state_before=result.state_before,
    )
    return ApiResponse.ok(
        CommitBeginResponse(
            weights_state=result.weights_state,
            began_at=result.began_at,
            watchdog_ms=result.watchdog_ms,
        )
    )


@router.post(
    "/commit-end",
    response_model=ApiResponse[CommitEndResponse],
    name="weights_commit_end",
    summary="Close the commit: the weights are clean",
)
async def commit_end(
    identity: WeightsIdentityDep, x_millm_attachment: AttachmentHeader = None
) -> ApiResponse[CommitEndResponse]:
    ref = _holder_ref(identity, x_millm_attachment)
    with _audited("weights_commit_end", identity, ref):
        result = get_weight_service().commit_end(identity, x_millm_attachment)
    weights_audit(
        "weights_commit_end",
        "accepted",
        identity,
        attachment_ref=ref,
        state_before=result.state_before,
        elapsed_ms=result.commit_ms,
        commit_ms=result.commit_ms,
        steps=result.steps_since_load,
    )
    return ApiResponse.ok(
        CommitEndResponse(
            weights_state=result.weights_state,
            steps_since_load=result.steps_since_load,
            commit_ms=result.commit_ms,
            overran=result.overran,
        )
    )


@router.post(
    "/clear-stale",
    response_model=ApiResponse[ClearStaleResponse],
    name="weights_clear_stale",
    summary="Clear a stale attachment (operator token)",
)
async def clear_stale(identity: OperatorIdentityDep) -> ApiResponse[ClearStaleResponse]:
    before = _state_now()
    with _audited("weights_clear_stale", identity, None):
        body: dict[str, Any] = await get_weight_service().clear_stale(identity)
    weights_audit("weights_clear_stale", "accepted", identity, state_before=before)
    return ApiResponse.ok(ClearStaleResponse.model_validate(body))


@router.get(
    "/status",
    response_model=ApiResponse[WeightsStatus],
    name="weights_status",
    summary="Weight state and attachment (no token)",
    description=(
        "Attached or not, the holder's role and label, quiesced or serving, the weights state, "
        "the resident model and steps since load. Never the attachment ID or a token."
    ),
)
async def status() -> ApiResponse[WeightsStatus]:
    return ApiResponse.ok(get_weight_service().status())
