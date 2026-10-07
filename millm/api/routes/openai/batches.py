"""OpenAI-shaped batches routes (Feature 26, FR-26.1.2 - 26.1.7, FR-26.6, FR-26.7).

POST /v1/batches                 create; synchronous checks only, then `validating`
GET  /v1/batches                 list, newest first
GET  /v1/batches/{id}            the batch object, counts current to the last recorded chunk
POST /v1/batches/{id}/cancel     `cancelling`; 409 on a batch that can no longer be cancelled
POST /v1/batches/{id}/lease      miLLM extension (FR-26.7.7, X-08): hand a live lease to a batch

Create checks, in order: the input file (exists, purpose `batch`, not expired or deleted); the
endpoint (one this app SERVES, read from `app.openapi()` — FR-26.1.4); `completion_window`
(`<N>h`, 1..BATCH_MAX_COMPLETION_WINDOW_HOURS — T-65); `output_expires_after` (OpenAI's
bounds); the lease (409 `model_leased` unless `X-miLLM-Lease` proves it — FR-26.7.3). Then the
row is written and validation starts in the background; validation takes no slot (FR-26.2.9).
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Header, Query, Request, Response

from millm.api.dependencies import DbSession, ModelServiceDep, get_inference_service
from millm.api.request_policy import (
    IGNORED_FIELDS_HEADER,
    STRICT_HEADER,
    evaluate_control,
    ignored_fields_header,
    parse_strict,
)
from millm.api.schemas.batch import (
    OUTPUT_EXPIRES_MAX_S,
    OUTPUT_EXPIRES_MIN_S,
    BatchCreateRequest,
)
from millm.core.batch_values import TERMINAL, BatchStatus, FilePurpose, FileStatus
from millm.core.config import settings
from millm.core.errors import (
    BatchFileDeletedError,
    BatchFileExpiredError,
    BatchFileNotFoundError,
    BatchNotFoundError,
    BatchStateConflictError,
    InvalidBatchRequestError,
    InvalidLeaseRequestError,
    LeaseNotFoundError,
    ModelLeasedError,
)
from millm.core.logging import get_logger
from millm.db.models.batch import Batch
from millm.db.repositories.batch_repository import BatchRepository
from millm.services.batch.files import new_batch_id
from millm.services.batch.runner import BatchRunner, get_batch_runner
from millm.services.batch.state import (
    SUPPORTED_ENDPOINTS,
    batch_object,
    can_transition,
    openai_list,
    served_batch_endpoints,
    transition,
    utcnow,
)
from millm.services.inference_service import InferenceService
from millm.services.model_lease import UNKNOWN_LEASE_MESSAGE

router = APIRouter()
logger = get_logger(__name__)

WINDOW = re.compile(r"^([1-9][0-9]*)h$")


def runner_dependency() -> BatchRunner:
    return get_batch_runner()


RunnerDep = Annotated[BatchRunner, Depends(runner_dependency)]


def parse_window(value: str) -> int:
    """`<N>h` → N hours, 1..BATCH_MAX_COMPLETION_WINDOW_HOURS (T-65). Anything else is refused."""
    maximum = int(settings.BATCH_MAX_COMPLETION_WINDOW_HOURS)
    match = WINDOW.match(value)
    hours = int(match.group(1)) if match else 0
    if not match or not 1 <= hours <= maximum:
        raise InvalidBatchRequestError(
            f"completion_window must be '24h' (OpenAI's value) or whole hours '<N>h' from 1h to "
            f"{maximum}h (a miLLM extension); got {value!r}.",
            details={"param": "completion_window", "min_hours": 1, "max_hours": maximum},
        )
    return hours


async def _get(repo: BatchRepository, batch_id: str) -> Batch:
    batch = await repo.get_batch(batch_id)
    if batch is None:
        raise BatchNotFoundError(f"No batch {batch_id!r}.", details={"param": "batch_id"})
    return batch


@router.post("/batches")
async def create_batch(
    body: BatchCreateRequest,
    request: Request,
    response: Response,
    session: DbSession,
    service: ModelServiceDep,
    runner: RunnerDep,
    inference: InferenceService = Depends(get_inference_service),
    x_millm_lease: Annotated[Optional[str], Header(alias="X-miLLM-Lease")] = None,
    x_millm_load_policy: Annotated[Optional[str], Header(alias="X-miLLM-Load-Policy")] = None,
) -> dict[str, Any]:
    """Create a batch. `X-miLLM-Load-Policy` is accepted and needs no action: a batch never loads
    a model, whatever it says (FR-26.1.7, FR-26.7.1)."""
    policy = evaluate_control(
        body, "/v1/batches", strict=parse_strict(request.headers.get(STRICT_HEADER))
    )
    repo = BatchRepository(session)
    in_file = await repo.get_file(body.input_file_id)
    if in_file is None:
        raise BatchFileNotFoundError(
            f"No file {body.input_file_id!r}.", details={"param": "input_file_id"}
        )
    if in_file.purpose != FilePurpose.BATCH.value:
        raise InvalidBatchRequestError(
            f"File {in_file.id} has purpose {in_file.purpose!r}; a batch input needs 'batch'.",
            details={"param": "input_file_id"},
        )
    if in_file.status == FileStatus.EXPIRED.value:
        raise BatchFileExpiredError(f"File {in_file.id} has expired.", details={"param": "input_file_id"})
    if in_file.status == FileStatus.DELETED.value:
        raise BatchFileDeletedError(f"File {in_file.id} was deleted.", details={"param": "input_file_id"})

    served = served_batch_endpoints(request.app)
    if body.endpoint not in served:
        raise InvalidBatchRequestError(
            f"endpoint must be one of {sorted(served)}; got {body.endpoint!r}."
            + (
                f" ({body.endpoint} is supported once its route is served.)"
                if body.endpoint in SUPPORTED_ENDPOINTS else ""
            ),
            details={"param": "endpoint", "supported": sorted(served)},
        )
    hours = parse_window(body.completion_window)
    expires_after = (
        body.output_expires_after.seconds if body.output_expires_after else 2_592_000
    )
    if not OUTPUT_EXPIRES_MIN_S <= expires_after <= OUTPUT_EXPIRES_MAX_S:
        raise InvalidBatchRequestError(
            f"output_expires_after.seconds must be from {OUTPUT_EXPIRES_MIN_S} to "
            f"{OUTPUT_EXPIRES_MAX_S}; got {expires_after}.",
            details={"param": "output_expires_after.seconds"},
        )

    # FR-26.7.3: one lease exists per server, on the resident model (029 FR-29.1.3), so the check
    # reads no line. A live lease refuses the create unless the request PRESENTS it (T-64).
    caller = service.resolve_lease(x_millm_lease) if x_millm_lease else None
    info = inference.get_loaded_model_info()
    if info is not None:
        live, _ended = await service.get_lease(info.model_id)
        if live is not None and (caller is None or caller.digest != live.digest):
            raise ModelLeasedError.for_lease(
                model_id=live.model_id, model_name=live.model_name, holder=live.holder,
                reason=live.reason, expires_at=live.expires_at.isoformat(), operation="batch",
                target_model_id=live.model_id,
            )

    now = utcnow()
    batch = Batch(
        id=new_batch_id(), endpoint=body.endpoint, completion_window=body.completion_window,
        status=BatchStatus.VALIDATING.value, input_file_id=in_file.id,
        pack=bool(settings.BATCH_PACK_DEFAULT if body.pack is None else body.pack),
        batch_metadata=body.metadata, output_expires_after_s=expires_after,
        created_at=now, expires_at=now + timedelta(hours=hours),
        lease_mode="caller" if caller is not None else None,
    )
    await repo.create_batch(batch)
    if caller is not None:
        runner.set_caller_lease(batch.id, x_millm_lease, caller.ttl_seconds)
    runner.submit_validation(batch.id)
    logger.info("batch_created", batch_id=batch.id, endpoint=batch.endpoint, pack=batch.pack,
                window_hours=hours, lease_mode=batch.lease_mode)
    header = ignored_fields_header(policy)
    if header:
        response.headers[IGNORED_FIELDS_HEADER] = header
    return batch_object(batch)


@router.get("/batches")
async def list_batches(
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    after: Annotated[Optional[str], Query()] = None,
) -> dict[str, Any]:
    rows, more = await BatchRepository(session).list_batches(limit=limit, after=after)
    return openai_list([batch_object(r) for r in rows], more)


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str, session: DbSession) -> dict[str, Any]:
    return batch_object(await _get(BatchRepository(session), batch_id))


@router.post("/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, session: DbSession, runner: RunnerDep) -> dict[str, Any]:
    """`cancelling`, then `cancelled` once the forward already running finishes (FR-26.6.4)."""
    repo = BatchRepository(session)
    batch = await _get(repo, batch_id)
    if batch.status == BatchStatus.CANCELLING.value:
        return batch_object(batch)
    if not can_transition(batch.status, BatchStatus.CANCELLING):
        raise BatchStateConflictError(
            f"Batch {batch_id} is {batch.status} and can no longer be cancelled.",
            details={"param": "batch_id", "status": batch.status},
        )
    transition(batch, BatchStatus.CANCELLING, utcnow())
    await repo.save()
    runner.request_cancel(batch_id)
    runner.emit(batch, force=True)
    return batch_object(batch)


@router.post("/batches/{batch_id}/lease")
async def hand_over_lease(
    batch_id: str,
    session: DbSession,
    service: ModelServiceDep,
    runner: RunnerDep,
    x_millm_lease: Annotated[Optional[str], Header(alias="X-miLLM-Lease")] = None,
) -> dict[str, Any]:
    """miLLM extension (FR-26.7.7): run a waiting batch under the caller's live lease (T-64)."""
    if not x_millm_lease or not x_millm_lease.strip():
        raise InvalidLeaseRequestError(
            "The lease ID is required in the X-miLLM-Lease header.",
            details={"param": "X-miLLM-Lease"},
        )
    repo = BatchRepository(session)
    batch = await _get(repo, batch_id)
    if batch.status in {s.value for s in TERMINAL}:
        raise BatchStateConflictError(
            f"Batch {batch_id} is {batch.status}; a lease cannot be handed to it.",
            details={"param": "batch_id", "status": batch.status},
        )
    record = service.resolve_lease(x_millm_lease)
    if record is None or (batch.model_id is not None and record.model_id != batch.model_id):
        # The same 404 Feature 29 answers for an unknown lease or one on ANOTHER model.
        raise LeaseNotFoundError(
            f"Batch {batch_id}: {UNKNOWN_LEASE_MESSAGE}.", details={"param": "X-miLLM-Lease"}
        )
    batch.lease_mode = "caller"
    await repo.save()
    runner.set_caller_lease(batch_id, x_millm_lease, record.ttl_seconds)
    logger.info("batch_lease", action="handed_over", batch_id=batch_id, holder=record.holder)
    return batch_object(batch)
