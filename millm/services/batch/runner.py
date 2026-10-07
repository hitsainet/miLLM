"""The batch runner: one asyncio task in the API process (Feature 26, FTDD §6-§7, FTID §3).

Loop, FIFO by `created_at`:

    pick the oldest batch needing work
      cancelling → assemble files → cancelled
      finalizing → (re-)assemble files → completed
      in_progress →
        past expires_at → assemble with unrun rows `batch_expired` → expired
        no live lease → acquire one, or wait (`model_not_resident` / `lease_unavailable`)
        next chunk of pending rows → ONE slot (`_admit(background=True)`) → run rows through the
        synchronous service code → record the chunk in ONE transaction → emit `batch:progress`
        no pending rows → finalizing → assemble → completed

Rules, each with a test that fails when it is removed:

* **Never loads, unloads or swaps a model** (FR-26.7.1). The only model-service calls are the
  Feature 29 lease calls and row lookups.
* **Never runs a row unleased** (FR-26.3.6, X-01). A restart forgets every lease, so a resumed
  batch acquires its own or waits; a caller's lease handed over by `POST /v1/batches/{id}/lease`
  is renewed and NEVER released (T-64). An OWN lease is released on every terminal path, in a
  `finally`.
* **Recorded means committed** (FR-26.3.2). A chunk interrupted before its commit leaves its rows
  `pending`; they run again and nothing was counted.
* **An unloading refusal is a wait, never a failure** (FTASKS 2.6, 5.6). Any OTHER exception in a
  chunk fails the batch, naming the exception class, and the own lease is released.

In-memory state (lease ids, cancel flags, emit times, the backlog) is rebuilt at startup by
`reconcile_batches_on_startup` — every in-memory flag here has a reconciliation (this estate has
shipped "a new in-memory flag left out of startup reconciliation" three times).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from millm.core.batch_values import (
    TERMINAL,
    BatchStatus,
    FilePurpose,
    FileStatus,
    RowKind,
    RowState,
    WaitingReason,
)
from millm.core.config import settings
from millm.core.errors import (
    LeaseExpiredError,
    LeaseNotFoundError,
    MiLLMError,
    ModelBusyError,
    ModelLeasedError,
    ModelNotResidentError,
)
from millm.core.logging import get_logger
from millm.db.models.batch import Batch, BatchFile
from millm.db.repositories.batch_repository import BatchRepository
from millm.services.batch.files import FileStore, get_file_store, new_file_id
from millm.services.batch.state import (
    batch_object,
    error_line,
    transition,
    utcnow,
)

logger = get_logger(__name__)

OWN = "own"
CALLER = "caller"

#: The unrun-row codes OpenAI writes to the error file (FTDD TD14).
_UNRUN_CODE = {
    BatchStatus.CANCELLED: ("batch_cancelled", "The batch was cancelled before this row ran."),
    BatchStatus.EXPIRED: ("batch_expired", "The batch expired before this row ran."),
    BatchStatus.FAILED: ("batch_failed", "The batch failed before this row ran."),
}


def aware(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite drops tzinfo; every stored time is UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@dataclass
class _Memory:
    """Per-batch state that lives ONLY in this process (the lease ID is never persisted)."""

    lease_id: Optional[str] = None
    lease_mode: Optional[str] = None
    lease_ttl: int = 0
    renewed_mono: float = 0.0
    cancel: threading.Event = field(default_factory=threading.Event)
    last_emit: float = 0.0


class BatchRunner:
    """One per process (`get_batch_runner()`)."""

    def __init__(
        self,
        session_factory: Any,
        *,
        inference_provider: Callable[[], Any],
        model_service_factory: Callable[[Any], Any],
        store_factory: Callable[[], FileStore] = get_file_store,
        emitter: Any = None,
        now: Callable[[], datetime] = utcnow,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_factory = session_factory
        self.inference_provider = inference_provider
        self.model_service_factory = model_service_factory
        self.store_factory = store_factory
        self.emitter = emitter
        self.now = now
        self.monotonic = monotonic
        self._memory: dict[str, _Memory] = {}
        self._backlog: dict[str, int] = {}
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._validations: set[asyncio.Task] = set()
        #: The retention loop, started by `start_batch_api` beside the runner.
        self._retention: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the loop and register the backlog with Feature 29's health (FR-26.4.7)."""
        from millm.core.backpressure import register_backlog_provider

        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop())
        register_backlog_provider(self.backlog_rows)

    async def stop(self) -> None:
        from millm.core.backpressure import register_backlog_provider

        register_backlog_provider(None)
        tasks = [t for t in [self._task, *self._validations] if t is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown
                pass
        self._task = None

    def wake(self) -> None:
        self._wake.set()

    def memory(self, batch_id: str) -> _Memory:
        return self._memory.setdefault(batch_id, _Memory())

    def backlog_rows(self) -> int:
        """Rows not yet run across active batches, from counters (FTID §13: no scan)."""
        return sum(self._backlog.values())

    def set_backlog(self, batch_id: str, rows: int) -> None:
        if rows > 0:
            self._backlog[batch_id] = rows
        else:
            self._backlog.pop(batch_id, None)

    # --------------------------------------------------------------- route hooks

    def submit_validation(self, batch_id: str) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(self.validate_batch(batch_id))
        self._validations.add(task)
        task.add_done_callback(self._validations.discard)
        return task

    def request_cancel(self, batch_id: str) -> None:
        self.memory(batch_id).cancel.set()
        self.wake()

    def set_caller_lease(self, batch_id: str, lease_id: str, ttl_seconds: int) -> None:
        """Run under the caller's lease (T-64, FR-26.7.5/26.7.7). Held in memory only."""
        mem = self.memory(batch_id)
        mem.lease_id = lease_id
        mem.lease_mode = CALLER
        mem.lease_ttl = int(ttl_seconds)
        mem.renewed_mono = self.monotonic()
        self.wake()

    # ------------------------------------------------------------------- emitting

    def emit(self, batch: Batch, *, force: bool = False) -> None:
        """`batch:progress` (FR-26.6.2): every transition (`force`), chunks throttled."""
        mem = self.memory(batch.id)
        now = self.monotonic()
        if not force and now - mem.last_emit < float(settings.BATCH_PROGRESS_MIN_INTERVAL_S):
            return
        mem.last_emit = now
        if self.emitter is None:
            return
        obj = batch_object(batch)
        try:
            self.emitter.emit_batch_progress(
                {"id": obj["id"], "status": obj["status"], "request_counts": obj["request_counts"]}
            )
        except Exception as e:  # noqa: BLE001 - never fail a batch over a socket
            logger.warning("batch_progress_emit_failed", error=str(e))

    # ----------------------------------------------------------------- validation

    async def validate_batch(self, batch_id: str) -> None:
        """Validate a `validating` batch, then move it to `in_progress`, `failed` or `cancelled`."""
        from millm.services.batch.validator import ValidationCancelled, validate_file

        store = self.store_factory()
        async with self.session_factory() as session:
            repo = BatchRepository(session)
            batch = await repo.get_batch(batch_id)
            if batch is None or batch.status != BatchStatus.VALIDATING.value:
                return
            in_file = await repo.get_file(batch.input_file_id) if batch.input_file_id else None
            endpoint = batch.endpoint
        mem = self.memory(batch_id)
        problem = await self._input_problem(in_file, store)
        if problem is not None:
            await self._fail_validation(batch_id, [problem], rows=None)
            return
        inference = self.inference_provider()
        try:
            async with self.session_factory() as session:
                service = self.model_service_factory(session)
                outcome = await validate_file(
                    lines=store.iter_lines(in_file.storage_path),
                    endpoint=endpoint,
                    resolve_model=service.find_model_by_name,
                    resident_row=lambda: self._resident_row(service),
                    cbm_enabled=inference.cbm_enabled() is True,
                    cancelled=mem.cancel.is_set,
                )
        except ValidationCancelled:
            await self._end_cancelled_while_validating(batch_id)
            return
        await self._finish_validation(batch_id, outcome)

    async def _input_problem(self, in_file: Optional[BatchFile], store: FileStore) -> Optional[dict]:
        if in_file is None or in_file.status != FileStatus.PROCESSED.value or not store.exists(
            in_file.storage_path
        ):
            return {"code": "input_file_missing", "line": None, "param": "input_file_id",
                    "message": "The input file is gone (deleted or expired)."}
        digest = await asyncio.to_thread(store.sha256, in_file.storage_path)
        if digest != in_file.sha256:
            # A different file would break "the first row without a recorded result" (FTDD §8).
            return {"code": "input_file_changed", "line": None, "param": "input_file_id",
                    "message": "The input file's bytes no longer match the sha256 recorded at "
                               "upload; the batch cannot run against a different file."}
        return None

    async def _resident_row(self, service: Any) -> Any:
        info = self.inference_provider().get_loaded_model_info()
        if info is None:
            return None
        return await service.repository.get_by_id(info.model_id)

    async def _finish_validation(self, batch_id: str, outcome: Any) -> None:
        async with self.session_factory() as session:
            repo = BatchRepository(session)
            batch = await repo.get_batch(batch_id)
            if batch is None:
                return
            if batch.status == BatchStatus.CANCELLING.value:
                # Cancelled while validating: no row ever ran (FR-26.6.4).
                transition(batch, BatchStatus.CANCELLED, self.now())
                await repo.save()
                self.emit(batch, force=True)
                return
            # A re-validation after a restart replaces whatever rows a crashed pass inserted.
            await repo.delete_rows(batch_id)
            await repo.bulk_insert_rows([{**r, "batch_id": batch_id} for r in outcome.rows])
            batch.request_total = outcome.total
            batch.request_failed = outcome.invalid
            batch.request_completed = 0
            errors = list(outcome.errors)
            failure: Optional[list[dict]] = None
            if outcome.valid == 0:
                # T-67: a file with no valid line ends `failed`; no lease taken, no row run.
                failure = errors or [{"code": "empty_file", "line": None, "param": "input_file_id",
                                      "message": "The input file has no lines."}]
            elif len(outcome.models) > 1:
                listing = ", ".join(f"{name} ({n} lines)" for name, n in outcome.models.values())
                failure = [{"code": "multiple_models", "line": None, "param": "body.model",
                            "message": f"Every line must name the same model; found {listing}."}]
            else:
                model_id, (model_name, _n) = next(iter(outcome.models.items()))
                batch.model_id, batch.model_name = model_id, model_name
                failure = await self._model_gate(batch)
            if failure is not None:
                batch.errors = {"object": "list", "data": failure}
                await repo.save()
            else:
                batch.errors = {"object": "list", "data": errors} if errors else None
                transition(batch, BatchStatus.IN_PROGRESS, self.now())
                batch.waiting_reason = WaitingReason.QUEUED.value
                await repo.save()
                self.set_backlog(batch_id, outcome.valid)
                logger.info("batch_validated", batch_id=batch_id, valid=outcome.valid,
                            invalid=outcome.invalid)
                self.emit(batch, force=True)
        if failure is not None:
            await self.finalize(batch_id, BatchStatus.FAILED)
        else:
            self.wake()

    async def _model_gate(self, batch: Batch) -> Optional[list[dict]]:
        """FR-26.7.2 (resident) and FR-26.7.3 (lease) at the move to `in_progress`."""
        info = self.inference_provider().get_loaded_model_info()
        if info is None or info.model_id != batch.model_id:
            resident = info.name if info is not None else "none"
            return [{"code": "model_not_resident", "line": None, "param": "body.model",
                     "message": f"The batch's model '{batch.model_name}' is not resident "
                                f"(resident: '{resident}'). A batch never loads a model."}]
        async with self.session_factory() as session:
            service = self.model_service_factory(session)
            live, _ended = await service.get_lease(batch.model_id)
            mem = self.memory(batch.id)
            if live is not None and not (
                mem.lease_id and service.resolve_lease(mem.lease_id) is not None
                and service.resolve_lease(mem.lease_id).digest == live.digest
            ):
                return [{"code": "model_leased", "line": None, "param": None,
                         "message": f"Model '{live.model_name}' is leased by '{live.holder}' "
                                    f"until {live.expires_at.isoformat()} ({live.reason}); "
                                    "present that lease in X-miLLM-Lease to run under it."}]
        return None

    async def _fail_validation(self, batch_id: str, errors: list[dict], rows: Any) -> None:
        async with self.session_factory() as session:
            repo = BatchRepository(session)
            batch = await repo.get_batch(batch_id)
            if batch is None or batch.status != BatchStatus.VALIDATING.value:
                return
            batch.errors = {"object": "list", "data": errors}
            await repo.save()
        await self.finalize(batch_id, BatchStatus.FAILED)

    async def _end_cancelled_while_validating(self, batch_id: str) -> None:
        async with self.session_factory() as session:
            repo = BatchRepository(session)
            batch = await repo.get_batch(batch_id)
            if batch is not None and batch.status == BatchStatus.CANCELLING.value:
                transition(batch, BatchStatus.CANCELLED, self.now())
                await repo.save()
                self.emit(batch, force=True)

    # ------------------------------------------------------------------- the loop

    async def _loop(self) -> None:
        while True:
            self._wake.clear()
            try:
                progressed = await self.step()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the runner must outlive one bad step
                logger.error("batch_runner_step_failed", error=str(e),
                             error_type=type(e).__name__, exc_info=True)
                progressed = False
            if not progressed:
                try:
                    await asyncio.wait_for(self._wake.wait(), float(settings.BATCH_WAIT_POLL_S))
                except asyncio.TimeoutError:
                    pass

    async def step(self) -> bool:
        """Do one unit of work; False when nothing could progress (the loop then waits)."""
        async with self.session_factory() as session:
            batches = await BatchRepository(session).batches_in(
                (BatchStatus.IN_PROGRESS, BatchStatus.CANCELLING, BatchStatus.FINALIZING)
            )
        for batch in batches:
            if batch.status == BatchStatus.CANCELLING.value:
                await self.finalize(batch.id, BatchStatus.CANCELLED)
                return True
            if batch.status == BatchStatus.FINALIZING.value:
                await self.finalize(batch.id, BatchStatus.COMPLETED)
                return True
            if self.now() >= aware(batch.expires_at):
                await self.finalize(batch.id, BatchStatus.EXPIRED)
                return True
            if not await self.ensure_lease(batch):
                continue
            if await self.run_chunk(batch):
                return True
        return False

    # ------------------------------------------------------------------- leases

    async def ensure_lease(self, batch: Batch) -> bool:
        """Hold a live lease for `batch` before any row runs (FR-26.7.4, X-01, T-64).

        Own lease: renewed when a third of its TTL has passed. Caller lease: renewed the same way
        with the caller's own TTL. A lease that is gone (renew answers 404/409) is replaced by an
        OWN acquisition; when that is refused the batch waits with the reason, and runs nothing.
        """
        mem = self.memory(batch.id)
        async with self.session_factory() as session:
            service = self.model_service_factory(session)
            if mem.lease_id is not None:
                if self.monotonic() - mem.renewed_mono >= max(mem.lease_ttl, 1) / 3.0:
                    try:
                        await service.renew_lease(batch.model_id, mem.lease_id, mem.lease_ttl)
                        mem.renewed_mono = self.monotonic()
                        logger.info("batch_lease", action="renewed", batch_id=batch.id,
                                    mode=mem.lease_mode)
                    except (LeaseNotFoundError, LeaseExpiredError):
                        logger.info("batch_lease", action="lost", batch_id=batch.id,
                                    mode=mem.lease_mode)
                        mem.lease_id, mem.lease_mode = None, None
                elif service.resolve_lease(mem.lease_id) is None:
                    mem.lease_id, mem.lease_mode = None, None
            if mem.lease_id is None:
                try:
                    grant = await service.acquire_lease(
                        batch.model_id,
                        holder=f"millm-batch:{batch.id}",
                        reason=f"batch {batch.id}",
                        ttl_seconds=int(settings.BATCH_LEASE_TTL_S),
                    )
                except ModelLeasedError:
                    await self._set_waiting(batch.id, WaitingReason.LEASE_UNAVAILABLE)
                    return False
                except (ModelNotResidentError, ModelBusyError, MiLLMError):
                    await self._set_waiting(batch.id, WaitingReason.MODEL_NOT_RESIDENT)
                    return False
                mem.lease_id = grant.lease_id
                mem.lease_mode = OWN
                mem.lease_ttl = int(settings.BATCH_LEASE_TTL_S)
                mem.renewed_mono = self.monotonic()
                logger.info("batch_lease", action="acquired", batch_id=batch.id, mode=OWN)
        await self._set_lease_mode(batch.id, mem.lease_mode)
        return True

    async def _set_waiting(self, batch_id: str, reason: WaitingReason) -> None:
        async with self.session_factory() as session:
            batch = await session.get(Batch, batch_id)
            if batch is not None and batch.status == BatchStatus.IN_PROGRESS.value and (
                batch.waiting_reason != reason.value
            ):
                batch.waiting_reason = reason.value
                await session.commit()
                logger.info("batch_waiting", batch_id=batch_id, reason=reason.value)
                self.emit(batch, force=True)

    async def _set_lease_mode(self, batch_id: str, mode: Optional[str]) -> None:
        async with self.session_factory() as session:
            batch = await session.get(Batch, batch_id)
            if batch is not None and batch.lease_mode != mode:
                batch.lease_mode = mode
                await session.commit()

    async def release_own_lease(self, batch: Any) -> None:
        """Release an OWN lease; a CALLER lease is never released (T-64, mutation control M8)."""
        mem = self._memory.get(batch.id)
        if mem is None or mem.lease_id is None:
            return
        if mem.lease_mode == OWN and batch.model_id is not None:
            try:
                async with self.session_factory() as session:
                    await self.model_service_factory(session).release_lease(
                        batch.model_id, mem.lease_id
                    )
                logger.info("batch_lease", action="released", batch_id=batch.id, mode=OWN)
            except Exception as e:  # noqa: BLE001 - an already-ended lease is fine
                logger.info("batch_lease_release_skipped", batch_id=batch.id, error=str(e))
        mem.lease_id, mem.lease_mode = None, None

    # ------------------------------------------------------------------- chunks

    def chunk_size(self, batch: Batch, kind: str) -> int:
        """FTDD §9: generation and probe chunks are one row; scoring/embedding a pack or
        BATCH_CHUNK_ROWS single forwards."""
        if kind in (RowKind.GENERATION.value, RowKind.PROBE.value):
            return 1
        if batch.pack and kind == RowKind.SCORING.value:
            return max(int(settings.BATCH_PACK_MAX_ROWS), 1)
        return max(int(settings.BATCH_CHUNK_ROWS), 1)

    async def run_chunk(self, batch: Batch) -> bool:
        """Run and record one chunk. False when there was nothing to run (the batch finished,
        or it must wait)."""
        from millm.services.batch.executors import RowExecutor

        horizon = max(int(settings.BATCH_CHUNK_ROWS), int(settings.BATCH_PACK_MAX_ROWS), 1)
        async with self.session_factory() as session:
            repo = BatchRepository(session)
            rows = await repo.next_pending(batch.id, horizon)
            in_file = await repo.get_file(batch.input_file_id) if batch.input_file_id else None
        if not rows:
            await self._to_finalizing(batch.id)
            await self.finalize(batch.id, BatchStatus.COMPLETED)
            return True
        kind = rows[0].kind
        same = []
        for row in rows:
            if row.kind != kind or len(same) >= self.chunk_size(batch, kind):
                break
            same.append(row)
        store = self.store_factory()
        if in_file is None or not store.exists(in_file.storage_path):
            await self._fail(batch, RuntimeError("the input file is gone"), "input_file_missing")
            return True
        inference = self.inference_provider()
        async with self.session_factory() as session:
            model_row = await self.model_service_factory(session).repository.get_by_id(
                batch.model_id
            )

        async def read_line(row: Any) -> bytes:
            return await asyncio.to_thread(
                store.read_line, in_file.storage_path, int(row.byte_offset), int(row.byte_length)
            )

        executor = RowExecutor(
            batch_id=batch.id, endpoint=batch.endpoint, inference=inference, model_row=model_row,
            read_line=read_line, session_factory=self.session_factory,
        )
        mem = self.memory(batch.id)
        started = self.monotonic()
        try:
            async with inference._admit(background=True):
                if batch.waiting_reason is not None:
                    await self._clear_waiting(batch.id)
                results = await executor.run(same, pack=bool(batch.pack), cancelled=mem.cancel.is_set)
        except ModelBusyError:
            # The model is being unloaded: nothing of this chunk was recorded; wait, never fail.
            await self._set_waiting(batch.id, WaitingReason.MODEL_NOT_RESIDENT)
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - FTID §12: the batch fails, naming the class
            await self._fail(batch, exc, "batch_failed")
            return True
        async with self.session_factory() as session:
            repo = BatchRepository(session)
            recorded = await repo.record_chunk(batch.id, results)
            fresh = await repo.get_batch(batch.id)
        self.set_backlog(batch.id, self._backlog.get(batch.id, 0) - recorded)
        logger.info("batch_chunk_recorded", batch_id=batch.id, rows=recorded,
                    packed=any(r.packed for r in results),
                    seconds=round(self.monotonic() - started, 4))
        if fresh is not None:
            self.emit(fresh)
        return True

    async def _clear_waiting(self, batch_id: str) -> None:
        async with self.session_factory() as session:
            batch = await session.get(Batch, batch_id)
            if batch is not None and batch.waiting_reason is not None:
                batch.waiting_reason = None
                await session.commit()

    async def _to_finalizing(self, batch_id: str) -> None:
        async with self.session_factory() as session:
            batch = await session.get(Batch, batch_id)
            if batch is not None and batch.status == BatchStatus.IN_PROGRESS.value:
                transition(batch, BatchStatus.FINALIZING, self.now())
                await session.commit()
                self.emit(batch, force=True)

    async def _fail(self, batch: Batch, exc: BaseException, code: str) -> None:
        logger.error("batch_failed", batch_id=batch.id, error_type=type(exc).__name__)
        async with self.session_factory() as session:
            row = await session.get(Batch, batch.id)
            if row is not None and row.status not in {s.value for s in TERMINAL}:
                existing = (row.errors or {}).get("data", []) if row.errors else []
                row.errors = {"object": "list", "data": [*existing, {
                    "code": code, "line": None, "param": None,
                    "message": f"{type(exc).__name__}: {exc}",
                }]}
                await session.commit()
        await self.finalize(batch.id, BatchStatus.FAILED)

    # --------------------------------------------------------------- finalisation

    async def finalize(self, batch_id: str, terminal: BatchStatus) -> None:
        """Assemble output and error files from the rows, in input order, and end the batch.

        `.partial` → fsync → rename for each file, THEN one transaction: insert the file rows,
        set the terminal status, delete the batch's rows (FTDD §7). A crash before the commit
        leaves the batch where it was; reconciliation assembles again (stray bytes are swept).
        The own lease is released in `finally`, on every terminal path (mutation control M7).
        """
        batch: Any = None
        try:
            store = self.store_factory()
            async with self.session_factory() as session:
                repo = BatchRepository(session)
                batch = await repo.get_batch(batch_id)
                if batch is None or batch.status in {s.value for s in TERMINAL}:
                    return
                if batch.status == BatchStatus.CANCELLING.value and terminal is not BatchStatus.CANCELLED:
                    # A cancel landed while this terminal was being reached (a chunk failing, the
                    # last row recorded, the window passing): FR-26.3.5 allows `cancelling` only
                    # to `cancelled`, and the operator's cancel is the later, explicit intent.
                    terminal = BatchStatus.CANCELLED
                rows = await repo.rows_in_order(batch_id)
                now = self.now()
                output, errors = [], []
                unrun = _UNRUN_CODE.get(terminal)
                for row in rows:
                    if row.state == RowState.DONE.value:
                        output.append(row.result)
                    elif row.state in (RowState.INVALID.value, RowState.FAILED.value):
                        errors.append(row.result)
                    elif unrun is not None:
                        errors.append(error_line(row.custom_id, row.line_no, None, *unrun))
                expires = now + timedelta(seconds=int(batch.output_expires_after_s))
                created: dict[str, BatchFile] = {}
                for kind, lines in (("output", output), ("error", errors)):
                    if not lines:
                        continue
                    file_id = new_file_id()
                    path = store.relative_path(file_id, now)
                    stored = await asyncio.to_thread(
                        store.write_lines, path,
                        (json.dumps(line, separators=(",", ":")).encode() for line in lines),
                    )
                    created[kind] = BatchFile(
                        id=file_id, purpose=FilePurpose.BATCH_OUTPUT.value,
                        filename=f"{batch_id}_{kind}.jsonl", bytes=stored.bytes,
                        line_count=stored.line_count, storage_path=stored.storage_path,
                        sha256=stored.sha256, status=FileStatus.PROCESSED.value,
                        created_at=now, expires_at=expires,
                    )
                for row_file in created.values():
                    session.add(row_file)
                await session.flush()
                batch.output_file_id = created["output"].id if "output" in created else None
                batch.error_file_id = created["error"].id if "error" in created else None
                if terminal is BatchStatus.COMPLETED and batch.status == BatchStatus.IN_PROGRESS.value:
                    transition(batch, BatchStatus.FINALIZING, now)
                transition(batch, terminal, now)
                await repo.delete_rows(batch_id)
                await session.commit()
                logger.info("batch_terminal", batch_id=batch_id, status=terminal.value,
                            completed=batch.request_completed, failed=batch.request_failed)
                self.emit(batch, force=True)
        finally:
            if batch is not None and terminal in TERMINAL:
                await self.release_own_lease(batch)
                self.set_backlog(batch_id, 0)
                self._memory.pop(batch_id, None)


# ----------------------------------------------------------------------- process singleton

_RUNNER: dict[str, Optional[BatchRunner]] = {"runner": None}


def _default_model_service(session: Any) -> Any:
    from millm.api.dependencies import get_inference_service, get_model_downloader, get_model_loader
    from millm.db.repositories.model_repository import ModelRepository
    from millm.services.model_service import ModelService
    from millm.sockets.progress import progress_emitter

    return ModelService(
        repository=ModelRepository(session),
        downloader=get_model_downloader(),
        loader=get_model_loader(),
        emitter=progress_emitter,
        inference_service=get_inference_service(),
    )


def get_batch_runner() -> BatchRunner:
    """The process's runner, created on first use over the production session factory."""
    runner = _RUNNER["runner"]
    if runner is None:
        from millm.api.dependencies import get_inference_service
        from millm.db.base import async_session_factory
        from millm.sockets.progress import progress_emitter

        runner = BatchRunner(
            async_session_factory,
            inference_provider=get_inference_service,
            model_service_factory=_default_model_service,
            emitter=progress_emitter,
        )
        _RUNNER["runner"] = runner
    return runner


def set_batch_runner(runner: Optional[BatchRunner]) -> None:
    """Replace the process runner (tests)."""
    _RUNNER["runner"] = runner
