"""Startup reconciliation for batches (Feature 26, FR-26.3.3, FTASKS 5.9).

A restart loses the runner's memory — lease ids, cancel flags, the backlog — and may interrupt a
batch in any non-terminal state. This named function puts every batch back on a path that ends:

* `validating`  → validated again (a crashed pass's rows are replaced, never appended to);
* `in_progress` → resumes from the first row without a recorded result. Its lease is gone (029
  FR-29.1.9 — `clear_leases_on_startup` ran first), so `lease_mode` is cleared and the batch
  reports `queued` until the runner re-acquires one or waits (X-01, T-66). The input file's
  sha256 is checked: a different file would break "the first row without a recorded result";
* `finalizing`  → its files are assembled again (a crash before the commit left no file rows);
* `cancelling`  → ends `cancelled`, with every recorded row in its files;
* every stray `.partial` (an assembly or upload the restart interrupted) is removed.

Never raises: batch bookkeeping must never be why the server does not start. It is a named
function, exercised for real by a test, with a second test that fails when `lifespan` stops
calling it (M11) — this estate has shipped "a new in-memory state left out of startup
reconciliation" three times.
"""

from __future__ import annotations

import asyncio
from typing import Any

from millm.core.batch_values import BatchStatus, WaitingReason
from millm.core.logging import get_logger
from millm.db.repositories.batch_repository import BatchRepository
from millm.services.batch.files import FileStore

logger = get_logger(__name__)


async def reconcile_batches_on_startup(
    session_factory: Any, runner: Any, store: FileStore
) -> dict[str, list[str]]:
    """Reconcile every non-terminal batch. Returns `{status: [batch ids]}` for what it found."""
    found: dict[str, list[str]] = {s.value: [] for s in (
        BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS, BatchStatus.FINALIZING,
        BatchStatus.CANCELLING,
    )}
    try:
        async with session_factory() as session:
            repo = BatchRepository(session)
            batches = await repo.batches_in([BatchStatus(s) for s in found])
            for batch in batches:
                found[batch.status].append(batch.id)
                if batch.status == BatchStatus.IN_PROGRESS.value:
                    batch.lease_mode = None
                    batch.waiting_reason = WaitingReason.QUEUED.value
                    remaining = int(batch.request_total or 0) - int(batch.request_completed or 0) \
                        - int(batch.request_failed or 0)
                    runner.set_backlog(batch.id, remaining)
            await repo.save()
            keep = await repo.live_storage_paths()
            changed: list[str] = []
            for batch_id in found[BatchStatus.IN_PROGRESS.value]:
                batch = await repo.get_batch(batch_id)
                in_file = await repo.get_file(batch.input_file_id) if batch.input_file_id else None
                problem = await runner._input_problem(in_file, store)
                if problem is not None:
                    changed.append(batch_id)
                    batch.errors = {"object": "list", "data": [problem]}
            await repo.save()
        removed = await asyncio.to_thread(store.sweep, keep, min_age_s=0)
        for batch_id in changed:
            await runner._fail(await _row(session_factory, batch_id), RuntimeError(
                "the input file changed or is gone"), "input_file_changed")
        for batch_id in found[BatchStatus.VALIDATING.value]:
            runner.submit_validation(batch_id)
        for batch_id in found[BatchStatus.FINALIZING.value]:
            await runner.finalize(batch_id, BatchStatus.COMPLETED)
        for batch_id in found[BatchStatus.CANCELLING.value]:
            await runner.finalize(batch_id, BatchStatus.CANCELLED)
        logger.info(
            "batches_reconciled_on_startup",
            **{k: len(v) for k, v in found.items()},
            partials_removed=len(removed),
        )
    except Exception as e:  # noqa: BLE001 - never stop the server starting
        logger.error("batch_startup_reconcile_failed", error=str(e),
                     error_type=type(e).__name__, exc_info=True)
    return found


async def _row(session_factory: Any, batch_id: str) -> Any:
    async with session_factory() as session:
        return await BatchRepository(session).get_batch(batch_id)


async def start_batch_api(session_factory: Any, runner: Any = None) -> Any:
    """Lifespan's ONE call (FTASKS 5.10): reconcile → start the runner → retention (startup pass
    plus the hourly loop). Never raises. Returns the runner."""
    from millm.core.config import settings
    from millm.services.batch.files import get_file_store
    from millm.services.batch.retention import prune_expired_files, retention_loop
    from millm.services.batch.runner import get_batch_runner

    runner = runner or get_batch_runner()
    try:
        await reconcile_batches_on_startup(session_factory, runner, get_file_store())
        runner.start()
        await prune_expired_files(session_factory, get_file_store(), sweep_min_age_s=0)
        runner._retention = asyncio.get_running_loop().create_task(retention_loop(
            session_factory, get_file_store, float(settings.BATCH_RETENTION_INTERVAL_S)
        ))
    except Exception as e:  # noqa: BLE001
        logger.error("batch_api_start_failed", error=str(e), error_type=type(e).__name__,
                     exc_info=True)
    return runner


async def stop_batch_api(runner: Any = None) -> None:
    """Shutdown: stop the runner and the retention loop."""
    from millm.services.batch.runner import get_batch_runner

    runner = runner or get_batch_runner()
    retention = getattr(runner, "_retention", None)
    if retention is not None:
        retention.cancel()
    await runner.stop()
