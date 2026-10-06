"""Batch file retention (Feature 26, FR-26.9, T-68).

Every batch file carries `expires_at` (input files 30 days after upload; output/error files after
the batch's `output_expires_after`). `prune_expired_files` deletes the bytes of expired files and
marks their rows `expired` — the RECORD remains, and the content route answers `404 file_expired`
(FR-26.9.3). It runs at startup and hourly (FR-26.9.4).

⚠ A file referenced by a NON-TERMINAL batch is never pruned (FR-26.9.4, mutation control M9): a
batch still validating or running reads its input file by offset, row by row, and resumes from it
after a restart. Its expiry is honoured once the batch ends.

Order per file: mark the row, commit, THEN unlink. A failed unlink leaves bytes no row owns, which
the orphan sweep removes next time; the reverse order would leave a row pointing at nothing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Optional

from millm.core.batch_values import FileStatus
from millm.core.logging import get_logger
from millm.db.repositories.batch_repository import BatchRepository
from millm.services.batch.files import FileStore
from millm.services.batch.state import utcnow

logger = get_logger(__name__)

#: The hourly sweep spares files touched in the last ten minutes (an upload in flight).
SWEEP_GRACE_S = 600.0


async def prune_expired_files(
    session_factory: Any,
    store: FileStore,
    now: Optional[datetime] = None,
    *,
    sweep_min_age_s: float = SWEEP_GRACE_S,
) -> dict[str, Any]:
    """Prune expired, unreferenced files, then sweep orphaned bytes. Returns what it did."""
    now = now or utcnow()
    pruned: list[str] = []
    kept: dict[str, str] = {}
    async with session_factory() as session:
        repo = BatchRepository(session)
        for row in await repo.expired_files(now):
            # ⚠ The reference guard (M9). Remove it and a running batch's input file is deleted
            # under it.
            referencing = await repo.batch_referencing(row.id)
            if referencing is not None:
                kept[row.id] = referencing
                continue
            await repo.mark_file(row.id, FileStatus.EXPIRED, now)
            await asyncio.to_thread(store.delete, row.storage_path)
            pruned.append(row.id)
        keep = await repo.live_storage_paths()
    swept = await asyncio.to_thread(store.sweep, keep, min_age_s=sweep_min_age_s)
    if pruned or swept:
        logger.info(
            "batch_files_pruned", pruned=len(pruned), swept=len(swept), kept_for_batches=len(kept)
        )
    return {"pruned": pruned, "kept": kept, "swept": swept}


async def retention_loop(
    session_factory: Any,
    store_factory: Callable[[], FileStore],
    interval_s: float,
    now: Callable[[], datetime] = utcnow,
) -> None:
    """Prune every `interval_s` until cancelled. A failed pass is logged and retried next time."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            await prune_expired_files(session_factory, store_factory(), now())
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - retention must never take the runner down
            logger.warning("batch_retention_failed", error=str(e), error_type=type(e).__name__)
