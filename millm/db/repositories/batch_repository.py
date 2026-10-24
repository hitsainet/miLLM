"""Persistence for the Batch API (Feature 26, FTID §4).

⚠ Every criteria UPDATE/DELETE passes ``synchronize_session=False``. A criteria statement otherwise
synchronises the session by EVALUATING its WHERE clause in Python against loaded objects; with an
aware datetime against SQLite's naive column that raised, and a caller's ``except`` turned it into
a silent no-op (the reason is recorded at ``probe_repository.prune_aged``).

⚠ ``record_chunk`` is THE place a row becomes recorded, and it is guarded by ``state='pending'`` in
the WHERE clause. A row already recorded can therefore never be overwritten, and a chunk re-run
after a crash records only the rows that were never recorded — which is what "no recorded row runs
twice" means at the database (FR-26.3.4). The whole chunk is ONE transaction (FR-26.3.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from millm.core.batch_values import TERMINAL, BatchStatus, FileStatus, RowState
from millm.db.models.batch import Batch, BatchFile, BatchRow

#: Rows per INSERT statement at validation (FTID §4).
INSERT_CHUNK = 1000

_NOSYNC = {"synchronize_session": False}
_ACTIVE = tuple(s.value for s in BatchStatus if s not in TERMINAL)


@dataclass(frozen=True)
class RowResult:
    """One row's outcome, as the runner hands it to `record_chunk`."""

    line_no: int
    state: RowState
    result: dict[str, Any]
    packed: bool


class BatchRepository:
    """Async reads and writes for batch files, batches and batch rows."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------------ files

    async def create_file(self, row: BatchFile) -> BatchFile:
        self.session.add(row)
        await self.session.commit()
        return row

    async def get_file(self, file_id: str) -> Optional[BatchFile]:
        return await self.session.get(BatchFile, file_id)

    async def list_files(
        self, *, purpose: Optional[str], limit: int, after: Optional[str], order: str
    ) -> tuple[list[BatchFile], bool]:
        """Newest first by default (FR-26.6.9); `after` is the id of the last file already seen."""
        descending = order != "asc"
        stmt = select(BatchFile)
        if purpose is not None:
            stmt = stmt.where(BatchFile.purpose == purpose)
        if after is not None:
            anchor = await self.session.get(BatchFile, after)
            if anchor is not None:
                key = (anchor.created_at, anchor.id)
                if descending:
                    stmt = stmt.where(
                        (BatchFile.created_at < key[0])
                        | ((BatchFile.created_at == key[0]) & (BatchFile.id < key[1]))
                    )
                else:
                    stmt = stmt.where(
                        (BatchFile.created_at > key[0])
                        | ((BatchFile.created_at == key[0]) & (BatchFile.id > key[1]))
                    )
        if descending:
            stmt = stmt.order_by(BatchFile.created_at.desc(), BatchFile.id.desc())
        else:
            stmt = stmt.order_by(BatchFile.created_at.asc(), BatchFile.id.asc())
        rows = list((await self.session.execute(stmt.limit(limit + 1))).scalars().all())
        return rows[:limit], len(rows) > limit

    async def batch_referencing(self, file_id: str) -> Optional[str]:
        """The id of a NON-TERMINAL batch that references `file_id`, or None (FR-26.6.10, 26.9.4)."""
        stmt = (
            select(Batch.id)
            .where(
                Batch.status.in_(_ACTIVE),
                (Batch.input_file_id == file_id)
                | (Batch.output_file_id == file_id)
                | (Batch.error_file_id == file_id),
            )
            .order_by(Batch.created_at)
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def mark_file(self, file_id: str, status: FileStatus, when: datetime) -> None:
        """`deleted` or `expired`: the bytes go after this commits (FTDD §7 "Side effects")."""
        await self.session.execute(
            update(BatchFile)
            .where(BatchFile.id == file_id)
            .values(status=status.value, deleted_at=when),
            execution_options=_NOSYNC,
        )
        await self.session.commit()

    async def expired_files(self, now: datetime) -> list[BatchFile]:
        """Files past `expires_at` whose bytes are still held."""
        stmt = select(BatchFile).where(
            BatchFile.expires_at <= now,
            BatchFile.status.in_((FileStatus.PROCESSED.value, FileStatus.ERROR.value)),
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def live_storage_paths(self) -> set[str]:
        """Storage paths of every file whose bytes should still exist (the orphan sweep's keep set)."""
        stmt = select(BatchFile.storage_path).where(
            BatchFile.status.in_((FileStatus.PROCESSED.value, FileStatus.ERROR.value))
        )
        return set((await self.session.execute(stmt)).scalars().all())

    # ---------------------------------------------------------------- batches

    async def create_batch(self, row: Batch) -> Batch:
        self.session.add(row)
        await self.session.commit()
        return row

    async def get_batch(self, batch_id: str) -> Optional[Batch]:
        return await self.session.get(Batch, batch_id)

    async def list_batches(self, *, limit: int, after: Optional[str]) -> tuple[list[Batch], bool]:
        """Newest first (FR-26.6.3)."""
        stmt = select(Batch)
        if after is not None:
            anchor = await self.session.get(Batch, after)
            if anchor is not None:
                stmt = stmt.where(
                    (Batch.created_at < anchor.created_at)
                    | ((Batch.created_at == anchor.created_at) & (Batch.id < anchor.id))
                )
        stmt = stmt.order_by(Batch.created_at.desc(), Batch.id.desc()).limit(limit + 1)
        rows = list((await self.session.execute(stmt)).scalars().all())
        return rows[:limit], len(rows) > limit

    async def batches_in(self, statuses: Iterable[BatchStatus]) -> list[Batch]:
        """Batches in any of `statuses`, oldest first (FIFO, FTID §3)."""
        stmt = (
            select(Batch)
            .where(Batch.status.in_([s.value for s in statuses]))
            .order_by(Batch.created_at.asc(), Batch.id.asc())
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def save(self) -> None:
        await self.session.commit()

    # ------------------------------------------------------------------- rows

    async def bulk_insert_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        """Insert validated rows, INSERT_CHUNK per statement. The caller commits."""
        for start in range(0, len(rows), INSERT_CHUNK):
            await self.session.execute(insert(BatchRow), list(rows[start:start + INSERT_CHUNK]))

    async def delete_rows(self, batch_id: str) -> None:
        """Drop a batch's rows (a re-validation, or finalisation). The caller commits."""
        await self.session.execute(
            delete(BatchRow).where(BatchRow.batch_id == batch_id), execution_options=_NOSYNC
        )

    async def next_pending(self, batch_id: str, n: int) -> list[BatchRow]:
        """The first `n` rows without a recorded result, in input order (FR-26.3.3)."""
        stmt = (
            select(BatchRow)
            .where(BatchRow.batch_id == batch_id, BatchRow.state == RowState.PENDING.value)
            .order_by(BatchRow.line_no)
            .limit(n)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def pending_count(self, batch_id: str) -> int:
        stmt = select(func.count()).select_from(BatchRow).where(
            BatchRow.batch_id == batch_id, BatchRow.state == RowState.PENDING.value
        )
        return int((await self.session.execute(stmt)).scalar_one())

    async def rows_in_order(self, batch_id: str) -> list[BatchRow]:
        """Every row of a batch in input-line order, for assembly (FR-26.6.8)."""
        stmt = select(BatchRow).where(BatchRow.batch_id == batch_id).order_by(BatchRow.line_no)
        return list((await self.session.execute(stmt)).scalars().all())

    async def record_chunk(self, batch_id: str, results: Sequence[RowResult]) -> int:
        """Record a chunk's results — all of them, or none — and bump the counters. Returns how
        many rows were recorded.

        ⚠ ONE TRANSACTION (FR-26.3.2): a crash before the commit leaves every row of the chunk
        `pending`, so it runs again and none of it was ever counted. ⚠ The `state='pending'` guard
        (FR-26.3.4): a row recorded by an earlier attempt is not overwritten, and is not counted a
        second time.
        """
        completed = failed = 0
        try:
            batch = await self.session.get(Batch, batch_id)
            chunk_seq = int(batch.chunks_recorded or 0) if batch is not None else 0
            for item in results:
                outcome = await self.session.execute(
                    update(BatchRow)
                    .where(
                        BatchRow.batch_id == batch_id,
                        BatchRow.line_no == item.line_no,
                        # ⚠ The exactly-once guard. Remove it and a re-run overwrites a recorded
                        # row and counts it twice.
                        BatchRow.state == RowState.PENDING.value,
                    )
                    .values(
                        state=item.state.value,
                        result=item.result,
                        packed=item.packed,
                        chunk_seq=chunk_seq,
                    ),
                    execution_options=_NOSYNC,
                )
                if outcome.rowcount == 1:
                    if item.state is RowState.DONE:
                        completed += 1
                    else:
                        failed += 1
            await self.session.execute(
                update(Batch)
                .where(Batch.id == batch_id)
                .values(
                    request_completed=Batch.request_completed + completed,
                    request_failed=Batch.request_failed + failed,
                    chunks_recorded=Batch.chunks_recorded + 1,
                ),
                execution_options=_NOSYNC,
            )
            await self.session.commit()
        except BaseException:
            await self.session.rollback()
            raise
        return completed + failed
