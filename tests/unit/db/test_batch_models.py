"""Feature 26 data layer (FTASKS 1.5): the constraints that make batch state trustworthy.

Run against SQLite with foreign keys enforced (`tests/conftest.py`), so the CHECK constraints, the
composite primary key and the cascade are exercised for real, not assumed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from millm.core.batch_values import BatchStatus, RowState
from millm.db.models.batch import Batch, BatchFile, BatchRow
from millm.db.repositories.batch_repository import BatchRepository, RowResult

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


def _file(file_id: str = "file-a") -> BatchFile:
    return BatchFile(
        id=file_id, purpose="batch", filename="in.jsonl", bytes=10, line_count=2,
        storage_path=f"2026-10/{file_id}.jsonl", sha256="0" * 64, status="processed",
        created_at=NOW, expires_at=NOW + timedelta(days=30),
    )


def _batch(batch_id: str = "batch_a", status: str = "in_progress") -> Batch:
    return Batch(
        id=batch_id, endpoint="/v1/completions", completion_window="24h", status=status,
        input_file_id="file-a", pack=True, output_expires_after_s=2_592_000,
        created_at=NOW, expires_at=NOW + timedelta(hours=24), request_total=2,
    )


def _row(line_no: int, batch_id: str = "batch_a", state: str = "pending") -> BatchRow:
    return BatchRow(
        batch_id=batch_id, line_no=line_no, custom_id=f"c{line_no}", kind="scoring",
        byte_offset=0, byte_length=5, state=state,
    )


async def _seed(session, rows=(1, 2)):
    session.add(_file())
    await session.commit()
    session.add(_batch())
    await session.commit()
    session.add_all([_row(n) for n in rows])
    await session.commit()


class TestStatusConstraint:
    async def test_every_openai_status_is_accepted(self, test_session):
        test_session.add(_file())
        await test_session.commit()
        for i, status in enumerate(BatchStatus):
            test_session.add(_batch(f"batch_{i}", status.value))
        await test_session.commit()

    async def test_a_ninth_status_is_rejected_by_the_database(self, test_session):
        test_session.add(_file())
        await test_session.commit()
        test_session.add(_batch("batch_x", "paused"))
        with pytest.raises(IntegrityError):
            await test_session.commit()

    def test_the_status_set_is_exactly_openais_eight(self):
        assert [s.value for s in BatchStatus] == [
            "validating", "in_progress", "finalizing", "completed", "failed", "cancelling",
            "cancelled", "expired",
        ]


class TestRowIdentity:
    async def test_batch_id_and_line_no_are_unique(self, test_session):
        await _seed(test_session, rows=(1,))
        test_session.add(_row(1))
        with pytest.raises(IntegrityError):
            await test_session.commit()

    async def test_rows_cascade_with_their_batch(self, test_session):
        await _seed(test_session)
        await test_session.execute(text("DELETE FROM batches WHERE id = 'batch_a'"))
        await test_session.commit()
        left = (await test_session.execute(text("SELECT count(*) FROM batch_rows"))).scalar_one()
        assert left == 0

    async def test_deleting_a_file_nulls_the_batch_reference(self, test_session):
        await _seed(test_session)
        await test_session.execute(text("DELETE FROM batch_files WHERE id = 'file-a'"))
        await test_session.commit()
        ref = (
            await test_session.execute(text("SELECT input_file_id FROM batches"))
        ).scalar_one()
        assert ref is None


class TestRecordChunk:
    async def test_records_rows_and_counters_together(self, test_session):
        await _seed(test_session)
        repo = BatchRepository(test_session)
        recorded = await repo.record_chunk("batch_a", [
            RowResult(1, RowState.DONE, {"custom_id": "c1"}, False),
            RowResult(2, RowState.FAILED, {"custom_id": "c2"}, True),
        ])
        assert recorded == 2
        batch = await repo.get_batch("batch_a")
        await test_session.refresh(batch)
        assert (batch.request_completed, batch.request_failed, batch.chunks_recorded) == (1, 1, 1)

    async def test_a_recorded_row_is_never_overwritten_or_recounted(self, test_session):
        """M5's target: the `state='pending'` guard. A re-run chunk must not record twice."""
        await _seed(test_session)
        repo = BatchRepository(test_session)
        await repo.record_chunk("batch_a", [RowResult(1, RowState.DONE, {"v": "first"}, False)])
        again = await repo.record_chunk(
            "batch_a", [RowResult(1, RowState.DONE, {"v": "second"}, False)]
        )
        assert again == 0
        result = (
            await test_session.execute(
                text("SELECT result FROM batch_rows WHERE line_no = 1")
            )
        ).scalar_one()
        assert "first" in str(result) and "second" not in str(result)
        batch = await repo.get_batch("batch_a")
        await test_session.refresh(batch)
        assert batch.request_completed == 1

    async def test_a_chunk_is_recorded_whole_or_not_at_all(self, test_session):
        """M4's target: one transaction. The second row's result cannot be serialised, so the
        chunk fails — and the first row must still be pending, uncounted."""
        await _seed(test_session)
        repo = BatchRepository(test_session)
        with pytest.raises(Exception):
            await repo.record_chunk("batch_a", [
                RowResult(1, RowState.DONE, {"ok": 1}, False),
                RowResult(2, RowState.DONE, {"bad": object()}, False),
            ])
        states = (
            await test_session.execute(text("SELECT state FROM batch_rows ORDER BY line_no"))
        ).scalars().all()
        assert list(states) == ["pending", "pending"]
        counts = (
            await test_session.execute(text("SELECT request_completed FROM batches"))
        ).scalar_one()
        assert counts == 0


class TestProbeEventOrigin:
    async def test_an_existing_event_reads_live(self, test_session):
        """The server default must be `live`: every row that predates the column was live traffic."""
        from tests.unit.services.test_probe_survives_no_restart import _probe_row

        test_session.add(_probe_row("pr_a", armed=False))
        await test_session.commit()
        await test_session.execute(
            text("INSERT INTO probe_events (probe_id, scored, window, provisional, "
                 "threshold_revision) VALUES ('pr_a', 1, 'all', 0, 1)")
        )
        await test_session.commit()
        origin = (await test_session.execute(text("SELECT origin FROM probe_events"))).scalar_one()
        assert origin == "live"

    async def test_one_batch_line_records_one_event_per_window(self, test_session):
        from tests.unit.services.test_probe_survives_no_restart import _probe_row

        test_session.add(_probe_row("pr_a", armed=False))
        await test_session.commit()
        insert = text(
            "INSERT INTO probe_events (probe_id, scored, window, provisional, threshold_revision,"
            " origin, batch_id, batch_line) VALUES ('pr_a', 1, :w, 0, 1, 'batch', 'b1', 7)"
        )
        await test_session.execute(insert, {"w": "all"})
        await test_session.execute(insert, {"w": "prompt"})
        await test_session.commit()
        with pytest.raises(IntegrityError):
            await test_session.execute(insert, {"w": "all"})
            await test_session.commit()


def test_the_orm_metadata_carries_the_new_tables():
    """The schema guards compare `target_metadata()` with the migrated database; a table missing
    from it would be invisible to them."""
    from millm.db.alembic_support import target_metadata

    tables = target_metadata().tables
    assert {"batch_files", "batches", "batch_rows"} <= set(tables)
    assert {"origin", "batch_id", "batch_line"} <= set(tables["probe_events"].columns.keys())


def test_the_migration_creates_what_the_orm_declares():
    """Columns per table, migration vs ORM, read from the migration's own operations."""
    import importlib.util
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from millm.db.alembic_support import target_metadata

    path = Path(__file__).resolve().parents[3] / "millm/db/migrations/versions/019_add_batch_api.py"
    spec = importlib.util.spec_from_file_location("m019", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    created: dict[str, set[str]] = {}
    added: dict[str, set[str]] = {}
    op = MagicMock()
    op.create_table.side_effect = lambda name, *cols, **kw: created.setdefault(
        name, {c.name for c in cols if hasattr(c, "name") and getattr(c, "name", None)
               and c.__class__.__name__ == "Column"}
    )
    op.add_column.side_effect = lambda table, col: added.setdefault(table, set()).add(col.name)
    with patch.object(module, "op", op):
        module.upgrade()
    tables = target_metadata().tables
    for name, cols in created.items():
        assert cols == set(tables[name].columns.keys()), name
    assert added["probe_events"] == {"origin", "batch_id", "batch_line"}
