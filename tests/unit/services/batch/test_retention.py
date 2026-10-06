"""Feature 26 task 3.5 / 3.7: retention with an injected clock, and the orphan sweep.

M9's target is `test_a_file_a_running_batch_needs_survives_its_expiry`.
"""

from __future__ import annotations

import io
import os
import time
from datetime import datetime, timedelta, timezone

from millm.db.models.batch import Batch, BatchFile
from millm.services.batch.files import FileStore
from millm.services.batch.retention import prune_expired_files
from tests.unit.batch_fixtures import batch_db, batch_dir  # noqa: F401

T0 = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


async def _add_file(factory, store: FileStore, file_id: str, expires: datetime) -> BatchFile:
    path = store.relative_path(file_id, T0)
    stored = store.write_upload(io.BytesIO(b'{"a":1}\n'), path, max_bytes=1000, max_rows=10)
    row = BatchFile(
        id=file_id, purpose="batch", filename="x.jsonl", bytes=stored.bytes,
        line_count=stored.line_count, storage_path=path, sha256=stored.sha256,
        status="processed", created_at=T0, expires_at=expires,
    )
    async with factory() as session:
        session.add(row)
        await session.commit()
    return row


async def _status(factory, file_id: str) -> str:
    async with factory() as session:
        return (await session.get(BatchFile, file_id)).status


async def test_prune_removes_only_expired_unreferenced_files(batch_db, batch_dir):
    store = FileStore(batch_dir)
    old = await _add_file(batch_db, store, "file-old", T0 + timedelta(days=30))
    young = await _add_file(batch_db, store, "file-young", T0 + timedelta(days=31))
    result = await prune_expired_files(batch_db, store, T0 + timedelta(days=30, seconds=1))
    assert result["pruned"] == ["file-old"]
    assert await _status(batch_db, "file-old") == "expired"
    assert await _status(batch_db, "file-young") == "processed"
    assert not store.exists(old.storage_path) and store.exists(young.storage_path)


async def test_nothing_is_pruned_before_expiry(batch_db, batch_dir):
    store = FileStore(batch_dir)
    await _add_file(batch_db, store, "file-a", T0 + timedelta(days=30))
    result = await prune_expired_files(batch_db, store, T0 + timedelta(days=29))
    assert result["pruned"] == []


async def test_a_file_a_running_batch_needs_survives_its_expiry(batch_db, batch_dir):
    """FR-26.9.4: a batch still running reads its input by offset; deleting it fails the batch."""
    store = FileStore(batch_dir)
    row = await _add_file(batch_db, store, "file-in", T0 + timedelta(days=1))
    async with batch_db() as session:
        session.add(Batch(
            id="batch_run", endpoint="/v1/completions", completion_window="168h",
            status="in_progress", input_file_id="file-in", pack=True,
            output_expires_after_s=2_592_000, created_at=T0, expires_at=T0 + timedelta(days=7),
        ))
        await session.commit()
    result = await prune_expired_files(batch_db, store, T0 + timedelta(days=2))
    assert result["pruned"] == [] and result["kept"] == {"file-in": "batch_run"}
    assert store.exists(row.storage_path)
    assert await _status(batch_db, "file-in") == "processed"


async def test_the_sweep_removes_orphans_and_stray_partials_only(batch_db, batch_dir):
    store = FileStore(batch_dir)
    kept = await _add_file(batch_db, store, "file-kept", T0 + timedelta(days=30))
    orphan = batch_dir / "2026-10" / "file-orphan.jsonl"
    partial = batch_dir / "2026-10" / "file-x.jsonl.partial"
    orphan.write_bytes(b"x\n")
    partial.write_bytes(b"x")
    result = await prune_expired_files(batch_db, store, T0, sweep_min_age_s=0)
    assert sorted(result["swept"]) == ["2026-10/file-orphan.jsonl", "2026-10/file-x.jsonl.partial"]
    assert store.exists(kept.storage_path)


def test_the_hourly_sweep_spares_a_file_written_moments_ago(tmp_path):
    """An upload renames its bytes into place BEFORE its row commits."""
    store = FileStore(tmp_path)
    (tmp_path / "2026-10").mkdir()
    fresh = tmp_path / "2026-10" / "file-new.jsonl"
    fresh.write_bytes(b"x\n")
    assert store.sweep(set(), min_age_s=600) == []
    old = time.time() - 3600
    os.utime(fresh, (old, old))
    assert store.sweep(set(), min_age_s=600) == ["2026-10/file-new.jsonl"]


def test_a_path_outside_the_root_is_refused(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        FileStore(tmp_path).absolute("../escape.jsonl")


def test_read_line_reads_one_line_by_offset(tmp_path):
    store = FileStore(tmp_path)
    store.write_upload(io.BytesIO(b"first\nsecond\nthird"), "a/f.jsonl", max_bytes=100, max_rows=5)
    lines = list(store.iter_lines("a/f.jsonl"))
    assert [line for _, line in lines] == [b"first", b"second", b"third"]
    offset, line = lines[1]
    assert store.read_line("a/f.jsonl", offset, len(line)) == b"second"
