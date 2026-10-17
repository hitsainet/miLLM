"""Batch file BYTES on the data volume (Feature 26, FTDD §4, §8).

Rules this module holds:

* **Generated paths only.** A stored path is `<yyyy-mm>/<file-id>.jsonl` under `BATCH_FILES_DIR`.
  The client's filename is metadata on the row and never touches the filesystem (FTDD §8).
* **Caps while copying (FR-26.8.2, 26.8.3).** An upload is copied in 1 MiB pieces, counting bytes,
  lines and the sha256 as it goes; over a cap, the partial file is deleted and the refusal names
  the limit and the MEASURED value. The source is Starlette's spooled upload (a temporary file past
  1 MiB), so nothing is ever held whole in memory.
* **Write `.partial`, fsync, rename.** A crash leaves a `.partial` the orphan sweep removes, never a
  half-written file under a final name.
* **Line reads by offset.** A row's line is read with `os.pread(offset, length)` — never the whole
  input file per row.

Everything here is synchronous; callers run it with `asyncio.to_thread`.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Optional

from millm.core.errors import BatchFileLimitError

COPY_CHUNK = 1 << 20
PARTIAL = ".partial"


def new_file_id() -> str:
    return f"file-{secrets.token_hex(12)}"


def new_batch_id() -> str:
    return f"batch_{secrets.token_hex(12)}"


@dataclass(frozen=True)
class StoredFile:
    """What a write measured."""

    storage_path: str
    bytes: int
    line_count: int
    sha256: str


def count_lines(total_bytes: int, newlines: int, last_byte: Optional[int]) -> int:
    """Lines in a file: every `\\n`, plus a final line that has none. An empty file has 0."""
    if total_bytes == 0:
        return 0
    return newlines + (0 if last_byte == 0x0A else 1)


class FileStore:
    """Bytes for batch files, under one root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ----------------------------------------------------------------- paths

    def relative_path(self, file_id: str, now: datetime) -> str:
        """`yyyy-mm/<id>.jsonl` — generated from the id and the clock, nothing the client sent."""
        return f"{now:%Y-%m}/{file_id}.jsonl"

    def absolute(self, storage_path: str) -> Path:
        path = (self.root / storage_path).resolve()
        if self.root.resolve() not in path.parents:
            # Defence in depth: rows only ever hold generated paths, but a path that escapes the
            # root is refused rather than read or deleted.
            raise ValueError(f"storage path escapes BATCH_FILES_DIR: {storage_path!r}")
        return path

    # ----------------------------------------------------------------- write

    def write_upload(
        self, source: BinaryIO, storage_path: str, *, max_bytes: int, max_rows: int
    ) -> StoredFile:
        """Copy `source` to `storage_path` within the caps, or store NOTHING and raise.

        Over a cap the copy stops writing at once and deletes the partial file, then keeps
        COUNTING (reading the spooled upload, writing nothing) so the refusal can name the measured
        value rather than "more than the limit".
        """
        final = self.absolute(storage_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(final.name + PARTIAL)
        digest = hashlib.sha256()
        total = newlines = 0
        last: Optional[int] = None
        over: Optional[str] = None
        out = open(partial, "wb")
        try:
            while True:
                chunk = source.read(COPY_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                newlines += chunk.count(b"\n")
                last = chunk[-1]
                if over is None:
                    rows_so_far = newlines  # complete lines; the final partial line comes later
                    if total > max_bytes:
                        over = "bytes"
                    elif rows_so_far > max_rows:
                        over = "rows"
                    if over is not None:
                        out.close()
                        partial.unlink(missing_ok=True)
                        continue
                    digest.update(chunk)
                    out.write(chunk)
            rows = count_lines(total, newlines, last)
            if over is None and rows > max_rows:
                over = "rows"
            if over is not None:
                if not out.closed:
                    out.close()
                partial.unlink(missing_ok=True)
                if over == "bytes":
                    raise BatchFileLimitError(
                        f"The file is {total} bytes; the limit is {max_bytes} bytes "
                        "(BATCH_MAX_FILE_BYTES). Nothing was stored; split the file.",
                        details={"param": "file", "limit": "bytes", "max": max_bytes,
                                 "measured": total},
                    )
                raise BatchFileLimitError(
                    f"The file has {rows} lines; the limit is {max_rows} requests "
                    "(BATCH_MAX_ROWS). Nothing was stored; split the file.",
                    details={"param": "file", "limit": "rows", "max": max_rows,
                             "measured": rows},
                )
            out.flush()
            os.fsync(out.fileno())
            out.close()
            os.replace(partial, final)
        except BaseException:
            if not out.closed:
                out.close()
            partial.unlink(missing_ok=True)
            raise
        return StoredFile(storage_path, total, rows, digest.hexdigest())

    def write_lines(self, storage_path: str, lines: Iterable[bytes]) -> StoredFile:
        """Assemble an output/error file: `.partial` → fsync → rename (FTDD §7)."""
        final = self.absolute(storage_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(final.name + PARTIAL)
        digest = hashlib.sha256()
        total = count = 0
        try:
            with open(partial, "wb") as out:
                for line in lines:
                    data = line if line.endswith(b"\n") else line + b"\n"
                    out.write(data)
                    digest.update(data)
                    total += len(data)
                    count += 1
                out.flush()
                os.fsync(out.fileno())
            os.replace(partial, final)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return StoredFile(storage_path, total, count, digest.hexdigest())

    # ------------------------------------------------------------------ read

    def exists(self, storage_path: str) -> bool:
        return self.absolute(storage_path).is_file()

    def read_line(self, storage_path: str, offset: int, length: int) -> bytes:
        """One line's bytes by offset (`os.pread`), without reading the file."""
        fd = os.open(self.absolute(storage_path), os.O_RDONLY)
        try:
            return os.pread(fd, length, offset)
        finally:
            os.close(fd)

    def iter_lines(self, storage_path: str) -> Iterator[tuple[int, bytes]]:
        """`(byte_offset, line_bytes_without_newline)` for every line, streaming."""
        offset = 0
        with open(self.absolute(storage_path), "rb") as handle:
            for raw in handle:
                yield offset, raw[:-1] if raw.endswith(b"\n") else raw
                offset += len(raw)

    def iter_bytes(self, storage_path: str) -> Iterator[bytes]:
        with open(self.absolute(storage_path), "rb") as handle:
            while True:
                chunk = handle.read(COPY_CHUNK)
                if not chunk:
                    return
                yield chunk

    def sha256(self, storage_path: str) -> str:
        digest = hashlib.sha256()
        for chunk in self.iter_bytes(storage_path):
            digest.update(chunk)
        return digest.hexdigest()

    # ---------------------------------------------------------------- delete

    def delete(self, storage_path: str) -> bool:
        """Remove the bytes; True if a file was removed. A missing file is not an error."""
        path = self.absolute(storage_path)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def sweep(self, keep: set[str], *, min_age_s: float = 0.0, now: Optional[float] = None) -> list[str]:
        """Delete every stored file no live row owns, and every stray `.partial` (FTDD §7).

        `keep` is the set of storage paths whose bytes should exist. ⚠ `min_age_s` spares files
        modified more recently than that: an upload renames its file into place BEFORE its row
        commits, and an assembly writes its `.partial` while the runner is live, so the hourly
        sweep must not take a file in flight. The startup sweep passes 0 — nothing is in flight
        in a process that has not started serving. Returns what was removed.
        """
        import time

        removed: list[str] = []
        if not self.root.is_dir():
            return removed
        clock = time.time() if now is None else now
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            try:
                young = clock - path.stat().st_mtime < min_age_s
            except FileNotFoundError:
                continue
            if young:
                continue
            if rel.endswith(PARTIAL) or (rel.endswith(".jsonl") and rel not in keep):
                try:
                    path.unlink()
                    removed.append(rel)
                except FileNotFoundError:
                    pass
        return removed


def get_file_store() -> FileStore:
    """The store under the configured BATCH_FILES_DIR (read per call, so a test can point it)."""
    from millm.core.config import settings

    return FileStore(settings.BATCH_FILES_DIR)
