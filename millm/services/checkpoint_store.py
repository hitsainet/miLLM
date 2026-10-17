"""Durable checkpoint storage (Feature 32, FTASKS 2.0).

Filesystem only: no barrier, no database, no service import, so every rule here is testable over a
`tmp_path`. The service decides when to save and load; this module decides where the bytes go and
that a published checkpoint is either whole or absent.

ROOTS (addendum D-21). The PRIMARY root (`WEIGHTS_CHECKPOINT_DIR`) is the only one written. The
READ roots (`WEIGHTS_CHECKPOINT_READ_ROOTS`, typically miForge's volume mounted read-only) are
searched by load and lineage and never written or cleaned. A `run_id` present under two roots is
refused (`CHECKPOINT_AMBIGUOUS`): guessing which run was meant would load a different model under
the same name.

PUBLISHING (`write_checkpoint_atomically`, FR-32.1.2/3, 032 review 5.7)
    <root>/<run_id>/.step-<NNNNNN>.tmp-<uuid>/   written, every file and directory fsynced
    rename → <root>/<run_id>/step-<NNNNNN>/       then the run directory fsynced
    latest.tmp-<uuid> → os.replace → latest      only AFTER the rename; run directory fsynced
A run directory created by this save is made durable by fsyncing the ROOT, or a power loss can
drop the directory entry and the checkpoint with it. A rename onto a non-empty step directory (a
concurrent save) raises ENOTEMPTY/EEXIST, which is `CHECKPOINT_EXISTS`. A rename onto an EMPTY
directory silently succeeds on Linux; the `exists()` check immediately before the rename is the
only guard against that, and it is a check, not a lock (the service holds a process-wide save
lock, addendum D-20).

CONFINEMENT (`resolve_inside_root`, 032 review 5.8). Every path from a caller, a manifest's
`parent` or a `latest` link is resolved with `realpath` and must stay inside its root by
`commonpath`; a file must be a regular file (a FIFO would hang the worker that opens it).
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from millm.core.config import settings
from millm.core.errors import (
    CheckpointAmbiguousError,
    CheckpointExistsError,
    CheckpointNotFoundError,
    CheckpointWriteFailedError,
    InsufficientStorageError,
    InvalidCheckpointPathError,
    InvalidRunIdError,
)
from millm.core.logging import get_logger
from millm.services.checkpoint_manifest import (
    parse_name,
    step_path_segment,
    validate_run_id,
)

logger = get_logger(__name__)

SHA256_CHUNK = 8 * 1024 * 1024
LATEST = "latest"

#: Exactly what this module creates, and nothing else: miForge's `.tmp-save-<uuid>` and any
#: operator file are never matched.
_STEP_TMP = re.compile(r"\.step-[0-9]{6,}\.tmp-[0-9a-f]{32}")
_LATEST_TMP = re.compile(r"latest\.tmp-[0-9a-f]{32}")

T = TypeVar("T")


# --------------------------------------------------------------------------- roots


def checkpoint_root() -> Path:
    """The primary root: the only one miLLM writes."""
    return Path(settings.WEIGHTS_CHECKPOINT_DIR)


def read_roots() -> list[Path]:
    """The read-only roots, de-duplicated, never including the primary."""
    primary = os.path.realpath(checkpoint_root())
    seen = {primary}
    roots: list[Path] = []
    for raw in settings.weights_checkpoint_read_roots:
        real = os.path.realpath(raw)
        if real not in seen:
            seen.add(real)
            roots.append(Path(raw))
    return roots


def checkpoint_roots() -> list[Path]:
    """Every root load and lineage search: the primary first, then the read roots."""
    return [checkpoint_root(), *read_roots()]


def _is_read_root(root: Path) -> bool:
    real = os.path.realpath(root)
    return any(os.path.realpath(r) == real for r in read_roots())


def find_run_root(run_id: str, roots: Iterable[Path] | None = None) -> Path | None:
    """The one root holding `<run_id>/`, or None. Two roots → `CHECKPOINT_AMBIGUOUS`."""
    validate_run_id(run_id, mode="load")
    found = [root for root in (checkpoint_roots() if roots is None else roots)
             if (Path(root) / run_id).is_dir()]
    if len(found) > 1:
        raise CheckpointAmbiguousError(
            f"run {run_id!r} exists under more than one checkpoint root; refusing to guess",
            details={"run_id": run_id, "roots": [str(r) for r in found]},
        )
    return Path(found[0]) if found else None


# --------------------------------------------------------------------------- confinement


def resolve_inside_root(
    root: str | os.PathLike[str],
    user_path: str | os.PathLike[str],
    *,
    expect: Literal["any", "dir", "file"] = "any",
) -> Path:
    """`user_path` (relative to `root`, or absolute) resolved, refused unless it stays inside
    `root` after every symbolic link is followed. `expect` adds existence and type checks."""
    shown = os.fspath(user_path)
    try:
        real_root = os.path.realpath(root)
        candidate = Path(user_path)
        joined = candidate if candidate.is_absolute() else Path(root) / candidate
        real = os.path.realpath(joined)
        inside = os.path.commonpath([real_root, real]) == real_root
    except (ValueError, OSError):  # NUL byte, mixed absolute/relative, unreadable
        inside = False
    if not inside:
        raise InvalidCheckpointPathError(
            "checkpoint path resolves outside the checkpoint root",
            details={"input": shown},
        )
    if expect == "any":
        return Path(real)
    try:
        mode = os.stat(real).st_mode
    except FileNotFoundError:
        raise CheckpointNotFoundError(
            f"no checkpoint at {shown!r}", details={"path": shown}
        ) from None
    if expect == "dir" and not stat.S_ISDIR(mode):
        raise CheckpointNotFoundError(f"no checkpoint directory at {shown!r}",
                                      details={"path": shown})
    if expect == "file" and not stat.S_ISREG(mode):
        raise InvalidCheckpointPathError(
            "checkpoint file is not a regular file",
            details={"input": shown, "reason": "not_regular_file"},
        )
    return Path(real)


def step_dir(root: str | os.PathLike[str], run_id: str, step: int) -> Path:
    """`<root>/<run_id>/step-<NNNNNN>` (not resolved; `resolve_inside_root` confines it)."""
    return Path(root) / validate_run_id(run_id, mode="load") / step_path_segment(step)


def resolve_checkpoint(run_id: str, step: int,
                       roots: Iterable[Path] | None = None) -> tuple[Path, Path]:
    """(root, real step directory) for a run and step across every root."""
    root = find_run_root(run_id, roots)
    if root is None:
        raise CheckpointNotFoundError(
            f"no run {run_id!r} under any checkpoint root",
            details={"run_id": run_id, "step": step},
        )
    relative = Path(validate_run_id(run_id, mode="load")) / step_path_segment(step)
    return root, resolve_inside_root(root, relative, expect="dir")


def resolve_by_name(name: str, roots: Iterable[Path] | None = None) -> tuple[Path, Path]:
    """(root, real step directory) for a checkpoint `name` (`<base>:<run_id>:step-<n>`)."""
    try:
        _, run_id, step = parse_name(name)
    except ValueError:
        raise CheckpointNotFoundError(
            f"{name!r} is not a checkpoint name", details={"name": name}
        ) from None
    return resolve_checkpoint(run_id, step, roots)


def resolve_path(user_path: str | os.PathLike[str],
                 roots: Iterable[Path] | None = None) -> tuple[Path, Path]:
    """(root, real directory) for a caller path: absolute inside some root, or relative
    `<run_id>/...` resolved in the one root holding that run."""
    roots = list(checkpoint_roots() if roots is None else roots)
    shown = os.fspath(user_path)
    candidate = Path(user_path)
    if candidate.is_absolute():
        for root in roots:
            try:
                return Path(root), resolve_inside_root(root, candidate, expect="dir")
            except InvalidCheckpointPathError:
                continue
        raise InvalidCheckpointPathError(
            "checkpoint path resolves outside every checkpoint root", details={"input": shown}
        )
    parts = candidate.parts
    if not parts:
        raise InvalidCheckpointPathError("empty checkpoint path", details={"input": shown})
    try:
        root = find_run_root(parts[0], roots)
    except InvalidRunIdError:  # an unsafe first segment is a path error here, not a run_id one
        raise InvalidCheckpointPathError(
            "checkpoint path resolves outside the checkpoint root", details={"input": shown}
        ) from None
    if root is None:
        raise CheckpointNotFoundError(f"no checkpoint at {shown!r}", details={"path": shown})
    return root, resolve_inside_root(root, candidate, expect="dir")


# --------------------------------------------------------------------------- sizing, hashing


def estimate_bytes(model: Any, *, overhead_mb: int | None = None) -> int:
    """Parameter and persistent-buffer bytes, each tied tensor once, plus the configured
    allowance for configuration and tokenizer files (FR-32.1.8)."""
    from millm.ml.weights_io import canonical_state

    tensor_bytes = sum(t.numel() * t.element_size() for t in canonical_state(model).values())
    mb = settings.WEIGHTS_SAVE_OVERHEAD_MB if overhead_mb is None else overhead_mb
    return tensor_bytes + mb * 1024 * 1024


def _existing_ancestor(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def check_free_space(root: str | os.PathLike[str], need_bytes: int) -> None:
    """Refuse (`INSUFFICIENT_STORAGE`) unless free space ≥ need × WEIGHTS_SAVE_FREE_SPACE_FACTOR.
    A root that does not exist yet is measured on its nearest existing ancestor."""
    factor = settings.WEIGHTS_SAVE_FREE_SPACE_FACTOR
    free = shutil.disk_usage(_existing_ancestor(Path(root))).free
    required = int(need_bytes * factor)
    if free < required:
        raise InsufficientStorageError(
            f"checkpoint needs {required} bytes free ({need_bytes} estimated x {factor}); "
            f"{free} bytes are free",
            details={"need_bytes": need_bytes, "free_bytes": free, "factor": factor,
                     "required_bytes": required},
        )


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory byte string, hex (the one copy: 032 manifests, 033 artifacts)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk: int = SHA256_CHUNK) -> str:
    """SHA-256 of the whole file, streamed in `chunk`-byte reads to the end."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileFingerprint:
    """What changes when a file is rewritten or replaced; compared before the first write."""

    size: int
    inode: int
    mtime_ns: int

    @classmethod
    def of(cls, path: str | os.PathLike[str]) -> FileFingerprint:
        st = os.stat(path)
        return cls(size=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns)

    @classmethod
    def of_fd(cls, fd: int) -> FileFingerprint:
        """The fingerprint of the INODE an open descriptor holds, wherever its name now points:
        a write into that inode changes it, a rename-replace of its name does not."""
        st = os.fstat(fd)
        return cls(size=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns)


# --------------------------------------------------------------------------- durability


def _fsync_dir(path: str | os.PathLike[str]) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: str | os.PathLike[str]) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(top: Path) -> None:
    """Every regular file, then every directory bottom-up, then `top` itself."""
    for dirpath, _dirnames, filenames in os.walk(top, topdown=False):
        for name in filenames:
            full = os.path.join(dirpath, name)
            if stat.S_ISREG(os.lstat(full).st_mode):
                _fsync_file(full)
        _fsync_dir(dirpath)


@dataclass(frozen=True)
class PublishedCheckpoint:
    path: Path          # the final step directory
    written: Any        # whatever `write_contents` returned


def _write_failed(stage: str, exc: BaseException) -> CheckpointWriteFailedError:
    return CheckpointWriteFailedError(
        f"checkpoint write failed at {stage}: {exc}",
        details={"stage": stage, "error": f"{type(exc).__name__}: {exc}"},
    )


def write_checkpoint_atomically(
    root: str | os.PathLike[str],
    run_id: str,
    step: int,
    write_contents: Callable[[Path], T],
) -> PublishedCheckpoint:
    """Publish `<root>/<run_id>/step-<NNNNNN>` whole or not at all, then point `latest` at it.

    `write_contents(tmp_dir)` writes every file (the manifest last) and returns what the caller
    wants back. Any failure removes the temporary directory and leaves `latest` unchanged.
    """
    root = Path(root)
    validate_run_id(run_id, mode="save")
    if _is_read_root(root):
        raise InvalidCheckpointPathError(
            "checkpoint read roots are read-only",
            details={"input": str(root), "reason": "read_only_root"},
        )
    try:
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / run_id
        created = not run_dir.exists()
        run_dir.mkdir(exist_ok=True)
        resolve_inside_root(root, run_id, expect="dir")
        if created:
            _fsync_dir(root)  # the new run directory's entry survives a power loss
    except OSError as exc:
        raise _write_failed("run_dir", exc) from exc

    final = run_dir / step_path_segment(step)
    relative = f"{run_id}/{final.name}"
    if os.path.lexists(final):
        raise CheckpointExistsError(f"checkpoint {relative} already exists",
                                    details={"path": relative})
    tmp = run_dir / f".{final.name}.tmp-{uuid.uuid4().hex}"
    stage = "temp_dir"
    try:
        tmp.mkdir()
        stage = "write"
        written = write_contents(tmp)
        stage = "fsync"
        _fsync_tree(tmp)
        if os.path.lexists(final):
            raise CheckpointExistsError(f"checkpoint {relative} already exists",
                                        details={"path": relative})
        stage = "rename"
        try:
            os.rename(tmp, final)
        except OSError as exc:
            if exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
                raise CheckpointExistsError(f"checkpoint {relative} already exists",
                                            details={"path": relative}) from exc
            raise
        stage = "fsync_run_dir"
        _fsync_dir(run_dir)
    except BaseException as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        if isinstance(exc, OSError):
            raise _write_failed(stage, exc) from exc
        raise
    try:
        replace_latest(run_dir, final.name)
    except OSError as exc:
        raise _write_failed("latest", exc) from exc
    return PublishedCheckpoint(path=final, written=written)


def replace_latest(run_dir: str | os.PathLike[str], target: str) -> None:
    """Point `<run_dir>/latest` at `target` (a sibling name, so the link is relative) atomically."""
    if not target or "/" in target or target in (".", ".."):
        raise ValueError(f"latest must name a sibling directory, got {target!r}")
    run_dir = Path(run_dir)
    tmp_link = run_dir / f"{LATEST}.tmp-{uuid.uuid4().hex}"
    os.symlink(target, tmp_link)
    try:
        os.replace(tmp_link, run_dir / LATEST)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_link)
        raise
    _fsync_dir(run_dir)


# --------------------------------------------------------------------------- startup cleanup


def remove_stale_temp_dirs(root: str | os.PathLike[str] | None = None) -> list[Path]:
    """Remove this module's crash leftovers under the PRIMARY root: `*/.step-*.tmp-<hex>` and
    `*/latest.tmp-<hex>`, one log line each. Never runs on a read root; never touches a symbolic
    run directory, miForge's `.tmp-save-*` or anything else. Call from `lifespan`."""
    # Called from the lifespan: an unreadable directory is logged and skipped, never raised —
    # an optional feature must not stop serving (the D-05/D-35 principle; R32 F12). That
    # includes the root itself: `Path.is_dir()` RAISES PermissionError when a parent of the
    # root cannot be searched (it ignores only ENOENT-like errors), and a bad read-roots
    # setting raises from `_is_read_root` (R2-B).
    try:
        root = checkpoint_root() if root is None else Path(root)
        if _is_read_root(root):
            logger.warning("checkpoint_temp_cleanup_refused", root=str(root), reason="read_root")
            return []
        if not root.is_dir():
            return []
        run_dirs = sorted(root.iterdir())
    except Exception as exc:  # noqa: BLE001 - never stop serving over a cleanup
        logger.warning("checkpoint_temp_scan_failed", path=str(root), error=str(exc))
        return []
    removed: list[Path] = []
    for run_dir in run_dirs:
        try:
            if run_dir.is_symlink() or not run_dir.is_dir():
                continue
            entries = sorted(run_dir.iterdir())
        except OSError as exc:
            logger.warning("checkpoint_temp_scan_failed", path=str(run_dir), error=str(exc))
            continue
        for entry in entries:
            try:
                if _STEP_TMP.fullmatch(entry.name) and entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                elif _LATEST_TMP.fullmatch(entry.name) and entry.is_symlink():
                    entry.unlink()
                else:
                    continue
            except OSError as exc:
                logger.warning("checkpoint_temp_remove_failed", path=str(entry), error=str(exc))
                continue
            logger.info("checkpoint_temp_removed", path=str(entry))
            removed.append(entry)
    return removed
