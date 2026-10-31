"""Checkpoint save, verified load and lineage over the resident weights (Feature 32).

Contract: `0xcc/reviews/attachable_weights_impl_addendum_2026-10-14.md` (§1 D-01, D-18..D-26,
D-30; §7; §8) over the 032 FPRD/FTDD/FTID; review `attachable_weights_032_doc_review_2026-10-14.md`.
Instantiate (FR-32.8) is DEFERRED and not here.

This module orchestrates. It never touches a tensor (`millm.ml.weights_io` does), never touches a
path without confinement (`checkpoint_store` does), never decides a manifest rule
(`checkpoint_manifest` does), and holds no barrier logic: every state transition is a call into
`WeightBarrier` (031).

SAVE (FR-32.1–FR-32.3; D-19, D-20)
    holder (X-miLLM-Attachment) → caller fields → resident transformers model → step absent
    → free disk → free host memory → process-wide save lock (a second save is
    PAUSE_IN_PROGRESS) → `serving_pause` + `require_clean`, snapshot to host ONLY → leave the
    pause → write, fsync, hash, publish → identity compare-and-set.
    ⚠ The identity is set with `expected_epoch` captured UNDER the pause (review 5.1): a load that
    completed during the seconds-long write has moved the epoch, and naming the new weights after
    this checkpoint would mislabel every `/v1` answer. On a mismatch the checkpoint is published
    and NOTHING else moves (no identity, no `record_save`); `weights_save_identity_superseded`.

LOAD (FR-32.5; D-01, D-21..D-26)
    resolve across roots (ambiguity refused) → manifest read once, validated, consistent →
    fingerprint → safe_open handle → hash (no barrier held) → fingerprint again → mode from the
    header → attachment rule → lease → busy → barrier.
    In place: `internal_write` → fingerprint re-check → `begin_base_replacement` →
    `mark_first_write` → `copy_in_place` from the handle hashed → `set_identity` INSIDE the block.
    Full reload: carrier row → `internal_write` → `ModelService.precheck_load` (a refusal leaves
    the state unchanged) → fingerprint re-check → `begin_base_replacement` (a raising callback
    leaves the state unchanged) → `mark_first_write` →
    `ModelService.load_model(under_barrier=True, allow_weights_origin=True)` + `wait_until_loaded`
    → `set_identity` INSIDE the block.
    ⚠ The identity is set before the barrier releases the slot (review 5.2): a request admitted
    the instant serving resumes runs on the new weights and must capture the new name.

LINEAGE (FR-32.7): manifests only, every link confined to its root, every break named,
`weights_verified: false`.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import stat
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from millm.core.config import settings
from millm.core.errors import (
    CheckpointAmbiguousError,
    CheckpointHashMismatchError,
    CheckpointNotFoundError,
    CheckpointNotLoadableError,
    CheckpointWriteFailedError,
    EngineUnsupportedError,
    InsufficientMemoryError,
    InvalidCheckpointPathError,
    InvalidManifestError,
    InvalidRunIdError,
    MiLLMError,
    ModelBusyError,
    ModelNotFoundError,
    ModelNotLoadedError,
    PauseInProgressError,
    WeightsLoadFailedError,
    WeightsNotCleanError,
)
from millm.core.logging import get_logger
from millm.core.weights_auth import WeightsIdentity
from millm.ml.memory_utils import get_available_cpu_memory_mb
from millm.ml.native_dtype import resolve_for_config
from millm.ml.model_loader import ENGINE_TRANSFORMERS, LoadedModelState
from millm.ml.weights_io import (
    InPlaceVerdict,
    SafetensorsHeaderError,
    TensorSpec,
    compare_specs,
    copy_in_place,
    decide_mode,
    expected_specs,
    read_header,
    allocate_pinned_snapshot,
    release_pinned_host_memory,
    snapshot_to_host,
)
from millm.services import checkpoint_manifest, checkpoint_store
from millm.services.checkpoint_manifest import (
    build_manifest,
    check_name_length,
    parse_name,
    render_name,
    serialise_manifest,
    step_path_segment,
    validate_caller_fields,
    validate_full,
    validate_run_id,
)
from millm.services.checkpoint_store import (
    FileFingerprint,
    checkpoint_root,
    find_run_root,
    read_roots,
    resolve_by_name,
    resolve_inside_root,
    resolve_path,
    sha256_file,
    step_dir,
    write_checkpoint_atomically,
)
from millm.services.resident_identity import (
    ResidentIdentity,
    base_epoch,
    begin_base_replacement,
    covered_base_replacement,
    identity_of,
    set_identity,
)
from millm.services.weight_attachment import get_weight_service
from millm.services.weight_barrier import get_weight_barrier
from millm.services.weights_audit import weights_audit

logger = get_logger(__name__)

#: Audit operation names (addendum D-42): one name per operation everywhere, `weights_<op>`. The
#: routes carry the same names (`name=` in `routes/management/weights_checkpoints.py`), so a token
#: refusal, which `require_weights_identity` audits under the route name, agrees with the
#: service's own accepted and refused lines.
AUDIT_SAVE = "weights_save"
AUDIT_LOAD = "weights_load"

WEIGHTS_FILE = "model.safetensors"
MANIFEST_FILE = "manifest.json"
#: Floating dtypes a full reload serves as stored (addendum D-25): the loader's native rule
#: serves 16-bit weights in their own dtype, so anything else would be cast, not verified.
SERVABLE_FLOAT_DTYPES = frozenset({"F16", "BF16"})
_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"})

#: One save at a time, process-wide (addendum D-20). A threading lock taken WITHOUT blocking:
#: a second save is refused, never queued, so it works from any event loop.
_SAVE_LOCK = threading.Lock()


#: The safetensors dtype each 16-bit load dtype name stores as.
_LOAD_DTYPE_CODES = {"bfloat16": "BF16", "float16": "F16"}
#: Any one of these makes a directory's tokenizer loadable (`AutoTokenizer.from_pretrained`).
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json")


async def _to_completion(awaitable: Any) -> Any:
    """Await `awaitable` as a task that a cancellation does not interrupt: the cancellation
    waits for the task to END, then propagates (the same rule as `_thread_to_completion`)."""
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:  # noqa: BLE001 - the cancellation is what propagates
                break
        raise


async def _thread_to_completion(fn: Any, *args: Any) -> Any:
    """`asyncio.to_thread(fn, *args)`, but a cancellation waits for the thread to END before it
    propagates (R32 F8). A thread cannot be stopped: re-raising at once would let the caller's
    `async with` release the barrier (or the save lock) while the thread was still writing the
    resident weights (or publishing a checkpoint), so the next load could race it."""
    return await _to_completion(asyncio.to_thread(fn, *args))


def _fingerprint_changed(path: Path, fingerprint: FileFingerprint) -> bool:
    """True when `path` no longer has `fingerprint`, including when it is gone (R32 F5)."""
    try:
        return FileFingerprint.of(path) != fingerprint
    except OSError:
        return True


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 1)


def _sha256_bytes(data: bytes) -> str:
    return checkpoint_store.sha256_bytes(data)


def schema_ownership_error() -> Optional[str]:
    """D-35: why the vendored checkpoint-manifest schema cannot be used, or None when it can.

    `assert_schema_ownership` raises `SchemaOwnershipError` naming every field with no derivable
    owner; a missing or unreadable schema raises whatever reading it raises. Either way no save
    can write a manifest anyone can validate and no load can trust one, so both refuse. Checked
    at the moment of each save and load (cheap: the schema is read once and cached), so the
    refusal does not depend on startup having run."""
    try:
        checkpoint_manifest.assert_schema_ownership()
    except Exception as exc:  # noqa: BLE001 - every failure means the same: unusable schema
        return f"{type(exc).__name__}: {exc}"
    return None


def check_schema_ownership_on_startup() -> Optional[str]:
    """Lifespan (D-35): run the ownership check and NEVER raise. Serving starts either way — an
    optional feature must not take serving down (the D-05 principle); a failure is logged at
    error level with the offending fields, and every save and load refuses with reason
    `schema_ownership` until the schema is fixed. Returns the failure, or None."""
    error = schema_ownership_error()
    if error is not None:
        logger.error("weights_schema_ownership_failed", error=error,
                     schema=str(checkpoint_manifest.SCHEMA_PATH),
                     effect="POST /api/weights/save and POST /api/weights/load refuse "
                            "(INVALID_MANIFEST, reason schema_ownership) until it is fixed")
    return error


def _refuse_if_schema_unusable(operation: str) -> None:
    """D-35: refuse a save or load over an unusable vendored schema, naming the reason."""
    error = schema_ownership_error()
    if error is not None:
        raise InvalidManifestError(
            f"{operation} refused: the vendored checkpoint-manifest schema cannot be used "
            f"({error}). Re-vendor docs/schemas/checkpoint-manifest-v1.json from miForge.",
            details={"reason": "schema_ownership", "operation": operation,
                     "errors": [{"json_path": "$", "message": error}]},
        )


@dataclass(frozen=True)
class SaveResult:
    name: str
    path: str
    weights_sha256: str
    manifest_sha256: str
    bytes: int
    snapshot_ms: float
    write_ms: float
    parent_check: str
    latest: str
    identity_set: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LoadResult:
    name: str
    mode: str
    weights_sha256: Optional[str]
    manifest_sha256: Optional[str]
    verify_ms: float
    write_ms: float
    weights_state: str
    steps_since_load: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class _Verified:
    """A checkpoint resolved, its manifest validated and its weights hashed."""

    root: Path
    directory: Path
    weights: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    fingerprint: FileFingerprint
    header_tensors: dict[str, TensorSpec]
    verify_ms: float
    #: A descriptor on the inode that was hashed, held until the load ends (the caller closes
    #: it): the in-place copy reads that inode through the `safe_open` mapping, so a write INTO
    #: it is what can change the bytes copied, and `fstat` on this descriptor sees exactly that.
    pinned_fd: int = -1

    @property
    def relative(self) -> str:
        return f"{self.manifest['run_id']}/{step_path_segment(self.manifest['step'])}"


class WeightsCheckpointService:
    """Save, load and lineage. One per request (it holds no state of its own).

    Args:
        model_service: the `ModelService` (031/032 contract, addendum §8): `get_model`,
            `repository`, `load_model(..., under_barrier=, allow_weights_origin=)`,
            `precheck_load`, `wait_until_loaded`, `unload_model(..., under_barrier=)`, `is_loading`,
            `_refuse_if_leased`.
    """

    def __init__(self, model_service: Any) -> None:
        self._models = model_service

    # --- shared reads -------------------------------------------------------------------------

    @staticmethod
    def _state() -> LoadedModelState:
        return LoadedModelState()

    async def _row(self, model_id: int) -> Any:
        try:
            return await self._models.get_model(model_id)
        except ModelNotFoundError:
            return None

    @staticmethod
    def _resident_base(loaded: Any, row: Any) -> str:
        """The lineage root of the resident (FR-32.4.4, review 5.14): the identity's base for a
        checkpoint; for a row load the HF repo id when the row has one (what miForge writes),
        else the row name."""
        identity = identity_of(loaded)
        if identity is not None:
            return identity.base_model
        if row is not None and row.repo_id:
            return str(row.repo_id)
        return str(loaded.model_name)

    @staticmethod
    def _base_names(loaded: Any, row: Any) -> set[str]:
        """Every base name a checkpoint may carry to be loaded in place over the resident."""
        identity = identity_of(loaded)
        if identity is not None:
            return {identity.base_model}
        names = {str(loaded.model_name)}
        if row is not None and row.repo_id:
            names.add(str(row.repo_id))
        return names

    @staticmethod
    def _require_transformers(loaded: Any, operation: str) -> Any:
        if loaded is None:
            raise ModelNotLoadedError(
                f"No model is loaded; {operation} needs a resident model.",
                details={"operation": operation},
            )
        if loaded.engine != ENGINE_TRANSFORMERS:
            raise EngineUnsupportedError(
                f"{operation} needs a transformers model; the resident model is served by "
                f"{loaded.engine}.",
                details={"operation": operation, "engine": loaded.engine},
            )
        return loaded

    # =========================================================================== save

    async def save(
        self, identity: WeightsIdentity, attachment_id: Optional[str], fields: Any
    ) -> SaveResult:
        """`POST /api/weights/save` (FR-32.1, FR-32.2, FR-32.3). Holder only."""
        ref: Optional[str] = None
        try:
            _refuse_if_schema_unusable("save")
            attachment = get_weight_service().require_holder(identity, attachment_id)
            ref = attachment.ref
            result = await self._save(identity, fields)
        except MiLLMError as e:
            logger.warning("weights_save_refused", code=e.code, label=identity.label)
            weights_audit(AUDIT_SAVE, "refused", identity, code=e.code,
                          reason=e.details.get("reason"), attachment_ref=ref)
            raise
        except Exception as e:
            # Audited like 033's (FR-31.15): every refusal leaves a line, a 500 included (R32 F7).
            logger.error("weights_save_failed_unexpectedly", error_type=type(e).__name__,
                         label=identity.label)
            weights_audit(AUDIT_SAVE, "refused", identity, code="INTERNAL_ERROR",
                          reason=type(e).__name__, attachment_ref=ref)
            raise
        except asyncio.CancelledError:
            # A cancelled save (client gone, shutdown) owes its audit line too, as 031's and
            # 033's cancelled calls do; the cancellation then goes on (R2-B).
            weights_audit(AUDIT_SAVE, "refused", identity, code="CANCELLED", reason="cancelled",
                          attachment_ref=ref)
            raise
        weights_audit(AUDIT_SAVE, "accepted", identity, attachment_ref=ref,
                      checkpoint_sha256=result.weights_sha256, path=result.path)
        return result

    @staticmethod
    def _refuse_if_run_in_read_root(run_id: str) -> None:
        """A save never creates a run a read root already holds (R32 F4). Load and lineage
        refuse a run_id present under two roots (D-21), so writing it under the primary root
        would make the checkpoint just saved, and the read root's run, unloadable by name."""
        found = [str(r) for r in read_roots() if (Path(r) / run_id).is_dir()]
        if found:
            raise CheckpointAmbiguousError(
                f"run {run_id!r} already exists under the read-only checkpoint root(s) {found}; "
                "saving it under the primary root would make that run ambiguous. Save under a "
                "different run_id.",
                details={"run_id": run_id, "roots": [str(checkpoint_root()), *found],
                         "operation": "save"},
            )

    def _check_host_memory(self, need_bytes: int) -> None:
        """FR-32.1.8a: refuse below the snapshot size plus 10%; an unmeasured figure (0) warns
        and proceeds, never refuses (FTID §1 "unmeasured is never zero")."""
        need_mb = int(need_bytes * 1.10 / (1024 * 1024)) + 1
        free_mb = get_available_cpu_memory_mb()
        if free_mb <= 0:
            logger.warning("weights_save_host_memory_unmeasured", need_mb=need_mb)
            return
        if free_mb < need_mb:
            raise InsufficientMemoryError(
                f"Saving needs {need_mb} MB of host memory for the snapshot; {free_mb} MB is free.",
                details={"need_mb": need_mb, "free_mb": free_mb, "where": "host"},
            )

    async def _save(self, identity: WeightsIdentity, fields: Any) -> SaveResult:
        caller = validate_caller_fields(fields)
        run_id, step = caller["run_id"], caller["step"]
        loaded = self._require_transformers(self._state().current, "save")
        root = checkpoint_root()
        relative = f"{run_id}/{step_path_segment(step)}"
        if os.path.lexists(step_dir(root, run_id, step)):
            from millm.core.errors import CheckpointExistsError

            raise CheckpointExistsError(
                f"checkpoint {relative} already exists", details={"path": relative}
            )
        self._refuse_if_run_in_read_root(run_id)
        check_name_length(self._resident_base(loaded, await self._row(loaded.model_id)),
                          run_id, step)
        need = checkpoint_store.estimate_bytes(loaded.model)
        checkpoint_store.check_free_space(root, need)
        self._check_host_memory(need)

        if not _SAVE_LOCK.acquire(blocking=False):
            raise PauseInProgressError.for_operation("save", "save")
        try:
            return await self._save_locked(identity, caller, root, relative)
        finally:
            _SAVE_LOCK.release()

    async def _save_locked(
        self, identity: WeightsIdentity, caller: dict[str, Any], root: Path, relative: str
    ) -> SaveResult:
        barrier = get_weight_barrier()
        logger.info("weights_save_started", path=relative, label=identity.label)
        # Page-locked buffers, allocated BEFORE the pause (finding 4, 2026-10-16): pinning is slow,
        # the copy into pinned memory is about three times faster than into pageable memory, and
        # only the copy belongs inside the hold.
        pinned, pin_reason = await _thread_to_completion(
            functools.partial(
                allocate_pinned_snapshot,
                self._require_transformers(self._state().current, "save").model,
                available_mb=get_available_cpu_memory_mb(),
            )
        )
        started = time.monotonic()
        async with barrier.serving_pause("save", identity) as guard:
            # Read UNDER the pause (FR-32.2.2): nothing can commit or load while it is held.
            barrier.require_clean("save")
            loaded = self._require_transformers(self._state().current, "save")
            epoch = base_epoch()
            before = identity_of(loaded)
            counter = guard.steps_since_load
            row = await self._row(loaded.model_id)
            base_model = self._resident_base(loaded, row)
            check_name_length(base_model, caller["run_id"], caller["step"])
            host = await _thread_to_completion(snapshot_to_host, loaded.model, pinned)
        snapshot_ms = _ms(started)
        used_pinned = pinned is not None
        pinned = None
        if pin_reason is not None:
            logger.info("weights_save_snapshot_pageable", reason=pin_reason, path=relative)

        expected_parent = before.name if before is not None and before.origin == "checkpoint" else None
        parent_check = "match" if caller["parent"] == expected_parent else "mismatch"
        if parent_check == "mismatch":
            logger.warning("manifest_parent_mismatch", run_id=caller["run_id"], step=caller["step"],
                           parent=caller["parent"], expected_parent=expected_parent)

        source_dir = self._source_dir(before, row)
        written: dict[str, Any] = {}
        write_started = time.monotonic()
        box = [host]
        del host
        try:
            published = await _thread_to_completion(
                write_checkpoint_atomically, root, caller["run_id"], caller["step"],
                lambda tmp: self._write_contents(tmp, box, loaded, caller, base_model,
                                                 source_dir, written),
            )
            latest_moved = True
        except CheckpointWriteFailedError as e:
            if e.details.get("stage") != "latest" or "name" not in written:
                raise
            # The step directory IS published (renamed, fsynced); only `latest` did not move.
            # Reported as a failure, honestly, after the identity work below (C2).
            latest_moved = False
            latest_error = e
        except MiLLMError:
            raise
        except Exception as e:  # tokenizer/config writers raise their own types
            raise CheckpointWriteFailedError(
                f"checkpoint write failed: {e}",
                details={"stage": "write", "error": f"{type(e).__name__}: {e}"},
            ) from e
        finally:
            box.clear()
            if used_pinned:
                await _thread_to_completion(release_pinned_host_memory)
        write_ms = _ms(write_started)

        final_dir = root / caller["run_id"] / step_path_segment(caller["step"])
        new_identity = ResidentIdentity(
            name=written["name"],
            base_model=base_model,
            origin="checkpoint",
            checkpoint_dir=os.path.realpath(final_dir),
            manifest_sha256=written["manifest_sha256"],
            expected_parent=written["name"],
            commits_at_identity=counter,
        )
        identity_set = set_identity(loaded, new_identity, expected_epoch=epoch)
        if identity_set:
            get_weight_barrier().record_save(
                written["weights_sha256"], relative, datetime.now(timezone.utc)
            )
        else:
            logger.warning("weights_save_identity_superseded", name=written["name"],
                           expected_epoch=epoch, epoch=base_epoch())

        if not latest_moved:
            raise CheckpointWriteFailedError(
                f"checkpoint {relative} was published, but `latest` could not be moved to it: "
                f"{latest_error.details.get('error')}",
                details={**latest_error.details, "published": True, "path": relative,
                         "name": written["name"], "weights_sha256": written["weights_sha256"],
                         "manifest_sha256": written["manifest_sha256"],
                         "identity_set": identity_set},
            )
        logger.info("weights_save_completed", name=written["name"], path=relative,
                    snapshot_ms=snapshot_ms, write_ms=write_ms, identity_set=identity_set)
        return SaveResult(
            name=written["name"], path=relative, weights_sha256=written["weights_sha256"],
            manifest_sha256=written["manifest_sha256"], bytes=written["bytes"],
            snapshot_ms=snapshot_ms, write_ms=write_ms, parent_check=parent_check,
            latest=published.path.name, identity_set=identity_set,
        )

    @staticmethod
    def _source_dir(identity: Optional[ResidentIdentity], row: Any) -> Optional[Path]:
        """Where the resident's remote-code files live, for copying `auto_map` files."""
        if identity is not None and identity.checkpoint_dir:
            return Path(identity.checkpoint_dir)
        if row is not None and row.cache_path:
            from millm.services.model_service import resolve_cache_path

            return Path(resolve_cache_path(row.cache_path))
        return None

    @staticmethod
    def _auto_map_files(config: Any, source_dir: Optional[Path]) -> list[Path]:
        """The `.py` files `auto_map` names that exist in the source directory (FR-32.1.5)."""
        auto_map = getattr(config, "auto_map", None)
        if not auto_map or source_dir is None:
            return []
        modules: set[str] = set()
        for value in dict(auto_map).values():
            for ref in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(ref, str) and "." in ref:
                    modules.add(ref.split("--")[-1].rsplit(".", 1)[0])
        files = []
        for module in sorted(modules):
            # transformers resolves `<module>.<Class>` to ONE file at the directory's top level
            # (`class_reference.split(".")`), so a module is a bare file stem. Anything else —
            # `..`, a separator, an absolute path — would copy a file from outside the source
            # directory into the checkpoint (R32 F9), and is skipped.
            if not module or any(ch in module for ch in "./\\\x00"):
                logger.warning("checkpoint_auto_map_module_refused", module=module)
                continue
            try:
                candidate = resolve_inside_root(source_dir, f"{module}.py", expect="file")
            except (CheckpointNotFoundError, InvalidCheckpointPathError):
                logger.warning("checkpoint_auto_map_file_missing", module=module)
                continue
            files.append(candidate)
        return files

    def _write_contents(
        self,
        tmp: Path,
        box: list[dict[str, Any]],
        loaded: Any,
        caller: dict[str, Any],
        base_model: str,
        source_dir: Optional[Path],
        written: dict[str, Any],
    ) -> dict[str, Any]:
        """Write the step directory (in the temporary directory). The manifest LAST, after
        `weights_sha256` is read back from the fsynced file (FR-32.1.6, FR-32.3.6)."""
        import shutil

        from safetensors.torch import save_file

        weights = tmp / WEIGHTS_FILE
        save_file(box[0], str(weights), metadata={"format": "pt"})
        box.clear()  # release the host snapshot before hashing (FTID §13)
        # READABLE BY ITS CONSUMERS. safetensors' writer creates the file 0600 whatever the
        # umask, while every sibling below is written 0644: a published checkpoint whose
        # weights only miLLM's uid can open is one no other process can verify or load -
        # found live 2026-10-18, where miForge (another uid, D47's read-only mount) reads
        # these checkpoints to continue a run (F11 14.5).
        os.chmod(weights, 0o644)
        fd = os.open(weights, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        model = loaded.model
        model.config.save_pretrained(tmp)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(tmp)
        if loaded.tokenizer is not None:
            # At the ROOT, beside config.json (R-01.10 corrected 2026-10-13), so the directory
            # loads unchanged through miLLM's loader and through any `from_pretrained`.
            loaded.tokenizer.save_pretrained(tmp)
        for path in self._auto_map_files(model.config, source_dir):
            shutil.copy2(path, tmp / path.name)
        weights_sha256 = sha256_file(weights)
        manifest = build_manifest(caller, base_model=base_model, weights_sha256=weights_sha256)
        manifest_path = tmp / MANIFEST_FILE
        manifest_path.write_bytes(serialise_manifest(manifest))
        written.update(
            name=manifest["name"],
            weights_sha256=weights_sha256,
            manifest_sha256=sha256_file(manifest_path),
            bytes=sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file()),
        )
        return written

    # =========================================================================== load

    async def load(
        self,
        identity: WeightsIdentity,
        request: Any,
        *,
        attachment_id: Optional[str] = None,
        lease_id: Optional[str] = None,
    ) -> LoadResult:
        """`POST /api/weights/load` (FR-32.5; D-01). `request` is a `LoadRequest`."""
        try:
            _refuse_if_schema_unusable("load")
            if request.model_id is not None:
                result = await self._load_row(identity, request, lease_id)
            else:
                result = await self._load_checkpoint(identity, request, attachment_id, lease_id)
        except MiLLMError as e:
            torn = get_weight_barrier().snapshot().weights_state == "torn"
            log = logger.error if isinstance(e, WeightsLoadFailedError) else logger.warning
            log("weights_load_failed_torn" if isinstance(e, WeightsLoadFailedError)
                else "weights_load_refused", code=e.code, label=identity.label, torn=torn)
            weights_audit(AUDIT_LOAD, "refused", identity, code=e.code,
                          reason=e.details.get("reason"))
            raise
        except Exception as e:
            logger.error("weights_load_failed_unexpectedly", error_type=type(e).__name__,
                         label=identity.label,
                         torn=get_weight_barrier().snapshot().weights_state == "torn")
            weights_audit(AUDIT_LOAD, "refused", identity, code="INTERNAL_ERROR",
                          reason=type(e).__name__)
            raise
        except asyncio.CancelledError:
            # Audited like 031's and 033's (R2-B): a cancelled load may have left `torn`, and
            # the audit trail must not be silent about the call that did it.
            logger.warning("weights_load_cancelled", label=identity.label,
                           torn=get_weight_barrier().snapshot().weights_state == "torn")
            weights_audit(AUDIT_LOAD, "refused", identity, code="CANCELLED", reason="cancelled")
            raise
        weights_audit(AUDIT_LOAD, "accepted", identity, checkpoint_sha256=result.weights_sha256,
                      mode=result.mode, path=None if request.model_id is not None else request.path)
        return result

    def _resolve(self, request: Any) -> tuple[Path, Path]:
        if request.name is not None:
            return resolve_by_name(request.name)
        return resolve_path(request.path)

    async def _verify(self, request: Any) -> tuple[_Verified, Any]:
        """Resolve, read the manifest once, validate, hash. Returns the verification and an open
        `safe_open` handle on the bytes that were hashed (the caller closes it)."""
        started = time.monotonic()
        root, directory = self._resolve(request)
        try:
            manifest_path = resolve_inside_root(root, directory / MANIFEST_FILE, expect="file")
        except CheckpointNotFoundError:
            raise CheckpointNotFoundError(
                f"{request.name or request.path!r} has no manifest.json: not a published "
                "checkpoint", details={"name": request.name} if request.name else
                {"path": request.path},
            ) from None
        raw = manifest_path.read_bytes()
        manifest_sha256 = _sha256_bytes(raw)
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidManifestError(
                "manifest.json is not JSON",
                details={"errors": [{"json_path": "$", "message": str(exc)}]},
            ) from None
        manifest = validate_full(doc)
        validate_run_id(manifest["run_id"], mode="load")
        rendered = render_name(manifest["base_model"], manifest["run_id"], manifest["step"])
        errors = []
        if manifest["name"] != rendered:
            errors.append({"json_path": "$.name",
                           "message": f"{manifest['name']!r} is not {rendered!r}"})
        expected_dir = os.path.realpath(
            Path(root) / manifest["run_id"] / step_path_segment(manifest["step"]))
        if os.path.realpath(directory) != expected_dir:
            errors.append({"json_path": "$.step",
                           "message": "the manifest's run_id and step do not name its directory"})
        if errors:
            raise InvalidManifestError(
                "the manifest does not describe its own checkpoint: "
                + "; ".join(e["message"] for e in errors),
                details={"errors": errors},
            )
        if request.name is not None and manifest["name"] != request.name:
            raise CheckpointNotFoundError(
                f"no checkpoint named {request.name!r} (that directory holds "
                f"{manifest['name']!r})", details={"name": request.name},
            )
        weights = resolve_inside_root(root, directory / WEIGHTS_FILE, expect="file")

        # Fingerprint, then open the handle, then hash, then fingerprint again: the handle maps
        # the same inode the hash read, unchanged throughout (review 5.5). The fingerprint is
        # taken from a descriptor that stays open (R32 F6), so the copy can be checked against
        # the inode it really read.
        pinned_fd = os.open(weights, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            return await self._verify_pinned(
                started, root, directory, manifest, manifest_sha256, weights, pinned_fd)
        except BaseException:
            os.close(pinned_fd)
            raise

    async def _verify_pinned(
        self, started: float, root: Path, directory: Path,
        manifest: dict[str, Any], manifest_sha256: str, weights: Path, pinned_fd: int,
    ) -> tuple[_Verified, Any]:
        fingerprint = FileFingerprint.of_fd(pinned_fd)
        handle, handle_error = None, None
        try:
            from safetensors import safe_open

            handle = safe_open(str(weights), framework="pt", device="cpu")
        except Exception as exc:  # noqa: BLE001 - reported after the hash, which outranks it
            handle_error = exc
        actual = await asyncio.to_thread(sha256_file, weights)
        if actual != manifest["weights_sha256"]:
            raise CheckpointHashMismatchError(
                f"{WEIGHTS_FILE} does not match its manifest",
                details={"expected": manifest["weights_sha256"], "actual": actual},
            )
        if _fingerprint_changed(weights, fingerprint):
            raise CheckpointHashMismatchError(
                f"{WEIGHTS_FILE} changed while it was being hashed",
                details={"changed_since_hash": True},
            )
        if handle_error is not None:
            raise CheckpointNotLoadableError(
                f"{WEIGHTS_FILE} is not a readable safetensors file: {handle_error}",
                details={"reason": "safetensors_header"},
            )
        try:
            header = await asyncio.to_thread(read_header, weights)
        except SafetensorsHeaderError as exc:
            raise CheckpointNotLoadableError(
                f"{WEIGHTS_FILE} is not a readable safetensors file: {exc}",
                details={"reason": "safetensors_header"},
            ) from None
        verified = _Verified(
            root=Path(root), directory=Path(directory), weights=weights, manifest=manifest,
            manifest_sha256=manifest_sha256, fingerprint=fingerprint,
            header_tensors=header.tensors, verify_ms=_ms(started), pinned_fd=pinned_fd,
        )
        return verified, handle

    def _check_load_permitted(
        self, identity: WeightsIdentity, attachment_id: Optional[str], mode: str
    ) -> None:
        """The attachment rule (FTID §7; FPRD FR-32.5.8–FR-32.5.10; review 3.3)."""
        state = get_weight_barrier().snapshot().weights_state
        if state == "writing":
            raise WeightsNotCleanError(
                "load refused: a commit is open (the resident weights are writing).",
                details={"operation": "weights_load", "weights_state": state},
            )
        service = get_weight_service()
        view = service.attachment_view()
        if view is None:
            return
        if mode == "full_reload":
            service.refuse_if_attached("weights_load")
            return
        if service.holder_of(identity, attachment_id) is not None:
            return
        if identity.role == "operator" and (state == "torn" or view.stale):
            return
        service.refuse_if_attached("weights_load")

    def _refuse_if_busy(self) -> None:
        if self._models.is_loading or self._state().is_unloading:
            raise ModelBusyError(
                "A model load or unload is in progress; retry once it finishes.",
                details={"operation": "weights_load"},
            )

    @staticmethod
    def _checkpoint_identity(verified: _Verified) -> ResidentIdentity:
        m = verified.manifest
        return ResidentIdentity(
            name=m["name"],
            base_model=m["base_model"],
            origin="checkpoint",
            checkpoint_dir=os.path.realpath(verified.directory),
            manifest_sha256=verified.manifest_sha256,
            expected_parent=m["name"],
            commits_at_identity=0,
        )

    @staticmethod
    def _refuse_changed(verified: _Verified) -> None:
        if _fingerprint_changed(verified.weights, verified.fingerprint):
            raise CheckpointHashMismatchError(
                f"{WEIGHTS_FILE} changed after it was hashed; nothing was written",
                details={"changed_since_hash": True},
            )

    async def _load_checkpoint(
        self,
        identity: WeightsIdentity,
        request: Any,
        attachment_id: Optional[str],
        lease_id: Optional[str],
    ) -> LoadResult:
        verified, handle = await self._verify(request)
        try:
            loaded = self._state().current
            row = await self._row(loaded.model_id) if loaded is not None else None
            transformers = loaded is not None and loaded.engine == ENGINE_TRANSFORMERS
            verdict = (compare_specs(expected_specs(loaded.model), verified.header_tensors)
                       if transformers else InPlaceVerdict(mismatches=()))
            decision = decide_mode(
                verdict,
                engine_is_transformers=transformers,
                base_model_matches=(loaded is not None and verified.manifest["base_model"]
                                    in self._base_names(loaded, row)),
            )
            logger.info("weights_load_started", name=verified.manifest["name"],
                        mode=decision.mode, reasons=list(decision.reasons), label=identity.label)
            self._check_load_permitted(identity, attachment_id, decision.mode)
            self._models._refuse_if_leased(
                "weights_load", loaded.model_id if loaded is not None else 0, lease_id)
            self._refuse_if_busy()
            if decision.mode == "in_place":
                write_ms = await self._in_place(identity, verified, loaded, handle, lease_id)
            else:
                write_ms = await self._full_reload(identity, verified, request, lease_id)
        finally:
            del handle  # the mapping closes with its last reference
            os.close(verified.pinned_fd)
        snap = get_weight_barrier().snapshot()
        logger.info("weights_load_completed", name=verified.manifest["name"],
                    mode=decision.mode, verify_ms=verified.verify_ms, write_ms=write_ms)
        return LoadResult(
            name=verified.manifest["name"], mode=decision.mode,
            weights_sha256=verified.manifest["weights_sha256"],
            manifest_sha256=verified.manifest_sha256, verify_ms=verified.verify_ms,
            write_ms=write_ms, weights_state=snap.weights_state,
            steps_since_load=snap.steps_since_load,
        )

    @staticmethod
    def _refuse_unverified_after_write(verified: _Verified, *, by_path: bool) -> None:
        """R32 F6: the file changed while it was being copied or reloaded, so the bytes now
        served are not known to be the ones hashed. Torn, never named after the checkpoint.

        In place (`by_path=False`) the copy read the hashed inode through the `safe_open`
        mapping, which a rename-replace of the NAME cannot change; only a write into that inode
        can, and the pinned descriptor's `fstat` sees it. A full reload (`by_path=True`) opens
        the directory BY NAME inside `from_pretrained`, so the name must still be the inode
        that was hashed, unchanged."""
        if by_path:
            changed = _fingerprint_changed(verified.weights, verified.fingerprint)
        else:
            try:
                changed = FileFingerprint.of_fd(verified.pinned_fd) != verified.fingerprint
            except OSError:
                changed = True
        if changed:
            raise WeightsLoadFailedError(
                f"{WEIGHTS_FILE} changed while it was being loaded; the weights served are not "
                "known to be the verified bytes, so they are torn",
                details={"stage": "verify", "changed_since_hash": True, "weights_state": "torn"},
            )

    async def _in_place(
        self, identity: WeightsIdentity, verified: _Verified, loaded: Any, handle: Any,
        lease_id: Optional[str] = None,
    ) -> float:
        new_identity = self._checkpoint_identity(verified)
        started = time.monotonic()
        async with get_weight_barrier().internal_write("weights_load", identity) as write:
            self._refuse_changed(verified)
            if self._state().current is not loaded:
                raise ModelBusyError(
                    "The resident model changed while the checkpoint was being verified; retry.",
                    details={"operation": "weights_load"},
                )
            expected = expected_specs(loaded.model)
            if not compare_specs(expected, verified.header_tensors).in_place:
                raise ModelBusyError(
                    "The resident model no longer matches the checkpoint; retry.",
                    details={"operation": "weights_load"},
                )
            # Again INSIDE the barrier (R32 F10): taking the slot waits for the request in
            # flight, and a lease granted meanwhile must still refuse the write. (The full
            # reload's `precheck_load` makes the same re-check.)
            self._models._refuse_if_leased("weights_load", loaded.model_id, lease_id)
            self._refuse_if_busy()
            # Callbacks (033's adapter drop) run here; one that raises aborts with the state
            # unchanged, because the first write is not marked yet.
            begin_base_replacement("weights_load_in_place")
            write.mark_first_write()
            try:
                await _thread_to_completion(copy_in_place, loaded.model, handle)
            except Exception as exc:
                raise WeightsLoadFailedError(
                    f"the in-place copy failed after it began; the weights are torn: {exc}",
                    details={"stage": "copy", "error": f"{type(exc).__name__}: {exc}",
                             "weights_state": "torn"},
                ) from exc
            self._refuse_unverified_after_write(verified, by_path=False)
            # INSIDE the block (review 5.2): set before the slot is released.
            set_identity(loaded, new_identity)
        return _ms(started)

    async def _full_reload(
        self,
        identity: WeightsIdentity,
        verified: _Verified,
        request: Any,
        lease_id: Optional[str],
    ) -> float:
        config = self._preflight_directory(verified)
        floats = {s.dtype for s in verified.header_tensors.values() if s.dtype in _FLOAT_DTYPES}
        if not floats <= SERVABLE_FLOAT_DTYPES or len(floats) > 1:
            raise CheckpointNotLoadableError(
                "A full reload serves 16-bit weights as stored and never casts them; this "
                f"checkpoint holds {sorted(floats)}.",
                details={"reason": "dtype", "dtypes": sorted(floats)},
            )
        if "quantization_config" in config:
            raise CheckpointNotLoadableError(
                "This checkpoint's config.json declares a quantization_config: the loader would "
                "serve it quantized, not as the verified 16-bit bytes (addendum D-25).",
                details={"reason": "quantized"},
            )
        # The loader picks its dtype from config.json (`ml/native_dtype.py`), which the hash does
        # not cover: a recorded dtype other than the weights' — or none at all over float16
        # weights, where the rule's default is bfloat16 — would CAST the verified bytes (R32 F1).
        load_dtype = resolve_for_config("FP16", config).name
        if floats and floats != {_LOAD_DTYPE_CODES.get(load_dtype)}:
            raise CheckpointNotLoadableError(
                f"The loader would load this checkpoint at {load_dtype} (from its config.json), "
                f"but its weights are {sorted(floats)}: that is a cast, not the verified bytes. "
                "Record the weights' dtype in config.json.",
                details={"reason": "dtype", "dtypes": sorted(floats), "load_dtype": load_dtype},
            )
        trust = False
        if "auto_map" in config:
            if not (request.trust_remote_code and identity.role == "operator"):
                raise CheckpointNotLoadableError(
                    "This checkpoint's config.json names remote code (auto_map). Loading it runs "
                    "that code, so it needs trust_remote_code: true from an operator token.",
                    details={"reason": "remote_code"},
                )
            trust = True
        m = verified.manifest
        check_name_length(m["base_model"], m["run_id"], m["step"])
        self._check_path_length(verified)
        row = await self._models.repository.upsert_checkpoint_carrier(
            local_path=os.path.realpath(verified.directory), name=m["name"],
            trust_remote_code=trust,
        )
        return await self._replace_resident(
            identity, row, request.gpu, lease_id,
            new_identity=self._checkpoint_identity(verified),
            before_first_write=lambda: self._refuse_changed(verified),
            after_write=lambda: self._refuse_unverified_after_write(verified, by_path=True),
        )

    @staticmethod
    def _check_path_length(verified: _Verified) -> None:
        """R32 F11: the carrier row stores the directory's real path in `local_path` and
        `cache_path` (both VARCHAR(500)); a longer one would be a database error (a 500 on
        Postgres, silently accepted by SQLite in the tests). Refused before the insert."""
        from millm.db.models.model import Model

        real = os.path.realpath(verified.directory)
        limit = min(int(Model.__table__.c.local_path.type.length),
                    int(Model.__table__.c.cache_path.type.length))
        if len(real) > limit:
            raise CheckpointNotLoadableError(
                f"The checkpoint directory's path is {len(real)} characters; a model row holds "
                f"at most {limit}. Move it to a shorter path or use a shorter run_id.",
                details={"reason": "path_length", "path_length": len(real), "limit": limit},
            )

    @staticmethod
    def _preflight_directory(verified: _Verified) -> dict[str, Any]:
        """What the loader will open, checked before the first write (R32 F2, F3); returns the
        parsed config.json.

        A full reload hands the whole directory to `from_pretrained` AFTER the first write is
        marked, so anything it cannot open there leaves the weights `torn` over a model that
        was replaced for nothing. Refused here instead:
        * an entry that is not a regular file or directory (a FIFO hangs the loader while it
          holds the barrier), or that resolves outside its root through a link (the loader
          would read an unverified file from anywhere) → reason `unsafe_file`;
        * no parseable config.json naming a `model_type`, or no tokenizer file → `incomplete`.
        Only `model.safetensors` is integrity-pinned; config and tokenizer files are not hashed
        (they are checked for shape here, not for content)."""
        unsafe: list[str] = []
        try:
            entries = sorted(os.scandir(verified.directory), key=lambda e: e.name)
        except OSError as exc:
            raise CheckpointNotLoadableError(
                f"the checkpoint directory cannot be listed: {exc}",
                details={"reason": "incomplete", "missing": ["directory"]},
            ) from None
        for entry in entries:
            try:
                resolve_inside_root(verified.root, entry.path)
                mode = os.stat(entry.path).st_mode
            except (InvalidCheckpointPathError, OSError):
                unsafe.append(entry.name)
                continue
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                unsafe.append(entry.name)
        if unsafe:
            raise CheckpointNotLoadableError(
                "The checkpoint directory holds entries the loader must not open (not a regular "
                f"file, or a link out of its checkpoint root): {unsafe}.",
                details={"reason": "unsafe_file", "files": unsafe},
            )
        missing: list[str] = []
        config: dict[str, Any] = {}
        try:
            parsed = json.loads((verified.directory / "config.json").read_text(encoding="utf-8"))
            if isinstance(parsed, dict) and isinstance(parsed.get("model_type"), str):
                config = parsed
            else:
                missing.append("config.json")
        except (OSError, ValueError):
            missing.append("config.json")
        if not any((verified.directory / name).is_file() for name in TOKENIZER_FILES):
            missing.append("tokenizer")
        if missing:
            raise CheckpointNotLoadableError(
                f"A full reload cannot open this checkpoint: missing or unreadable {missing}. "
                "It needs a config.json naming its model_type and a tokenizer beside it.",
                details={"reason": "incomplete", "missing": missing},
            )
        return config

    async def _replace_resident(
        self,
        identity: WeightsIdentity,
        row: Any,
        gpu: Any,
        lease_id: Optional[str],
        *,
        new_identity: Optional[ResidentIdentity],
        before_first_write: Any = None,
        after_write: Any = None,
    ) -> float:
        """A full reload of `row` inside `internal_write` (FR-32.5.8; D-01)."""
        started = time.monotonic()
        state = self._state()
        allow_weights_origin = row.weights_origin is not None
        async with get_weight_barrier().internal_write("weights_load", identity) as write:
            # Every refusal `load_model` can make before it changes anything (placement, memory,
            # a row still loading, a load in progress), made BEFORE the first write is marked:
            # a refusal here leaves the state as it was, never `torn` over an untouched model
            # (W2-G debt). Then the fingerprint, as late as possible before the write.
            await self._models.precheck_load(row.id, gpu, lease_id=lease_id, under_barrier=True,
                                             allow_weights_origin=allow_weights_origin)
            if before_first_write is not None:
                before_first_write()
            # Callbacks (033's adapter drop) run HERE, before the first write is marked (RX F5):
            # one that raises aborts with the state unchanged and nothing unloaded. Run only
            # inside `LoadedModelState.clear`/`set`, after the mark, it left the weights `torn`
            # over a model never touched.
            # The chokepoints this replacement makes, each covered once (P-EPOCH): the
            # resident's unload (`load_model`'s own, or ours for the same row), then the load.
            resident_id = state.loaded_model_id
            chokepoints = ([("model_unload", resident_id)] if resident_id is not None else [])
            chokepoints.append(("model_load", row.id))

            async def replace() -> None:
                with covered_base_replacement("weights_load_full_reload", chokepoints):
                    write.mark_first_write()
                    if resident_id == row.id:
                        await self._models.unload_model(row.id, lease_id=lease_id,
                                                        under_barrier=True)
                    await self._models.load_model(row.id, gpu, lease_id=lease_id,
                                                  under_barrier=True,
                                                  allow_weights_origin=allow_weights_origin)
                    await self._models.wait_until_loaded(row.id)

            try:
                # To completion, like the in-place copy's thread (R32 F8): the load runs in the
                # model service's worker thread, which a cancellation cannot stop. Released at
                # once, the barrier would leave `torn` while that worker went on, and its
                # Feature-1 completion (`on_parameter_set_replaced`, no longer inside an
                # internal write) then turned `torn` into `clean` with the checkpoint served
                # under its own name and never re-checked against the hash (R2-B).
                await _to_completion(replace())
            except Exception as exc:
                if not write.first_write_marked:
                    raise
                raise WeightsLoadFailedError(
                    f"the full reload failed after the resident model was replaced; the weights "
                    f"are torn: {exc}",
                    details={"stage": "reload", "error": f"{type(exc).__name__}: {exc}",
                             "weights_state": "torn"},
                ) from exc
            loaded = state.current
            if loaded is None or loaded.model_id != row.id:
                raise WeightsLoadFailedError(
                    "the full reload finished without the checkpoint resident; the weights are "
                    "torn", details={"stage": "reload", "weights_state": "torn"},
                )
            if after_write is not None:
                after_write()
            # INSIDE the block (review 5.2). A row load's identity is None (origin `model`).
            if new_identity is not None:
                set_identity(loaded, new_identity)
        return _ms(started)

    async def _load_row(
        self, identity: WeightsIdentity, request: Any, lease_id: Optional[str]
    ) -> LoadResult:
        """D-01: reload a plain model row inside the barrier (the recovery from `torn`)."""
        row = await self._models.get_model(request.model_id)
        if row.weights_origin is not None:
            raise CheckpointNotLoadableError(
                f"Model {row.id} is a checkpoint carrier; load it by its name or path so its "
                "files are verified against the manifest.",
                details={"reason": "carrier_row", "model_id": row.id},
            )
        loaded = self._state().current
        self._check_load_permitted(identity, None, "full_reload")
        self._models._refuse_if_leased(
            "weights_load", loaded.model_id if loaded is not None else row.id, lease_id)
        self._refuse_if_busy()
        write_ms = await self._replace_resident(identity, row, request.gpu, lease_id,
                                                new_identity=None)
        snap = get_weight_barrier().snapshot()
        return LoadResult(
            name=row.name, mode="full_reload", weights_sha256=None, manifest_sha256=None,
            verify_ms=0.0, write_ms=write_ms, weights_state=snap.weights_state,
            steps_since_load=snap.steps_since_load,
        )

    # =========================================================================== lineage

    async def lineage(self) -> dict[str, Any]:
        """`GET /api/weights/lineage` (FR-32.7). Reads manifests only, never a weights file."""
        loaded = self._state().current
        if loaded is None:
            raise ModelNotLoadedError("No model is loaded; lineage starts at the resident model.",
                                      details={"operation": "lineage"})
        identity = identity_of(loaded)
        row = await self._row(loaded.model_id)
        steps = get_weight_barrier().snapshot().steps_since_load
        commits = steps - (identity.commits_at_identity if identity is not None else 0)
        resident = {
            "name": identity.name if identity is not None else loaded.model_name,
            "origin": identity.origin if identity is not None else "model",
            "manifest_sha256": identity.manifest_sha256 if identity is not None else None,
            "commits_since_identity": max(commits, 0),
        }
        start = identity.name if identity is not None and identity.origin == "checkpoint" else None
        chain, broken = await asyncio.to_thread(self._walk, start)
        return {
            "resident": resident,
            "base_model": self._resident_base(loaded, row),
            "chain": chain,
            "complete": broken is None,
            "break": broken,
            "weights_verified": False,
        }

    @staticmethod
    def _walk(start: Optional[str]) -> tuple[list[dict[str, Any]], Optional[dict[str, str]]]:
        """Follow `parent` names to manifests (FTID §7), each link confined to its root."""
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        name = start
        max_depth = int(settings.WEIGHTS_LINEAGE_MAX_DEPTH)

        def broken(at: str, reason: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
            logger.info("weights_lineage_break", at=at, reason=reason, links=len(chain))
            return chain, {"at": at, "reason": reason}

        while name is not None:
            if name in seen:
                return broken(name, "cycle")
            if len(chain) >= max_depth:
                return broken(name, "depth_limit")
            seen.add(name)
            try:
                _, run_id, step = parse_name(name)
                validate_run_id(run_id, mode="load")
            except (ValueError, InvalidRunIdError):
                return broken(name, "manifest_invalid")
            try:
                root = find_run_root(run_id)
            except CheckpointAmbiguousError:
                return broken(name, "ambiguous")
            if root is None:
                return broken(name, "manifest_missing")
            relative = f"{run_id}/{step_path_segment(step)}"
            try:
                path = resolve_inside_root(root, f"{relative}/{MANIFEST_FILE}", expect="file")
            except CheckpointNotFoundError:
                return broken(name, "manifest_missing")
            except InvalidCheckpointPathError:
                return broken(name, "manifest_invalid")
            try:
                raw = path.read_bytes()
                doc = validate_full(json.loads(raw.decode("utf-8")))
            except (OSError, UnicodeDecodeError, ValueError, InvalidManifestError):
                return broken(name, "manifest_invalid")
            if doc["name"] != name:
                return broken(name, "name_mismatch")
            if (doc["run_id"], doc["step"]) != (run_id, step):
                return broken(name, "manifest_invalid")
            chain.append({
                "name": doc["name"],
                "run_id": doc["run_id"],
                "step": doc["step"],
                "path": relative,
                "root": str(root),
                "manifest_sha256": _sha256_bytes(raw),
                "weights_sha256": doc["weights_sha256"],
                "parent": doc["parent"],
                "merged_adapter": doc.get("merged_adapter"),
            })
            name = doc["parent"]
        return chain, None


def get_weights_checkpoint_service(model_service: Any) -> WeightsCheckpointService:
    """The service for one request (route dependency)."""
    return WeightsCheckpointService(model_service)
