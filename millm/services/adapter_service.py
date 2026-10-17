"""Serve the resident base with one LoRA adapter applied, and take it off again (Feature 33).

The adapter changes what the model answers without writing a single base tensor: forward hooks
on the targeted `nn.Linear` modules (`millm/ml/adapter_hooks.py`) add `scaling · B·(A·x)`. So it
needs no attachment, no commit bracket and no Feature 31 at all — only a serving pause, so that no
request is mid-forward while `module._forward_hooks` changes under it. **It works with nothing
attached**, which is the common case: miForge trains adapters standalone.

`AdapterService.load` refuses in this order (033 FTDD §5.1 as amended by the addendum), each step
with no side effect, cheapest first:

1. the token — at the route (`WeightsIdentityDep`, D-07): a `trainer` or `operator` token;
2. a model is loaded (`MODEL_NOT_LOADED`), on an engine that runs `nn.Module` hooks and with
   unquantized weights (`ADAPTER_NOT_SERVABLE`, reasons `engine` / `quantization`);
3. continuous batching is off (`ADAPTER_NOT_SERVABLE`, reason `cbm`, D-13): the CBM manager
   generates without the admission slot, so a pause would stop nothing;
4. one admission slot (`BARRIER_UNAVAILABLE`, reason `concurrency`, D-04) — refused here rather
   than after the artifact was read, which is where the pause would refuse it;
5. the lease (`MODEL_LEASED`, D-15): an adapter changes every answer, so a run that leased the
   resident model must not see it change; the holder's `X-miLLM-Lease` passes;
6. the weights are `clean` (`WEIGHTS_NOT_CLEAN`); the pause checks again;
7. the slot (`ADAPTER_ALREADY_LOADED`): claimed with NO `await` between the check and the claim,
   under `AdapterState`'s lock, so two concurrent loads give one adapter and one 409;
8. the base epoch is captured, then the artifact is read, hashed, validated, matched to the base
   and planned OFF the event loop (`adapter_artifact.prepare_adapter`, its own refusals);
9. inside `serving_pause("adapter/load")`: the lease is re-checked (`MODEL_LEASED` — one may
   have been granted while the artifact was being read), the epoch is re-checked
   (`ADAPTER_BASE_CHANGED` — the base was replaced while the artifact was being read; the
   module is read only after this check, since an unload deletes `LoadedModel.model`), the
   hooks are installed all or nothing
   (`ADAPTER_INSTALL_FAILED`), and the state is set — the served name flips in the same step.
   The pause itself may refuse (`QUEUE_FULL`, `MODEL_BUSY`, `PAUSE_IN_PROGRESS`); then nothing
   has been installed.

Three properties are load-bearing, each with a test and a mutation control:

**The epoch check, the install and the state flip happen under one `threading.Lock`, and so does
the drop.** A forced unload runs `LoadedModelState.clear()` in an executor thread while the
event loop may be inside a load's pause (033 review 11, 13). The chokepoint advances the base
epoch BEFORE it runs the drop, so either the load sees the moved epoch and refuses, or the drop
waits for the lock and then removes the adapter the load just installed. Without the lock the
drop can run in the gap between the check and the flip, find nothing, and leave a stale adapter
hooked onto a base it was never verified against.

**A base replacement drops the adapter and never refuses.** `drop_adapter_for_base_change` is
registered with `resident_identity.register_on_base_replaced` when this module is imported (the
router imports it, so the LIVE app has it). A load is the recovery path and gains no
precondition here; a raising callback would abort the replacement, so the drop catches
everything, logs and audits it. It releases the adapter's tensor references before the
replacement frees the base, or the card would stay reserved.

**The served name is the adapter's own `name` while it is applied** (D11), through
`register_served_name_overlay`. The overlay is read inside `_admit` once a request holds the slot
(W0b), and the flip happens inside the pause that holds that slot, so a request can never run
under the hooks and report the base, or the reverse.

Nothing here reads whether a process is attached (FR-33.5.4): an AST test pins it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Sequence

import torch

from millm.core.errors import (
    AdapterAlreadyLoadedError,
    AdapterBaseChangedError,
    AdapterInstallFailedError,
    AdapterInvalidError,
    AdapterNotServableError,
    InsufficientMemoryError,
    MiLLMError,
    ModelNotLoadedError,
    NoAdapterLoadedError,
    WeightsBarrierUnavailableError,
)
from millm.core.logging import get_logger
from millm.ml.adapter_hooks import AdapterHookError, LoraHookSet, PreparedAdapter
from millm.services import model_lease, resident_identity
from millm.services.adapter_artifact import (
    AdapterArtifact,
    adapter_roots,
    map_hook_error,
    prepare_adapter,
    resident_base,
)
from millm.services.adapter_manifest import ValidatorFactory, vendored_validator
from millm.services.weights_audit import weights_audit

if TYPE_CHECKING:
    from millm.core.weights_auth import WeightsIdentity

logger = get_logger(__name__)

OP_LOAD = "adapter/load"
OP_UNLOAD = "adapter/unload"
OP_DROP = "adapter/drop"


@dataclass
class AppliedAdapter:
    """The adapter being served, and the only references to its tensors and hook handles."""

    name: str
    manifest_sha256: str
    weights_sha256: str
    base_name: str
    base_manifest_sha256: Optional[str]
    rank: int
    alpha: float
    scaling: float
    target_count: int
    root_relative: str
    loaded_at: datetime
    #: The barrier's `steps_since_load` when the adapter was applied; None = not recorded.
    steps_at_load: Optional[int]
    #: The base epoch the artifact was verified against.
    base_epoch: int
    prepared: PreparedAdapter
    handles: list = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "manifest_sha256": self.manifest_sha256,
            "weights_sha256": self.weights_sha256,
            "base": {"name": self.base_name, "manifest_sha256": self.base_manifest_sha256},
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": self.scaling,
            "target_count": self.target_count,
        }

    def release_tensors(self) -> None:
        """Drop every reference to A and B (and the modules they were built for)."""
        self.prepared.weights.clear()
        self.prepared.modules.clear()


@dataclass
class _Progress:
    """How far a load or unload got, for the audit line a cancellation still owes."""

    started: float = field(default_factory=time.monotonic)
    #: Set the moment the change took effect (the commit, or the take).
    applied: Optional[AppliedAdapter] = None


class AdapterState:
    """The process's one adapter slot. Every read and write is under `_lock` (033 review 11):
    the drop runs in the unload worker's thread, load and unload on the event loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[AppliedAdapter] = None
        self._loading = False

    @property
    def current(self) -> Optional[AppliedAdapter]:
        with self._lock:
            return self._current

    @property
    def loading(self) -> bool:
        with self._lock:
            return self._loading

    def applied_name(self) -> Optional[str]:
        with self._lock:
            return self._current.name if self._current is not None else None

    def claim(self) -> None:
        """Take the load slot, or refuse `ADAPTER_ALREADY_LOADED`. Synchronous on purpose: the
        caller places it with no `await` between the check and the claim."""
        with self._lock:
            if self._current is not None:
                raise AdapterAlreadyLoadedError(
                    f"Adapter '{self._current.name}' is applied; unload it before loading another.",
                    details={"reason": "applied", "adapter": self._current.name},
                )
            if self._loading:
                raise AdapterAlreadyLoadedError(
                    "Another adapter load is in progress.",
                    details={"reason": "load_in_progress"},
                )
            self._loading = True

    def release_claim(self) -> None:
        with self._lock:
            self._loading = False

    def commit(self, applied: AppliedAdapter, loaded: Any, expected_epoch: int) -> None:
        """Inside the pause: re-check the base epoch, install every hook, flip the state — one
        critical section, so a concurrent drop sees either nothing or the whole adapter.

        Takes the `LoadedModel`, not its `.model`: an unload deletes `LoadedModel.model`, so the
        module is read only AFTER the epoch check has proved the base is still the one verified
        (R1-A finding 1: reading it first turned a replaced base into an AttributeError 500)."""
        with self._lock:
            now = resident_identity.base_epoch()
            if now != expected_epoch:
                raise AdapterBaseChangedError(
                    "The base model was replaced while the adapter was being verified; nothing "
                    "was installed. Load it again against the new base.",
                    details={"verified_epoch": expected_epoch, "epoch": now},
                )
            try:
                applied.handles = LoraHookSet.install(loaded.model, applied.prepared)
            except AdapterHookError as e:
                raise map_hook_error(e) from None
            self._current = applied

    def take(self) -> Optional[AppliedAdapter]:
        """Remove the hooks and empty the slot; return what was applied (None if nothing).

        The handles come off under the lock; the Dynamo reset runs AFTER it is released (033
        review R1-A finding 7). The reset can take long, and the served-name overlay reads this
        lock on the event loop (`_admit` -> `applied_name()`): a drop running the reset in the
        unload worker's thread while holding the lock would stall every request for its length.
        Still before `take` returns, so an unload's reset happens inside its serving pause."""
        with self._lock:
            applied, self._current = self._current, None
            if applied is not None:
                LoraHookSet.detach(applied.handles)
        if applied is not None:
            LoraHookSet.reset_after_change()
        return applied


#: The process-wide slot. Every `AdapterService` built without a `state` shares it, and the drop
#: and the served-name overlay read it, so all three always agree on what is applied.
_STATE = AdapterState()


def get_adapter_state() -> AdapterState:
    return _STATE


def applied_adapter_name() -> Optional[str]:
    """The applied adapter's `name`, or None. Registered as the served-name overlay (D11)."""
    return _STATE.applied_name()


def drop_adapter_for_base_change(reason: str) -> bool:
    """Run inside every base replacement, before the base is moved or freed. NEVER raises.

    Removes the hooks, empties the slot and releases the tensors, then logs and audits the drop.
    Returns whether anything was dropped.
    """
    try:
        applied = _STATE.take()
    except Exception as e:  # noqa: BLE001 — a raising callback would abort the replacement
        logger.error("adapter_drop_failed", reason=reason, error_type=type(e).__name__, error=str(e))
        return False
    if applied is None:
        return False
    try:
        applied.release_tensors()
        logger.warning(
            "adapter_dropped_by_base_change", name=applied.name, reason=reason,
            targets=applied.target_count,
        )
        weights_audit(
            OP_DROP, "accepted", None, reason=reason, adapter=applied.name,
            weights_sha256=applied.weights_sha256, path=applied.root_relative,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("adapter_drop_report_failed", reason=reason, error_type=type(e).__name__)
    return True


# The LIVE registrations. The router imports this module, so `create_app()` has both; each has a
# live-registry test and a removal control.
resident_identity.register_on_base_replaced(drop_adapter_for_base_change)
resident_identity.register_served_name_overlay(applied_adapter_name)


# ---------------------------------------------------------------------------------------------
# Defaults, resolved at call time so tests and the live app can swap each one.
# ---------------------------------------------------------------------------------------------


def _default_loaded() -> Any:
    from millm.ml.model_loader import LoadedModelState

    return LoadedModelState().current


def _default_inference() -> Any:
    from millm.api.dependencies import get_inference_service

    return get_inference_service()


def _default_barrier() -> Any:
    from millm.services.weight_barrier import get_weight_barrier

    return get_weight_barrier()


def _default_leases() -> Any:
    return model_lease.get_lease_registry()


async def _default_row_lookup(model_id: int) -> Any:
    from millm.db.base import async_session_factory
    from millm.db.repositories.model_repository import ModelRepository

    async with async_session_factory() as session:
        return await ModelRepository(session).get_by_id(model_id)


async def _default_registered_names() -> list[str]:
    """Every name a request can address a model row by (D-32): each row's name, its repo id,
    and — for a row that is not a weights-origin carrier — each `:`-prefix of its name, because
    `ModelRepository.find_by_name` falls back from a bare name to `<name>:<tag>`. An adapter
    named any of them would shadow that row (resolver rule 1 sends it to the resident).

    Read through the async session, so the query does not block the event loop."""
    from millm.db.base import async_session_factory
    from millm.db.repositories.model_repository import ModelRepository

    async with async_session_factory() as session:
        rows = await ModelRepository(session).get_all()
    return sorted(addressable_names(rows))


def addressable_names(rows: Sequence[Any]) -> set[str]:
    """The names `rows` can be requested by (see `_default_registered_names`)."""
    names: set[str] = set()
    for row in rows:
        name = getattr(row, "name", None)
        if isinstance(name, str) and name:
            names.add(name)
            if getattr(row, "weights_origin", None) is None:
                parts = name.split(":")
                names.update(":".join(parts[:i]) for i in range(1, len(parts)))
        repo_id = getattr(row, "repo_id", None)
        if isinstance(repo_id, str) and repo_id:
            names.add(repo_id)
    names.discard("")
    return names


def prepare_adapter_mapped(*args: Any, **kwargs: Any) -> AdapterArtifact:
    """`prepare_adapter`, with every exception it can raise made a public refusal (033 review
    R1-B). Runs in the worker thread.

    `prepare_adapter` maps what it expects; what escapes it is an environment failure — an
    `OSError` from the filesystem outside the guarded reads (`ADAPTER_INVALID`, reason
    `unreadable`), or a torch / safetensors `RuntimeError` while placing tensors
    (`ADAPTER_INSTALL_FAILED`, reason `prepare_failed`). Without this they reached the route as an
    unaudited-by-code 500. Only the exception TYPE is carried: its message is not ours and may
    name a path outside the root."""
    try:
        return prepare_adapter(*args, **kwargs)
    except MiLLMError:
        raise
    except (torch.OutOfMemoryError, MemoryError) as e:
        # Out of memory anywhere in the preparation (reading the file, parsing it, placing A and
        # B) is the 507 `prepare_adapter` already gives for the placement — not an
        # install failure, which would send the operator looking for a broken adapter (R2-C).
        raise InsufficientMemoryError(
            "not enough memory to prepare the adapter's tensors; nothing was installed",
            details={"reason": "prepare_out_of_memory", "error": type(e).__name__},
        ) from None
    except OSError as e:
        raise AdapterInvalidError(
            "the adapter could not be read from disk",
            details={"reason": "unreadable", "error": type(e).__name__},
        ) from None
    except Exception as e:  # noqa: BLE001 — mapped, never swallowed
        raise AdapterInstallFailedError(
            "preparing the adapter's tensors failed; nothing was installed",
            details={"reason": "prepare_failed", "error": type(e).__name__},
        ) from None


def _held_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 1)


class AdapterService:
    """Load, unload and report the one adapter. Built per process (`get_adapter_service`)."""

    def __init__(
        self,
        *,
        state: Optional[AdapterState] = None,
        loaded: Callable[[], Any] = _default_loaded,
        inference: Callable[[], Any] = _default_inference,
        barrier: Callable[[], Any] = _default_barrier,
        leases: Callable[[], Any] = _default_leases,
        row_lookup: Callable[[int], Awaitable[Any]] = _default_row_lookup,
        names_lookup: Callable[[], Awaitable[Sequence[str]]] = _default_registered_names,
        roots: Callable[[], Sequence[tuple[str, str]]] = adapter_roots,
        validator_factory: ValidatorFactory = vendored_validator,
    ) -> None:
        self.state = state if state is not None else _STATE
        self._loaded = loaded
        self._inference = inference
        self._barrier = barrier
        self._leases = leases
        self._row_lookup = row_lookup
        self._names_lookup = names_lookup
        self._roots = roots
        self._validator_factory = validator_factory

    # --- refusals ----------------------------------------------------------------------------

    def _resident(self) -> Any:
        loaded = self._loaded()
        if loaded is None:
            raise ModelNotLoadedError(
                "No model is loaded; load a base model before applying an adapter."
            )
        return loaded

    @staticmethod
    def _refuse_unservable_model(loaded: Any) -> None:
        if not loaded.supports_hooks:
            raise AdapterNotServableError(
                f"The resident model runs on {loaded.engine}, which has no forward hooks; an "
                "adapter can be served only on the transformers engine.",
                details={"reason": "engine", "engine": loaded.engine},
            )
        if loaded.quantization_method != "none":
            raise AdapterNotServableError(
                "The resident model's weights are quantized "
                f"({loaded.quantization_method}); an adapter is served only over unquantized "
                "weights.",
                details={"reason": "quantization", "quantization": loaded.quantization_method},
            )

    @staticmethod
    def _refuse_unpausable(inference: Any, operation: str) -> None:
        if inference.cbm_enabled():
            raise AdapterNotServableError(
                f"{operation} refused: continuous batching is enabled and generates without the "
                "admission slot, so serving cannot be paused to change the adapter.",
                details={"reason": "cbm", "operation": operation},
            )
        if int(inference.request_queue.max_concurrent) != 1:
            raise WeightsBarrierUnavailableError.for_reason("concurrency", operation)

    def _refuse_if_leased(self, operation: str, loaded: Any, lease_id: Optional[str]) -> None:
        """The resident model's lease (D-15), through THE shared rule `ModelService` uses
        (`model_lease.refuse_if_leased`): one implementation, so the two cannot drift. The
        target is the resident itself — an adapter changes the model that is loaded."""
        model_lease.refuse_if_leased(
            self._leases(),
            operation=operation,
            resident_model_id=loaded.model_id,
            target_model_id=loaded.model_id,
            lease_id=lease_id,
        )

    def _refused(
        self, operation: str, identity: Optional["WeightsIdentity"], exc: MiLLMError,
        path: Optional[str] = None,
    ) -> None:
        details = exc.details if isinstance(exc.details, dict) else {}
        logger.warning(
            "adapter_refused", operation=operation, code=exc.code,
            reason=details.get("reason"), first=details.get("first"),
            role=identity.role if identity is not None else None,
            label=identity.label if identity is not None else None,
        )
        weights_audit(
            operation, "refused", identity, code=exc.code, reason=details.get("reason"),
            path=path,
        )

    def _failed(
        self, operation: str, identity: Optional["WeightsIdentity"], exc: Exception,
        path: Optional[str] = None,
    ) -> None:
        """An unexpected exception (a 500) still writes the one audit line every gated call owes
        (FR-31.15): refused, `INTERNAL_ERROR`, the exception TYPE as the reason. The message is
        not logged here — it is not ours, and may carry a path or a value."""
        logger.error(
            "adapter_failed", operation=operation, error_type=type(exc).__name__,
            role=identity.role if identity is not None else None,
            label=identity.label if identity is not None else None,
        )
        weights_audit(
            operation, "refused", identity, code="INTERNAL_ERROR", reason=type(exc).__name__,
            path=path,
        )

    # --- load ----------------------------------------------------------------------------------

    def _report(
        self, operation: str, identity: Optional["WeightsIdentity"], applied: AppliedAdapter,
        held_ms: float, *, cancelled: bool = False,
    ) -> None:
        """The success log line and the one `accepted` audit line for a load or an unload that
        took effect — also when the caller was cancelled after it took effect, because the
        adapter IS applied (or gone) and the audit log must say so (R1-A finding 8)."""
        event = "adapter_loaded" if operation == OP_LOAD else "adapter_unloaded"
        logger.info(
            event, name=applied.name, manifest_sha256=applied.manifest_sha256,
            weights_sha256=applied.weights_sha256, targets=applied.target_count,
            rank=applied.rank, held_ms=held_ms, cancelled=cancelled,
            role=identity.role if identity is not None else None,
            label=identity.label if identity is not None else None,
        )
        weights_audit(
            operation, "accepted", identity, adapter=applied.name,
            weights_sha256=applied.weights_sha256, path=applied.root_relative,
        )

    def _cancelled(
        self, operation: str, identity: Optional["WeightsIdentity"], progress: "_Progress",
        path: Optional[str] = None,
    ) -> None:
        """A cancellation (shutdown) still owes the call its one audit line: `accepted` when the
        change had already taken effect, otherwise `refused` with code `CANCELLED`."""
        if progress.applied is not None:
            self._report(
                operation, identity, progress.applied, _held_ms(progress.started),
                cancelled=True,
            )
            return
        logger.warning("adapter_cancelled", operation=operation)
        weights_audit(
            operation, "refused", identity, code="CANCELLED", reason="cancelled", path=path,
        )

    async def load(
        self, identity: Optional["WeightsIdentity"], path: str, lease_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Verify and apply the adapter at `path` (relative to a checkpoint root)."""
        audit_path = path if isinstance(path, str) else None
        progress = _Progress()
        try:
            applied, held_ms = await self._load(identity, path, lease_id, progress)
        except MiLLMError as exc:
            self._refused(OP_LOAD, identity, exc, path=audit_path)
            raise
        except Exception as exc:
            self._failed(OP_LOAD, identity, exc, path=audit_path)
            raise
        except asyncio.CancelledError:
            self._cancelled(OP_LOAD, identity, progress, path=audit_path)
            raise
        self._report(OP_LOAD, identity, applied, held_ms)
        return {
            "adapter": applied.info(),
            "resolved_model_name": resident_identity.served_name(),
            "held_ms": held_ms,
        }

    async def _load(
        self, identity: Optional["WeightsIdentity"], path: str, lease_id: Optional[str],
        progress: "_Progress",
    ) -> tuple[AppliedAdapter, float]:
        loaded = self._resident()
        self._refuse_unservable_model(loaded)
        self._refuse_unpausable(self._inference(), OP_LOAD)
        self._refuse_if_leased(OP_LOAD, loaded, lease_id)
        barrier = self._barrier()
        barrier.require_clean(OP_LOAD)
        # ⚠ NO `await` BETWEEN HERE AND THE CLAIM: a second load must see the claim.
        self.state.claim()
        release_claim = True
        try:
            epoch = resident_identity.base_epoch()
            row = await self._row_lookup(loaded.model_id)
            if row is None:
                row = SimpleNamespace(
                    name=loaded.model_name, repo_id=None, source=None, local_path=None
                )
            # Every name a request can address a row by (D-32): an adapter named like another
            # row would shadow it.
            registered = tuple(await self._names_lookup())
            # Off the loop: a LOCAL row's identity hashes `<local_path>/manifest.json` (D-09),
            # file I/O that would stall every request on a slow or hung volume.
            resident = await asyncio.to_thread(resident_base, row, loaded)
            # Three awaits since `_resident()`: a forced unload in between deletes
            # `LoadedModel.model`, so reading it now would be an AttributeError 500 — the defect
            # R1-A fixed in `commit`, on this earlier path (R2-C). The epoch advances BEFORE the
            # delete, so check it first; the `getattr` covers an unload landing between the two.
            model = self._model_if_unchanged(loaded, epoch)
            prepare = asyncio.ensure_future(asyncio.to_thread(
                prepare_adapter_mapped, path, model, resident,
                validator_factory=self._validator_factory, roots=self._roots(),
                registered_names=registered,
            ))
            try:
                # Shielded: a cancellation cannot stop the worker thread, so the CLAIM must
                # outlive it — released (and any tensors it built dropped) only when the thread
                # has finished, or a second load could prepare alongside it (R1-A finding 8).
                artifact: AdapterArtifact = await asyncio.shield(prepare)
            except asyncio.CancelledError:
                if not prepare.done():
                    release_claim = False
                    prepare.add_done_callback(self._finish_orphaned_prepare)
                else:
                    # The thread finished but the cancellation won the race to this frame: the
                    # tensors it built are in `prepare`, which the propagating exception's
                    # traceback keeps alive. Drop them here; the claim goes in `finally`.
                    self._drop_prepared(prepare)
                raise
            fields = artifact.fields
            applied = AppliedAdapter(
                name=fields.name,
                manifest_sha256=artifact.manifest_sha256,
                weights_sha256=artifact.weights_sha256,
                base_name=fields.base_name,
                base_manifest_sha256=fields.base_manifest_sha256,
                rank=fields.rank,
                alpha=float(fields.alpha),
                scaling=fields.scaling,
                target_count=len(artifact.prepared.targets),
                root_relative=artifact.root_relative,
                loaded_at=datetime.now(timezone.utc),
                steps_at_load=None,
                base_epoch=epoch,
                prepared=artifact.prepared,
            )
            started = time.monotonic()
            progress.started = started
            committed = False
            try:
                async with barrier.serving_pause(OP_LOAD, identity) as guard:
                    # Again inside the pause: a lease granted while the artifact was being read
                    # and hashed (seconds for a large adapter) must still stop the change.
                    self._refuse_if_leased(OP_LOAD, loaded, lease_id)
                    applied.steps_at_load = guard.steps_since_load
                    self.state.commit(applied, loaded, epoch)
                    committed = True
                    progress.applied = applied
            finally:
                if not committed:
                    applied.release_tensors()
            return applied, _held_ms(started)
        finally:
            if release_claim:
                self.state.release_claim()

    @staticmethod
    def _model_if_unchanged(loaded: Any, epoch: int) -> Any:
        """`loaded.model`, or `ADAPTER_BASE_CHANGED` when the base was replaced since `epoch`."""
        now = resident_identity.base_epoch()
        model = getattr(loaded, "model", None) if now == epoch else None
        if model is None:
            raise AdapterBaseChangedError(
                "The base model was replaced while the adapter load was starting; nothing was "
                "read or installed. Load it again against the new base.",
                details={"verified_epoch": epoch, "epoch": resident_identity.base_epoch()},
            )
        return model

    def _finish_orphaned_prepare(self, task: "asyncio.Future[AdapterArtifact]") -> None:
        """The worker thread of a cancelled load has finished: drop what it built, then free the
        slot. Runs on the event loop; never raises."""
        try:
            self._drop_prepared(task)
        finally:
            self.state.release_claim()
            logger.warning("adapter_orphaned_prepare_finished")

    @staticmethod
    def _drop_prepared(task: "asyncio.Future[AdapterArtifact]") -> None:
        """Release the tensors a finished prepare built, if it built any. Never raises."""
        try:
            if not task.cancelled() and task.exception() is None:
                task.result().prepared.weights.clear()
                task.result().prepared.modules.clear()
        except Exception as e:  # noqa: BLE001 — called from a done-callback and a cancel path
            logger.error("adapter_orphaned_prepare_cleanup_failed", error_type=type(e).__name__)

    # --- unload --------------------------------------------------------------------------------

    async def unload(
        self, identity: Optional["WeightsIdentity"], lease_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Remove the applied adapter; the base serves again, bit for bit."""
        progress = _Progress()
        try:
            applied, held_ms = await self._unload(identity, lease_id, progress)
        except MiLLMError as exc:
            self._refused(OP_UNLOAD, identity, exc)
            raise
        except Exception as exc:
            self._failed(OP_UNLOAD, identity, exc)
            raise
        except asyncio.CancelledError:
            if progress.applied is not None:
                progress.applied.release_tensors()
            self._cancelled(OP_UNLOAD, identity, progress)
            raise
        self._report(OP_UNLOAD, identity, applied, held_ms)
        return {
            "unloaded": {"name": applied.name, "manifest_sha256": applied.manifest_sha256},
            "resolved_model_name": resident_identity.served_name(),
            "held_ms": held_ms,
        }

    async def _unload(
        self, identity: Optional["WeightsIdentity"], lease_id: Optional[str],
        progress: "_Progress",
    ) -> tuple[AppliedAdapter, float]:
        if self.state.current is None:
            raise NoAdapterLoadedError("No adapter is applied.")
        self._refuse_unpausable(self._inference(), OP_UNLOAD)
        loaded = self._loaded()
        if loaded is not None:
            self._refuse_if_leased(OP_UNLOAD, loaded, lease_id)
        barrier = self._barrier()
        barrier.require_clean(OP_UNLOAD)
        started = time.monotonic()
        progress.started = started
        async with barrier.serving_pause(OP_UNLOAD, identity):
            # Again inside the pause, as for a load (R1-A finding 2): the pause waits for the
            # request in flight — a whole stream — and a lease granted meanwhile must still stop
            # the change, since removing the adapter changes every answer the holder sees (R2-C).
            current = self._loaded()
            if current is not None:
                self._refuse_if_leased(OP_UNLOAD, current, lease_id)
            applied = self.state.take()
            if applied is None:  # a base replacement dropped it while the pause was taken
                raise NoAdapterLoadedError("No adapter is applied.")
            progress.applied = applied
        held_ms = _held_ms(started)
        # After the pause, so the slot is not held while references go; no cache release here.
        applied.release_tensors()
        return applied, held_ms

    # --- status --------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The adapter block (D-28; `AdapterStatus`). `base_steps_since_load` is None when not
        recorded. `base_name` is the resident's served name; `manifest_base_name` /
        `manifest_base_sha256` are the base the adapter's manifest names (FR-33.4.5)."""
        applied = self.state.current
        if applied is None:
            return {
                "applied": False, "name": None, "base_name": None, "manifest_sha256": None,
                "manifest_base_name": None, "manifest_base_sha256": None,
                "weights_sha256": None, "rank": None, "target_count": None, "loaded_at": None,
                "base_steps_since_load": None,
            }
        return {
            "applied": True,
            "name": applied.name,
            # The RESIDENT base's served name; the base the manifest names is below.
            "base_name": resident_identity.base_served_name(),
            "manifest_sha256": applied.manifest_sha256,
            "manifest_base_name": applied.base_name,
            "manifest_base_sha256": applied.base_manifest_sha256,
            "weights_sha256": applied.weights_sha256,
            "rank": applied.rank,
            "target_count": applied.target_count,
            "loaded_at": applied.loaded_at,
            "base_steps_since_load": self._base_steps_since(applied),
        }

    def _base_steps_since(self, applied: AppliedAdapter) -> Optional[int]:
        """Commits to the base since the adapter was applied (D-16). None when the count at
        load was not recorded, the barrier cannot be read, or the counter has since been reset
        (a reset means the parameter set was replaced, which would have dropped the adapter)."""
        if applied.steps_at_load is None:
            return None
        try:
            now = int(self._barrier().snapshot().steps_since_load)
        except Exception as e:  # noqa: BLE001 — status never 500s over a counter
            logger.warning("adapter_status_steps_unavailable", error_type=type(e).__name__)
            return None
        if now < applied.steps_at_load:
            return None
        return now - applied.steps_at_load


_service: Optional[AdapterService] = None
_service_lock = threading.Lock()


def get_adapter_service() -> AdapterService:
    """The process-wide service (a FastAPI dependency; tests override it)."""
    global _service
    with _service_lock:
        if _service is None:
            _service = AdapterService()
        return _service
