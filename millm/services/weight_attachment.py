"""Weight attachment: which trainer, if any, holds the resident weights (Feature 31).

Contract: addendum §2.1 and §7; 031 FTID §3.3 and §7.3; 031 review D3, D14, D16, I10, I11.

- **One attachment at a time** (FR-31.2). `attach` checks and RESERVES under one lock with no
  `await` between, so two concurrent attaches give one attachment; the reservation is visible
  to `refuse_if_attached` while the handles are exported off the event loop, so no load can
  start in between (FTID §7.3).
- **The attachment ID is returned once and never stored.** Only its SHA-256 digest is kept;
  logs carry `ref`, the digest's first 8 hex (FR-31.1.6). A holder call digests the presented
  ID and compares in constant time, plus the token label that attached (FPRD D7).
- **Check order for a holder call** (031 review I11): the token (401/403, the route dependency)
  -> nothing attached `NOT_ATTACHED` -> missing or mismatched ID or label
  `NOT_ATTACHMENT_HOLDER` -> state checks.
- **Staleness is computed on every read** from a monotonic clock (FR-31.4.2); the background
  `tick` only logs it once. Stale changes nothing about serving (FR-31.4.3), still blocks an
  attach and every model load (FR-31.2.1, FR-31.5.3), and a heartbeat from the same, uncleared
  attachment revives it (FPRD D9). Only an operator clears it, and never an active one (D11).
- **Detach keeps the values** (FR-31.3): a clean quiesce is resumed; a commit open (`writing`)
  becomes `torn` at once (D16); `torn` stays `torn`.
- **Addresses are fixed** (FR-31.5.4): every parameter's `data_ptr()` is recorded at attach and
  checked at quiesce, commit-end and detach; a moved address marks the set `torn`.

Public names other features use (addendum §2.1): `get_weight_service`, `is_attached`,
`attachment_view`, `holder_of` (never raises; R-01.31), `require_holder`, `refuse_if_attached`,
`status`.

`weight_attachment` imports the barrier; the barrier never imports this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

from millm.core.errors import (
    AlreadyAttachedError,
    AttachmentNotStaleError,
    NotAttachedError,
    NotAttachmentHolderError,
    WeightsForbiddenError,
    WeightsTornError,
)
from millm.core.logging import get_logger
from millm.core.weights_auth import WeightsIdentity
from millm.services import weight_handles
from millm.services.weight_barrier import get_weight_barrier

if TYPE_CHECKING:
    from millm.api.schemas.weights import WeightsStatus
    from millm.api.schemas.weights_adapters import AdapterStatus

logger = get_logger(__name__)

#: How often `weights_tick_loop` calls `WeightService.tick` (031 FTID §7.7).
TICK_INTERVAL_S = 5.0

#: How many moved parameter names the error log line lists (FTID §12).
_MOVED_LOGGED = 5


def _digest(value: str) -> str:
    """SHA-256 hex of an attachment ID."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ref(digest: str) -> str:
    """The only form of an attachment ID a log line or a response carries."""
    return digest[:8]


@dataclass(frozen=True)
class AttachmentView:
    """A public read of the attachment. `ref` is the first 8 hex of the ID's digest."""

    present: bool
    stale: bool
    holder_label: Optional[str]
    ref: Optional[str]


@dataclass
class Attachment:
    """One attachment (031 FTDD §4). `digest` is the SHA-256 of the ID; the ID itself is never
    kept. `pending` is True between the reservation and the end of the export: the attachment
    blocks other attaches and model loads, but has no holder yet."""

    digest: str
    ref: str
    identity: WeightsIdentity
    attached_at: datetime
    model_id: int
    last_heartbeat_mono: float = 0.0
    last_heartbeat_at: Optional[datetime] = None
    stale_logged: bool = False
    pending: bool = True
    addresses: dict[str, int] = field(default_factory=dict, repr=False)
    #: Every shared storage, held for the attachment's lifetime so nothing is freed under an
    #: open handle (FTID §3.4).
    keep_alive: list[Any] = field(default_factory=list, repr=False)
    #: The model object the handles point into; the address check reads it.
    model: Any = field(default=None, repr=False)


@dataclass
class AttachResult:
    """What `attach` returns to the route (031 FTDD §5.3). The ID is in it exactly once; the
    repr never shows the ID or a handle, so an accidental log of the object leaks neither."""

    attachment_id: str = field(repr=False)
    ref: str
    model: dict[str, Any]
    weights_state: str
    steps_since_load: int
    heartbeat: dict[str, float]
    producer: dict[str, Optional[str]]
    storages: list[dict[str, Any]] = field(repr=False)
    parameters: list[dict[str, Any]] = field(repr=False)
    #: The release contract (addendum D-47): what stays pinned until every handle is released.
    release: dict[str, Any] = field(default_factory=dict)
    #: The served model's architecture config exactly as loaded (D39), so a trainer rebuilds
    #: the graph this process serves rather than one fetched from elsewhere.
    config: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """The attach response body (FTDD §5.3)."""
        return {
            "attachment_id": self.attachment_id,
            "model": dict(self.model),
            "weights_state": self.weights_state,
            "steps_since_load": self.steps_since_load,
            "heartbeat": dict(self.heartbeat),
            "producer": dict(self.producer),
            "storages": list(self.storages),
            "parameters": list(self.parameters),
            "release": dict(self.release),
            "config": None if self.config is None else dict(self.config),
        }


#: The release contract stated in every attach response (addendum D-47; hardware record
#: 2026-10-16, finding 2). CUDA IPC counts a reference for every exported storage that only the
#: CONSUMER's release closes; until it closes, torch cannot free the block. Measured on the node: an
#: attach whose handles were never rebuilt, a trainer killed with handles open, or one that exited
#: without dropping its tensors left the whole parameter set reserved through detach, clear-stale,
#: a reload and an unload, until miLLM restarted.
RELEASE_CONTRACT = (
    "Rebuild every storage exactly once (one rebuild per storage entry, then a view per "
    "parameter), and before you exit or detach for good, drop every rebuilt tensor and call "
    "torch.cuda.ipc_collect(). Handles that are never rebuilt, or a process that dies or exits "
    "still holding them, keep pinned_bytes of miLLM's GPU memory reserved until miLLM restarts; "
    "detach, clear-stale, a reload and an unload do not return it."
)


def _default_inference() -> Any:
    from millm.api.dependencies import get_inference_service

    return get_inference_service()


def _default_loaded_state() -> Any:
    from millm.ml.model_loader import LoadedModelState

    return LoadedModelState()


def _default_load_claimed() -> bool:
    """Whether a Feature-1 load has claimed the process-wide load slot (031 review D3), through
    `ModelService.is_loading` (addendum §8). Imported lazily: `model_service` imports this
    module."""
    from millm.services.model_service import ModelService

    return bool(ModelService.is_loading)


def _ipc_collect() -> None:
    """Reclaim CUDA IPC blocks the consumer released (FR-31.3.3). Never creates a context."""
    import torch

    if torch.cuda.is_initialized():
        torch.cuda.ipc_collect()


def _wall() -> datetime:
    return datetime.now(timezone.utc)


def _settings() -> Any:
    from millm.core.config import settings

    return settings


class WeightService:
    """Process-wide attachment state (one instance, `get_weight_service()`).

    Every collaborator is injectable so the rules are testable without a GPU, a model or a
    running server: the barrier, the inference service (continuous batching, slot count), the
    resident-model state, the load-slot probe, the exporter, the clocks and `ipc_collect`.
    """

    def __init__(
        self,
        *,
        barrier: Callable[[], Any] = get_weight_barrier,
        inference: Callable[[], Any] = _default_inference,
        loaded_state: Callable[[], Any] = _default_loaded_state,
        load_claimed: Callable[[], bool] = _default_load_claimed,
        export: Callable[..., weight_handles.ExportedHandles] = weight_handles.export_handles,
        producer: Callable[[], dict[str, Optional[str]]] = weight_handles.producer_versions,
        moved: Callable[[Any, dict[str, int]], list[str]] = weight_handles.moved_addresses,
        ipc_collect: Callable[[], None] = _ipc_collect,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = _wall,
    ) -> None:
        self._barrier = barrier
        self._inference = inference
        self._loaded_state = loaded_state
        self._load_claimed = load_claimed
        self._export = export
        self._producer = producer
        self._moved = moved
        self._ipc_collect = ipc_collect
        self._clock = clock
        self._wall = wall
        self._lock = threading.Lock()
        self._attachment: Optional[Attachment] = None
        #: Handle exports this process has made (each opens IPC reference counts; finding 2).
        self._exports_since_start = 0

    # --- reads -----------------------------------------------------------------------------

    @staticmethod
    def _timeout_s() -> float:
        return float(_settings().WEIGHTS_HEARTBEAT_TIMEOUT_S)

    def _is_stale(self, attachment: Attachment, now: float) -> bool:
        """Stale = no heartbeat within the window. Computed on read (FR-31.4.2); a pending
        attachment is still being exported and is never stale."""
        if attachment.pending:
            return False
        return now - attachment.last_heartbeat_mono > self._timeout_s()

    def _stale_since(self, attachment: Attachment) -> Optional[str]:
        if attachment.last_heartbeat_at is None:
            return None
        return (attachment.last_heartbeat_at + timedelta(seconds=self._timeout_s())).isoformat()

    def is_attached(self) -> bool:
        """Whether any attachment exists: active, stale, or still being exported."""
        return self._attachment is not None

    def attachment_view(self) -> Optional[AttachmentView]:
        """Presence, staleness, holder label and ref; None when nothing is attached."""
        with self._lock:
            attachment = self._attachment
            if attachment is None:
                return None
            stale = self._is_stale(attachment, self._clock())
            return AttachmentView(
                present=True,
                stale=stale,
                holder_label=attachment.identity.label,
                ref=attachment.ref,
            )

    def _match(
        self, identity: WeightsIdentity, attachment_id: Optional[str]
    ) -> tuple[Optional[Attachment], Optional[str]]:
        """(the attachment when this caller holds it, else None; why not, else None).

        Why not is `not_attached` (nothing attached, or only a reservation nobody holds yet)
        or `not_holder`. The digest comparison is constant time, and the label check runs
        whatever the digest said.
        """
        with self._lock:
            attachment = self._attachment
        if attachment is None or attachment.pending:
            return None, "not_attached"
        presented = _digest(attachment_id) if attachment_id else ""
        digest_ok = hmac.compare_digest(presented, attachment.digest)
        label_ok = hmac.compare_digest(
            identity.label.encode("utf-8"), attachment.identity.label.encode("utf-8")
        )
        if digest_ok and label_ok:
            return attachment, None
        return None, "not_holder"

    def holder_of(
        self, identity: WeightsIdentity, attachment_id: Optional[str]
    ) -> Optional[Attachment]:
        """The attachment when `identity` presenting `attachment_id` holds it, else None.

        Never raises, so a load can recognise the holder without requiring one (R-01.31).
        """
        attachment, _ = self._match(identity, attachment_id)
        return attachment

    def require_holder(
        self, identity: WeightsIdentity, attachment_id: Optional[str]
    ) -> Attachment:
        """The held attachment, or `NOT_ATTACHED` (nothing attached) / `NOT_ATTACHMENT_HOLDER`
        (missing or wrong ID, or another label), in that order (031 review I11)."""
        attachment, why = self._match(identity, attachment_id)
        if attachment is not None:
            return attachment
        if why == "not_attached":
            raise NotAttachedError(
                "Nothing is attached to the resident weights; attach first.",
                details={"label": identity.label},
            )
        raise NotAttachmentHolderError(
            "This call needs the attachment's holder: the X-miLLM-Attachment header is "
            "missing or names another attachment, or the token is not the one that attached.",
            details={"label": identity.label, "header_present": bool(attachment_id)},
        )

    def _already_attached(self, attachment: Attachment, operation: str) -> AlreadyAttachedError:
        state = "stale" if self._is_stale(attachment, self._clock()) else "active"
        return AlreadyAttachedError.for_attachment(
            role=attachment.identity.role,
            label=attachment.identity.label,
            state=state,
            attached_at=attachment.attached_at.isoformat(),
            operation=operation,
        )

    def refuse_if_attached(self, operation: str) -> None:
        """Raise `ALREADY_ATTACHED` while any attachment exists, stale or still being exported
        (FR-31.5.2, FR-31.5.3). Names the holder, never the ID (FR-31.2.2)."""
        with self._lock:
            attachment = self._attachment
            if attachment is not None:
                raise self._already_attached(attachment, operation)

    # --- attach ----------------------------------------------------------------------------

    def _transition(self) -> Optional[str]:
        """The model transition in progress, or None (031 review D3).

        An unload in progress, a claimed load slot, or a barrier-owned write or pause
        (`hold_owner` `internal:<op>`). A trainer's own hold left in `torn` after a detach is
        not a transition: re-attaching to a torn set is how the holder finds out (acceptance 7).
        """
        state = self._loaded_state()
        if state.is_unloading:
            return "unloading"
        if self._load_claimed():
            return "loading"
        owner = self._barrier().snapshot().hold_owner
        if owner is not None and owner.startswith("internal:"):
            return "internal_write"
        return None

    def _check(self) -> Any:
        """Run `check_attachable` against the live facts; returns the resident model."""
        inference = self._inference()
        loaded = self._loaded_state().current
        weight_handles.check_attachable(
            loaded,
            cbm_enabled=bool(inference.cbm_enabled()),
            max_concurrent=int(inference.request_queue.max_concurrent),
            transition=self._transition(),
            weights_state=self._barrier().snapshot().weights_state,
        )
        return loaded

    async def attach(self, identity: WeightsIdentity) -> AttachResult:
        """Attach `identity` to the resident weights (FR-31.1, FR-31.2; FTID §7.3).

        Raises:
            AlreadyAttachedError: an attachment, active, stale or pending, exists.
            NotAttachableError: the model or the deployment cannot be attached (any reason).
            ModelNotLoadedError: nothing is resident.
        """
        with self._lock:
            if self._attachment is not None:
                raise self._already_attached(self._attachment, "attach")
            loaded = self._check()
            attachment_id = secrets.token_urlsafe(24)
            digest = _digest(attachment_id)
            now_wall = self._wall()
            reservation = Attachment(
                digest=digest,
                ref=_ref(digest),
                identity=identity,
                attached_at=now_wall,
                model_id=loaded.model_id,
                last_heartbeat_mono=self._clock(),
                last_heartbeat_at=now_wall,
                pending=True,
                model=loaded.model,
            )
            self._attachment = reservation
        try:
            exported = await asyncio.to_thread(self._export, loaded.model)
            with self._lock:
                self._exports_since_start += 1
            producer = self._producer()
            with self._lock:
                # The model must still be the one exported, and nothing may have started
                # moving it meanwhile (a pause or an unload that did not ask this service).
                # And the reservation must still be the registered attachment: one removed
                # meanwhile would hand out an ID nothing recognises, with handles open on weights
                # no `refuse_if_attached` protects any more.
                if self._attachment is not reservation:
                    raise NotAttachedError(
                        "The attachment was removed while its handles were exported; attach "
                        "again.",
                        details={"operation": "attach", "label": identity.label},
                    )
                transition = self._transition()
                if self._loaded_state().current is not loaded or transition is not None:
                    raise weight_handles.not_attachable(
                        "model_transition", transition=transition or "model_replaced"
                    )
                reservation.addresses = dict(exported.addresses)
                reservation.keep_alive = list(exported.keep_alive)
                reservation.last_heartbeat_mono = self._clock()
                reservation.last_heartbeat_at = self._wall()
                reservation.pending = False
        except BaseException:
            with self._lock:
                if self._attachment is reservation:
                    self._attachment = None
            raise
        snap = self._barrier().snapshot()
        from millm.services.resident_identity import base_served_name, identity_of

        resident = identity_of(loaded)
        logger.info(
            "weights_attached",
            role=identity.role,
            label=identity.label,
            attachment_ref=reservation.ref,
            model_id=loaded.model_id,
            storages=len(exported.storages),
            parameters=len(exported.parameters),
        )
        settings = _settings()
        return AttachResult(
            attachment_id=attachment_id,
            ref=reservation.ref,
            model={
                "model_id": loaded.model_id,
                "name": base_served_name(loaded),
                "manifest_hash": resident.manifest_sha256 if resident is not None else None,
            },
            weights_state=snap.weights_state,
            steps_since_load=snap.steps_since_load,
            heartbeat={
                "timeout_s": float(settings.WEIGHTS_HEARTBEAT_TIMEOUT_S),
                "interval_hint_s": float(settings.WEIGHTS_HEARTBEAT_INTERVAL_HINT_S),
            },
            producer=producer,
            storages=exported.storages,
            parameters=exported.parameters,
            release={
                "pinned_bytes": sum(int(s["storage_size_bytes"]) for s in exported.storages),
                "contract": RELEASE_CONTRACT,
            },
            config=weight_handles.served_config(loaded.model),
        )

    # --- fixed addresses -------------------------------------------------------------------

    def verify_addresses(self, context: str) -> list[str]:
        """Compare every parameter's address with the attach-time record (FR-31.5.4).

        A difference marks the set `torn` and logs `weights_address_moved` at error level.
        Returns the moved names (empty when nothing moved or nothing is attached).
        """
        with self._lock:
            attachment = self._attachment
        if attachment is None or attachment.pending:
            return []
        moved = self._moved(attachment.model, attachment.addresses)
        if moved:
            logger.error(
                "weights_address_moved",
                parameters=moved[:_MOVED_LOGGED],
                moved_count=len(moved),
                context=context,
                attachment_ref=attachment.ref,
            )
            self._barrier().mark_torn("address_moved")
        return moved

    def _raise_if_moved(self, context: str) -> None:
        moved = self.verify_addresses(context)
        if moved:
            raise WeightsTornError(
                "A parameter's address moved while attached, so the resident weights are "
                "torn and not served. A load of a checkpoint restores serving.",
                details={"reason": "address_moved", "context": context, "count": len(moved)},
            )

    # --- holder calls ----------------------------------------------------------------------

    async def quiesce(self, identity: WeightsIdentity, attachment_id: Optional[str]) -> Any:
        """The holder's quiesce, then the address check (FR-31.5.4). Returns the barrier's
        result."""
        attachment = self.require_holder(identity, attachment_id)
        barrier = self._barrier()
        # A detach or clear DURING the drain is caught by the barrier itself once it holds the
        # slot (it raises NOT_ATTACHED and releases); this re-check covers one landing after
        # the barrier's last check and before this line (031 review D14).
        result = await barrier.quiesce(attachment.digest, identity)
        with self._lock:
            still_held = self._attachment is attachment
        if not still_held:
            # Idempotent: the detach already released the hold unless it raced the drain.
            barrier.on_holder_detached(attachment.digest, tear_writing=False)
            raise NotAttachedError(
                "The attachment was detached or cleared while quiesce waited; serving resumed.",
                details={"label": identity.label},
            )
        self._raise_if_moved("quiesce")
        return result

    def resume(self, identity: WeightsIdentity, attachment_id: Optional[str]) -> Any:
        """The holder's resume; returns the barrier's result (FR-31.8)."""
        attachment = self.require_holder(identity, attachment_id)
        return self._barrier().resume(attachment.digest)

    def commit_begin(self, identity: WeightsIdentity, attachment_id: Optional[str]) -> Any:
        """The holder's commit-begin; returns the barrier's result (FR-31.10)."""
        attachment = self.require_holder(identity, attachment_id)
        return self._barrier().commit_begin(attachment.digest)

    def commit_end(self, identity: WeightsIdentity, attachment_id: Optional[str]) -> Any:
        """The holder's commit-end, then the address check. Checked AFTER: a tear recorded
        before would be undone by the late commit-end that follows it."""
        attachment = self.require_holder(identity, attachment_id)
        result = self._barrier().commit_end(attachment.digest)
        self._raise_if_moved("commit_end")
        return result

    async def detach(
        self, identity: WeightsIdentity, attachment_id: Optional[str]
    ) -> dict[str, Any]:
        """Release the attachment and keep the values (FR-31.3). Holder only."""
        attachment = self.require_holder(identity, attachment_id)
        self.verify_addresses("detach")
        # The barrier decides what leaving does, under its own lock (FR-31.3.2, D16): a clean
        # quiesce is released, an open commit becomes `torn`, `torn` stays `torn`, and a hold
        # an internal op adopted is released when that op ends instead of handed back.
        self._barrier().on_holder_detached(attachment.digest)
        with self._lock:
            if self._attachment is attachment:
                self._attachment = None
        await asyncio.to_thread(self._ipc_collect)
        snap = self._barrier().snapshot()
        logger.info(
            "weights_detached",
            role=identity.role,
            label=identity.label,
            attachment_ref=attachment.ref,
            weights_state=snap.weights_state,
        )
        return {"weights_state": snap.weights_state, "barrier": snap.barrier}

    def heartbeat(
        self, identity: WeightsIdentity, attachment_id: Optional[str]
    ) -> dict[str, Any]:
        """Prove the holder is alive (FR-31.4). Revives a stale, uncleared attachment (D9)."""
        attachment = self.require_holder(identity, attachment_id)
        with self._lock:
            if self._attachment is not attachment:
                raise NotAttachedError(
                    "The attachment was cleared or detached; attach again.",
                    details={"label": identity.label},
                )
            now = self._clock()
            was_stale = self._is_stale(attachment, now)
            stale_s = now - attachment.last_heartbeat_mono
            attachment.last_heartbeat_mono = now
            attachment.last_heartbeat_at = self._wall()
            attachment.stale_logged = False
        if was_stale:
            logger.info(
                "weights_attachment_revived",
                role=identity.role,
                label=identity.label,
                attachment_ref=attachment.ref,
                stale_s=round(stale_s, 3),
            )
        return {
            "attachment_state": "active",
            "timeout_s": self._timeout_s(),
            "interval_hint_s": float(_settings().WEIGHTS_HEARTBEAT_INTERVAL_HINT_S),
            "weights_state": self._barrier().snapshot().weights_state,
        }

    async def clear_stale(self, identity: WeightsIdentity) -> dict[str, Any]:
        """Clear a stale attachment (FR-31.4.5). Operator only; refuses an active one, because
        the holder may be mid-commit (BRD-02 R-02.34). A clean quiesce it held is resumed;
        `writing` and `torn` are left as they are."""
        if identity.role != "operator":
            raise WeightsForbiddenError.for_roles(identity.role, ("operator",))
        with self._lock:
            attachment = self._attachment
            if attachment is None or attachment.pending:
                raise NotAttachedError(
                    "Nothing is attached to the resident weights; there is nothing to clear.",
                    details={"label": identity.label},
                )
            if not self._is_stale(attachment, self._clock()):
                raise AttachmentNotStaleError(
                    f"The attachment held by '{attachment.identity.label}' is active (its "
                    "heartbeat is current), so it is not cleared: the holder may be mid-commit.",
                    details={
                        "role": attachment.identity.role,
                        "label": attachment.identity.label,
                    },
                )
            # Removed in the same locked section as the stale check, so a heartbeat cannot
            # revive it in between and be cleared while active.
            self._attachment = None
        # A clean quiesce it held is released; `writing` is left for the watchdog and `torn`
        # stays (FR-31.4.5): the holder may only be slow, so clearing never tears by itself.
        self._barrier().on_holder_detached(attachment.digest, tear_writing=False)
        await asyncio.to_thread(self._ipc_collect)
        after = self._barrier().snapshot()
        cleared = {
            "role": attachment.identity.role,
            "label": attachment.identity.label,
            "stale_since": self._stale_since(attachment),
        }
        logger.warning(
            "weights_attachment_cleared",
            operator=identity.label,
            attachment_ref=attachment.ref,
            **cleared,
        )
        return {"cleared": cleared, "weights_state": after.weights_state, "barrier": after.barrier}

    # --- background and status ---------------------------------------------------------------

    def tick(self) -> None:
        """Log `weights_attachment_stale` once per stale episode (FR-31.4.2). Correctness never
        depends on it: every read computes staleness itself."""
        with self._lock:
            attachment = self._attachment
            if attachment is None or attachment.stale_logged:
                return
            if not self._is_stale(attachment, self._clock()):
                return
            attachment.stale_logged = True
        logger.warning(
            "weights_attachment_stale",
            role=attachment.identity.role,
            label=attachment.identity.label,
            attachment_ref=attachment.ref,
            last_heartbeat_at=(
                attachment.last_heartbeat_at.isoformat()
                if attachment.last_heartbeat_at is not None
                else None
            ),
        )

    def status(self) -> "WeightsStatus":
        """The status block (FR-31.16.1): never the attachment ID, never a token (FR-31.16.2).

        Takes no slot. The model forbids extra fields, so nothing beyond the contract can leak
        through it.
        """
        from millm.api.schemas.weights import WeightsStatus

        settings = _settings()
        snap = self._barrier().snapshot()
        with self._lock:
            attachment = self._attachment
            view: Optional[dict[str, Any]] = None
            if attachment is not None:
                stale = self._is_stale(attachment, self._clock())
                view = {
                    "state": "stale" if stale else "active",
                    "role": attachment.identity.role,
                    "label": attachment.identity.label,
                    "attached_at": attachment.attached_at.isoformat(),
                    "last_heartbeat_at": (
                        attachment.last_heartbeat_at.isoformat()
                        if attachment.last_heartbeat_at is not None
                        else None
                    ),
                    "stale_since": self._stale_since(attachment) if stale else None,
                }
        loaded = self._loaded_state().current
        resident: Optional[dict[str, Any]] = None
        if loaded is not None:
            from millm.services.resident_identity import base_served_name, identity_of

            identity = identity_of(loaded)
            resident = {
                "model_id": loaded.model_id,
                "name": base_served_name(loaded),
                "manifest_hash": identity.manifest_sha256 if identity is not None else None,
            }
        commit: Optional[dict[str, Any]] = None
        if snap.commit_began_at is not None:
            commit = {
                "began_at": snap.commit_began_at,
                "open_ms": snap.commit_open_ms,
                "watchdog_ms": int(settings.WEIGHTS_COMMIT_WATCHDOG_MS),
                "overran": snap.commit_overran,
            }
        held_requests: Optional[int] = None
        if snap.barrier == "quiesced":
            queue = self._inference().request_queue
            held_requests = max(int(queue.pending_count) - int(queue.holding_count), 0)
        try:
            pause_unavailable = self._barrier().unservable_reason(self._inference())
        except Exception:  # status never fails over this read
            logger.warning("weights_status_pause_check_failed", exc_info=True)
            pause_unavailable = None
        return WeightsStatus(
            attached=attachment is not None,
            attachment=view,
            barrier=snap.barrier,
            quiesced_at=snap.quiesced_at,
            held_requests=held_requests,
            weights_state=snap.weights_state,
            resident_model=resident,
            # Steps count commits on the RESIDENT parameter set; with nothing resident there is
            # nothing to count, so null (not recorded), never 0.
            steps_since_load=snap.steps_since_load if loaded is not None else None,
            last_save=snap.last_save,
            commit=commit,
            commit_overran=snap.commit_overran,
            last_hold_timeout_at=snap.last_hold_timeout_at,
            last_overrun=snap.last_overrun,
            pause_unavailable=pause_unavailable,
            adapter=_adapter_status(),
            cuda_memory=_cuda_memory(),
            exports_since_start=self._exports_since_start,
        )


def _cuda_memory() -> list[Any] | None:
    """What the CUDA caching allocator of THIS process reports per card (finding 2): allocated
    and reserved MiB. With nothing resident, allocated memory well above zero is memory a
    consumer never released (handles never rebuilt, or a trainer that died or exited holding
    them); only a restart returns it. None when CUDA is not available or cannot be read."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return [
            {
                "device_index": i,
                "allocated_mb": int(torch.cuda.memory_allocated(i) // (1024 * 1024)),
                "reserved_mb": int(torch.cuda.memory_reserved(i) // (1024 * 1024)),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception:  # status never fails over this read
        logger.warning("weights_status_cuda_memory_failed", exc_info=True)
        return None


def _adapter_status() -> Optional["AdapterStatus"]:
    """033's adapter block for the 031 status (addendum D-28). NEVER raises: the weights status
    must answer when the adapter block cannot be read, so a failure is logged and reported as
    null (not recorded) rather than failing — or faking — the whole status."""
    try:
        from millm.api.schemas.weights_adapters import AdapterStatus
        from millm.services.adapter_service import get_adapter_service

        return AdapterStatus.model_validate(get_adapter_service().status())
    except Exception as e:  # noqa: BLE001 - an optional block must never fail the status
        logger.warning("weights_status_adapter_unavailable", error_type=type(e).__name__)
        return None


_service: Optional[WeightService] = None
_service_lock = threading.Lock()


def get_weight_service() -> WeightService:
    """The process-wide weight service, created on first use."""
    global _service
    with _service_lock:
        if _service is None:
            _service = WeightService()
        return _service


def set_weight_service(service: Optional[WeightService]) -> None:
    """Replace the process-wide service (tests, startup reset); None drops it."""
    global _service
    with _service_lock:
        _service = service


def reset_weights_on_startup() -> None:
    """A restart begins `clean`, `serving`, with no attachment and no counters (PADR v1.6 §10;
    031 FTID §7.7). Called from `lifespan` right after `clear_leases_on_startup()`. Never raises:
    an optional feature must not stop the server starting.

    The registry is process memory, so it is empty after a real restart anyway; the call makes
    the reconciliation explicit and testable, as `clear_leases_on_startup` does for leases.
    """
    try:
        get_weight_barrier().reset("startup")
        set_weight_service(None)
        logger.info("weights_reset_on_startup")
    except Exception as e:  # noqa: BLE001 - never stop the server for an optional feature
        logger.error("weights_reset_on_startup_failed", error_type=type(e).__name__, error=str(e))


async def weights_tick_loop(interval_s: float = TICK_INTERVAL_S) -> None:
    """Call `WeightService.tick()` every `interval_s` until cancelled (started and cancelled by
    `lifespan`). The tick only logs a stale attachment once (FR-31.4.2); a failure is logged
    and the loop goes on."""
    while True:
        try:
            get_weight_service().tick()
        except Exception as e:  # noqa: BLE001 - a log line must never end the loop
            logger.warning("weights_tick_failed", error_type=type(e).__name__, error=str(e))
        await asyncio.sleep(interval_s)
