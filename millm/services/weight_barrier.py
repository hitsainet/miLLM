"""The step barrier: who may pause serving, and what state the resident weights are in.

Shared by Features 31, 32 and 33 (contract: addendum §2.1 and §7). Three kinds of hold take
THE admission slot:

- a trainer's **quiesce** (`quiesce` / `resume` / `commit_begin` / `commit_end`, Feature 31),
  owned by the holder (`hold_owner == "holder"`);
- a **serving pause** (`serving_pause`, 032's save snapshot and 033's adapter load and unload),
  which never sets `writing`;
- an **internal write** (`internal_write`, 032's load), which can recover a `torn` set.

How the barrier pauses serving. It takes the ONE admission slot through
`InferenceService.hold_admission()`, from a barrier-owned HOLDER TASK, and keeps it until the
hold ends. Load-bearing consequences:

- **The slot belongs to the barrier, not the caller.** `_admit` is re-entrant only for the task
  that holds the slot, and the holder task is not the caller's task, so a caller that enters
  `_admit()` (or `run_model_work`) inside a pause waits for a slot the barrier will not release
  until the block exits: a deadlock, bounded by nothing. Do not admit inside a pause.
- **It is correct only with one slot and no continuous batching.** With two slots, holding one
  stops nothing; the continuous batching manager generates without a slot at all. Both are
  refused (`BARRIER_UNAVAILABLE`, reasons `concurrency` and `cbm`; addendum D-04, D-13) rather
  than asserted, because 033 pauses with nothing attached and an assertion would be a 500.
- **Holding the slot proves the last request returned, not that its kernels finished** (031
  review D7), so every hold synchronizes the resident model's cards before it is reported held.

HOLD OWNERSHIP (031 review D5, X6, D10). `hold_owner` is `"holder"` or `"internal:<op>"`;
`episode` increases with every hold, adopted or taken.

| Situation | Effect |
|---|---|
| `resume`, the hold timer, detach | act only on a holder-owned hold |
| an internal op while the holder holds a `clean` quiesce | ADOPTS it: the hold timer is suspended, `commit-begin` and `resume` refuse with `PAUSE_IN_PROGRESS`; on exit ownership returns to the holder and the timer is re-armed in full |
| `internal_write` while the holder's hold is `torn` (the holder possibly gone) | INHERITS it; on success the slot is released (`clean, serving`); on an early failure everything is as it was |
| `internal_write` exits | `clean` only if the state is still ITS OWN `writing` episode, so a concurrent `mark_torn` is never overwritten |
| a second pause or internal write | refused, `PAUSE_IN_PROGRESS` |
| `on_parameter_set_replaced` | a no-op while an internal write runs; otherwise `clean`, steps 0, and a holder-owned hold (a torn set's) is released |

`weights_state` starts `clean`. While `torn`, `admission_refusal()` refuses every new request and
the abort event wakes and refuses every queued one (FR-31.11.3).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Literal, Optional

from millm.core.errors import (
    AlreadyQuiescedError,
    ModelBusyError,
    NoOpenCommitError,
    NotAttachedError,
    NotAttachmentHolderError,
    NotQuiescedError,
    PauseInProgressError,
    WeightsBarrierUnavailableError,
    WeightsNotCleanError,
    WeightsTornError,
)
from millm.core.logging import get_logger

if TYPE_CHECKING:
    from millm.core.weights_auth import WeightsIdentity

logger = get_logger(__name__)

WeightsState = Literal["clean", "writing", "torn"]
BarrierMode = Literal["serving", "quiesced"]

HOLDER = "holder"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class BarrierSnapshot:
    """A consistent read of the barrier, for status, health and the audit line.

    The first ten fields are the contract (addendum §2.1); the rest are additive, for the status
    route (FPRD FR-31.16.1), and default to None (not recorded).
    """

    weights_state: WeightsState
    barrier: BarrierMode
    hold_owner: Optional[str]
    episode: int
    steps_since_load: int
    commit_overran: bool
    last_save: Optional[dict[str, Any]]
    last_hold_timeout_at: Optional[str]
    last_overrun: Optional[dict[str, Any]]
    held_model_id: Optional[int]
    quiesced_at: Optional[str] = None
    hold_deadline: Optional[str] = None
    commit_began_at: Optional[str] = None
    commit_open_ms: Optional[float] = None


@dataclass(frozen=True)
class PauseGuard:
    """What a `serving_pause` block sees once serving is paused (032 reads both counters)."""

    weights_state: WeightsState
    steps_since_load: int
    episode: int


@dataclass(frozen=True)
class QuiesceResult:
    """The quiesce route's body (FPRD FR-31.7.2, FR-31.7.3)."""

    held_requests: int
    held_batch_chunks: int
    weights_state: WeightsState
    quiesced_at: str
    hold_deadline: str
    state_before: WeightsState


@dataclass(frozen=True)
class ResumeResult:
    """`released` is False for the no-op resume while serving (FR-31.8.1)."""

    barrier: BarrierMode
    weights_state: WeightsState
    released: bool
    state_before: WeightsState


@dataclass(frozen=True)
class CommitBeginResult:
    weights_state: WeightsState
    began_at: str
    watchdog_ms: int
    state_before: WeightsState


@dataclass(frozen=True)
class CommitResult:
    """`commit-end`'s body (FR-31.10.2); `overran` for a late commit-end out of `torn`."""

    weights_state: WeightsState
    steps_since_load: int
    commit_ms: float
    overran: bool
    state_before: WeightsState


class WriteHandle:
    """Yielded by `internal_write`. Before `mark_first_write()` a failure leaves the state
    unchanged; after it, a failure marks the weights `torn` (031 G2). 032 calls it immediately
    before its first tensor copy or unload."""

    def __init__(
        self, episode: int, on_first_write: Optional[Callable[["WriteHandle"], None]] = None
    ) -> None:
        self.episode = episode
        self.first_write_marked = False
        self._on_first_write = on_first_write

    def mark_first_write(self) -> None:
        if self.first_write_marked:
            return
        if self._on_first_write is not None:
            self._on_first_write(self)
        self.first_write_marked = True


@dataclass
class _Adopted:
    """A holder-owned hold an internal op took over; restored (or released) on exit."""

    digest: Optional[str]
    identity: Optional["WeightsIdentity"]
    state: WeightsState  # the state when adopted: `clean` (a quiesce) or `torn` (inherited)


def _default_inference() -> Any:
    from millm.api.dependencies import get_inference_service

    return get_inference_service()


def _current_model_id() -> Optional[int]:
    from millm.ml.model_loader import LoadedModelState

    return LoadedModelState().loaded_model_id


class WeightBarrier:
    """Process-wide barrier state. One instance (`get_weight_barrier()`); tests build their own.

    Every check-and-transition runs under one `threading.Lock` with no `await` inside (031 FTDD
    §7.1). Timers use `loop.call_later` and re-check generation, state and the injected clock
    when they fire, so a stale timer is a no-op and a fake clock decides expiry in tests.

    Args:
        inference: returns the inference service the barrier pauses. It must offer
            `hold_admission()`, `cbm_enabled()`, `invalidate_request_caches()` and
            `request_queue.max_concurrent`.
        clock: monotonic seconds, for every timer decision.
        wall: the displayed time (UTC).
    """

    def __init__(
        self,
        *,
        inference: Callable[[], Any] = _default_inference,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._inference = inference
        self._clock = clock
        self._wall = wall
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._weights_state: WeightsState = "clean"
        self._barrier: BarrierMode = "serving"
        self._hold_owner: Optional[str] = None
        self._episode = 0
        self._steps_since_load = 0
        self._commit_overran = False
        self._last_save: Optional[dict[str, Any]] = None
        self._last_hold_timeout_at: Optional[str] = None
        self._last_overrun: Optional[dict[str, Any]] = None
        self._held_model_id: Optional[int] = None
        self._internal_write_active = False
        # The holder (a trainer's quiesce).
        self._quiescing = False
        self._holder_digest: Optional[str] = None
        self._holder_identity: Optional["WeightsIdentity"] = None
        self._adopted: Optional[_Adopted] = None
        self._quiesced_at: Optional[datetime] = None
        # The hold timer.
        self._hold_generation = 0
        self._hold_deadline_mono: Optional[float] = None
        self._hold_deadline_at: Optional[datetime] = None
        self._hold_handle: Optional[asyncio.TimerHandle] = None
        # The commit and its watchdog.
        self._commit_owner_digest: Optional[str] = None
        # Who began the open commit. Kept beside the digest because the holder's identity is
        # cleared when it detaches or is cleared, while a torn line must still say who was
        # writing (R-01.22, FR-31.11.2; R1 RX F3): a watchdog after clear-stale fires later.
        self._commit_owner_identity: Optional["WeightsIdentity"] = None
        self._commit_began_mono: Optional[float] = None
        self._commit_began_at: Optional[datetime] = None
        self._watchdog_generation = 0
        self._watchdog_deadline_mono: Optional[float] = None
        self._watchdog_handle: Optional[asyncio.TimerHandle] = None
        self._writing_episode: Optional[int] = None
        # The abort event: set exactly while new admissions are refused (FTID §7.3).
        self._abort_flag = False
        self._abort_event: Optional[asyncio.Event] = None
        self._abort_loop: Optional[asyncio.AbstractEventLoop] = None
        # Set while the barrier waits for the slot: new arrivals park on it (031 review D9).
        self._gate: Optional[asyncio.Event] = None
        # The holder task and its two events (slot held; release requested).
        self._holder_task: Optional[asyncio.Task[None]] = None
        self._slot_held: Optional[asyncio.Event] = None
        self._release: Optional[asyncio.Event] = None

    # --- reads -----------------------------------------------------------------------------

    def snapshot(self) -> BarrierSnapshot:
        with self._lock:
            open_ms = None
            if self._commit_began_mono is not None and self._weights_state != "clean":
                open_ms = round((self._clock() - self._commit_began_mono) * 1000.0, 1)
            return BarrierSnapshot(
                weights_state=self._weights_state,
                barrier=self._barrier,
                hold_owner=self._hold_owner,
                episode=self._episode,
                steps_since_load=self._steps_since_load,
                commit_overran=self._commit_overran,
                last_save=dict(self._last_save) if self._last_save is not None else None,
                last_hold_timeout_at=self._last_hold_timeout_at,
                last_overrun=dict(self._last_overrun) if self._last_overrun is not None else None,
                held_model_id=self._held_model_id,
                quiesced_at=_iso(self._quiesced_at),
                hold_deadline=_iso(self._hold_deadline_at),
                commit_began_at=_iso(self._commit_began_at),
                commit_open_ms=open_ms,
            )

    def holds_slot(self) -> bool:
        """Whether the barrier's holder task holds the admission slot, whoever owns the hold
        (031 review X6): a holder's quiesce, a pause, an internal write, or a torn set's hold
        whose holder is gone. `unload_model(under_barrier=True)` skips its drain on it."""
        task, held = self._holder_task, self._slot_held
        return task is not None and not task.done() and held is not None and held.is_set()

    def is_holder_task(self, task: Optional["asyncio.Task[Any]"]) -> bool:
        """Whether `task` is the barrier's own holder task (admission lets it past the torn
        refusal and the quiescing gate, and keeps its long hold out of the queue statistics)."""
        return task is not None and task is self._holder_task

    def admission_refusal(self) -> Optional[WeightsTornError]:
        """The refusal for new work while the resident weights are not servable; None otherwise.

        Lock-free read, called by `InferenceService._admit` for every request.
        """
        if self._weights_state == "torn" or self._abort_flag:
            return WeightsTornError(
                "The resident weights were left mid-write and are not served; retry once a "
                "late commit-end or a weights load restores them.",
                details={"weights_state": self._weights_state},
            )
        return None

    @property
    def abort_event(self) -> asyncio.Event:
        """The event `RequestQueue.acquire` races its semaphore against: set while `torn`.

        Bound to the running loop; rebuilt (carrying the flag) if the loop changed.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            event = self._abort_event
            if event is None or self._abort_loop is not loop:
                event = asyncio.Event()
                if self._abort_flag:
                    event.set()
                self._abort_event, self._abort_loop = event, loop
            return event

    def admission_gate(self, task: Optional["asyncio.Task[Any]"]) -> Optional[asyncio.Event]:
        """The event a new arrival waits on while the barrier is taking the slot (031 review D9):
        parked outside the queue, it cannot overtake the barrier's interactive waiter. None when
        nothing is being taken, or for the holder task itself."""
        gate = self._gate
        if gate is None or gate.is_set() or self.is_holder_task(task):
            return None
        return gate

    def require_clean(self, operation: str) -> None:
        """Raise `WEIGHTS_NOT_CLEAN` unless the weights are `clean` (a save's refusal, R-01.11)."""
        with self._lock:
            state = self._weights_state
        if state != "clean":
            raise WeightsNotCleanError(
                f"{operation} refused: the resident weights are {state}, not clean.",
                details={"operation": operation, "weights_state": state},
            )

    # --- counters 032 drives ---------------------------------------------------------------

    def reset_steps_since_load(self, reason: str = "parameter_set_replaced") -> None:
        with self._lock:
            before = self._steps_since_load
            self._steps_since_load = 0
        logger.info("weights_steps_reset", reason=reason, steps_before=before)

    def record_save(self, checkpoint_sha256: str, path: str, at: datetime) -> None:
        with self._lock:
            self._last_save = {
                "checkpoint_sha256": checkpoint_sha256,
                "path": path,
                "at": at.isoformat(),
            }

    def on_parameter_set_replaced(self, reason: str) -> None:
        """A Feature-1 load or unload replaced the parameter set: clean, steps 0, abort cleared,
        and a holder-owned hold released (a torn set's hold would otherwise leak the slot and
        every request would wait forever after the reload, 031 review D10).

        A no-op while an `internal_write` is active: that write performs its own transition
        (031 review X5). Never releases a pause an internal op owns.
        """
        with self._lock:
            if self._internal_write_active:
                return
            before = self._weights_state
            self._weights_state = "clean"
            self._steps_since_load = 0
            self._commit_overran = False
            self._clear_commit_locked()
            self._set_abort_locked(False)
            released = None
            if self._hold_owner == HOLDER and not self._quiescing:
                released = self._release_holder_locked()
        if released is not None:
            self._signal_release(*released)
        logger.info(
            "weights_parameter_set_replaced",
            reason=reason,
            weights_state_before=before,
            released_hold=released is not None,
        )

    def reset(self, reason: str) -> None:
        """Back to `clean, serving` with no hold and no counters (startup and tests)."""
        with self._lock:
            released = self._release_holder_locked() if self._holder_task is not None else None
            self._weights_state = "clean"
            self._barrier = "serving"
            self._hold_owner = None
            self._steps_since_load = 0
            self._commit_overran = False
            self._last_save = None
            self._last_hold_timeout_at = None
            self._last_overrun = None
            self._held_model_id = None
            self._adopted = None
            self._clear_commit_locked()
            self._set_abort_locked(False)
        if released is not None:
            self._signal_release(*released)
        logger.info("weights_barrier_reset", reason=reason)

    # --- the holder's side (Feature 31) ----------------------------------------------------

    async def quiesce(self, owner_digest: str, identity: "WeightsIdentity") -> QuiesceResult:
        """The holder's quiesce (031 FTID §7.1): drain in-flight work, take the slot, hold it.

        The attachment-holder check is the caller's (`WeightService`); the barrier records
        `owner_digest` so that `commit_begin`, `commit_end` and `resume` can match it.

        Raises:
            WeightsBarrierUnavailableError: continuous batching on, or more than one slot.
            WeightsNotCleanError: the weights are `writing` or `torn`.
            AlreadyQuiescedError: the holder already holds (or is taking) the slot.
            PauseInProgressError: a pause or internal write holds the barrier.
            ModelBusyError: in-flight work did not drain (`reason: drain_timeout`), or the model
                is unloading.
            QueueFullError: the request queue is full.
        """
        inference = self._inference()
        self._refuse_unservable(inference, "quiesce")
        with self._lock:
            owner = self._hold_owner
            if owner is not None and owner.startswith("internal:"):
                raise PauseInProgressError.for_operation(owner.split(":", 1)[1], "quiesce")
            if self._weights_state != "clean":
                raise WeightsNotCleanError(
                    f"quiesce refused: the resident weights are {self._weights_state}.",
                    details={"operation": "quiesce", "weights_state": self._weights_state},
                )
            if owner == HOLDER:
                raise AlreadyQuiescedError(
                    "Serving is already quiesced for this attachment.",
                    details={"quiesced_at": _iso(self._quiesced_at)},
                )
            self._episode += 1
            episode = self._episode
            self._hold_owner = HOLDER
            self._quiescing = True
            self._holder_digest = owner_digest
            self._holder_identity = identity
        try:
            await self._take_slot(inference, episode, "quiesce")
            with self._lock:
                self._barrier = "quiesced"
                self._held_model_id = _current_model_id()
            await self._after_hold(inference)
        except BaseException:
            with self._lock:
                self._quiescing = False
                self._hold_owner = None
                self._barrier = "serving"
                self._held_model_id = None
                self._holder_digest = self._holder_identity = None
            await self._release_slot()
            raise
        gone = None
        with self._lock:
            self._quiescing = False
            if self._holder_digest != owner_digest:
                # The holder detached or was cleared while the drain ran (031 review D14):
                # `on_holder_detached` could not release a hold still being taken. A hold with
                # no holder would refuse every request until the hold timer freed it.
                gone = self._release_holder_locked()
            else:
                self._quiesced_at = self._wall()
                self._arm_hold_timer_locked()
            quiesced_at, deadline = self._quiesced_at, self._hold_deadline_at
        if gone is not None:
            self._signal_release(*gone)
            logger.info("weights_quiesce_holder_gone", episode=episode,
                        attachment_ref=owner_digest[:8])
            raise NotAttachedError(
                "The attachment was detached or cleared while quiesce waited; serving resumed.",
                details={"operation": "quiesce"},
            )
        queue = getattr(inference, "request_queue", None)
        held_requests = max(
            int(getattr(queue, "pending_count", 0)) - int(getattr(queue, "holding_count", 0)), 0
        )
        held_chunks = int(getattr(queue, "background_waiting_count", 0))
        logger.info(
            "weights_quiesced",
            episode=episode,
            attachment_ref=owner_digest[:8],
            held_requests=held_requests,
            held_batch_chunks=held_chunks,
        )
        assert quiesced_at is not None and deadline is not None
        return QuiesceResult(
            held_requests=held_requests,
            held_batch_chunks=held_chunks,
            weights_state="clean",
            quiesced_at=quiesced_at.isoformat(),
            hold_deadline=deadline.isoformat(),
            state_before="clean",
        )

    def resume(self, owner_digest: str) -> ResumeResult:
        """Release the holder's quiesce (FR-31.8). A no-op while serving (FR-31.8.1).

        Raises:
            PauseInProgressError: an internal op adopted the holder's hold; it hands it back.
            NotQuiescedError: the holder's quiesce is still draining in-flight work (addendum
                D-39). A no-op success here would be followed by the quiesce completing and
                holding for the full hold timeout while the holder believes it resumed.
            NotAttachmentHolderError: `owner_digest` is not the attachment that quiesced.
            WeightsNotCleanError: a commit is open or the weights are torn.
        """
        with self._lock:
            state = self._weights_state
            adopted = self._adopted
            if adopted is not None and self._hold_owner is not None:
                raise PauseInProgressError.for_operation(
                    self._hold_owner.split(":", 1)[1], "resume"
                )
            if self._hold_owner == HOLDER and self._quiescing:
                raise NotQuiescedError(
                    "resume refused: quiesce is still draining in-flight work; resume once "
                    "quiesce has returned.",
                    details={
                        "operation": "resume",
                        "barrier": self._barrier,
                        "reason": "quiesce_in_progress",
                    },
                )
            if self._hold_owner != HOLDER:
                return ResumeResult(self._barrier, state, False, state)
            if state == "torn":
                # D-44 (as D-43 for commit-begin): a torn set refuses before the owner check, so
                # a holder re-attached after a tear learns the real state, not 403.
                raise WeightsNotCleanError(
                    "resume refused: the resident weights are torn.",
                    details={"operation": "resume", "weights_state": state},
                )
            if self._holder_digest != owner_digest:
                raise NotAttachmentHolderError(
                    "Only the attachment that quiesced may resume.",
                    details={"operation": "resume"},
                )
            if state != "clean":
                raise WeightsNotCleanError(
                    f"resume refused: the resident weights are {state}.",
                    details={"operation": "resume", "weights_state": state},
                )
            released = self._release_holder_locked()
        self._signal_release(*released)
        logger.info("weights_resumed", attachment_ref=owner_digest[:8])
        return ResumeResult("serving", "clean", True, state)

    def commit_begin(self, owner_digest: str) -> CommitBeginResult:
        """Open a commit (FR-31.10, FTID §7.2): `writing`, hold timer suspended, watchdog armed.

        Raises:
            PauseInProgressError: a pause or internal write holds the barrier (031 3A.1),
                including one that adopted this holder's own quiesce.
            NotQuiescedError: the holder does not hold the slot.
            NotAttachmentHolderError: `owner_digest` is not the attachment that quiesced.
            WeightsNotCleanError: a commit is open or the weights are torn.
        """
        from millm.core.config import settings

        with self._lock:
            owner = self._hold_owner
            if owner is not None and owner.startswith("internal:"):
                raise PauseInProgressError.for_operation(owner.split(":", 1)[1], "commit_begin")
            if owner != HOLDER or self._quiescing or self._barrier != "quiesced":
                raise NotQuiescedError(
                    "commit-begin needs serving quiesced by this attachment.",
                    details={"operation": "commit_begin", "barrier": self._barrier},
                )
            # The state BEFORE the owner check (addendum D-43, FR-31.10.1): after a watchdog tear
            # the slot stays held under the departed holder's quiesce, so a re-attached holder
            # (a new digest) would otherwise be told 403 NOT_ATTACHMENT_HOLDER for a set whose
            # real answer is that it is not clean. The caller is already the attachment's holder
            # (`WeightService.require_holder`); the digest check guards only a clean quiesce.
            state = self._weights_state
            if state != "clean":
                raise WeightsNotCleanError(
                    f"commit-begin refused: the resident weights are {state}.",
                    details={"operation": "commit_begin", "weights_state": state},
                )
            if self._holder_digest != owner_digest:
                raise NotAttachmentHolderError(
                    "Only the attachment that quiesced may begin a commit.",
                    details={"operation": "commit_begin"},
                )
            self._weights_state = "writing"
            self._writing_episode = None
            self._commit_overran = False
            self._commit_owner_digest = owner_digest
            self._commit_owner_identity = self._holder_identity
            self._commit_began_mono = self._clock()
            self._commit_began_at = self._wall()
            self._suspend_hold_timer_locked()
            watchdog_ms = int(settings.WEIGHTS_COMMIT_WATCHDOG_MS)
            self._arm_watchdog_locked(watchdog_ms / 1000.0)
            began_at = self._commit_began_at
        logger.info("weights_commit_begin", attachment_ref=owner_digest[:8], watchdog_ms=watchdog_ms)
        return CommitBeginResult("writing", began_at.isoformat(), watchdog_ms, state)

    def commit_end(self, owner_digest: str) -> CommitResult:
        """Close the commit (FR-31.10.2), or recover a torn set by a late commit-end from the
        commit's owner (FR-31.12.1): `clean`, steps + 1, abort cleared, hold timer re-armed.

        Raises:
            PauseInProgressError: an internal op holds the barrier.
            NoOpenCommitError: no commit is open (or the torn set is not a commit's).
            NotAttachmentHolderError: `owner_digest` did not begin this commit (FR-31.12.2).
        """
        with self._lock:
            owner = self._hold_owner
            if owner is not None and owner.startswith("internal:"):
                raise PauseInProgressError.for_operation(owner.split(":", 1)[1], "commit_end")
            state = self._weights_state
            if state == "clean" or self._commit_owner_digest is None:
                raise NoOpenCommitError(
                    "No commit is open.",
                    details={"operation": "commit_end", "weights_state": state},
                )
            if self._commit_owner_digest != owner_digest:
                raise NotAttachmentHolderError(
                    "Only the attachment that began the commit may end it.",
                    details={"operation": "commit_end"},
                )
            overran = state == "torn"
            assert self._commit_began_mono is not None
            commit_ms = round((self._clock() - self._commit_began_mono) * 1000.0, 3)
            self._weights_state = "clean"
            self._set_abort_locked(False)
            self._steps_since_load += 1
            self._cancel_watchdog_locked()
            self._clear_commit_locked()
            self._commit_overran = overran
            if overran:
                self._last_overrun = {"at": self._wall().isoformat(), "commit_ms": commit_ms}
            if self._hold_owner == HOLDER and self.holds_slot():
                self._arm_hold_timer_locked()
            steps = self._steps_since_load
            identity = self._holder_identity
        if overran:
            from millm.core.config import settings

            logger.warning(
                "weights_commit_overran",
                role=identity.role if identity is not None else None,
                label=identity.label if identity is not None else None,
                attachment_ref=owner_digest[:8],
                commit_ms=commit_ms,
                watchdog_ms=int(settings.WEIGHTS_COMMIT_WATCHDOG_MS),
            )
        logger.info(
            "weights_commit_end",
            attachment_ref=owner_digest[:8],
            commit_ms=commit_ms,
            overran=overran,
            steps_since_load=steps,
        )
        return CommitResult("clean", steps, commit_ms, overran, state)

    def on_holder_detached(self, owner_digest: str, *, tear_writing: bool = True) -> WeightsState:
        """The holder detached or was cleared (031 FPRD D16): a clean quiesce is released, an
        open commit becomes `torn` (its slot kept, as the watchdog's), and an adopted hold is
        released when the internal op ends instead of being handed back. Returns the state.

        `tear_writing=False` is clear-stale's form (FR-31.4.5): an open commit is left `writing`
        for the watchdog to tear, because the operator cleared a holder that may only be slow.
        """
        released = None
        torn = None
        with self._lock:
            adopted = self._adopted
            if adopted is not None and adopted.digest == owner_digest:
                adopted.digest = adopted.identity = None
            elif self._hold_owner == HOLDER and self._holder_digest == owner_digest:
                if self._weights_state == "writing":
                    if tear_writing:
                        torn = self._mark_torn_locked()
                elif self._weights_state == "clean" and not self._quiescing:
                    released = self._release_holder_locked()
                self._holder_digest = self._holder_identity = None
            state = self._weights_state
        if released is not None:
            self._signal_release(*released)
        if torn is not None:
            self._log_torn("detach", torn, attachment_ref=owner_digest[:8])
        return state

    def mark_torn(
        self, reason: str, *, writer: Optional["WeightsIdentity"] = None, **fields: Any
    ) -> None:
        """Mark the resident weights `torn` (watchdog, detach while writing, address moved, a
        failed internal write): waiters are woken and refused, timers stop, the slot is kept.

        `writer` names who was writing when the caller knows better than the barrier: a failed
        internal write was miLLM's own write for the caller of `internal_write`, not the
        holder's, even when it adopted the holder's hold."""
        with self._lock:
            facts = self._mark_torn_locked()
        if writer is not None:
            facts = (facts[0], writer, facts[2])
        self._log_torn(reason, facts, **fields)

    def _mark_torn_locked(self) -> tuple[WeightsState, Any, Optional[int]]:
        """The torn transition itself (lock held). A caller that DECIDED to tear under the lock
        tears in the same locked section: released in between, a commit-end or a parameter-set
        replacement from another thread could land first and be overwritten with `torn`."""
        before = self._weights_state
        self._weights_state = "torn"
        self._set_abort_locked(True)
        self._cancel_watchdog_locked()
        self._suspend_hold_timer_locked()
        # The holder's identity, else the commit's owner: a watchdog after clear-stale (or a
        # detach) finds the holder already gone, and the line must still say who was writing.
        identity = self._holder_identity or self._commit_owner_identity
        return before, identity, self._held_model_id

    @staticmethod
    def _log_torn(reason: str, facts: tuple[Any, Any, Optional[int]], **fields: Any) -> None:
        before, identity, model_id = facts
        logger.error(
            "weights_torn",
            reason=reason,
            weights_state_before=before,
            role=identity.role if identity is not None else None,
            label=identity.label if identity is not None else None,
            resident_model_id=model_id,
            **fields,
        )

    # --- timers ----------------------------------------------------------------------------

    def _call_later(self, delay: float, callback: Callable[[int], None], generation: int) -> Any:
        """Schedule `callback(generation)` on the barrier's loop. Off the loop (a worker
        thread) the timer is handed to the loop and not kept: its generation voids it."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            other = self._loop
            if other is not None and not other.is_closed():
                other.call_soon_threadsafe(other.call_later, delay, callback, generation)
            return None
        return loop.call_later(delay, callback, generation)

    def _arm_hold_timer_locked(self) -> None:
        from millm.core.config import settings

        timeout = float(settings.WEIGHTS_HOLD_TIMEOUT_S)
        self._hold_generation += 1
        if self._hold_handle is not None:
            self._hold_handle.cancel()
        self._hold_deadline_mono = self._clock() + timeout
        self._hold_deadline_at = self._wall() + timedelta(seconds=timeout)
        self._hold_handle = self._call_later(timeout, self._on_hold_timer, self._hold_generation)

    def _suspend_hold_timer_locked(self) -> None:
        self._hold_generation += 1
        if self._hold_handle is not None:
            self._hold_handle.cancel()
        self._hold_handle = None
        self._hold_deadline_mono = None
        self._hold_deadline_at = None

    def _on_hold_timer(self, generation: int) -> None:
        """Self-resume, only in `clean, quiesced` with the holder owning the hold (FR-31.9)."""
        released = None
        with self._lock:
            if generation != self._hold_generation or self._hold_deadline_mono is None:
                return
            if (
                self._hold_owner != HOLDER
                or self._quiescing
                or self._weights_state != "clean"
                or self._barrier != "quiesced"
            ):
                return
            remaining = self._hold_deadline_mono - self._clock()
            if remaining > 0:
                self._hold_handle = self._call_later(remaining, self._on_hold_timer, generation)
                return
            identity, digest = self._holder_identity, self._holder_digest
            self._last_hold_timeout_at = self._wall().isoformat()
            released = self._release_holder_locked()
        self._signal_release(*released)
        logger.warning(
            "weights_hold_timeout",
            role=identity.role if identity is not None else None,
            label=identity.label if identity is not None else None,
            attachment_ref=digest[:8] if digest else None,
        )

    def _arm_watchdog_locked(self, delay: float) -> None:
        self._watchdog_generation += 1
        self._watchdog_deadline_mono = self._clock() + delay
        self._watchdog_handle = self._call_later(
            delay, self._on_watchdog, self._watchdog_generation
        )

    def _cancel_watchdog_locked(self) -> None:
        self._watchdog_generation += 1
        if self._watchdog_handle is not None:
            self._watchdog_handle.cancel()
        self._watchdog_handle = None
        self._watchdog_deadline_mono = None

    def _on_watchdog(self, generation: int) -> None:
        """A commit that did not end in time: `torn` (FR-31.11), slot kept, waiters refused."""
        with self._lock:
            if generation != self._watchdog_generation or self._watchdog_deadline_mono is None:
                return
            if self._weights_state != "writing" or self._commit_owner_digest is None:
                return
            remaining = self._watchdog_deadline_mono - self._clock()
            if remaining > 0:
                self._watchdog_handle = self._call_later(remaining, self._on_watchdog, generation)
                return
            assert self._commit_began_mono is not None
            elapsed_ms = round((self._clock() - self._commit_began_mono) * 1000.0, 1)
            digest = self._commit_owner_digest
            self._commit_overran = True
            facts = self._mark_torn_locked()
        self._log_torn("watchdog", facts, attachment_ref=digest[:8], elapsed_ms=elapsed_ms)

    # --- slot plumbing ---------------------------------------------------------------------

    @staticmethod
    def unservable_reason(inference: Any) -> Optional[str]:
        """Why the barrier cannot pause serving right now (`cbm` or `concurrency`), or None.

        The one source for both the refusal (`_refuse_unservable`) and the status field
        `pause_unavailable`, so a client can learn before submitting what a quiesce, a
        weights load or an adapter change would be refused for."""
        if inference.cbm_enabled():
            return "cbm"
        if int(inference.request_queue.max_concurrent) != 1:
            return "concurrency"
        return None

    def _refuse_unservable(self, inference: Any, operation: str) -> None:
        reason = self.unservable_reason(inference)
        if reason is not None:
            raise WeightsBarrierUnavailableError.for_reason(reason, operation)

    def _set_abort_locked(self, flag: bool) -> None:
        self._abort_flag = flag
        event, loop = self._abort_event, self._abort_loop
        if event is None or loop is None:
            return
        action = event.set if flag else event.clear
        try:
            running: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            action()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(action)

    def _clear_commit_locked(self) -> None:
        self._cancel_watchdog_locked()
        self._commit_owner_digest = None
        self._commit_owner_identity = None
        self._commit_began_mono = None
        self._commit_began_at = None
        self._writing_episode = None

    def _release_holder_locked(self) -> tuple[Any, Any, Any, Any]:
        """End the current hold's bookkeeping; the caller then calls `_signal_release`."""
        self._suspend_hold_timer_locked()
        self._barrier = "serving"
        self._hold_owner = None
        self._held_model_id = None
        self._quiesced_at = None
        self._holder_digest = self._holder_identity = None
        self._adopted = None
        released = (self._holder_task, self._slot_held, self._release, self._loop)
        self._holder_task = self._slot_held = self._release = None
        return released

    @staticmethod
    def _signal_release(task: Any, held: Any, release: Any, loop: Any) -> None:
        """Let the holder task leave `hold_admission()` (thread-safe: a load completes in a
        worker thread). A task that never held the slot is cancelled instead."""
        if task is None:
            return

        def signal() -> None:
            if held is not None and held.is_set() and release is not None:
                release.set()
            else:
                task.cancel()

        try:
            running: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is None or running is loop:
            signal()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(signal)

    async def _take_slot(self, inference: Any, episode: int, operation: str) -> None:
        """Start the holder task and wait until it holds the slot.

        While waiting, new arrivals park on the gate (031 review D9). Refuses with `MODEL_BUSY`
        (reason `drain_timeout`) when in-flight work does not drain within
        `WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S`; re-raises what `_admit` refused with (`QUEUE_FULL`,
        `MODEL_BUSY` during an unload) having held nothing.
        """
        from millm.core.config import settings

        self._loop = asyncio.get_running_loop()
        held, release = asyncio.Event(), asyncio.Event()
        gate = asyncio.Event()

        async def hold() -> None:
            async with inference.hold_admission():
                held.set()
                await release.wait()

        self._gate = gate
        task = asyncio.create_task(hold(), name=f"weights-barrier-hold-{episode}")
        self._holder_task, self._slot_held, self._release = task, held, release
        waiter = asyncio.create_task(held.wait())
        timeout = float(settings.WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S)
        try:
            await asyncio.wait({task, waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
            gate.set()
            if self._gate is gate:
                self._gate = None
        if held.is_set() and not task.done():
            return
        if task.done():
            self._forget_holder(task)
            task.result()  # the refusal `_admit` raised
            raise RuntimeError("the barrier's holder task ended without holding the slot")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._forget_holder(task)
        raise ModelBusyError(
            f"{operation} refused: in-flight requests did not finish within {timeout:g} s.",
            details={"reason": "drain_timeout", "operation": operation, "timeout_s": timeout},
        )

    def _forget_holder(self, task: Any) -> None:
        if self._holder_task is task:
            self._holder_task = self._slot_held = self._release = None

    async def _release_slot(self) -> None:
        task, held, release = self._holder_task, self._slot_held, self._release
        if task is None:
            return
        self._signal_release(task, held, release, None)
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._forget_holder(task)

    async def _after_hold(self, inference: Any) -> None:
        """Once the slot is held or adopted: drop cross-request caches (FR-31.13, 031 review
        X7), then wait for the resident model's cards (031 review D7)."""
        inference.invalidate_request_caches()
        await asyncio.to_thread(_synchronize_resident_devices)

    def _begin_internal_locked(self, operation: str, *, may_adopt_torn: bool) -> tuple[int, bool]:
        """Reserve the barrier for an internal op (lock held). Returns (episode, adopted)."""
        owner = self._hold_owner
        if owner is not None and owner.startswith("internal:"):
            raise PauseInProgressError.for_operation(owner.split(":", 1)[1], operation)
        if self._quiescing:
            raise PauseInProgressError.for_operation("quiesce", operation)
        state = self._weights_state
        if state == "writing" or (state == "torn" and not may_adopt_torn):
            raise WeightsNotCleanError(
                f"{operation} refused: the resident weights are {state}.",
                details={"operation": operation, "weights_state": state},
            )
        self._episode += 1
        adopted = False
        if owner == HOLDER:
            # A holder-owned hold: a clean quiesce, or a torn set's hold whose holder may be
            # gone. Adopt it: the slot stays held, the hold timer stops, resume and commit-begin
            # now refuse.
            self._adopted = _Adopted(self._holder_digest, self._holder_identity, state)
            self._suspend_hold_timer_locked()
            adopted = True
        self._hold_owner = f"internal:{operation}"
        return self._episode, adopted

    def _end_internal_locked(self, adopted: bool) -> Optional[tuple[Any, Any, Any, Any]]:
        """Hand an adopted hold back to the holder, or release it (lock held).

        Back to the holder only when it adopted a CLEAN quiesce, the holder is still attached
        and the weights are clean: `clean, quiesced`, hold timer re-armed in full. An inherited
        torn hold that is clean again is released (`clean, serving`); one still torn stays the
        holder's (the slot kept, as after the watchdog).
        """
        record = self._adopted
        self._adopted = None
        if not adopted or record is None:
            self._hold_owner = None
            return None
        state = self._weights_state
        holder_present = record.digest is not None
        if state == "torn" or (record.state == "clean" and holder_present and state == "clean"):
            self._hold_owner = HOLDER
            self._holder_digest, self._holder_identity = record.digest, record.identity
            if state == "clean":
                self._arm_hold_timer_locked()
            return None
        return self._release_holder_locked()

    @contextlib.asynccontextmanager
    async def serving_pause(
        self, operation: str, identity: Optional["WeightsIdentity"]
    ) -> AsyncIterator[PauseGuard]:
        """Pause serving for the block: hold the admission slot, refuse unless `clean`, never
        set `writing`. For a save snapshot (032) and adapter load and unload (033).

        Adopts the holder's clean quiesce when there is one; `commit-begin` and `resume` refuse
        with `PAUSE_IN_PROGRESS` until the block exits (031 3A.1).

        Raises:
            WeightsBarrierUnavailableError: continuous batching on, or more than one slot.
            PauseInProgressError: another pause or internal write holds the barrier.
            WeightsNotCleanError: a commit is open or the weights are torn.
            ModelBusyError: in-flight work did not drain in time, or the model is unloading.
            QueueFullError: the request queue is full.
        """
        inference = self._inference()
        self._refuse_unservable(inference, operation)
        with self._lock:
            episode, adopted = self._begin_internal_locked(operation, may_adopt_torn=False)
        started = self._clock()
        took_slot = False
        try:
            if not adopted:
                await self._take_slot(inference, episode, operation)
                took_slot = True
                with self._lock:
                    self._barrier = "quiesced"
                    self._held_model_id = _current_model_id()
            await self._after_hold(inference)
            with self._lock:
                guard = PauseGuard(self._weights_state, self._steps_since_load, episode)
            logger.info(
                "weights_serving_pause_held",
                operation=operation,
                episode=episode,
                adopted=adopted,
                label=identity.label if identity is not None else None,
            )
            yield guard
        finally:
            released = None
            with self._lock:
                if took_slot or not adopted:
                    self._hold_owner = None
                    self._barrier = "serving"
                    self._held_model_id = None
                else:
                    released = self._end_internal_locked(adopted)
            if took_slot or not adopted:
                await self._release_slot()
            elif released is not None:
                self._signal_release(*released)
            logger.info(
                "weights_serving_pause_released",
                operation=operation,
                episode=episode,
                held_ms=round((self._clock() - started) * 1000.0, 1),
            )

    @contextlib.asynccontextmanager
    async def internal_write(
        self, operation: str, identity: Optional["WeightsIdentity"]
    ) -> AsyncIterator[WriteHandle]:
        """A multi-tensor write under the barrier (032's load). Yields a `WriteHandle`.

        Accepts `clean` and `torn` (the recovery path, FR-31.12.3); refuses an open commit
        (`WEIGHTS_NOT_CLEAN`). Takes the slot, or adopts the holder's hold (a clean quiesce, or a
        torn set's hold whose holder may be gone, 031 review X6). `handle.mark_first_write()`
        moves the state to `writing`; before it a failure changes nothing, after it a failure is
        `torn`. On success the state becomes `clean` with steps 0 — only if it is still this
        write's own `writing` episode, so a concurrent `mark_torn` survives.

        Raises:
            WeightsBarrierUnavailableError, PauseInProgressError, WeightsNotCleanError,
            ModelBusyError, QueueFullError: as `serving_pause`.
        """
        inference = self._inference()
        self._refuse_unservable(inference, operation)
        with self._lock:
            episode, adopted = self._begin_internal_locked(operation, may_adopt_torn=True)
            self._internal_write_active = True
        handle = WriteHandle(episode, self._on_first_write)
        started = self._clock()
        took_slot = False
        outcome = "refused"
        try:
            if not adopted:
                await self._take_slot(inference, episode, operation)
                took_slot = True
                with self._lock:
                    self._barrier = "quiesced"
                    self._held_model_id = _current_model_id()
            await self._after_hold(inference)
            logger.info(
                "weights_internal_write_held",
                operation=operation,
                episode=episode,
                adopted=adopted,
                label=identity.label if identity is not None else None,
            )
            try:
                yield handle
            except BaseException:
                if handle.first_write_marked:
                    # `_on_first_write` already voided the holder's commit: no late commit-end
                    # can recover a set this write has started to replace.
                    self.mark_torn("internal_write_failed", writer=identity,
                                   operation=operation, episode=episode)
                    outcome = "torn"
                raise
            with self._lock:
                if self._weights_state == "writing" and self._writing_episode == episode:
                    self._weights_state = "clean"
                    self._steps_since_load = 0
                    self._commit_overran = False
                    self._clear_commit_locked()
                    self._set_abort_locked(False)
                    outcome = "clean"
                elif handle.first_write_marked:
                    outcome = self._weights_state  # torn by someone else meanwhile: kept
                else:
                    outcome = "unchanged"
        finally:
            released = None
            with self._lock:
                self._internal_write_active = False
                if took_slot or not adopted:
                    self._hold_owner = None
                    self._barrier = "serving"
                    self._held_model_id = None
                else:
                    released = self._end_internal_locked(adopted)
            if took_slot or not adopted:
                await self._release_slot()
            elif released is not None:
                self._signal_release(*released)
            logger.info(
                "weights_internal_write_released",
                operation=operation,
                episode=episode,
                outcome=outcome,
                held_ms=round((self._clock() - started) * 1000.0, 1),
            )

    def _on_first_write(self, handle: WriteHandle) -> None:
        with self._lock:
            if self._hold_owner is None or not self._internal_write_active:
                raise RuntimeError("mark_first_write outside its internal_write block")
            # The holder's commit (an inherited torn set's) is void from here on: no late
            # commit-end may recover a set this write has started to replace, and status must
            # not keep reporting it as an open commit with a growing `open_ms`.
            self._clear_commit_locked()
            self._weights_state = "writing"
            self._writing_episode = handle.episode


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _synchronize_resident_devices() -> None:
    """`torch.cuda.synchronize` on each card the resident transformers model holds memory on."""
    import torch

    from millm.ml.model_loader import ENGINE_TRANSFORMERS, LoadedModelState

    loaded = LoadedModelState().current
    if loaded is None or loaded.engine != ENGINE_TRANSFORMERS or not torch.cuda.is_available():
        return
    for index in sorted(set(loaded.gpu_indices)):
        torch.cuda.synchronize(index)


_barrier: Optional[WeightBarrier] = None
_barrier_lock = threading.Lock()


def get_weight_barrier() -> WeightBarrier:
    """The process-wide barrier, created on first use."""
    global _barrier
    with _barrier_lock:
        if _barrier is None:
            _barrier = WeightBarrier()
        return _barrier


def set_weight_barrier(barrier: Optional[WeightBarrier]) -> None:
    """Replace the process-wide barrier (tests, startup reset); None drops it, so the next
    `get_weight_barrier()` starts a fresh, clean one."""
    global _barrier
    with _barrier_lock:
        _barrier = barrier
