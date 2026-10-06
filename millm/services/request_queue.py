"""
Request queue for managing concurrent inference.

Manages concurrent access to GPU resources for inference requests.
Uses a semaphore to limit concurrent operations and a pending counter
to prevent queue overflow.

Implementation notes:
- Semaphore limits concurrent GPU operations (default: 1)
- Pending counter prevents queue overflow (default: 5)
- Context manager ensures proper cleanup
"""

import asyncio
import statistics
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from millm.core.errors import MiLLMError
from millm.core.logging import get_logger

logger = get_logger(__name__)


class QueueFullError(MiLLMError):
    """Raised when the request queue is at capacity.

    A MiLLMError with code QUEUE_FULL so the OpenAI error handler maps it to a
    proper 503 backpressure response (ERROR_STATUS_MAP). It was previously a bare
    Exception, so it fell through to a generic HTTP 500 — the intended 503 path
    was never reached and the docs (which said 429) were also wrong.
    """

    code = "QUEUE_FULL"
    status_code = 503


class RequestQueue:
    """
    Manages concurrent inference requests.

    Provides controlled access to GPU resources by limiting:
    - max_concurrent: How many requests can run simultaneously (default: 1)
    - max_pending: How many requests can wait in queue (default: 5)

    Default settings assume single GPU that can only run one inference
    at a time, with a small queue to prevent request overload.

    Usage:
        queue = RequestQueue(max_concurrent=1, max_pending=5)

        async with queue.acquire():
            result = await generate(...)

    Attributes:
        pending_count: Current number of pending requests
        is_available: Whether the queue can accept new requests
    """

    def __init__(self, max_concurrent: int = 1, max_pending: int = 5) -> None:
        """
        Initialize the request queue.

        Args:
            max_concurrent: Maximum concurrent GPU operations
            max_pending: Maximum pending requests in queue
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending = 0
        self._max_pending = max_pending
        self._max_concurrent = max_concurrent
        self._lock = asyncio.Lock()
        #: Over `_lock`: a background waiter sleeps here until no interactive request is waiting
        #: (Feature 26, FTDD §7 constraint 2). Every change to `_pending` or `_holding` notifies it.
        self._cond = asyncio.Condition(self._lock)
        #: Set while no request holds or waits for a slot — interactive OR background (wait_idle).
        self._idle = asyncio.Event()
        self._idle.set()
        #: Interactive requests HOLDING a slot now (Feature 29). `_pending` counts waiting
        #: plus holding, so `pending_count - holding_count` is the number waiting.
        self._holding = 0
        #: Feature 26's batch chunks waiting for, and holding, the slot. ⚠ NEVER in `_pending`:
        #: a batch must not count against MAX_PENDING_REQUESTS nor ever receive QUEUE_FULL, and an
        #: interactive request's QUEUE_FULL threshold must not move while a batch runs (FR-26.4.4).
        self._background_waiting = 0
        self._background_holding = 0
        #: Recent slot-holding durations in seconds, for the estimated wait (FR-29.7.4).
        from millm.core.config import settings

        self._durations: deque[float] = deque(maxlen=max(settings.QUEUE_DURATION_WINDOW, 1))

    @asynccontextmanager
    async def acquire(
        self, timeout: Optional[float] = None
    ) -> AsyncGenerator[None, None]:
        """
        Acquire a slot in the request queue.

        This is an async context manager that should be used with `async with`.
        It first increments the pending count (checking for overflow), then
        waits for a semaphore slot to become available for actual execution.

        Usage:
            async with request_queue.acquire():
                # Run inference here - you have the GPU slot
                result = await generate(...)

            async with request_queue.acquire(timeout=30.0):
                # With 30 second timeout
                result = await generate(...)

        Args:
            timeout: Optional timeout in seconds for waiting for a slot

        Yields:
            None - just provides the context

        Raises:
            QueueFullError: If queue is at max_pending capacity
            asyncio.TimeoutError: If timeout expires waiting for slot
        """
        # Check pending count and increment if space available
        async with self._lock:
            if self._pending >= self._max_pending:
                logger.warning(
                    "request_queue_full",
                    pending=self._pending,
                    max_pending=self._max_pending,
                )
                raise QueueFullError(
                    f"Request queue full ({self._pending} pending). Try again later."
                )
            self._pending += 1
            self._idle.clear()
            logger.debug(
                "request_queued",
                pending=self._pending,
                max_pending=self._max_pending,
            )

        # Release ONLY what was actually acquired.
        #
        # The `finally` used to release unconditionally, so any exit before the
        # semaphore was obtained handed back a permit that was never taken.
        # asyncio.Semaphore is unbounded, so each such exit raised the effective
        # concurrency limit BY ONE, PERMANENTLY.
        #
        # This was not theoretical. Cancellation while awaiting the semaphore is
        # ordinary client behaviour — an SSE client hanging up, or a caller
        # timing out, while its request is queued behind another. Measured on
        # this code at max_concurrent=1: one cancelled waiter left the semaphore
        # holding 2 permits, and two generations then ran concurrently.
        #
        # That matters far beyond throughput. Per-request steering apply/restore,
        # the sensing buffers and monitoring attribution are all process-global
        # and rely on this semaphore for isolation (see MAX_CONCURRENT_REQUESTS
        # in core/config.py). A leaked permit silently removes the only thing
        # preventing two generations interleaving their steering state.
        #
        # The timeout branch also decremented `_pending` and then let `finally`
        # decrement it a second time, driving the count negative. `finally` now
        # owns that decrement exactly once.
        acquired = False
        try:
            # Wait for semaphore (actual GPU slot)
            if timeout:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            else:
                await self._semaphore.acquire()
            acquired = True
            self._holding += 1
            started = time.monotonic()
            # A background waiter may now find no interactive request waiting.
            await self._notify()

            logger.debug("request_slot_acquired", pending=self._pending)
            yield

        finally:
            if acquired:
                self._holding -= 1
                self._durations.append(time.monotonic() - started)
                self._semaphore.release()
            async with self._cond:
                self._pending -= 1
                self._set_idle_if_empty()
                # A waiter that left (or a holder that finished) changes the background predicate.
                self._cond.notify_all()
                logger.debug(
                    "request_slot_released",
                    pending=self._pending,
                )

    def _background_may_enter(self) -> bool:
        """No interactive request waiting (`_pending - _holding == 0`) and a free slot."""
        return self._pending - self._holding == 0 and not self._semaphore.locked()

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    def _set_idle_if_empty(self) -> None:
        """`_idle` only when NOTHING holds or waits — a batch chunk included (FTDD §7)."""
        if self.occupied_count == 0:
            self._idle.set()

    @asynccontextmanager
    async def acquire_background(self) -> AsyncGenerator[None, None]:
        """A slot for a batch chunk (Feature 26). Never `QUEUE_FULL`; always behind interactive work.

        ⚠ THREE RULES, each load-bearing (FR-26.4.3, FR-26.4.4):

        * It never reads `max_pending` and never touches `_pending`. A batch waits outside the
          interactive count, so it can never be refused with QUEUE_FULL, and the 11th interactive
          request is refused exactly as it was before any batch existed.
        * Before it takes the semaphore it waits until NO interactive request is waiting
          (`_pending - _holding == 0`). So at every chunk boundary an interactive waiter goes
          first, whatever the interpreter's semaphore fairness — the image runs Python 3.11,
          whose `Semaphore` lets a newcomer overtake a woken waiter (FTDD §7).
        * It counts itself in `_background_waiting` / `_background_holding`, so `wait_idle` (the
          unload drain) waits for a running chunk and `in_flight` sees it.

        A race where an interactive request arrives between the wait and the semaphore costs that
        request one chunk, which is what "answered within one chunk's duration" allows.

        Taken ONLY through `InferenceService._admit(background=True)`
        (`test_every_request_queue_slot_is_taken_through_admission`).
        """
        acquired = False
        started = time.monotonic()
        try:
            async with self._cond:
                self._background_waiting += 1
                self._idle.clear()
                # ⚠ Interactive priority (mutation control M2). The chunk moves only when NO
                # interactive request is waiting AND the slot is free, so it never queues on the
                # semaphore at all: it cannot be ahead of a chat request there, on 3.11's
                # unfair Semaphore or 3.12's fair one.
                await self._cond.wait_for(self._background_may_enter)
            # The predicate saw a free semaphore under the lock, with no await since: this
            # acquire does not suspend, so nothing can overtake it in between.
            await self._semaphore.acquire()
            acquired = True
            started = time.monotonic()
            async with self._cond:
                self._background_waiting -= 1
                self._background_holding += 1
            logger.debug("background_slot_acquired", pending=self._pending)
            yield
        finally:
            async with self._cond:
                if acquired:
                    self._background_holding -= 1
                    self._durations.append(time.monotonic() - started)
                    self._semaphore.release()
                else:
                    self._background_waiting -= 1
                self._set_idle_if_empty()
                self._cond.notify_all()

    async def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Wait until no request holds or waits for a slot. False if `timeout` passes first.

        The unload's drain (ModelService.unload_model): woken by the last request
        leaving, not by polling. A batch chunk counts (`occupied_count`), so an unload never moves
        the weights under a running chunk.
        """
        if self.occupied_count == 0:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True

    @property
    def pending_count(self) -> int:
        """Current number of pending requests."""
        return self._pending

    @property
    def holding_count(self) -> int:
        """Interactive requests holding a slot now — the idle cache release included (T-90)."""
        return self._holding

    @property
    def background_holding_count(self) -> int:
        """Batch chunks holding a slot now (Feature 26)."""
        return self._background_holding

    @property
    def background_waiting_count(self) -> int:
        """Batch chunks waiting for a slot now (Feature 26)."""
        return self._background_waiting

    @property
    def occupied_count(self) -> int:
        """Everything holding or waiting: interactive pending + background waiting + holding.

        What the unload drain and the idle-cache release read (026 FTDD §7). `pending_count` keeps
        its interactive-only meaning, so QUEUE_FULL and the health field are unchanged.
        """
        return self._pending + self._background_waiting + self._background_holding

    def median_hold_seconds(self) -> Optional[float]:
        """Median of the recent slot-holding durations; None with fewer than three samples."""
        if len(self._durations) < 3:
            return None
        return float(statistics.median(self._durations))

    @property
    def is_available(self) -> bool:
        """Check if queue can accept new requests."""
        return self._pending < self._max_pending

    @property
    def max_pending(self) -> int:
        """Maximum number of pending requests allowed."""
        return self._max_pending

    @property
    def max_concurrent(self) -> int:
        """Maximum number of concurrent requests allowed."""
        return self._max_concurrent
