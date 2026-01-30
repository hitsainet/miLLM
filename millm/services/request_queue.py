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
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from millm.core.logging import get_logger

logger = get_logger(__name__)


class QueueFullError(Exception):
    """Raised when request queue is at capacity."""

    pass


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
            logger.debug(
                "request_queued",
                pending=self._pending,
                max_pending=self._max_pending,
            )

        try:
            # Wait for semaphore (actual GPU slot)
            if timeout:
                try:
                    await asyncio.wait_for(
                        self._semaphore.acquire(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Decrement pending on timeout
                    async with self._lock:
                        self._pending -= 1
                    raise
            else:
                await self._semaphore.acquire()

            logger.debug("request_slot_acquired", pending=self._pending)
            yield

        finally:
            self._semaphore.release()
            async with self._lock:
                self._pending -= 1
                logger.debug(
                    "request_slot_released",
                    pending=self._pending,
                )

    @property
    def pending_count(self) -> int:
        """Current number of pending requests."""
        return self._pending

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
