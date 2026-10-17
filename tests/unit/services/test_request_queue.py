"""Unit tests for RequestQueue."""

import asyncio
from unittest.mock import patch

import pytest

from millm.services.request_queue import QueueFullError, RequestQueue


@pytest.fixture
def queue():
    """Create a RequestQueue with default settings."""
    return RequestQueue(max_concurrent=1, max_pending=5)


@pytest.fixture
def small_queue():
    """Create a RequestQueue with small limits for testing overflow."""
    return RequestQueue(max_concurrent=1, max_pending=2)


class TestRequestQueueAcquire:
    """Tests for acquire context manager."""

    @pytest.mark.asyncio
    async def test_acquire_returns_context_manager(self, queue):
        """Test that acquire can be used as an async context manager."""
        async with queue.acquire():
            # If we reach here, the context manager worked
            assert True

    @pytest.mark.asyncio
    async def test_acquire_allows_execution_inside_block(self, queue):
        """Test that code executes normally inside the acquired block."""
        result = None
        async with queue.acquire():
            result = "executed"

        assert result == "executed"


class TestRequestQueueConcurrency:
    """Tests for concurrent access behavior."""

    @pytest.mark.asyncio
    async def test_second_request_waits_for_first(self, queue):
        """Test that second request waits when first holds the lock."""
        order = []

        async def first_task():
            async with queue.acquire():
                order.append("first_start")
                await asyncio.sleep(0.05)
                order.append("first_end")

        async def second_task():
            # Small delay to ensure first_task acquires first
            await asyncio.sleep(0.01)
            async with queue.acquire():
                order.append("second_start")
                order.append("second_end")

        await asyncio.gather(first_task(), second_task())

        assert order == ["first_start", "first_end", "second_start", "second_end"]


class TestRequestQueuePendingCount:
    """Tests for pending_count tracking."""

    @pytest.mark.asyncio
    async def test_pending_count_starts_at_zero(self, queue):
        """Test that pending count starts at zero."""
        assert queue.pending_count == 0

    @pytest.mark.asyncio
    async def test_pending_count_increments_on_acquire(self, queue):
        """Test that pending count increments when a request is queued."""
        # Use a queue with max_concurrent=1, so the second request will be pending
        event = asyncio.Event()
        observed_count = None

        async def holder():
            async with queue.acquire():
                event.set()
                await asyncio.sleep(0.1)

        async def waiter():
            await event.wait()
            # At this point, holder has the slot but hasn't released.
            # If we check pending while holder is running, it should be 1.
            # We can't easily observe from inside acquire, so check before acquiring.
            observed_count_before = queue.pending_count
            return observed_count_before

        task1 = asyncio.create_task(holder())
        await event.wait()
        # Holder has the slot, pending count should be 1 (holder is still in acquire block)
        assert queue.pending_count == 1
        task1.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_pending_count_decrements_on_release(self, queue):
        """Test that pending count goes back to zero after release."""
        async with queue.acquire():
            assert queue.pending_count == 1

        assert queue.pending_count == 0


class TestRequestQueueOverflow:
    """Tests for queue overflow behavior."""

    @pytest.mark.asyncio
    async def test_raises_queue_full_error(self, small_queue):
        """Test that QueueFullError is raised when max_pending exceeded.

        Strategy: use a release_gate to keep holders running while we
        verify that a third acquire raises QueueFullError. With
        max_concurrent=1 and max_pending=2 we need task1 holding the
        semaphore AND task2 waiting for it simultaneously so that a
        third attempt sees pending_count >= max_pending.
        """
        release_gate = asyncio.Event()
        task1_acquired = asyncio.Event()

        async def hold():
            async with small_queue.acquire():
                task1_acquired.set()
                await release_gate.wait()

        # Task1 fills the single concurrent slot
        task1 = asyncio.create_task(hold())
        await task1_acquired.wait()

        # Task2 queues behind task1 (pending=2)
        task2 = asyncio.create_task(hold())
        await asyncio.sleep(0)  # Let task2 increment pending_count

        # Third acquire must overflow: pending_count >= max_pending
        with pytest.raises(QueueFullError) as exc_info:
            async with small_queue.acquire():
                pass

        assert "full" in str(exc_info.value).lower()

        # Unblock holders so they can finish cleanly
        release_gate.set()
        await asyncio.gather(task1, task2, return_exceptions=True)

    def test_queue_full_error_is_a_millm_error_with_503_backpressure_code(self):
        """QueueFullError must carry the QUEUE_FULL contract so the OpenAI error
        handler maps it to HTTP 503 (backpressure), not a generic 500. It used to
        be a bare Exception, which never reached the handler → 500, while the docs
        claimed 429. Both were wrong; this pins the 503 contract."""
        from millm.core.errors import MiLLMError
        from millm.api.routes.openai.errors import ERROR_STATUS_MAP

        err = QueueFullError("Request queue is full")
        assert isinstance(err, MiLLMError), (
            "QueueFullError must subclass MiLLMError or the OpenAI handler never "
            "sees it and it falls through to a generic 500")
        assert err.code == "QUEUE_FULL"
        # The handler resolves the HTTP status from this map by err.code.
        status, _type = ERROR_STATUS_MAP[err.code]
        assert status == 503, "queue-full is backpressure → 503, not 500 or 429"

    @pytest.mark.asyncio
    async def test_queue_full_error_message_includes_count(self, small_queue):
        """Test that QueueFullError message includes pending count."""
        release_gate = asyncio.Event()
        task1_acquired = asyncio.Event()

        async def hold():
            async with small_queue.acquire():
                task1_acquired.set()
                await release_gate.wait()

        task1 = asyncio.create_task(hold())
        await task1_acquired.wait()

        task2 = asyncio.create_task(hold())
        await asyncio.sleep(0)

        with pytest.raises(QueueFullError) as exc_info:
            async with small_queue.acquire():
                pass

        assert str(small_queue.max_pending) in str(exc_info.value)

        release_gate.set()
        await asyncio.gather(task1, task2, return_exceptions=True)


class TestRequestQueueProperties:
    """Tests for queue property accessors."""

    def test_is_available_when_empty(self, queue):
        """Test that is_available returns True when queue is empty."""
        assert queue.is_available is True

    def test_max_pending_returns_configured_value(self, small_queue):
        """Test that max_pending returns the configured maximum."""
        assert small_queue.max_pending == 2

    def test_max_concurrent_returns_configured_value(self, queue):
        """Test that max_concurrent returns the configured maximum."""
        assert queue.max_concurrent == 1


class TestTheSemaphoreIsNotOverReleased:
    """A permit must be returned only if it was actually taken.

    The `finally` released unconditionally, so any exit before the semaphore was
    obtained handed back a permit never taken. asyncio.Semaphore is unbounded,
    so each such exit raised the effective concurrency limit BY ONE, PERMANENTLY.

    This is ordinary client behaviour, not an edge case: an SSE client hanging up
    — or a caller timing out — while queued behind another request. Measured on
    the pre-fix code at max_concurrent=1, ONE cancelled waiter left 2 permits and
    two generations then ran at once.

    It matters far beyond throughput. Per-request steering apply/restore, the
    sensing buffers and monitoring attribution are process-global and depend on
    this semaphore for isolation, so a leaked permit silently removes the only
    thing keeping two generations from interleaving their steering state.

    Mutation controls:
      C70 release unconditionally in `finally`
           -> test_a_cancelled_waiter_does_not_leak_a_permit
      C71 restore the duplicate `_pending -= 1` in the timeout branch
           -> test_a_timeout_decrements_pending_exactly_once
    """

    @pytest.mark.asyncio
    async def test_a_cancelled_waiter_does_not_leak_a_permit(self):
        q = RequestQueue(max_concurrent=1, max_pending=10)

        async def hold(secs):
            async with q.acquire():
                await asyncio.sleep(secs)

        holder = asyncio.create_task(hold(0.4))
        await asyncio.sleep(0.05)

        waiter = asyncio.create_task(hold(0.1))   # queues behind holder
        await asyncio.sleep(0.05)
        waiter.cancel()                            # the client hangs up
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await holder

        assert q._semaphore._value == 1, (
            f"semaphore holds {q._semaphore._value} permits after one cancelled "
            f"waiter; the concurrency limit has been permanently raised and "
            f"steering isolation is gone"
        )
        assert q.pending_count == 0

    @pytest.mark.asyncio
    async def test_the_limit_still_holds_after_a_cancellation(self):
        """The consequence, stated as behaviour rather than internals."""
        q = RequestQueue(max_concurrent=1, max_pending=10)

        async def hold(secs):
            async with q.acquire():
                await asyncio.sleep(secs)

        holder = asyncio.create_task(hold(0.3))
        await asyncio.sleep(0.05)
        waiter = asyncio.create_task(hold(0.1))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await holder

        running = 0
        peak = 0

        async def probe():
            nonlocal running, peak
            async with q.acquire():
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0.1)
                running -= 1

        await asyncio.gather(*(probe() for _ in range(3)))
        assert peak == 1, (
            f"{peak} generations ran concurrently at max_concurrent=1 after a "
            f"cancellation — interleaved steering apply/restore is now possible"
        )

    @pytest.mark.asyncio
    async def test_a_timeout_decrements_pending_exactly_once(self):
        """C71. The timeout branch decremented `_pending` and then let `finally`
        decrement it again, driving the count negative."""
        q = RequestQueue(max_concurrent=1, max_pending=10)

        async def hold():
            async with q.acquire():
                await asyncio.sleep(0.4)

        holder = asyncio.create_task(hold())
        await asyncio.sleep(0.05)

        with pytest.raises(asyncio.TimeoutError):
            async with q.acquire(timeout=0.05):
                pass
        await holder

        assert q.pending_count == 0, (
            f"pending is {q.pending_count} after a timeout; it was decremented "
            f"twice and the queue's capacity accounting is now wrong"
        )
        assert q._semaphore._value == 1, "a timed-out waiter released a permit it never held"
