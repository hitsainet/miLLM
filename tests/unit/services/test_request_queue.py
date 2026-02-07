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
        """Test that QueueFullError is raised when max_pending exceeded."""
        # Fill up the queue
        holders = []
        events = []
        for _ in range(small_queue.max_pending):
            event = asyncio.Event()
            events.append(event)

            async def hold(evt=event):
                async with small_queue.acquire():
                    evt.set()
                    await asyncio.sleep(1.0)

            holders.append(asyncio.create_task(hold()))

        # Wait for all holders to acquire
        for event in events:
            await event.wait()

        # Now the queue should be full
        with pytest.raises(QueueFullError) as exc_info:
            async with small_queue.acquire():
                pass

        assert "full" in str(exc_info.value).lower()

        # Cleanup
        for task in holders:
            task.cancel()
        await asyncio.gather(*holders, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_queue_full_error_message_includes_count(self, small_queue):
        """Test that QueueFullError message includes pending count."""
        holders = []
        events = []
        for _ in range(small_queue.max_pending):
            event = asyncio.Event()
            events.append(event)

            async def hold(evt=event):
                async with small_queue.acquire():
                    evt.set()
                    await asyncio.sleep(1.0)

            holders.append(asyncio.create_task(hold()))

        for event in events:
            await event.wait()

        with pytest.raises(QueueFullError) as exc_info:
            async with small_queue.acquire():
                pass

        assert str(small_queue.max_pending) in str(exc_info.value)

        for task in holders:
            task.cancel()
        await asyncio.gather(*holders, return_exceptions=True)


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
