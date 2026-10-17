"""Feature 29 task 7.1: the holding counter and duration window survive every exit path."""

from __future__ import annotations

import asyncio

import pytest

from millm.services.request_queue import QueueFullError, RequestQueue


async def test_a_waiter_cancelled_before_its_slot_never_counted_as_holding():
    queue = RequestQueue(max_concurrent=1, max_pending=5)
    release = asyncio.Event()

    async def holder():
        async with queue.acquire():
            await release.wait()

    async def waiter():
        async with queue.acquire():
            pass

    h = asyncio.create_task(holder())
    await asyncio.sleep(0)
    w = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    assert (queue.pending_count, queue.holding_count) == (2, 1)
    w.cancel()
    with pytest.raises(asyncio.CancelledError):
        await w
    assert (queue.pending_count, queue.holding_count) == (1, 1)
    release.set()
    await h
    assert (queue.pending_count, queue.holding_count) == (0, 0)
    assert len(queue._durations) == 1, "only the holder that held a slot is timed"


async def test_a_holder_that_raises_is_released_and_timed():
    queue = RequestQueue(max_concurrent=1, max_pending=5)
    with pytest.raises(RuntimeError):
        async with queue.acquire():
            assert queue.holding_count == 1
            raise RuntimeError("generation failed")
    assert queue.holding_count == 0
    assert len(queue._durations) == 1


async def test_a_full_queue_changes_no_counter():
    queue = RequestQueue(max_concurrent=1, max_pending=1)
    release = asyncio.Event()

    async def holder():
        async with queue.acquire():
            await release.wait()

    h = asyncio.create_task(holder())
    await asyncio.sleep(0)
    with pytest.raises(QueueFullError):
        async with queue.acquire():
            pass
    assert (queue.pending_count, queue.holding_count) == (1, 1)
    release.set()
    await h


async def test_window_is_bounded_and_median_needs_three():
    from millm.core.config import settings

    queue = RequestQueue()
    for i in range(settings.QUEUE_DURATION_WINDOW + 7):
        if i == 2:
            assert queue.median_hold_seconds() is None
        async with queue.acquire():
            pass
    assert len(queue._durations) == settings.QUEUE_DURATION_WINDOW
    assert queue.median_hold_seconds() is not None
    assert queue.background_holding_count == 0
