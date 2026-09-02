"""The event loop must survive a blocking iterator.

Before `aiter_blocking`, `stream_chat_completion` iterated TextIteratorStreamer
synchronously. Measured against granite-4.2-8b on 2026-09-02: all 16 SSE chunks
arrived at 6.12s with spread 0.00s -- including the role chunk yielded before
generation began. The client saw nothing for six seconds and then everything at
once, which reads as a hung server.

The test that matters is the CONCURRENCY one: a competing task must make
progress while the iterator blocks. Asserting only that the right items come
out would pass against the blocking version that caused the outage.
"""

import asyncio
import time

import pytest

from millm.services.async_iter import aiter_blocking


class _BlockingIterator:
    """Mimics TextIteratorStreamer: __next__ sleeps, holding the caller."""

    def __init__(self, items, delay=0.02):
        self._items = list(items)
        self._delay = delay

    def __iter__(self):
        return self

    def __next__(self):
        if not self._items:
            raise StopIteration
        time.sleep(self._delay)          # blocking, like a queue.get()
        return self._items.pop(0)


@pytest.mark.asyncio
async def test_yields_every_item_in_order():
    out = [x async for x in aiter_blocking(_BlockingIterator("abcd", 0.001))]
    assert out == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_empty_iterator_terminates():
    assert [x async for x in aiter_blocking(_BlockingIterator([]))] == []


@pytest.mark.asyncio
async def test_a_competing_task_RUNS_while_the_iterator_blocks():
    """The regression guard. Fails against synchronous iteration."""
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    task = asyncio.create_task(ticker())
    try:
        out = [x async for x in aiter_blocking(_BlockingIterator("abcde", 0.02))]
    finally:
        task.cancel()

    assert out == list("abcde")
    # ~100ms of blocking iteration; a loop that was pinned would tick ~0 times.
    assert ticks >= 5, f"event loop was starved: only {ticks} ticks"


@pytest.mark.asyncio
async def test_falsy_items_are_not_swallowed():
    """`next(it, sentinel)` must distinguish '' from exhaustion."""
    out = [x async for x in aiter_blocking(_BlockingIterator(["a", "", "b"], 0.001))]
    assert out == ["a", "", "b"]
