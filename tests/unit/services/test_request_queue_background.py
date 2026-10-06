"""Feature 26 task 2.5: a batch chunk's slot — not pending, never QUEUE_FULL, always second.

Each test drives the REAL `RequestQueue` with real asyncio tasks; nothing is mocked. The ordering
tests record the order in which holders actually entered the slot.
"""

from __future__ import annotations

import asyncio

import pytest

from millm.services.request_queue import QueueFullError, RequestQueue


async def _spin(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


class _Py311Semaphore(asyncio.Semaphore):
    """3.11's `locked()`: true only at zero permits, so a newcomer can overtake a woken waiter.

    ⚠ THE IMAGE RUNS PYTHON 3.11 (`Dockerfile:7`); this venv runs 3.12, whose `locked()` is also
    true while waiters are queued and would order these tests correctly with NO priority rule at
    all. Without this fixture the interactive-priority mutation (M2) survives on 3.12 while the
    defect ships on 3.11 — the "fixture agrees with the code by construction" trap.
    """

    def locked(self) -> bool:  # noqa: D401
        return self._value == 0


def _queue_311(max_pending: int = 10) -> RequestQueue:
    queue = RequestQueue(max_concurrent=1, max_pending=max_pending)
    queue._semaphore = _Py311Semaphore(1)
    return queue


async def test_a_background_holder_does_not_raise_pending_count():
    queue = RequestQueue(max_concurrent=1, max_pending=10)
    async with queue.acquire_background():
        assert queue.pending_count == 0
        assert queue.background_holding_count == 1
        assert queue.occupied_count == 1
    assert (queue.background_holding_count, queue.occupied_count) == (0, 0)


async def test_background_waiters_never_get_queue_full_and_never_count():
    """M1's target: many waiting chunks, a max_pending of 1, and not one QUEUE_FULL."""
    queue = RequestQueue(max_concurrent=1, max_pending=1)
    release = asyncio.Event()

    async def chunk():
        async with queue.acquire_background():
            await release.wait()

    tasks = [asyncio.create_task(chunk()) for _ in range(5)]
    await _spin()
    assert queue.pending_count == 0
    assert queue.background_waiting_count == 4 and queue.background_holding_count == 1
    release.set()
    await asyncio.gather(*tasks)
    assert queue.occupied_count == 0


async def test_the_interactive_queue_full_threshold_is_unchanged_while_a_batch_runs():
    """With a chunk holding the slot and another waiting, the 11th interactive request is refused
    exactly as before (MAX_PENDING_REQUESTS=10) — not the 9th, not the 12th."""
    queue = RequestQueue(max_concurrent=1, max_pending=10)
    release = asyncio.Event()

    async def chunk():
        async with queue.acquire_background():
            await release.wait()

    async def chat():
        async with queue.acquire():
            pass

    chunks = [asyncio.create_task(chunk()) for _ in range(2)]
    await _spin()
    chats = [asyncio.create_task(chat()) for _ in range(10)]
    await _spin()
    assert queue.pending_count == 10
    with pytest.raises(QueueFullError):
        async with queue.acquire():
            pass
    release.set()
    await asyncio.gather(*chunks, *chats)


async def test_an_interactive_waiter_runs_before_the_next_chunk():
    """M2's target. A chunk holds the slot; a chat request and the batch's next chunk both wait.
    The chat must enter first — whichever arrived first."""
    queue = _queue_311()
    order: list[str] = []
    release = asyncio.Event()

    async def chunk(name: str, gate: asyncio.Event | None = None):
        async with queue.acquire_background():
            order.append(name)
            if gate is not None:
                await gate.wait()

    async def chat():
        async with queue.acquire():
            order.append("chat")

    first = asyncio.create_task(chunk("chunk0", release))
    await _spin()
    # The NEXT chunk queues before the chat request does.
    nxt = asyncio.create_task(chunk("chunk1"))
    await _spin()
    talk = asyncio.create_task(chat())
    await _spin()
    release.set()
    await asyncio.gather(first, nxt, talk)
    assert order == ["chunk0", "chat", "chunk1"]


async def test_wait_idle_waits_for_a_background_holder():
    """M14's queue half: the unload drain must not see an idle queue while a chunk runs."""
    queue = RequestQueue(max_concurrent=1, max_pending=10)
    release = asyncio.Event()

    async def chunk():
        async with queue.acquire_background():
            await release.wait()

    task = asyncio.create_task(chunk())
    await _spin()
    assert await queue.wait_idle(timeout=0.05) is False
    release.set()
    await task
    assert await queue.wait_idle(timeout=0.05) is True


async def test_a_cancelled_background_waiter_leaves_no_count_behind():
    queue = RequestQueue(max_concurrent=1, max_pending=10)
    release = asyncio.Event()

    async def holder():
        async with queue.acquire():
            await release.wait()

    async def chunk():
        async with queue.acquire_background():
            pass

    h = asyncio.create_task(holder())
    await _spin()
    c = asyncio.create_task(chunk())
    await _spin()
    assert queue.background_waiting_count == 1
    c.cancel()
    with pytest.raises(asyncio.CancelledError):
        await c
    assert queue.background_waiting_count == 0
    release.set()
    await h
    assert queue.occupied_count == 0
    assert queue._idle.is_set()


async def test_a_chunk_that_raises_releases_its_slot():
    queue = RequestQueue(max_concurrent=1, max_pending=10)
    with pytest.raises(RuntimeError):
        async with queue.acquire_background():
            raise RuntimeError("row failed")
    assert queue.occupied_count == 0
    async with queue.acquire():
        pass


async def test_a_background_chunk_waits_while_interactive_requests_are_queued():
    """Interactive waiters arriving one after another all go first; the chunk enters last."""
    queue = _queue_311()
    order: list[str] = []
    release = asyncio.Event()

    async def chat(name: str, gate: asyncio.Event | None = None):
        async with queue.acquire():
            order.append(name)
            if gate is not None:
                await gate.wait()

    async def chunk():
        async with queue.acquire_background():
            order.append("chunk")

    first = asyncio.create_task(chat("chat0", release))
    await _spin()
    c = asyncio.create_task(chunk())
    await _spin()
    others = [asyncio.create_task(chat(f"chat{i}")) for i in (1, 2)]
    await _spin()
    release.set()
    await asyncio.gather(first, c, *others)
    assert order == ["chat0", "chat1", "chat2", "chunk"]


async def test_the_priority_holds_on_the_real_312_semaphore_too():
    queue = RequestQueue(max_concurrent=1, max_pending=10)
    order: list[str] = []
    release = asyncio.Event()

    async def chunk(name, gate=None):
        async with queue.acquire_background():
            order.append(name)
            if gate is not None:
                await gate.wait()

    async def chat():
        async with queue.acquire():
            order.append("chat")

    first = asyncio.create_task(chunk("chunk0", release))
    await _spin()
    nxt = asyncio.create_task(chunk("chunk1"))
    await _spin()
    talk = asyncio.create_task(chat())
    await _spin()
    release.set()
    await asyncio.gather(first, nxt, talk)
    assert order == ["chunk0", "chat", "chunk1"]
