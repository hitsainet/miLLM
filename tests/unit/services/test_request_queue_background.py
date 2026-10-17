"""Feature 26 task 2.5: a batch chunk's slot — not pending, never QUEUE_FULL, always second.

Each test drives the REAL `RequestQueue` with real asyncio tasks; nothing is mocked. The ordering
tests record the order in which holders actually entered the slot.
"""

from __future__ import annotations

import asyncio

import pytest

from millm.services.request_queue import QueueFullError, RequestQueue



def bounded(fn):
    """A deadlocked slot must FAIL the test, not hang the suite: a mutation that leaves a waiter
    asleep forever (M1 makes `_pending` never return to zero) otherwise never turns red."""
    import functools

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await asyncio.wait_for(fn(*args, **kwargs), timeout=10)

    return wrapper

async def _spin(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


class _Py311Semaphore:
    """Python 3.11's `asyncio.Semaphore` semantics, which the IMAGE runs (`Dockerfile:7`).

    `release()` adds a permit and wakes the next waiter WITHOUT handing the permit to it; the woken
    waiter re-checks when it next runs, so a task that acquires before it runs takes the permit and
    the woken waiter waits again. 3.12 hands the permit over inside `release()` (and its
    `locked()` is true while waiters queue), which orders these tests correctly with NO priority
    rule at all — the first version of this fixture subclassed 3.12's Semaphore, overrode only
    `locked()`, and let the interactive-priority mutation (M2) SURVIVE.
    """

    def __init__(self, value: int = 1) -> None:
        import collections

        self._value = value
        self._waiters: "collections.deque[asyncio.Future]" = collections.deque()

    def locked(self) -> bool:
        return self._value == 0

    async def acquire(self) -> bool:
        while self._value <= 0:
            fut = asyncio.get_running_loop().create_future()
            self._waiters.append(fut)
            try:
                await fut
            finally:
                if fut in self._waiters:
                    self._waiters.remove(fut)
        self._value -= 1
        return True

    def release(self) -> None:
        self._value += 1
        while self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(True)
                break


def _queue_311(max_pending: int = 10) -> RequestQueue:
    queue = RequestQueue(max_concurrent=1, max_pending=max_pending)
    queue._semaphore = _Py311Semaphore(1)
    return queue


@bounded
async def test_a_background_holder_does_not_raise_pending_count():
    queue = RequestQueue(max_concurrent=1, max_pending=10)
    async with queue.acquire_background():
        assert queue.pending_count == 0
        assert queue.background_holding_count == 1
        assert queue.occupied_count == 1
    assert (queue.background_holding_count, queue.occupied_count) == (0, 0)


@bounded
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


@bounded
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


@bounded
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


@bounded
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


@bounded
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


@bounded
async def test_a_chunk_that_raises_releases_its_slot():
    queue = RequestQueue(max_concurrent=1, max_pending=10)
    with pytest.raises(RuntimeError):
        async with queue.acquire_background():
            raise RuntimeError("row failed")
    assert queue.occupied_count == 0
    async with queue.acquire():
        pass


@bounded
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


@bounded
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


@bounded
async def test_a_chunk_re_requested_at_once_still_lets_a_woken_chat_go_first():
    """M2's target, on 3.11 semantics. The batch releases its slot and asks for the next one
    WITHOUT yielding in between; a chat request was already waiting. Without the interactive
    priority rule the chunk takes the permit the chat was just woken for."""
    queue = _queue_311()
    order: list[str] = []
    in_first = asyncio.Event()
    release_first = asyncio.Event()

    async def batch():
        async with queue.acquire_background():
            order.append("chunk0")
            in_first.set()
            await release_first.wait()
        # No await between the release above and this request: the woken chat has not run yet.
        async with queue.acquire_background():
            order.append("chunk1")

    async def chat():
        async with queue.acquire():
            order.append("chat")

    runner = asyncio.create_task(batch())
    await in_first.wait()
    talk = asyncio.create_task(chat())
    await _spin()
    release_first.set()
    await asyncio.gather(runner, talk)
    assert order == ["chunk0", "chat", "chunk1"]


@bounded
async def test_a_chunk_requested_while_the_interactive_queue_is_full_is_not_refused():
    """X1's target: `acquire_background` never reads `max_pending`. With the interactive queue
    at its cap, a batch chunk still queues (and runs after) instead of getting QUEUE_FULL."""
    queue = RequestQueue(max_concurrent=1, max_pending=2)
    release = asyncio.Event()
    order: list[str] = []

    async def chat(name, gate=None):
        async with queue.acquire():
            order.append(name)
            if gate is not None:
                await gate.wait()

    async def chunk():
        async with queue.acquire_background():
            order.append("chunk")

    first = asyncio.create_task(chat("chat0", release))
    await _spin()
    second = asyncio.create_task(chat("chat1"))
    await _spin()
    assert queue.pending_count == queue.max_pending
    with pytest.raises(QueueFullError):
        async with queue.acquire():
            pass
    background = asyncio.create_task(chunk())
    await _spin()
    assert not background.done(), "the chunk must wait, not fail"
    release.set()
    await asyncio.gather(first, second, background)
    assert order == ["chat0", "chat1", "chunk"]
