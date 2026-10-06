"""Feature 26 task 2.5: `_admit` is re-entrant for the task that holds the slot — and only it.

A batch chunk holds the one slot and runs each row through the synchronous service method, which
enters `_admit()` itself. Without re-entry that nested call waits forever for a slot its own task
holds (deadlock at MAX_CONCURRENT_REQUESTS=1). With re-entry keyed on the CONTEXT rather than the
TASK, a child task created inside the slot would inherit it and run concurrently with its parent —
the isolation the single slot exists to provide (mutation control M3).
"""

from __future__ import annotations

import asyncio

import pytest

from millm.core.errors import ModelBusyError
from millm.ml.model_loader import LoadedModelState
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer


@pytest.fixture
def service():
    svc = make_service(word_model(), word_tokenizer())
    yield svc
    LoadedModelState().cancel_unload()
    clear_loaded()


async def test_a_nested_admit_in_the_owner_task_re_enters(service):
    queue = service.request_queue
    async with service._admit(background=True):
        assert queue.background_holding_count == 1
        # Would deadlock without re-entry; the timeout turns a deadlock into a failure.
        # ⚠ `asyncio.timeout`, NOT `asyncio.wait_for`: on Python 3.11 (the image) wait_for wraps
        # its coroutine in a NEW task, which is not the slot's owner and would queue behind it.
        async with asyncio.timeout(1.0):
            async with service._admit():
                pending_inside = queue.pending_count
        assert pending_inside == 0, "re-entry must take no second slot and count nothing"
        assert queue.background_holding_count == 1
    assert queue.occupied_count == 0


async def test_re_entry_is_also_what_an_interactive_holder_gets(service):
    async with service._admit():
        async with asyncio.timeout(1.0):
            async with service._admit():
                assert service.request_queue.pending_count == 1


async def test_a_child_task_created_inside_the_slot_queues(service):
    """M3's target: the child inherits `_SLOT_OWNER` but is a different task, so it must wait."""
    queue = service.request_queue
    entered = asyncio.Event()

    async def child():
        async with service._admit():
            entered.set()

    async with service._admit(background=True):
        task = asyncio.create_task(child())
        for _ in range(10):
            await asyncio.sleep(0)
        assert not entered.is_set(), "a child task ran inside its parent's slot"
        assert queue.pending_count == 1, "the child must be queued as an interactive request"
    await asyncio.wait_for(task, timeout=1.0)
    assert entered.is_set()


async def test_re_entry_still_refuses_while_the_model_unloads(service):
    state = LoadedModelState()
    async with service._admit(background=True):
        state.begin_unload()
        try:
            with pytest.raises(ModelBusyError):
                async with service._admit():
                    pass
        finally:
            state.cancel_unload()


async def test_the_owner_is_cleared_when_the_slot_is_released(service):
    from millm.services.inference_service import _SLOT_OWNER

    async with service._admit(background=True):
        assert _SLOT_OWNER.get() is asyncio.current_task()
    assert _SLOT_OWNER.get() is None
