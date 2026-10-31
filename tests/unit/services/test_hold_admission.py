"""`InferenceService.hold_admission()`: the weight barrier's hold on THE admission slot (031 §7.2,
033 review finding 17.1, addendum §2.1 "W0b").

With one slot (D-04) the holder has the model to itself: it waits for in-flight work, nothing is
admitted while it holds, and the slot comes back on exit — including when the block raises. It
takes the slot through `_admit()` as an INTERACTIVE waiter (never `background`), and runs no model
work itself.

MUTATION CONTROLS (recorded in the W0b report):
  C5  `hold_admission` yields without entering `_admit()`   -> waits / blocks tests
  C6  `_admit(background=True)` in `hold_admission`         -> the interactive-waiter test
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from millm.core.errors import ModelBusyError
from millm.ml.model_loader import LoadedModelState
from millm.services.inference_service import InferenceService
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer


@pytest.fixture
def svc():
    service = make_service(word_model(), word_tokenizer())
    yield service
    LoadedModelState()._unloading = False
    clear_loaded()


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_it_waits_for_in_flight_work(svc):
    release = asyncio.Event()
    entered = asyncio.Event()

    async def in_flight():
        async with svc._admit():
            await release.wait()

    async def holder():
        async with svc.hold_admission():
            entered.set()

    work = asyncio.create_task(in_flight())
    await _settle()
    hold = asyncio.create_task(holder())
    await _settle()
    assert not entered.is_set(), "the hold must wait for the request already holding the slot"
    release.set()
    await asyncio.wait_for(hold, 2)
    await work
    assert entered.is_set()


async def test_nothing_is_admitted_while_it_is_held(svc):
    admitted = asyncio.Event()

    async def request():
        async with svc._admit():
            admitted.set()

    async def background_chunk():
        async with svc._admit(background=True):
            admitted.set()

    async def model_work():
        await svc.run_model_work(lambda: admitted.set())

    async with svc.hold_admission():
        tasks = [asyncio.create_task(f()) for f in (request, background_chunk, model_work)]
        await _settle()
        assert not admitted.is_set(), "work was admitted while the barrier held the slot"
    await asyncio.wait_for(asyncio.gather(*tasks), 5)
    assert admitted.is_set()


async def test_the_slot_is_released_when_the_block_raises(svc):
    with pytest.raises(RuntimeError, match="write failed"):
        async with svc.hold_admission():
            raise RuntimeError("write failed")

    async def request():
        async with svc._admit():
            return "ran"

    assert await asyncio.wait_for(request(), 2) == "ran"
    assert svc.request_queue.occupied_count == 0


async def test_it_queues_as_an_interactive_waiter_ahead_of_background_work(svc):
    """An interactive waiter goes before a waiting batch chunk (026 FTDD §7): the barrier must
    not sit behind a batch."""
    order: list[str] = []
    release = asyncio.Event()

    async def in_flight():
        async with svc._admit():
            await release.wait()

    async def chunk():
        async with svc._admit(background=True):
            order.append("chunk")

    async def holder():
        async with svc.hold_admission():
            order.append("hold")

    work = asyncio.create_task(in_flight())
    await _settle()
    c = asyncio.create_task(chunk())
    await _settle()
    h = asyncio.create_task(holder())
    await _settle()
    release.set()
    await asyncio.wait_for(asyncio.gather(work, c, h), 5)
    assert order == ["hold", "chunk"], order


async def test_it_refuses_while_the_model_unloads(svc):
    LoadedModelState()._unloading = True
    with pytest.raises(ModelBusyError):
        async with svc.hold_admission():
            pytest.fail("the hold must not be granted on a model being unloaded")


async def test_it_runs_no_model_work(svc):
    calls = []
    svc._model.register_forward_pre_hook(lambda *a: calls.append(1))
    async with svc.hold_admission():
        pass
    assert calls == []


def test_it_is_the_only_admission_api_the_barrier_needs():
    """The barrier's seam exists on the class (a live attribute, not a module import)."""
    assert callable(getattr(InferenceService, "hold_admission", None))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        service = InferenceService(model_service=None)
    assert hasattr(service.hold_admission(), "__aenter__")
