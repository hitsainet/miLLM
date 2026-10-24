"""Torch's unused cache goes back to the model's cards once they are idle — and never during a request.

Hardware acceptance, 2026-09-14 (0xcc/reviews/multi_gpu_phase2_acceptance_2026-09-14.md, Failure 1,
"Also seen"): after three successful requests on OLMo-2-13B the RTX 3080 Ti sat at 12,004 MiB
used / 155 MiB free until the model was unloaded. torch keeps the blocks a finished request
freed, nvidia-smi counts them as used, and miStudio's placement and every other tenant of the
node read nvidia-smi. Only the out-of-memory path emptied the cache.

Now, once a request leaves the queue idle, InferenceService schedules a release after
TRANSFORMERS_IDLE_CACHE_RELEASE_S (5 s): if no work has been admitted since, it takes the
queue's slot — so it cannot run during a request — and empties torch's cache on the model's
cards. A GGUF model is not released (llama.cpp's memory is not torch's), nor a model on no card,
nor anything while continuous batching runs (its manager generates without a queue slot).

Everything drives the real route and the real InferenceService; torch's allocator calls are
stand-ins, since there is no GPU here.

MUTATION CONTROLS (millm-p2-accept-fix/mutate.py; each restored and its sha256 verified, no backup
left behind; run over this file, test_unload_admission and the six fit and attach files listed in
test_working_memory.py). First run, by the session that wrote the fix (its per-test output was not
kept):
  ACR-M1  _admit never schedules a release                                     -> RED
  ACR-M2  admitted work does not advance the release generation                -> RED
  ACR-M3  the release runs without taking the queue's slot                     -> RED
  ACR-M4  a negative delay schedules a release anyway                          -> RED
  ACR-M5  a llama.cpp model gets a release scheduled                           -> RED
  ACR-M6  the release never calls torch.cuda.empty_cache                       -> RED
  ACR-M7  a release scheduled before later work is not voided by it            -> RED
  ACR-M8  an error while scheduling escapes into the request (except ZeroDivisionError) -> RED
Review of the fix (millm-p2-accept-fix2): continuous batching generates without a queue slot, so
the queue reads idle during a CBM request, and a serial request finishing beside one scheduled a
release that ran during it. Fixed; test_nothing_is_released_while_continuous_batching_runs:
  ACR-M9  scheduling does not check for continuous batching  -> 1 red, that test
  ACR-M10 the release does not check for continuous batching -> 1 red, that test
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import torch

pytest.importorskip("transformers")
from transformers import BatchEncoding, Qwen2Config  # noqa: E402

from millm.core.config import settings  # noqa: E402
from millm.db.models.model import ModelStatus  # noqa: E402
from millm.main import create_app  # noqa: E402
from millm.ml.model_loader import LoadedModel, LoadedModelState  # noqa: E402
from millm.services import inference_service  # noqa: E402
from millm.services.inference_service import InferenceService  # noqa: E402
from tests.support.factories import make_model  # noqa: E402

CHAT = {"model": "wanted", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
WAIT_S = 10


@pytest.fixture(autouse=True)
def _no_model_left_behind():
    state = LoadedModelState()
    state._loaded = None
    state._unloading = False
    yield
    state._loaded = None
    state._unloading = False


class _Model:
    def __init__(self):
        self.config = Qwen2Config(max_position_embeddings=4_096, num_hidden_layers=2)
        self.device = "cpu"
        self.generation_config = None
        self.generating = False
        self.hold = False
        self.in_generate = threading.Event()
        self.let_generate = threading.Event()
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        self.generating = True
        try:
            if self.hold:
                self.in_generate.set()
                assert self.let_generate.wait(WAIT_S)
            return torch.cat([kwargs["input_ids"], torch.tensor([[7, 8, 9]])], dim=-1)
        finally:
            self.generating = False


def _tokenizer() -> MagicMock:
    ids = torch.ones((1, 5), dtype=torch.long)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.chat_template = None
    tokenizer.side_effect = lambda *args, **kwargs: BatchEncoding(
        {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    )
    tokenizer.decode = MagicMock(return_value="ok")
    return tokenizer


def _stack(gpu_indices=(1,), engine="transformers"):
    from millm.api.dependencies import get_inference_service, get_model_service

    model = _Model()
    LoadedModelState().set(
        LoadedModel(
            model_id=3, model_name="wanted", model=model, tokenizer=_tokenizer(),
            loaded_at=datetime(2026, 9, 14), memory_used_mb=1_000, num_parameters=1_000_000,
            device="cuda:1", dtype="bfloat16", gpu_indices=list(gpu_indices), engine=engine,
        )
    )
    with patch("millm.services.inference_service.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        inference = InferenceService(model_service=None)
    inference._get_input_device = lambda: "cpu"  # no GPU here: the model's inputs stay on the CPU
    inference.active_circuit_rung = AsyncMock(return_value=None)
    svc = MagicMock()
    svc.find_model_by_name = AsyncMock(return_value=make_model(id=3, name="wanted", status=ModelStatus.LOADED))
    svc.get_locked_model = AsyncMock(return_value=None)
    svc.load_model_and_wait = AsyncMock()
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_inference_service] = lambda: inference
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    )
    return inference, model, client


class _Releases:
    """_release_cached_gpu_memory: records the cards, whether a generation was running and the queue's slots."""

    def __init__(self, inference: InferenceService, model: _Model, hold: bool = False):
        self.inference, self.model, self.hold = inference, model, hold
        self.calls: list[tuple[list[int], bool, int]] = []
        self.released = threading.Event()
        self.releasing = threading.Event()
        self.let_release = threading.Event()

    def __call__(self, indices):
        self.calls.append((list(indices), self.model.generating, self.inference.request_queue.pending_count))
        if self.hold:
            self.releasing.set()
            assert self.let_release.wait(WAIT_S)
        self.released.set()
        return {f"cuda:{index}": 100 for index in indices}


async def test_a_finished_request_gives_the_cards_their_cache_back_once_the_queue_is_idle():
    inference, model, client = _stack()
    releases = _Releases(inference, model)
    with patch.object(settings, "TRANSFORMERS_IDLE_CACHE_RELEASE_S", 0.0), \
            patch.object(inference_service, "_release_cached_gpu_memory", releases):
        async with client:
            response = await client.post("/v1/chat/completions", json=CHAT)
            assert await asyncio.to_thread(releases.released.wait, WAIT_S), "no release after the request"

    assert response.status_code == 200, response.text
    assert releases.calls == [([1], False, 1)], "once, on the model's card, with only its own slot taken"


async def test_work_admitted_before_the_release_runs_voids_it():
    inference, model, client = _stack()
    releases = _Releases(inference, model)
    with patch.object(settings, "TRANSFORMERS_IDLE_CACHE_RELEASE_S", -1.0), \
            patch.object(inference_service, "_release_cached_gpu_memory", releases):
        async with client:
            assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200
            scheduled_after_first = inference._idle_release_generation
            assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200

            await inference._release_idle_cache(scheduled_after_first)
            assert releases.calls == [], "a request came after it was scheduled"

            await inference._release_idle_cache(inference._idle_release_generation)

    assert releases.calls == [([1], False, 1)]


async def test_a_request_arriving_during_a_release_waits_for_it():
    inference, model, client = _stack()
    releases = _Releases(inference, model, hold=True)
    with patch.object(settings, "TRANSFORMERS_IDLE_CACHE_RELEASE_S", -1.0), \
            patch.object(inference_service, "_release_cached_gpu_memory", releases):
        async with client:
            release = asyncio.create_task(inference._release_idle_cache(inference._idle_release_generation))
            assert await asyncio.to_thread(releases.releasing.wait, WAIT_S)
            request = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
            await _turns_until(lambda: inference.request_queue.pending_count == 2)

            assert model.calls == 0, "a request generated while the cache was being released"
            releases.let_release.set()
            await asyncio.wait_for(release, WAIT_S)
            response = await asyncio.wait_for(request, WAIT_S)

    assert response.status_code == 200, response.text
    assert model.calls == 1


async def _turns_until(predicate, turns: int = 100_000) -> None:
    for _ in range(turns):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("the condition never held")


@pytest.mark.parametrize(
    "delay, gpu_indices, engine, scheduled",
    [
        (5.0, (1,), "transformers", True),
        (-1.0, (1,), "transformers", False),
        (5.0, (), "transformers", False),
        (5.0, (1,), "llamacpp", False),
    ],
)
async def test_what_is_scheduled(delay, gpu_indices, engine, scheduled):
    inference, _, _ = _stack(gpu_indices=gpu_indices, engine=engine)
    loop = asyncio.get_running_loop()
    with patch.object(settings, "TRANSFORMERS_IDLE_CACHE_RELEASE_S", delay), \
            patch.object(loop, "call_later") as call_later:
        inference._schedule_idle_cache_release()

    assert call_later.call_count == (1 if scheduled else 0)
    if scheduled:
        assert call_later.call_args.args[0] == delay


async def test_a_release_that_cannot_be_scheduled_never_replaces_the_requests_answer():
    """Scheduling runs in _admit's `finally`: an exception there would replace the request's
    own outcome. A setting that is not a number leaves the chat answered and says why."""
    inference, model, client = _stack()
    with patch.object(settings, "TRANSFORMERS_IDLE_CACHE_RELEASE_S", "soon"), \
            patch.object(inference_service, "logger") as logger:
        async with client:
            response = await client.post("/v1/chat/completions", json=CHAT)

    assert response.status_code == 200, response.text
    assert model.calls == 1
    assert [c.args for c in logger.warning.call_args_list] == [("idle_cache_release_not_scheduled",)]


def _cbm_running() -> MagicMock:
    backend = MagicMock()
    backend.is_running = True
    return backend


async def test_nothing_is_released_while_continuous_batching_runs():
    """CBM generates without a queue slot, so the queue reads idle in the middle of a CBM
    request. A serial request (sampling CBM cannot serve) finishing beside it scheduled a
    release that ran during CBM generation. Not scheduled, and a release scheduled before
    CBM started is dropped when it fires."""
    inference, model, client = _stack()
    releases = _Releases(inference, model)
    loop = asyncio.get_running_loop()
    with patch.object(settings, "TRANSFORMERS_IDLE_CACHE_RELEASE_S", 5.0), \
            patch.object(inference_service, "_release_cached_gpu_memory", releases):
        generation = inference._idle_release_generation
        inference._cbm_backend = _cbm_running()
        with patch.object(loop, "call_later") as call_later:
            inference._schedule_idle_cache_release()
        await inference._release_idle_cache(generation)

        assert call_later.call_count == 0, "scheduled while CBM generates outside the queue"
        assert releases.calls == [], "released while CBM generates outside the queue"

        inference._cbm_backend = None
        with patch.object(loop, "call_later") as call_later:
            inference._schedule_idle_cache_release()
        await inference._release_idle_cache(generation)

    assert call_later.call_count == 1
    assert releases.calls == [([1], False, 1)], "the same service releases once CBM is off"


def test_the_release_reports_what_each_card_got_back_and_never_raises():
    reserved = iter([3_000 * 1024 * 1024, 5_000 * 1024 * 1024, 1_000 * 1024 * 1024, 4_500 * 1024 * 1024])
    with patch.object(torch.cuda, "memory_reserved", side_effect=lambda index: next(reserved)), \
            patch.object(torch.cuda, "empty_cache") as empty_cache:
        assert inference_service._release_cached_gpu_memory([0, 1]) == {"cuda:0": 2_000, "cuda:1": 500}
    assert empty_cache.call_count == 1

    with patch.object(torch.cuda, "memory_reserved", side_effect=RuntimeError("no CUDA")), \
            patch.object(inference_service, "logger") as logger:
        assert inference_service._release_cached_gpu_memory([0]) == {}
    assert logger.warning.call_args.args == ("idle_cache_release_failed",)
