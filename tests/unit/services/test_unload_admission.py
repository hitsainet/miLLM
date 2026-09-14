"""A request that arrives while its model is being unloaded is told to retry; nothing runs on a moving model.

Hardware acceptance, 2026-09-14, item 11 (0xcc/reviews/multi_gpu_phase2_acceptance_2026-09-14.md).
ModelService.unload_model kept the row LOADED, and the loader reporting the model, until its
worker returned, while LoadedModelState.clear() moved the weights `.to("cpu")` first. A /v1
chat 1-3 s into the unload of a split OLMo-2-13B passed the route's "is that model loaded?"
check, was dispatched into generate() and failed in embed_tokens: HTTP 500, "Expected all
tensors to be on the same device, but got index is on cuda:0, different from other tensors
on cpu". Round 4's retry answer (f821eb7) covered only the moment after the loader was
cleared.

Now the unload marks the model (LoadedModelState.begin_unload) BEFORE anything moves, and
InferenceService._admit refuses work on a marked model — before it queues, and again once
it holds its slot, because an unload can begin while a request waits. The unload drains the
requests already admitted (RequestQueue.wait_idle, up to GRACEFUL_UNLOAD_TIMEOUT, which was
documented for this and read by nothing) before its worker moves a weight. A refused request
gets 503 server_error `model_busy`, the same answer as a request that catches a load in
progress. An unload stopped before its worker starts takes the mark back.

Everything runs through the real ModelService, InferenceService, ModelLoader and
LoadedModelState and the real routes. The model is a stand-in whose `.to("cpu")` signals and
waits, and whose generate() fails the way accelerate does once its weights have moved. The
interleaving is driven by events, never by sleeping.

PROVED FIRST against 2c84fe8 (a scratch worktree, one test at a time): the request during the
unload answered HTTP 500 server_error, "An internal server error occurred." — the node's
symptom; a request for another model started a second unload; the drain had no idle event to
wait on and continuous batching had no mark to read; the admission guard found nine direct
callers of the queue's acquire. (Only test_a_model_loaded_after_an_unload_serves passed: it
guards the mark's reset, and that code had no mark.)

MUTATION CONTROLS (millm-p2-accept-fix/mutate.py, run against this file and
test_openai_load_refusal_status.py; each restored, sha256 verified, git diff unchanged):
  AF3-M1  unload_model does not mark the model before it moves it   -> 4 red, the request
          during the move, the queued request, the late request, the other model
  AF3-M2  _admit does not check again once it holds the slot        -> the queued request
  AF3-M3  _admit does not check before it queues                     -> the late request (it
          waited behind the generation in flight)
  AF3-M4  the drain waits 0 s instead of GRACEFUL_UNLOAD_TIMEOUT     -> SURVIVED at first: the
          drain test looked at the weights when the drain began, before a drain that gives up
          could move them. The test now records what the drain waited for and whether it found
          the queue idle only after the generation finished; re-run, red.
  AF3-M5  a cancelled unload does not take the mark back             -> the stopped unload
  AF3-M6  MODEL_BUSY has no /v1 row (409 instead of 503)             -> 3 red
  AF3-M7  check_stream_admission does not refuse an unloading model  -> the request during the move
  AF3-M8  load_model starts a second unload                          -> the other model
  AF3-M9  unload_model accepts a second unload of the same model     -> the other model
  AF3-M10 set() leaves the mark set on the next model                -> the model loaded after
  AF3-M11 continuous batching takes a request for an unloading model -> its test
  AF3-M12 embeddings take a slot outside _admit                      -> the admission guard
  AF3-M13 the serial stream raises the refusal after its 200         -> its stream test
  AF3-M14 the llama.cpp stream raises the refusal after its 200      -> its stream test
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import torch

pytest.importorskip("transformers")
from transformers import BatchEncoding, Qwen2Config  # noqa: E402

from millm.api.schemas.openai import ChatCompletionRequest  # noqa: E402
from millm.core.errors import ModelBusyError  # noqa: E402
from millm.db.models.model import ModelStatus  # noqa: E402
from millm.main import create_app  # noqa: E402
from millm.ml.model_loader import LoadedModel, LoadedModelState, ModelLoader  # noqa: E402
from millm.services import inference_service  # noqa: E402
from millm.services.inference_service import InferenceService  # noqa: E402
from millm.services.model_service import ModelService  # noqa: E402
from tests.support.factories import make_model  # noqa: E402

CHAT = {"model": "wanted", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
DEVICE_MISMATCH = (
    "Expected all tensors to be on the same device, but got index is on cuda:0, "
    "different from other tensors on cpu"
)
WAIT_S = 10


@pytest.fixture(autouse=True)
def _no_model_left_behind():
    state = LoadedModelState()
    state._loaded = None
    state._unloading = False
    yield
    state._loaded = None
    state._unloading = False


class _Weights:
    """The loaded model. `.to("cpu")` says it started and waits to be let go; generate()
    fails on moved weights as accelerate's hooks do, and can be held mid-generation."""

    def __init__(self):
        self.config = Qwen2Config(max_position_embeddings=4_096, num_hidden_layers=2)
        self.device = "cpu"
        self.generation_config = None
        self.moving = threading.Event()
        self.let_move = threading.Event()
        self.moved = False
        self.hold_generation = False
        self.generating = threading.Event()
        self.let_generate = threading.Event()
        self.generate_calls = 0
        self.generated_while_moving = False

    def to(self, device):
        self.moving.set()
        assert self.let_move.wait(WAIT_S), "the test never let the weights move"
        self.moved = True
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1
        if self.moving.is_set() or self.moved:
            # Part of the model is already on the CPU: accelerate's hooks meet a
            # cuda:0 index and CPU weights, as on the node.
            self.generated_while_moving = True
            raise RuntimeError(DEVICE_MISMATCH)
        if self.hold_generation:
            self.generating.set()
            assert self.let_generate.wait(WAIT_S), "the test never let generation finish"
        generated = torch.tensor([[7, 8, 9]])
        streamer = kwargs.get("streamer")
        if streamer is not None:  # generate() feeds and ends its streamer when it finishes
            streamer.put(generated[0])
            streamer.end()
        return torch.cat([kwargs["input_ids"], generated], dim=-1)


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


def _load(weights: _Weights, model_id: int = 3, name: str = "wanted") -> None:
    LoadedModelState().set(
        LoadedModel(
            model_id=model_id, model_name=name, model=weights, tokenizer=_tokenizer(),
            loaded_at=datetime(2026, 9, 14), memory_used_mb=1_000, num_parameters=1_000_000,
            device="cpu", dtype="bfloat16",
        )
    )


def _stack(weights: _Weights):
    """The real services over one loaded model (id 3, "wanted") and a READY one (id 4, "other")."""
    from millm.api.dependencies import get_inference_service, get_model_service

    _load(weights)
    with patch("millm.services.inference_service.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        inference = InferenceService(model_service=None)
    inference._device = "cpu"
    inference.active_circuit_rung = AsyncMock(return_value=None)

    rows = {
        3: make_model(id=3, name="wanted", status=ModelStatus.LOADED),
        4: make_model(id=4, name="other", status=ModelStatus.READY),
    }
    repo = MagicMock()
    repo.get_by_id = AsyncMock(side_effect=lambda model_id: rows.get(model_id))
    repo.find_by_name = AsyncMock(
        side_effect=lambda name: next((row for row in rows.values() if row.name == name), None)
    )
    repo.get_locked_model = AsyncMock(return_value=None)
    repo.update = AsyncMock(side_effect=lambda model_id, **fields: rows[model_id])
    repo.update_status = AsyncMock(side_effect=lambda model_id, **fields: rows[model_id])
    svc = ModelService(
        repository=repo, downloader=MagicMock(), loader=ModelLoader(), emitter=None,
        inference_service=inference,
    )
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_inference_service] = lambda: inference
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    )
    return inference, svc, repo, client


def _watch_the_drain(
    inference: InferenceService, record: list | None = None, after=lambda: None
) -> asyncio.Event:
    """An event set when the unload starts waiting for admitted requests.

    Each wait appends (the timeout the drain was given, whether it found the queue idle,
    `after()` when it returned) to `record`.
    """
    draining = asyncio.Event()
    queue = inference.request_queue
    original = queue.wait_idle

    async def watched(timeout=None):
        draining.set()
        idle = await original(timeout)
        if record is not None:
            record.append((timeout, idle, after()))
        return idle

    queue.wait_idle = watched
    return draining


async def _turns_until(predicate, turns: int = 100_000) -> None:
    """Hand the event loop over until `predicate` holds: no clock involved."""
    for _ in range(turns):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("the condition never held")


def _assert_busy(response) -> None:
    assert response.status_code == 503, response.text
    assert response.headers["content-type"].startswith("application/json"), response.headers
    error = response.json()["error"]
    assert (error["type"], error["code"]) == ("server_error", "model_busy")
    assert "retry" in error["message"].lower()


async def test_a_request_while_the_weights_move_is_told_to_retry_and_nothing_runs_on_them():
    weights = _Weights()
    inference, svc, repo, client = _stack(weights)
    async with client:
        unload = asyncio.create_task(svc.unload_model(3))
        assert await asyncio.to_thread(weights.moving.wait, WAIT_S), "the unload never moved a weight"

        response = await client.post("/v1/chat/completions", json=CHAT)
        streamed = await client.post("/v1/chat/completions", json={**CHAT, "stream": True})

        weights.let_move.set()
        await asyncio.wait_for(unload, WAIT_S)

    _assert_busy(response)
    _assert_busy(streamed)
    assert "data:" not in streamed.text
    assert weights.generate_calls == 0
    repo.update.assert_awaited_with(3, status=ModelStatus.READY, loaded_at=None, locked=False)


async def test_the_unload_waits_for_the_request_in_flight_before_any_weight_moves():
    from millm.core.config import settings

    weights = _Weights()
    weights.hold_generation = True
    inference, svc, _, client = _stack(weights)
    drains: list = []
    draining = _watch_the_drain(inference, drains, after=weights.let_generate.is_set)
    async with client:
        in_flight = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
        assert await asyncio.to_thread(weights.generating.wait, WAIT_S)

        unload = asyncio.create_task(svc.unload_model(3))
        await asyncio.wait_for(draining.wait(), WAIT_S)
        assert not weights.moving.is_set(), "a weight moved while a request was generating"

        weights.let_generate.set()
        response = await asyncio.wait_for(in_flight, WAIT_S)
        assert await asyncio.to_thread(weights.moving.wait, WAIT_S), "the unload never went on"
        weights.let_move.set()
        await asyncio.wait_for(unload, WAIT_S)

    assert response.status_code == 200, response.text
    assert not weights.generated_while_moving
    assert drains == [(settings.GRACEFUL_UNLOAD_TIMEOUT, True, True)], (
        "the drain waits up to GRACEFUL_UNLOAD_TIMEOUT and finds the queue idle only after the "
        "generation in flight was let finish"
    )


async def test_a_request_waiting_for_a_slot_when_the_unload_begins_is_refused_once_it_gets_one():
    weights = _Weights()
    weights.hold_generation = True
    inference, svc, _, client = _stack(weights)
    draining = _watch_the_drain(inference)
    async with client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
        assert await asyncio.to_thread(weights.generating.wait, WAIT_S)
        second = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
        await _turns_until(lambda: inference.request_queue.pending_count == 2)

        unload = asyncio.create_task(svc.unload_model(3))
        await asyncio.wait_for(draining.wait(), WAIT_S)
        weights.let_generate.set()
        first_response = await asyncio.wait_for(first, WAIT_S)
        second_response = await asyncio.wait_for(second, WAIT_S)
        weights.let_move.set()
        await asyncio.wait_for(unload, WAIT_S)

    assert first_response.status_code == 200, first_response.text
    _assert_busy(second_response)
    assert weights.generate_calls == 1


async def test_a_request_arriving_during_the_drain_is_refused_without_waiting_behind_it():
    weights = _Weights()
    weights.hold_generation = True
    inference, svc, _, client = _stack(weights)
    draining = _watch_the_drain(inference)
    async with client:
        in_flight = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
        assert await asyncio.to_thread(weights.generating.wait, WAIT_S)
        unload = asyncio.create_task(svc.unload_model(3))
        await asyncio.wait_for(draining.wait(), WAIT_S)

        # Generation is still held: a request that queued behind it would never answer.
        late = await asyncio.wait_for(client.post("/v1/chat/completions", json=CHAT), WAIT_S)

        weights.let_generate.set()
        await asyncio.wait_for(in_flight, WAIT_S)
        weights.let_move.set()
        await asyncio.wait_for(unload, WAIT_S)

    _assert_busy(late)
    assert weights.generate_calls == 1


async def test_a_request_for_another_model_during_the_unload_does_not_start_a_second_one():
    weights = _Weights()
    inference, svc, repo, client = _stack(weights)
    async with client:
        unload = asyncio.create_task(svc.unload_model(3))
        assert await asyncio.to_thread(weights.moving.wait, WAIT_S)

        response = await client.post("/v1/chat/completions", json={**CHAT, "model": "other"})
        with pytest.raises(ModelBusyError, match="already being unloaded"):
            await asyncio.wait_for(svc.unload_model(3), WAIT_S)

        weights.let_move.set()
        await asyncio.wait_for(unload, WAIT_S)

    _assert_busy(response)
    assert "being unloaded" in response.json()["error"]["message"]
    assert all(call.args[:1] != (4,) for call in repo.update_status.await_args_list), (
        "the other model's row was touched"
    )
    assert svc._loading_model_id is None, "the refusal must not hold the load slot"


async def test_an_unload_stopped_while_draining_leaves_the_model_serving():
    weights = _Weights()
    weights.hold_generation = True
    inference, svc, _, client = _stack(weights)
    draining = _watch_the_drain(inference)
    async with client:
        in_flight = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
        assert await asyncio.to_thread(weights.generating.wait, WAIT_S)
        unload = asyncio.create_task(svc.unload_model(3))
        await asyncio.wait_for(draining.wait(), WAIT_S)

        unload.cancel()
        with pytest.raises(asyncio.CancelledError):
            await unload
        weights.let_generate.set()
        await asyncio.wait_for(in_flight, WAIT_S)
        weights.hold_generation = False
        after = await asyncio.wait_for(client.post("/v1/chat/completions", json=CHAT), WAIT_S)

    assert after.status_code == 200, after.text
    assert not weights.moving.is_set()


async def test_a_model_loaded_after_an_unload_serves():
    """The mark belongs to the model it was set on: a model set afterwards is not being unloaded."""
    weights = _Weights()
    inference, svc, _, client = _stack(weights)
    async with client:
        unload = asyncio.create_task(svc.unload_model(3))
        assert await asyncio.to_thread(weights.moving.wait, WAIT_S)
        weights.let_move.set()
        await asyncio.wait_for(unload, WAIT_S)
        assert not LoadedModelState().is_loaded
        LoadedModelState()._unloading = True  # the mark as the unload left it, before the next load

        _load(_Weights())
        response = await client.post("/v1/chat/completions", json=CHAT)

    assert response.status_code == 200, response.text


async def test_continuous_batching_does_not_take_a_request_for_a_model_being_unloaded():
    """The manager holds no queue slot to refuse; the request goes to the serial path, which refuses."""
    weights = _Weights()
    inference, _, _, _ = _stack(weights)
    backend = MagicMock()
    backend.is_running = True
    backend.sampling_params_match = MagicMock(return_value=True)
    backend.generate = AsyncMock(return_value=([7, 8, 9], "stop"))
    inference._cbm_backend = backend
    LoadedModelState().begin_unload()

    with pytest.raises(ModelBusyError, match="being unloaded"):
        await inference.create_chat_completion(ChatCompletionRequest(**CHAT))

    assert backend.generate.await_count == 0
    assert weights.generate_calls == 0


async def test_a_stream_whose_model_starts_unloading_after_the_route_ends_with_the_refusal_and_done():
    """The route refuses first (check_stream_admission); an unload that begins after the 200 is
    committed must still end the stream, with the refusal and [DONE], and run nothing."""
    weights = _Weights()
    inference, _, _, _ = _stack(weights)
    LoadedModelState().begin_unload()

    events = [chunk async for chunk in inference.stream_chat_completion(ChatCompletionRequest(**{**CHAT, "stream": True}))]

    assert events[-1] == "data: [DONE]\n\n"
    error = __import__("json").loads(events[-2].removeprefix("data: "))["error"]
    assert (error["type"], error["code"]) == ("server_error", "model_busy")
    assert len(events) == 2
    assert weights.generate_calls == 0
    assert inference.request_queue.pending_count == 0


async def test_a_llamacpp_stream_whose_model_starts_unloading_ends_with_the_refusal_and_done():
    weights = _Weights()
    weights.create_chat_completion = MagicMock()
    weights.create_completion = MagicMock()
    inference, _, _, _ = _stack(weights)
    inference._engine_is_llamacpp = lambda: True
    LoadedModelState().begin_unload()

    events = [chunk async for chunk in inference.stream_chat_completion(ChatCompletionRequest(**{**CHAT, "stream": True}))]

    assert events[-1] == "data: [DONE]\n\n"
    error = __import__("json").loads(events[-2].removeprefix("data: "))["error"]
    assert (error["type"], error["code"]) == ("server_error", "model_busy")
    assert weights.create_chat_completion.call_count == 0
    assert weights.create_completion.call_count == 0


def test_every_request_queue_slot_is_taken_through_admission():
    """A new `self._request_queue.acquire()` elsewhere would skip the unloading check.

    The idle cache release takes a slot too, so it never runs during a request; it
    does no work on the model."""
    tree = ast.parse(inspect.getsource(inference_service))
    callers: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def _function(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = visit_AsyncFunctionDef = _function

        def visit_Call(self, node):
            target = node.func
            if (
                isinstance(target, ast.Attribute) and target.attr == "acquire"
                and isinstance(target.value, ast.Attribute) and target.value.attr == "_request_queue"
            ):
                callers.append(self.stack[-1] if self.stack else "<module>")
            self.generic_visit(node)

    _Visitor().visit(tree)

    assert sorted(callers) == ["_admit", "_release_idle_cache"]
