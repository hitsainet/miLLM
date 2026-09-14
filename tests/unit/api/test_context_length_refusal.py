"""A request past the model's context is a 400 with the error envelope, on every route, streamed or not.

Hardware acceptance, 2026-09-14, item 7 (0xcc/reviews/multi_gpu_phase2_acceptance_2026-09-14.md).
InferenceService._check_context_length raised a bare ValueError (since de474c63, 2026-02-07)
although ContextLengthExceededError (400) existed, and imported the /v1 helper for it without
using it. On the node:

  * Qwen2.5-7B, 32,699 prompt + 512 max_tokens > 32,768, non-streaming: HTTP 500 server_error;
  * the same, streaming: HTTP 200 with an empty body, no error event and no [DONE];
  * OLMo-2-13B, 4,272 + 64 > 4,096: HTTP 500.

A 500 tells a client to retry a request that can never succeed. Now the check raises
ContextLengthExceededError, which /v1 answers 400 invalid_request_error
`context_length_exceeded` naming the limit and what was asked for. A streamed request is
refused by the route BEFORE the 200 is committed (InferenceService.check_stream_admission);
a refusal the generator still finds (the model changed in between) is an error event and
[DONE]. The continuous-batching paths, which never checked at all, and embeddings, whose
input can exceed the model's positions when the tokenizer does not truncate, check too.

Everything drives the real routes and the real InferenceService; only the model, its
tokenizer and the model service are stand-ins. The model serves 64 tokens.

PROVED FIRST: against the code before the fix (production files stashed), 8 of these tests
failed — every refusal below — and only the at-the-limit request passed.

MUTATION CONTROLS (millm-p2-accept-fix/mutate.py, run against this file,
test_inference_service.py::TestCheckContextLength and test_fit_admitted_context.py; each
restored, sha256 verified, git diff unchanged afterwards):
  AF-M1  _check_context_length raises ValueError again             -> 12 red: every refusal here
         and the three unit tests of the check
  AF-M2  the chat route no longer calls check_stream_admission     -> the streamed 400 test
  AF-M3  the stream's setup re-raises a refusal instead of an event -> the generator test
  AF-M4  embeddings no longer check their input                    -> the embeddings test
  AF-M5  the continuous-batching chat path does not check           -> its chat case
  AF-M6  the continuous-batching completion path does not check     -> its completions case
  AF-M7  the continuous-batching stream does not check              -> its stream test
  AF-M8  check_stream_admission returns without checking            -> the streamed 400 test
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

pytest.importorskip("transformers")
from fastapi.testclient import TestClient  # noqa: E402
from transformers import BatchEncoding, Qwen2Config  # noqa: E402

from millm.api.schemas.openai import ChatCompletionRequest  # noqa: E402
from millm.db.models.model import ModelStatus  # noqa: E402
from millm.main import create_app  # noqa: E402
from millm.ml.model_loader import LoadedModel, LoadedModelState  # noqa: E402
from millm.services.inference_service import InferenceService  # noqa: E402
from tests.support.factories import make_model  # noqa: E402

MAX_CONTEXT = 64
CHAT = {"model": "wanted", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}


@pytest.fixture(autouse=True)
def _no_model_left_behind():
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None


def _tokenizer(tokens: int) -> MagicMock:
    """Every text tokenizes to `tokens` ids, as the real tokenizer's BatchEncoding."""
    ids = torch.ones((1, tokens), dtype=torch.long)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.chat_template = None
    tokenizer.side_effect = lambda *args, **kwargs: BatchEncoding(
        {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    )
    tokenizer.encode = MagicMock(return_value=ids)
    tokenizer.decode = MagicMock(return_value="ok")
    return tokenizer


def _service(prompt_tokens: int) -> tuple[InferenceService, MagicMock]:
    model = MagicMock()
    model.config = Qwen2Config(max_position_embeddings=MAX_CONTEXT, num_hidden_layers=2)
    model.device = "cpu"
    model.generate = MagicMock(
        side_effect=lambda **kwargs: torch.cat(
            [kwargs["input_ids"], torch.tensor([[7, 8, 9]])], dim=-1
        )
    )
    LoadedModelState().set(
        LoadedModel(
            model_id=3, model_name="wanted", model=model, tokenizer=_tokenizer(prompt_tokens),
            loaded_at=datetime(2026, 9, 14), memory_used_mb=1_000, num_parameters=1_000_000,
            device="cpu", dtype="bfloat16",
        )
    )
    with patch("millm.services.inference_service.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        inference = InferenceService(model_service=None)
    inference._device = "cpu"
    # The rung echo reads the circuits table; nothing steers here.
    inference.active_circuit_rung = AsyncMock(return_value=None)
    return inference, model


def _client(inference: InferenceService) -> TestClient:
    from millm.api.dependencies import get_inference_service, get_model_service

    svc = MagicMock()
    svc.find_model_by_name = AsyncMock(
        return_value=make_model(id=3, name="wanted", status=ModelStatus.LOADED)
    )
    svc.get_locked_model = AsyncMock(return_value=None)
    svc.load_model_and_wait = AsyncMock()
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_inference_service] = lambda: inference
    return TestClient(app)


def _assert_refused(response, requested: int, asked: str) -> None:
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json"), response.headers
    error = response.json()["error"]
    assert (error["type"], error["code"]) == ("invalid_request_error", "context_length_exceeded")
    assert f"maximum context length is {MAX_CONTEXT} tokens" in error["message"]
    assert f"you requested {requested} tokens ({asked})" in error["message"]


def test_a_chat_request_past_the_context_is_a_400_and_nothing_is_generated():
    inference, model = _service(prompt_tokens=60)

    response = _client(inference).post("/v1/chat/completions", json=CHAT)

    _assert_refused(response, 68, "60 in the prompt, 8 for the completion")
    assert model.generate.call_count == 0


def test_a_chat_request_at_the_context_is_served():
    inference, model = _service(prompt_tokens=56)

    response = _client(inference).post("/v1/chat/completions", json=CHAT)

    assert response.status_code == 200, response.text
    assert model.generate.call_count == 1


def test_a_streamed_chat_request_past_the_context_is_a_400_before_the_stream_starts():
    inference, model = _service(prompt_tokens=60)

    response = _client(inference).post("/v1/chat/completions", json={**CHAT, "stream": True})

    _assert_refused(response, 68, "60 in the prompt, 8 for the completion")
    assert "data:" not in response.text
    assert model.generate.call_count == 0


async def test_a_stream_whose_generator_finds_the_prompt_too_long_ends_with_the_error_and_done():
    """The generator checks again: the route's check can be passed by a model the request
    no longer runs on. After the 200 the refusal must be an event, and the stream must end."""
    inference, model = _service(prompt_tokens=60)
    request = ChatCompletionRequest(**{**CHAT, "stream": True})

    events = [chunk async for chunk in inference.stream_chat_completion(request)]

    assert events[-1] == "data: [DONE]\n\n"
    error = json.loads(events[-2].removeprefix("data: "))["error"]
    assert (error["type"], error["code"]) == ("invalid_request_error", "context_length_exceeded")
    assert "you requested 68 tokens" in error["message"]
    assert len(events) == 2, "no role chunk: nothing about a completion was sent"
    assert model.generate.call_count == 0
    assert inference.request_queue.pending_count == 0


def test_a_text_completion_past_the_context_is_a_400():
    inference, model = _service(prompt_tokens=60)

    response = _client(inference).post(
        "/v1/completions", json={"model": "wanted", "prompt": "hi", "max_tokens": 8}
    )

    _assert_refused(response, 68, "60 in the prompt, 8 for the completion")
    assert model.generate.call_count == 0


def test_a_batched_chat_request_past_the_context_is_a_400():
    inference, model = _service(prompt_tokens=60)

    response = _client(inference).post(
        "/v1/chat/completions",
        json={**CHAT, "extra_messages": [[{"role": "user", "content": "again"}]]},
    )

    _assert_refused(response, 68, "60 in the prompt, 8 for the completion")
    assert model.generate.call_count == 0


def test_an_embeddings_input_past_the_context_is_a_400_and_the_model_is_not_run():
    inference, model = _service(prompt_tokens=70)

    response = _client(inference).post("/v1/embeddings", json={"model": "wanted", "input": "hi"})

    _assert_refused(response, 70, "70 in the input")
    assert model.call_count == 0


@pytest.mark.parametrize(
    "path, body",
    [
        ("/v1/chat/completions", CHAT),
        ("/v1/completions", {"model": "wanted", "prompt": "hi", "max_tokens": 8}),
    ],
)
def test_continuous_batching_refuses_before_the_manager_is_asked(path, body):
    """The continuous-batching paths never checked the context at all."""
    inference, model = _service(prompt_tokens=60)
    backend = MagicMock()
    backend.is_running = True
    backend.sampling_params_match = MagicMock(return_value=True)
    backend.generate = AsyncMock(return_value=([7, 8, 9], "stop"))
    inference._cbm_backend = backend

    response = _client(inference).post(path, json=body)

    _assert_refused(response, 68, "60 in the prompt, 8 for the completion")
    assert backend.generate.await_count == 0


async def test_a_continuous_batching_stream_that_finds_the_prompt_too_long_ends_with_the_error_and_done():
    inference, _ = _service(prompt_tokens=60)
    backend = MagicMock()
    backend.is_running = True
    backend.sampling_params_match = MagicMock(return_value=True)
    backend.generate_stream = MagicMock()
    inference._cbm_backend = backend
    request = ChatCompletionRequest(**{**CHAT, "stream": True})

    events = [chunk async for chunk in inference.stream_chat_completion(request)]

    assert events[-1] == "data: [DONE]\n\n"
    error = json.loads(events[-2].removeprefix("data: "))["error"]
    assert (error["type"], error["code"]) == ("invalid_request_error", "context_length_exceeded")
    assert len(events) == 2, "no role chunk: nothing about a completion was sent"
    assert not backend.method_calls or all(
        call[0] in ("sampling_params_match",) for call in backend.method_calls
    ), backend.method_calls
