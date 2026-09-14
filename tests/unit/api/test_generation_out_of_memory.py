"""A CUDA out-of-memory error during generation is a typed refusal naming the card, and the server keeps serving.

Review round 6, 2026-09-14. The per-card fit keeps each card room for a KV cache at
TRANSFORMERS_MIN_CONTEXT, but a longer prompt, a batch or another tenant on the card
can still run generation out of memory. What a client got:

  * non-streaming (/v1/chat/completions, /v1/completions): torch.OutOfMemoryError
    reached the generic handler — a bare 500 "An internal server error occurred.",
    typed server_error, which an OpenAI client retries unchanged;
  * streaming: NOTHING. generate() ends its streamer only when it finishes
    (transformers 5.15.1 generation/utils.py:2944, not in a finally), and
    TextIteratorStreamer waits with no timeout. The role chunk went out and the
    stream then hung: no error event, no [DONE], the request queue slot held until
    the client gave up, and one executor thread blocked for good. The existing
    TestStreamThreadErrorPropagation tests could not see it: they replace both
    _generate_in_thread and the streamer, so the error was recorded and the
    iterator ended by construction. PROOF: millm-p2-review6/stream_hang_probe.py,
    the real streamer and a generate() that raises, "HUNG ... within 8 s".

Now (inference_service._generate_sync / _generate_in_thread): the out-of-memory error
becomes GenerationOutOfMemoryError — naming the card torch named, the prompt size and
max_tokens, and what to change — raised only after the failed pass's tensors are
collected and torch's cache is emptied; /v1 answers 503 invalid_request_error
`insufficient_memory` (the same envelope as a stream's error event), the management
API 507. A streaming generation that raises anything is recorded and its streamer
ended, so every stream finishes.

Everything drives the real routes and the real InferenceService; only the model, its
tokenizer and the model service are stand-ins. No management route generates text, so
the 507 rendering is the class's status for the management handler, not exercised here.

MUTATION CONTROLS (millm-p2-review6/mutate.py; each restored, sha256 verified, git diff clean):
  R6-M2  _generate_sync lets torch's error through           -> both non-streaming 503 tests (500)
  R6-M3  _generate_sync never releases the memory            -> both non-streaming tests (cache not emptied)
  R6-M4  the release does not collect before emptying         -> both non-streaming tests and the stream test
         (the failed pass's cache still alive when emptied)
  R6-M5  the thread does not end the streamer on failure      -> both stream tests (the deadline fires)
  R6-M6  the streamer ended BEFORE the error is recorded      -> both stream tests (a stream with no error event)
  R6-M7  a stream's OOM answered with the generic event       -> test_a_stream_that_runs_out_of_memory_...
  R6-M8  the /v1 handler ignores the error's own type         -> both non-streaming tests (server_error)
  R6-M9  a generate() call outside the two mapped helpers     -> test_every_generate_call_goes_through_...
"""

from __future__ import annotations

import ast
import inspect
import json
import threading
import time
import weakref
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

pytest.importorskip("transformers")
import transformers  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from millm.db.models.model import ModelStatus  # noqa: E402
from millm.main import create_app  # noqa: E402
from millm.ml.model_loader import LoadedModel, LoadedModelState  # noqa: E402
from millm.services import inference_service  # noqa: E402
from millm.services.inference_service import InferenceService  # noqa: E402
from tests.support.factories import make_model  # noqa: E402

#: torch 2.10's message for an allocation that failed on the 3090 (index 1).
OOM = (
    "CUDA out of memory. Tried to allocate 1.46 GiB. GPU 1 has a total capacity of 23.56 GiB "
    "of which 612.00 MiB is free. Including non-PyTorch memory, this process has 22.95 GiB "
    "memory in use."
)
CHAT = {"model": "wanted", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}
COMPLETION = {"model": "wanted", "prompt": "hi", "max_tokens": 64}
PROMPT_IDS = torch.tensor([[1, 2, 3, 4, 5]])
STREAM_DEADLINE_S = 20


@pytest.fixture(autouse=True)
def _no_model_left_behind():
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None


class _Generate:
    """model.generate: runs out of memory on the calls named, as a failed pass does.

    The failing call allocates a stand-in for its partial KV cache and leaves it in a
    reference cycle, so only the garbage collector frees it — tracebacks and frames
    form such cycles, which is why the release collects before it empties the cache.
    """

    def __init__(self, fail_on=(1,), error=None):
        self.calls = 0
        self.fail_on = set(fail_on)
        self.error = error
        self.cache_refs: list[weakref.ref] = []

    def __call__(self, **kwargs):
        self.calls += 1
        if self.calls in self.fail_on:
            cache = torch.empty(4096)
            holder = {"kv": cache}
            holder["self"] = holder
            self.cache_refs.append(weakref.ref(cache))
            del cache
            raise self.error if self.error is not None else torch.OutOfMemoryError(OOM)
        return torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12]])


def _tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.chat_template = None
    encoded = MagicMock()
    encoded.input_ids = PROMPT_IDS
    encoded.__getitem__ = lambda self, key: {"input_ids": PROMPT_IDS, "attention_mask": torch.ones_like(PROMPT_IDS)}[key]
    encoded.items = MagicMock(return_value=[("input_ids", PROMPT_IDS), ("attention_mask", torch.ones_like(PROMPT_IDS))])
    encoded.to = MagicMock(return_value=encoded)
    tokenizer.return_value = encoded
    tokenizer.decode = MagicMock(return_value="Hello, world!")
    return tokenizer


def _app(generate: _Generate):
    from millm.api.dependencies import get_inference_service, get_model_service

    model = MagicMock()
    model.config = MagicMock()
    model.config.max_position_embeddings = 4_096
    model.config.is_encoder_decoder = False
    model.device = "cpu"
    model.generate = MagicMock(side_effect=generate)
    LoadedModelState().set(
        LoadedModel(
            model_id=3, model_name="wanted", model=model, tokenizer=_tokenizer(),
            loaded_at=datetime(2026, 9, 14), memory_used_mb=14_000, num_parameters=7_000_000_000,
            device="cpu", dtype="float16",
        )
    )
    with patch("millm.services.inference_service.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        inference = InferenceService(model_service=None)
    inference._device = "cpu"
    # The rung echo reads the circuits table; no circuit steers here, and a unit test
    # must not reach for a database.
    inference.active_circuit_rung = AsyncMock(return_value=None)

    svc = MagicMock()
    svc.find_model_by_name = AsyncMock(return_value=make_model(id=3, name="wanted", status=ModelStatus.LOADED))
    svc.get_locked_model = AsyncMock(return_value=None)
    svc.load_model_and_wait = AsyncMock()
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_inference_service] = lambda: inference
    return TestClient(app), svc


def _record_empty_cache(generate: _Generate):
    """Each torch.cuda.empty_cache call, as whether every failed pass's cache was already freed."""
    seen: list[list[bool]] = []

    def empty_cache():
        seen.append([ref() is None for ref in generate.cache_refs])

    return seen, patch("torch.cuda.empty_cache", side_effect=empty_cache)


class _TrackedStreamer(transformers.TextIteratorStreamer):
    """The real streamer. end() lingers after waking the consumer, so an error recorded
    only AFTER end() is recorded too late for the consumer to see."""

    made: list["_TrackedStreamer"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _TrackedStreamer.made.append(self)

    def end(self):
        super().end()
        time.sleep(0.3)


def _post_with_deadline(client: TestClient, body: dict):
    """POST a stream, failing (not hanging the suite) when it never ends."""
    _TrackedStreamer.made.clear()
    result: dict = {}
    worker = threading.Thread(
        target=lambda: result.setdefault("response", client.post("/v1/chat/completions", json=body)),
        daemon=True,
    )
    worker.start()
    worker.join(STREAM_DEADLINE_S)
    if worker.is_alive():
        for streamer in _TrackedStreamer.made:
            streamer.on_finalized_text("", stream_end=True)  # release the blocked consumer
        worker.join(10)
        pytest.fail(f"the stream did not end within {STREAM_DEADLINE_S} s: no error event and no [DONE]")
    return result["response"]


def _events(response) -> list[str]:
    return [part.removeprefix("data: ") for part in response.text.split("\n\n") if part.startswith("data: ")]


@pytest.mark.parametrize("path, body", [("/v1/chat/completions", CHAT), ("/v1/completions", COMPLETION)])
def test_out_of_memory_is_a_503_naming_the_card_and_the_next_request_is_served(path, body):
    generate = _Generate(fail_on={1})
    client, svc = _app(generate)
    seen, empty_cache = _record_empty_cache(generate)

    with empty_cache:
        first = client.post(path, json=body)

    assert svc.load_model_and_wait.await_count == 0, "the model named is resident: no load"
    assert first.status_code == 503, first.text
    error = first.json()["error"]
    assert (error["type"], error["code"]) == ("invalid_request_error", "insufficient_memory")
    assert "ran out of memory on cuda:1" in error["message"]
    assert "5 prompt tokens, up to 64 new tokens" in error["message"]
    assert "shorter prompt" in error["message"]
    assert seen == [[True]], "the cache is emptied once, after the failed pass's tensors are freed"

    second = client.post(path, json=body)

    assert second.status_code == 200, second.text
    assert generate.calls == 2


def test_a_stream_that_runs_out_of_memory_ends_with_the_error_and_done():
    generate = _Generate(fail_on={1})
    client, _ = _app(generate)
    seen, empty_cache = _record_empty_cache(generate)

    with empty_cache, patch("transformers.TextIteratorStreamer", _TrackedStreamer):
        response = _post_with_deadline(client, {**CHAT, "stream": True})

    assert response.status_code == 200
    events = _events(response)
    assert events[-1] == "[DONE]"
    error = json.loads(events[-2])["error"]
    assert (error["type"], error["code"]) == ("invalid_request_error", "insufficient_memory")
    assert "ran out of memory on cuda:1" in error["message"]
    assert seen == [[True]]
    # The queue slot came back: the next request is served.
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_a_stream_whose_generation_raises_anything_else_still_ends():
    generate = _Generate(fail_on={1}, error=RuntimeError("Expected all tensors to be on the same device"))
    client, _ = _app(generate)

    with patch("transformers.TextIteratorStreamer", _TrackedStreamer):
        response = _post_with_deadline(client, {**CHAT, "stream": True})

    events = _events(response)
    assert events[-1] == "[DONE]"
    error = json.loads(events[-2])["error"]
    assert (error["type"], error["code"]) == ("server_error", "generation_error")
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_every_generate_call_goes_through_the_out_of_memory_mapping():
    """A new `self._model.generate(...)` call elsewhere would bypass the mapping."""
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
                isinstance(target, ast.Attribute) and target.attr == "generate"
                and isinstance(target.value, ast.Attribute) and target.value.attr == "_model"
                and isinstance(target.value.value, ast.Name) and target.value.value.id == "self"
            ):
                callers.append(self.stack[-1] if self.stack else "<module>")
            self.generic_visit(node)

    _Visitor().visit(tree)

    assert sorted(callers) == ["_generate_in_thread", "_generate_sync"]
