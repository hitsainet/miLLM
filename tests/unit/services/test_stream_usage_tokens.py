"""F-1 (038 window, 2026-10-29): a streamed response's `usage.completion_tokens` and `finish_reason`
must agree with the non-streamed response to the same request.

The live finding: Llama-3.1-8B, temperature 0, `max_tokens 64`, identical text both ways —
non-streamed `64` / `length`, streamed `58` / `stop`. The streamed path counted
`TextIteratorStreamer` TEXT CHUNKS (about a word each), not tokens, so `_determine_finish_reason`
never saw the limit.

The fake `generate` below does what transformers does: it appends one token per step, runs every
stopping criterion on the whole sequence after each step, and hands the streamer TEXT — one chunk
per TWO tokens here, and nothing at all for the EOS token (skipped as special), so a chunk count
is visibly wrong. Control (findings_1029_controls.md): count chunks again → these go red.
"""

from __future__ import annotations

import json
import queue
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage
from tests.unit.services.test_inference_service import (  # noqa: F401 - fixtures
    loaded_model_state,
    mock_model,
    mock_tokenizer,
    reset_loaded_model_state,
    service,
)
from tests.unit.services.test_llamacpp_generation_mode import (  # noqa: F401 - fixture
    _service,
    llama_cpp,
)

PROMPT = [1, 2, 3, 4, 5]
EOS = 2  # mock_tokenizer.eos_token_id


class _FakeStreamer:
    """`TextIteratorStreamer`'s consumer side: text pushed by `generate`, read by the server."""

    def __init__(self, *_a: Any, **_k: Any) -> None:
        self.q: "queue.Queue[Optional[str]]" = queue.Queue()

    def push(self, text: str) -> None:
        self.q.put(text)

    def end(self) -> None:
        self.q.put(None)

    def on_finalize(self, *_a: Any) -> None:
        self.end()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self.q.get(timeout=10)
        if item is None:
            raise StopIteration
        return item


def _fake_generate(n_tokens: int, *, eos: bool):
    """Generates `n_tokens` (the last one EOS when `eos`), honouring `max_new_tokens`."""

    def generate(**kwargs: Any) -> torch.Tensor:
        ids = kwargs["input_ids"].clone()
        limit = int(kwargs.get("max_new_tokens") or n_tokens)
        streamer = kwargs.get("streamer")
        criteria = list(kwargs.get("stopping_criteria") or [])
        steps = min(n_tokens, limit)
        pending = 0
        for i in range(steps):
            last = eos and i == n_tokens - 1
            token = EOS if last else 100 + i
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
            stop = False
            for criterion in criteria:
                stop = bool(criterion(ids, None)) or stop
            if streamer is not None and not last:
                pending += 1
                if pending == 2:  # a "word": two tokens, one text chunk
                    streamer.push("w ")
                    pending = 0
            if stop:
                break
        if streamer is not None:
            if pending:
                streamer.push("w ")
            streamer.end()
        return ids

    return generate


def _request(stream: bool, max_tokens: int) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content="Hi")],
        stream=stream, max_tokens=max_tokens, temperature=0,
    )


def _final(chunks: list[str]) -> dict[str, Any]:
    """The chunk carrying `finish_reason` and `usage`."""
    for raw in chunks:
        body = raw.removeprefix("data: ").strip()
        if body == "[DONE]" or not body:
            continue
        data = json.loads(body)
        for choice in data.get("choices") or []:
            if choice.get("finish_reason"):
                return {"finish_reason": choice["finish_reason"], "usage": data.get("usage")}
    raise AssertionError("no final chunk")


async def _both(service, mock_model, max_tokens: int, n_tokens: int, eos: bool):  # noqa: F811
    mock_model.generate = MagicMock(side_effect=_fake_generate(n_tokens, eos=eos))
    plain = await service.create_chat_completion(_request(False, max_tokens))
    with patch("transformers.TextIteratorStreamer", _FakeStreamer):
        chunks = [c async for c in service.stream_chat_completion(_request(True, max_tokens))]
    return plain, _final(chunks)


@pytest.mark.asyncio
async def test_hit_max_tokens_streamed_agrees_with_not_streamed(service, mock_model):  # noqa: F811
    """The live case: the model runs to `max_tokens`. Both must say `length` and the limit."""
    plain, streamed = await _both(service, mock_model, max_tokens=64, n_tokens=200, eos=False)
    assert plain.usage.completion_tokens == 64
    assert plain.choices[0].finish_reason == "length"
    # 32 text chunks for 64 tokens: a chunk count reported 32 / `stop` here.
    assert streamed["usage"]["completion_tokens"] == 64
    assert streamed["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_eos_stop_streamed_agrees_with_not_streamed(service, mock_model):  # noqa: F811
    """EOS before the limit: `stop`, and the same count (the EOS token included, as the
    non-streamed path counts the generated ids)."""
    plain, streamed = await _both(service, mock_model, max_tokens=64, n_tokens=9, eos=True)
    assert plain.choices[0].finish_reason == "stop"
    assert streamed["finish_reason"] == "stop"
    assert streamed["usage"]["completion_tokens"] == plain.usage.completion_tokens == 9


@pytest.mark.asyncio
async def test_usage_is_counted_without_any_monitoring_armed(service, mock_model):  # noqa: F811
    """⚠ The id capture was installed only for sensing, probes or activations; the count must not
    depend on them (nothing is armed in this test)."""
    _plain, streamed = await _both(service, mock_model, max_tokens=10, n_tokens=50, eos=False)
    assert streamed["usage"]["completion_tokens"] == 10
    assert streamed["finish_reason"] == "length"


# ── the other engines (F-1's sweep) ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cbm_stream_counts_every_generated_id(service, mock_tokenizer):  # noqa: F811
    """CBM counted only ids whose decoded text was non-empty: an id that decodes to nothing (EOS
    with special tokens skipped, half a character) went uncounted, and a run to the limit read
    `stop`. Four ids, one silent, `max_tokens` 4: 4 / `length`."""
    backend = MagicMock()
    backend.is_running = True

    async def gen(input_ids, max_new_tokens, request_id):
        for ids in ([10], [11], [12], [13]):
            yield ids

    backend.generate_stream = gen
    service._cbm_backend = backend
    mock_tokenizer.encode = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    mock_tokenizer.decode = MagicMock(side_effect=["Hel", "", "lo", "!"])
    chunks = [c async for c in service._cbm_stream_chat_completion(_request(True, 4))]
    final = _final(chunks)
    assert final["finish_reason"] == "length"
    # CBM's final chunk carries no usage; the count is what finish_reason was derived from.


class _WordLlama:
    """llama.cpp's stream shape: TEXT deltas, several tokens to a delta (it holds text back
    while it might be a stop sequence). One token per space-separated word here."""

    def __init__(self, deltas: list[str], finish: str):
        from types import SimpleNamespace

        self.deltas, self.finish = deltas, finish
        self.context_params = SimpleNamespace(embeddings=False)
        self._ctx = SimpleNamespace(ctx=SimpleNamespace(embeddings=False))
        self.metadata = {}

    def create_chat_completion(self, messages, stream=False, **params):
        def gen():
            for piece in self.deltas:
                yield {"choices": [{"index": 0, "delta": {"content": piece},
                                    "finish_reason": None}]}
            yield {"choices": [{"index": 0, "delta": {}, "finish_reason": self.finish}]}
        return gen()

    def tokenize(self, data, add_bos=True, special=False):
        return [1] * len(data.decode("utf-8").split())


@pytest.mark.asyncio
@pytest.mark.parametrize("deltas, finish, max_tokens, expected", [
    (["one two ", "three four ", "five"], "stop", 64, 5),
    # `length` is counted the same way: llama.cpp also says `length` when the context fills.
    (["one two ", "three"], "length", 3, 3),
], ids=["eos-tokenized", "length-tokenized"])
async def test_llamacpp_stream_counts_tokens_not_deltas(llama_cpp, deltas, finish,  # noqa: F811
                                                        max_tokens, expected):
    from tests.unit.services.test_gguf_refusals import _NullQueue, _stream_request

    svc = _service(_WordLlama(deltas, finish))
    svc._request_queue = _NullQueue()
    chunks = [c async for c in svc._llamacpp_stream_chat_completion(
        _stream_request(max_tokens=max_tokens))]
    final = _final(chunks)
    assert final["finish_reason"] == finish
    assert final["usage"]["completion_tokens"] == expected
