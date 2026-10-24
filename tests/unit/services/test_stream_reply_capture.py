"""Each streaming path records, as the turn's reply, exactly the content the client received —
reasoning excluded (036 FTDD §10 "Reply capture tests").

`accumulated_text` is NOT usable for this (it is filled only when stop sequences are set and holds
the raw stream, think block included), so each path collects its splitter's visible output. A spy
on `_probe_record` reads the `reply_text` each path passes, and compares it with the deltas.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage
from tests.unit.services.test_gguf_refusals import _NullQueue, _stream_request
from tests.unit.services.test_inference_service import (  # noqa: F401 - fixtures
    loaded_model_state,
    mock_model,
    mock_tokenizer,
    reset_loaded_model_state,
    service,
)
from tests.unit.services.test_llamacpp_generation_mode import (  # noqa: F401 - fixture
    _FakeLlama,
    _service,
    llama_cpp,
)


def _content(chunks: list[str]) -> str:
    out = []
    for raw in chunks:
        body = raw.removeprefix("data: ").strip()
        if body == "[DONE]":
            continue
        for choice in json.loads(body).get("choices") or []:
            out.append((choice.get("delta") or {}).get("content") or "")
    return "".join(out)


def _spy(svc) -> list[dict]:
    seen: list[dict] = []
    original = svc._probe_record

    async def record(*a, **k):
        seen.append(k)
        return await original(*a, **k)

    svc._probe_record = record
    return seen


def _request(**over):
    return ChatCompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content="Hi")], stream=True, **over
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens, reply", [
    (["<think>", "plan", "</think>", "Hel", "lo"], "Hello"),
    # A trailing "<" is withheld as a possible tag and released only by the final flush; the
    # flush is part of what the client receives, so it is part of the reply.
    (["Hel", "lo", "<"], "Hello<"),
], ids=["reasoning-excluded", "flush-included"])
async def test_hf_stream_records_the_visible_reply(service, tokens, reply):  # noqa: F811
    seen = _spy(service)
    streamer = MagicMock()
    streamer.__iter__ = MagicMock(return_value=iter(tokens))
    with patch("transformers.TextIteratorStreamer", return_value=streamer):
        chunks = [c async for c in service.stream_chat_completion(_request())]
    assert [k["reply_text"] for k in seen] == [_content(chunks)] == [reply]


@pytest.mark.asyncio
async def test_cbm_stream_records_the_visible_reply(service, mock_tokenizer):  # noqa: F811
    backend = MagicMock()
    backend.is_running = True

    async def gen(input_ids, max_new_tokens, request_id):
        for ids in ([10], [11], [12]):
            yield ids

    backend.generate_stream = gen
    service._cbm_backend = backend
    mock_tokenizer.encode = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    mock_tokenizer.decode = MagicMock(side_effect=["<think>plan", "</think>Hello", " world"])
    seen = _spy(service)
    chunks = [c async for c in service._cbm_stream_chat_completion(_request())]
    assert [k["reply_text"] for k in seen] == [_content(chunks)] == ["Hello world"]


class _Thinking(_FakeLlama):
    def _text(self, text: str) -> str:
        return "<think>private plan</think>The answer."


@pytest.mark.asyncio
async def test_llamacpp_stream_records_the_visible_reply(llama_cpp):  # noqa: F811
    svc = _service(_Thinking(embedding=False))
    svc._request_queue = _NullQueue()
    seen = _spy(svc)
    chunks = [c async for c in svc._llamacpp_stream_chat_completion(_stream_request())]
    assert [k["reply_text"] for k in seen] == [_content(chunks)] == ["The answer."]
