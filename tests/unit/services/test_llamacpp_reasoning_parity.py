"""A GGUF reply splits its `<think>` block the same way streamed and not (pre-existing defect, 036).

`_llamacpp_stream_chat_completion` routes the think block to `reasoning_content` through
`StreamingReasoningSplitter(False)`; `_llamacpp_chat_completion` put the raw text — reasoning
included — in `content`. Found while wiring Feature 36's reply capture (the turn's reply must be
what the client reads as the answer, on every path). The non-streaming path now applies
`split_reasoning(text, False)`, the same rule. Reverting it turns the first test red.
"""

from __future__ import annotations

import json

import pytest

from tests.unit.services.test_gguf_refusals import _NullQueue, _stream_request
from tests.unit.services.test_llamacpp_generation_mode import (  # noqa: F401 - fixture
    _FakeLlama,
    _service,
    llama_cpp,
)

RAW = "<think>private plan</think>The answer."


class _Thinking(_FakeLlama):
    def _text(self, text: str) -> str:
        return RAW


@pytest.mark.asyncio
async def test_non_streaming_gguf_chat_splits_reasoning_like_the_stream(llama_cpp):  # noqa: F811
    from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage

    svc = _service(_Thinking(embedding=False))
    svc._request_queue = _NullQueue()
    out = await svc._llamacpp_chat_completion(ChatCompletionRequest(
        model="lfm-gguf", messages=[ChatMessage(role="user", content="hi")]
    ))
    message = out.choices[0].message
    assert message.content == "The answer."
    assert message.reasoning_content == "private plan"

    stream_svc = _service(_Thinking(embedding=False))
    stream_svc._request_queue = _NullQueue()
    content, reasoning = [], []
    async for chunk in stream_svc._llamacpp_stream_chat_completion(_stream_request()):
        body = chunk.removeprefix("data: ").strip()
        if body == "[DONE]":
            continue
        for choice in json.loads(body).get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
    assert ("".join(content), "".join(reasoning)) == (message.content, message.reasoning_content)
