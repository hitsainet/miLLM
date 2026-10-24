"""A stop sequence inside an open `<think>` block must not stream the reasoning as content.

⚠ PRE-EXISTING DEFECT, FOUND AT THE ENH-005 HANDOFF (036 FTDD §4.2) AND FIXED IN FEATURE 36.
`stream_chat_completion` checks stop sequences on the RAW token text. When one is found it yields
the text before it (`remaining`) as a `content` delta directly — WITHOUT passing it through the
reasoning splitter every other token goes through. So a stop sequence that lands while a think
block is open sent the model's private reasoning to the client as its answer (and, since Feature
36, would have recorded it as the turn's reply).

The fix feeds `remaining` through `_splitter.feed` like any token. Reproduced red on the old line
before the fix (recorded in `0xcc/reviews/036_037_controls_2026-10-21.md`).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage
from tests.unit.services.test_inference_service import (  # noqa: F401 - fixtures
    loaded_model_state,
    mock_model,
    mock_tokenizer,
    reset_loaded_model_state,
    service,
)

SECRET = "secretplan"


def _deltas(chunks: list[str]) -> tuple[str, str]:
    content, reasoning = [], []
    for raw in chunks:
        body = raw.removeprefix("data: ").strip()
        if body == "[DONE]":
            continue
        data = json.loads(body)
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
    return "".join(content), "".join(reasoning)


async def _stream(svc, tokens: list[str]) -> list[str]:
    streamer = MagicMock()
    streamer.__iter__ = MagicMock(return_value=iter(tokens))
    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hi")],
        stop=["<stop>"],
        stream=True,
    )
    with patch("transformers.TextIteratorStreamer", return_value=streamer):
        return [chunk async for chunk in svc.stream_chat_completion(request)]


@pytest.mark.asyncio
async def test_a_stop_sequence_inside_think_does_not_stream_reasoning_as_content(service):  # noqa: F811 - the imported fixture
    chunks = await _stream(service, ["<think>", f"{SECRET}<stop> trailing"])
    content, reasoning = _deltas(chunks)
    assert SECRET not in content, f"reasoning streamed as the answer: {content!r}"
    assert SECRET in reasoning


@pytest.mark.asyncio
async def test_a_stop_sequence_after_the_think_block_keeps_the_answer(service):  # noqa: F811 - the imported fixture
    """The control: the tail after `</think>` is the answer and still streams as content."""
    chunks = await _stream(service, ["<think>", "plan", "</think>", "the answer<stop> x"])
    content, reasoning = _deltas(chunks)
    assert content == "the answer"
    assert "plan" in reasoning and "answer" not in reasoning
