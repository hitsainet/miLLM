"""GGUF generation leaves embedding mode for its duration (suite tracker 1.9).

THE DEFECT: miLLM loads every GGUF model with `embedding=True` + MEAN pooling so one instance can
answer `/v1/embeddings`. A context in that mode does not produce usable next-token logits:
`LFM2.5-1.2B-Instruct-GGUF:Q4_K_M` answered every chat 200 with EMPTY content and non-zero
completion tokens. Reproduced on CPU with llama-cpp-python 0.3.36 and the real file (2026-10-16):
`embedding=False` -> "Hello! How can I help?"; `embedding=True, pooling=MEAN` -> ''. Toggling
`llama_set_embeddings(ctx, False)` around the generation -> correct text, and embeddings computed
afterwards bit-identical to a fresh embedding-only instance (cosine 1.0).

The doubles below behave like that: `_FakeLlama` returns empty content when asked to generate
while its context is in embedding mode. A path that forgets to switch modes therefore reproduces
the production symptom here, not just a missing call.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _FakeLlamaCpp(types.ModuleType):
    """`llama_cpp.llama_set_embeddings`, recording into the context it is given."""

    def __init__(self):
        super().__init__("llama_cpp")
        self.calls: list[bool] = []

    def llama_set_embeddings(self, ctx, enabled):
        self.calls.append(bool(enabled))
        ctx.embeddings = bool(enabled)


class _FakeLlama:
    def __init__(self, embedding: bool):
        self.context_params = SimpleNamespace(embeddings=embedding)
        self._ctx = SimpleNamespace(ctx=SimpleNamespace(embeddings=embedding))
        self.metadata = {}

    def _text(self, text: str) -> str:
        return "" if self._ctx.ctx.embeddings else text  # the production symptom

    def create_chat_completion(self, messages, stream=False, **params):
        content = self._text("Hello! How can I help?")
        if stream:
            def gen():
                for piece in ([content] if content else []):
                    yield {"choices": [{"index": 0, "delta": {"content": piece},
                                        "finish_reason": None}]}
                yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            return gen()
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}}

    def create_completion(self, prompt, stream=False, **params):
        return {"choices": [{"index": 0, "text": self._text(" world"), "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def tokenize(self, data):
        return [1, 2, 3]


@pytest.fixture
def llama_cpp(monkeypatch):
    fake = _FakeLlamaCpp()
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    return fake


def _service(model):
    from millm.services.inference_service import InferenceService

    svc = InferenceService.__new__(InferenceService)
    svc._cbm_backend = None
    state = MagicMock()
    state.is_loaded = True
    state.current.model = model
    state.current.model_name = "lfm-gguf"
    svc._model_state = state
    return svc


class TestTheHelper:
    def test_switches_off_for_the_block_and_back_on(self, llama_cpp):
        from millm.ml.llamacpp_modes import generation_mode

        model = _FakeLlama(embedding=True)
        with generation_mode(model):
            assert model._ctx.ctx.embeddings is False
        assert model._ctx.ctx.embeddings is True
        assert llama_cpp.calls == [False, True]

    def test_restores_even_when_the_generation_raises(self, llama_cpp):
        from millm.ml.llamacpp_modes import generation_mode

        model = _FakeLlama(embedding=True)
        with pytest.raises(RuntimeError), generation_mode(model):
            raise RuntimeError("decode failed")
        assert model._ctx.ctx.embeddings is True

    def test_a_model_built_without_embeddings_is_left_alone(self, llama_cpp):
        from millm.ml.llamacpp_modes import generation_mode

        with generation_mode(_FakeLlama(embedding=False)):
            pass
        with generation_mode(MagicMock()):  # a MagicMock's truthy attribute is not opt-in
            pass
        assert llama_cpp.calls == []


class TestEveryGenerationPathUsesIt:
    def test_blocking_chat_has_content(self, llama_cpp):
        model = _FakeLlama(embedding=True)
        out = _service(model)._llamacpp_sync([{"role": "user", "content": "hi"}], {})
        assert out["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert model._ctx.ctx.embeddings is True  # /v1/embeddings still works afterwards

    def test_continuation_has_content(self, llama_cpp):
        model = _FakeLlama(embedding=True)
        svc = _service(model)
        svc._llamacpp_continuation_prompt = lambda messages: "PROMPT"
        out = svc._llamacpp_sync([{"role": "user", "content": "hi"}], {})
        assert out["choices"][0]["message"]["content"] == " world"

    @pytest.mark.asyncio
    async def test_streaming_chat_has_content_and_restores_after_close(self, llama_cpp):
        from tests.unit.services.test_gguf_refusals import _NullQueue, _stream_request

        model = _FakeLlama(embedding=True)
        svc = _service(model)
        svc._request_queue = _NullQueue()
        seen_during = []
        original = model.create_chat_completion

        def spy(*a, **k):
            seen_during.append(model._ctx.ctx.embeddings)
            return original(*a, **k)

        model.create_chat_completion = spy
        # Restored IN the `finally`, after the stream is closed and before the slot is released
        # — not later, when the garbage collector happens to finalise the frame.
        at_record = []
        original_record = svc._probe_record

        async def record_spy(*a, **k):
            at_record.append(model._ctx.ctx.embeddings)
            return await original_record(*a, **k)

        svc._probe_record = record_spy
        chunks = [c async for c in svc._llamacpp_stream_chat_completion(_stream_request())]
        assert any("Hello! How can I help?" in c for c in chunks), chunks
        assert seen_during == [False]
        assert at_record == [True]
        assert model._ctx.ctx.embeddings is True

    @pytest.mark.asyncio
    async def test_text_completion_has_content(self, llama_cpp):
        from millm.api.schemas.openai import TextCompletionRequest
        from tests.unit.services.test_gguf_refusals import _NullQueue

        model = _FakeLlama(embedding=True)
        svc = _service(model)
        svc._request_queue = _NullQueue()
        out = await svc._llamacpp_text_completion(
            TextCompletionRequest(model="lfm-gguf", prompt="hello", max_tokens=4)
        )
        assert out.choices[0].text == " world"
        assert model._ctx.ctx.embeddings is True
