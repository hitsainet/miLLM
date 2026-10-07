"""`/v1/embeddings` service behaviour with a TINY REAL Llama (Feature 30, FR-30.2, FR-30.3).

The tokenizer is real and its `model_max_length` is 8, as is the model's
`max_position_embeddings`. The suite's mock tokenizer cannot truncate, so it agreed with the
silent-truncation defect by construction (FPRD §12): only a real tokenizer proves `truncation=False`.

Forward passes are counted with a forward pre-hook on the real model, so "measured before any
forward" is asserted as a call count of zero, not inferred.

MUTATION CONTROLS (0xcc/reviews/030_implementation_controls_2026-10-07.md):
  M1  `truncation=True` in `_embed_inputs`            -> the 12-token refusal
  M2  the length check moved inside the per-input loop -> the [ok, long, ok, long] test
  M3  the length-check call deleted                    -> the 12-token refusal
  M10 a fixed "mean" passed instead of options.pooling -> the spy payload test
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from millm.api.schemas.openai import EmbeddingRequest
from millm.core.errors import EmbeddingInputTooLongError, EmbeddingVectorInvalidError
from millm.ml import embedding_pooling
from millm.ml.model_loader import LoadedModel, LoadedModelState
from millm.services import inference_service as inference_module
from millm.services.inference_service import InferenceService

WORDS = ["<s>", "[UNK]"] + [f"w{i}" for i in range(30)]
LIMIT = 8


def _tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 0)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", unk_token="[UNK]", model_max_length=LIMIT
    )
    fast.pad_token = "[UNK]"
    return fast


def _text(tokens: int) -> str:
    """A string the tokenizer turns into exactly `tokens` ids (BOS included)."""
    return " ".join(f"w{i % 30}" for i in range(tokens - 1))


@pytest.fixture(autouse=True)
def _clean_state():
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None


@pytest.fixture
def model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=LIMIT)
    return LlamaForCausalLM(config).eval()


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    return _tokenizer()


@pytest.fixture
def forwards(model) -> list[int]:
    calls: list[int] = []
    model.register_forward_pre_hook(lambda _module, _args: calls.append(1))
    return calls


@pytest.fixture
def service(model, tokenizer) -> InferenceService:
    LoadedModelState().set(LoadedModel(
        model_id=1, model_name="tiny", model=model, tokenizer=tokenizer,
        loaded_at=datetime(2026, 10, 7), memory_used_mb=1, num_parameters=1, device="cpu",
        dtype="float32",
    ))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        svc = InferenceService(model_service=None)
    svc._device = "cpu"
    return svc


def _manual(model, tokenizer, text: str, mode: str) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**ids, output_hidden_states=True).hidden_states[-1][0]
    return {"mean": hidden.mean(dim=0), "cls": hidden[0], "last": hidden[-1]}[mode]


class TestNoSilentTruncation:
    async def test_a_twelve_token_string_is_refused_and_nothing_runs(self, service, forwards):
        with pytest.raises(EmbeddingInputTooLongError) as caught:
            await service.create_embeddings(EmbeddingRequest(model="tiny", input=_text(12)))
        exc = caught.value
        assert exc.code == "CONTEXT_LENGTH_EXCEEDED"
        assert exc.details["param"] == "input", "a string input is named `input`, not input[0]"
        assert exc.details["over_limit"] == [{"index": 0, "tokens": 12}]
        assert "Input 0 has 12 tokens" in exc.message and f"limit is {LIMIT} tokens" in exc.message
        assert "never truncated" in exc.message
        assert forwards == []

    async def test_every_over_limit_index_is_named_before_any_forward(self, service, forwards):
        request = EmbeddingRequest(
            model="tiny", input=[_text(3), _text(12), _text(4), _text(9)]
        )
        with pytest.raises(EmbeddingInputTooLongError) as caught:
            await service.create_embeddings(request)
        exc = caught.value
        assert exc.details["param"] == "input[1]", "param names the FIRST over-limit index"
        assert [o["index"] for o in exc.details["over_limit"]] == [1, 3]
        assert "input 1 has 12 tokens" in exc.message.lower()
        assert "input 3 has 9 tokens" in exc.message
        assert forwards == [], "input 0 fits and must not have been embedded first"

    async def test_input_zero_fits_input_one_over(self, service, forwards):
        with pytest.raises(EmbeddingInputTooLongError) as caught:
            await service.create_embeddings(
                EmbeddingRequest(model="tiny", input=[_text(LIMIT), _text(LIMIT + 1)])
            )
        assert caught.value.details["param"] == "input[1]"
        assert forwards == []

    async def test_the_list_is_bounded_and_the_rest_counted(self, service, forwards):
        with pytest.raises(EmbeddingInputTooLongError) as caught:
            await service.create_embeddings(
                EmbeddingRequest(model="tiny", input=[_text(10)] * 20)
            )
        exc = caught.value
        assert len(exc.details["over_limit"]) == 16
        assert exc.details["omitted"] == 4
        assert "(4 more over the limit not listed)" in exc.message
        assert "input 15 has" in exc.message and "input 16 has" not in exc.message
        assert forwards == []

    async def test_at_the_limit_is_served(self, service, forwards):
        result = await service.create_embeddings(EmbeddingRequest(model="tiny", input=_text(LIMIT)))
        assert len(result.data) == 1 and forwards == [1]

    async def test_with_no_stated_limit_a_long_input_runs_whole(self, service, forwards):
        """FTDD R3: a model whose limit is unknown is served untruncated and unchecked."""
        with patch.object(inference_module, "_served_max_context", return_value=None):
            result = await service.create_embeddings(EmbeddingRequest(model="tiny", input=_text(12)))
        assert result.usage.prompt_tokens == 12, "the full input was embedded, not 8 tokens of it"
        assert forwards == [1]

    async def test_a_refusal_logs_indices_and_never_text(self, service, caplog):
        secret = _text(12).replace("w1 ", "w29 ")
        caplog.set_level(logging.WARNING)
        with patch.object(inference_module.logger, "warning") as warn, \
             pytest.raises(EmbeddingInputTooLongError):
            await service.create_embeddings(EmbeddingRequest(model="tiny", input=["w2", secret]))
        refused = [c for c in warn.call_args_list if c.args and c.args[0] == "embedding_refused"]
        assert len(refused) == 1
        kwargs = refused[0].kwargs
        assert kwargs["reason"] == "input_too_long" and kwargs["indices"] == [1]
        logged = repr(refused[0]) + caplog.text
        assert secret not in logged and "w29" not in logged


class TestPooling:
    @pytest.mark.parametrize("mode", ["mean", "last", "cls"])
    async def test_each_mode_matches_a_manual_pool(self, service, model, tokenizer, mode):
        texts = [_text(3), _text(7)]
        result = await service.create_embeddings(
            EmbeddingRequest(model="tiny", input=texts, pooling=mode)
        )
        for i, text in enumerate(texts):
            expected = _manual(model, tokenizer, text, mode)
            assert torch.allclose(torch.tensor(result.data[i].embedding), expected, atol=1e-6)

    async def test_the_default_is_todays_vector_exactly(self, service, model, tokenizer):
        """FR-30.2.2: no new field → `hidden_states[-1].mean(dim=1).squeeze().cpu().tolist()`."""
        text = _text(6)
        result = await service.create_embeddings(EmbeddingRequest(model="tiny", input=text))
        ids = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**ids, output_hidden_states=True).hidden_states[-1]
        assert result.data[0].embedding == hidden.mean(dim=1).squeeze().cpu().tolist()

    async def test_last_with_normalize_returns_unit_vectors(self, service):
        """US-1, BRD-04 acceptance 13."""
        result = await service.create_embeddings(
            EmbeddingRequest(model="tiny", input=[_text(2), _text(5), _text(8)],
                             pooling="last", normalize=True)
        )
        for item in result.data:
            assert abs(math.sqrt(sum(x * x for x in item.embedding)) - 1.0) < 1e-5

    async def test_the_request_options_reach_the_pooling_calls(self, service):
        """Payload AND count: one pool and one finalize per input, with the request's values."""
        with patch.object(inference_module, "pool_hidden",
                          wraps=embedding_pooling.pool_hidden) as pool, \
             patch.object(inference_module, "finalize_vector",
                          wraps=embedding_pooling.finalize_vector) as finalize:
            await service.create_embeddings(
                EmbeddingRequest(model="tiny", input=[_text(3), _text(4), _text(5)],
                                 pooling="cls", normalize=True)
            )
        assert pool.call_count == 3
        assert [c.args[2] for c in pool.call_args_list] == ["cls"] * 3
        assert finalize.call_count == 3
        assert [c.args[1] for c in finalize.call_args_list] == [True] * 3

    async def test_usage_is_the_sum_of_full_counts(self, service):
        result = await service.create_embeddings(
            EmbeddingRequest(model="tiny", input=[_text(3), _text(8), _text(5)])
        )
        assert result.usage.prompt_tokens == 16 and result.usage.total_tokens == 16

    async def test_base64_carries_the_normalised_vector(self, service):
        import base64
        import struct

        plain = await service.create_embeddings(
            EmbeddingRequest(model="tiny", input=_text(4), normalize=True)
        )
        encoded = await service.create_embeddings(
            EmbeddingRequest(model="tiny", input=_text(4), normalize=True, encoding_format="base64")
        )
        raw = base64.b64decode(encoded.data[0].embedding)
        decoded = struct.unpack(f"<{len(raw) // 4}f", raw)
        assert decoded == pytest.approx(plain.data[0].embedding, abs=1e-7)

    async def test_a_non_finite_vector_is_a_500_naming_its_index(self, service):
        real = embedding_pooling.pool_hidden

        def poisoned(hidden, mask, mode):
            out = real(hidden, mask, mode)
            if hidden.shape[1] == 4:
                out = out * float("nan")
            return out

        with patch.object(inference_module, "pool_hidden", side_effect=poisoned), \
             pytest.raises(EmbeddingVectorInvalidError) as caught:
            await service.create_embeddings(
                EmbeddingRequest(model="tiny", input=[_text(3), _text(4)])
            )
        assert caught.value.status_code == 500
        assert caught.value.details["param"] == "input[1]"
        assert "Input 1" in caught.value.message


class TestEmbeddingsOpenNoProbeContext:
    """FR-30.2.9: no probe context is opened, so an armed probe records nothing."""

    async def test_no_probe_context_is_opened(self, service, model):
        from millm.services.probe_runtime import ProbeRuntimeState

        seen: list[object] = []
        model.register_forward_pre_hook(
            lambda _module, _args: seen.append(ProbeRuntimeState().current_request())
        )
        with patch.object(ProbeRuntimeState, "begin_request") as begin, \
             patch.object(InferenceService, "_probe_begin") as probe_begin, \
             patch.object(InferenceService, "_probe_begin_detached") as detached:
            await service.create_embeddings(
                EmbeddingRequest(model="tiny", input=[_text(3), _text(4)], pooling="last")
            )
        assert begin.call_count == 0
        assert probe_begin.call_count == 0 and detached.call_count == 0
        assert seen == [None, None], "a forward ran inside an open probe context"


class TestEmbedInputsIsTheSharedEntryPoint:
    """FR-30.2.10: Feature 26's executor calls `_embed_inputs` directly (it holds the slot)."""

    def test_embed_inputs_returns_vectors_and_counts(self, service, model, tokenizer):
        vectors, counts = service._embed_inputs(
            [_text(3), _text(5)], embedding_pooling.EmbeddingOptions("last", False)
        )
        assert counts == [3, 5]
        assert torch.allclose(torch.tensor(vectors[1]),
                              _manual(model, tokenizer, _text(5), "last"), atol=1e-6)

    async def test_create_embeddings_calls_it_once_inside_a_slot(self, service):
        entered: list[bool] = []
        real = service._embed_inputs

        def spy(texts, options, **kwargs):
            entered.append(service._request_queue.holding_count == 1)
            return real(texts, options, **kwargs)

        with patch.object(service, "_embed_inputs", side_effect=spy) as embed:
            await service.create_embeddings(
                EmbeddingRequest(model="tiny", input=["w1", "w2 w3"], pooling="cls")
            )
        assert embed.call_count == 1
        assert entered == [True], "the body ran without holding an admission slot"
        texts, options = embed.call_args.args
        assert texts == ["w1", "w2 w3"]
        assert options == embedding_pooling.EmbeddingOptions("cls", False)
        assert embed.call_args.kwargs == {"param_for_string": False}
