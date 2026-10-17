"""Shared fixtures for Feature 25 tests: TINY REAL transformers, never a mocked model.

A mock returns whatever the test told it to — the "fixture agrees with the code by construction"
trap this estate keeps recording. Every fixture here runs real tokenizers and a real (random)
Llama, as `tests/unit/services/test_scoring_completions.py` does.

* `word_tokenizer(bos, template)` — WordLevel over a small vocabulary. With `bos=True` the
  post-processor prepends `<s>` (so `add_special_tokens=True` adds a second BOS to a rendered
  template, the defect FR-25.5.5 guards). `template=True` installs a chat template that itself
  begins with `<s>`.
* `char_tokenizer()` — one token per character over JSON punctuation, digits, lowercase letters,
  space and newline, with `</s>` as EOS: what constrained-decoding tests need, since a random
  model under an xgrammar mask then emits valid JSON.
* `make_service(model, tokenizer)` — an InferenceService over a LoadedModel on the CPU.
* `make_client(inference, row)` — the real FastAPI app with the model and inference services
  overridden, plus the spies the "before auto-load" assertions read.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from millm.ml.model_loader import LoadedModel, LoadedModelState

WORDS = ["<s>", "[UNK]", "user", "assistant", "system", "</s>"] + [f"w{i}" for i in range(30)]

CHAT_TEMPLATE = (
    "<s>{% for m in messages %} {{ m['role'] }} {{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %} assistant{% endif %}"
)


def word_tokenizer(bos: bool = True, template: bool = True) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if bos:
        tok.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", 0)]
        )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", unk_token="[UNK]", eos_token="</s>"
    )
    fast.pad_token = "[UNK]"
    if template:
        fast.chat_template = CHAT_TEMPLATE
    return fast


def word_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(WORDS), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
        bos_token_id=0, eos_token_id=5, pad_token_id=1,
    )
    return LlamaForCausalLM(config).eval()


CHARS = list('{}[]:,"') + list("0123456789") + list("abcdefghijklmnopqrstuvwxyz") + [
    " ", "\n", ".", "-", "_", "+",
]
CHAR_VOCAB = ["</s>", "<unk>"] + CHARS


def char_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(CHAR_VOCAB)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tok.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="</s>", unk_token="<unk>")
    fast.pad_token = "<unk>"
    fast.chat_template = (
        "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}>{% endif %}"
    )
    return fast


def char_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(CHAR_VOCAB), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
        bos_token_id=0, eos_token_id=0, pad_token_id=1,
    )
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = 0
    model.generation_config.pad_token_id = 1
    return model


def make_service(model: Any, tokenizer: Any, *, name: str = "tiny", dtype: str = "float32"):
    from millm.services.inference_service import InferenceService

    LoadedModelState().set(LoadedModel(
        model_id=1, model_name=name, model=model, tokenizer=tokenizer,
        loaded_at=datetime(2026, 10, 6), memory_used_mb=1, num_parameters=1, device="cpu",
        dtype=dtype,
    ))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        svc = InferenceService(model_service=None)
    svc._device = "cpu"
    return svc


def clear_loaded() -> None:
    LoadedModelState()._loaded = None


def model_row(
    name: str = "tiny", gguf_files: Optional[list] = None, revision: Optional[str] = "abc123",
    quantization: str = "FP16",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1, name=name, architecture="text-generation", gguf_files=gguf_files,
        revision=revision, quantization=SimpleNamespace(value=quantization), gguf_label="",
    )


def make_client(inference: Any, row: Any):
    """The real app; returns (client, model_service_mock)."""
    from fastapi.testclient import TestClient

    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app

    svc = MagicMock()
    svc.find_model_by_name = AsyncMock(return_value=row)
    svc.get_locked_model = AsyncMock(return_value=None)
    svc.load_model_and_wait = AsyncMock()
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_inference_service] = lambda: inference
    return TestClient(app), svc


def unloaded_inference() -> MagicMock:
    """An inference stand-in with nothing resident: any request that reaches it would auto-load
    first, which `load_model_and_wait` records — the "before auto-load" spy."""
    inference = MagicMock()
    inference.backend_name = "serial"
    inference.get_loaded_model_info = lambda: None
    return inference
