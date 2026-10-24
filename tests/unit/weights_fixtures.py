"""Shared fixtures for Feature 32 (checkpoints, load, lineage): tiny REAL models on the CPU.

Real transformers models, never stubs: tie detection reads real storages and real tie
declarations, and the inbound-compatibility rule is about what real `save_pretrained` writes.

* `tiny_llama(tied, dtype, seed)` and `tiny_lfm2(...)` — 2-layer models (FTID §8.1). The two
  architectures differ where it matters: LFM2 is hybrid (a conv layer and an attention layer) and
  carries different buffers. Both declare `lm_head.weight -> model.embed_tokens.weight` as a tie in
  transformers 5.15 whether or not the config ties, which is exactly the trap the untied variants
  exist to catch.
* `tiny_tokenizer()` — a WordLevel tokenizer over the 256-id vocabulary, no network.
* `save_pretrained_dir(model, path)` — writes the directory the way miForge does (transformers'
  own `save_pretrained`, safetensors, tied key dropped, `{"format": "pt"}` metadata).
* `checkpoint_root` — a checkpoint root under `tmp_path`. Patching it into settings waits for
  `WEIGHTS_CHECKPOINT_DIR` (owned by the configuration wave); the service tests add that.

Seeds matter: two models built with different seeds hold different values in the same shapes,
which is the fixture an in-place copy needs to prove it copied anything at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Lfm2Config,
    Lfm2ForCausalLM,
    PreTrainedTokenizerFast,
)

VOCAB_SIZE = 256


def _seeded(seed: int):
    torch.manual_seed(seed)


def tiny_llama(
    tied: bool = True, dtype: torch.dtype = torch.float32, seed: int = 0, vocab_size: int = VOCAB_SIZE
) -> LlamaForCausalLM:
    _seeded(seed)
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        tie_word_embeddings=tied,
    )
    return LlamaForCausalLM(config).to(dtype).eval()


def tiny_lfm2(
    tied: bool = True, dtype: torch.dtype = torch.float32, seed: int = 0, vocab_size: int = VOCAB_SIZE
) -> Lfm2ForCausalLM:
    _seeded(seed)
    config = Lfm2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        tie_word_embeddings=tied,
        layer_types=["conv", "full_attention"],
    )
    return Lfm2ForCausalLM(config).to(dtype).eval()


BUILDERS = {"llama": tiny_llama, "lfm2": tiny_lfm2}


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    words = ["<unk>", "<s>", "</s>"] + [f"w{i}" for i in range(VOCAB_SIZE - 3)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )


def save_pretrained_dir(model, path: Path, *, tokenizer=None) -> Path:
    """Write `path` exactly as transformers (and therefore miForge) writes a checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def checkpoint_root(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoints"
    root.mkdir()
    return root
