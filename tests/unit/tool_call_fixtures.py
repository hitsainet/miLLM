"""Fixtures for function calling (2026-10-28): REAL chat templates over character tokenizers.

The templates are the models' own, byte for byte (`tests/fixtures/chat_templates/`):

* `lfm2.5-1.2b-instruct.jinja` — LiquidAI/LFM2.5-1.2B-Instruct at revision 0f604ada (the revision
  production serves), `chat_template.jinja`;
* `qwen2.5-7b-instruct.jinja`  — Qwen/Qwen2.5-7B-Instruct `tokenizer_config.json`;
* `llama-3.1-8b-instruct.jinja` — meta-llama/Llama-3.1-8B-Instruct, read off the production cache.

The tokenizers are character-level with the families' marker tokens added WITH THE SPECIAL-NESS
THE REAL VOCABULARIES GIVE THEM (read from each `added_tokens_decoder`): LFM2.5's
`<|tool_call_start|>`/`<|tool_call_end|>` and Qwen2.5's `<tool_call>`/`</tool_call>` are NOT
special, so they survive the `skip_special_tokens=True` decode; Llama 3.1's `<|python_tag|>` is.
Each family's BOS behaviour is the real one too: LFM2.5 and Llama 3.1 templates write the BOS text
and their tokenizers add one; Qwen2.5 has no BOS.

`scripted_model` is a tiny REAL Llama whose `generate` is the real one with one extra logits
processor that forces a scripted reply — the streamer, stopping criteria, decode and every miLLM
step around them run for real; only the model's choice of tokens is fixed.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import AddedToken

TEMPLATES = Path(__file__).resolve().parent.parent / "fixtures" / "chat_templates"

#: name -> (template file, bos, eos, special tokens, non-special added tokens, tokenizer adds BOS)
FAMILIES: dict[str, dict[str, Any]] = {
    "lfm2": {
        "template": "lfm2.5-1.2b-instruct.jinja",
        "bos": "<|startoftext|>", "eos": "<|im_end|>",
        "special": ["<|startoftext|>", "<|im_end|>", "<|im_start|>", "<|pad|>"],
        "added": ["<|tool_call_start|>", "<|tool_call_end|>"],
        "adds_bos": True,
    },
    "hermes": {
        "template": "qwen2.5-7b-instruct.jinja",
        "bos": None, "eos": "<|im_end|>",
        "special": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        "added": ["<tool_call>", "</tool_call>"],
        "adds_bos": False,
    },
    "llama3_json": {
        "template": "llama-3.1-8b-instruct.jinja",
        "bos": "<|begin_of_text|>", "eos": "<|eot_id|>",
        "special": ["<|begin_of_text|>", "<|eot_id|>", "<|eom_id|>", "<|start_header_id|>",
                    "<|end_header_id|>", "<|python_tag|>", "<|finetune_right_pad_id|>"],
        "added": [],
        "adds_bos": True,
    },
}

CHARS = [c for c in string.printable if c not in "\x0b\x0c"]


def template_text(family: str) -> str:
    return (TEMPLATES / FAMILIES[family]["template"]).read_text()


def family_tokenizer(family: str) -> PreTrainedTokenizerFast:
    spec = FAMILIES[family]
    vocab = {"<unk>": 0}
    for c in CHARS:
        vocab.setdefault(c, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tok.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")
    fast.add_special_tokens({"additional_special_tokens": [
        AddedToken(t, special=True, normalized=False) for t in spec["special"]
    ]})
    if spec["added"]:
        fast.add_tokens([AddedToken(t, special=False, normalized=False) for t in spec["added"]])
    if spec["bos"]:
        fast.bos_token = spec["bos"]
    fast.eos_token = spec["eos"]
    fast.pad_token = spec["special"][-1]
    if spec["adds_bos"]:
        bos_id = fast.convert_tokens_to_ids(spec["bos"])
        fast._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{spec['bos']} $A", special_tokens=[(spec["bos"], bos_id)]
        )
    fast.chat_template = template_text(family)
    return fast


class _Script(LogitsProcessor):
    """Forces `ids` after the prompt, then EOS."""

    def __init__(self, prompt_len: int, ids: list[int], eos: int) -> None:
        self.prompt_len, self.ids, self.eos = prompt_len, ids, eos

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = input_ids.shape[1] - self.prompt_len
        target = self.ids[step] if step < len(self.ids) else self.eos
        forced = torch.full_like(scores, float("-inf"))
        forced[:, target] = 0.0
        return forced


def scripted_model(tokenizer: PreTrainedTokenizerFast, reply: str, seed: int = 0):
    """A tiny real Llama that answers `reply` (then EOS) to whatever it is asked.

    `generate_calls` records each call's prompt ids (what the model actually ran)."""
    torch.manual_seed(seed)
    eos = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token else eos,
        eos_token_id=eos, pad_token_id=pad,
    )
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = eos
    model.generation_config.pad_token_id = pad
    script = list(tokenizer(reply, add_special_tokens=False)["input_ids"])
    real_generate = model.generate
    model.generate_calls = []  # type: ignore[attr-defined]

    def generate(*args: Any, **kwargs: Any) -> Any:
        ids = kwargs.get("input_ids", args[0] if args else None)
        model.generate_calls.append(ids[0].tolist())  # type: ignore[attr-defined]
        processors_ = LogitsProcessorList(kwargs.pop("logits_processor", None) or [])
        processors_.append(_Script(ids.shape[1], script, eos))
        kwargs["logits_processor"] = processors_
        kwargs.setdefault("max_new_tokens", len(script) + 4)
        return real_generate(*args, **kwargs)

    model.generate = generate  # type: ignore[method-assign]
    return model


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}
TIME_TOOL = {
    "type": "function",
    "function": {"name": "get_time", "description": "Time now",
                 "parameters": {"type": "object", "properties": {}}},
}

#: What each family's model writes for `get_weather(city="Paris")`, per its template.
CALL_REPLY = {
    "lfm2": "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>",
    "hermes": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
    "llama3_json": '{"name": "get_weather", "parameters": {"city": "Paris"}}',
}
