"""kv_cache_spec agrees with the cache transformers really allocates, layer by layer.

Review round 5, 2026-09-14. The per-card fit's KV figure is derived from the
config (model_loader.kv_cache_spec). Every earlier test of it compared the
formula against a hand-worked number, which proves the arithmetic and not that
transformers stores what the arithmetic says. Here each architecture is BUILT
(tiny widths, the real classes and their own layer-type logic), generate() runs
with the cache miLLM's generation path asks for, and the tensors the cache holds
are measured: bytes a token per layer must equal the spec's, and no layer may hold
more tokens than the spec's window allows. A transformers upgrade that changes
what a layer type caches turns this red.

FOUND BY IT: `attention_k_eq_v` (Gemma 4's global layers use their keys as values)
was in _KV_UNSIZED_FIELDS, so every checkpoint that declares it fell back to the
20% slack as "not sizable", loudly, on every load. The cache still stores keys and
values as two tensors of the per-layer override's shape: 2 x kv heads x head_dim
x 2 B, exactly the formula.

MUTATION CONTROLS (mutate.py, millm-p2-review5; each restored, sha256 verified,
git diff clean):
  R5-M6  "attention_k_eq_v" back in _KV_UNSIZED_FIELDS -> the gemma4-k_eq_v case
  R5-M7  _layer_config ignores per_layer_config         -> both gemma4 cases
  R5-M8  a windowed layer's cap not recorded (None)     -> qwen2-sliding, mistral-sliding,
         gemma2, gemma3, both gemma4 cases
  R5-M9  TRANSFORMERS_KV_BYTES = 4 (float32)            -> every case
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")
import torch  # noqa: E402
import transformers as tf  # noqa: E402

from millm.ml.model_loader import kv_cache_spec  # noqa: E402

PROMPT = 30
NEW_TOKENS = 3
WINDOW = 8
BASE = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_attention_heads=4, max_position_embeddings=512)
GEMMA4 = dict(
    num_hidden_layers=6, num_key_value_heads=2, head_dim=16, global_head_dim=32,
    num_global_key_value_heads=1, sliding_window=WINDOW, vocab_size_per_layer_input=128,
    hidden_size_per_layer_input=8, **BASE,
)


def _cases():
    return {
        "llama-mha": (tf.LlamaConfig(num_hidden_layers=2, num_key_value_heads=4, **BASE), tf.LlamaForCausalLM),
        "llama-gqa": (tf.LlamaConfig(num_hidden_layers=2, num_key_value_heads=2, **BASE), tf.LlamaForCausalLM),
        "llama-explicit-head-dim": (
            tf.LlamaConfig(num_hidden_layers=2, num_key_value_heads=2, head_dim=32, **BASE), tf.LlamaForCausalLM,
        ),
        "qwen2": (tf.Qwen2Config(num_hidden_layers=3, num_key_value_heads=2, **BASE), tf.Qwen2ForCausalLM),
        "qwen2-sliding": (
            tf.Qwen2Config(num_hidden_layers=4, num_key_value_heads=2, use_sliding_window=True,
                           sliding_window=WINDOW, max_window_layers=2, **BASE),
            tf.Qwen2ForCausalLM,
        ),
        "olmo2": (tf.Olmo2Config(num_hidden_layers=2, num_key_value_heads=4, **BASE), tf.Olmo2ForCausalLM),
        "mistral-sliding": (
            tf.MistralConfig(num_hidden_layers=2, num_key_value_heads=2, sliding_window=WINDOW, **BASE),
            tf.MistralForCausalLM,
        ),
        "mistral-explicit-head-dim": (
            tf.MistralConfig(num_hidden_layers=2, num_key_value_heads=2, sliding_window=None, head_dim=24, **BASE),
            tf.MistralForCausalLM,
        ),
        "gemma2": (
            tf.Gemma2Config(num_hidden_layers=4, num_key_value_heads=2, head_dim=32, sliding_window=WINDOW, **BASE),
            tf.Gemma2ForCausalLM,
        ),
        "gemma3": (
            tf.Gemma3TextConfig(num_hidden_layers=7, num_key_value_heads=2, head_dim=32, sliding_window=WINDOW, **BASE),
            tf.Gemma3ForCausalLM,
        ),
        "gemma4": (tf.Gemma4TextConfig(**GEMMA4), tf.Gemma4ForCausalLM),
        "gemma4-k_eq_v": (tf.Gemma4TextConfig(attention_k_eq_v=True, **GEMMA4), tf.Gemma4ForCausalLM),
        "lfm2-hybrid": (
            tf.Lfm2Config(num_hidden_layers=4, num_key_value_heads=2, full_attn_idxs=[1, 3], **BASE),
            tf.Lfm2ForCausalLM,
        ),
        "granitemoehybrid": (
            tf.GraniteMoeHybridConfig(
                num_hidden_layers=4, num_key_value_heads=2, layer_types=["mamba", "attention", "mamba", "attention"],
                mamba_n_heads=4, mamba_d_head=32, mamba_d_state=8, mamba_n_groups=1, mamba_expand=2,
                num_local_experts=0, shared_intermediate_size=128, **BASE,
            ),
            tf.GraniteMoeHybridForCausalLM,
        ),
    }


def _held(layer) -> tuple[int, int]:
    """(bytes, tokens) of the keys and values a cache layer holds."""
    total, tokens = 0, 0
    for name in ("keys", "values"):
        tensor = getattr(layer, name, None)
        if isinstance(tensor, torch.Tensor) and tensor.numel():
            total += tensor.numel() * tensor.element_size()
            tokens = tensor.shape[-2]
    return total, tokens


@pytest.mark.parametrize("name", sorted(_cases()))
def test_the_spec_matches_what_generate_caches(name):
    config, model_class = _cases()[name]
    spec, reason = kv_cache_spec(config)
    assert spec is not None, f"{name} is sizable from its config, but: {reason}"

    torch.manual_seed(0)
    model = model_class(config).to(torch.bfloat16).eval()
    kwargs = {}
    if "hybrid" in config.model_type or "mamba" in config.model_type:
        kwargs["cache_implementation"] = "hybrid"  # as InferenceService._build_generate_kwargs asks
    ids = torch.randint(3, 100, (1, PROMPT))
    with torch.no_grad():
        out = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=NEW_TOKENS, do_sample=False,
            return_dict_in_generate=True, use_cache=True, pad_token_id=0, **kwargs,
        )
    layers = out.past_key_values.layers
    held_tokens = PROMPT + NEW_TOKENS - 1

    assert len(layers) == spec.num_layers
    for index, layer in enumerate(layers):
        stored, tokens = _held(layer)
        per_token = stored // tokens if tokens else 0
        assert per_token == spec.bytes_per_token[index], (
            f"{name} layer {index}: the cache holds {per_token} B a token, the spec says "
            f"{spec.bytes_per_token[index]}"
        )
        if not per_token:
            continue
        cap = spec.token_cap[index]
        if cap is None:
            assert tokens == held_tokens, f"{name} layer {index}: {tokens} tokens held of {held_tokens}"
        else:
            # DynamicSlidingWindowLayer keeps `sliding_window - 1` tokens once full;
            # the spec counts the whole window, one token over, never under.
            assert cap - 1 <= tokens <= cap, (
                f"{name} layer {index}: {tokens} tokens held under a window the spec records as {cap}"
            )
