"""A load's KV cache is sized at TRANSFORMERS_MIN_CONTEXT or at the model's own limit, whichever is shorter.

Review round 5, 2026-09-14. Every request to a transformers model is refused past
its text config's `max_position_embeddings` (InferenceService._check_context_length).
The per-card fit sized the cache at TRANSFORMERS_MIN_CONTEXT regardless, so
OLMo-2-1124-13B and Vicuna-13B-v1.5 — `max_position_embeddings` 4,096 each — were
refused at 8,192 for memory their 8k cache would take, a cache no request can
fill. Round 4 wrote that refusal as a test ("OLMo-2-13B refused at 8k, cuda:0 690
MiB short"); it pinned the defect.

And a multimodal checkpoint keeps `max_position_embeddings` in its TEXT config
only, so the context check, which read the top-level config, found nothing and
accepted every request at any length.

Shapes are the real configs' fields (config.json), built on the meta device only.

MUTATION CONTROLS (mutate.py, millm-p2-review5; run against this file,
test_split_preflight.py and test_per_card_fit.py, or test_inference_service.py for
M4/M5; each restored, sha256 verified, git diff clean):
  R5-M1  admitted_context returns the setting, ignoring the model's limit -> 3 red:
         test_a_model_that_serves_4096_is_sized_at_4096_under_an_8192_floor,
         test_the_refusal_says_which_limit_sized_the_cache,
         test_split_preflight::test_kept_in_fp8_it_is_sized_by_what_it_stores
  R5-M2  kv_cache_spec does not read max_position_embeddings (max_context None)
         -> 5 red: the same three, test_a_multimodal_text_config_carries_the_limit,
         test_per_card_fit::test_qwen25_14b_is_refused_where_cuda1_cannot_hold_a_32k_context
  R5-M3  transformers_fit sizes at TRANSFORMERS_MIN_CONTEXT, not admitted_context
         -> 3 red, the same as R5-M1
  R5-M4  _served_max_context reads the top-level config only
         -> test_a_multimodal_models_context_is_checked_against_its_text_config
  R5-M5  _check_context_length reads the top-level attribute again
         -> test_a_multimodal_models_context_is_checked_against_its_text_config
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("transformers")
from transformers import Gemma3Config, Olmo2Config, Qwen2Config  # noqa: E402

from millm.core.config import settings  # noqa: E402
from millm.core.errors import InsufficientMemoryError  # noqa: E402
from millm.ml import model_loader  # noqa: E402
from millm.ml.gpu_placement import MODE_SHARD, list_gpus  # noqa: E402
from millm.ml.model_loader import kv_cache_spec, plan_transformers_load  # noqa: E402
from millm.services.inference_service import InferenceService  # noqa: E402
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus  # noqa: E402

NODE = ((TI_3080, 11_500, 12_288), (RTX_3090, 23_500, 24_576))

#: allenai/OLMo-2-1124-13B-Instruct config.json.
OLMO2_13B = dict(
    vocab_size=100_352, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=40,
    num_attention_heads=40, num_key_value_heads=40, max_position_embeddings=4_096,
    tie_word_embeddings=False,
)
#: Qwen/Qwen2.5-14B-Instruct config.json.
QWEN25_14B = dict(
    vocab_size=152_064, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=48,
    num_attention_heads=40, num_key_value_heads=8, max_window_layers=70, sliding_window=131_072,
    use_sliding_window=False, max_position_embeddings=32_768, tie_word_embeddings=False,
)
#: google/gemma-3-12b-it config.json: text and vision towers; max_position_embeddings
#: lives in the text config (131,072 by the class default the checkpoint relies on).
GEMMA3_12B = dict(
    text_config=dict(
        vocab_size=262_208, hidden_size=3_840, intermediate_size=15_360, num_hidden_layers=48,
        num_attention_heads=16, num_key_value_heads=8, head_dim=256, sliding_window=1_024,
    ),
    vision_config=dict(
        hidden_size=1_152, intermediate_size=4_304, num_hidden_layers=27, num_attention_heads=16,
        image_size=896, patch_size=14,
    ),
)


def _save(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(directory)
    return str(directory)


def _accepted(path, context):
    """The plan and the per-card figures it logged on accepting a split."""
    with patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", context), \
            patch.object(model_loader, "logger") as logger, fake_gpus(*NODE):
        placement = plan_transformers_load(0, "FP16", requested=None, gpus=list_gpus(), cache_path=path)
    [accepted] = [c for c in logger.info.call_args_list if c.args == ("transformers_fit_split_accepted",)]
    return placement, accepted.kwargs


class TestTheCacheIsSizedAtWhatTheModelServes:
    def test_a_model_that_serves_4096_is_sized_at_4096_under_an_8192_floor(self, tmp_path):
        """cuda:0 holds 14 of 40 layers: 14 x 2 x 40 x 128 x 2 B x 4,096 = 1,120 MiB,
        need 9,451 + 1,120 + 500 = 11,071 of 11,500. At 8,192 it was 2,240 and a
        refusal, 690 short."""
        path = _save(tmp_path, Olmo2Config(**OLMO2_13B))

        placement, logged = _accepted(path, 8_192)

        assert placement.mode == MODE_SHARD
        assert logged["min_context_tokens"] == 4_096
        cuda0 = next(card for card in logged["per_card"] if card["device"] == "cuda:0")
        assert (cuda0["layers"], cuda0["kv_mb"], cuda0["need_mb"]) == (14, 1_120, 11_071)

    def test_a_model_that_serves_more_is_sized_at_the_floor(self, tmp_path):
        """Qwen2.5-14B serves 32,768: at an 8,192 floor, cuda:0's 15 layers hold
        15 x 2 x 8 x 128 x 2 B x 8,192 = 480 MiB."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_14B))

        _, logged = _accepted(path, 8_192)

        assert logged["min_context_tokens"] == 8_192
        cuda0 = next(card for card in logged["per_card"] if card["device"] == "cuda:0")
        assert cuda0["kv_mb"] == 480

    def test_the_refusal_says_which_limit_sized_the_cache(self, tmp_path):
        """A 65,536 floor on a model that serves 32,768 is sized at 32,768, and the
        refusal carries both figures."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_14B))

        with patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", 65_536), fake_gpus(*NODE), \
                pytest.raises(InsufficientMemoryError) as raised:
            plan_transformers_load(0, "FP16", requested=None, gpus=list_gpus(), cache_path=path)

        details = raised.value.details
        assert (details["min_context_tokens"], details["model_max_context_tokens"]) == (32_768, 32_768)
        assert details["kv_mb"] == 48 * 128, "48 layers x 128 MiB at 32,768 tokens, not 256 at 65,536"
        assert "32768-token context" in raised.value.message

    def test_a_multimodal_text_config_carries_the_limit(self):
        spec, _ = kv_cache_spec(Gemma3Config(**GEMMA3_12B))
        assert spec.max_context == 131_072
        assert getattr(Gemma3Config(**GEMMA3_12B), "max_position_embeddings", None) is None, (
            "the top-level config does not carry it: that is why the text config is read"
        )


class TestTheRequestCheckReadsTheSameLimit:
    @staticmethod
    def _service(config):
        service = InferenceService.__new__(InferenceService)
        model = SimpleNamespace(config=config)
        service._model_state = SimpleNamespace(is_loaded=True, current=SimpleNamespace(model=model))
        return service

    def test_a_multimodal_models_context_is_checked_against_its_text_config(self):
        """gemma-3-12b-it: 131,000 prompt + 1,000 new tokens > 131,072. Read from the
        top-level config it was accepted at any length."""
        service = self._service(Gemma3Config(**GEMMA3_12B))

        with pytest.raises(ValueError, match="Context length exceeded"):
            service._check_context_length(prompt_tokens=131_000, max_new_tokens=1_000)
        service._check_context_length(prompt_tokens=130_000, max_new_tokens=1_000)

    def test_a_text_model_is_checked_as_before(self):
        service = self._service(Olmo2Config(**OLMO2_13B))

        with pytest.raises(ValueError, match="4097 > 4096"):
            service._check_context_length(prompt_tokens=4_000, max_new_tokens=97)
        service._check_context_length(prompt_tokens=4_000, max_new_tokens=96)
