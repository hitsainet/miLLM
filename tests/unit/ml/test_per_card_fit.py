"""A transformers load is judged per card: its weights, its KV cache at a minimum context, its CUDA context.

Decision 7 (user, 2026-09-14). The 20% slack on the weight estimate grew with the
weights, not with what each card needs. On 11,500 / 23,500 MB free it refused
Qwen2.5-14B at FP16, whose real map leaves both cards room for about 8k tokens;
it accepted OLMo-2-13B, whose cuda:0 cannot hold an 8k cache; and it split models
one card holds. A transformers load — a split, Auto's one-card-or-split choice,
or a named card — is now accepted only when each card it uses holds, beside the
weights transformers' own device map puts there, its CUDA context
(TRANSFORMERS_CUDA_CONTEXT_MB, 500 MB until measured on the node) and the KV
cache of its layers at TRANSFORMERS_MIN_CONTEXT tokens (4,096). A model whose KV
cache cannot be sized from its config keeps the slack, and says so as an error.

Shapes are the real configs' fields (config.json), built on the meta device only.
KV figures, worked by hand (bf16, keys and values):
  Qwen2.5-14B (GQA)   2 x 8 kv heads x 128 head_dim x 2 B = 4,096 B a token a layer;
                      48 layers x 4,096 tokens = 805,306,368 B = 768 MiB
  OLMo-2-1124-13B     2 x 40 x 128 x 2 B = 20,480 B; 40 layers x 4,096 = 3,200 MiB
                      (14 layers on cuda:0: 1,120 MiB at 4k, 2,240 at 8k)
  gemma-3-1b (sliding 512, every 6th layer full)
                      2 x 1 x 256 x 2 B = 1,024 B; at 4,096 tokens the 4 full layers
                      16,777,216 B + the 22 sliding ones capped at 512 tokens
                      11,534,336 B = 28,311,552 B -> 27 MiB (rounded up)
  LFM2-1.2B (hybrid: attention at layers 2, 5, 8, 10, 12, 14; convolution elsewhere)
                      2 x 8 x 64 x 2 B = 2,048 B, on 6 layers only:
                      6 x 2,048 x 4,096 = 50,331,648 B = 48 MiB
Weights, by hand:
  Qwen2.5-7B FP16     a layer: q, o 3,584^2 = 12,845,056 each; k, v 3,584 x 512 =
                      1,835,008 each; gate, up, down 3,584 x 18,944 = 67,895,296 each;
                      q/k/v biases 4,608; two norms 7,168 -> 233,057,792 params x 2 B;
                      28 layers + embed_tokens and lm_head 544,997,376 params each x 2 B
                      + final norm = 15,231,233,024 B = 14,526 MiB (rounded up). Its KV at 4k is
                      28 x 2 x 4 x 128 x 2 B x 4,096 = 224 MiB. One card needs
                      14,526 + 224 + 500 = 15,250; the slack's row estimate was
                      7.6B x 2 B x 1.2 = 17,395.
The per-card layouts are transformers' own map over max_memory 11,000 / 23,000: each
card's free memory less its 500 MB CUDA context (review round 5 replaced SHARD_RESERVE_MB
in a sized split's budget, and re-plans a split whose card is short; test_fit_rebalance.py).
Qwen2.5-14B: 9,361 MiB and 15 layers on cuda:0, 18,812 and 33 on cuda:1. OLMo-2-13B:
9,451 and 14 / 16,712 and 26 (MiB rounded up once per card).

MUTATION CONTROLS (mutate.py; each restored and its sha256 verified; the placement,
preflight, load, API and wiring test files run, 323 tests):
  F-M1  a card's need leaves out the CUDA context      -> 9 red (the per-card figures, the
        7B one-card test, the named card, the context setting, both route tests, kept FP8)
  F-M2  a card's KV cache counted as nothing           -> 8 red (the same, less the setting)
  F-M3  a sliding window does not cap its tokens       -> test_sliding_window_layers_keep_only_their_window
  F-M4  a layer with no cache sized as attention       -> test_a_hybrid_model_counts_only_its_attention_layers
  F-M5  the fields that change a token's cost ignored  -> 12 red: test_latent_attention_is_not_sized, the
        fallback test, and every test that plans an unsizable checkpoint on purpose
  F-M6  an unmodelled layer type sized as attention    -> test_a_layer_type_miLLM_does_not_model_is_not_sized
  F-M7  a split's cards not checked (off-GPU part only) -> the Qwen2.5-14B 32k refusal (was OLMo-2 8k
        until review round 5), the context setting,
        both route tests
  F-M8  Auto's one card chosen on its weights alone     -> test_the_cuda_context_allowance_is_read
  F-M9  a named card not checked                        -> test_a_named_card_that_cannot_hold_its_context_is_refused_naming_it
  F-M10 the plan never uses the per-card fit            -> 12 red (every placement test here, both
        FP8 tests, two "all" tests in test_split_preflight.py)
  F-M11 TRANSFORMERS_MIN_CONTEXT not read               -> the Qwen2.5-14B 32k refusal, both route tests
  F-M12 TRANSFORMERS_CUDA_CONTEXT_MB not read           -> test_the_cuda_context_allowance_is_read
  F-M13 every decoder layer put on the split's first card -> the Qwen2.5-14B and both OLMo tests,
        both route tests
  F-M14 the fit sizes a Q4 load without bitsandbytes    -> the Qwen2.5-7B Q4 "all" test, a Q4 map test
  F-M15 an unsizable KV cache does not fall back        -> 11 red, the fallback test among them
  F-M16 the fit decision gets no quantizer factor       -> the wiring test, the Qwen2.5-7B Q4 "all" test
  F-M17 the fit's "all" does not refuse a card left out -> the wiring test, both TestAllNamesEveryVisibleCard
  F-M18 SHARD_RESERVE_MB = 0                            -> 44 red, both acceptances here among them:
        the reserve still shapes every split's map (see gpu_placement.SHARD_RESERVE_MB)
Controls re-run in review round 5 on the lines it moved (mutate.py, millm-p2-review5; this
file, test_split_preflight, test_fit_rebalance, test_split_layout_attribution and the
wiring test; restored, sha256 verified):
  F-M7  (short = [] in _check_split_fit's re-plan loop)                    -> 11 red
  F-M13 (every layer attributed to cuda:0, now in _layer_device_labels)   -> 9 red
  F-M17 (plan_all without refuse_cards_left_out_of_all)                    -> 3 red
  F-M16a ("all" re-plans without the quantizer factor)                     -> 1 red
  F-M16b (Auto re-plans without it) SURVIVED at first: the factor only decided the slack
        fallback's budget, which that fallback no longer took from the slack's own rule.
        Fixed (the fallback re-plans with transformers_shard_rule) and re-run: 1 red,
        test_fit_rebalance::test_the_slack_counts_the_quantizers_factor.
  R2 preflight off-GPU refusal (off_gpu = [] beside the new MiB sum)       -> 3 red
Controls re-run on lines Decision 7 moved into helpers:
  R4-M1 (the "all" left-out rule, now refuse_cards_left_out_of_all) -> the GGUF case, both
        TestAllNamesEveryVisibleCard tests
  R2-M2 (the pre-quantized factor, now split_max_memory_factor)     -> the bitsandbytes checkpoint test
  R1-M4 (the Q2 refusal, now refuse_unsupported_quantization)       -> both Q2 tests
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("transformers")
from fastapi.testclient import TestClient  # noqa: E402
from transformers import (  # noqa: E402
    DeepseekV3Config,
    Gemma3TextConfig,
    LlamaConfig,
    Lfm2Config,
    Olmo2Config,
    Qwen2Config,
)

from millm.core.config import (  # noqa: E402
    TRANSFORMERS_CUDA_CONTEXT_MB_DEFAULT,
    TRANSFORMERS_MIN_CONTEXT_DEFAULT,
    Settings,
    settings,
)
from millm.core.errors import InsufficientMemoryError  # noqa: E402
from millm.db.models.model import ModelStatus, QuantizationType  # noqa: E402
from millm.main import create_app  # noqa: E402
from millm.ml import model_loader  # noqa: E402
from millm.ml.gpu_placement import MODE_SHARD, MODE_SINGLE  # noqa: E402
from millm.ml.memory_utils import estimate_memory_mb  # noqa: E402
from millm.ml.model_loader import (  # noqa: E402
    LoadedModel,
    decide_transformers_placement,
    kv_cache_spec,
    plan_transformers_load,
)
from millm.ml.working_memory import ALLOCATOR_OVERHEAD_FRACTION, WorkingMemory  # noqa: E402
from millm.services.model_service import ModelService  # noqa: E402
from tests.support.factories import make_model  # noqa: E402
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus  # noqa: E402

QWEN25_14B = dict(
    vocab_size=152_064, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=48,
    num_attention_heads=40, num_key_value_heads=8, max_window_layers=70, sliding_window=131_072,
    use_sliding_window=False, tie_word_embeddings=False,
)
OLMO2_13B = dict(
    vocab_size=100_352, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=40,
    num_attention_heads=40, num_key_value_heads=40, max_position_embeddings=4_096,
    tie_word_embeddings=False,
)
QWEN25_7B = dict(
    vocab_size=152_064, hidden_size=3_584, intermediate_size=18_944, num_hidden_layers=28,
    num_attention_heads=28, num_key_value_heads=4, tie_word_embeddings=False,
)
GEMMA3_1B = dict(
    vocab_size=262_144, hidden_size=1_152, intermediate_size=6_912, num_hidden_layers=26,
    num_attention_heads=4, num_key_value_heads=1, head_dim=256, sliding_window=512,
    tie_word_embeddings=True,
)
LFM2_1_2B = dict(
    vocab_size=65_536, hidden_size=2_048, intermediate_size=8_192, num_hidden_layers=16,
    num_attention_heads=32, num_key_value_heads=8, full_attn_idxs=[2, 5, 8, 10, 12, 14],
)
NODE = ((TI_3080, 11_500, 12_288), (RTX_3090, 23_500, 24_576))


def _save(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(directory)
    return str(directory)


def _plan(cards, path, quantization="FP16", requested=None, estimate=0):
    from millm.ml.gpu_placement import list_gpus

    with fake_gpus(*cards):
        return plan_transformers_load(
            estimate, quantization, requested=requested, gpus=list_gpus(), cache_path=path
        )


def _accepted_cards(cards, path, **kwargs):
    """The per-card figures the plan logged when it accepted a split."""
    with patch.object(model_loader, "logger") as logger:
        placement = _plan(cards, path, **kwargs)
    [accepted] = [c for c in logger.info.call_args_list if c.args == ("transformers_fit_split_accepted",)]
    return placement, {card["device"]: card for card in accepted.kwargs["per_card"]}


class TestACardOfASplitIsChargedOnlyThePhasesItRuns:
    """A split's card holds a request's working memory for the phases that run on it: its decoder
    layers, the embeddings and generate's sampling if it holds the input, the final norm and
    lm_head if it holds the output (TransformersFit.layout's input_devices / output_devices).

    In every real model traced here the decoder layer's peak is the largest, so charging every
    card for every phase changed no figure, and a mutation doing exactly that survived (control
    AF1-M9). A class that computes all the logits (none of the 9 transformers 5.15.1
    *ForCausalLM classes without `logits_to_keep` is served here today) or a very large
    vocabulary's sampling makes the output or input phase the largest, and the card that does
    not run it must not be charged for it. The phases below differ on purpose, so a card's
    figure says which phases it was charged for; the layout is transformers' own map of
    OLMo-2-13B."""

    MIB = 1024 * 1024
    LAYER_MB, INPUT_MB, SAMPLING_MB, OUTPUT_MB = 100, 800, 100, 600

    def _phases(self, tokens: int) -> WorkingMemory:
        return WorkingMemory(
            tokens=tokens,
            layer_bytes=(self.LAYER_MB * self.MIB,) * 40,
            input_bytes=self.INPUT_MB * self.MIB,
            output_bytes=self.OUTPUT_MB * self.MIB,
            sampling_bytes=self.SAMPLING_MB * self.MIB,
            method="traced",
        )

    def _cards(self, cards, path, **kwargs):
        with patch.object(model_loader, "size_working_memory", side_effect=lambda *a: self._phases(a[3])):
            return _accepted_cards(cards, path, **kwargs)

    @staticmethod
    def _working(transient_mb: int, kv_mb: int) -> int:
        return transient_mb + math.ceil(ALLOCATOR_OVERHEAD_FRACTION * (transient_mb + kv_mb))

    def test_two_cards_the_first_runs_the_input_and_the_second_the_output(self, tmp_path):
        path = _save(tmp_path, Olmo2Config(**OLMO2_13B))

        placement, cards = self._cards(NODE, path)

        assert placement.mode == MODE_SHARD
        assert set(cards) == {"cuda:0", "cuda:1"}
        first, second = cards["cuda:0"], cards["cuda:1"]
        assert first["working_mb"] == self._working(self.INPUT_MB + self.SAMPLING_MB, first["kv_mb"])
        assert second["working_mb"] == self._working(self.OUTPUT_MB, second["kv_mb"])
        assert placement.to_dict()["working_mb_by_device"] == {
            "cuda:0": first["working_mb"], "cuda:1": second["working_mb"],
        }

    def test_a_middle_card_runs_neither_and_is_charged_its_layers_alone(self, tmp_path):
        path = _save(tmp_path, Olmo2Config(**OLMO2_13B))
        three = ((TI_3080, 11_500, 12_288), (TI_3080, 11_500, 12_288), (RTX_3090, 23_500, 24_576))

        placement, cards = self._cards(three, path, requested="all")

        assert placement.mode == MODE_SHARD
        assert set(cards) == {"cuda:0", "cuda:1", "cuda:2"}
        assert all(card["layers"] for card in cards.values())
        assert cards["cuda:0"]["working_mb"] == self._working(self.INPUT_MB + self.SAMPLING_MB, cards["cuda:0"]["kv_mb"])
        assert cards["cuda:1"]["working_mb"] == self._working(self.LAYER_MB, cards["cuda:1"]["kv_mb"])
        assert cards["cuda:2"]["working_mb"] == self._working(self.OUTPUT_MB, cards["cuda:2"]["kv_mb"])


class TestTheKvCacheIsReadFromTheConfig:
    def test_grouped_query_attention_qwen25_14b(self):
        spec, reason = kv_cache_spec(Qwen2Config(**QWEN25_14B))
        assert reason == ""
        assert spec.bytes_per_token == (4_096,) * 48
        assert spec.token_cap == (None,) * 48, "use_sliding_window is false: no layer is windowed"
        assert spec.mb(range(48), 4_096) == 768

    def test_full_multi_head_attention_olmo2_13b(self):
        spec, _ = kv_cache_spec(Olmo2Config(**OLMO2_13B))
        assert spec.bytes_per_token == (20_480,) * 40
        assert spec.mb(range(40), 4_096) == 3_200
        assert spec.mb(range(14), 8_192) == 2_240

    def test_sliding_window_layers_keep_only_their_window(self):
        spec, _ = kv_cache_spec(Gemma3TextConfig(**GEMMA3_1B))
        full = [index for index, cap in enumerate(spec.token_cap) if cap is None]
        assert full == [5, 11, 17, 23]
        assert set(cap for cap in spec.token_cap if cap is not None) == {512}
        assert spec.mb(range(26), 4_096) == 27
        assert spec.mb(range(26), 256) == 7, "below the window a sliding layer is not capped"

    def test_a_hybrid_model_counts_only_its_attention_layers(self):
        spec, _ = kv_cache_spec(Lfm2Config(**LFM2_1_2B))
        assert [index for index, cost in enumerate(spec.bytes_per_token) if cost] == [2, 5, 8, 10, 12, 14]
        assert set(spec.bytes_per_token) == {0, 2_048}
        assert spec.mb(range(16), 4_096) == 48

    def test_latent_attention_is_not_sized(self):
        spec, reason = kv_cache_spec(DeepseekV3Config(num_hidden_layers=4, kv_lora_rank=128))
        assert spec is None
        assert "kv_lora_rank" in reason

    def test_a_layer_type_miLLM_does_not_model_is_not_sized(self):
        config = LlamaConfig(num_hidden_layers=2)
        config.layer_types = ["full_attention", "deepseek_sparse_attention"]
        spec, reason = kv_cache_spec(config)
        assert spec is None
        assert "deepseek_sparse_attention" in reason


class TestThePlacement:
    def test_qwen25_14b_at_fp16_on_11500_and_23500_is_accepted(self, tmp_path):
        """The slack refused it (row estimate 33,874 > budgets 10,476 + 22,476)."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_14B))
        from millm.ml.gpu_placement import list_gpus

        with fake_gpus(*NODE), pytest.raises(InsufficientMemoryError):
            decide_transformers_placement(33_874, "FP16", requested=None, gpus=list_gpus())

        placement, cards = _accepted_cards(NODE, path)

        assert placement.mode == MODE_SHARD
        # Every card keeps its 500 MB context and a request's transient peak with the
        # allocator's share of it: T = 124,432 B a token (8 x 5,120 + 6 x 13,824 + 528,
        # test_working_memory.py) x 4,096 = 487 MiB, share ceil(0.4 x 487) = 195.
        assert placement.transformers_max_memory() == {0: "10318MiB", 1: "22318MiB"}
        assert cards["cuda:0"] == {
            "device": "cuda:0", "name": TI_3080, "free_mb": 11_500, "weights_mb": 8_311,
            "kv_mb": 208, "working_mb": 765, "staging_mb": 0, "context_mb": 500, "layers": 13,
            "need_mb": 9_784, "short_mb": 0,
        }, "working 487 + ceil(0.4 x (487 + 208))"
        assert (cards["cuda:1"]["weights_mb"], cards["cuda:1"]["layers"], cards["cuda:1"]["kv_mb"]) == (19_862, 35, 560)
        assert (cards["cuda:1"]["working_mb"], cards["cuda:1"]["need_mb"]) == (906, 21_828)

    def test_qwen25_14b_is_refused_where_cuda1_cannot_hold_a_32k_context(self, tmp_path):
        """Qwen2.5-14B serves 32,768 tokens. A request that long has a transient peak of
        124,432 B x 32,768 = 3,889 MiB on every card, so each card's limit is its free
        memory less 500 + 3,889 + 1,556: 5,555 / 17,555. transformers' map then puts
        7,261 MiB on disk, refused before any re-plan. cuda:1 would need 17,327 + 33
        layers x 128 MiB (4,224) + 3,889 + ceil(0.4 x (3,889 + 4,224)) (7,135 in all) +
        500 = 29,186 of its 23,500: 5,686 short. (Round 4 refused OLMo-2-13B at 8,192
        here, a context that model never serves: review round 5.)"""
        path = _save(tmp_path, Qwen2Config(**QWEN25_14B))

        with patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", 32_768), \
                pytest.raises(InsufficientMemoryError) as raised:
            _plan(NODE, path)

        details = raised.value.details
        assert details["short_devices"] == ["cuda:1"]
        assert (details["min_context_tokens"], details["model_max_context_tokens"]) == (32_768, 32_768)
        assert details["per_card"][1] == {
            "device": "cuda:1", "name": RTX_3090, "free_mb": 23_500, "weights_mb": 17_327,
            "kv_mb": 4_224, "working_mb": 7_135, "staging_mb": 0, "context_mb": 500, "layers": 33,
            "need_mb": 29_186, "short_mb": 5_686,
        }
        assert (details["per_card"][0]["need_mb"], details["per_card"][0]["short_mb"]) == (10_248, 0)
        assert details["mapped_mb_by_device"]["disk"] == 7_261
        message = raised.value.message
        for figure in (
            "cuda:1", "23500 MiB free", "17327 MiB of weights", "4224 MiB of KV cache",
            "7135 MiB of working memory", "500 MiB CUDA context", "5686 MiB short",
        ):
            assert figure in message

    def test_olmo2_13b_is_accepted_at_the_default_4k_context(self, tmp_path):
        """Re-planned until cuda:0 holds 12 layers: 8,241 + 960 of KV + 1,119 of working
        memory (T 525 — OLMo-2's 134,152 B a token x 4,096 — + ceil(0.4 x (525 + 960))) +
        500 = 10,820 of 11,500. The minimum context is read. (Before working memory was
        counted it took 14 layers with 429 MiB to spare, and on the node, where the card
        had 11,767 MiB free, 15 layers with 11 — and ran a 3,879-token request out of
        memory: hardware acceptance, 2026-09-14.)"""
        path = _save(tmp_path, Olmo2Config(**OLMO2_13B))

        placement, cards = _accepted_cards(NODE, path)

        assert placement.mode == MODE_SHARD
        assert (cards["cuda:0"]["layers"], cards["cuda:0"]["kv_mb"], cards["cuda:0"]["working_mb"]) == (12, 960, 1_119)
        assert cards["cuda:0"]["need_mb"] == 10_820

    def test_a_7b_width_model_that_fits_one_card_goes_on_it(self, tmp_path):
        """Card 1 has 17,000 free. The slack's 17,395 did not fit it and split the model."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_7B))
        cards = ((TI_3080, 11_500, 12_288), (RTX_3090, 17_000, 24_576))
        from millm.ml.gpu_placement import list_gpus

        row_estimate = estimate_memory_mb("7.6B", "FP16")
        assert row_estimate == 17_395
        with fake_gpus(*cards):
            assert decide_transformers_placement(
                row_estimate, "FP16", requested=None, gpus=list_gpus()
            ).mode == MODE_SHARD

        placement = _plan(cards, path, estimate=row_estimate)

        # 14,526 of weights + 224 of KV + a 4,096-token request's working memory (T 559 —
        # 142,864 B a token, test_working_memory.py — + ceil(0.4 x (559 + 224)) = 873) + 500.
        assert (placement.mode, placement.index, placement.required_mb) == (MODE_SINGLE, 1, 16_123)
        assert placement.to_dict()["working_mb_by_device"] == {"cuda:1": 873}

    def test_a_named_card_that_cannot_hold_its_context_is_refused_naming_it(self, tmp_path):
        path = _save(tmp_path, Qwen2Config(**QWEN25_7B))
        cards = ((TI_3080, 15_000, 16_000), (RTX_3090, 23_500, 24_576))

        with pytest.raises(InsufficientMemoryError) as raised:
            _plan(cards, path, requested=0)

        [card] = raised.value.details["per_card"]
        assert card == {
            "device": "cuda:0", "name": TI_3080, "free_mb": 15_000, "weights_mb": 14_526,
            "kv_mb": 224, "working_mb": 873, "staging_mb": 0, "context_mb": 500, "layers": 28,
            "need_mb": 16_123, "short_mb": 1_123,
        }
        assert "not swapped for another one" in raised.value.message
        assert "873 MiB of working memory for a 4096-token request" in raised.value.message

    def test_a_named_card_that_holds_the_model_carries_its_working_memory(self, tmp_path):
        """The placement a named card is accepted with carries that card's working memory, which
        SAE attachment keeps free (sae_service._working_reserve_mb). Auto's one card did
        (test_a_7b_width_model_that_fits_one_card_goes_on_it); a named card's placement
        dropped it and nothing failed (control AF1-M13)."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_7B))
        cards = ((TI_3080, 11_500, 12_288), (RTX_3090, 17_000, 24_576))

        placement = _plan(cards, path, requested=1)

        assert (placement.mode, placement.index, placement.required_mb) == (MODE_SINGLE, 1, 16_123)
        assert placement.to_dict()["working_mb_by_device"] == {"cuda:1": 873}

    def test_the_cuda_context_allowance_is_read(self, tmp_path):
        """With 3,000 MB a card, one card needs 14,526 + 224 + 873 + 3,000 = 18,623, more
        than card 1's 17,000, so the model splits — and each card of the split keeps its
        3,000 beside a request's transient peak and the allocator's share of it (559 + 224):
        budgets 11,500 - 3,783 and 17,000 - 3,783."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_7B))
        cards = ((TI_3080, 11_500, 12_288), (RTX_3090, 17_000, 24_576))

        with patch.object(settings, "TRANSFORMERS_CUDA_CONTEXT_MB", 3_000):
            placement, by_device = _accepted_cards(cards, path)

        assert placement.mode == MODE_SHARD
        assert placement.transformers_max_memory() == {0: "7717MiB", 1: "13217MiB"}
        assert [card["context_mb"] for card in by_device.values()] == [3_000, 3_000]
        assert all(card["need_mb"] <= card["free_mb"] for card in by_device.values())

    def test_an_architecture_whose_cache_cannot_be_sized_keeps_the_slack_loudly(self, tmp_path):
        config = DeepseekV3Config(num_hidden_layers=4, kv_lora_rank=128)
        config.architectures = ["DeepseekV3ForCausalLM"]
        path = _save(tmp_path, config)
        from millm.ml.gpu_placement import list_gpus

        with patch.object(model_loader, "logger") as logger:
            placement = _plan(NODE, path, estimate=30_000)
        with fake_gpus(*NODE):
            slack = decide_transformers_placement(30_000, "FP16", requested=None, gpus=list_gpus())

        assert placement == slack
        assert placement.required_mb == 30_000
        [fallback] = [c for c in logger.error.call_args_list if c.args == ("transformers_fit_falls_back_to_slack",)]
        assert fallback.kwargs["architecture"] == "DeepseekV3ForCausalLM"
        assert "kv_lora_rank" in fallback.kwargs["reason"]


class TestTheSettings:
    def test_defaults(self):
        fresh = Settings()
        assert fresh.TRANSFORMERS_MIN_CONTEXT == TRANSFORMERS_MIN_CONTEXT_DEFAULT == 4_096
        assert fresh.TRANSFORMERS_CUDA_CONTEXT_MB == TRANSFORMERS_CUDA_CONTEXT_MB_DEFAULT == 500

    @pytest.mark.parametrize("name, value", [("TRANSFORMERS_MIN_CONTEXT", 0), ("TRANSFORMERS_CUDA_CONTEXT_MB", -1)])
    def test_nonsense_is_refused_at_startup(self, name, value, monkeypatch):
        monkeypatch.setenv(name, str(value))
        with pytest.raises(ValueError):
            Settings()


class TestThePreCheckRefusesPerCardBeforeTheUnload:
    """The resident model holds 16,000 MB of card 1: projected, 11,500 / 23,500."""

    CARDS = ((TI_3080, 11_500, 12_288), (RTX_3090, 7_500, 24_576))

    @staticmethod
    def _service(model):
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=model)
        repo.find_by_name = AsyncMock(return_value=model)
        repo.get_locked_model = AsyncMock(return_value=None)
        repo.update_status = AsyncMock(return_value=model)
        loader = MagicMock()
        loader.is_loaded = True
        loader.loaded_model_id = 9
        loader.state.current = LoadedModel(
            9, "resident", MagicMock(), MagicMock(), datetime.utcnow(),
            memory_used_mb=16_000, device="cuda:1", gpu_indices=[1],
            memory_by_device_mb={"cuda:1": 16_000},
        )
        svc = ModelService(repository=repo, downloader=MagicMock(), loader=loader, emitter=None)
        svc.unload_model = AsyncMock()
        svc._executor = MagicMock()
        return svc

    def _model(self, tmp_path):
        return make_model(
            id=3, name="qwen2.5-14b", status=ModelStatus.READY, quantization=QuantizationType.FP16,
            estimated_memory_mb=33_874, cache_path=_save(tmp_path, Qwen2Config(**QWEN25_14B)),
        )

    def test_the_management_route_answers_507_naming_the_card(self, tmp_path):
        from millm.api.dependencies import get_model_service

        svc = self._service(self._model(tmp_path))
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        with fake_gpus(*self.CARDS), patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", 32_768):
            response = TestClient(app).post("/api/models/3/load", json={})

        assert response.status_code == 507, response.text
        error = response.json()["error"]
        assert error["code"] == "INSUFFICIENT_MEMORY"
        assert error["details"]["short_devices"] == ["cuda:1"]
        assert "5686 MiB short" in error["message"]
        assert not svc.unload_model.called
        assert not svc._executor.method_calls and not svc._executor.called

    def test_an_openai_request_that_loads_it_answers_503(self, tmp_path):
        from millm.api.dependencies import get_inference_service, get_model_service

        svc = self._service(self._model(tmp_path))
        inference = MagicMock()
        other = MagicMock()
        other.name = "resident"
        inference.get_loaded_model_info = lambda: other
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        app.dependency_overrides[get_inference_service] = lambda: inference
        with fake_gpus(*self.CARDS), patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", 32_768):
            response = TestClient(app).post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-14b", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 503, response.text
        body = response.json()["error"]
        assert body["code"] == "insufficient_memory"
        assert "cuda:1" in body["message"] and "5686 MiB short" in body["message"]
        assert not svc.unload_model.called
