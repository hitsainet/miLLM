"""A split with a short card is re-planned until every card holds its weights, its KV cache and its CUDA context.

Review round 5, 2026-09-14. Round 4's per-card fit planned a split with each card's
budget at free memory less a flat 1,024 MB (SHARD_RESERVE_MB), computed
transformers' map, and refused the load if any card was short — "nothing re-plans
a split around a short card". A card a few hundred MiB short beside one with
gigabytes spare was refused, although moving a layer or two fixed it.

Now (model_loader._check_split_fit): each card's budget starts at its free memory
less its CUDA context and the plan's need includes the whole KV cache, so Auto
takes enough cards for the cache as well as the weights. When transformers' map
leaves a card short, that card's limit is cut by its shortfall — doubled while
the map does not move on it — and the split is planned again. Limits only fall, so
no map repeats; FIT_REBALANCE_MAX_PASSES bounds it anyway; and a placement is
returned only when the map for exactly that placement fits every card. A split no
re-plan fits is refused with the last split that stayed on the GPUs.

Shapes are the real configs' fields (lmsys/vicuna-13b-v1.5, Qwen/Qwen2.5-7B-Instruct,
Qwen/Qwen2.5-14B-Instruct, allenai/OLMo-2-1124-13B-Instruct, meta-llama/Llama-3.2-1B),
built on the meta device only. Per-pass figures are transformers' own map.

PROOF: run against the tree before the rebalance (11416d7, a scratch worktree), 8 of
these 13 tests fail — the rescue of OLMo-2-13B Q4 among them (round 4 refused it,
cuda:0 1,712 MiB short). A grid of 2,940 plans (seven real checkpoints x FP16/Q8/Q4 x
four contexts x 42 free-memory pairs) compared the two trees: no load round 4 accepted
is refused, 227 it refused are accepted (138 only by re-planning), and no model's
one-card-or-split choice changed.

MUTATION CONTROLS (mutate.py, millm-p2-review5; run against this file,
test_per_card_fit.py and test_split_preflight.py; each restored, sha256 verified,
git diff clean):
  R5-M20 a short card is refused, never re-planned          -> 9 red
  R5-M21 the cut is not doubled when the map does not move  -> test_vicuna_13b_moves_a_layer_off_the_3080_ti,
         test_qwen25_14b_cannot_hold_32k_on_these_cards_however_it_is_split (pass counts)
  R5-M22 the rule ignores the cut limits                    -> 8 red
  R5-M23 a split is accepted with a card short              -> 11 red, the overcommit grid among them
  R5-M24 an unknown layout falls back without saying so     -> test_the_slack_judges_it_and_says_so
  R5-M25 the plan's need leaves out the KV cache            -> 3 red: the card-choice test, the Qwen2.5-7B
         32k rebalance, test_split_preflight's Q4 "all" refusal
  R5-M26 the pass bound ignored                             -> test_the_search_is_bounded
  R5-M27 a card's budget starts at free less 1,024 MB again -> 14 red
  R5-M28 "all" with a card the layout leaves empty is not refused by the plan
         -> test_a_card_whose_budget_holds_no_layer_is_refused_by_the_plan and
            test_split_preflight::test_a_card_whose_share_holds_no_layer_is_refused

FOUND BY RE-RUNNING ROUND 4'S F-M16 ON THE MOVED LINES: with Auto's re-plan given no
quantizer factor, the suite stayed green. The factor still decided one thing — the
budget of the slack fallback taken when the layout cannot be computed — and that
fallback (as first written in this round) judged the split against the FIT's budget
(free less the CUDA context) rather than the slack's own (free less SHARD_RESERVE_MB,
times the factor): it admitted OLMo-2-13B FP16 on 16,000 / 16,500 MB, which the slack
refuses. Both TestAnUnknownLayoutFallsBackLoudly refusal tests failed on that code;
the fallback now re-plans with transformers_shard_rule.
  R5-M29 the fallback judges the fit's placement again     -> 3 red: all of TestAnUnknownLayoutFallsBackLoudly
  R5-M30 the slack's rule without the quantizer factor     -> test_the_slack_counts_the_quantizers_factor
  F-M16b re-run (Auto re-plans without the factor)         -> test_the_slack_counts_the_quantizers_factor
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("transformers")
from transformers import LlamaConfig, Olmo2Config, Qwen2Config  # noqa: E402

from millm.core.config import settings  # noqa: E402
from millm.core.errors import InsufficientMemoryError, SplitNotHonouredError  # noqa: E402
from millm.ml import model_loader  # noqa: E402
from millm.ml.gpu_placement import MODE_SHARD, list_gpus  # noqa: E402
from millm.ml.model_loader import (  # noqa: E402
    decide_transformers_placement,
    plan_transformers_load,
    preflight_split,
    transformers_fit,
)
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus  # noqa: E402

NODE = ((TI_3080, 11_500, 12_288), (RTX_3090, 23_500, 24_576))
CARD_1_BUSY = ((TI_3080, 11_500, 12_288), (RTX_3090, 15_100, 24_576))

VICUNA_13B = dict(
    vocab_size=32_000, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=40,
    num_attention_heads=40, num_key_value_heads=40, max_position_embeddings=4_096,
    rms_norm_eps=1e-5, tie_word_embeddings=False,
)
QWEN25_7B = dict(
    vocab_size=152_064, hidden_size=3_584, intermediate_size=18_944, num_hidden_layers=28,
    num_attention_heads=28, num_key_value_heads=4, max_position_embeddings=32_768,
    tie_word_embeddings=False,
)
QWEN25_14B = dict(
    vocab_size=152_064, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=48,
    num_attention_heads=40, num_key_value_heads=8, max_window_layers=70, sliding_window=131_072,
    use_sliding_window=False, max_position_embeddings=32_768, tie_word_embeddings=False,
)
OLMO2_13B = dict(
    vocab_size=100_352, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=40,
    num_attention_heads=40, num_key_value_heads=40, max_position_embeddings=4_096,
    tie_word_embeddings=False,
)
LLAMA32_1B = dict(
    vocab_size=128_256, hidden_size=2_048, intermediate_size=8_192, num_hidden_layers=16,
    num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=True,
)


def _save(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(directory)
    return str(directory)


def _plan(cards, path, context=4_096, requested=None, quantization="FP16"):
    """The plan, and the per-card figures and passes it logged if it accepted a split."""
    with patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", context), \
            patch.object(model_loader, "logger") as logger, fake_gpus(*cards):
        placement = plan_transformers_load(
            0, quantization, requested=requested, gpus=list_gpus(), cache_path=path
        )
    accepted = [c.kwargs for c in logger.info.call_args_list if c.args == ("transformers_fit_split_accepted",)]
    return placement, (accepted[-1] if accepted else None)


def _by_device(logged):
    return {
        card["device"]: (card["weights_mb"], card["layers"], card["kv_mb"], card["need_mb"], card["free_mb"])
        for card in logged["per_card"]
    }


class TestAShortCardIsRebalanced:
    def test_vicuna_13b_moves_a_layer_off_the_3080_ti(self, tmp_path):
        """Pass 1, limits 11,000 / 23,000: cuda:0 takes 16 layers, 9,993 + 1,280 + 500 =
        11,773 of 11,500, short 273. Pass 2 cuts its limit by 273 to 10,727: the map
        does not move. Pass 3 cuts twice that, to 10,181: layer 15 moves, and cuda:0
        holds 15 layers, 9,388 + 1,200 + 500 = 11,088; cuda:1 25, 15,438 + 2,000 + 500."""
        path = _save(tmp_path, LlamaConfig(**VICUNA_13B))

        placement, logged = _plan(NODE, path)

        assert placement.mode == MODE_SHARD
        assert placement.transformers_max_memory() == {0: "10181MiB", 1: "23000MiB"}
        assert (logged["passes"], logged["lowered_limits_mb"]) == (3, {"cuda:0": 10_181})
        assert _by_device(logged) == {
            "cuda:0": (9_388, 15, 1_200, 11_088, 11_500),
            "cuda:1": (15_438, 25, 2_000, 17_938, 23_500),
        }

    def test_olmo2_13b_q4_that_round_4_refused_is_split_across_both_cards(self, tmp_path):
        """10,000 / 9,000 MB free, Q4: 8,012 MiB of weights and a 3,200 MiB cache at 4,096.
        Round 4 sized the plan on the weights, which cuda:0's budget ((10,000 - 1,024) x
        0.9 = 8,078) covered, so it planned cuda:0 alone — all 40 layers, 8,012 + 3,200
        + 500 = 11,712 of 10,000 — and refused, 1,712 short. Now the need includes the
        cache, so both cards are planned; pass 1 (limits 9,500 / 8,500) still maps every
        layer to cuda:0, short 1,712, and pass 2 cuts it to 7,788: 33 layers there,
        5,973 + 2,640 + 500 = 9,113, and 7 on cuda:1, 2,040 + 560 + 500 = 3,100."""
        path = _save(tmp_path, Olmo2Config(**OLMO2_13B))
        cards = ((TI_3080, 10_000, 12_288), (RTX_3090, 9_000, 24_576))

        placement, logged = _plan(cards, path, quantization="Q4")

        assert placement.transformers_max_memory() == {0: "7788MiB", 1: "8500MiB"}
        assert (logged["passes"], logged["lowered_limits_mb"]) == (2, {"cuda:0": 7_788})
        assert _by_device(logged) == {
            "cuda:0": (5_973, 33, 2_640, 9_113, 10_000),
            "cuda:1": (2_040, 7, 560, 3_100, 9_000),
        }

    def test_qwen25_7b_serves_32k_across_a_busy_3090(self, tmp_path):
        """cuda:1 has 15,100 MB free. Pass 1: cuda:0 20 layers, 9,930 + 1,280 + 500 =
        11,710, short 210. Pass 2, limit 10,790: 19 layers, 9,486 + 1,216 + 500 = 11,202;
        cuda:1 9 layers, 5,041 + 576 + 500 = 6,117. (Round 4's flat 1,024 MB reserve
        happened to plan these same 19 layers; the rebalance reaches them from a budget
        that does not hold back room the card does not need.)"""
        path = _save(tmp_path, Qwen2Config(**QWEN25_7B))

        placement, logged = _plan(CARD_1_BUSY, path, context=32_768)

        assert placement.transformers_max_memory() == {0: "10790MiB", 1: "14600MiB"}
        assert (logged["passes"], logged["min_context_tokens"]) == (2, 32_768)
        assert _by_device(logged) == {
            "cuda:0": (9_486, 19, 1_216, 11_202, 11_500),
            "cuda:1": (5_041, 9, 576, 6_117, 15_100),
        }


class TestWhatNoReplanFits:
    def test_qwen25_14b_cannot_hold_32k_on_these_cards_however_it_is_split(self, tmp_path):
        """Weights 28,173 MiB + a 32,768-token cache of 6,144 + two 500 MB contexts =
        35,317 against 35,000 free: no split holds it. Five passes: cuda:0 is cut to 14
        layers and fits; cuda:1, then short 689, is cut until its tail goes to disk.
        The refusal carries the last split that stayed on the GPUs."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_14B))
        fit = transformers_fit(path, "FP16", False)
        with patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", 32_768):
            total_need = fit.weights_mb + transformers_fit(path, "FP16", False).kv_mb() + 2 * 500
        assert total_need > 11_500 + 23_500

        with pytest.raises(InsufficientMemoryError) as raised:
            _plan(NODE, path, context=32_768)

        details = raised.value.details
        assert details["short_devices"] == ["cuda:1"]
        assert details["rebalance_passes"] == 5
        assert [(c["device"], c["weights_mb"], c["layers"], c["short_mb"]) for c in details["per_card"]] == [
            ("cuda:0", 8_836, 14, 0), ("cuda:1", 19_337, 34, 689),
        ]
        assert "leaves the other cards without room" in raised.value.message

    def test_the_search_is_bounded(self, tmp_path):
        """Vicuna-13B needs three passes; allowed two, it is refused, not accepted short."""
        path = _save(tmp_path, LlamaConfig(**VICUNA_13B))

        with patch.object(model_loader, "FIT_REBALANCE_MAX_PASSES", 2), \
                pytest.raises(InsufficientMemoryError) as raised:
            _plan(NODE, path)

        assert raised.value.details["rebalance_passes"] == 2
        assert raised.value.details["short_devices"] == ["cuda:0"]
        assert "did not settle within 2 re-plans" in raised.value.message


class TestTheSearchIsSafe:
    @pytest.mark.parametrize(
        "shape, config_class, contexts",
        [
            ("vicuna", LlamaConfig, (4_096,)),
            ("qwen7b", Qwen2Config, (4_096, 32_768)),
            ("qwen14b", Qwen2Config, (4_096, 16_384)),
            ("olmo", Olmo2Config, (4_096,)),
        ],
    )
    def test_no_accepted_split_overcommits_a_card_and_the_preflight_agrees(
        self, tmp_path, shape, config_class, contexts
    ):
        fields = {"vicuna": VICUNA_13B, "qwen7b": QWEN25_7B, "qwen14b": QWEN25_14B, "olmo": OLMO2_13B}[shape]
        path = _save(tmp_path, config_class(**fields))
        accepted = 0
        for context in contexts:
            for free in ((8_000, 23_500), (11_500, 9_000), (11_500, 15_100), (11_500, 23_500)):
                cards = ((TI_3080, free[0], 12_288), (RTX_3090, free[1], 24_576))
                try:
                    placement, logged = _plan(cards, path, context=context)
                except (InsufficientMemoryError, SplitNotHonouredError):
                    continue
                if logged is None:
                    continue  # one card
                accepted += 1
                for card in logged["per_card"]:
                    assert card["need_mb"] <= card["free_mb"], (shape, context, free, card)
                with patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", context), fake_gpus(*cards):
                    mapped = preflight_split(shape, path, "FP16", placement)
                assert mapped == {card["device"]: card["weights_mb"] for card in logged["per_card"]}
        assert accepted, f"{shape}: the grid must accept at least one split"

    def test_the_same_cards_give_the_same_placement(self, tmp_path):
        path = _save(tmp_path, LlamaConfig(**VICUNA_13B))

        first, _ = _plan(NODE, path)
        second, _ = _plan(NODE, path)

        assert first == second


class TestAutoTakesCardsForTheCache:
    def test_a_card_that_holds_the_weights_but_not_the_cache_is_not_split_over_alone(self, tmp_path):
        """Qwen2.5-7B: 14,526 MiB of weights, 224 of KV at 4,096. cuda:1's budget, 15,100 -
        500 = 14,600, covers the weights but not weights + KV (14,750). With the need
        sized on the weights alone (R5-M25), Auto chose cuda:1 alone, whose map puts
        lm_head on disk (room for the embedding is held back), and refused a model the
        two cards hold."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_7B))

        placement, logged = _plan(CARD_1_BUSY, path)

        assert placement.gpu_indices == [0, 1]
        assert logged["passes"] == 1


class TestAllIsRefusedFromTheLayout:
    def test_a_card_whose_budget_holds_no_layer_is_refused_by_the_plan(self, tmp_path):
        """cuda:0 has 900 MB free: a 400 MiB budget, less than llama-3.2-1b's 490 MiB
        embedding. The map puts all 2,358 MiB on cuda:1."""
        path = _save(tmp_path, LlamaConfig(**LLAMA32_1B))
        cards = ((TI_3080, 900, 12_288), (RTX_3090, 23_500, 24_576))

        with pytest.raises(SplitNotHonouredError) as raised:
            _plan(cards, path, requested="all")

        details = raised.value.details
        assert details["unused_devices"] == ["cuda:0"]
        assert details["mapped_mb_by_device"] == {"cuda:1": 2_358}
        assert details["max_memory"] == {"cuda:0": "400MiB", "cuda:1": "23000MiB"}


class TestAnUnknownLayoutFallsBackLoudly:
    def test_the_slack_judges_it_and_says_so(self, tmp_path):
        """When transformers' map cannot be computed here the split is judged by the
        20% slack, and that is logged as an error. Round 4 logged only why the layout
        failed, never that the fit had fallen back. Qwen2.5-14B FP16 on 11,500 / 23,500:
        28,172 MiB x 1.2 = 33,806 against the slack's budgets 10,476 + 22,476 = 32,952 —
        refused, exactly as the slack refuses it on its own (Decision 7's example)."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_14B))

        with patch.object(model_loader.TransformersFit, "layout", return_value=None), \
                patch.object(model_loader, "logger") as logger, fake_gpus(*NODE), \
                pytest.raises(InsufficientMemoryError) as raised:
            plan_transformers_load(0, "FP16", requested=None, gpus=list_gpus(), cache_path=path)

        [fallback] = [c for c in logger.error.call_args_list if c.args == ("transformers_fit_falls_back_to_slack",)]
        assert fallback.kwargs["architecture"] == "qwen2"
        assert "layout" in fallback.kwargs["reason"]
        assert (raised.value.details["required_mb"], raised.value.details["available_mb"]) == (33_806, 32_952)
        with fake_gpus(*NODE), pytest.raises(InsufficientMemoryError) as slack:
            decide_transformers_placement(33_806, "FP16", requested=None, gpus=list_gpus())
        assert slack.value.details["available_mb"] == 32_952

    def test_the_slack_is_the_slacks_own_budget_not_the_fits(self, tmp_path):
        """OLMo-2-13B FP16, 26,162 MiB of weights: the slack needs x1.2 = 31,394. The
        slack budgets a card at free less SHARD_RESERVE_MB: 14,976 + 15,476 = 30,452, and
        refuses. Judged against the fit's own budget (free less the 500 MB context,
        15,500 + 16,000 = 31,500) the fallback accepted it: a load "judged by the 20%
        slack" that the slack itself refuses. Review round 5."""
        path = _save(tmp_path, Olmo2Config(**OLMO2_13B))
        cards = ((TI_3080, 16_000, 24_576), (RTX_3090, 16_500, 24_576))

        with patch.object(model_loader.TransformersFit, "layout", return_value=None), \
                patch.object(model_loader, "logger"), fake_gpus(*cards), \
                pytest.raises(InsufficientMemoryError) as raised:
            plan_transformers_load(0, "FP16", requested=None, gpus=list_gpus(), cache_path=path)

        assert raised.value.details["required_mb"] == int(26_162 * 1.2)
        assert "judged by the 20% slack" in raised.value.message

    def test_the_slack_counts_the_quantizers_factor(self, tmp_path):
        """OLMo-2-13B Q4, 8,012 MiB: the slack needs 9,614. Slack limits 4,976 / 4,676;
        bitsandbytes places x0.9 of them, 4,478 + 4,208 = 8,686, and it is refused.
        Without the factor, 9,652 would have accepted it."""
        path = _save(tmp_path, Olmo2Config(**OLMO2_13B))
        cards = ((TI_3080, 6_000, 12_288), (RTX_3090, 5_700, 24_576))

        with patch.object(model_loader.TransformersFit, "layout", return_value=None), \
                patch.object(model_loader, "logger"), fake_gpus(*cards), \
                pytest.raises(InsufficientMemoryError) as raised:
            plan_transformers_load(0, "Q4", requested=None, gpus=list_gpus(), cache_path=path)

        assert raised.value.details["required_mb"] == int(8_012 * 1.2)
        assert raised.value.details["available_mb"] == 4_478 + 4_208
