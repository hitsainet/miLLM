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

REVIEW ROUND 6 (2026-09-14) — TWO SHORT CARDS IN ONE PASS RAISED KeyError. A step was
recorded only when a cut was taken. When the lower-index card's shortfall is at least
its limit (no cut) while the last card is cut without its map moving, the next pass
finds the first card short at the same weights and reads its step to double it:
`steps[0]`, never written. Llama-2-7B-32K at Q8 with a 32,768-token cache on 3,000 /
16,000 MB free — 24 GB needed of 19 GB — raised KeyError out of the pre-unload check on
both Auto and "all", a bare 500 where round 4 gave a refusal with every card's figures.
Found by a targeted probe (millm-p2-review6/keyerror_probe.py): no grid round 5 ran
combined a KV-heavy MHA model, a long context and a nearly full lower-index card.
  R6-M1  the step recorded only when the cut is taken (round 5's code)
         -> both TestTwoShortCardsInOnePass cases (KeyError: 0)
  Round 5's controls on the re-plan loop re-run after the fix (this file,
  test_per_card_fit.py, test_split_preflight.py), all red: R5-M20 11, R5-M21 4,
  R5-M22 10, R5-M23 13, R5-M26 1.
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
#: togethercomputer/LLaMA-2-7B-32K: multi-head attention with 32,768 positions, so a
#: layer's cache at full context (512 MiB) outweighs its Q8 weights (~193 MiB).
LLAMA2_7B_32K = dict(
    vocab_size=32_000, hidden_size=4_096, intermediate_size=11_008, num_hidden_layers=32,
    num_attention_heads=32, num_key_value_heads=32, max_position_embeddings=32_768,
    rms_norm_eps=1e-5, tie_word_embeddings=False,
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
    def test_vicuna_13b_moves_layers_off_the_3080_ti(self, tmp_path):
        """Each card's limit starts at its free memory less its 500 MB context and a
        4,096-token request's transient peak with the allocator's share of it: 10,318 /
        22,318. cuda:0 is short at first and is cut, doubling while its map does not move,
        until pass 5 at 9,337: 13 layers there, 8,178 + 1,040 of KV + 1,098 of working
        memory + 500 = 10,816; 27 on cuda:1, 16,649 + 2,160 + 1,546 + 500 = 20,855."""
        path = _save(tmp_path, LlamaConfig(**VICUNA_13B))

        placement, logged = _plan(NODE, path)

        assert placement.mode == MODE_SHARD
        assert placement.transformers_max_memory() == {0: "9337MiB", 1: "22318MiB"}
        assert (logged["passes"], logged["lowered_limits_mb"]) == (5, {"cuda:0": 9_337})
        assert _by_device(logged) == {
            "cuda:0": (8_178, 13, 1_040, 10_816, 11_500),
            "cuda:1": (16_649, 27, 2_160, 20_855, 23_500),
        }

    def test_olmo2_13b_q4_is_split_across_both_cards_by_a_replan(self, tmp_path):
        """11,000 / 9,000 MB free, Q4: 8,012 MiB of weights and a 3,200 MiB cache at 4,096.
        Round 4 sized the plan on the weights alone and refused this shape, cuda:0 1,712
        MiB short. Now pass 1 leaves cuda:0 short and pass 2 cuts it to 5,343: 18 layers
        there, 3,704 + 1,440 of KV + 1,311 of working memory + 763 left by quantizing its
        weights as they load + 500 = 7,718; 22 on cuda:1, 4,309 + 1,760 + 1,439 + 932 +
        500 = 8,940. (On 10,000 / 9,000 it no longer fits: a request's working memory and
        the staging put cuda:1 1,163 MiB short however the layers are divided.)"""
        path = _save(tmp_path, Olmo2Config(**OLMO2_13B))
        cards = ((TI_3080, 11_000, 12_288), (RTX_3090, 9_000, 24_576))

        placement, logged = _plan(cards, path, quantization="Q4")

        assert placement.transformers_max_memory() == {0: "5343MiB", 1: "7765MiB"}
        assert (logged["passes"], logged["lowered_limits_mb"]) == (2, {"cuda:0": 5_343})
        assert _by_device(logged) == {
            "cuda:0": (3_704, 18, 1_440, 7_718, 11_000),
            "cuda:1": (4_309, 22, 1_760, 8_940, 9_000),
        }
        assert {c["device"]: (c["working_mb"], c["staging_mb"]) for c in logged["per_card"]} == {
            "cuda:0": (1_311, 763), "cuda:1": (1_439, 932),
        }

    def test_qwen25_7b_at_32k_is_refused_where_a_long_requests_working_memory_does_not_fit(self, tmp_path):
        """Round 5 served Qwen2.5-7B at 32,768 tokens across these cards with cuda:1 at
        15,100 MB free. A 32,768-token request's transient peak is 142,864 B a token (8 x
        3,584 + 6 x 18,944 + 528) = 4,465 MiB on each card before the allocator's share,
        and no free memory up to 24,000 MB on cuda:1 holds that beside the model. On
        11,500 / 17,800 it is re-planned once and refused: cuda:1 would need 10,820 +
        1,408 of KV + 6,815 of working memory + 500 = 19,543, 1,743 short."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_7B))

        with pytest.raises(InsufficientMemoryError) as raised:
            _plan(((TI_3080, 11_500, 12_288), (RTX_3090, 17_800, 24_576)), path, context=32_768)

        details = raised.value.details
        assert (details["rebalance_passes"], details["short_devices"]) == (2, ["cuda:1"])
        assert [(c["device"], c["weights_mb"], c["layers"], c["kv_mb"], c["working_mb"], c["short_mb"]) for c in details["per_card"]] == [
            ("cuda:0", 3_707, 6, 384, 6_405, 0), ("cuda:1", 10_820, 22, 1_408, 6_815, 1_743),
        ]


class TestWhatNoReplanFits:
    def test_qwen25_14b_cannot_hold_32k_on_these_cards_however_it_is_split(self, tmp_path):
        """Weights 28,173 MiB + a 32,768-token cache of 6,144 + two 500 MB contexts =
        35,317 against 35,000 free, before a request's working memory: no split holds it.
        With a 32,768-token request's transient peak kept on each card, transformers' map
        puts 7,261 MiB on disk at the first pass, and it is refused there."""
        path = _save(tmp_path, Qwen2Config(**QWEN25_14B))
        fit = transformers_fit(path, "FP16", False)
        with patch.object(settings, "TRANSFORMERS_MIN_CONTEXT", 32_768):
            total_need = fit.weights_mb + transformers_fit(path, "FP16", False).kv_mb() + 2 * 500
        assert total_need > 11_500 + 23_500

        with pytest.raises(InsufficientMemoryError) as raised:
            _plan(NODE, path, context=32_768)

        details = raised.value.details
        assert details["short_devices"] == ["cuda:1"]
        assert "rebalance_passes" not in details, "refused at the first pass"
        assert details["mapped_mb_by_device"]["disk"] == 7_261
        assert [(c["device"], c["weights_mb"], c["layers"], c["short_mb"]) for c in details["per_card"]] == [
            ("cuda:0", 3_586, 4, 0), ("cuda:1", 17_327, 33, 5_686),
        ]
        assert "never offloaded" in raised.value.message

    def test_the_search_is_bounded(self, tmp_path):
        """Vicuna-13B needs three passes; allowed two, it is refused, not accepted short."""
        path = _save(tmp_path, LlamaConfig(**VICUNA_13B))

        with patch.object(model_loader, "FIT_REBALANCE_MAX_PASSES", 2), \
                pytest.raises(InsufficientMemoryError) as raised:
            _plan(NODE, path)

        assert raised.value.details["rebalance_passes"] == 2
        assert raised.value.details["short_devices"] == ["cuda:0"]
        assert "did not settle within 2 re-plans" in raised.value.message


class TestTwoShortCardsInOnePass:
    @pytest.mark.parametrize("requested", [None, "all"])
    def test_a_card_that_cannot_be_cut_beside_one_that_is_is_refused_not_a_key_error(
        self, tmp_path, requested
    ):
        """Q8, 32,768 tokens, 8,500 / 24,000 MB free. A 32,768-token request's working
        memory is several GiB on each card, so both cards are short at the first pass:
        cuda:0 holds 4 layers, 1,023 + 2,048 of KV + 7,296 of working memory + 109 of
        staging + 500 = 10,976 of 8,500, short 2,476 — no smaller than its limit, so it
        cannot be cut; cuda:1 holds 28, short 9,459, and is cut. Pass 2 finds cuda:0 short
        at the same weights and doubles its step — the read round 6 found raising KeyError
        — and neither card can give up any more, so the answer is a refusal with the
        split's figures. (Before working memory was counted this shape was 3,000 / 16,000
        MB; with it, that refusal comes at the first pass.)"""
        path = _save(tmp_path, LlamaConfig(**LLAMA2_7B_32K))
        cards = ((TI_3080, 8_500, 12_288), (RTX_3090, 24_000, 24_576))

        with pytest.raises(InsufficientMemoryError) as raised:
            _plan(cards, path, context=32_768, requested=requested, quantization="Q8")

        details = raised.value.details
        assert details["rebalance_passes"] == 2
        assert details["short_devices"] == ["cuda:0", "cuda:1"]
        assert [(c["device"], c["weights_mb"], c["layers"], c["kv_mb"], c["short_mb"]) for c in details["per_card"]] == [
            ("cuda:0", 1_023, 4, 2_048, 2_476), ("cuda:1", 5_655, 28, 14_336, 9_459),
        ]
        assert "cannot give up any more" in raised.value.message


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
        """cuda:0 has 900 MB free: less its 500 MB context and a 4,096-token request's
        transient peak with the allocator's share of it (181 MiB), a 219 MiB budget — less
        than llama-3.2-1b's 490 MiB embedding. The map puts all 2,358 MiB on cuda:1."""
        path = _save(tmp_path, LlamaConfig(**LLAMA32_1B))
        cards = ((TI_3080, 900, 12_288), (RTX_3090, 23_500, 24_576))

        with pytest.raises(SplitNotHonouredError) as raised:
            _plan(cards, path, requested="all")

        details = raised.value.details
        assert details["unused_devices"] == ["cuda:0"]
        assert details["mapped_mb_by_device"] == {"cuda:1": 2_358}
        assert details["max_memory"] == {"cuda:0": "219MiB", "cuda:1": "22819MiB"}


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
