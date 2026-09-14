"""The load-time fit check measures capacity where the placement puts the model.

Phase 0 checked FP16/FP32 against free memory summed over every card (they
loaded with device_map="auto") and Q8 against GPU 0 (pinned by
max_memory={0: ...}). Phase 1 decided the card first: a model that fits one
card goes on the most-free such card and fits by construction.

Phase 2 (2026-09-14): a model no single card holds is SPLIT across GPUs and
checked against the budgets the load will pass as max_memory — each card's free
memory less SHARD_RESERVE_MB, and for bitsandbytes less transformers' own 0.9.
EVERY quantization is checked: nothing may offload to the CPU any more, so
nothing may skip the check.

Cards here are the node's: RTX 3080 Ti (12 GB) at index 0 with 11 GB free, RTX
3090 (24 GB) at index 1 with 23 GB free. A split's budgets: 9,976 + 21,976 =
31,952 MB (bitsandbytes: 8,978 + 19,778 = 28,756 MB).

MUTATION CONTROLS:
  * check the fallback against one card instead of the sum -> "split" fails
  * drop the summed-capacity refusal                       -> "refused" fails
  * pin Q8 back to choose_gpu(requested=0)                 -> "q8 on the 3090" fails
Phase 2, 2026-09-14 (mutate.py; restored and sha256-verified):
  M7  restore `skip_mem_check` for Q4/Q2/pre-quantized    -> test_every_quantization_is_checked_against_gpu_memory[Q4,Q2], test_a_pre_quantized_model_is_checked_too
  M19 per-card limit without SHARD_RESERVE_MB              -> test_each_cards_reserve_is_counted, test_refused_when_the_split_lacks_room
  M20 plan bitsandbytes without its 0.9                    -> test_bitsandbytes_is_planned_with_its_own_09_once
"""

from unittest.mock import MagicMock, patch

import pytest

from millm.core.errors import InsufficientMemoryError
from millm.ml.gpu_placement import MODE_SHARD, MODE_SINGLE
from millm.ml.model_loader import ModelLoader
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus


def _loader():
    loader = ModelLoader()
    loader.state = MagicMock()
    return loader


def _load(quantization: str, estimated_mb: int, gpu=None, **extra):
    """Run ModelLoader.load with the context patched; return the ctx.load kwargs."""
    context = MagicMock()
    context.__enter__.return_value.load.return_value = "loaded-model"
    with patch("millm.ml.model_loader.ModelLoadContext", return_value=context):
        result = _loader().load(
            model_id=1,
            model_name="test-model",
            cache_path="/tmp/model",
            quantization=quantization,
            estimated_memory_mb=estimated_mb,
            gpu=gpu,
            **extra,
        )
    assert result == "loaded-model"
    return context.__enter__.return_value.load.call_args.kwargs


NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576))


class TestAModelThatFitsOneCard:
    def test_fp16_larger_than_gpu0_goes_whole_onto_the_3090(self):
        with fake_gpus(*NODE):
            kwargs = _load("FP16", 18_000)
        placement = kwargs["placement"]
        assert (placement.mode, placement.index) == (MODE_SINGLE, 1)

    def test_q8_is_no_longer_checked_against_gpu0(self):
        """Phase 0 refused this: 8 GB against GPU 0's 5 GB, with 23 GB on the 3090."""
        with fake_gpus((TI_3080, 5_000, 12_288), (RTX_3090, 23_000, 24_576)):
            kwargs = _load("Q8", 8_000)
        assert kwargs["placement"].index == 1


class TestAModelThatFitsNoSingleCard:
    def test_split_across_the_cards_when_their_budgets_hold_it(self):
        with fake_gpus(*NODE):
            kwargs = _load("FP16", 30_000)
        placement = kwargs["placement"]
        assert placement.mode == MODE_SHARD
        assert placement.gpu_indices == [0, 1]
        assert placement.planned_mb_by_index == {0: 9_976, 1: 20_024}

    def test_refused_when_the_split_lacks_room(self):
        with fake_gpus(*NODE):
            with pytest.raises(InsufficientMemoryError) as raised:
                _load("FP16", 40_000)
        assert raised.value.details["available_mb"] == 31_952
        assert raised.value.details["budget_mb_by_device"] == {"cuda:0": 9_976, "cuda:1": 21_976}
        assert [gpu["index"] for gpu in raised.value.details["gpus"]] == [0, 1]
        assert "never offloaded to the CPU" in str(raised.value)

    def test_each_cards_reserve_is_counted(self):
        """34 GB are free in total, but a split holds 31,952 MB once each card
        keeps room for its CUDA context and activations."""
        with fake_gpus(*NODE):
            with pytest.raises(InsufficientMemoryError):
                _load("FP16", 33_000)

    # Q2 is refused before the memory check unless the checkpoint is
    # pre-quantized (review round 1): tests/unit/ml/test_placement_reads_the_checkpoint.py.
    @pytest.mark.parametrize("quantization", ["Q4", "Q8"])
    def test_every_quantization_is_checked_against_gpu_memory(self, quantization):
        with fake_gpus(*NODE):
            with pytest.raises(InsufficientMemoryError):
                _load(quantization, 40_000)

    def test_a_pre_quantized_model_is_checked_too(self):
        with fake_gpus(*NODE):
            with pytest.raises(InsufficientMemoryError):
                _load("FP16", 40_000, is_pre_quantized=True)

    def test_bitsandbytes_is_planned_with_its_own_09_once(self):
        """29 GB fits the FP16 budgets (31,952 MB) but not bitsandbytes' (28,756 MB)."""
        with fake_gpus(*NODE):
            assert _load("FP16", 29_000)["placement"].is_shard
        with fake_gpus(*NODE):
            with pytest.raises(InsufficientMemoryError) as raised:
                _load("Q8", 29_000)
        assert raised.value.details["available_mb"] == 28_756


class TestAnExplicitCard:
    def test_honoured_when_it_fits(self):
        with fake_gpus(*NODE):
            kwargs = _load("FP16", 8_000, gpu=0)
        assert (kwargs["placement"].index, kwargs["placement"].reason) == (0, "requested_card")

    def test_refused_when_it_does_not_and_nothing_is_loaded(self):
        context = MagicMock()
        with fake_gpus(*NODE), patch(
            "millm.ml.model_loader.ModelLoadContext", return_value=context
        ):
            with pytest.raises(InsufficientMemoryError) as raised:
                _loader().load(
                    model_id=1,
                    model_name="m",
                    cache_path="/tmp/m",
                    quantization="FP16",
                    estimated_memory_mb=18_000,
                    gpu=0,
                )
        assert raised.value.details["available_mb"] == 11_000
        assert not context.__enter__.return_value.load.called, (
            "a refused card must stop the load before any weights are read"
        )

    def test_all_is_honoured_for_a_model_one_card_holds(self):
        with fake_gpus(*NODE):
            kwargs = _load("FP16", 2_870, gpu="all")
        placement = kwargs["placement"]
        assert (placement.mode, placement.reason, placement.gpu_indices) == (
            MODE_SHARD, "requested_all_cards", [0, 1],
        )

    def test_all_is_refused_before_anything_loads_when_the_cards_lack_room(self):
        context = MagicMock()
        with fake_gpus(*NODE), patch("millm.ml.model_loader.ModelLoadContext", return_value=context):
            with pytest.raises(InsufficientMemoryError):
                _loader().load(1, "m", "/tmp/m", "FP16", 40_000, gpu="all")
        assert not context.__enter__.return_value.load.called
