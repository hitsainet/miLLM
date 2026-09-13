"""The load-time fit check measures capacity where the placement puts the model.

Phase 0 checked FP16/FP32 against free memory summed over every card (they
loaded with device_map="auto") and Q8 against GPU 0 (pinned by
max_memory={0: ...}). Phase 1 decides the card first: a model that fits one
card goes on the most-free such card and fits by construction; only the
all-cards fallback is checked, against the summed free memory. Q8 is no longer
tied to GPU 0.

Cards here are the node's: RTX 3080 Ti (12 GB) at index 0, RTX 3090 (24 GB) at
index 1, with the most free memory on index 1.

MUTATION CONTROLS:
  * check the fallback against one card instead of the sum -> "spread" fails
  * drop the summed-capacity refusal                       -> "refused" fails
  * pin Q8 back to choose_gpu(requested=0)                 -> "q8 on the 3090" fails
"""

from unittest.mock import MagicMock, patch

import pytest

from millm.core.errors import InsufficientMemoryError
from millm.ml.gpu_placement import MODE_ALL, MODE_SINGLE
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
    def test_spread_across_every_card_when_the_sum_holds_it(self):
        with fake_gpus(*NODE):
            kwargs = _load("FP16", 30_000)
        placement = kwargs["placement"]
        assert placement.mode == MODE_ALL
        assert placement.gpu_indices == [0, 1]

    def test_refused_when_all_cards_together_lack_room(self):
        with fake_gpus(*NODE):
            with pytest.raises(InsufficientMemoryError) as raised:
                _load("FP16", 40_000)
        assert raised.value.details["available_mb"] == 34_000
        assert [gpu["index"] for gpu in raised.value.details["gpus"]] == [0, 1]

    def test_q4_can_offload_so_it_is_not_refused(self):
        with fake_gpus(*NODE):
            kwargs = _load("Q4", 40_000)
        assert kwargs["placement"].mode == MODE_ALL


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
