"""The load-time fit check measures capacity where the model will be placed.

FP16/FP32 load with device_map="auto", which spreads a model across every
visible GPU, so capacity is free memory summed over all cards. Checking GPU 0
alone refused a model needing more than 12 GB while the 3090 had 24 GB free.
Q8 is still pinned to GPU 0 by max_memory, so it is checked against GPU 0.
"""

from unittest.mock import MagicMock, patch

import pytest

from millm.core.errors import InsufficientMemoryError
from millm.ml.model_loader import ModelLoader


def _loader():
    loader = ModelLoader()
    loader.state = MagicMock()
    return loader


def _gpus(total_free_mb: int, gpu0_free_mb: int):
    """Two cards: GPU 0 with `gpu0_free_mb`, the rest on the other card."""
    context = MagicMock()
    context.__enter__.return_value.load.return_value = "loaded-model"
    return (
        patch("torch.cuda.is_available", return_value=True),
        patch("millm.ml.model_loader.get_total_free_memory_mb", return_value=total_free_mb),
        patch("millm.ml.model_loader.get_available_memory_mb", return_value=gpu0_free_mb),
        patch(
            "millm.ml.model_loader.list_gpu_memory",
            return_value=[
                {"index": 0, "free_mb": gpu0_free_mb, "total_mb": 12_288},
                {"index": 1, "free_mb": total_free_mb - gpu0_free_mb, "total_mb": 24_576},
            ],
        ),
        patch("millm.ml.model_loader.ModelLoadContext", return_value=context),
    )


def _load(quantization: str, estimated_mb: int):
    return _loader().load(
        model_id=1,
        model_name="test-model",
        cache_path="/tmp/model",
        quantization=quantization,
        estimated_memory_mb=estimated_mb,
    )


class TestFp16FitsAcrossCards:
    def test_a_model_larger_than_gpu0_loads_when_all_cards_have_room(self):
        cuda, total, gpu0, listing, ctx = _gpus(total_free_mb=34_000, gpu0_free_mb=11_000)
        with cuda, total, gpu0, listing, ctx:
            assert _load("FP16", 18_000) == "loaded-model"

    def test_refused_when_all_cards_together_lack_room(self):
        cuda, total, gpu0, listing, ctx = _gpus(total_free_mb=16_000, gpu0_free_mb=11_000)
        with cuda, total, gpu0, listing, ctx:
            with pytest.raises(InsufficientMemoryError) as raised:
                _load("FP16", 18_000)

        assert raised.value.details["available_mb"] == 16_000
        assert [gpu["index"] for gpu in raised.value.details["gpus"]] == [0, 1]


class TestQ8StaysOnGpu0:
    def test_q8_is_checked_against_gpu0(self):
        cuda, total, gpu0, listing, ctx = _gpus(total_free_mb=34_000, gpu0_free_mb=5_000)
        with cuda, total, gpu0, listing, ctx:
            with pytest.raises(InsufficientMemoryError) as raised:
                _load("Q8", 8_000)

        assert raised.value.details["available_mb"] == 5_000
