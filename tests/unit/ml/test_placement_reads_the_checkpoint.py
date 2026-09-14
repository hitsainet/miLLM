"""The placement decision reads the checkpoint the load will read.

Review round 1, 2026-09-14, two defects in the decision both checks share
(decide_transformers_placement):

  * `is_pre_quantized` was a parameter no production caller passed. The load
    detects a GPTQ/AWQ checkpoint from its config.json and gives it no
    bitsandbytes, but the plan applied bitsandbytes' 0.9 to every Q4/Q8 row, so
    a split that fits was refused — and since Phase 2 checks every quantization,
    that was a new refusal, not an old one.
  * Q2 (pre-existing): bitsandbytes has no 2-bit mode, so a Q2 checkpoint that
    is not already quantized loads unquantized in bfloat16, while its estimate
    assumed 0.25 bytes a parameter. It is now refused before anything is loaded.

Cards are the node's: RTX 3080 Ti (index 0, 11 GB free), RTX 3090 (index 1,
23 GB free). Split budgets 9,976 + 21,976 = 31,952 MB; with bitsandbytes' 0.9,
8,978 + 19,778 = 28,756 MB. Worked out by hand.

MUTATION CONTROLS (review round 1, 2026-09-14; mutate.py, restored and sha256-verified):
  R1-M3a ModelLoader.load drops `or checkpoint_is_pre_quantized(cache_path)`
         -> test_a_gptq_checkpoint_on_a_q8_row_is_planned_without_bitsandbytes_factor,
            test_a_pre_quantized_q2_checkpoint_is_placed
  R1-M3b checkpoint_is_pre_quantized reads key presence, not a quantization object
         -> test_a_null_quantization_config_is_not_pre_quantized
  R1-M4  drop the Q2 refusal
         -> test_a_q2_checkpoint_that_is_not_quantized_is_refused_before_any_weight (and the
            pre-check's test_a_q2_transformers_checkpoint_is_refused_before_the_unload)
The pre-unload check's half (R1-M3c, R1-M3d, R1-M5, R1-M5c) is recorded in
tests/unit/api/test_load_refusal_keeps_resident_model.py.
Review round 2 moved the reading into plan_transformers_load and re-ran R1-M3a there
(it drops the checkpoint's quantization): 6 red — both tests named above, the
pre-check's three, and the bitsandbytes-checkpoint test in
tests/unit/ml/test_split_preflight.py, where round 2's factor and estimate controls live.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from millm.core.errors import InsufficientMemoryError, UnsupportedQuantizationError
from millm.ml.gpu_placement import MODE_SHARD, MODE_SINGLE
from millm.ml.model_loader import ModelLoader, checkpoint_is_pre_quantized
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576))


def _checkpoint(directory, quantization_config=None, raw=None):
    directory.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (directory / "config.json").write_text(raw)
        return str(directory)
    # kv_lora_rank: a KV cache miLLM cannot size, so the plan is the 20% slack's
    # over the row's estimate — the plan these tests pin the quantizer factor on.
    # A checkpoint it can size is judged per card (Decision 7, test_per_card_fit.py).
    config = {"model_type": "llama", "kv_lora_rank": 1}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (directory / "config.json").write_text(json.dumps(config))
    return str(directory)


def _load(context, cache_path, quantization, estimated_mb):
    loader = ModelLoader()
    loader.state = MagicMock()
    with patch("millm.ml.model_loader.ModelLoadContext", return_value=context):
        loader.load(
            model_id=1,
            model_name="m",
            cache_path=cache_path,
            quantization=quantization,
            estimated_memory_mb=estimated_mb,
        )
    return context.__enter__.return_value.load.call_args.kwargs["placement"]


class TestReadingTheCheckpoint:
    def test_a_quantization_config_means_pre_quantized(self, tmp_path):
        assert checkpoint_is_pre_quantized(_checkpoint(tmp_path, {"quant_method": "awq"}))

    def test_a_null_quantization_config_is_not_pre_quantized(self, tmp_path):
        assert not checkpoint_is_pre_quantized(_checkpoint(tmp_path, raw='{"quantization_config": null}'))

    def test_no_config_or_an_unreadable_one_is_not_pre_quantized(self, tmp_path):
        assert not checkpoint_is_pre_quantized(str(tmp_path / "missing"))
        assert not checkpoint_is_pre_quantized(_checkpoint(tmp_path, raw="{not json"))
        assert not checkpoint_is_pre_quantized(None)


class TestAPreQuantizedCheckpointGetsItsWholeBudget:
    def test_a_gptq_checkpoint_on_a_q8_row_is_planned_without_bitsandbytes_factor(self, tmp_path):
        path = _checkpoint(tmp_path, {"quant_method": "gptq", "bits": 8})
        with fake_gpus(*NODE):
            placement = _load(MagicMock(), path, "Q8", 29_000)
        assert placement.mode == MODE_SHARD
        assert placement.budget_mb == 31_952

    def test_a_plain_checkpoint_on_the_same_row_is_planned_with_it(self, tmp_path):
        context = MagicMock()
        with fake_gpus(*NODE), pytest.raises(InsufficientMemoryError) as raised:
            _load(context, _checkpoint(tmp_path), "Q8", 29_000)
        assert raised.value.details["available_mb"] == 28_756
        assert not context.__enter__.called


class TestQ2:
    def test_a_q2_checkpoint_that_is_not_quantized_is_refused_before_any_weight(self, tmp_path):
        context = MagicMock()
        with fake_gpus(*NODE), pytest.raises(UnsupportedQuantizationError) as raised:
            _load(context, _checkpoint(tmp_path), "Q2", 3_600)
        assert raised.value.status_code == 400
        assert raised.value.details == {
            "quantization": "Q2",
            "estimated_memory_mb": 3_600,
            "loads_as": "bfloat16",
        }
        assert not context.__enter__.called, "nothing may be loaded for a refused quantization"

    def test_a_pre_quantized_q2_checkpoint_is_placed(self, tmp_path):
        path = _checkpoint(tmp_path, {"quant_method": "bitnet"})
        with fake_gpus(*NODE):
            placement = _load(MagicMock(), path, "Q2", 3_600)
        assert (placement.mode, placement.index) == (MODE_SINGLE, 1)
