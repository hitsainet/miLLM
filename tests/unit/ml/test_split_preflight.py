"""A split is judged by the device map transformers will compute, before any weight is read.

Review round 2, 2026-09-14. Round 1 made a split fill its cards in index order,
so accelerate's held-back largest layer lands on memory that exists. That holds
only while the estimate carries enough slack over the weights: transformers maps
MODULES, holding back room for the largest layer on the lowest-index card and
stranding the tail of every card but the last. Run over meta-device models of
six shapes, an estimate within 5% of the weights put lm_head on disk under
accepted plans, and the production estimate did for a Q4 "all" split near
capacity. An FP16 map to disk was found only after every weight had loaded; a
bitsandbytes one after the resident model had been unloaded.

Everything here runs transformers' REAL map inference over a meta-device model
built from a real config.json (no weights, no GPU). Expected figures were worked
out by hand from the shapes, as follows.

Llama, 70B widths, 16 layers, untied, bf16 (the FIXTURE below):
  embed_tokens = lm_head = 128,256 x 8,192 x 2 B = 2,004 MiB, the largest layer
  one decoder layer = 855,752,704 params x 2 B = 1,632.2 MiB (int 1,632)
  row estimate 31,626 MB (5% over the weights) on cards of 11,500 / 23,500 free:
    no card holds it; the split fills index 0 first -> max_memory 10,476 / 22,476
    card 0 holds back 2,004 -> embed + 3 layers = 6,900; the 4th would be 8,533
    card 1 takes 13 layers = 21,216; 1,260 left, lm_head needs 2,004 -> disk
  with 1,000 MB more on card 1 (24,500 free): 2,260 left -> lm_head fits (23,220)

MUTATION CONTROLS (review round 2, 2026-09-14; mutate.py: one replacement, the 11
placement/load/inference test files run, file restored and its sha256 verified):
  R2-M1  plan_shard's "all" index-order rule disabled
         -> both TestAllNearCapacityMapsOntoTheGpus tests
            (+ test_gpu_placement::test_a_model_the_lower_cards_cannot_hold_is_filled_like_auto)
  R2-M2  decide_transformers_placement ignores the pre-quantized factor
         -> test_a_bitsandbytes_checkpoint_gets_the_09_its_quantizer_applies
  R2-M2b the factor lookup drops bitsandbytes' _4bit/_8bit suffix
         -> the two bitsandbytes factor cases and the same checkpoint test
  R2-M2c plan_transformers_load does not pass the factor
         -> the same checkpoint test (+ the wiring test)
  R2-M3  a pre-quantized checkpoint keeps the row's estimate
         -> test_sized_from_its_weights_not_its_rows_label
  R2-M3b the weights are counted from every weight file
         -> test_sharded_files_without_an_index_are_summed
  R2-M4  ModelLoader.load skips the preflight
         -> test_model_loader_load_runs_the_preflight (+ the wiring test)
  R2-M5  the pre-check skips the preflight
         -> test_a_split_that_would_map_to_disk_is_refused_and_the_served_model_kept (+ wiring)
  R2-M6  the preflight never finds a device off the GPU
         -> the disk-map test, the loader test and the pre-check test
  R2-M6b the preflight skips bitsandbytes' own refusal
         -> test_refused_from_the_quantizers_own_refusal
  R2-M11 the generic INSUFFICIENT_MEMORY message is back in error_messages
         -> the pre-check test (+ test_exception_handlers)
Round 1 controls re-run on lines this round moved:
  R1-M1  transformers splits fill most-free-first again -> 17 red, 7 of them in this file
  M16    the highest-index card capped at its share     -> 20 red, 9 of them in this file

The sweep behind this file (a scratch script, not a test: 6 shapes x FP16/Q8/Q4 x 19
depths x 7 card sets, real map inference). Accepted plans that mapped a module off
their GPUs, of 2,394 per cell, before -> after this round's plan_shard fix:
  production estimate   Auto 0 -> 0     "all" 1 -> 0
  weights + 5%          Auto 4 -> 4     "all" 9 -> 2
  exact weights         Auto 25 -> 25   "all" 43 -> 16
What remains is invisible to a plan in MB; the preflight computes the map itself.

REVIEW ROUND 3, 2026-09-14. The preflight's map was checked against the map
from_pretrained itself computes (its `_get_device_map` wrapped to record and abort,
tied and untied checkpoints, three budgets each): identical in all six cases.

Round 2's sizing of a pre-quantized checkpoint from the weights it stores was
wrong for a quantizer that DEQUANTIZES on these cards (FP8 below compute
capability 8.9). Mutation controls (mutate.py; restored and sha256-verified):
  R3-M4  the estimate uses the stored weights only — round 2's code
         -> test_dequantized_on_these_cards_it_is_not_placed_on_one,
            test_a_bitsandbytes_checkpoint_gets_the_09_its_quantizer_applies (+ the wiring test)
  R3-M4b the materialised size skips the quantizer's module replacement
         -> test_kept_in_fp8_it_is_sized_by_what_it_stores, the bitsandbytes factor test
  R3-M4c the materialised size is computed without the quantizer -> the bitsandbytes factor test
  R3-M4g the quantizer is dropped after it decided (AttributeError, swallowed as unknown)
         -> the dequantized test, the bitsandbytes factor test
test_a_bitsandbytes_checkpoint_gets_the_09_its_quantizer_applies now uses a
52-layer checkpoint: round 2's 16-layer one had no weights and was sized from its
row's 29,000; sized as it loads (12,643 MB) it no longer reached the 0.9 budgets.

"all" could come out on ONE card of transformers' real map (TestAllIsHonouredOrRefused);
the preflight now refuses that before anything is unloaded:
  R3-M5  the "all" check disabled
         -> the two refusal tests and test_the_pre_check_refuses_it_before_the_unload
  R3-M5b the check widened to every split, Auto included
         -> test_an_auto_split_that_lands_on_fewer_cards_is_not_refused
  R3-M5c unused cards looked up in the map's own keys (so never any) -> the same three as R3-M5
  R2-M6 re-run (the off-GPU refusal the new check follows) -> the disk-map, loader and pre-check tests

REVIEW ROUND 4, 2026-09-14. Round 3's check looked for an unused card among the
PLANNED cards, and "all" planned only cards with a budget: a card too full to take
a share was left out and "all" ran on the rest (TestAllNamesEveryVisibleCard).
Mutation controls (mutate.py; restored and sha256-verified):
  R4-M1  decide_placement accepts an "all" that leaves a visible card out
         -> both TestAllNamesEveryVisibleCard tests (+ the GGUF case in test_gguf_placement.py)
  R3-M5 re-run (the preflight's own check, now reached only with every card planned)
         -> the two refusal tests and test_the_pre_check_refuses_it_before_the_unload

A failure in transformers' own machinery (a private function gone or changed)
logged the same warning as an unverifiable checkpoint, so the preflight could go
dark on every load unnoticed (TestAnEngineFailureIsNotAnUnverifiableCheckpoint):
  R4-M6  the preflight's map call logged as the checkpoint's -> test_the_device_map_call_raising_is_an_error_of_its_own
  R4-M6b its private imports counted as the checkpoint's     -> test_the_private_function_gone_is_an_error_of_its_own
  R4-M6c it never enters the checkpoint stage                -> test_an_unverifiable_checkpoint_is_still_only_a_warning
  R4-M6d _log_unverified logs an engine failure as the warning -> the three engine-failure tests
  R4-M7  the materialised sizing's engine stage never entered -> test_the_materialised_size_engine_failure_is_an_error_too
  R4-M7b it never enters the checkpoint stage                -> test_an_unbuildable_checkpoint_is_still_an_ordinary_unknown
  R3-M4b re-run (the line beside the new stage marker)      -> the bitsandbytes factor test, test_kept_in_fp8_...
  R3-X6 re-run (the offload refusal branch the logging now follows)
         -> test_refused_from_the_quantizers_own_refusal

DECISION 7, 2026-09-14: a transformers load is judged per card (test_per_card_fit.py).
The plan now sizes a checkpoint as it loads and refuses a map to disk itself, so the
tests here that exercise the PREFLIGHT, or the slack's estimate and factor, use a
checkpoint whose KV cache miLLM cannot size (`_checkpoint(..., unsized=True)`), which
keeps the slack's plan; F-M5 and F-M15 (test_per_card_fit.py) turn exactly those red.
Re-derived under the per-card fit: the FP8 pair (30,120 MiB dequantized with lm_head on
disk; 17,823 on the 3090 kept) and the "all" tests (the 7B Q4 share is now 1,835 MiB
from its 5,191 MiB of weights; gemma-3-1b's pre-check leaves cuda:0 empty, not cuda:1).
Round 3's two slack-planned "all" directions keep their plans, built directly.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("bitsandbytes")
from fastapi.testclient import TestClient  # noqa: E402
from transformers import BitsAndBytesConfig, LlamaConfig  # noqa: E402
from transformers.integrations.accelerate import _get_device_map, compute_module_sizes  # noqa: E402
from transformers.quantizers.auto import AutoHfQuantizer  # noqa: E402

from millm.core.errors import InsufficientMemoryError, SplitNotHonouredError  # noqa: E402
from millm.db.models.model import ModelStatus, QuantizationType  # noqa: E402
from millm.main import create_app  # noqa: E402
from millm.ml.gpu_placement import MODE_SHARD, MODE_SINGLE, GpuInfo  # noqa: E402
from millm.ml.model_loader import (  # noqa: E402
    LoadedModel,
    ModelLoader,
    checkpoint_weights_mb,
    decide_transformers_placement,
    plan_transformers_load,
    pre_quantized_max_memory_factor,
    preflight_split,
)
from millm.services.model_service import ModelService  # noqa: E402
from tests.support.factories import make_model  # noqa: E402
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus  # noqa: E402

MIB = 1024 * 1024
WIDE = dict(
    vocab_size=128_256, hidden_size=8_192, intermediate_size=28_672,
    num_attention_heads=64, num_key_value_heads=8, tie_word_embeddings=False,
)
ESTIMATE_MB = 31_626
SHORT = ((TI_3080, 11_500, 12_288), (RTX_3090, 23_500, 24_576))
ROOMY = ((TI_3080, 11_500, 12_288), (RTX_3090, 24_500, 24_576))


def _checkpoint(directory, layers=16, quantization_config=None, unsized=False):
    """WIDE at `layers` layers, config only.

    `unsized` adds the field multi-head latent attention sets (`kv_lora_rank`).
    miLLM then cannot size the checkpoint's KV cache, so the plan is the 20%
    slack's — the plan these tests were written against, whose splits only the
    preflight can see past. A checkpoint it CAN size is judged per card by the
    plan itself (Decision 7, test_per_card_fit.py), which refuses the same disk
    map before the preflight is reached.
    """
    directory.mkdir(parents=True, exist_ok=True)
    LlamaConfig(num_hidden_layers=layers, **WIDE).save_pretrained(directory)
    path = directory / "config.json"
    config = json.loads(path.read_text())
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    if unsized:
        config["kv_lora_rank"] = 1
    path.write_text(json.dumps(config))
    return str(directory)


def _plan(cards, estimate=ESTIMATE_MB, quantization="FP16", requested=None, cache_path=None):
    with fake_gpus(*cards):
        from millm.ml.gpu_placement import list_gpus

        return plan_transformers_load(
            estimate, quantization, requested=requested, gpus=list_gpus(), cache_path=cache_path
        )


class TestThePreflightReadsTheRealMap:
    def test_a_split_whose_lm_head_would_go_to_disk_is_refused_with_the_map(self, tmp_path):
        path = _checkpoint(tmp_path, unsized=True)
        placement = _plan(SHORT, cache_path=path)
        assert placement.mode == MODE_SHARD
        assert placement.transformers_max_memory() == {0: "10476MiB", 1: "22476MiB"}
        assert placement.budget_mb >= ESTIMATE_MB, "the estimate alone accepts this split"

        with pytest.raises(InsufficientMemoryError) as raised:
            preflight_split("wide-16", path, "FP16", placement)

        details = raised.value.details
        assert details["off_gpu"] == ["disk"]
        assert details["mapped_mb_by_device"] == {"cuda:0": 6_901, "cuda:1": 21_217, "disk": 2_004}
        assert details["before_loading"] is True

    def test_the_same_split_with_room_for_lm_head_maps_onto_the_gpus(self, tmp_path):
        path = _checkpoint(tmp_path)
        placement = _plan(ROOMY, cache_path=path)
        assert placement.transformers_max_memory() == {0: "11000MiB", 1: "24000MiB"}

        assert preflight_split("wide-16", path, "FP16", placement) == {
            "cuda:0": 8_533,
            "cuda:1": 21_589,
        }

    def test_nothing_to_compute_is_not_a_refusal(self, tmp_path):
        """No config, or one no class builds, leaves the check to the load."""
        placement = _plan(SHORT)
        assert preflight_split("m", str(tmp_path / "missing"), "FP16", placement) is None
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "not-a-real-model"}))
        assert preflight_split("m", str(tmp_path), "FP16", placement) is None

    def test_a_single_card_is_not_mapped(self, tmp_path):
        path = _checkpoint(tmp_path, unsized=True)
        placement = _plan(SHORT, estimate=8_000, cache_path=path)
        assert placement.mode == MODE_SINGLE
        assert preflight_split("wide-16", path, "FP16", placement) is None

    def test_the_preflight_creates_no_cuda_context(self, tmp_path):
        path = _checkpoint(tmp_path)
        with fake_gpus(*ROOMY) as fake:
            from millm.ml.gpu_placement import list_gpus

            placement = plan_transformers_load(
                ESTIMATE_MB, "FP16", requested=None, gpus=list_gpus(), cache_path=path
            )
            fake.forbid(0, 1)
            assert preflight_split("wide-16", path, "FP16", placement) is not None
        assert fake.calls == []


class TestTheLoaderRefusesBeforeReadingAWeight:
    def test_model_loader_load_runs_the_preflight(self, tmp_path):
        path = _checkpoint(tmp_path, unsized=True)
        context = MagicMock()
        loader = ModelLoader()
        loader.state = MagicMock()
        with fake_gpus(*SHORT), patch("millm.ml.model_loader.ModelLoadContext", return_value=context):
            with pytest.raises(InsufficientMemoryError) as raised:
                loader.load(
                    model_id=1, model_name="wide-16", cache_path=path,
                    quantization="FP16", estimated_memory_mb=ESTIMATE_MB,
                )
        assert raised.value.details["mapped_mb_by_device"]["disk"] == 2_004
        assert not context.__enter__.called, "no weight may be read for a split the map refuses"

    def test_a_split_that_maps_onto_the_gpus_is_loaded_with_that_placement(self, tmp_path):
        path = _checkpoint(tmp_path)
        context = MagicMock()
        loader = ModelLoader()
        loader.state = MagicMock()
        with fake_gpus(*ROOMY), patch("millm.ml.model_loader.ModelLoadContext", return_value=context):
            loader.load(
                model_id=1, model_name="wide-16", cache_path=path,
                quantization="FP16", estimated_memory_mb=ESTIMATE_MB,
            )
        assert context.__enter__.return_value.load.call_count == 1
        placement = context.__enter__.return_value.load.call_args.kwargs["placement"]
        assert placement.transformers_max_memory() == {0: "11000MiB", 1: "24000MiB"}


class TestThePreCheckRefusesBeforeTheUnload:
    """The resident model holds 16,000 MB of card 1; projected, card 1 has 23,500."""

    CARDS = ((TI_3080, 11_500, 12_288), (RTX_3090, 7_500, 24_576))

    @staticmethod
    def _service(model):
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=model)
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

    def _post(self, tmp_path):
        from millm.api.dependencies import get_model_service

        model = make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.FP16,
            estimated_memory_mb=ESTIMATE_MB, cache_path=_checkpoint(tmp_path, unsized=True),
        )
        svc = self._service(model)
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        with fake_gpus(*self.CARDS):
            response = TestClient(app).post("/api/models/3/load", json={})
        return svc, response

    def test_a_split_that_would_map_to_disk_is_refused_and_the_served_model_kept(self, tmp_path):
        svc, response = self._post(tmp_path)

        assert response.status_code == 507, response.text
        error = response.json()["error"]
        assert error["code"] == "INSUFFICIENT_MEMORY"
        assert error["details"]["mapped_mb_by_device"] == {
            "cuda:0": 6_901, "cuda:1": 21_217, "disk": 2_004,
        }
        assert "would run from disk" in error["message"], "the toast shows the refusal, not a generic sentence"
        assert not svc.unload_model.called
        assert not svc._executor.method_calls and not svc._executor.called


class TestAllNearCapacityMapsOntoTheGpus:
    """Q4, 70B widths, 48 layers, on cards of 11,000 / 20,000 free, estimated as
    miLLM estimates it (params x 0.5 B x 1.2 = 24,719 MB).

    Limits 9,976 / 18,976; bitsandbytes budgets int(x 0.9) = 8,978 / 17,078. The
    card below the highest index holds 8,978 < 24,719, so filling in index order
    already reaches card 1: card 0 is planned whole (8,978 -> max_memory
    ceil(8,978 / 0.9) = 9,976) and card 1 keeps its whole limit. Proportional,
    card 0 was capped at ceil(ceil(24,719 x 8,978 / 26,056) / 0.9) = 9,465 and
    lm_head (bf16, 2,004 MiB) went to disk."""

    def test_the_plan_gives_each_card_its_whole_limit(self):
        cards = [
            GpuInfo(index=0, name=TI_3080, uuid=None, total_mb=12_288, free_mb=11_000),
            GpuInfo(index=1, name=RTX_3090, uuid=None, total_mb=24_576, free_mb=20_000),
        ]
        placement = decide_transformers_placement(24_719, "Q4", requested="all", gpus=cards)
        assert placement.planned_mb_by_index == {0: 8_978, 1: 15_741}
        assert placement.transformers_max_memory() == {0: "9976MiB", 1: "18976MiB"}

    def test_the_real_map_uses_both_cards_and_nothing_else(self):
        cards = [
            GpuInfo(index=0, name=TI_3080, uuid=None, total_mb=12_288, free_mb=11_000),
            GpuInfo(index=1, name=RTX_3090, uuid=None, total_mb=24_576, free_mb=20_000),
        ]
        placement = decide_transformers_placement(24_719, "Q4", requested="all", gpus=cards)
        with torch.device("meta"):
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_config(
                LlamaConfig(num_hidden_layers=48, **WIDE), dtype=torch.bfloat16
            )
        quantizer = AutoHfQuantizer.from_config(
            BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            ),
            pre_quantized=False,
        )
        quantizer.preprocess_model(
            model=model, dtype=torch.bfloat16, device_map="sequential",
            checkpoint_files=None, use_kernels=False,
        )
        with patch("torch.cuda.device_count", return_value=2):
            device_map = _get_device_map(
                model, placement.transformers_device_map(),
                dict(placement.transformers_max_memory()), quantizer,
            )
        sizes, _ = compute_module_sizes(model, quantizer)
        assert sizes["lm_head"] / MIB == pytest.approx(2_004, abs=1), "the fixture's largest layer"
        assert set(device_map.values()) == {0, 1}


class TestAPreQuantizedCheckpointIsPlannedWithItsOwnQuantizer:
    @pytest.mark.parametrize(
        "config, factor",
        [
            ({"quant_method": "bitsandbytes", "load_in_4bit": True}, 0.9),
            ({"quant_method": "bitsandbytes", "load_in_8bit": True}, 0.9),
            ({"quant_method": "bitnet"}, 0.9),
            ({"quant_method": "awq", "bits": 4}, 1.0),
            # GPTQ's quantizer cannot even be constructed here (optimum is not
            # installed): the factor is read from the class, not an instance.
            ({"quant_method": "gptq", "bits": 4}, 1.0),
            ({"quant_method": "made-up"}, 1.0),
            (None, 1.0),
        ],
    )
    def test_the_factor_is_what_transformers_quantizer_applies(self, config, factor):
        assert pre_quantized_max_memory_factor(config) == pytest.approx(factor)

    def test_a_bitsandbytes_checkpoint_gets_the_09_its_quantizer_applies(self, tmp_path):
        """Budgets on these cards: 9,976 + 21,976 = 31,952 whole; with 0.9,
        8,978 + 19,778 = 28,756. The row says FP16; the checkpoint says bitsandbytes.

        Sized as transformers loads it (review round 3 — round 2's fixture had no
        weights and was sized from its row's 29,000): WIDE at 52 layers, 4-bit.
          a layer's linears 855,638,016 params x 0.5 B + two bf16 norms 32,768 B
            = 427,851,776 B; embed_tokens + lm_head stay bf16 = 4,202,692,608 B
          4,202,692,608 + 52 x 427,851,776 + 16,384 = 26,451,001,344 B = 25,225 MiB
          x1.2 = 30,270 MB: inside the whole budgets, outside the 0.9 ones."""
        path = _checkpoint(tmp_path, layers=52, unsized=True, quantization_config={
            "quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        })
        with pytest.raises(InsufficientMemoryError) as raised:
            _plan(((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576)), 29_000, cache_path=path)
        assert raised.value.details["available_mb"] == 28_756
        assert raised.value.details["required_mb"] == 30_270

    def test_an_awq_checkpoint_keeps_its_whole_budget(self, tmp_path):
        path = _checkpoint(tmp_path, unsized=True, quantization_config={"quant_method": "awq", "bits": 4})
        placement = _plan(((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576)), 29_000, cache_path=path)
        assert placement.budget_mb == 31_952


class TestAPreQuantizedCheckpointIsSizedByWhatItStores:
    def test_sized_from_its_weights_not_its_rows_label(self, tmp_path):
        """A 4-bit GPTQ checkpoint on an FP16 row: 18 GiB stored -> 18,432 x 1.2
        = 22,118 MB, which the 3090's 23,000 holds. At the row's label (74,387 MB
        for 32.5B params) it was refused outright."""
        path = _checkpoint(tmp_path, unsized=True, quantization_config={"quant_method": "gptq", "bits": 4})
        with open(tmp_path / "model.safetensors", "wb") as handle:
            handle.truncate(18 * 1024 ** 3)  # sparse
        placement = _plan(((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576)), 74_387, cache_path=path)
        assert (placement.mode, placement.index, placement.required_mb) == (MODE_SINGLE, 1, 22_118)

    def test_a_checkpoint_that_is_not_pre_quantized_keeps_the_rows_estimate(self, tmp_path):
        path = _checkpoint(tmp_path, unsized=True)
        with open(tmp_path / "model.safetensors", "wb") as handle:
            handle.truncate(18 * 1024 ** 3)
        with pytest.raises(InsufficientMemoryError) as raised:
            _plan(((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576)), 74_387, cache_path=path)
        assert raised.value.details["required_mb"] == 74_387

    def test_the_index_total_is_used_and_a_second_copy_is_not_counted(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 5 * 1024 ** 3}, "weight_map": {}})
        )
        with open(tmp_path / "consolidated.safetensors", "wb") as handle:
            handle.truncate(5 * 1024 ** 3)
        assert checkpoint_weights_mb(str(tmp_path)) == 5_120

    def test_sharded_files_without_an_index_are_summed(self, tmp_path):
        for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            with open(tmp_path / shard, "wb") as handle:
                handle.truncate(3 * 1024 ** 3)
        with open(tmp_path / "consolidated.safetensors", "wb") as handle:
            handle.truncate(6 * 1024 ** 3)
        assert checkpoint_weights_mb(str(tmp_path)) == 6_144
        assert checkpoint_weights_mb(str(tmp_path / "missing")) == 0


class TestABitsandbytesMapIsRefusedByTheQuantizerItself:
    """transformers' bitsandbytes quantizer refuses a map with a CPU or disk entry
    inside `_get_device_map` (validate_environment), before any weight. The
    preflight must turn that into the placement refusal, not a skipped check.

    Q4, 70B widths, 48 layers: a layer is 855.7M params x 0.5 B = 408 MiB,
    embed_tokens and lm_head stay bf16 at 2,004 MiB each. Row estimate 24,719 MB.
      cards 11,000 / 18,700 free: budgets 8,978 + 15,908 = 24,886, the plan accepts;
        card 0: 8,978 - 2,004 held back - 2,004 embed -> 12 layers
        card 1: 36 layers = 14,677 of 15,908 -> 1,231 left, lm_head 2,004 -> disk
      cards 11,000 / 20,000 free: card 1 has 17,078 -> lm_head fits"""

    def test_refused_from_the_quantizers_own_refusal(self, tmp_path):
        path = _checkpoint(tmp_path, layers=48, unsized=True)
        placement = _plan(((TI_3080, 11_000, 12_288), (RTX_3090, 18_700, 24_576)), 24_719, "Q4", cache_path=path)
        assert placement.budget_mb == 24_886

        with pytest.raises(InsufficientMemoryError) as raised:
            preflight_split("wide-48-q4", path, "Q4", placement)

        details = raised.value.details
        assert details["off_gpu"] == ["cpu or disk"]
        assert details["before_loading"] is True
        assert "dispatched on the CPU or the disk" in details["engine_message"]

    def test_the_same_model_with_room_maps_onto_both_cards(self, tmp_path):
        path = _checkpoint(tmp_path, layers=48)
        placement = _plan(((TI_3080, 11_000, 12_288), (RTX_3090, 20_000, 24_576)), 24_719, "Q4", cache_path=path)

        mapped = preflight_split("wide-48-q4", path, "Q4", placement)

        assert set(mapped) == {"cuda:0", "cuda:1"}


FP8 = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": [128, 128]}


class TestACheckpointTransformersDequantizesIsSizedAsItLoads:
    """Review round 3, 2026-09-14. Round 2 sized a pre-quantized checkpoint from
    the weights it stores. transformers does not always load them that way: the
    FineGrainedFP8 quantizer DEQUANTIZES to bf16 on a card below compute
    capability 8.9, and both of this node's cards are 8.6. Measured on a
    Qwen2.5-14B-shaped FP8 checkpoint: 15,575 MiB kept, 28,171 MiB dequantized,
    so sized from its files (x1.2 = 18,690 MB) it was planned whole onto the
    3090 and would run out of memory mid-load, after the resident model left.

    The FIXTURE: Llama, 70B widths, 16 layers, untied (WIDE). Dequantized it is
    plain bf16, worked by hand:
      embed_tokens = lm_head = 128,256 x 8,192 = 1,050,673,152 params each
      one decoder layer: q, o 8,192 x 8,192 = 67,108,864 each; k, v 8,192 x 1,024
      = 8,388,608 each; gate, up, down 8,192 x 28,672 = 234,881,024 each; two
      norms 8,192 each -> 855,654,400 params; final norm 8,192
      total 2,101,346,304 + 16 x 855,654,400 + 8,192 = 15,791,824,896 params
      x 2 B = 31,583,649,792 B = 30,120 MiB -> x1.2 = 36,144 MB
    Stored: a sparse 16 GiB model.safetensors -> x1.2 = 19,660 MB, which the
    3090's 23,500 MB free holds whole. Dequantized, no single card and no split
    of these cards (budgets 10,476 + 22,476 = 32,952 MB) holds it.

    Since Decision 7 (2026-09-14) the plan sizes it as it loads and judges it per
    card, with no x1.2. Dequantized: 30,121 MiB, which no card holds, so it splits
    (round 4's 1,024 MB reserve per card put lm_head on disk; review round 5 budgets
    each card at free less its CUDA context, and the split holds it). Kept in FP8: each layer's
    linears 855,638,016 B at 1 B, their block scales (128 x 128 blocks: q, o
    4,096; k, v 512; gate, up, down 14,336 -> 52,224 x 4 B) and two bf16 norms
    (32,768 B) = 855,879,680 B; 16 layers + bf16 embed_tokens and lm_head
    (4,202,692,608 B) + final norm (16,384 B) = 17,896,783,872 B = 17,068 MiB (rounded up);
    + KV 128 MiB (16 layers x 2 x 8 x 128 x 2 B x 2,048 tokens — LlamaConfig's
    max_position_embeddings, below the 4,096 floor; review round 5) + 500 MiB
    context = 17,696 on the 3090."""

    CARDS = ((TI_3080, 11_500, 12_288), (RTX_3090, 23_500, 24_576))

    @staticmethod
    def _fp8_checkpoint(tmp_path):
        path = _checkpoint(tmp_path, quantization_config=FP8)
        with open(tmp_path / "model.safetensors", "wb") as handle:
            handle.truncate(16 * 1024 ** 3)  # sparse
        return path

    def test_dequantized_on_these_cards_it_is_not_placed_on_one(self, tmp_path):
        import millm.ml.model_loader as module

        path = self._fp8_checkpoint(tmp_path)
        with patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
                patch.object(module, "logger") as logger:
            placement = _plan(self.CARDS, 19_660, cache_path=path)
        assert placement.mode == MODE_SHARD
        [accepted] = [c for c in logger.info.call_args_list if c.args == ("transformers_fit_split_accepted",)]
        weights = {card["device"]: card["weights_mb"] for card in accepted.kwargs["per_card"]}
        assert weights == {"cuda:0": 8_533, "cuda:1": 21_589}, "30,122 MiB of bf16 across both cards"

    def test_kept_in_fp8_it_is_sized_by_what_it_stores(self, tmp_path):
        """On a card that runs FP8 the checkpoint is not refused for a bf16 size
        it never takes: one card holds it."""
        path = self._fp8_checkpoint(tmp_path)
        with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
            placement = _plan(self.CARDS, 19_660, cache_path=path)
        assert (placement.mode, placement.index) == (MODE_SINGLE, 1)
        assert placement.required_mb == 17_696


# Real configurations' shapes (config.json fields), built on the meta device only.
QWEN25_7B = dict(
    vocab_size=152_064, hidden_size=3_584, intermediate_size=18_944, num_hidden_layers=28,
    num_attention_heads=28, num_key_value_heads=4, tie_word_embeddings=False,
)
GEMMA3_1B = dict(
    vocab_size=262_144, hidden_size=1_152, intermediate_size=6_912, num_hidden_layers=26,
    num_attention_heads=4, num_key_value_heads=1, head_dim=256, tie_word_embeddings=True,
)
LLAMA32_1B = dict(
    vocab_size=128_256, hidden_size=2_048, intermediate_size=8_192, num_hidden_layers=16,
    num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=True,
)


def _config_checkpoint(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(directory)
    return str(directory)


class TestAllIsHonouredOrRefused:
    """Review round 3, 2026-09-14. "all" promises a split across every card,
    honoured or refused, never swapped. The plan divides MB; transformers places
    whole layers, in index order, keeping room for the largest one free on the
    lowest-index card. Swept on its real map inference, "all" came out on ONE card
    for every small Q4 model even with both cards idle, and for Qwen2.5-7B FP16 —
    the equivalence-check model — whenever the 3080 Ti was busy. Both of
    plan_shard's "all" branches do it, including round 2's index-order one.

    Plan figures, worked by hand:
      Qwen2.5-7B Q4, cards 11,500 / 23,500, sized as it loads (Decision 7) and
        budgeted per card (review round 5: limits = free - the 500 MB CUDA context):
        a layer's 4-bit linears 233,046,016 params x 0.5 B + bf16 biases and
        norms 23,552 B = 116,546,560 B; 28 layers + bf16 embed_tokens and lm_head
        1,089,994,752 B each + final norm 7,168 B = 5,443,300,352 B = 5,192 MiB
        rounded up; need 5,192 + 224 MiB of KV = 5,416. bitsandbytes budgets
        int(x 0.9) = 9,900 / 20,700; 5,416 <= 9,900, so proportional: card 0 share
        ceil(5,416 x 9,900 / 30,600) = 1,753 -> max_memory ceil(1,753 / 0.9) = 1,948.
        transformers gives 1,753 back and holds room for the largest layer, the
        untied bf16 embedding (1,039 MiB), so the embedding — first in order — does
        not fit card 0, and nothing lands there. (Round 3 planned the row's 4,348.)
      gemma-3-1b FP16, cards 3,000 / 23,500, row estimate 2,288: limits 1,976 /
        22,476; 2,288 > 1,976, so round 2's index-order branch: card 0 whole
        (1,976). The ~1,907 MiB of weights fit it, and card 1 gets nothing.
    """

    IDLE = ((TI_3080, 11_500, 12_288), (RTX_3090, 23_500, 24_576))
    BUSY = ((TI_3080, 3_000, 12_288), (RTX_3090, 23_500, 24_576))

    def test_a_card_whose_share_holds_no_layer_is_refused(self, tmp_path):
        """Refused by the plan itself, from the layout the per-card fit computes
        (review round 5); the preflight refused it one step later before."""
        from transformers import Qwen2Config

        path = _config_checkpoint(tmp_path, Qwen2Config(**QWEN25_7B))

        with pytest.raises(SplitNotHonouredError) as raised:
            _plan(self.IDLE, 4_348, "Q4", requested="all", cache_path=path)

        details = raised.value.details
        assert details["unused_devices"] == ["cuda:0"]
        assert details["max_memory"] == {"cuda:0": "1948MiB", "cuda:1": "23000MiB"}
        assert details["mapped_mb_by_device"] == {"cuda:1": 5_192}
        assert details["before_loading"] is True

    def test_a_first_card_that_holds_the_whole_model_is_refused(self, tmp_path):
        """The other direction, on the plan the 20% slack makes (a model whose KV
        cache miLLM cannot size is still planned that way). Sized as it loads,
        "all" plans from the weights, whose smaller share never let card 0 take
        everything in a sweep of four shapes over ten card-0 budgets (review round
        4) — so the slack's plan is built here directly."""
        from transformers import Gemma3TextConfig

        from millm.ml.gpu_placement import list_gpus

        path = _config_checkpoint(tmp_path, Gemma3TextConfig(**GEMMA3_1B))
        with fake_gpus(*self.BUSY):
            placement = decide_transformers_placement(2_288, "FP16", requested="all", gpus=list_gpus())
        assert placement.transformers_max_memory() == {0: "1976MiB", 1: "22476MiB"}

        with pytest.raises(SplitNotHonouredError) as raised:
            preflight_split("gemma-3-1b", path, "FP16", placement)

        assert raised.value.details["unused_devices"] == ["cuda:1"]
        assert set(raised.value.details["mapped_mb_by_device"]) == {"cuda:0"}

    def test_a_split_that_reaches_every_card_is_honoured(self, tmp_path):
        from transformers import LlamaConfig

        path = _config_checkpoint(tmp_path, LlamaConfig(**LLAMA32_1B))
        placement = _plan(self.IDLE, 2_746, requested="all", cache_path=path)

        mapped = preflight_split("llama-3.2-1b", path, "FP16", placement)

        assert set(mapped) == {"cuda:0", "cuda:1"}

    def test_an_auto_split_that_lands_on_fewer_cards_is_not_refused(self, tmp_path):
        """Only "all" promises every card. Llama at 7B widths, 26 layers, 32,000
        vocab, untied, worked by hand: a layer 4 x 4,096^2 + 3 x 4,096 x 11,008 +
        8,192 = 202,383,360 params = 386 MiB; embed_tokens + lm_head 2 x 250 MiB;
        500 + 26 x 386.02 = 10,536.5, 10,537 MiB rounded up. Row estimate x1.2 = 12,643, more than
        either card has free (12,000 / 11,000), so Auto plans both
        (max_memory 10,976 / 9,976) — and the weights fit the first card whole.
        That load runs on one card and fits; refusing it would turn away a model
        the node holds.

        That plan is the slack's. Judged per card (Decision 7) an Auto split that
        lands on one card is refused on that card instead: it needed a split
        because the most-free card could not hold the model with its context, so
        no card can. The preflight still guards a load the slack plans."""
        from transformers import LlamaConfig

        path = _config_checkpoint(tmp_path, LlamaConfig(
            vocab_size=32_000, hidden_size=4_096, intermediate_size=11_008, num_hidden_layers=26,
            num_attention_heads=32, num_key_value_heads=32, tie_word_embeddings=False,
        ))
        cards = ((TI_3080, 12_000, 12_288), (RTX_3090, 11_000, 24_576))
        from millm.ml.gpu_placement import list_gpus

        with fake_gpus(*cards):
            placement = decide_transformers_placement(12_643, "FP16", requested=None, gpus=list_gpus())
        assert placement.mode == MODE_SHARD
        assert placement.transformers_max_memory() == {0: "10976MiB", 1: "9976MiB"}

        assert preflight_split("llama-7b-widths-26", path, "FP16", placement) == {"cuda:0": 10_537}

    def test_the_pre_check_refuses_it_before_the_unload(self, tmp_path):
        """The resident model holds 16,000 MB of card 1: projected 3,000 / 23,500.
        Sized as it loads (Decision 7), gemma-3-1b is 1,908 MiB of weights and 104 of
        KV cache at 4,096 tokens; budgeted per card (review round 5) the limits are
        free less the 500 MB context, 2,500 / 23,000, and 2,012 <= 2,500, so
        proportional: card 0 share ceil(2,012 x 2,500 / 25,500) = 198, under its
        576 MiB embedding — card 0 takes nothing, and the plan refuses it from the
        layout."""
        from transformers import Gemma3TextConfig

        from millm.api.dependencies import get_model_service

        model = make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.FP16,
            estimated_memory_mb=2_288,
            cache_path=_config_checkpoint(tmp_path, Gemma3TextConfig(**GEMMA3_1B)),
        )
        svc = TestThePreCheckRefusesBeforeTheUnload._service(model)
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        with fake_gpus((TI_3080, 3_000, 12_288), (RTX_3090, 7_500, 24_576)):
            response = TestClient(app).post("/api/models/3/load", json={"gpu": "all"})

        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "SPLIT_NOT_HONOURED"
        assert error["details"]["unused_devices"] == ["cuda:0"]
        assert "cuda:0" in error["message"]
        assert not svc.unload_model.called
        assert not svc._executor.method_calls and not svc._executor.called


class TestAllNamesEveryVisibleCard:
    """Review round 4, 2026-09-14. Round 3's check looks for a card with nothing
    among the cards the PLAN named. plan_shard's "all" uses "every card with any
    budget", so a card with less than the 1,024 MB overhead free — the 3080 Ti
    while a miStudio job holds it — was never named: the plan was the 3090 alone,
    the map filled it, and nothing was unused. Probed on llama-3.2-1b FP16 with
    card 0 at 900 MB free: planned [1], mapped {"cuda:1": 2,357}, accepted as
    "all". At 1,100 MB free the same request was refused by round 3's check.

    Since review round 5 a sized split budgets each card at its free memory less
    its 500 MB CUDA context, so a card too full to take any share is one with 500 MB
    free or less: 400 MB here. A card whose budget is too small for any layer (900 MB)
    is refused from the layout instead (TestAllIsHonouredOrRefused)."""

    BUSY_CARD_0 = ((TI_3080, 400, 12_288), (RTX_3090, 23_500, 24_576))

    def test_a_card_with_no_room_to_take_part_is_refused_not_left_out(self, tmp_path):
        path = _config_checkpoint(tmp_path, LlamaConfig(**LLAMA32_1B))

        with pytest.raises(SplitNotHonouredError) as raised:
            _plan(self.BUSY_CARD_0, 2_746, requested="all", cache_path=path)

        details = raised.value.details
        assert details["unused_devices"] == ["cuda:0"]
        assert details["requested"] == "all"
        assert "cuda:0" in raised.value.message and "400 MB" in raised.value.message

    def test_the_pre_check_refuses_it_before_the_unload(self, tmp_path):
        """The resident model holds 16,000 MB of card 1: projected 400 / 23,500."""
        from millm.api.dependencies import get_model_service

        model = make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.FP16,
            estimated_memory_mb=2_746,
            cache_path=_config_checkpoint(tmp_path, LlamaConfig(**LLAMA32_1B)),
        )
        svc = TestThePreCheckRefusesBeforeTheUnload._service(model)
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        with fake_gpus((TI_3080, 400, 12_288), (RTX_3090, 7_500, 24_576)):
            response = TestClient(app).post("/api/models/3/load", json={"gpu": "all"})

        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "SPLIT_NOT_HONOURED"
        assert error["details"]["unused_devices"] == ["cuda:0"]
        assert not svc.unload_model.called
        assert not svc._executor.method_calls and not svc._executor.called

class TestAnEngineFailureIsNotAnUnverifiableCheckpoint:
    """Review round 4, 2026-09-14 (a leftover from round 3). The preflight and the
    materialised sizing call transformers PRIVATE functions (`_get_device_map`,
    `compute_module_sizes`). When one of those changes under an upgrade, every
    split preflight skipped with the same warning as a checkpoint whose config no
    class builds — so the check went quietly dark on every load and read as a run
    of odd checkpoints. A failure inside transformers' own machinery, after the
    checkpoint's config, class and quantizer were all built, is now a separate
    ERROR event naming the call and the transformers version."""

    def test_the_device_map_call_raising_is_an_error_of_its_own(self, tmp_path):
        import millm.ml.model_loader as module

        path = _checkpoint(tmp_path, unsized=True)
        placement = _plan(SHORT, cache_path=path)
        changed = TypeError("_get_device_map() takes 3 positional arguments but 4 were given")
        with patch("transformers.integrations.accelerate._get_device_map", side_effect=changed), \
                patch.object(module, "logger") as logger:
            assert preflight_split("wide-16", path, "FP16", placement) is None

        [error] = logger.error.call_args_list
        assert error.args == ("split_preflight_engine_failed",)
        assert error.kwargs["error_type"] == "TypeError"
        assert error.kwargs["transformers_version"]
        assert "split_preflight_skipped" not in [c.args[0] for c in logger.warning.call_args_list]

    def test_the_private_function_gone_is_an_error_of_its_own(self, tmp_path, monkeypatch):
        import transformers.integrations.accelerate as accelerate

        import millm.ml.model_loader as module

        path = _checkpoint(tmp_path, unsized=True)
        placement = _plan(SHORT, cache_path=path)
        monkeypatch.delattr(accelerate, "_get_device_map")
        with patch.object(module, "logger") as logger:
            assert preflight_split("wide-16", path, "FP16", placement) is None

        assert [c.args[0] for c in logger.error.call_args_list] == ["split_preflight_engine_failed"]
        assert logger.error.call_args.kwargs["error_type"] == "ImportError"

    def test_an_unverifiable_checkpoint_is_still_only_a_warning(self, tmp_path):
        import millm.ml.model_loader as module

        placement = _plan(SHORT)
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "not-a-real-model"}))
        with patch.object(module, "logger") as logger:
            assert preflight_split("m", str(tmp_path), "FP16", placement) is None

        assert [c.args[0] for c in logger.warning.call_args_list] == ["split_preflight_skipped"]
        assert not logger.error.called

    def test_the_materialised_size_engine_failure_is_an_error_too(self, tmp_path):
        """Its fallback is the stored weights, which for FP8 on these cards is the
        mid-load out-of-memory round 3 fixed: an API change must not look like an
        ordinary unknown."""
        import millm.ml.model_loader as module

        path = _checkpoint(tmp_path, quantization_config=FP8)
        changed = TypeError("compute_module_sizes() got an unexpected keyword argument")
        with patch("transformers.integrations.accelerate.compute_module_sizes", side_effect=changed), \
                patch.object(module, "logger") as logger:
            assert module.checkpoint_materialised_mb(path) == 0

        assert [c.args[0] for c in logger.error.call_args_list] == ["checkpoint_materialised_engine_failed"]
        assert "checkpoint_materialised_size_unknown" not in [
            c.args[0] for c in logger.warning.call_args_list
        ]

    def test_an_unbuildable_checkpoint_is_still_an_ordinary_unknown(self, tmp_path):
        import millm.ml.model_loader as module

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "not-a-real-model", "quantization_config": FP8})
        )
        with patch.object(module, "logger") as logger:
            assert module.checkpoint_materialised_mb(str(tmp_path)) == 0

        assert [c.args[0] for c in logger.warning.call_args_list] == ["checkpoint_materialised_size_unknown"]
        assert not logger.error.called
