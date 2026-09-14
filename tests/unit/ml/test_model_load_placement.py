"""A transformers load goes where the placement says, and nowhere else.

Driven through the REAL ModelLoadContext.load with only the HuggingFace
factories replaced, so the device_map and max_memory under test are the ones
from_pretrained actually receives — not a helper's return value that a caller
might ignore.

Cards are the node's: RTX 3080 Ti (index 0, 11 GB free), RTX 3090 (index 1,
23 GB free). A single-card model is placed on index 1, so any implicit GPU 0
read or write shows up as a wrong answer. A split's expected limits were worked
out by hand: free - 1024 per card, every card but the highest index whole (the
order accelerate fills them — review round 1, 2026-09-14), the last card its
whole limit with the remainder planned on it.

MUTATION CONTROLS (each must turn this file red):
  * device_map back to "auto" for a single-card load   -> "whole on one card" fails
  * cleanup loop over range(1) / no index              -> "cleans up card 1" fails
  * drop the split-model compile guard                 -> "not compiled" fails
Phase 2, 2026-09-14 (mutate.py; restored and sha256-verified):
  M1  put `"cpu"` back into a split's max_memory       -> test_a_split_is_limited_to_gpus
  M2  skip the refusal of a model that landed off the GPU
                                                       -> test_a_split_that_landed_on_disk_is_refused_and_released,
                                                          test_a_parameter_left_on_meta_is_refused
  M3a drop the short-circuit that skips class fallbacks on bitsandbytes' refusal
                                                       -> test_bitsandbytes_refusing_the_map_is_a_placement_refusal_not_retried
  M3b drop the conversion of that refusal              -> the same test
  M6  put llm_int8_enable_fp32_cpu_offload=True back   -> test_q8_no_longer_permits_cpu_offload
  M8  hand transformers the factor-discounted limits   -> test_a_bitsandbytes_split_passes_its_limits_undiscounted
Review round 1, 2026-09-14 (mutate.py; restored and sha256-verified):
  R1-M7 the CausalLM fallback retries an offload refusal as AutoModel again
                                                       -> test_a_refusal_from_the_causal_lm_fallback_is_not_retried_as_auto_model
  M3a re-run (the first class's short-circuit dropped) -> test_bitsandbytes_refusing_the_map_is_a_placement_refusal_not_retried
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from millm.core.errors import InsufficientMemoryError
from millm.ml import model_loader
from millm.ml.gpu_placement import (
    MODE_SHARD,
    MODE_SINGLE,
    GpuInfo,
    Placement,
    choose_gpu,
    list_gpus,
)
from millm.ml.model_loader import (
    LoadedModel,
    LoadedModelState,
    ModelLoadContext,
    decide_transformers_placement,
)
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576))
CUDA0, CUDA1 = torch.device("cuda", 0), torch.device("cuda", 1)


class FakeModel:
    """Just enough of a PreTrainedModel for the load path."""

    def __init__(self, devices, hf_device_map=None):
        self._tensors = [SimpleNamespace(device=d) for d in devices]
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map
        self.dtype = torch.bfloat16
        self.forward = MagicMock(name="forward")
        self.generate = MagicMock(side_effect=lambda **_: torch.ones(1, 12, dtype=torch.long))

    def parameters(self):
        return iter(self._tensors)

    def buffers(self):
        return iter([])

    def num_parameters(self):
        return 1_000


@pytest.fixture(autouse=True)
def _reset_state():
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None


def _load(
    fake,
    placement,
    quantization="FP16",
    model=None,
    consume=None,
    torch_compile=False,
    factory=None,
    bnb=None,
    config=None,
):
    """Run ModelLoadContext.load; `consume` maps card index -> MB the load takes."""
    model = model or FakeModel([CUDA1])
    factory = factory or MagicMock()

    def _from_pretrained(path, **kwargs):
        for index, mb in (consume or {}).items():
            fake.free_mb[index] -= mb
        return model

    if factory.from_pretrained.side_effect is None:
        factory.from_pretrained.side_effect = _from_pretrained
    tokenizer = MagicMock(eos_token="</s>", pad_token="</s>")
    with patch.object(model_loader, "AutoConfig") as auto_config, \
            patch.object(model_loader, "AutoTokenizer") as auto_tokenizer, \
            patch.object(model_loader, "AutoModelForCausalLM", factory), \
            patch.object(model_loader, "BitsAndBytesConfig", bnb or MagicMock()):
        if config is None:
            auto_config.from_pretrained.side_effect = OSError("no config in the fixture")
        else:
            auto_config.from_pretrained.return_value = config
        auto_tokenizer.from_pretrained.return_value = tokenizer
        with ModelLoadContext(1, "m") as ctx:
            loaded = ctx.load(
                cache_path="/nonexistent/model",
                quantization=quantization,
                placement=placement,
                torch_compile=torch_compile,
            )
    return loaded, factory.from_pretrained.call_args.kwargs


def _decide(estimated_mb, quantization="FP16"):
    return decide_transformers_placement(estimated_mb, quantization, requested=None, gpus=list_gpus())


class TestTheDeviceMap:
    def test_a_model_that_fits_one_card_goes_whole_onto_it(self):
        with fake_gpus(*NODE) as fake:
            placement = choose_gpu(8_000)
            _, kwargs = _load(fake, placement)
        assert kwargs["device_map"] == {"": "cuda:1"}
        assert "max_memory" not in kwargs

    def test_a_bitsandbytes_model_that_fits_one_card_goes_whole_onto_it_too(self):
        with fake_gpus(*NODE) as fake:
            _, kwargs = _load(fake, _decide(8_000, "Q8"), quantization="Q8")
        assert kwargs["device_map"] == {"": "cuda:1"}
        assert "max_memory" not in kwargs, (
            "a single-card bitsandbytes load needs no budget; the old one carried a cpu entry"
        )

    def test_a_split_is_limited_to_gpus(self):
        with fake_gpus(*NODE) as fake:
            placement = _decide(30_000)
            _, kwargs = _load(fake, placement, model=FakeModel([CUDA0, CUDA1]))
        assert placement.mode == MODE_SHARD
        assert kwargs["device_map"] == "sequential"
        assert kwargs["max_memory"] == {0: "9976MiB", 1: "21976MiB"}

    def test_a_bitsandbytes_split_passes_its_limits_undiscounted(self):
        """transformers applies its 0.9 to these itself; applying it here too
        left a bitsandbytes load 81% of its budget."""
        with fake_gpus(*NODE) as fake:
            _, kwargs = _load(
                fake, _decide(25_000, "Q8"), quantization="Q8", model=FakeModel([CUDA0, CUDA1])
            )
        assert kwargs["device_map"] == "sequential"
        assert kwargs["max_memory"] == {0: "9976MiB", 1: "21976MiB"}

    def test_q8_no_longer_permits_cpu_offload(self):
        bnb = MagicMock()
        with fake_gpus(*NODE) as fake:
            _load(fake, choose_gpu(8_000), quantization="Q8", bnb=bnb)
        assert bnb.call_args.kwargs == {"load_in_8bit": True}

    def test_no_placement_given_means_every_card_gpu_only(self):
        with fake_gpus(*NODE) as fake:
            _, kwargs = _load(fake, None, model=FakeModel([CUDA0, CUDA1]))
        assert kwargs["device_map"] == "auto"
        assert kwargs["max_memory"] == {0: "9976MiB", 1: "21976MiB"}


class TestNothingRunsOffTheGpu:
    def test_a_split_that_landed_on_disk_is_refused_and_released(self):
        """accelerate always adds "disk" as a last device, and transformers serves
        a safetensors "disk" entry from the checkpoint without complaint."""
        model = FakeModel(
            [CUDA0, CUDA1],
            hf_device_map={
                "model.embed_tokens": 0,
                "model.layers.0": 0,
                "model.layers.1": 1,
                "lm_head": "disk",
            },
        )
        with fake_gpus(*NODE) as fake, _cleanup_mocks():
            placement = _decide(30_000)
            with pytest.raises(InsufficientMemoryError) as raised:
                _load(fake, placement, model=model)
            assert torch.cuda.synchronize.call_args_list == [call(0), call(1)], (
                "what was loaded before the refusal must be released"
            )
        assert raised.value.details["off_gpu"] == ["disk"]
        assert raised.value.details["placement"]["mode"] == MODE_SHARD
        assert LoadedModelState().current is None

    def test_a_parameter_left_on_meta_is_refused(self):
        model = FakeModel([CUDA1, torch.device("meta")])
        with fake_gpus(*NODE) as fake, _cleanup_mocks():
            with pytest.raises(InsufficientMemoryError) as raised:
                _load(fake, choose_gpu(8_000), model=model)
        assert raised.value.details["off_gpu"] == ["meta"]

    def test_bitsandbytes_refusing_the_map_is_a_placement_refusal_not_retried(self):
        refusal = ValueError(
            "Some modules are dispatched on the CPU or the disk. Make sure you have "
            "enough GPU RAM to fit the quantized model."
        )
        primary = MagicMock(__name__="Gemma4ForConditionalGeneration")
        primary.from_pretrained.side_effect = refusal
        fallback = MagicMock()
        fallback.from_pretrained.return_value = FakeModel([CUDA0, CUDA1])
        with fake_gpus(*NODE) as fake, _cleanup_mocks(), patch.object(
            model_loader, "_get_auto_model_class", return_value=primary
        ):
            with pytest.raises(InsufficientMemoryError) as raised:
                _load(
                    fake, _decide(25_000, "Q8"), quantization="Q8", factory=fallback,
                    config=SimpleNamespace(quantization_config=None),
                )
        assert primary.from_pretrained.call_count == 1
        assert not fallback.from_pretrained.called, (
            "another model class computes the same map; retrying only repeats the refusal"
        )
        assert "dispatched on the CPU" in raised.value.details["engine_message"]

    def test_a_refusal_from_the_causal_lm_fallback_is_not_retried_as_auto_model(self):
        """The chosen class fails for its own reason, then AutoModelForCausalLM
        computes the map and is refused. AutoModel computes the same map, so
        trying it only repeats the refusal — and when AutoModel fails differently
        (it does not take every config), its error REPLACED the memory refusal.
        Review round 1, 2026-09-14."""
        refusal = ValueError(
            "Some modules are dispatched on the CPU or the disk. Make sure you have "
            "enough GPU RAM to fit the quantized model."
        )
        primary = MagicMock(__name__="Gemma4ForConditionalGeneration")
        primary.from_pretrained.side_effect = RuntimeError("this class cannot take the checkpoint")
        causal_lm = MagicMock()
        causal_lm.from_pretrained.side_effect = refusal
        auto_model = MagicMock()
        auto_model.from_pretrained.side_effect = ValueError("Unrecognized configuration class")
        with fake_gpus(*NODE) as fake, _cleanup_mocks(), patch.object(
            model_loader, "_get_auto_model_class", return_value=primary
        ), patch("transformers.AutoModel", auto_model):
            with pytest.raises(InsufficientMemoryError) as raised:
                _load(
                    fake, _decide(25_000, "Q8"), quantization="Q8", factory=causal_lm,
                    config=SimpleNamespace(quantization_config=None),
                )
        assert causal_lm.from_pretrained.call_count == 1
        assert auto_model.from_pretrained.call_count == 0
        assert "dispatched on the CPU" in raised.value.details["engine_message"]


class TestWhatTheLoadRecords:
    def test_memory_is_the_per_card_drop_on_the_cards_used(self):
        with fake_gpus(*NODE) as fake:
            placement = choose_gpu(8_000)
            loaded, _ = _load(fake, placement, consume={1: 9_000, 0: 300})
        assert loaded.gpu_indices == [1]
        assert loaded.device == "cuda:1"
        assert loaded.memory_by_device_mb == {"cuda:1": 9_000}, (
            "memory must be the drop on the card the model is on — not GPU 0, "
            "and not 'total minus free', which charges the model for the card's "
            "other tenants"
        )
        assert loaded.memory_used_mb == 9_000
        assert loaded.placement["mode"] == MODE_SINGLE
        assert loaded.placement["devices"] == ["cuda:1"]

    def test_a_split_model_sums_its_cards_and_reports_its_plan(self):
        model = FakeModel([CUDA0, CUDA1])
        with fake_gpus(*NODE) as fake:
            loaded, _ = _load(fake, _decide(30_000), model=model, consume={0: 8_000, 1: 20_000})
        assert loaded.memory_by_device_mb == {"cuda:0": 8_000, "cuda:1": 20_000}
        assert loaded.memory_used_mb == 28_000
        assert loaded.gpu_indices == [0, 1]
        assert loaded.placement["mode"] == MODE_SHARD
        assert loaded.placement["planned_mb_by_device"] == {"cuda:0": 9_976, "cuda:1": 20_024}


class TestTorchCompile:
    def test_a_model_split_across_devices_is_not_compiled(self):
        model = FakeModel([CUDA0, CUDA1])
        with fake_gpus(*NODE) as fake, patch("torch.compile") as compile_:
            _load(fake, choose_gpu(30_000), model=model, torch_compile=True)
        assert not compile_.called

    def test_a_single_card_model_still_compiles(self):
        # The soak that follows compile needs a real card and fails here; it
        # reverts to eager, which is its job. Only the compile call is asserted.
        with fake_gpus(*NODE) as fake, patch("torch.compile", return_value=MagicMock()) as compile_:
            _load(fake, choose_gpu(8_000), torch_compile=True)
        assert compile_.call_count == 1


def _cleanup_mocks():
    return patch.multiple(
        "torch.cuda",
        synchronize=MagicMock(),
        empty_cache=MagicMock(),
        ipc_collect=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
    )


class TestCleanupActsOnTheCardsTheModelUsed:
    def test_unload_synchronizes_and_measures_card_1_only(self):
        state = LoadedModelState()
        state.set(
            LoadedModel(1, "m", MagicMock(), MagicMock(), datetime.utcnow(), gpu_indices=[1])
        )
        with fake_gpus(*NODE) as fake, _cleanup_mocks():
            state.clear()
            assert torch.cuda.synchronize.call_args_list == [call(1)]
            assert torch.cuda.reset_peak_memory_stats.call_args_list == [call(1)]
            assert torch.cuda.empty_cache.call_count == 1
        assert {index for _, index in fake.calls} == {1}
        assert state.current is None

    def test_a_split_model_cleans_every_card_it_used(self):
        state = LoadedModelState()
        state.set(
            LoadedModel(1, "m", MagicMock(), MagicMock(), datetime.utcnow(), gpu_indices=[0, 1])
        )
        with fake_gpus(*NODE), _cleanup_mocks():
            state.clear()
            assert torch.cuda.synchronize.call_args_list == [call(0), call(1)]

    def test_nothing_recorded_touches_no_card(self):
        """No card used means no card touched: a torch call on a card creates a
        CUDA context there, taking memory to clean up nothing."""
        state = LoadedModelState()
        state.set(LoadedModel(1, "m", MagicMock(), MagicMock(), datetime.utcnow()))
        with fake_gpus(*NODE) as fake, _cleanup_mocks():
            fake.forbid(0, 1)
            state.clear()
            assert torch.cuda.synchronize.call_args_list == []
            assert torch.cuda.reset_peak_memory_stats.call_args_list == []
        assert fake.calls == []

    def test_a_gguf_model_gets_no_torch_calls_on_its_card(self):
        """llama.cpp's memory is not torch's; torch calls on its card would only
        create a torch context there."""
        from millm.ml.model_loader import ENGINE_LLAMACPP

        handle = MagicMock()
        state = LoadedModelState()
        state.set(LoadedModel(1, "m", handle, None, datetime.utcnow(),
                              gpu_indices=[1], engine=ENGINE_LLAMACPP))
        with fake_gpus(*NODE) as fake, _cleanup_mocks():
            fake.forbid(0, 1)
            state.clear()
            assert torch.cuda.synchronize.call_args_list == []
        assert handle.close.called
        assert fake.calls == []

    def test_a_failed_load_cleans_the_placement_cards(self):
        placement = Placement(
            mode=MODE_SINGLE,
            reason="requested_card",
            required_mb=1,
            gpus=(GpuInfo(1, RTX_3090, None, 24_576, 23_000),),
            index=1,
        )
        with fake_gpus(*NODE), _cleanup_mocks():
            with pytest.raises(RuntimeError):
                with ModelLoadContext(1, "m") as ctx:
                    ctx.gpu_indices = placement.gpu_indices
                    raise RuntimeError("from_pretrained blew up")
            assert torch.cuda.synchronize.call_args_list == [call(1)]
