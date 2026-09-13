"""A transformers load goes where the placement says, and cleans up there.

Driven through the REAL ModelLoadContext.load with only the HuggingFace
factories replaced, so the device_map and max_memory under test are the ones
from_pretrained actually receives — not a helper's return value that a caller
might ignore.

Cards are the node's: RTX 3080 Ti (index 0, 11 GB free), RTX 3090 (index 1,
23 GB free). The model is placed on index 1, so any implicit GPU 0 read or
write shows up as a wrong answer.

MUTATION CONTROLS (each must turn this file red):
  * device_map back to "auto" for a single-card load   -> "whole on one card" fails
  * max_memory back to {0: ...}                        -> "keyed to card 1" fails
  * cleanup loop over range(1) / no index              -> "cleans up card 1" fails
  * drop the split-model compile guard                 -> "not compiled" fails
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from millm.ml import model_loader
from millm.ml.gpu_placement import GpuInfo, MODE_ALL, MODE_SINGLE, Placement, choose_gpu
from millm.ml.model_loader import LoadedModel, LoadedModelState, ModelLoadContext
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576))


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


def _load(fake, placement, quantization="FP16", model=None, consume=None, torch_compile=False):
    """Run ModelLoadContext.load; `consume` maps card index -> MB the load takes."""
    model = model or FakeModel([torch.device("cuda", 1)])
    factory = MagicMock()

    def _from_pretrained(path, **kwargs):
        for index, mb in (consume or {}).items():
            fake.free_mb[index] -= mb
        return model

    factory.from_pretrained.side_effect = _from_pretrained
    tokenizer = MagicMock(eos_token="</s>", pad_token="</s>")
    with patch.object(model_loader, "AutoConfig") as auto_config, \
            patch.object(model_loader, "AutoTokenizer") as auto_tokenizer, \
            patch.object(model_loader, "AutoModelForCausalLM", factory), \
            patch.object(model_loader, "BitsAndBytesConfig", MagicMock()), \
            patch.object(model_loader, "get_available_cpu_memory_mb", return_value=32_000):
        auto_config.from_pretrained.side_effect = OSError("no config in the fixture")
        auto_tokenizer.from_pretrained.return_value = tokenizer
        with ModelLoadContext(1, "m") as ctx:
            loaded = ctx.load(
                cache_path="/nonexistent/model",
                quantization=quantization,
                placement=placement,
                torch_compile=torch_compile,
            )
    return loaded, factory.from_pretrained.call_args.kwargs


class TestTheDeviceMap:
    def test_a_model_that_fits_one_card_goes_whole_onto_it(self):
        with fake_gpus(*NODE) as fake:
            placement = choose_gpu(8_000)
            _, kwargs = _load(fake, placement)
        assert kwargs["device_map"] == {"": "cuda:1"}
        assert "max_memory" not in kwargs

    def test_a_model_that_fits_no_single_card_keeps_the_spread(self):
        with fake_gpus(*NODE) as fake:
            placement = choose_gpu(30_000)
            _, kwargs = _load(fake, placement, model=FakeModel([torch.device("cuda", 0), torch.device("cuda", 1)]))
        assert placement.mode == MODE_ALL
        assert kwargs["device_map"] == "auto"

    def test_bitsandbytes_budget_is_keyed_to_the_chosen_card(self):
        with fake_gpus(*NODE) as fake:
            placement = choose_gpu(8_000)
            _, kwargs = _load(fake, placement, quantization="Q8")
        assert kwargs["device_map"] == "auto"
        assert set(kwargs["max_memory"]) == {1, "cpu"}, (
            "max_memory named a card other than the chosen one; bitsandbytes "
            "would place layers there"
        )

    def test_no_placement_given_means_every_card_not_gpu0(self):
        with fake_gpus(*NODE) as fake:
            _, kwargs = _load(fake, None, model=FakeModel([torch.device("cuda", 0), torch.device("cuda", 1)]))
        assert kwargs["device_map"] == "auto"


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

    def test_a_split_model_sums_its_cards(self):
        model = FakeModel([torch.device("cuda", 0), torch.device("cuda", 1)])
        with fake_gpus(*NODE) as fake:
            loaded, _ = _load(fake, choose_gpu(30_000), model=model, consume={0: 8_000, 1: 20_000})
        assert loaded.memory_by_device_mb == {"cuda:0": 8_000, "cuda:1": 20_000}
        assert loaded.memory_used_mb == 28_000
        assert loaded.gpu_indices == [0, 1]


class TestTorchCompile:
    def test_a_model_split_across_devices_is_not_compiled(self):
        model = FakeModel([torch.device("cuda", 0), torch.device("cuda", 1)])
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
