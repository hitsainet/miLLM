"""GGUF placement: one card in llama.cpp single-GPU mode, or a layer split.

llama.cpp's default (LLAMA_SPLIT_MODE_LAYER, main_gpu 0) spread every GGUF
model over both cards of the node and put its scratch buffers on the 3080 Ti.
A model that fits one card now goes on the most-free card with
split_mode=NONE and main_gpu=<that card>; one that does not keeps the layer
split, and CPU spill stays allowed (operator decision 3 — GGUF only).

The context-window prediction is budgeted against the card(s) actually used.

MUTATION CONTROLS (each must turn this file red):
  * main_gpu=placement.index -> main_gpu=0                -> "main_gpu is card 1" fails
  * predicted budget from GPU 0 instead of capacity_mb    -> "budgeted against" fails
  * size an explicit card at the target context, not min  -> "shorter context" fails
"""

from unittest.mock import MagicMock, patch

import pytest

from millm.core.config import settings
from millm.core.errors import GpuNotFoundError, InsufficientMemoryError
from millm.ml import model_loader
from millm.ml.gpu_placement import MODE_ALL, MODE_CPU, MODE_SINGLE
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576))

#: ~0.1 MB of KV per token: 32768 tokens is ~3.1 GB, which fits either card.
SMALL_KV = 100_000.0
#: 1 MB per token: 32768 tokens is ~31 GB, which fits no single card.
LARGE_KV = 1_000_000.0


@pytest.fixture
def offload():
    with patch.object(model_loader, "llama_supports_gpu_offload", lambda: True):
        yield


class TestPlan:
    def test_fits_one_card_goes_on_the_most_free(self, offload):
        with fake_gpus(*NODE):
            placement = model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768)
        assert (placement.mode, placement.index) == (MODE_SINGLE, 1)

    def test_fits_no_single_card_is_a_layer_split(self, offload):
        with fake_gpus(*NODE):
            placement = model_loader.plan_gguf_placement(4_000, LARGE_KV, 32_768)
        assert placement.mode == MODE_ALL
        assert placement.gpu_indices == [0, 1]

    def test_an_explicit_card_is_sized_at_the_shorter_context(self, offload):
        """Card 0 cannot hold 32768 tokens of this cache, but holds 2048 — and
        the ladder will shrink the window to fit, so the card is honoured."""
        with fake_gpus(*NODE):
            placement = model_loader.plan_gguf_placement(4_000, LARGE_KV, 32_768, requested=0)
        assert (placement.mode, placement.index) == (MODE_SINGLE, 0)

    def test_an_explicit_card_that_cannot_hold_even_that_is_refused(self, offload):
        with fake_gpus((TI_3080, 3_000, 12_288), (RTX_3090, 23_000, 24_576)):
            with pytest.raises(InsufficientMemoryError) as raised:
                model_loader.plan_gguf_placement(4_000, LARGE_KV, 32_768, requested=0)
        assert raised.value.details["gpu"]["index"] == 0

    def test_no_card_is_the_cpu(self, offload):
        with fake_gpus():
            placement = model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768)
        assert placement.mode == MODE_CPU

    def test_a_cpu_only_build_is_the_cpu_even_with_cards(self):
        with fake_gpus(*NODE), patch.object(
            model_loader, "llama_supports_gpu_offload", lambda: False
        ):
            placement = model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768)
        assert placement.mode == MODE_CPU

    def test_a_named_card_on_a_cpu_only_build_is_refused_not_run_on_cpu(self):
        with fake_gpus(), patch.object(model_loader, "llama_supports_gpu_offload", lambda: True):
            with pytest.raises(InsufficientMemoryError):
                model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768, requested=1)

    def test_a_named_card_that_fits_on_a_cpu_only_build_is_refused(self):
        """The case the test above is named for: cards EXIST and the named one
        has room, but the llama.cpp build offloads nothing.

        This returned choose_gpu's single-card placement, so the model ran on
        the CPU under a placement reading "requested_card". The test above has
        no cards at all, so it passed against that defect.

        MUTATION CONTROL (round 1, 2026-09-13): `return choose_gpu(...)` in place
        of the raise -> this test and the GGUF_GPU_LAYERS=0 one fail.
        """
        with fake_gpus(*NODE), patch.object(
            model_loader, "llama_supports_gpu_offload", lambda: False
        ):
            with pytest.raises(InsufficientMemoryError) as raised:
                model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768, requested=1)
        assert raised.value.details["gpu_offload"] is False
        assert raised.value.details["requested"] == 1

    def test_a_named_card_with_gpu_layers_off_is_refused(self, offload):
        with fake_gpus(*NODE), patch.object(model_loader, "GGUF_GPU_LAYERS", 0):
            with pytest.raises(InsufficientMemoryError) as raised:
                model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768, requested=1)
        assert raised.value.details["gguf_gpu_layers"] == 0

    def test_an_unknown_card_on_a_cpu_only_build_is_still_not_found(self):
        with fake_gpus(*NODE), patch.object(
            model_loader, "llama_supports_gpu_offload", lambda: False
        ):
            with pytest.raises(GpuNotFoundError):
                model_loader.plan_gguf_placement(
                    4_000, SMALL_KV, 32_768,
                    requested="GPU-00000000-0000-0000-0000-000000000000",
                )


def _load(tmp_path, kv_bytes, gpu=None, fake=None, consume=None, predicted=None):
    (tmp_path / "m-Q4_K_M.gguf").write_bytes(b"\x00" * 64)
    llama = MagicMock()

    def _construct(**kwargs):
        for index, mb in (consume or {}).items():
            fake.free_mb[index] -= mb
        return MagicMock(close=MagicMock())

    llama.side_effect = _construct
    predict = MagicMock(return_value=predicted)
    with patch.object(model_loader, "Llama", llama), \
            patch.object(model_loader, "declared_context", return_value=32_768), \
            patch.object(model_loader, "gguf_kv_bytes_per_token", return_value=kv_bytes), \
            patch.object(model_loader, "predicted_max_context", predict), \
            patch.object(settings, "GGUF_ENABLE_EMBEDDINGS", False), \
            patch.object(settings, "GGUF_CONTEXT_LENGTH", 32_768):
        loaded = model_loader.load_gguf_model(1, "m", str(tmp_path), "m-Q4_K_M.gguf", gpu=gpu)
    return loaded, llama, predict


class TestLoad:
    def test_one_card_single_gpu_mode_on_that_card(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            loaded, llama, _ = _load(tmp_path, SMALL_KV, fake=fake, consume={1: 6_000})
        kwargs = llama.call_args.kwargs
        assert kwargs["split_mode"] == 0, "LLAMA_SPLIT_MODE_NONE"
        assert kwargs["main_gpu"] == 1
        assert loaded.device == "cuda:1"
        assert loaded.gpu_indices == [1]
        assert loaded.memory_by_device_mb == {"cuda:1": 6_000}
        assert loaded.placement["mode"] == MODE_SINGLE

    def test_no_single_card_keeps_the_layer_split(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            loaded, llama, _ = _load(tmp_path, LARGE_KV, fake=fake)
        kwargs = llama.call_args.kwargs
        assert kwargs["split_mode"] == 1, "LLAMA_SPLIT_MODE_LAYER"
        assert "main_gpu" not in kwargs
        assert loaded.device == "cuda:0, cuda:1"
        assert loaded.gpu_indices == [0, 1]

    def test_the_context_is_budgeted_against_the_chosen_card(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            _, _, predict = _load(tmp_path, SMALL_KV, fake=fake)
        free_vram_mb = predict.call_args.args[2]
        assert free_vram_mb == 23_000, (
            f"predicted the window from {free_vram_mb} MB; the model is on card 1 "
            "(23000 MB free), and GPU 0 has 11000"
        )

    def test_a_layer_split_is_budgeted_against_every_card(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            _, _, predict = _load(tmp_path, LARGE_KV, fake=fake)
        assert predict.call_args.args[2] == 34_000

    def test_an_explicit_card_is_honoured(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            loaded, llama, _ = _load(tmp_path, SMALL_KV, gpu=0, fake=fake)
        assert llama.call_args.kwargs["main_gpu"] == 0
        assert loaded.placement["reason"] == "requested_card"

    def test_a_cpu_placement_offloads_no_layers(self, tmp_path, offload):
        """A placement that says CPU must RUN on the CPU.

        CUDA and a GPU-capable llama.cpp build are present, but the inventory is
        empty (nvidia-smi absent, failing or past its 5 s timeout), so placement
        answers `cpu`. The kwargs still carried n_gpu_layers=-1 with no
        split_mode/main_gpu, so llama.cpp spread every layer over every card
        with its default main_gpu 0 while the load recorded device "cpu", no
        gpu_indices and no per-card memory — the pre-Phase-1 defect, unreported.

        MUTATION CONTROL (round 2, 2026-09-13): pass GGUF_GPU_LAYERS whatever
        the placement -> this test fails.
        """
        with fake_gpus(*NODE, smi_available=False) as fake:
            loaded, llama, _ = _load(tmp_path, SMALL_KV, fake=fake)
        kwargs = llama.call_args.kwargs
        assert loaded.placement["mode"] == MODE_CPU
        assert loaded.device == "cpu"
        assert kwargs["n_gpu_layers"] == 0, (
            f"placement is cpu but llama.cpp was asked to offload "
            f"n_gpu_layers={kwargs['n_gpu_layers']}"
        )
        assert "main_gpu" not in kwargs and "split_mode" not in kwargs

    def test_a_card_placement_still_offloads_every_layer(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            _, llama, _ = _load(tmp_path, SMALL_KV, fake=fake)
        assert llama.call_args.kwargs["n_gpu_layers"] == model_loader.GGUF_GPU_LAYERS

    def test_a_refused_card_reaches_the_caller_as_itself_and_nothing_loads(self, tmp_path, offload):
        with fake_gpus((TI_3080, 3_000, 12_288), (RTX_3090, 23_000, 24_576)) as fake:
            with pytest.raises(InsufficientMemoryError):
                _load(tmp_path, LARGE_KV, gpu=0, fake=fake)

    def test_through_the_model_loader_entry_point(self, tmp_path, offload):
        """ModelLoader.load must hand the request to the GGUF path."""
        (tmp_path / "m-Q4_K_M.gguf").write_bytes(b"\x00" * 64)
        loader = model_loader.ModelLoader()
        loader.state = MagicMock()
        with fake_gpus(*NODE), \
                patch.object(model_loader, "Llama", MagicMock()) as llama, \
                patch.object(model_loader, "declared_context", return_value=None), \
                patch.object(model_loader, "gguf_kv_bytes_per_token", return_value=SMALL_KV), \
                patch.object(settings, "GGUF_ENABLE_EMBEDDINGS", False):
            loader.load(1, "m", str(tmp_path), "Q4", 0, gguf_file="m-Q4_K_M.gguf", gpu=0)
        assert llama.call_args.kwargs["main_gpu"] == 0
