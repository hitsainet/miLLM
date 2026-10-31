"""GGUF placement: one card in llama.cpp single-GPU mode, or a planned layer split.

llama.cpp's default (LLAMA_SPLIT_MODE_LAYER, main_gpu 0) spread every GGUF
model over both cards of the node and put its scratch buffers on the 3080 Ti.
A model that fits one card now goes on the most-free card with
split_mode=NONE and main_gpu=<that card>. One that does not is split over the
most-free cards, only as many as hold it — each counted with its own runtime
overhead — and `tensor_split` names those cards and no other, because a layer
split with no list uses every visible card. When even every card together is
short, the split uses every card and the context ladder shrinks the window.
Nothing is left on the CPU: operator decision 3 allows a partial CPU offload for
GGUF, and none is implemented (review round 2 corrected a claim here that it
"spills").

The context-window prediction is budgeted against the card(s) actually used,
with the runtime overhead counted once per card.

Expected shares were worked out by hand: a card offers int(free x 0.94) - 2048.

MUTATION CONTROLS (each must turn this file red):
  * main_gpu=placement.index -> main_gpu=0                -> "main_gpu is card 1" fails
  * predicted budget from GPU 0 instead of capacity_mb    -> "budgeted against" fails
  * size an explicit card at the target context, not min  -> "shorter context" fails
Phase 2, 2026-09-14 (mutate.py; restored and sha256-verified):
  M10 ignore GGUF_TENSOR_SPLIT (always the plan's shares) -> test_the_operators_split_is_used_in_index_order,
                                                             test_the_operators_split_maps_onto_the_cards_used
  M11 drop the tensor_split length check                  -> test_a_split_with_the_wrong_number_of_values_is_refused_before_loading
  M12 count the runtime overhead once for any split       -> test_each_card_of_a_split_carries_its_own_overhead
  M12b call predicted_max_context without n_cards        -> test_the_context_is_budgeted_against_the_chosen_card,
                                                             test_a_layer_split_is_budgeted_against_every_card,
                                                             test_a_split_is_budgeted_against_the_cards_it_uses
  M21 layer split without tensor_split                    -> test_a_layer_split_names_only_the_cards_it_uses (and 4 more)
  M22 GGUF plans with the transformers rule (no per-card overhead)
                                                          -> test_a_split_takes_only_the_cards_it_needs_each_with_its_own_overhead
                                                             (and 3 more)
  M29 no CPU fallback when no card has room for its overhead
                                                          -> test_no_card_with_room_for_its_overhead_is_the_cpu
Review round 4, 2026-09-14 (mutate.py; restored and sha256-verified). "all" left
out a card with no budget and was accepted over the rest (one card, on this node):
  R4-M1 decide_placement accepts an "all" that leaves a visible card out
                                                          -> test_all_with_a_card_that_has_no_room_for_its_overhead_is_refused_not_left_out
                                                             (and TestAllNamesEveryVisibleCard in test_split_preflight.py)
"""

from unittest.mock import MagicMock, patch

import pytest

from millm.core.config import settings
from millm.core.errors import (
    GpuNotFoundError,
    InsufficientMemoryError,
    ModelLoadError,
    SplitNotHonouredError,
)
from millm.ml import model_loader
from millm.ml.gpu_placement import MODE_CPU, MODE_SHARD, MODE_SINGLE
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576))
THREE = NODE + (("NVIDIA RTX A6000", 40_000, 48_000),)

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
        assert placement.mode == MODE_SHARD
        assert placement.gpu_indices == [0, 1]

    def test_a_split_takes_only_the_cards_it_needs_each_with_its_own_overhead(self, offload):
        """40 GB of weights and 3.1 GB of cache: the A6000 offers 35,552 MB, the
        3090 the remaining 7,573 MB, and the 3080 Ti is not needed."""
        with fake_gpus(*THREE):
            placement = model_loader.plan_gguf_placement(40_000, SMALL_KV, 32_768)
        assert placement.mode == MODE_SHARD
        assert placement.gpu_indices == [1, 2]
        assert placement.planned_mb_by_index == {2: 35_552, 1: 7_573}

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

    def test_all_is_a_split_over_every_card(self, offload):
        with fake_gpus(*NODE):
            placement = model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768, requested="all")
        assert (placement.mode, placement.reason, placement.gpu_indices) == (
            MODE_SHARD, "requested_all_cards", [0, 1],
        )

    def test_all_is_refused_when_the_cards_cannot_hold_even_the_smallest_context(self, offload):
        with fake_gpus((TI_3080, 3_000, 12_288), (RTX_3090, 3_000, 24_576)):
            with pytest.raises(InsufficientMemoryError) as raised:
                model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768, requested="all")
        assert raised.value.details["requested"] == "all"

    def test_all_with_a_card_that_has_no_room_for_its_overhead_is_refused_not_left_out(self, offload):
        """Review round 4, 2026-09-14. int(1,500 x 0.94) - 2,048 < 0: the 3080 Ti has
        no budget, and "all" was planned on the 3090 alone and accepted — "all"
        swapped for one card."""
        with fake_gpus((TI_3080, 1_500, 12_288), (RTX_3090, 23_000, 24_576)):
            with pytest.raises(SplitNotHonouredError) as raised:
                model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768, requested="all")
        assert raised.value.details["unused_devices"] == ["cuda:0"]

    def test_no_card_with_room_for_its_overhead_is_the_cpu(self, offload):
        with fake_gpus((TI_3080, 2_000, 12_288), (RTX_3090, 2_000, 24_576)):
            placement = model_loader.plan_gguf_placement(4_000, SMALL_KV, 32_768)
        assert placement.mode == MODE_CPU

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


def _load(
    tmp_path, kv_bytes, gpu=None, fake=None, consume=None, predicted=None,
    weights_mb=0, tensor_split="", llama=None,
):
    with open(tmp_path / "m-Q4_K_M.gguf", "wb") as handle:
        handle.write(b"\x00" * 64)
        if weights_mb:
            handle.truncate(weights_mb * 1024 * 1024)  # sparse: no disk used
    llama = llama or MagicMock()

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
            patch.object(settings, "GGUF_TENSOR_SPLIT", tensor_split), \
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
        assert "tensor_split" not in kwargs
        assert loaded.device == "cuda:1"
        assert loaded.gpu_indices == [1]
        assert loaded.memory_by_device_mb == {"cuda:1": 6_000}
        assert loaded.placement["mode"] == MODE_SINGLE

    def test_no_single_card_is_a_layer_split_in_proportion_to_each_cards_budget(self, tmp_path, offload):
        """Every card together is short here, so every card is used, in
        proportion to what each offers (8,292 : 19,572 MB)."""
        with fake_gpus(*NODE) as fake:
            loaded, llama, _ = _load(tmp_path, LARGE_KV, fake=fake)
        kwargs = llama.call_args.kwargs
        assert kwargs["split_mode"] == 1, "LLAMA_SPLIT_MODE_LAYER"
        assert kwargs["tensor_split"] == [0.2976, 0.7024]
        assert "main_gpu" not in kwargs
        assert loaded.device == "cuda:0, cuda:1"
        assert loaded.gpu_indices == [0, 1]
        assert loaded.placement["mode"] == MODE_SHARD

    def test_the_context_is_budgeted_against_the_chosen_card(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            _, _, predict = _load(tmp_path, SMALL_KV, fake=fake)
        free_vram_mb = predict.call_args.args[2]
        assert free_vram_mb == 23_000, (
            f"predicted the window from {free_vram_mb} MB; the model is on card 1 "
            "(23000 MB free), and GPU 0 has 11000"
        )
        assert predict.call_args.kwargs["n_cards"] == 1

    def test_a_layer_split_is_budgeted_against_every_card(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            _, _, predict = _load(tmp_path, LARGE_KV, fake=fake)
        assert predict.call_args.args[2] == 34_000
        assert predict.call_args.kwargs["n_cards"] == 2

    def test_a_split_is_budgeted_against_the_cards_it_uses(self, tmp_path, offload):
        with fake_gpus(*THREE) as fake:
            _, _, predict = _load(tmp_path, SMALL_KV, fake=fake, weights_mb=40_000)
        assert predict.call_args.args[2] == 63_000, "the 3090 and the A6000, not the 3080 Ti"
        assert predict.call_args.kwargs["n_cards"] == 2

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
        assert "tensor_split" not in kwargs

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


class TestTensorSplit:
    def test_a_layer_split_names_only_the_cards_it_uses(self, tmp_path, offload):
        """llama.cpp indexes tensor_split by device and, given no list, splits
        over every visible card. Card 0 must get an explicit 0."""
        with fake_gpus(*THREE) as fake:
            loaded, llama, _ = _load(tmp_path, SMALL_KV, fake=fake, weights_mb=40_000)
        kwargs = llama.call_args.kwargs
        assert kwargs["split_mode"] == 1
        assert kwargs["tensor_split"] == [0.0, 0.1756, 0.8244]
        assert loaded.gpu_indices == [1, 2]
        assert loaded.device == "cuda:1, cuda:2"

    def test_the_operators_split_is_used_in_index_order(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            _, llama, _ = _load(tmp_path, LARGE_KV, fake=fake, tensor_split="1,3")
        assert llama.call_args.kwargs["tensor_split"] == [0.25, 0.75]

    def test_the_operators_split_maps_onto_the_cards_used(self, tmp_path, offload):
        with fake_gpus(*THREE) as fake:
            _, llama, _ = _load(tmp_path, SMALL_KV, fake=fake, weights_mb=40_000, tensor_split="1,1")
        assert llama.call_args.kwargs["tensor_split"] == [0.0, 0.5, 0.5]

    def test_a_split_with_the_wrong_number_of_values_is_refused_before_loading(self, tmp_path, offload):
        llama = MagicMock()
        with fake_gpus(*NODE) as fake:
            with pytest.raises(ModelLoadError) as raised:
                _load(tmp_path, LARGE_KV, fake=fake, tensor_split="1,2,3", llama=llama)
        assert raised.value.details["gpu_indices"] == [0, 1]
        assert raised.value.details["tensor_split"] == [1.0, 2.0, 3.0]
        assert "GGUF_TENSOR_SPLIT" in str(raised.value)
        assert not llama.called

    def test_a_model_on_one_card_ignores_the_setting(self, tmp_path, offload):
        with fake_gpus(*NODE) as fake:
            _, llama, _ = _load(tmp_path, SMALL_KV, fake=fake, tensor_split="1,2,3")
        assert llama.call_args.kwargs["split_mode"] == 0
        assert "tensor_split" not in llama.call_args.kwargs


class TestContextPrediction:
    def test_each_card_of_a_split_carries_its_own_overhead(self):
        """34 GB free over two cards, 20 GB of weights, 1 MiB of cache a token:
        31,960 - 20,000 - 2 x 2,048 = 7,864 MB of cache, 7,168 tokens on a 1024
        boundary. Counting the overhead once said 9,216."""
        one_mib = float(1024 * 1024)
        assert model_loader.predicted_max_context(
            "unused.gguf", "q8_0", 34_000, 20_000, bytes_per_token=one_mib, n_cards=2
        ) == 7_168
        assert model_loader.predicted_max_context(
            "unused.gguf", "q8_0", 34_000, 20_000, bytes_per_token=one_mib
        ) == 9_216
