"""Placement queries create no CUDA context on cards the model does not use.

`torch.cuda.mem_get_info(i)` initialises a CUDA context on card i — a few
hundred MB per card, kept for the life of the process. Reading every card that
way to decide where a model goes took memory from every card; miStudio places
its own jobs by live free memory on the same node, so a phantom miLLM context on
the other card was memory it could not use. The inventory now comes from
nvidia-smi (no context) and is mapped to torch indices by UUID.

The fake GPUs here RAISE when mem_get_info is called for a forbidden card.

MUTATION CONTROLS (each must turn this file red):
  * list_gpus reads free memory with mem_get_info(i) over every card
  * list_gpus maps cards by nvidia-smi position instead of UUID
  * GGUF measures its cards through torch instead of nvidia-smi
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from millm.core.config import settings
from millm.core.errors import InsufficientMemoryError
from millm.ml import model_loader, nvidia_smi
from millm.ml.gpu_placement import MODE_CPU, choose_gpu, list_gpus
from tests.support.fake_gpus import NODE_UUIDS, RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576))


class TestNvidiaSmiParsing:
    def test_the_free_column_is_parsed(self):
        stdout = (
            f"0, {NODE_UUIDS[0]}, 3, 1288, 12288, 11000, 41, {TI_3080}\n"
            f"1, {NODE_UUIDS[1]}, 60, 1576, 24576, 23000, 66, {RTX_3090}\n"
        )
        cards = nvidia_smi.parse_nvidia_smi_gpus(stdout)
        assert [(c["index"], c["memory_free_mb"], c["memory_total_mb"], c["name"]) for c in cards] == [
            (0, 11_000, 12_288, TI_3080),
            (1, 23_000, 24_576, RTX_3090),
        ]

    def test_the_query_asks_for_free_memory_and_ends_with_the_name(self):
        fields = nvidia_smi.NVIDIA_SMI_QUERY.split(",")
        assert "memory.free" in fields
        assert fields[-1] == "name"

    def test_absent_nvidia_smi_is_no_cards(self):
        with patch("millm.ml.nvidia_smi.subprocess.run", side_effect=FileNotFoundError):
            assert nvidia_smi.query_gpus() == []

    def test_a_failing_nvidia_smi_is_no_cards(self):
        failed = subprocess.CompletedProcess(args=["nvidia-smi"], returncode=9, stdout="", stderr="")
        with patch("millm.ml.nvidia_smi.subprocess.run", return_value=failed):
            assert nvidia_smi.query_gpus() == []


class TestUuidMapsToTorchIndex:
    def test_torch_index_follows_the_uuid_not_the_nvidia_smi_position(self):
        # nvidia-smi lists the 3090 first; torch has it at index 1.
        with fake_gpus(*NODE, smi_order=[1, 0]):
            gpus = list_gpus()
        assert [(g.index, g.name, g.free_mb, g.uuid) for g in gpus] == [
            (0, TI_3080, 11_000, NODE_UUIDS[0]),
            (1, RTX_3090, 23_000, NODE_UUIDS[1]),
        ]

    def test_a_card_torch_cannot_see_is_left_out(self):
        with fake_gpus(*NODE), patch("torch.cuda.device_count", return_value=1):
            assert [g.index for g in list_gpus()] == [0]


class TestNoContextOnCardsTheModelDoesNotUse:
    def test_the_inventory_touches_no_card(self):
        with fake_gpus(*NODE) as fake:
            fake.forbid(0, 1)
            assert [g.free_mb for g in list_gpus()] == [11_000, 23_000]
        assert fake.calls == []

    def test_deciding_a_transformers_load_touches_no_card(self):
        loader = model_loader.ModelLoader()
        loader.state = MagicMock()
        context = MagicMock()
        with fake_gpus(*NODE) as fake, patch.object(model_loader, "ModelLoadContext", return_value=context):
            fake.forbid(0, 1)
            loader.load(1, "m", "/tmp/m", "FP16", 18_000)
        assert context.__enter__.return_value.load.call_args.kwargs["placement"].index == 1
        assert fake.calls == []

    def test_loading_onto_card_1_never_reads_card_0(self):
        """The load measures the card the model goes on, and only that card."""
        from tests.unit.ml.test_model_load_placement import FakeModel, _load

        import torch

        with fake_gpus(*NODE) as fake:
            fake.forbid(0)
            loaded, kwargs = _load(fake, choose_gpu(8_000), consume={1: 9_000},
                                   model=FakeModel([torch.device("cuda", 1)]))
        assert kwargs["device_map"] == {"": "cuda:1"}
        assert loaded.memory_by_device_mb == {"cuda:1": 9_000}
        assert {index for _, index in fake.calls} == {1}

    def test_a_bitsandbytes_load_onto_card_1_never_reads_card_0(self):
        from tests.unit.ml.test_model_load_placement import _load

        with fake_gpus(*NODE) as fake:
            fake.forbid(0)
            _, kwargs = _load(fake, choose_gpu(8_000), quantization="Q8")
        assert kwargs["device_map"] == {"": "cuda:1"}
        assert "max_memory" not in kwargs
        assert all(index == 1 for _, index in fake.calls)

    def test_a_gguf_load_reads_no_card_through_torch(self, tmp_path):
        (tmp_path / "m-Q4_K_M.gguf").write_bytes(b"\x00" * 64)

        def _construct(**kwargs):
            fake.free_mb[1] -= 6_000
            return MagicMock()

        with fake_gpus(*NODE) as fake, \
                patch.object(model_loader, "llama_supports_gpu_offload", lambda: True), \
                patch.object(model_loader, "Llama", MagicMock(side_effect=_construct)) as llama, \
                patch.object(model_loader, "declared_context", return_value=32_768), \
                patch.object(model_loader, "gguf_kv_bytes_per_token", return_value=100_000.0), \
                patch.object(settings, "GGUF_ENABLE_EMBEDDINGS", False):
            fake.forbid(0, 1)
            loaded = model_loader.load_gguf_model(1, "m", str(tmp_path), "m-Q4_K_M.gguf")
        assert llama.call_args.kwargs["main_gpu"] == 1
        assert loaded.memory_by_device_mb == {"cuda:1": 6_000}
        assert fake.calls == []


class TestNoNvidiaSmi:
    def test_gguf_takes_the_cpu_path(self):
        with fake_gpus(*NODE, smi_available=False), \
                patch.object(model_loader, "llama_supports_gpu_offload", lambda: True):
            assert model_loader.plan_gguf_placement(4_000, None, 4_096).mode == MODE_CPU

    def test_a_transformers_load_is_refused_as_no_gpu(self):
        with fake_gpus(*NODE, smi_available=False):
            with pytest.raises(InsufficientMemoryError) as raised:
                choose_gpu(8_000)
        assert raised.value.details["gpus"] == []
