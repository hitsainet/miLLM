"""Feature 29 task 8.5: per-card memory, read without creating a CUDA context.

Fixtures disagree with the defect: two cards, one touched by a transformers model and one not.
torch's allocator stand-in FAILS when asked about the untouched card (control M17).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from millm.ml import model_loader
from millm.ml.model_loader import (
    ENGINE_LLAMACPP,
    ENGINE_TRANSFORMERS,
    LoadedModel,
    LoadedModelState,
)
from millm.ml.nvidia_smi import parse_compute_apps

UUID_A = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
UUID_B = "GPU-f47ba814-49a2-603f-3595-275284140251"

SMI_TWO_CARDS = [
    {"index": 0, "uuid": UUID_A, "utilization": 0, "memory_used_mb": 17396,
     "memory_total_mb": 24576, "memory_free_mb": 7180, "temperature": 40, "name": "RTX 3090"},
    {"index": 1, "uuid": UUID_B, "utilization": 0, "memory_used_mb": 300,
     "memory_total_mb": 12288, "memory_free_mb": 11988, "temperature": 35,
     "name": "RTX 3080 Ti"},
]


@pytest.fixture(autouse=True)
def _restore():
    saved = set(model_loader._TORCH_TOUCHED_INDICES)
    LoadedModelState()._loaded = None
    yield
    model_loader._TORCH_TOUCHED_INDICES.clear()
    model_loader._TORCH_TOUCHED_INDICES.update(saved)
    LoadedModelState()._loaded = None


@pytest.fixture
def gpus(monkeypatch):
    """Install nvidia-smi and torch stand-ins; returns the call log for torch reads."""
    import millm.services.gpu_memory as gm

    calls: list[tuple[str, int]] = []
    state = SimpleNamespace(
        smi=list(SMI_TWO_CARDS),
        apps=[{"gpu_uuid": UUID_A, "pid": 4242, "used_mb": 17100},
              {"gpu_uuid": UUID_B, "pid": 9, "used_mb": 250}],
        torch_map={UUID_A: 0, UUID_B: 1},
        initialised=True,
        calls=calls,
    )
    monkeypatch.setattr(gm.nvidia_smi, "query_gpus", lambda: state.smi)
    monkeypatch.setattr(gm.nvidia_smi, "query_compute_apps", lambda: state.apps)
    monkeypatch.setattr(gm, "_torch_index_by_uuid", lambda: state.torch_map)
    monkeypatch.setattr(gm.torch.cuda, "is_initialized", lambda: state.initialised)

    def allocated(index):
        calls.append(("allocated", index))
        if index not in model_loader._TORCH_TOUCHED_INDICES:
            raise AssertionError(f"torch read on untouched card {index} would create a context")
        return 1000 * 1024 * 1024

    def reserved(index):
        calls.append(("reserved", index))
        if index not in model_loader._TORCH_TOUCHED_INDICES:
            raise AssertionError(f"torch read on untouched card {index} would create a context")
        return 1500 * 1024 * 1024

    monkeypatch.setattr(gm.torch.cuda, "memory_allocated", allocated)
    monkeypatch.setattr(gm.torch.cuda, "memory_reserved", reserved)
    return state


def _read():
    from millm.services.gpu_memory import read_gpu_memory

    return read_gpu_memory()


def _set(engine, gpu_indices):
    LoadedModelState().set(LoadedModel(
        model_id=1, model_name="m1", model=SimpleNamespace(), tokenizer=None,
        loaded_at=datetime(2026, 10, 6), engine=engine, gpu_indices=list(gpu_indices),
    ))


class TestTouchedSet:
    def test_a_transformers_placement_is_recorded_and_kept_after_unload(self):
        model_loader._TORCH_TOUCHED_INDICES.clear()
        _set(ENGINE_TRANSFORMERS, [1])
        assert model_loader.torch_touched_indices() == frozenset({1})
        LoadedModelState()._loaded = None
        assert model_loader.torch_touched_indices() == frozenset({1})

    def test_a_gguf_placement_is_not(self):
        model_loader._TORCH_TOUCHED_INDICES.clear()
        _set(ENGINE_LLAMACPP, [0])
        assert model_loader.torch_touched_indices() == frozenset()


class TestTwoCards:
    def test_only_the_touched_card_is_measured(self, gpus):
        model_loader._TORCH_TOUCHED_INDICES.clear()
        model_loader._TORCH_TOUCHED_INDICES.add(0)
        result = _read()
        a, b = result["cards"]
        assert result["reason"] is None
        assert (a["smi_index"], a["torch_index"], a["torch_measured"]) == (0, 0, True)
        assert (a["millm_allocated_mb"], a["millm_reserved_mb"]) == (1000, 1500)
        assert (a["total_mb"], a["used_mb"], a["free_mb"]) == (24576, 17396, 7180)
        assert b["torch_measured"] is False
        assert b["millm_allocated_mb"] is None and b["millm_reserved_mb"] is None
        assert {i for _, i in gpus.calls} == {0}, "torch was asked about the untouched card"

    def test_processes_attach_by_uuid(self, gpus):
        a, b = _read()["cards"]
        assert a["processes"] == [{"pid": 4242, "used_mb": 17100}]
        assert b["processes"] == [{"pid": 9, "used_mb": 250}]
        assert a["processes_reason"] is None

    def test_a_card_torch_cannot_see_is_listed_not_dropped(self, gpus):
        model_loader._TORCH_TOUCHED_INDICES.update({0, 1})
        gpus.torch_map = {UUID_A: 0}
        cards = _read()["cards"]
        assert len(cards) == 2
        assert cards[1]["torch_index"] is None and cards[1]["torch_measured"] is False
        assert cards[1]["millm_reserved_mb"] is None

    def test_cuda_not_initialised_measures_nothing(self, gpus):
        model_loader._TORCH_TOUCHED_INDICES.update({0, 1})
        gpus.initialised = False
        cards = _read()["cards"]
        assert all(c["torch_measured"] is False for c in cards)
        assert gpus.calls == []

    def test_compute_apps_unavailable(self, gpus):
        gpus.apps = None
        a, _ = _read()["cards"]
        assert a["processes"] is None
        assert "compute-apps" in a["processes_reason"]

    def test_compute_apps_empty_is_an_empty_list(self, gpus):
        gpus.apps = []
        a, _ = _read()["cards"]
        assert a["processes"] == [] and a["processes_reason"] is None

    def test_one_card(self, gpus):
        gpus.smi = [SMI_TWO_CARDS[1]]
        cards = _read()["cards"]
        assert len(cards) == 1 and cards[0]["smi_index"] == 1 and cards[0]["torch_index"] == 1


class TestGguf:
    def test_a_resident_gguf_card_is_labelled(self, gpus):
        model_loader._TORCH_TOUCHED_INDICES.clear()
        _set(ENGINE_LLAMACPP, [0])
        a, b = _read()["cards"]
        assert a["engine_memory"] == "not_measured_by_torch"
        assert a["torch_measured"] is False  # never touched by torch
        assert b["engine_memory"] is None
        assert gpus.calls == []


class TestNoNvidiaSmi:
    def test_empty_cards_and_a_reason_never_zeros(self, gpus):
        gpus.smi = []
        result = _read()
        assert result["cards"] == [] and result["reason"] == "nvidia-smi unavailable"


class TestParser:
    def test_parses_and_skips_bad_lines(self):
        out = f"{UUID_A}, 4242, 17100\n{UUID_B}, 9, 250\ngarbage\n{UUID_A}, x, 1\n"
        assert parse_compute_apps(out) == [
            {"gpu_uuid": UUID_A, "pid": 4242, "used_mb": 17100},
            {"gpu_uuid": UUID_B, "pid": 9, "used_mb": 250},
        ]

    def test_query_returns_none_when_nvidia_smi_is_absent(self, monkeypatch):
        import millm.ml.nvidia_smi as smi

        def missing(*a, **k):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(smi.subprocess, "run", missing)
        assert smi.query_compute_apps() is None

    def test_query_uses_a_timeout(self, monkeypatch):
        import millm.ml.nvidia_smi as smi

        seen = {}

        def run(args, **kwargs):
            seen.update(kwargs, args=args)
            return SimpleNamespace(returncode=0, stdout=f"{UUID_A}, 1, 2\n")

        monkeypatch.setattr(smi.subprocess, "run", run)
        assert smi.query_compute_apps() == [{"gpu_uuid": UUID_A, "pid": 1, "used_mb": 2}]
        assert seen["timeout"] == 5
        assert "--query-compute-apps=gpu_uuid,pid,used_memory" in seen["args"]
