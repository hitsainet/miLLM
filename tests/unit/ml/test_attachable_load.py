"""The attachable load (Feature 31, addendum D-03).

miLLM runs the CUDA allocator with expandable segments (`tests/unit/test_cuda_allocator.py`), and
torch refuses to share those across processes, so a trainer can attach only to parameters
allocated with them OFF. `attachable_allocation` turns them off around `from_pretrained`, routes
the allocations into a `torch.cuda.MemPool` per card, and restores the configured setting.

Off by default: every existing load is unchanged, and `check_attachable` refuses it with reason
`allocator`. W2-F enables it from `model_service` with `ModelLoader.load(..., attachable=True)`.

MUTATION CONTROLS: drop the restore in the `finally` (the restore tests go red); stop passing
`attachable` through `ModelLoader.load`; record `attachable_allocation=True` unconditionally.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from millm.core.cuda_allocator import NEW_NAME, OLD_NAME
from millm.ml import model_loader
from millm.ml.gpu_placement import MODE_SINGLE, GpuInfo, Placement
from millm.ml.model_loader import (
    LoadedModel,
    ModelLoadContext,
    ModelLoader,
    attachable_allocation,
    configured_expandable_segments,
)


# --- the configured value restored afterwards ------------------------------------------------


@pytest.mark.parametrize(
    "environ,expected",
    [
        ({OLD_NAME: "expandable_segments:True"}, "expandable_segments:True"),
        ({OLD_NAME: "max_split_size_mb:128,expandable_segments:True"}, "expandable_segments:True"),
        ({OLD_NAME: "expandable_segments:False"}, "expandable_segments:False"),
        ({NEW_NAME: "expandable_segments:True", OLD_NAME: "expandable_segments:False"},
         "expandable_segments:True"),
        ({OLD_NAME: "max_split_size_mb:128"}, "expandable_segments:False"),
        ({}, "expandable_segments:False"),
    ],
)
def test_the_configured_setting(environ, expected):
    assert configured_expandable_segments(environ) == expected


def test_the_process_default_is_what_millm_set_at_import():
    """millm sets expandable segments at import, so that is what a load restores."""
    assert configured_expandable_segments() == "expandable_segments:True"


# --- attachable_allocation ------------------------------------------------------------------


class Recorder:
    def __init__(self) -> None:
        self.events: list[Any] = []


@pytest.fixture
def recorded(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(model_loader, "_set_allocator_settings", lambda v: rec.events.append(("set", v)))

    class FakePool:
        def __init__(self) -> None:
            rec.events.append(("pool",))

    @contextlib.contextmanager
    def fake_use(pool, device=None):
        rec.events.append(("enter", device))
        try:
            yield
        finally:
            rec.events.append(("exit", device))

    monkeypatch.setattr(torch.cuda, "MemPool", FakePool)
    monkeypatch.setattr(torch.cuda, "use_mem_pool", fake_use)
    return rec


def test_off_inside_one_pool_per_card_and_restored(recorded):
    with attachable_allocation([1, 0, 1]) as pools:
        recorded.events.append(("body",))
        assert len(pools) == 2
    assert recorded.events == [
        ("set", "expandable_segments:False"),
        ("pool",), ("enter", 0),
        ("pool",), ("enter", 1),
        ("body",),
        ("exit", 1), ("exit", 0),
        ("set", "expandable_segments:True"),
    ]


def test_restored_when_the_block_raises(recorded):
    with pytest.raises(RuntimeError):
        with attachable_allocation([0]):
            raise RuntimeError("from_pretrained failed")
    assert recorded.events[-1] == ("set", "expandable_segments:True")
    assert recorded.events.count(("set", "expandable_segments:False")) == 1


def test_restores_the_operators_explicit_choice(recorded, monkeypatch):
    monkeypatch.setenv(OLD_NAME, "expandable_segments:False")
    monkeypatch.delenv(NEW_NAME, raising=False)
    with attachable_allocation([0]):
        pass
    assert recorded.events[-1] == ("set", "expandable_segments:False")


@pytest.mark.parametrize("before", [None, "0", "1"])
def test_transformers_loads_on_the_calling_thread_inside_the_block(recorded, monkeypatch, before):
    """`use_mem_pool` routes only the calling thread's allocations, and transformers 5
    materialises weights on its own thread pool unless HF_DEACTIVATE_ASYNC_LOAD is true. On the
    node every parameter then landed outside the pool (hardware record 2026-10-16, spike 0.2).
    So the block turns async loading off, and puts back exactly what was there."""
    from millm.ml.model_loader import ASYNC_LOAD_ENV

    if before is None:
        monkeypatch.delenv(ASYNC_LOAD_ENV, raising=False)
    else:
        monkeypatch.setenv(ASYNC_LOAD_ENV, before)
    with attachable_allocation([0]):
        assert os.environ[ASYNC_LOAD_ENV] == "1"
    assert os.environ.get(ASYNC_LOAD_ENV) == before


def test_async_loading_is_restored_when_the_block_raises(recorded, monkeypatch):
    from millm.ml.model_loader import ASYNC_LOAD_ENV

    monkeypatch.delenv(ASYNC_LOAD_ENV, raising=False)
    with pytest.raises(RuntimeError):
        with attachable_allocation([0]):
            raise RuntimeError("from_pretrained failed")
    assert ASYNC_LOAD_ENV not in os.environ


def test_the_setter_is_torchs_non_deprecated_one(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", seen.append)
    model_loader._set_allocator_settings("expandable_segments:False")
    assert seen == ["expandable_segments:False"]


# --- the load path ---------------------------------------------------------------------------


def _placement() -> Placement:
    return Placement(
        mode=MODE_SINGLE,
        reason="requested_card",
        required_mb=0,
        gpus=(GpuInfo(1, "card", None, 24_576, 20_000),),
        index=1,
    )


def _run_ctx_load(attachable: bool, allocation_events: list[Any]) -> LoadedModel:
    @contextlib.contextmanager
    def fake_allocation(indices):
        allocation_events.append(("enter", list(indices)))
        yield ["pool-1"]
        allocation_events.append(("exit",))

    with patch("millm.ml.model_loader.AutoModelForCausalLM") as auto_model, patch(
        "millm.ml.model_loader.AutoTokenizer"
    ) as auto_tok, patch("millm.ml.model_loader.torch") as mock_torch, patch(
        "millm.ml.model_loader.attachable_allocation", fake_allocation
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0
        tokenizer = MagicMock(pad_token=None, eos_token="<eos>")
        auto_tok.from_pretrained.return_value = tokenizer
        auto_model.from_pretrained.side_effect = lambda *a, **k: (
            allocation_events.append(("from_pretrained",)) or MagicMock()
        )
        with ModelLoadContext(model_id=1, model_name="m") as ctx:
            return ctx.load(
                cache_path="/data/models/m",
                quantization="FP16",
                placement=_placement(),
                attachable=attachable,
            )


def test_an_attachable_load_allocates_the_parameters_inside_the_allocation():
    events: list[Any] = []
    loaded = _run_ctx_load(True, events)
    assert events == [("enter", [1]), ("from_pretrained",), ("exit",)]
    assert loaded.attachable_allocation is True
    assert loaded.attachable_mem_pools == ["pool-1"]


def test_the_default_load_is_unchanged():
    events: list[Any] = []
    loaded = _run_ctx_load(False, events)
    assert events == [("from_pretrained",)]
    assert loaded.attachable_allocation is False and loaded.attachable_mem_pools == []


def test_model_loader_passes_attachable_through():
    loader = ModelLoader()
    seen: dict[str, Any] = {}

    def fake_ctx_load(self, **kwargs):
        seen.update(kwargs)
        return MagicMock()

    with patch.object(ModelLoadContext, "load", fake_ctx_load), patch(
        "millm.ml.model_loader.torch"
    ) as mock_torch, patch(
        "millm.ml.model_loader.plan_transformers_load", return_value=_placement()
    ), patch("millm.ml.model_loader.preflight_split"), patch(
        "millm.ml.model_loader.list_gpus", return_value=[]
    ), patch.object(loader.state, "set"):
        mock_torch.cuda.is_available.return_value = True
        loader.load(1, "m", "/data/models/m", "FP16", 100, attachable=True)
        assert seen["attachable"] is True
        loader.load(1, "m", "/data/models/m", "FP16", 100)
        assert seen["attachable"] is False


_NO_CUDA = (
    "NEEDS CUDA: whether torch shares a tensor allocated by attachable_allocation (and refuses "
    "one allocated with expandable segments) is a hardware fact; re-run on the node (D-03 spike)."
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_NO_CUDA)
def test_from_pretrained_puts_every_parameter_in_the_pool(tmp_path):  # pragma: no cover
    """Hardware: a real `from_pretrained` inside the block allocates every parameter in the
    MemPool (none in the default pool), so a shared segment holds parameters only (review D2)."""
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=1024, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(tmp_path)
    with attachable_allocation([0]) as pools:
        model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16,
                                                     device_map={"": 0})
    pool_ids = {tuple(p.id) for p in pools}
    segments = [(s["address"], s["address"] + s["total_size"])
                for s in torch.cuda.memory_snapshot()
                if tuple(s.get("segment_pool_id", (0, 0))) in pool_ids]
    outside = [n for n, p in model.named_parameters()
               if not any(lo <= p.untyped_storage().data_ptr() < hi for lo, hi in segments)]
    assert pools and not outside, outside


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_NO_CUDA)
def test_a_tensor_from_the_allocation_can_be_shared():  # pragma: no cover - hardware only
    device = torch.device("cuda", 0)
    with attachable_allocation([0]) as pools:
        t = torch.ones(1024, dtype=torch.bfloat16, device=device)
    assert pools
    assert t._typed_storage()._share_cuda_()[1]
