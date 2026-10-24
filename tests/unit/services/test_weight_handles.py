"""Attachability and CUDA IPC handle export (Feature 31, FTASKS 5.1 and 5.7).

Every `NOT_ATTACHABLE` reason has a fixture where ONLY that check fails: the baseline below passes
every check, and each test changes one fact. The export is tested on the CPU with a stand-in for
`_share_cuda_` (the grouping, the dedupe and the field set are torch-independent) and on a real
card by a test that skips loudly without CUDA (the GPU node was down on 2026-10-14).

MUTATION CONTROLS (each turns this file red): accept `quantization_method="bitsandbytes"` (M12);
accept the llama.cpp engine (M13); one storage per parameter (M25); `remove_duplicate=True`;
drop any single check in `check_attachable`.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from millm.core.errors import ModelNotLoadedError, NotAttachableError
from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModel
from millm.services import weight_handles
from millm.services.weight_handles import (
    address_fingerprint,
    check_attachable,
    export_handles,
    moved_addresses,
    read_driver_version,
)
from tests.unit.weights_fixtures import tiny_llama

CUDA0 = torch.device("cuda", 0)


# --- check_attachable --------------------------------------------------------------------


class FakeModel:
    """Parameters that claim a device and a dtype; nothing else is read by the check."""

    def __init__(self, params: list[tuple[str, Any]], keep_fp32: Any = None) -> None:
        self._params = params
        self._keep_in_fp32_modules = keep_fp32
        self.remove_duplicate_seen: list[bool] = []

    def named_parameters(self, remove_duplicate: bool = True):
        self.remove_duplicate_seen.append(remove_duplicate)
        return iter(self._params)


def _param(device: torch.device = CUDA0, dtype: torch.dtype = torch.bfloat16) -> Any:
    return SimpleNamespace(device=device, dtype=dtype)


def _loaded(model: Any = None, **overrides: Any) -> LoadedModel:
    fields: dict[str, Any] = dict(
        model_id=7,
        model_name="tiny",
        model=model
        if model is not None
        else FakeModel([("embed.weight", _param()), ("head.weight", _param(dtype=torch.float16))]),
        tokenizer=None,
        loaded_at=datetime(2026, 10, 14),
        quantization_method="none",
        attachable_allocation=True,
        gpu_indices=[0],
    )
    fields.update(overrides)
    return LoadedModel(**fields)


def _check(loaded: Any, **facts: Any) -> None:
    args: dict[str, Any] = dict(cbm_enabled=False, max_concurrent=1, transition=None)
    args.update(facts)
    check_attachable(loaded, **args)


def _reason(loaded: Any, **facts: Any) -> dict[str, Any]:
    with pytest.raises(NotAttachableError) as info:
        _check(loaded, **facts)
    assert info.value.status_code == 400 and info.value.code == "NOT_ATTACHABLE"
    return info.value.details


def test_the_baseline_passes_every_check():
    assert _check(_loaded()) is None


def test_bfloat16_and_float16_are_both_accepted():
    model = FakeModel([("a", _param(dtype=torch.bfloat16)), ("b", _param(dtype=torch.float16))])
    assert _check(_loaded(model)) is None


def test_nothing_resident_is_model_not_loaded():
    with pytest.raises(ModelNotLoadedError):
        _check(None)


@pytest.mark.parametrize("transition", ["unloading", "loading", "internal_write"])
def test_a_model_transition_refuses(transition):
    details = _reason(_loaded(), transition=transition)
    assert details == {"reason": "model_transition", "transition": transition}


def test_a_transition_refuses_even_with_nothing_resident():
    """A load in progress with nothing resident yet is a transition, not 'no model'."""
    assert _reason(None, transition="loading")["reason"] == "model_transition"


def test_an_open_commit_refuses_and_torn_does_not():
    """Addendum D-40: `writing` is refused; `torn` and `clean` are attachable (acceptance 7)."""
    details = _reason(_loaded(), weights_state="writing")
    assert details == {"reason": "weights_writing", "weights_state": "writing"}
    assert _check(_loaded(), weights_state="torn") is None
    assert _check(_loaded(), weights_state="clean") is None


def test_gguf_engine_refuses():
    details = _reason(_loaded(engine=ENGINE_LLAMACPP, quantization_method="none"))
    assert details["reason"] == "gguf"


@pytest.mark.parametrize(
    "method", ["bitsandbytes", "gptq", "awq", "unknown", "gguf:Q4_K_M", "None", ""]
)
def test_quantized_is_a_whitelist_of_none(method):
    """Only the exact string "none" is attachable (031 review C5): the field's default is
    "unknown", and a pre-quantized `quant_method` or a gguf label are not plain weights."""
    details = _reason(_loaded(quantization_method=method))
    assert details == {"reason": "quantized", "quantization_method": method}


def test_a_cpu_parameter_refuses_naming_it():
    model = FakeModel([("embed.weight", _param()), ("head.weight", _param(torch.device("cpu")))])
    details = _reason(_loaded(model))
    assert details["reason"] == "not_on_gpu" and details["parameter"] == "head.weight"


def test_float32_refuses_naming_the_first_offender():
    model = FakeModel(
        [
            ("embed.weight", _param()),
            ("norm.weight", _param(dtype=torch.float32)),
            ("head.weight", _param(dtype=torch.float32)),
        ]
    )
    details = _reason(_loaded(model))
    assert details == {"reason": "dtype", "parameter": "norm.weight", "dtype": "float32"}


def test_a_keep_in_fp32_model_says_so():
    """031 review D16: a model that keeps modules in float32 by design is refused, and the
    refusal names the declaration, not only the tensor."""
    model = FakeModel(
        [("embed.weight", _param()), ("router.weight", _param(dtype=torch.float32))],
        keep_fp32=["router"],
    )
    details = _reason(_loaded(model))
    assert details["reason"] == "dtype" and details["keep_in_fp32_modules"] == ["router"]


def test_a_load_that_was_not_attachable_refuses_with_allocator():
    assert _reason(_loaded(attachable_allocation=False)) == {"reason": "allocator"}


def test_continuous_batching_refuses():
    assert _reason(_loaded(), cbm_enabled=True) == {"reason": "continuous_batching_running"}


@pytest.mark.parametrize("slots", [2, 4, 0])
def test_any_slot_count_but_one_refuses(slots):
    assert _reason(_loaded(), max_concurrent=slots) == {
        "reason": "concurrency",
        "max_concurrent": slots,
    }


def test_the_check_reads_tied_names_too():
    """`remove_duplicate=False`: a tied name must not hide an offender (031 review I5)."""
    model = FakeModel([("embed.weight", _param())])
    _check(_loaded(model))
    assert model.remove_duplicate_seen == [False]


def test_a_real_loaded_model_default_is_not_attachable():
    """Every existing load keeps the default, so attach refuses it with `allocator`."""
    assert LoadedModel(1, "m", None, None, datetime(2026, 1, 1)).attachable_allocation is False


# --- export_handles ----------------------------------------------------------------------


class FakeShare:
    """Stands in for `_share_cuda_`: one fake tuple per storage, device chosen by name."""

    def __init__(self, device_for=lambda p: 0) -> None:
        self.calls = 0
        self.device_for = device_for

    def __call__(self, param: Any) -> tuple[Any, ...]:
        self.calls += 1
        storage = param.untyped_storage()
        ptr = storage.data_ptr()
        return (
            self.device_for(param),
            f"handle-{ptr}".encode(),
            storage.nbytes(),
            0,
            b"refcounter",
            self.calls,
            None,
            True,
        )


def _uuid(index: int) -> str:
    return f"GPU-0000000{index}-0000-0000-0000-000000000000"


def test_tied_weights_give_one_storage_and_both_names():
    model = tiny_llama(tied=True, dtype=torch.bfloat16)
    share = FakeShare()
    out = export_handles(model, uuid_for=_uuid, share=share)
    names = [p["name"] for p in out.parameters]
    assert "lm_head.weight" in names and "model.embed_tokens.weight" in names
    assert len(names) == len(list(model.named_parameters(remove_duplicate=False)))
    by_name = {p["name"]: p for p in out.parameters}
    assert by_name["lm_head.weight"]["storage"] == by_name["model.embed_tokens.weight"]["storage"]
    distinct = {p.untyped_storage().data_ptr() for p in model.parameters()}
    assert len(out.storages) == len(distinct) == share.calls
    assert len(out.keep_alive) == len(out.storages)


def test_untied_weights_give_two_storages():
    model = tiny_llama(tied=False, dtype=torch.float16)
    out = export_handles(model, uuid_for=_uuid, share=FakeShare())
    by_name = {p["name"]: p for p in out.parameters}
    assert by_name["lm_head.weight"]["storage"] != by_name["model.embed_tokens.weight"]["storage"]


def test_the_storage_and_parameter_field_sets():
    model = tiny_llama(tied=True, dtype=torch.bfloat16)
    out = export_handles(model, uuid_for=_uuid, share=FakeShare())
    assert set(out.storages[0]) == {
        "index", "device_index", "device_uuid", "dtype", "handle", "storage_size_bytes",
        "storage_offset_bytes", "ref_counter_handle", "ref_counter_offset", "event_handle",
        "event_sync_required",
    }
    assert set(out.parameters[0]) == {
        "name", "storage", "shape", "stride", "storage_offset", "dtype", "requires_grad",
    }
    first = out.storages[0]
    assert first["dtype"] == "bfloat16" and first["event_handle"] is None
    assert first["device_uuid"] == _uuid(0)
    import base64

    assert base64.b64decode(first["handle"]).startswith(b"handle-")
    embed = next(p for p in out.parameters if p["name"] == "model.embed_tokens.weight")
    assert embed["shape"] == [256, 64] and embed["stride"] == [64, 1]


def test_a_view_records_its_storage_offset():
    """Two parameters viewing one storage: one entry, each parameter with its own offset."""
    base = torch.zeros(8, 4, dtype=torch.float16)
    module = torch.nn.Module()
    module.a = torch.nn.Parameter(base[:4])
    module.b = torch.nn.Parameter(base[4:])
    out = export_handles(module, uuid_for=_uuid, share=FakeShare())
    assert len(out.storages) == 1
    offsets = {p["name"]: p["storage_offset"] for p in out.parameters}
    assert offsets == {"a": 0, "b": 16}


def test_a_split_model_names_two_cards_by_uuid():
    model = tiny_llama(tied=False, dtype=torch.bfloat16)
    second_half = {id(p) for n, p in model.named_parameters() if "layers.1" in n or "lm_head" in n}
    asked: list[int] = []

    def uuid_for(index: int) -> str:
        asked.append(index)
        return _uuid(index)

    share = FakeShare(device_for=lambda p: 1 if id(p) in second_half else 0)
    out = export_handles(model, uuid_for=uuid_for, share=share)
    assert {s["device_uuid"] for s in out.storages} == {_uuid(0), _uuid(1)}
    assert sorted(asked) == [0, 1], "each card's UUID is resolved once"


def test_addresses_are_recorded_for_every_name():
    model = tiny_llama(tied=True, dtype=torch.bfloat16)
    out = export_handles(model, uuid_for=_uuid, share=FakeShare())
    assert out.addresses == address_fingerprint(model)
    assert "lm_head.weight" in out.addresses


def test_the_repr_carries_no_handle():
    out = export_handles(tiny_llama(dtype=torch.bfloat16), uuid_for=_uuid, share=FakeShare())
    assert "handle" not in repr(out) and "refcounter" not in repr(out)


def test_the_export_log_line_carries_counts_only():
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        export_handles(tiny_llama(dtype=torch.bfloat16), uuid_for=_uuid, share=FakeShare())
    assert "handle-" not in str(logs) and "refcounter" not in str(logs)
    assert [e["event"] for e in logs] == ["weights_handles_exported"]


def test_export_of_a_thousand_parameters_is_fast():
    """FTASKS 5.7: attach latency under 1 s for a 1,000-parameter stand-in."""
    module = torch.nn.Module()
    for i in range(1000):
        module.register_parameter(f"p{i}", torch.nn.Parameter(torch.zeros(4, dtype=torch.float16)))
    started = time.perf_counter()
    out = export_handles(module, uuid_for=_uuid, share=FakeShare())
    assert len(out.parameters) == 1000
    assert time.perf_counter() - started < 1.0


# --- addresses -----------------------------------------------------------------------------


def test_a_replaced_tensor_is_a_moved_address():
    model = tiny_llama(tied=False, dtype=torch.bfloat16)
    before = address_fingerprint(model)
    assert moved_addresses(model, before) == []
    model.model.norm.weight.data = model.model.norm.weight.data.clone()
    assert moved_addresses(model, before) == ["model.norm.weight"]


def test_an_in_place_write_is_not_a_move():
    model = tiny_llama(tied=True, dtype=torch.bfloat16)
    before = address_fingerprint(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1)
    assert moved_addresses(model, before) == []


# --- versions --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.07  Fri Aug 29 2025\n", "580.82.07"),
        (
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.95.05  Release Build\n",
            "580.95.05",
        ),
        ("garbage\n", None),
    ],
)
def test_driver_version_from_the_kernel_module_file(tmp_path: Path, text, expected):
    path = tmp_path / "version"
    path.write_text(text)
    assert read_driver_version(path) == expected


def test_a_missing_driver_file_is_none_not_a_guess(tmp_path: Path):
    assert read_driver_version(tmp_path / "absent") is None


def test_producer_versions_name_torch_cuda_and_driver():
    versions = weight_handles.producer_versions()
    assert set(versions) == {"torch", "cuda", "driver"}
    assert versions["torch"] == torch.__version__


def test_uuid_for_index_normalises_torchs_bare_uuid(monkeypatch):
    props = SimpleNamespace(uuid="247aa582-0d1b-e161-8156-983ed1fefc57")
    asked: list[int] = []

    def fake_props(index: int):
        asked.append(index)
        return props

    monkeypatch.setattr(torch.cuda, "get_device_properties", fake_props)
    assert weight_handles.uuid_for_index(1) == "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
    assert asked == [1]


# --- a real card -------------------------------------------------------------------------------

_NO_CUDA = (
    "NEEDS CUDA: the real `_share_cuda_` export runs only on a GPU (the node was down on "
    "2026-10-14); re-run on the node (FTASKS 9.2, hardware)."
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_NO_CUDA)
def test_real_export_inside_an_attachable_allocation():  # pragma: no cover - hardware only
    from millm.ml.model_loader import attachable_allocation

    with attachable_allocation([0]):
        model = tiny_llama(tied=True, dtype=torch.bfloat16).to(torch.device("cuda", 0))
    out = export_handles(model)
    assert len(out.storages) == len({p.untyped_storage().data_ptr() for p in model.parameters()})
    assert all(s["handle"] for s in out.storages)
    assert all(s["device_uuid"] and s["device_uuid"].startswith("GPU-") for s in out.storages)
