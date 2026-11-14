"""The rebuild recipe in `manual/docs/api/weights.md` rebuilds each storage ONCE (finding 3).

Hardware record 2026-10-16: the recipe used to call `rebuild_cuda_tensor` once per PARAMETER. A
tied storage (LFM2.5's embedding and `lm_head`) is exported once (`export_handles` dedupes by
storage, FR-31.1.2), so its second rebuild hit PyTorch's `shared_cache` and released an IPC count
the producer never opened; the counts stopped balancing and the 256 MiB storage was never freed —
every attach cycle. Rebuilding once per storage and viewing every parameter into it returned the
memory each cycle. The export side is unchanged (one storage entry per storage is the contract);
the consumer recipe is what had to change, so the recipe itself is executed here.

MUTATION CONTROL: put the rebuild back inside the parameter loop -> the call-count test goes red.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.multiprocessing.reductions as reductions

MANUAL = Path(__file__).resolve().parents[2] / "manual" / "docs" / "api" / "weights.md"


def _recipe() -> str:
    text = MANUAL.read_text()
    section = text[text.index("## Rebuilding the tensors in the trainer"):]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def _body() -> dict:
    """Two storages; the first is tied (two parameter names), as LFM2.5's embedding is."""
    def storage(i: int, size: int) -> dict:
        return {"index": i, "device_index": 0, "device_uuid": "GPU-x", "dtype": "bfloat16",
                "handle": "aGFuZGxl", "storage_size_bytes": size, "storage_offset_bytes": 0,
                "ref_counter_handle": "cmVm", "ref_counter_offset": i, "event_handle": None,
                "event_sync_required": False}

    return {
        "storages": [storage(0, 4 * 3 * 2), storage(1, 8 * 2)],
        "parameters": [
            {"name": "model.embed_tokens.weight", "storage": 0, "shape": [4, 3],
             "stride": [3, 1], "storage_offset": 0, "dtype": "bfloat16", "requires_grad": True},
            {"name": "lm_head.weight", "storage": 0, "shape": [4, 3], "stride": [3, 1],
             "storage_offset": 0, "dtype": "bfloat16", "requires_grad": True},
            {"name": "model.norm.weight", "storage": 1, "shape": [6], "stride": [1],
             "storage_offset": 2, "dtype": "bfloat16", "requires_grad": False},
        ],
    }


def _run(monkeypatch):
    calls = []

    def fake_rebuild(tensor_cls, size, stride, offset, storage_cls, dtype, device, handle,
                     storage_size_bytes, storage_offset_bytes, requires_grad, ref_counter_handle,
                     ref_counter_offset, event_handle, event_sync_required):
        calls.append(ref_counter_offset)
        return torch.zeros(size, dtype=dtype)

    monkeypatch.setattr(reductions, "rebuild_cuda_tensor", fake_rebuild)
    namespace: dict = {}
    exec(compile(_recipe(), str(MANUAL), "exec"), namespace)  # noqa: S102 - our own manual
    return namespace["rebuild"](_body()), calls


def test_each_storage_is_rebuilt_exactly_once(monkeypatch):
    tensors, calls = _run(monkeypatch)
    assert sorted(calls) == [0, 1], "one rebuild per storage entry, never per parameter"


def test_every_parameter_is_a_view_with_its_own_shape_offset_and_dtype(monkeypatch):
    tensors, _ = _run(monkeypatch)
    embed, head, norm = (tensors[n] for n in
                         ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"))
    assert embed.data_ptr() == head.data_ptr(), "tied names share one storage"
    assert embed.shape == (4, 3) and embed.dtype == torch.bfloat16
    assert norm.shape == (6,) and norm.storage_offset() == 2
    assert norm.data_ptr() - norm.untyped_storage().data_ptr() == 2 * 2  # elements, not bytes
    assert embed.requires_grad and not norm.requires_grad
