"""
Per-card memory, including what miLLM itself holds (Feature 29, FR-29.8; T-89).

`read_gpu_memory()` is synchronous: it runs nvidia-smi twice (cards, then compute processes),
each with a five-second timeout, so the route calls it through `asyncio.to_thread`.

⚠ **Reading a card never creates a CUDA context on it** (FR-29.8.4). A context costs memory that
other tenants of the node place against. torch's allocator is read ONLY on cards a transformers
model was placed on in this process (`model_loader.torch_touched_indices`) and only once CUDA is
initialised; on any other card the two miLLM fields are null with `torch_measured: false` —
never 0, which would claim a reading that was not taken.

llama.cpp's memory is not torch's: a card a resident GGUF model occupies says
`engine_memory: "not_measured_by_torch"`, and nvidia-smi's per-process list (where the pod can
read it) is the only measurement of it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import torch

from millm.core.logging import get_logger
from millm.ml import nvidia_smi
from millm.ml.gpu_placement import _torch_index_by_uuid, normalize_gpu_uuid
from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModelState, torch_touched_indices

logger = get_logger(__name__)

ENGINE_NOT_MEASURED = "not_measured_by_torch"
_MIB = 1024 * 1024


def _gguf_cards() -> set[int]:
    """Torch indices a resident GGUF model occupies."""
    current = LoadedModelState().current
    if current is None or current.engine != ENGINE_LLAMACPP:
        return set()
    return set(current.gpu_indices)


def read_gpu_memory() -> dict[str, Any]:
    """`{read_at, cards: [...], reason}`; `cards: []` and a reason when nvidia-smi is absent."""
    read_at = datetime.now(timezone.utc)
    reported = nvidia_smi.query_gpus()
    if not reported:
        return {"read_at": read_at, "cards": [], "reason": "nvidia-smi unavailable"}

    torch_index_by_uuid = _torch_index_by_uuid()
    touched = torch_touched_indices()
    try:
        initialised = bool(torch.cuda.is_initialized())  # type: ignore[no-untyped-call]
    except Exception:  # noqa: BLE001 - no CUDA at all is "not initialised"
        initialised = False
    gguf = _gguf_cards()
    apps = nvidia_smi.query_compute_apps()

    cards: list[dict[str, Any]] = []
    for card in reported:
        gpu_uuid = normalize_gpu_uuid(card.get("uuid"))
        torch_index = torch_index_by_uuid.get(gpu_uuid) if gpu_uuid is not None else None
        measured = initialised and torch_index is not None and torch_index in touched
        allocated_mb: int | None = None
        reserved_mb: int | None = None
        if measured and torch_index is not None:
            allocated_mb = int(torch.cuda.memory_allocated(torch_index) // _MIB)
            reserved_mb = int(torch.cuda.memory_reserved(torch_index) // _MIB)
        if apps is None:
            processes: list[dict[str, int]] | None = None
            processes_reason: str | None = "nvidia-smi --query-compute-apps unavailable"
        else:
            processes = [
                {"pid": app["pid"], "used_mb": app["used_mb"]}
                for app in apps
                if normalize_gpu_uuid(app["gpu_uuid"]) == gpu_uuid
            ]
            processes_reason = None
        cards.append(
            {
                "smi_index": card["index"],
                "uuid": card.get("uuid"),
                "name": card.get("name"),
                "total_mb": card.get("memory_total_mb"),
                "used_mb": card.get("memory_used_mb"),
                "free_mb": card.get("memory_free_mb"),
                "torch_index": torch_index,
                "torch_measured": measured,
                "millm_allocated_mb": allocated_mb,
                "millm_reserved_mb": reserved_mb,
                "engine_memory": (
                    ENGINE_NOT_MEASURED
                    if torch_index is not None and torch_index in gguf
                    else None
                ),
                "processes": processes,
                "processes_reason": processes_reason,
            }
        )
    return {"read_at": read_at, "cards": cards, "reason": None}
