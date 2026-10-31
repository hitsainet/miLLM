"""Which resident models can be attached, and the CUDA IPC handles an attach hands out (Feature 31).

Pure functions over a loaded model: this module knows torch and nothing about the barrier, tokens
or the attachment registry, so every rule here is testable on the CPU with stand-ins (031 FTDD §6).
`weight_attachment.WeightService` gathers the runtime facts (continuous batching, the slot count,
a model transition in progress) and passes them in.

**`check_attachable`** refuses with `NOT_ATTACHABLE` and one `reason` (FR-31.5.1, addendum D-03,
D-04, 031 review D3, D16):

- `model_transition`: a load, an unload, or a barrier-owned write or pause is in progress. Handles
  exported now would point at memory about to move or be freed.
- `weights_writing`: a commit is still open (`weights_state == writing`), which happens when
  clear-stale removed a holder mid-commit and left the commit for the watchdog (addendum D-40). A
  new holder does not join mid-commit; re-attaching to a `torn` set stays allowed (acceptance 7).
- `gguf`: the llama.cpp engine has no torch tensors to share.
- `quantized`: `quantization_method` is anything but `"none"`. A WHITELIST, because the field
  also takes `"unknown"` (its default), a pre-quantized `quant_method` (GPTQ, AWQ) and
  `"gguf:<label>"` (031 review C5); a blacklist of the names known today admits the next one.
- `not_on_gpu`, `dtype`: every parameter must be a CUDA tensor in float16 or bfloat16. The first
  offender is named. A model that keeps modules in float32 (`_keep_in_fp32_modules`) is refused
  too, and the refusal says so (D16).
- `allocator`: the parameters were not allocated by an attachable load. miLLM runs the CUDA
  allocator with expandable segments, and torch refuses to share those across processes; an
  attachable load allocates the parameters with them off, in a pool of their own (D-03).
- `continuous_batching_running`: the continuous batching manager generates without the admission
  slot, so the barrier cannot hold it (D-13).
- `concurrency`: with more than one admission slot, holding one stops nothing (D-04).

**`export_handles`** walks `named_parameters(remove_duplicate=False)` so a tied name such as
`lm_head.weight` is listed (031 review I5), and shares each STORAGE once, keyed by its device and
`data_ptr` (I6): every `_share_cuda_` call opens a reference count the consumer must release, so a
second call for a tied storage would leak one. The consumer must therefore rebuild each STORAGE
once and view every parameter into it (the manual's recipe): rebuilding a tied storage once per
name makes PyTorch's receiving side release a count that was never opened, and the storage is then
never freed (hardware record 2026-10-16, finding 3; addendum D-47). Devices are named by UUID (C6), the driver by the
kernel module's own version file (D17).

**Handles are never logged.** The only log line here carries counts.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from millm.core.errors import ModelNotLoadedError, NotAttachableError
from millm.core.logging import get_logger

logger = get_logger(__name__)

#: The only `quantization_method` that is attachable (whitelist; 031 review C5).
ATTACHABLE_QUANTIZATION = "none"

#: Parameter dtypes a consumer can rebuild and train in place (FR-31.5.1).
ATTACHABLE_DTYPES = ("float16", "bfloat16")

#: Where the NVIDIA kernel module states its version (031 review D17).
NVIDIA_VERSION_FILE = Path("/proc/driver/nvidia/version")

#: "NVIDIA UNIX x86_64 Kernel Module  580.82.07" and the open module's
#: "NVIDIA UNIX Open Kernel Module for x86_64  580.82.07".
_DRIVER_VERSION = re.compile(r"Kernel Module(?: for \S+)?\s+(\d+(?:\.\d+)+)")

#: The model-transition states attach refuses during (031 review D3).
TRANSITIONS = ("unloading", "loading", "internal_write")


def not_attachable(
    reason: str, parameter: Optional[str] = None, **extra: Any
) -> NotAttachableError:
    """`NOT_ATTACHABLE` with `reason`, the offending parameter, and any further facts."""
    error = NotAttachableError.for_reason(reason, parameter)
    if extra:
        error.details.update(extra)
    return error


def _dtype_name(dtype: Any) -> str:
    """`torch.bfloat16` -> `"bfloat16"`."""
    return str(dtype).rsplit(".", 1)[-1]


def _keep_in_fp32_modules(model: Any) -> list[str]:
    """The model class's `_keep_in_fp32_modules`, as a list (empty when it declares none)."""
    declared = getattr(model, "_keep_in_fp32_modules", None)
    if not declared:
        return []
    return sorted(str(m) for m in declared)


def check_attachable(
    loaded: Any,
    *,
    cbm_enabled: bool,
    max_concurrent: int,
    transition: Optional[str] = None,
    weights_state: Optional[str] = None,
) -> None:
    """Raise unless `loaded` can be attached (FR-31.5.1). Returns None when it can.

    Args:
        loaded: the resident `LoadedModel`, or None.
        cbm_enabled: whether the continuous batching manager is configured in this process.
        max_concurrent: the request queue's slot count.
        transition: one of `TRANSITIONS` when a model transition is in progress, else None.
        weights_state: the barrier's `weights_state`; `writing` is refused (D-40), `torn` is not.

    Raises:
        NotAttachableError: with `details.reason` naming the first check that failed.
        ModelNotLoadedError: nothing is resident.
    """
    from millm.ml.model_loader import ENGINE_TRANSFORMERS

    if transition is not None:
        raise not_attachable("model_transition", transition=transition)
    if weights_state == "writing":
        raise not_attachable("weights_writing", weights_state=weights_state)
    if loaded is None:
        raise ModelNotLoadedError("No model is loaded; load one before attaching.")
    if loaded.engine != ENGINE_TRANSFORMERS:
        raise not_attachable("gguf", engine=loaded.engine)
    if loaded.quantization_method != ATTACHABLE_QUANTIZATION:
        raise not_attachable("quantized", quantization_method=loaded.quantization_method)
    model = loaded.model
    for name, param in model.named_parameters(remove_duplicate=False):
        if param.device.type != "cuda":
            raise not_attachable("not_on_gpu", name, device=str(param.device))
        dtype = _dtype_name(param.dtype)
        if dtype not in ATTACHABLE_DTYPES:
            kept = _keep_in_fp32_modules(model)
            if kept:
                # D16: these models keep some modules in float32 by design; say so, rather
                # than leave the operator to wonder why a bfloat16 load holds a float32 tensor.
                raise not_attachable("dtype", name, dtype=dtype, keep_in_fp32_modules=kept)
            raise not_attachable("dtype", name, dtype=dtype)
    if not loaded.attachable_allocation:
        raise not_attachable("allocator")
    if cbm_enabled:
        raise not_attachable("continuous_batching_running")
    if int(max_concurrent) != 1:
        raise not_attachable("concurrency", max_concurrent=int(max_concurrent))


# --- export ---------------------------------------------------------------------------------


def _b64(value: Optional[bytes]) -> Optional[str]:
    """Base64 of a handle's bytes; None stays None (an event handle can be absent)."""
    if value is None:
        return None
    return base64.b64encode(bytes(value)).decode("ascii")


def _share_cuda(param: Any) -> tuple[Any, ...]:
    """The eight-field tuple torch's own reducer takes (`reductions.py:343-353`)."""
    return param._typed_storage()._share_cuda_()


def _storage_key(param: Any) -> tuple[Optional[int], int]:
    """One key per storage: its device index and the address of its first byte (I6)."""
    storage = param.untyped_storage()
    return (storage.device.index, storage.data_ptr())


def uuid_for_index(index: int) -> Optional[str]:
    """The `GPU-…` UUID of torch device `index` (C6). Called only for the resident model's cards,
    so no CUDA context is created on any other card."""
    import torch

    from millm.ml.gpu_placement import normalize_gpu_uuid

    return normalize_gpu_uuid(str(torch.cuda.get_device_properties(index).uuid))


def read_driver_version(path: Path = NVIDIA_VERSION_FILE) -> Optional[str]:
    """The NVIDIA kernel module's version, or None when the file is absent or unreadable.

    torch reports the CUDA driver API version, not the driver, and pynvml is not installed
    (031 review D17). None means "not recorded", never a guess.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    match = _DRIVER_VERSION.search(text)
    return match.group(1) if match else None


def served_config(model: Any) -> Optional[dict[str, Any]]:
    """The served model's architecture config, EXACTLY as loaded (suite decision D39).

    A trainer in another process rebuilds the model graph around the exported tensors, and the
    only authoritative description of that graph is the config this process instantiated - a
    copy fetched from the Hub can differ (a checkpoint, a local model, a revision). Serialised
    through the config's own JSON writer with ``use_diff=False``, so every attribute is stated
    and the value is plain JSON. ``None`` when the model carries no Hugging Face config.
    """
    config = getattr(model, "config", None)
    to_json = getattr(config, "to_json_string", None)
    if not callable(to_json):
        return None
    loaded = json.loads(to_json(use_diff=False))
    return loaded if isinstance(loaded, dict) else None


def producer_versions() -> dict[str, Optional[str]]:
    """torch, CUDA and driver versions, so a consumer can refuse an incompatible build first
    (FR-31.1.4)."""
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": read_driver_version(),
    }


@dataclass
class ExportedHandles:
    """What `export_handles` produces. `keep_alive` holds every shared storage for the
    attachment's lifetime (as the reducer's `shared_cache` does), so nothing is freed while a
    handle is open. The repr carries counts only: handles are never logged."""

    storages: list[dict[str, Any]] = field(repr=False)
    parameters: list[dict[str, Any]] = field(repr=False)
    addresses: dict[str, int] = field(repr=False)
    keep_alive: list[Any] = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"ExportedHandles(storages={len(self.storages)}, "
            f"parameters={len(self.parameters)})"
        )


def export_handles(
    model: Any,
    *,
    uuid_for: Callable[[int], Optional[str]] = uuid_for_index,
    share: Callable[[Any], tuple[Any, ...]] = _share_cuda,
) -> ExportedHandles:
    """One storage entry per distinct storage and one parameter entry per name (FR-31.1).

    Args:
        model: the resident `nn.Module`.
        uuid_for: device index -> UUID; resolved once per card.
        share: param -> the `_share_cuda_` tuple. Injectable so the grouping is testable on the
            CPU; production uses torch's own call.
    """
    storages: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    addresses: dict[str, int] = {}
    keep_alive: list[Any] = []
    index_of: dict[tuple[Optional[int], int], int] = {}
    uuids: dict[int, Optional[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        key = _storage_key(param)
        dtype = _dtype_name(param.dtype)
        slot = index_of.get(key)
        if slot is None:
            (
                device,
                handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ) = share(param)
            device_index = int(device)
            if device_index not in uuids:
                uuids[device_index] = uuid_for(device_index)
            slot = len(storages)
            index_of[key] = slot
            storages.append(
                {
                    "index": slot,
                    "device_index": device_index,
                    "device_uuid": uuids[device_index],
                    "dtype": dtype,
                    "handle": _b64(handle),
                    "storage_size_bytes": int(storage_size_bytes),
                    "storage_offset_bytes": int(storage_offset_bytes),
                    "ref_counter_handle": _b64(ref_counter_handle),
                    "ref_counter_offset": int(ref_counter_offset),
                    "event_handle": _b64(event_handle),
                    "event_sync_required": bool(event_sync_required),
                }
            )
            keep_alive.append(param.untyped_storage())
        parameters.append(
            {
                "name": name,
                "storage": slot,
                "shape": [int(s) for s in param.shape],
                "stride": [int(s) for s in param.stride()],
                "storage_offset": int(param.storage_offset()),
                "dtype": dtype,
                "requires_grad": bool(param.requires_grad),
            }
        )
        addresses[name] = int(param.data_ptr())
    logger.info("weights_handles_exported", storages=len(storages), parameters=len(parameters))
    return ExportedHandles(
        storages=storages, parameters=parameters, addresses=addresses, keep_alive=keep_alive
    )


def address_fingerprint(model: Any) -> dict[str, int]:
    """Every parameter name (tied names included) -> its `data_ptr()` (FR-31.5.4)."""
    return {
        name: int(param.data_ptr())
        for name, param in model.named_parameters(remove_duplicate=False)
    }


def moved_addresses(model: Any, fingerprint: dict[str, int]) -> list[str]:
    """The names whose address differs from `fingerprint`, or that appeared or vanished."""
    now = address_fingerprint(model)
    moved = [name for name, ptr in fingerprint.items() if now.get(name) != ptr]
    moved.extend(name for name in now if name not in fingerprint)
    return moved
