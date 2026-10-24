"""Tensor input and output for checkpoint save and load (Feature 32, FTASKS 3.0).

Pure functions over a resident `torch.nn.Module` and a safetensors file. Nothing here imports a
service, the barrier or the database, so every rule is testable on a tiny model on the CPU. The
service decides *when* to call these; this module decides only *what* the bytes are.

FOUR RULES CARRY THE WEIGHT

* **One name per storage, and the same name `save_pretrained` writes.** A tied model holds one
  tensor under two names (`model.embed_tokens.weight` and `lm_head.weight`). safetensors refuses
  shared storage, and an in-place load that wrote both names would write one storage twice. Tensors
  are grouped by (storage pointer, storage offset, shape, stride) — a pointer alone would merge
  distinct views of one buffer and every zero-size tensor (pointer 0), so zero-size tensors are
  never grouped. Within a group the kept name is the tie SOURCE transformers declares, and only
  when the storages really match: transformers 5.15 declares
  `{'lm_head.weight': 'model.embed_tokens.weight'}` on Llama and LFM2 even with
  `tie_word_embeddings=False`, where the two are separate tensors and both are saved. With this
  rule the names miLLM writes equal the names `save_pretrained` writes, which is what lets a
  directory written by miForge (through transformers) be loaded in place here, and the reverse.

* **The mode is decided from the header alone.** `read_header` parses the 8-byte length and the
  JSON header and reads no tensor data, so deciding between an in-place copy and a full reload
  costs one small read, before the barrier is entered (FR-32.5.5).

* **An in-place copy keeps every address.** `copy_in_place` does `dst.copy_(src_cpu)` into the live
  tensor — never `dst.data = ...`, which would replace the storage and leave an attached trainer's
  CUDA IPC handles pointing at freed memory. The source stays on the host: a cross-device `copy_`
  needs no device temporary, whereas `src.to(dst.device)` would allocate one as large as the
  biggest tensor (the embedding) on a card that may be full of KV cache, and fail after the first
  write had already happened (032 review §5.4).

* **No cast, ever.** A dtype difference is a refusal (FTDD T8): a cast copy is not the verified
  bytes. Every check runs before the first write (FR-32.5.6).

Inbound compatibility: a `save_pretrained` directory (tied key dropped, `{"format": "pt"}`
metadata) is accepted; the metadata is returned and otherwise ignored.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from millm.core.logging import get_logger

_logger = get_logger(__name__)

#: torch dtype -> the safetensors header's dtype string.
SAFETENSORS_DTYPES: dict[torch.dtype, str] = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
for _attr, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    if hasattr(torch, _attr):
        SAFETENSORS_DTYPES[getattr(torch, _attr)] = _code

#: safetensors' own ceiling on the JSON header (`safetensors` refuses larger headers).
MAX_HEADER_BYTES = 100 * 1024 * 1024

#: Groups of state_dict names that share one tensor, each in state_dict order.
TieGroups = list[tuple[str, ...]]


class SafetensorsHeaderError(ValueError):
    """The file is not a well-formed safetensors file. Internal: the caller maps it."""


class InPlaceMismatchError(ValueError):
    """`copy_in_place` was handed a file that does not match the model. Raised before any write."""


def safetensors_dtype(dtype: torch.dtype) -> str:
    """The header string for `dtype`; an unsupported dtype gets a string no header contains."""
    return SAFETENSORS_DTYPES.get(dtype, f"unsupported:{dtype}")


# --------------------------------------------------------------------------- tie groups


def _declared_tie_sources(model: Any) -> dict[str, str]:
    """target name -> source name, as the installed transformers declares them.

    `all_tied_weights_keys` is the expanded, model-wide mapping in transformers 5.x (empty when the
    config does not tie); `_tied_weights_keys` is the class declaration, which may be a list in
    older versions or carry patterns. Only exact string pairs are used, and only as a NAME choice:
    whether two names are tied is decided by their storage, never by this mapping.
    """
    declared: dict[str, str] = {}
    for attr in ("_tied_weights_keys", "all_tied_weights_keys"):
        value = getattr(model, attr, None)
        if isinstance(value, Mapping):
            declared.update({str(k): str(v) for k, v in value.items()})
    return declared


def _storage_key(name: str, tensor: torch.Tensor) -> tuple:
    if tensor.numel() == 0:
        # Every empty tensor reports storage pointer 0; grouping them would "tie" unrelated
        # names. An empty tensor is its own group.
        return ("empty", name)
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def tie_groups(model: Any) -> TieGroups:
    """Every state_dict name, grouped by the tensor it really is, in state_dict order."""
    groups: dict[tuple, list[str]] = {}
    for name, tensor in model.state_dict(keep_vars=True).items():
        groups.setdefault(_storage_key(name, tensor), []).append(name)
    return [tuple(names) for names in groups.values()]


def canonical_state(model: Any) -> dict[str, torch.Tensor]:
    """One name per tensor: the declared tie source when it is in the group, else the first name.

    The values are the LIVE tensors (`keep_vars=True`), so `copy_in_place` writes the model's own
    storage. Non-persistent buffers (rotary frequencies) are not in `state_dict()` and are never
    written, matching `save_pretrained`.
    """
    live = model.state_dict(keep_vars=True)
    declared = _declared_tie_sources(model)
    state: dict[str, torch.Tensor] = {}
    for group in tie_groups(model):
        members = set(group)
        sources = {declared[n] for n in group if n in declared and declared[n] in members}
        keep = next((n for n in group if n in sources), group[0])
        state[keep] = live[keep]
    return state


#: Pin the snapshot only when it is at most this share of the host memory available (the save
#: has already refused below the snapshot plus 10%). Pinned pages cannot be swapped or reclaimed
#: while held, so a snapshot that would pin most of the host falls back to pageable memory.
PINNED_SNAPSHOT_MAX_FRACTION = 0.5


def allocate_pinned_snapshot(
    model: Any, *, available_mb: int
) -> tuple[dict[str, torch.Tensor] | None, str | None]:
    """Page-locked host buffers shaped like the canonical state, allocated BEFORE the serving pause.

    Measured on the node (hardware record 2026-10-16, finding 4): a device-to-host copy of
    LFM2.5-1.2B (2.34 GB) into fresh pageable memory takes 1.09-1.14 s, over the 1,000 ms target
    for the save hold; into pinned memory with `non_blocking` copies and one synchronise, 348 ms.
    Pinning fresh memory is itself slow (about 1.2 s for the same size), so it happens here, outside
    the pause, and only the copy runs inside it.

    Returns `(buffers, None)`, or `(None, reason)` when the snapshot should use pageable memory:
    `no_cuda` (nothing to copy from a card), `too_large` (above
    `PINNED_SNAPSHOT_MAX_FRACTION` of `available_mb`, or that figure is unmeasured), or
    `pin_failed` (the allocation raised). The caller logs the reason; nothing here refuses.
    """
    state = canonical_state(model)
    if not state or not any(t.is_cuda for t in state.values()) or not torch.cuda.is_available():
        return None, "no_cuda"
    need = sum(t.numel() * t.element_size() for t in state.values())
    if available_mb <= 0 or need > available_mb * 1024 * 1024 * PINNED_SNAPSHOT_MAX_FRACTION:
        return None, "too_large"
    try:
        return {
            name: torch.empty(tuple(t.shape), dtype=t.dtype, pin_memory=True)
            for name, t in state.items()
        }, None
    except (RuntimeError, MemoryError):
        return None, "pin_failed"


def release_pinned_host_memory() -> None:
    """Return the caching host allocator's freed pinned blocks to the OS after the write, so a
    save does not keep the model's size of host memory page-locked until the process exits."""
    empty = getattr(torch._C, "_host_emptyCache", None)
    if empty is not None and torch.cuda.is_available():
        empty()


def snapshot_to_host(
    model: Any, into: Mapping[str, torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    """A contiguous host COPY of the canonical state, safe to write after serving resumes.

    With `into` (from `allocate_pinned_snapshot`), each tensor is copied into its pinned buffer
    with `non_blocking=True` and every source card is synchronised once before returning, so the
    bytes are on the host when this returns. When `into` does not describe the state now resident
    (another model loaded between the allocation and the pause), the pageable path runs instead
    and the fallback is logged (`weights_save_snapshot_pageable`, reason `state_changed`).

    `copy=True` matters on a CPU model: without it `.to("cpu")` returns the live tensor and the
    "snapshot" would change with the next commit.
    """
    with torch.no_grad():
        state = canonical_state(model)
        if into is not None:
            if set(into) == set(state) and all(
                tuple(into[n].shape) == tuple(t.shape) and into[n].dtype == t.dtype
                for n, t in state.items()
            ):
                devices = set()
                for name, tensor in state.items():
                    into[name].copy_(tensor.detach(), non_blocking=tensor.is_cuda)
                    if tensor.is_cuda:
                        devices.add(tensor.device)
                for device in devices:
                    torch.cuda.synchronize(device)
                return dict(into)
            _logger.info("weights_save_snapshot_pageable", reason="state_changed")
        return {
            name: tensor.detach().to(
                "cpu", copy=True, memory_format=torch.contiguous_format
            )
            for name, tensor in state.items()
        }


# --------------------------------------------------------------------------- header


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class SafetensorsHeader:
    tensors: dict[str, TensorSpec]
    metadata: dict[str, str] | None


def read_header(path: str | os.PathLike[str]) -> SafetensorsHeader:
    """Parse a safetensors header without reading any tensor data.

    Reads exactly 8 + N bytes. The data offsets are checked against the file size from `fstat`,
    so a truncated file is refused here rather than failing after the first write.
    """
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise SafetensorsHeaderError(f"{path}: shorter than the 8-byte header length")
        length = int.from_bytes(prefix, "little")
        if length <= 0 or length > MAX_HEADER_BYTES or 8 + length > size:
            raise SafetensorsHeaderError(f"{path}: header length {length} is not plausible")
        raw = handle.read(length)
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsHeaderError(f"{path}: header is not JSON") from exc
    if not isinstance(doc, dict):
        raise SafetensorsHeaderError(f"{path}: header is not an object")

    metadata = doc.pop("__metadata__", None)
    if metadata is not None and not (
        isinstance(metadata, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
    ):
        raise SafetensorsHeaderError(f"{path}: __metadata__ must map strings to strings")

    data_bytes = size - 8 - length
    tensors: dict[str, TensorSpec] = {}
    for name, entry in doc.items():
        try:
            dtype = entry["dtype"]
            shape = tuple(int(d) for d in entry["shape"])
            begin, end = (int(o) for o in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetensorsHeaderError(f"{path}: malformed entry for {name!r}") from exc
        if not isinstance(dtype, str) or any(d < 0 for d in shape):
            raise SafetensorsHeaderError(f"{path}: malformed entry for {name!r}")
        if not 0 <= begin <= end <= data_bytes:
            raise SafetensorsHeaderError(f"{path}: {name!r} lies outside the file")
        tensors[name] = TensorSpec(shape=shape, dtype=dtype)
    return SafetensorsHeader(tensors=tensors, metadata=metadata)


# --------------------------------------------------------------------------- mode decision

MismatchReason = Literal["missing", "unexpected", "shape", "dtype"]


@dataclass(frozen=True)
class Mismatch:
    reason: MismatchReason
    name: str
    expected: str | None
    found: str | None


@dataclass(frozen=True)
class InPlaceVerdict:
    mismatches: tuple[Mismatch, ...]

    @property
    def in_place(self) -> bool:
        return not self.mismatches

    @property
    def reasons(self) -> tuple[str, ...]:
        """The distinct mismatch reasons, in a stable order."""
        order = ("missing", "unexpected", "shape", "dtype")
        present = {m.reason for m in self.mismatches}
        return tuple(r for r in order if r in present)


def expected_specs(model: Any) -> dict[str, TensorSpec]:
    """What a file must contain to be copied in place into `model`."""
    return {
        name: TensorSpec(shape=tuple(t.shape), dtype=safetensors_dtype(t.dtype))
        for name, t in canonical_state(model).items()
    }


def compare_specs(
    expected: Mapping[str, TensorSpec], found: Mapping[str, TensorSpec]
) -> InPlaceVerdict:
    """Every name, shape and dtype difference between two tensor listings."""
    mismatches: list[Mismatch] = []
    for name, want in expected.items():
        have = found.get(name)
        if have is None:
            mismatches.append(Mismatch("missing", name, f"{want.dtype}{list(want.shape)}", None))
            continue
        if have.shape != want.shape:
            mismatches.append(Mismatch("shape", name, str(list(want.shape)), str(list(have.shape))))
        if have.dtype != want.dtype:
            mismatches.append(Mismatch("dtype", name, want.dtype, have.dtype))
    for name, have in found.items():
        if name not in expected:
            mismatches.append(Mismatch("unexpected", name, None, f"{have.dtype}{list(have.shape)}"))
    return InPlaceVerdict(mismatches=tuple(mismatches))


def compare_for_in_place(model: Any, header: SafetensorsHeader) -> InPlaceVerdict:
    """Names, shapes and dtypes of `header` against the model after tie canonicalisation."""
    return compare_specs(expected_specs(model), header.tensors)


LoadMode = Literal["in_place", "full_reload"]


@dataclass(frozen=True)
class ModeDecision:
    mode: LoadMode
    #: Why the load cannot be in place: "engine", "base_model", then the verdict's reasons.
    reasons: tuple[str, ...]
    verdict: InPlaceVerdict


def decide_mode(
    verdict: InPlaceVerdict, *, engine_is_transformers: bool, base_model_matches: bool
) -> ModeDecision:
    """FR-32.5.5: in place only when every condition holds; every failing one is reported.

    The service supplies the two facts it owns (the resident engine and the identity's
    `base_model`); the tensor comparison came from the header alone.
    """
    reasons: list[str] = []
    if not engine_is_transformers:
        reasons.append("engine")
    if not base_model_matches:
        reasons.append("base_model")
    reasons.extend(verdict.reasons)
    return ModeDecision(
        mode="full_reload" if reasons else "in_place", reasons=tuple(reasons), verdict=verdict
    )


# --------------------------------------------------------------------------- in-place copy


class _SafeOpenHandle(Protocol):
    def keys(self) -> Any: ...
    def get_slice(self, name: str) -> Any: ...
    def get_tensor(self, name: str) -> torch.Tensor: ...


def copy_in_place(model: Any, source: str | os.PathLike[str] | _SafeOpenHandle) -> int:
    """Copy every tensor of `source` into the model's live storage. Returns the tensor count.

    `source` is a path, or a `safe_open` handle the caller opened before hashing (so a
    rename-replace between the hash and the copy cannot change the bytes copied; 032 review §5.5).

    Every name, shape and dtype is checked BEFORE the first write; a mismatch raises
    `InPlaceMismatchError` with the model untouched. After the first write only an I/O or device
    error can fail, and the caller marks the weights `torn`.
    """
    if isinstance(source, (str, os.PathLike)):
        from safetensors import safe_open

        with safe_open(Path(source), framework="pt", device="cpu") as handle:
            return _copy_from(model, handle)
    return _copy_from(model, source)


def _copy_from(model: Any, handle: _SafeOpenHandle) -> int:
    targets = canonical_state(model)
    found = {}
    # A `safe_open` handle has `keys()` but is not iterable; `in handle` raises TypeError.
    for name in handle.keys():  # noqa: SIM118
        piece = handle.get_slice(name)
        found[name] = TensorSpec(shape=tuple(piece.get_shape()), dtype=str(piece.get_dtype()))
    expected = {
        name: TensorSpec(shape=tuple(t.shape), dtype=safetensors_dtype(t.dtype))
        for name, t in targets.items()
    }
    verdict = compare_specs(expected, found)
    if not verdict.in_place:
        raise InPlaceMismatchError(
            f"checkpoint does not match the resident model ({', '.join(verdict.reasons)}); "
            f"first: {verdict.mismatches[0]}"
        )

    devices: set[torch.device] = set()
    with torch.no_grad():
        for name, dst in targets.items():
            src = handle.get_tensor(name)  # host tensor; never moved to the device first
            dst.copy_(src)
            devices.add(dst.device)
    for device in devices:
        if device.type == "cuda":
            # Surface an asynchronous device error HERE, inside the caller's barrier.
            torch.cuda.synchronize(device)
    return len(targets)
