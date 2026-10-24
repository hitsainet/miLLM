"""Float32 scoring: one request's scoring forward over a float32 COPY of the resident model.

Feature 25, FR-25.15 — suite decision D61 (2026-10-18), built for miForge F11 R-02.2 (miForge
`0xcc/tdds/011_FTDD|Optional_Hot_Update.md` §6.10, FTID §3.14).

WHY. miForge verifies a hot update by comparing the next-token log-probabilities it computes with
the ones miLLM serves. In bfloat16 the two processes differ by kernel numerics alone (0.06-0.22 on
the node), which hides a ~1 % reconstruction error. In float32 on both sides the clean gap falls to
~1e-6 while a 1 % error keeps its size, so a float32 comparison can see what a bfloat16 one cannot.

THE COPY, NOT THE SERVED MODULE. The forward runs on a module built for this request:

* its structure from the served model's own class and config, instantiated on the `meta` device
  (no memory), then every parameter and buffer assigned a float32 copy of the served tensor,
  placed on one device. Tied tensors stay tied (copied once). Non-floating tensors keep their
  dtype. Any tensor the copy has that the served model does not (or the reverse) refuses the
  request (`model_not_float32_castable`): an uninitialised tensor would score garbage.
* The SERVED module is never touched: not cast, not moved, not re-parametrised. Re-parametrising
  it in place (`torch.func.functional_call`) was rejected: the continuous batching manager
  generates WITHOUT the admission slot, so a concurrent chat would run on float32 tensors on
  another device for the length of the forward.
* The copy has no SAE, probe or sensing hooks, which is what scoring wants (always unsteered, X-09,
  and unrecorded). A served LoRA adapter IS part of the model being scored, and it is forward hooks,
  so it is re-registered on the copy (`adapter_hooks.install_on_copy`) — otherwise the copy would
  score the base while the response named the adapter.
* Released when the request ends, success or failure: every reference dropped and, on a card,
  `torch.cuda.empty_cache()` so the ~4 bytes/parameter go back to the card's other tenants (miForge
  time-slices the same RTX 3090). Never cached across requests: the weights can change between two
  requests (a hot-update commit), and a cached copy would score the old ones.

A MODEL ALREADY IN FLOAT32 needs no copy: its native forward IS the float32 forward, so it is
scored natively and echoed `float32` (true of what ran).

REFUSALS, NEVER A DOWNGRADE (409 `SCORING_PRECISION_UNAVAILABLE`, `details.reason`):
`model_not_float32_castable` (llama.cpp / GGUF, bitsandbytes or pre-quantized weights, integer
parameters, a structure the copy cannot reproduce), `memory` (the estimate plus headroom does not
fit where the copy would go, or the allocation failed), `device` (a pinned device that does not
exist or does not hold the model), `busy` (another float32 copy is live).
"""

from __future__ import annotations

import copy
import gc
import math
import threading
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn

from millm.core.errors import ScoringPrecisionUnavailableError
from millm.core.logging import get_logger

logger = get_logger(__name__)

MIB = 1024 * 1024

#: One float32 copy at a time, process-wide. Taken without waiting: a second copy would double
#: the memory the estimate admitted, so a concurrent float32 request is refused `busy`.
_COPY_LOCK = threading.Lock()


def _base_module(model: Any) -> Any:
    """The module under a `torch.compile` wrapper (whose parameter names carry `_orig_mod.`)."""
    return getattr(model, "_orig_mod", model)


def _named_tensors(module: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Every parameter and buffer by every name it has (tied tensors appear once per name)."""
    return list(module.named_parameters(remove_duplicate=False)) + list(
        module.named_buffers(remove_duplicate=False)
    )


def refuse_if_not_castable(loaded: Any, *, llamacpp: bool) -> None:
    """`model_not_float32_castable` for a model whose float32 copy would not be the served model."""
    name = getattr(loaded, "model_name", "the loaded model")
    if llamacpp:
        raise ScoringPrecisionUnavailableError(
            "model_not_float32_castable",
            f"'{name}' is a GGUF model served by llama.cpp; its quantized weights have no float32 "
            "form here and llama.cpp exposes no module to copy",
        )
    method = str(getattr(loaded, "quantization_method", "none") or "none").lower()
    model = _base_module(loaded.model)
    config_quant = getattr(getattr(model, "config", None), "quantization_config", None)
    if method not in ("none", "unknown") or config_quant is not None:
        raise ScoringPrecisionUnavailableError(
            "model_not_float32_castable",
            f"'{name}' is loaded quantized ({method}); a float32 copy of dequantized weights is "
            "not the model being served",
            details={"quantization": method},
        )
    integer = [n for n, p in model.named_parameters() if not p.is_floating_point()]
    if integer:
        raise ScoringPrecisionUnavailableError(
            "model_not_float32_castable",
            f"'{name}' has non-floating parameters ({', '.join(integer[:3])}"
            f"{'…' if len(integer) > 3 else ''}), which a float32 copy cannot represent",
        )


def already_float32(model: Any) -> bool:
    """Every floating parameter and buffer is float32: the native forward is the float32 forward."""
    floating = [t for _, t in _named_tensors(_base_module(model)) if t.is_floating_point()]
    return bool(floating) and all(t.dtype == torch.float32 for t in floating)


def float32_copy_bytes(model: Any) -> int:
    """Bytes the copy holds: 4 per floating element, native size otherwise; tied tensors once."""
    seen: set[int] = set()
    total = 0
    for _, tensor in _named_tensors(_base_module(model)):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * (4 if tensor.is_floating_point() else tensor.element_size())
    return total


def _cgroup_available_mb() -> int | None:
    """What the container may still allocate (cgroup v2), or None when it is unlimited/unknown."""
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        if limit == "max":
            return None
        with open("/sys/fs/cgroup/memory.current") as f:
            current = int(f.read().strip())
        return max(int(limit) - current, 0) // MIB
    except (OSError, ValueError):
        return None


def host_available_mb() -> int:
    """Host memory available to THIS process: the smaller of the host's and the cgroup's.

    A pod limit below the host's free memory turns an over-large CPU copy into an OOM kill of the
    whole server rather than a refusal, so the cgroup figure is read too. 0 when unknown, which
    refuses (an unknown amount is not known to fit)."""
    from millm.ml.memory_utils import get_available_cpu_memory_mb

    host = get_available_cpu_memory_mb()
    cgroup = _cgroup_available_mb()
    return host if cgroup is None else min(host, cgroup)


def cuda_free_mb(device: str) -> int:
    """Free memory on `device`, after returning torch's unused cached blocks to the card.

    Without the `empty_cache()` the blocks an earlier request freed (still reserved by torch)
    count as used, and a copy that fits would be refused `memory`."""
    index = torch.device(device).index or 0
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(index)
    return int(free // MIB)


@dataclass(frozen=True)
class Placement:
    device: str
    needed_mb: int
    available_mb: int


def choose_device(
    setting: str, input_device: str, needed_mb: int, headroom_mb: int, *,
    cuda_free: Any = cuda_free_mb, host_free: Any = host_available_mb,
) -> Placement:
    """Where the copy goes (FR-25.15.4), or the refusal naming why it cannot go anywhere.

    `setting` is `SCORING_FLOAT32_DEVICE`. A pinned device is never swapped for another: a caller
    who pinned `cpu` to keep the card free for training must not find 5 GB allocated on it.
    """
    setting = setting.strip().lower()
    on_card = input_device.startswith("cuda")
    details: dict[str, Any] = {"needed_mb": needed_mb, "headroom_mb": headroom_mb}

    def card_fits(card: str) -> Placement | None:
        free = int(cuda_free(card))
        details[f"{card}_free_mb"] = free
        return Placement(card, needed_mb, free) if free - headroom_mb >= needed_mb else None

    def cpu_fits() -> Placement | None:
        free = int(host_free())
        details["cpu_available_mb"] = free
        return Placement("cpu", needed_mb, free) if free - headroom_mb >= needed_mb else None

    if setting == "cuda" or setting.startswith("cuda:"):
        if setting == "cuda":
            if not on_card:
                raise ScoringPrecisionUnavailableError(
                    "device",
                    f"SCORING_FLOAT32_DEVICE is cuda but the model's inputs are on {input_device}",
                    details=details,
                )
            card = input_device
        else:
            card = setting
            index = int(setting.split(":", 1)[1])
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ScoringPrecisionUnavailableError(
                    "device", f"SCORING_FLOAT32_DEVICE is {setting}, which this process cannot see",
                    details=details,
                )
        placed = card_fits(card)
    elif setting == "cpu":
        placed = cpu_fits()
    else:  # auto
        placed = (card_fits(input_device) if on_card else None) or cpu_fits()
    if placed is None:
        where = ", ".join(f"{k} {v}" for k, v in details.items() if k.endswith("_mb"))
        raise ScoringPrecisionUnavailableError(
            "memory",
            f"the float32 copy needs {needed_mb} MB plus {headroom_mb} MB headroom and does not "
            f"fit on {'the configured device' if setting != 'auto' else 'the card or the CPU'} "
            f"({where})",
            details=details,
        )
    return placed


def _copy_tensor(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """A detached float32 copy of a floating tensor (any other dtype kept) on `device`."""
    dtype = torch.float32 if tensor.is_floating_point() else tensor.dtype
    return tensor.detach().to(device=device, dtype=dtype, copy=True)


def _flash_to_sdpa(config: Any) -> None:
    """Replace a flash-attention choice with sdpa on the config and each sub-config.

    flash-attention kernels have no float32 form; sdpa computes the same attention.
    Only the levels that chose flash change. Assigning the TOP level's setter would
    push "sdpa" onto every sub-config as an explicit request, and a tower with no SDPA
    path (Gemma 4's audio, which resolved to eager) refuses to be built.
    """
    sub_configs = getattr(config, "sub_configs", None)
    for key in sub_configs if isinstance(sub_configs, dict) else ():
        sub = getattr(config, key, None)
        if sub is not None:
            _flash_to_sdpa(sub)
    if getattr(config, "_attn_implementation", None) not in (None, "eager", "sdpa"):
        # The internal field: the property setter would recurse into the sub-configs.
        if hasattr(config, "_attn_implementation_internal"):
            config._attn_implementation_internal = "sdpa"
        else:
            config._attn_implementation = "sdpa"


def _clone_skeleton(model: nn.Module) -> nn.Module:
    """The served model's class and config, instantiated on `meta` (allocates nothing)."""
    config: Any = copy.deepcopy(model.config)
    _flash_to_sdpa(config)
    model_class: Any = type(model)
    with torch.device("meta"):
        return cast(nn.Module, model_class._from_config(config, dtype=torch.float32))


@dataclass
class Float32Copy:
    """A float32 copy of the resident model for ONE request. `close()` releases it."""

    module: nn.Module | None
    device: str
    needed_mb: int
    handles: list[Any] = field(default_factory=list)

    @classmethod
    def build(cls, model: Any, device: str, needed_mb: int, adapter: Any = None) -> Float32Copy:
        """Build the copy on `device`. Refuses (never half-builds) when it cannot."""
        base = _base_module(model)
        clone: nn.Module | None = None
        try:
            try:
                clone = _clone_skeleton(base)
            except Exception as exc:  # noqa: BLE001 - any construction failure is a refusal
                raise ScoringPrecisionUnavailableError(
                    "model_not_float32_castable",
                    f"the model's structure could not be rebuilt for a float32 copy "
                    f"({type(exc).__name__}: {exc})",
                ) from exc
            source = _named_tensors(base)
            mine = {n for n, _ in _named_tensors(clone)}
            theirs = {n for n, _ in source}
            if mine != theirs:
                differ = sorted(mine ^ theirs)
                raise ScoringPrecisionUnavailableError(
                    "model_not_float32_castable",
                    "a float32 copy of this model would not have the served model's tensors "
                    f"({len(differ)} differ, e.g. {', '.join(differ[:3])})",
                )
            params = {n for n, _ in base.named_parameters(remove_duplicate=False)}
            copies: dict[int, Any] = {}
            for name, tensor in source:
                key = id(tensor)
                if key not in copies:
                    data = _copy_tensor(tensor, device)
                    copies[key] = (
                        nn.Parameter(data, requires_grad=False) if name in params else data
                    )
                owner, _, leaf = name.rpartition(".")
                sub = clone.get_submodule(owner)
                if name in params:
                    sub._parameters[leaf] = copies[key]
                else:
                    sub._buffers[leaf] = copies[key]
            copies.clear()
            if any(t.is_meta for _, t in _named_tensors(clone)):
                raise ScoringPrecisionUnavailableError(
                    "model_not_float32_castable", "the float32 copy was left with unset tensors"
                )
            clone.eval()
            built = cls(module=clone, device=device, needed_mb=needed_mb)
            clone = None
            if adapter is not None:
                from millm.ml.adapter_hooks import install_on_copy

                assert built.module is not None
                built.handles = install_on_copy(built.module, adapter)
            return built
        except torch.cuda.OutOfMemoryError as exc:
            clone = None
            _release(device)
            raise ScoringPrecisionUnavailableError(
                "memory", f"allocating the float32 copy on {device} failed ({exc})",
                details={"needed_mb": needed_mb, "device": device},
            ) from exc
        except (MemoryError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "alloc" not in str(exc).lower():
                raise
            clone = None
            _release(device)
            raise ScoringPrecisionUnavailableError(
                "memory", f"allocating the float32 copy on {device} failed ({exc})",
                details={"needed_mb": needed_mb, "device": device},
            ) from exc
        finally:
            del clone

    def next_token_logits(self, inputs: Any) -> torch.Tensor:
        """The last position's logits as float32 on the CPU (the scorer's input), from the copy."""
        module = self.module
        if module is None:
            raise RuntimeError("the float32 copy was already released")
        # ⚠ FULL float32 matmuls. "high"/"medium" let float32 matmuls run as TF32 (cuBLAS) or
        # bfloat16 (oneDNN on the CPU) — ~1e-3 relative error, the very noise this mode removes.
        # The default is "highest", but any library may lower it process-wide; it is forced for
        # this forward and restored after (it governs float32 matmuls only, so the bfloat16
        # served model is unaffected).
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            with torch.no_grad():
                try:
                    outputs = module(**inputs, use_cache=False, logits_to_keep=1)
                except TypeError as exc:
                    if "logits_to_keep" not in str(exc):
                        raise
                    outputs = module(**inputs, use_cache=False)
            return cast(torch.Tensor, outputs.logits[0, -1].detach().float().cpu().clone())
        except torch.cuda.OutOfMemoryError as exc:
            raise ScoringPrecisionUnavailableError(
                "memory", f"the float32 forward ran out of memory on {self.device} ({exc})",
                details={"needed_mb": self.needed_mb, "device": self.device},
            ) from exc
        finally:
            torch.set_float32_matmul_precision(previous)

    def close(self) -> None:
        """Drop every reference to the copy and give its memory back (FR-25.15.6)."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.module = None
        _release(self.device)


def _release(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def try_acquire() -> bool:
    return _COPY_LOCK.acquire(blocking=False)


def release_lock() -> None:
    _COPY_LOCK.release()


def needed_mb(model: Any) -> int:
    return int(math.ceil(float32_copy_bytes(model) / MIB))
