"""
Which GPU a model goes on.

The node that serves miLLM carries more than one card (an RTX 3080 Ti at index
0 beside an RTX 3090 at index 1 since 2026-09-13), and more may be added. Every
decision about which card a job uses lives here, and nowhere else: a bare
"cuda", a `mem_get_info()` with no argument or a `max_memory={0: ...}` means
GPU 0, whatever card that happens to be. Those were how the two-GPU node first
went wrong — metrics read as "No GPU", SAEs landed on the wrong card, and a
model that fit the 3090 was refused because the 3080 Ti was full.
`tests/unit/test_no_hardcoded_gpu_device.py` keeps such choices out of the rest
of `millm/`.

Policy (operator decisions, 2026-09-13):
  * Auto picks the card with the MOST free memory, read live, that fits.
  * Nothing is reserved for other applications; live free memory is the budget.
  * An explicitly requested card is honoured if it fits, and refused if it does
    not. It is never silently swapped for another card.
  * A model no single card can hold falls back to spreading across every card.
    That is Phase 1's stand-in for real sharding; Phase 2 replaces it.
"""

from __future__ import annotations

import itertools
import uuid as _uuid
from dataclasses import dataclass
from typing import Any, Optional, Union

import structlog
import torch

from millm.core.errors import GpuNotFoundError, InsufficientMemoryError
from millm.ml.memory_utils import list_gpu_memory

logger = structlog.get_logger()

_MIB = 1024 * 1024
_GIB = 1024 ** 3

#: The request value meaning "let the resolver choose".
AUTO = "auto"

#: The whole model on one card.
MODE_SINGLE = "single"
#: Spread across every visible card (device_map="auto" / llama.cpp layer split).
MODE_ALL = "all"
#: No card is used. Only GGUF may run like this.
MODE_CPU = "cpu"

REASON_MOST_FREE = "most_free_card_fits"
REASON_REQUESTED = "requested_card"
REASON_NO_SINGLE_CARD = "no_single_card_fits"
REASON_SIZE_UNKNOWN = "size_unknown"
REASON_NO_GPU = "no_gpu"

#: None or "auto", a CUDA index, or a UUID as nvidia-smi prints it.
GpuRequest = Union[None, int, str]

#: What bitsandbytes may place on a card, as a share of that card's free memory.
_BNB_GPU_FRACTION = 0.9


@dataclass(frozen=True)
class GpuInfo:
    """One visible card at the moment it was read."""

    index: int
    name: str
    uuid: Optional[str]
    total_mb: int
    free_mb: int

    @property
    def device_label(self) -> str:
        return f"cuda:{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "uuid": self.uuid,
            "total_mb": self.total_mb,
            "free_mb": self.free_mb,
        }


def normalize_gpu_uuid(value: Any) -> Optional[str]:
    """A GPU UUID in the `GPU-xxxxxxxx-xxxx-...` form nvidia-smi prints.

    torch reports the bare UUID (no prefix), nvidia-smi and the k8s device
    plugin report it with one, and operators paste either. Comparing them
    unnormalised would make a correct UUID read as an unknown card.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 16:
            return None
        text = str(_uuid.UUID(bytes=bytes(value)))
    else:
        text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("mig-") and len(lowered) > len("mig-"):
        # MIG instance ids are not plain UUIDs across driver generations
        # (MIG-<uuid> and MIG-GPU-<uuid>/gi/ci); pass them through as given.
        return "MIG-" + lowered[len("mig-"):]
    if lowered.startswith("gpu-"):
        lowered = lowered[len("gpu-"):]
    try:
        # The remainder must be a real UUID. Accepting any "GPU-..." string
        # let a typo through as a card id that could never match.
        return "GPU-" + str(_uuid.UUID(lowered))
    except ValueError:
        return None


def list_gpus() -> list[GpuInfo]:
    """Every visible card with name, UUID and live free/total memory.

    With CUDA_DEVICE_ORDER=PCI_BUS_ID (set in k8s and compose) the index here
    matches nvidia-smi and llama.cpp. A card whose properties cannot be read is
    still listed — its memory is what placement needs — with name "Unknown".
    """
    gpus: list[GpuInfo] = []
    for entry in list_gpu_memory():
        index = int(entry["index"])
        name, gpu_uuid = "Unknown", None
        try:
            props = torch.cuda.get_device_properties(index)
            name = str(getattr(props, "name", "") or "Unknown")
            gpu_uuid = normalize_gpu_uuid(getattr(props, "uuid", None))
        except Exception as e:  # noqa: BLE001 - a missing name must not hide a card
            logger.warning("gpu_properties_unavailable", index=index, error=str(e))
        gpus.append(
            GpuInfo(
                index=index,
                name=name,
                uuid=gpu_uuid,
                total_mb=int(entry["total_mb"]),
                free_mb=int(entry["free_mb"]),
            )
        )
    return gpus


def parse_gpu_request(requested: Any) -> Optional[Union[int, str]]:
    """Normalise a request to None (auto), a card index, or a normalised UUID.

    Raises:
        ValueError: for anything that is none of those. A typo must not quietly
            become "auto" — that would place the model on a card nobody chose.
    """
    if requested is None:
        return None
    if isinstance(requested, bool):
        raise ValueError("gpu must be 'auto', a GPU index, or a GPU UUID, not a boolean")
    if isinstance(requested, int):
        if requested < 0:
            raise ValueError(f"gpu index must be >= 0, got {requested}")
        return requested
    if isinstance(requested, str):
        text = requested.strip()
        if not text or text.lower() == AUTO:
            return None
        if text.isdigit():
            return int(text)
        normalised = normalize_gpu_uuid(text)
        if normalised is None:
            raise ValueError(
                f"gpu must be 'auto', a GPU index, or a GPU UUID (GPU-...), got {requested!r}"
            )
        return normalised
    raise ValueError(f"gpu must be 'auto', a GPU index, or a GPU UUID, got {requested!r}")


def find_gpu(gpus: list[GpuInfo], requested: Union[int, str]) -> GpuInfo:
    """The card a parsed request names.

    Raises:
        GpuNotFoundError: when no visible card matches.
    """
    for gpu in gpus:
        if isinstance(requested, int) and gpu.index == requested:
            return gpu
        if isinstance(requested, str) and gpu.uuid is not None and gpu.uuid == requested:
            return gpu
    raise GpuNotFoundError(
        f"No visible GPU matches {requested!r}",
        details={"requested": requested, "gpus": [gpu.to_dict() for gpu in gpus]},
    )


@dataclass(frozen=True)
class Placement:
    """Where one load goes: a single card, every card, or (GGUF only) the CPU."""

    mode: str
    reason: str
    required_mb: int
    gpus: tuple[GpuInfo, ...] = ()
    index: Optional[int] = None
    requested: Optional[Union[int, str]] = None

    @property
    def is_single(self) -> bool:
        return self.mode == MODE_SINGLE

    @property
    def chosen(self) -> Optional[GpuInfo]:
        if not self.is_single:
            return None
        return next((gpu for gpu in self.gpus if gpu.index == self.index), None)

    @property
    def gpu_indices(self) -> list[int]:
        """The cards this placement may allocate on."""
        if self.mode == MODE_SINGLE:
            return [int(self.index)] if self.index is not None else []
        if self.mode == MODE_ALL:
            return [gpu.index for gpu in self.gpus]
        return []

    @property
    def device_labels(self) -> list[str]:
        if self.mode == MODE_CPU:
            return ["cpu"]
        return [f"cuda:{index}" for index in self.gpu_indices]

    @property
    def device_label(self) -> Optional[str]:
        """"cuda:N" for a single card; None when there is no one device."""
        return f"cuda:{self.index}" if self.is_single else None

    @property
    def capacity_mb(self) -> int:
        """Free memory this placement can draw on, as read at decision time."""
        chosen = self.chosen
        if chosen is not None:
            return chosen.free_mb
        if self.mode == MODE_ALL:
            return sum(gpu.free_mb for gpu in self.gpus)
        return 0

    def transformers_device_map(self, bitsandbytes: bool) -> Any:
        """The `device_map` for `from_pretrained`.

        One card: `{"": "cuda:N"}` — everything on that card, no accelerate
        dispatch hooks and no cross-card copies in the forward pass.

        bitsandbytes keeps "auto" on purpose: its `max_memory` (see
        `bitsandbytes_max_memory`) names only the chosen card(s) and the CPU, so
        "auto" places on exactly those, and the CPU entry is what lets
        bitsandbytes stage weights while it quantizes.

        PHASE 1 FALLBACK: a model no single card holds gets "auto" across every
        card, which is how every load behaved before placement existed. Phase 2
        replaces this with planned sharding.
        """
        if self.mode == MODE_CPU:
            return None
        if self.is_single and not bitsandbytes:
            return {"": self.device_label}
        return "auto"

    def bitsandbytes_max_memory(self, max_cpu: str) -> dict[Any, str]:
        """`max_memory` for a bitsandbytes load, keyed to this placement's cards.

        It was `{0: ..., "cpu": ...}`, which put every Q8/Q4 model on GPU 0 —
        the smaller card on this node — whatever the other cards held.

        PHASE 2 MUST VERIFY: whether bitsandbytes needs the "cpu" entry only as
        load-time staging (weights quantized in host RAM, then moved) or also
        leaves layers there at run time. Decision 3 says transformers loads run
        on GPUs; if layers stay on the CPU, this entry violates it.
        """
        budget: dict[Any, str] = {}
        for index in self.gpu_indices:
            free_bytes, _ = torch.cuda.mem_get_info(index)
            budget[index] = f"{int(free_bytes * _BNB_GPU_FRACTION) // _GIB}GiB"
        budget["cpu"] = max_cpu
        return budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "requested": self.requested,
            "required_mb": self.required_mb,
            "capacity_mb": self.capacity_mb,
            "gpu_indices": self.gpu_indices,
            "devices": self.device_labels,
        }


def cpu_placement(reason: str = REASON_NO_GPU, required_mb: int = 0) -> Placement:
    """No card. Only GGUF may be placed like this (operator decision 3)."""
    return Placement(mode=MODE_CPU, reason=reason, required_mb=max(int(required_mb), 0))


def choose_gpu(
    required_mb: int,
    requested: GpuRequest = None,
    gpus: Optional[list[GpuInfo]] = None,
) -> Placement:
    """Decide where a job needing `required_mb` goes.

    Args:
        required_mb: Memory the job needs on a card. 0 or less means unknown.
        requested: None / "auto", a CUDA index, or a GPU UUID.
        gpus: The inventory to decide over; read live when omitted.

    Returns:
        A single-card Placement when one card fits (or the requested card fits),
        otherwise an all-cards Placement (Phase 1's stand-in for sharding).

    Raises:
        GpuNotFoundError: the requested card is not visible.
        InsufficientMemoryError: no GPU is visible, or the requested card lacks
            the free memory. The requested card is never swapped for another.
    """
    inventory = list_gpus() if gpus is None else list(gpus)
    wanted = parse_gpu_request(requested)
    required = max(int(required_mb or 0), 0)
    listing = [gpu.to_dict() for gpu in inventory]

    if not inventory:
        raise InsufficientMemoryError(
            "No GPU is visible to miLLM.",
            details={"required_mb": required, "available_mb": 0, "gpus": []},
        )

    if wanted is not None:
        card = find_gpu(inventory, wanted)
        if card.free_mb < required:
            raise InsufficientMemoryError(
                f"GPU {card.index} ({card.name}) has {card.free_mb} MB free; the model "
                f"needs ~{required} MB. The requested card is not swapped for another "
                "one; choose a different card or Auto.",
                details={
                    "required_mb": required,
                    "available_mb": card.free_mb,
                    "gpu": card.to_dict(),
                    "gpus": listing,
                },
            )
        return Placement(
            mode=MODE_SINGLE,
            reason=REASON_REQUESTED,
            required_mb=required,
            gpus=tuple(inventory),
            index=card.index,
            requested=wanted,
        )

    if required <= 0:
        # An unknown size cannot be shown to fit any card. Spreading across
        # every card is what every load did before placement existed, so an
        # unmeasured model keeps that rather than gambling on one card.
        return Placement(
            mode=MODE_ALL,
            reason=REASON_SIZE_UNKNOWN,
            required_mb=required,
            gpus=tuple(inventory),
        )

    # Most free first; on a tie the lower index, so the choice is stable.
    best = max(inventory, key=lambda gpu: (gpu.free_mb, -gpu.index))
    if best.free_mb >= required:
        return Placement(
            mode=MODE_SINGLE,
            reason=REASON_MOST_FREE,
            required_mb=required,
            gpus=tuple(inventory),
            index=best.index,
        )

    # PHASE 1: no single card holds it. Spread across every card as before;
    # Phase 2 replaces this with planned sharding.
    return Placement(
        mode=MODE_ALL,
        reason=REASON_NO_SINGLE_CARD,
        required_mb=required,
        gpus=tuple(inventory),
    )


def free_mb_by_index(indices: list[int]) -> dict[int, int]:
    """Live free memory for each named card. A card that cannot be read is omitted."""
    free: dict[int, int] = {}
    for index in indices:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
            free[index] = int(free_bytes / _MIB)
        except Exception as e:  # noqa: BLE001 - measurement, never a gate
            logger.warning("gpu_free_memory_unreadable", index=index, error=str(e))
    return free


def memory_used_by_device(
    before: dict[int, int], after: dict[int, int], indices: Optional[list[int]] = None
) -> dict[str, int]:
    """Per-card memory a load consumed: free before minus free after, in MB.

    Deltas, not "total minus free": the card is shared, and "used" would charge
    this model for miStudio's work on the same card.
    """
    keys = sorted(before) if indices is None else sorted(set(indices))
    return {
        f"cuda:{index}": max(before[index] - after[index], 0)
        for index in keys
        if index in before and index in after
    }


def _device_label(value: Any) -> Optional[str]:
    """Normalise an accelerate/torch device value to "cuda:N", "cpu", "disk"..."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"cuda:{value}"
    if isinstance(value, torch.device):
        if value.type == "cuda":
            return f"cuda:{value.index}" if value.index is not None else None
        return value.type
    text = str(value).strip()
    if text.isdigit():
        return f"cuda:{text}"
    if text.startswith("cuda:") and text[5:].isdigit():
        return text
    if text in ("cpu", "disk", "meta", "mps"):
        return text
    return None


def model_device_labels(model: Any) -> list[str]:
    """Every device a loaded transformers model actually holds tensors on.

    Read from `hf_device_map` AND the tensors themselves: a single-device
    `device_map` may leave no `hf_device_map`, and an accelerate map can name a
    device where a module ended up holding nothing.
    """
    labels: set[str] = set()
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            label = _device_label(value)
            if label is not None:
                labels.add(label)
    try:
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            label = _device_label(getattr(tensor, "device", None))
            if label is not None:
                labels.add(label)
    except Exception:  # noqa: BLE001 - a stub or exotic module: map alone
        pass
    return sorted(labels)


def gpu_indices_of(labels: list[str]) -> list[int]:
    """The CUDA indices among device labels."""
    return sorted(
        {int(label[5:]) for label in labels if label.startswith("cuda:") and label[5:].isdigit()}
    )
