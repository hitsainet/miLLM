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

Deciding where a job goes must not itself take memory from a card. The
inventory is read from nvidia-smi, which needs no CUDA context in this process,
and mapped to torch indices by UUID. `torch.cuda.mem_get_info(i)` creates a
context on card i (a few hundred MB, kept for the life of the process), so it is
used only on cards a model is already on or is being placed on.

Policy (operator decisions, 2026-09-13):
  * Auto picks the card with the MOST free memory, read live, that fits.
  * Nothing is reserved for other applications; live free memory is the budget.
  * An explicitly requested card is honoured if it fits, and refused if it does
    not. It is never silently swapped for another card. "all" (split across
    every visible card) is an explicit request too.
  * A model no single card can hold is SPLIT across GPUs: the cards with the
    most free memory first, and only as many as it needs. A transformers model
    never spills to the CPU or disk — a split that cannot hold it is refused.
    Only GGUF may run on the CPU, and today only wholly, when no card has room:
    a partial CPU offload is allowed for GGUF and not implemented.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import uuid as _uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import structlog
import torch

from millm.core.errors import GpuNotFoundError, InsufficientMemoryError, SplitNotHonouredError
from millm.ml import nvidia_smi

logger = structlog.get_logger()

_MIB = 1024 * 1024

#: The request value meaning "let the resolver choose".
AUTO = "auto"
#: The request value meaning "split across every visible card". Needed to
#: force-split a model that fits one card (the sharded-vs-single equivalence
#: check), and honoured or refused like any explicit choice.
ALL = "all"

#: The whole model on one card.
MODE_SINGLE = "single"
#: Split across several cards, each with a planned share of GPU memory and no
#: CPU or disk entry. Phase 1 reported a split as "all": `device_map="auto"`
#: with no `max_memory`, which accelerate was free to finish on the CPU.
MODE_SHARD = "shard"
#: No card is used. Only GGUF may run like this.
MODE_CPU = "cpu"

REASON_MOST_FREE = "most_free_card_fits"
REASON_REQUESTED = "requested_card"
REASON_REQUESTED_ALL = "requested_all_cards"
REASON_NO_SINGLE_CARD = "no_single_card_fits"
REASON_SIZE_UNKNOWN = "size_unknown"
REASON_NO_GPU = "no_gpu"

#: None or "auto", "all", a CUDA index, or a UUID as nvidia-smi prints it.
GpuRequest = Union[None, int, str]

#: Memory kept back on every card of a split, in MB: its CUDA context, cuBLAS
#: workspaces, and the activations and KV cache of the layers it runs.
#: `max_memory` is a budget for WEIGHTS — accelerate fills a card up to it — so a
#: split planned to each card's whole free memory would leave the forward pass
#: nothing on any card. A single card needs no such term: the estimate already
#: carries a 20% runtime overhead (memory_utils.MEMORY_OVERHEAD_FACTOR) and all
#: of it lands on that card.
SHARD_RESERVE_MB = 1024

#: What transformers multiplies `max_memory` by for a bitsandbytes load before
#: placing layers (`quantizer_bnb_{4,8}bit.adjust_max_memory`, transformers
#: 5.15.1). Counted in the plan and never applied to the map: miLLM applied its
#: own 0.9 on top, so a bitsandbytes load could use 81% of what was budgeted.
BNB_MAX_MEMORY_FACTOR = 0.9

#: Labels that mean a module is NOT on a GPU. "meta" is how a disk-offloaded
#: parameter looks after dispatch.
OFF_GPU_LABELS = ("cpu", "disk", "meta")


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


def _torch_index_by_uuid() -> dict[str, int]:
    """Torch's index for each card it can see, keyed by normalised UUID.

    `get_device_properties` reads the device's properties without creating a
    context. Matching by UUID, not by position, stays correct if torch's order
    ever differs from nvidia-smi's (CUDA_VISIBLE_DEVICES, a CUDA_DEVICE_ORDER
    that is not PCI_BUS_ID), where equal indices would be a coincidence.
    """
    mapping: dict[str, int] = {}
    try:
        count = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001 - no card is an answer, not an error
        logger.warning("gpu_device_count_unavailable", error=str(e))
        return mapping
    for index in range(count):
        try:
            props = torch.cuda.get_device_properties(index)
        except Exception as e:  # noqa: BLE001
            logger.warning("gpu_properties_unavailable", index=index, error=str(e))
            continue
        gpu_uuid = normalize_gpu_uuid(getattr(props, "uuid", None))
        if gpu_uuid is not None:
            mapping[gpu_uuid] = index
    return mapping


def list_gpus() -> list[GpuInfo]:
    """Every card this process can use, with live free/total memory, by torch index.

    Memory, name and UUID come from nvidia-smi; the index is torch's, found by
    UUID. No CUDA context is created on any card. A card nvidia-smi reports but
    torch cannot see is left out, and so is everything when either says there
    are no cards — nvidia-smi absent means the CPU path, as when torch has no
    CUDA.
    """
    if not torch.cuda.is_available():
        return []
    reported = nvidia_smi.query_gpus()
    if not reported:
        return []
    torch_index = _torch_index_by_uuid()
    gpus: list[GpuInfo] = []
    for card in reported:
        gpu_uuid = normalize_gpu_uuid(card.get("uuid"))
        index = torch_index.get(gpu_uuid) if gpu_uuid is not None else None
        if index is None:
            logger.warning(
                "gpu_not_mapped_to_torch",
                nvidia_smi_index=card.get("index"),
                uuid=card.get("uuid"),
                detail="nvidia-smi reports this card but torch has no device with its UUID",
            )
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=str(card.get("name") or "Unknown"),
                uuid=gpu_uuid,
                total_mb=int(card.get("memory_total_mb", 0)),
                free_mb=int(card.get("memory_free_mb", 0)),
            )
        )
    return sorted(gpus, key=lambda gpu: gpu.index)


def project_free_after_unload(
    gpus: list[GpuInfo], resident_memory_by_device_mb: Optional[dict[str, int]]
) -> list[GpuInfo]:
    """What each card will have once the resident model is unloaded.

    Live free memory plus that model's recorded usage on the card, capped at the
    card's total. A load is judged against this BEFORE the resident model is
    unloaded, so a refusal no longer costs the operator the model they were
    serving — and a switch that only fits once the old model is gone is not
    refused for the memory that model is about to give back.
    """
    usage = resident_memory_by_device_mb or {}
    return [
        dataclasses.replace(
            gpu,
            free_mb=min(gpu.total_mb, gpu.free_mb + max(int(usage.get(gpu.device_label, 0)), 0)),
        )
        for gpu in gpus
    ]


def parse_gpu_request(requested: Any) -> Optional[Union[int, str]]:
    """Normalise a request to None (auto), ALL, a card index, or a normalised UUID.

    Raises:
        ValueError: for anything that is none of those. A typo must not quietly
            become "auto" — that would place the model on a card nobody chose.
    """
    if requested is None:
        return None
    if isinstance(requested, bool):
        raise ValueError(
            "gpu must be 'auto', 'all', a GPU index, or a GPU UUID, not a boolean"
        )
    if isinstance(requested, int):
        if requested < 0:
            raise ValueError(f"gpu index must be >= 0, got {requested}")
        return requested
    if isinstance(requested, str):
        text = requested.strip()
        if not text or text.lower() == AUTO:
            return None
        if text.lower() == ALL:
            return ALL
        if text.isdigit():
            return int(text)
        normalised = normalize_gpu_uuid(text)
        if normalised is None:
            raise ValueError(
                "gpu must be 'auto', 'all', a GPU index, or a GPU UUID (GPU-...), "
                f"got {requested!r}"
            )
        return normalised
    raise ValueError(
        f"gpu must be 'auto', 'all', a GPU index, or a GPU UUID, got {requested!r}"
    )


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
class ShardRule:
    """How a split is sized.

    `need_mb` is what the model needs across every card it uses, and
    `limit_mb(card)` is what one card can be given. Transformers and llama.cpp
    count a card differently (a flat reserve here, a planning fraction and
    per-card runtime overhead there), so the rule comes from the engine.

    `max_memory_factor` is what the engine itself multiplies a limit by before
    placing; the plan counts it, and the limit handed over does not repeat it.

    `fill_in_index_order` says how the engine FILLS the cards a split chose.
    llama.cpp takes proportions, so the cards taken first can carry the most.
    accelerate cannot be given an order at all (see
    `Placement.transformers_max_memory`), so its shares are planned the way it
    fills: lowest index first, each card to its whole budget.
    """

    need_mb: int
    limit_mb: Callable[[GpuInfo], int]
    max_memory_factor: float = 1.0
    fill_in_index_order: bool = False


def transformers_shard_rule(
    need_mb: int, bitsandbytes: bool = False, max_memory_factor: Optional[float] = None
) -> ShardRule:
    """A split for `from_pretrained`: each card's free memory less SHARD_RESERVE_MB.

    `max_memory_factor` is what transformers' quantizer multiplies `max_memory`
    by for this load (`adjust_max_memory`). `bitsandbytes` is the shorthand for
    miLLM's own Q4/Q8 load; a checkpoint that ships its own quantization passes
    the factor its quantizer asks for (model_loader.pre_quantized_max_memory_factor).
    """
    if max_memory_factor is None:
        max_memory_factor = BNB_MAX_MEMORY_FACTOR if bitsandbytes else 1.0
    return ShardRule(
        need_mb=need_mb,
        limit_mb=lambda gpu: max(gpu.free_mb - SHARD_RESERVE_MB, 0),
        max_memory_factor=max_memory_factor,
        fill_in_index_order=True,
    )


@dataclass(frozen=True)
class Placement:
    """Where one load goes: a single card, a split across cards, or (GGUF only) the CPU."""

    mode: str
    reason: str
    required_mb: int
    gpus: tuple[GpuInfo, ...] = ()
    index: Optional[int] = None
    requested: Optional[Union[int, str]] = None
    #: A split's planned share per card, (index, MB), in the order the engine
    #: fills them: most free first for llama.cpp, index order for accelerate.
    shares: tuple[tuple[int, int], ...] = ()
    #: A split's limit per card, (index, MB): what the engine may place there
    #: before its own `max_memory_factor`.
    limits: tuple[tuple[int, int], ...] = ()
    max_memory_factor: float = 1.0

    @property
    def is_single(self) -> bool:
        return self.mode == MODE_SINGLE

    @property
    def is_shard(self) -> bool:
        return self.mode == MODE_SHARD

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
        if self.mode == MODE_SHARD:
            return sorted(index for index, _ in self.shares)
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
    def planned_mb_by_index(self) -> dict[int, int]:
        return dict(self.shares)

    @property
    def budget_mb_by_index(self) -> dict[int, int]:
        """What each card of a split can hold, after the engine's own factor."""
        return {index: int(limit * self.max_memory_factor) for index, limit in self.limits}

    @property
    def capacity_mb(self) -> int:
        """Free memory of the card(s) this placement uses, as read at decision time."""
        chosen = self.chosen
        if chosen is not None:
            return chosen.free_mb
        if self.is_shard:
            used = set(self.gpu_indices)
            return sum(gpu.free_mb for gpu in self.gpus if gpu.index in used)
        return 0

    @property
    def budget_mb(self) -> int:
        """What this placement can hold: a card's free memory, or a split's summed budgets."""
        if self.is_shard:
            return sum(self.budget_mb_by_index.values())
        return self.capacity_mb

    def transformers_device_map(self) -> Any:
        """The `device_map` for `from_pretrained`.

        One card, whatever the quantization: `{"": "cuda:N"}` — everything on
        that card, no accelerate dispatch hooks and no cross-card copies in the
        forward pass. bitsandbytes used to take "auto" with a "cpu" entry in
        `max_memory`, on the belief that it staged weights in host RAM while
        quantizing. It does not (transformers 5.15.1 materialises each weight on
        its device-map target and quantizes it there:
        `core_model_loading.py:1677-1697`, `integrations/bitsandbytes.py:58`);
        the entry only let a model that did not fit finish on the CPU.

        A split: "sequential" over `transformers_max_memory()`. NOT "auto".
        "auto" is accelerate's balanced map, which caps every card but the
        highest-index one at about model/N (`get_balanced_memory`): on cards of
        uneven size a model the planned budgets hold was put partly on disk.
        "sequential" fills the cards in index order, each up to the limit given
        — less what accelerate holds back on the first of them (see
        `transformers_max_memory`).

        A split of an unmeasured model has no plan to follow, so it keeps
        "auto": balanced across every card, GPU-only, checked after the load.
        """
        if self.mode == MODE_CPU:
            return None
        if self.is_single:
            return {"": self.device_label}
        return "sequential" if self.required_mb > 0 else "auto"

    def transformers_max_memory(self) -> Optional[dict[int, str]]:
        """`max_memory` for a split: GPU indices only, never "cpu" or "disk".

        accelerate fills the cards in INDEX order whatever order they are given
        in (`get_max_memory` sorts integer keys). On the way it gives two things
        away, both measured against transformers 5.15.1's copy of
        `infer_auto_device_map` (integrations/accelerate.py):
          * room for the model's largest layer, held back on the LOWEST-index
            card (`main_devices = [gpus[0], "cpu"]`) and never returned;
          * the tail of every card but the last, when the next layer does not
            fit whole.
        Both spill onto the next card, and past the last card onto "disk".

        So a card is not capped below its limit to express "most free first".
        That was Phase 2's first plan, and on this node — the 3080 Ti at index
        0 with the least free memory — it capped the 3080 Ti at the remainder
        and planned the 3090 whole, leaving the spill nowhere to go: on the
        Qwen2.5-14B shape with an estimate 5% over its weights, lm_head went to
        disk with 4.3 GB of planned budget unused. An Auto split instead names
        only the cards the plan CHOSE (most free first, as few as hold it) and
        plans their shares in index order (`ShardRule.fill_in_index_order`), so
        every card but the last gets its whole limit and the spill lands on
        memory that exists. Pinned on real map inference by
        tests/unit/ml/test_shard_plan_against_accelerate.py.

        A share below the limit is left only where "all" must divide a model the
        cards below the highest index could hold between them (see plan_shard).
        The highest-index card always keeps its whole limit, so that spill has
        somewhere to go. Whether it is enough depends on the model's layer
        sizes, which a plan in MB cannot see: model_loader.preflight_split runs
        the real map inference before anything is loaded.

        A share is divided by `max_memory_factor` because transformers
        multiplies it back (bitsandbytes' 0.9), so the planned share is what the
        load gets.
        """
        if not self.is_shard or not self.limits:
            return None
        limits = dict(self.limits)
        planned = self.planned_mb_by_index
        last = max(limits)
        budget: dict[int, str] = {}
        for index in sorted(limits):
            mb = limits[index]
            if index != last and self.required_mb > 0:
                mb = min(mb, math.ceil(planned[index] / self.max_memory_factor))
            budget[index] = f"{mb}MiB"
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
            "planned_mb_by_device": {
                f"cuda:{index}": mb for index, mb in sorted(self.shares)
            },
            "budget_mb_by_device": {
                f"cuda:{index}": mb for index, mb in sorted(self.budget_mb_by_index.items())
            },
        }


def cpu_placement(reason: str = REASON_NO_GPU, required_mb: int = 0) -> Placement:
    """No card. Only GGUF may be placed like this (operator decision 3)."""
    return Placement(mode=MODE_CPU, reason=reason, required_mb=max(int(required_mb), 0))


def plan_shard(
    inventory: list[GpuInfo],
    rule: ShardRule,
    reason: str,
    required_mb: int,
    requested: Optional[Union[int, str]] = None,
    every_card: bool = False,
) -> Placement:
    """A split of `rule.need_mb` over `inventory`.

    Auto CHOOSES the cards with the most free memory first and stops as soon as
    their budgets cover the need, so a model two cards hold does not also claim
    a third. Their shares follow the engine's fill order: most free first for
    llama.cpp (the last card taken gets the remainder), index order for
    accelerate (`ShardRule.fill_in_index_order`; the highest-index card chosen
    gets the remainder).

    `every_card` (an explicit "all") uses every card with any budget. A model
    the cards below the highest index could hold between them gets a share on
    each card in proportion to its budget, so it is actually divided rather than
    landing whole on the first card. A larger one, for an engine that fills in
    index order, is planned exactly as Auto plans it — each card but the last
    to its whole budget — because that fill already reaches every card, and a
    proportional cap below a card's budget only gave accelerate's held-back
    layer nowhere to go (review round 2, 2026-09-14).

    An unmeasured model (need 0) gets every card's whole budget. When every card
    together is short, the split still names every usable card with its whole
    budget and the caller decides: a transformers load is refused, and a GGUF
    load is attempted on those cards with every layer offloaded. A card with no
    budget at all is never part of a split.
    """
    # Most free first; on a tie the lower index, matching the single-card choice.
    ordered = sorted(inventory, key=lambda gpu: (-gpu.free_mb, gpu.index))
    limits = {gpu.index: max(int(rule.limit_mb(gpu)), 0) for gpu in ordered}
    budgets = {index: int(limit * rule.max_memory_factor) for index, limit in limits.items()}
    usable = [gpu for gpu in ordered if budgets[gpu.index] > 0]
    need = max(int(rule.need_mb), 0)

    shares: dict[int, int] = {}
    by_index = sorted(gpu.index for gpu in usable)
    if (
        need > 0
        and every_card
        and rule.fill_in_index_order
        and need > sum(budgets[index] for index in by_index[:-1])
    ):
        # "all" for a model the cards below the highest index cannot hold between
        # them: filling in index order already reaches every card, so no card is
        # capped below its budget. Proportional caps here gave accelerate's
        # held-back layer and stranded tails nowhere to go near capacity —
        # measured on transformers 5.15.1's map inference (a 70B-shaped Q4 model,
        # cards of 11 and 20 GB free): lm_head mapped to disk under a plan with
        # 1.3 GB to spare. Review round 2, 2026-09-14.
        remaining = need
        for index in by_index:
            shares[index] = min(budgets[index], remaining)
            remaining -= shares[index]
    elif need > 0 and every_card:
        total = sum(budgets[gpu.index] for gpu in usable)
        shares = {
            gpu.index: min(budgets[gpu.index], math.ceil(need * budgets[gpu.index] / total))
            for gpu in usable
        }
    elif need > 0:
        remaining = need
        for gpu in usable:
            if remaining <= 0:
                break
            take = min(budgets[gpu.index], remaining)
            shares[gpu.index] = take
            remaining -= take
        if remaining > 0:
            shares = {gpu.index: budgets[gpu.index] for gpu in usable}
        elif rule.fill_in_index_order:
            # The same cards, filled lowest index first. Every card but the
            # last chosen gets its whole budget, so accelerate's held-back
            # layer and stranded tails land on real memory, not on disk.
            remaining = need
            by_index: dict[int, int] = {}
            for index in sorted(shares):
                by_index[index] = min(budgets[index], remaining)
                remaining -= by_index[index]
            shares = by_index
    else:
        shares = {gpu.index: budgets[gpu.index] for gpu in usable}

    return Placement(
        mode=MODE_SHARD,
        reason=reason,
        required_mb=max(int(required_mb), 0),
        gpus=tuple(inventory),
        requested=requested,
        shares=tuple(shares.items()),
        limits=tuple((index, limits[index]) for index in shares),
        max_memory_factor=rule.max_memory_factor,
    )


def shard_refusal(placement: Placement, need_mb: int, detail: str) -> InsufficientMemoryError:
    """The refusal for a split that cannot hold `need_mb`, with every card's figures."""
    budgets = placement.budget_mb_by_index
    per_card = ", ".join(f"GPU {index}: {budgets[index]} MB" for index in sorted(budgets))
    return InsufficientMemoryError(
        f"Not enough GPU memory. Need ~{need_mb} MB; split across GPUs this can "
        f"hold {placement.budget_mb} MB ({per_card or 'no card has usable memory'}). "
        + detail,
        details={
            "required_mb": need_mb,
            "available_mb": placement.budget_mb,
            "budget_mb_by_device": {f"cuda:{i}": mb for i, mb in sorted(budgets.items())},
            "requested": placement.requested,
            "gpus": [gpu.to_dict() for gpu in placement.gpus],
        },
    )


def all_cards_refusal(
    placement: Placement, left_out: list[GpuInfo], rule: ShardRule
) -> SplitNotHonouredError:
    """The refusal for "all" when a visible card has no room to take any share of it."""
    cards = ", ".join(
        f"cuda:{gpu.index} ({gpu.name}) has {gpu.free_mb} MB free" for gpu in left_out
    )
    return SplitNotHonouredError(
        f"A split across every GPU was requested, but {cards}: not enough to take any of "
        "the model once the room each card keeps free is set aside. \"all\" is not "
        "narrowed to the other cards. Free memory on that card, or choose Auto or a "
        "named card.",
        details={
            "requested": placement.requested,
            "unused_devices": [f"cuda:{gpu.index}" for gpu in left_out],
            "limit_mb_by_device": {
                f"cuda:{gpu.index}": max(int(rule.limit_mb(gpu)), 0) for gpu in placement.gpus
            },
            "gpus": [gpu.to_dict() for gpu in placement.gpus],
            "placement": placement.to_dict(),
        },
    )


def choose_gpu(
    required_mb: int,
    requested: GpuRequest = None,
    gpus: Optional[list[GpuInfo]] = None,
    shard: Optional[ShardRule] = None,
) -> Placement:
    """Decide where a job needing `required_mb` on one card goes.

    Args:
        required_mb: Memory the job needs on a single card. 0 or less means unknown.
        requested: None / "auto", "all", a CUDA index, or a GPU UUID.
        gpus: The inventory to decide over; read live when omitted.
        shard: How to size a split. Defaults to a transformers split of
            `required_mb`.

    Returns:
        A single-card Placement when one card fits (or the requested card
        fits), otherwise a split. Whether a split holds the model is for the
        caller to judge (`budget_mb` against `shard.need_mb`): the engines
        disagree about what to do when it does not.

    Raises:
        GpuNotFoundError: the requested card is not visible.
        InsufficientMemoryError: no GPU is visible, the requested card lacks the
            free memory, or "all" was requested and every card together cannot
            hold the model. An explicit choice is never swapped for another.
    """
    inventory = list_gpus() if gpus is None else list(gpus)
    wanted = parse_gpu_request(requested)
    required = max(int(required_mb or 0), 0)
    listing = [gpu.to_dict() for gpu in inventory]
    rule = shard if shard is not None else transformers_shard_rule(required)

    if not inventory:
        raise InsufficientMemoryError(
            "No GPU is visible to miLLM.",
            details={"required_mb": required, "available_mb": 0, "gpus": []},
        )

    if wanted == ALL:
        placement = plan_shard(
            inventory, rule, REASON_REQUESTED_ALL, required, requested=ALL, every_card=True
        )
        need = max(int(rule.need_mb), 0)
        if not placement.gpu_indices or (need > 0 and placement.budget_mb < need):
            raise shard_refusal(
                placement,
                need,
                "A split across every GPU was requested; it is not swapped for "
                "one card or for the CPU.",
            )
        left_out = [gpu for gpu in inventory if gpu.index not in placement.gpu_indices]
        if left_out:
            # plan_shard names only the cards with a budget, so a card too full
            # to take any share (the 3080 Ti with a miStudio job on it) was left
            # out, and "all" became a split over the rest — one card, on this
            # node — that every later check accepted, since they look for an
            # unused card among the planned ones. Review round 4, 2026-09-14.
            raise all_cards_refusal(placement, left_out, rule)
        return placement

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
        # An unknown size cannot be shown to fit any card, and there is nothing
        # to plan a split with. Every card, GPU memory only, balanced by
        # accelerate — the spread every load had before placement existed, less
        # the CPU — and the load checks afterwards where it actually landed.
        return plan_shard(inventory, rule, REASON_SIZE_UNKNOWN, required)

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

    return plan_shard(inventory, rule, REASON_NO_SINGLE_CARD, required)


def reported_free_mb_by_index(indices: list[int]) -> dict[int, int]:
    """Free memory per named card as nvidia-smi reports it — no CUDA context.

    For measuring memory that torch does not own (llama.cpp's), where creating a
    torch context on the card would itself cost the memory being measured.
    """
    wanted = set(indices)
    if not wanted:
        return {}
    return {gpu.index: gpu.free_mb for gpu in list_gpus() if gpu.index in wanted}


def free_mb_by_index(indices: list[int]) -> dict[int, int]:
    """Live free memory for each named card, from torch. A card that cannot be read is omitted.

    ONLY for cards a transformers model is on or is being placed on: each call
    creates a CUDA context on its card, which is harmless where the model will
    create one anyway and a loss of memory everywhere else.
    """
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


#: Where the input embedding table sits in `hf_device_map`, for models whose
#: `get_input_embeddings()` cannot answer: the flat layouts by full name...
_EMBEDDING_MAP_KEYS = (
    "",
    "model.embed_tokens",
    "transformer.wte",
    "model.embedding",
    "model.shared",
    "model.embed",
)
#: ...and any nesting by the embedding's own name. Multimodal checkpoints put
#: the text stack a level deeper (gemma-4's `model.language_model.embed_tokens`,
#: after a vision tower in the map), and the flat names alone sent their inputs
#: to whichever card the map listed first. Only names that are unambiguously an
#: input embedding: "shared" or "embed" as a leaf can be an MoE expert or a
#: projection.
_EMBEDDING_LEAF_NAMES = ("embed_tokens", "wte", "word_embeddings", "embed_in")


def model_input_device(model: Any) -> Optional[str]:
    """The device `input_ids` must go to: where the input embedding table lives.

    Asked of the model first (`get_input_embeddings()`), which knows its own
    nesting; then of `hf_device_map` by known names, then by any key ending in an
    embedding's name; then the first card in the map. None when nothing answers.

    A split model's `model.device` is not this: it names the first parameter's
    device, and under accelerate dispatch that need not be the embedding's.
    """
    try:
        embeddings = model.get_input_embeddings()
        label = _device_label(getattr(getattr(embeddings, "weight", None), "device", None))
        if label is not None and label not in ("disk", "meta"):
            return label
    except Exception:  # noqa: BLE001 - a stub or a model without the accessor
        pass

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        for key in _EMBEDDING_MAP_KEYS:
            if key in device_map:
                label = _device_label(device_map[key])
                if label is not None and label not in ("disk", "meta"):
                    return label
        for key, value in device_map.items():
            if str(key).rsplit(".", 1)[-1] in _EMBEDDING_LEAF_NAMES:
                label = _device_label(value)
                if label is not None and label not in ("disk", "meta"):
                    return label
        labels = [_device_label(value) for value in device_map.values()]
        for label in labels:
            if label is not None and label.startswith("cuda:"):
                return label
        return "cpu"

    try:
        return _device_label(next(model.parameters()).device)
    except Exception:  # noqa: BLE001
        return None


def layer_share_by_index(model: Any, indices: list[int]) -> dict[int, float]:
    """The fraction of the model's layers on each card, from `hf_device_map`.

    A KV cache is allocated beside the attention layers, so a card holding a
    quarter of the layers holds about a quarter of the cache. A layer is an
    entry whose last name is a number (`model.layers.12`) — accelerate lists a
    split stack layer by layer. A card with no layers holds no cache. When the
    map names no layers at all, the cards share it evenly.
    """
    wanted = sorted(set(indices))
    if not wanted:
        return {}
    if len(wanted) == 1:
        return {wanted[0]: 1.0}
    counts = {index: 0 for index in wanted}
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for key, value in device_map.items():
            if not str(key).rsplit(".", 1)[-1].isdigit():
                continue
            label = _device_label(value)
            if label is not None and label.startswith("cuda:") and int(label[5:]) in counts:
                counts[int(label[5:])] += 1
    total = sum(counts.values())
    if total == 0:
        return {index: 1 / len(wanted) for index in wanted}
    return {index: count / total for index, count in counts.items()}
