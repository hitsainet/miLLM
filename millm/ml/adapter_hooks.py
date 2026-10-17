"""LoRA adapter forward hooks (Feature 33), with no `peft` dependency.

An adapter is served by adding `scaling · B·(A·x)` to the output of each targeted `nn.Linear`
through a forward hook. The base weights are never written, never re-registered and never replaced,
so removing the hooks restores the base model exactly (`torch.equal`, not `allclose`).

Five properties are load-bearing, and each has a test and a mutation control in
`tests/unit/ml/test_adapter_hooks.py`.

**1. The low-rank path runs in float32 (addendum D-12).** A and B are kept in float32 and the
input is cast to float32 on A's device before the two products, the delta is added to the output in
float32, and only that sum is cast back to the output's dtype (one rounding, as PEFT does). That is
PEFT's default (`autocast_adapter_dtype=True`), so an adapter scored by miForge or `peft` and the
same adapter served here agree bit for bit (pinned against a committed PEFT golden fixture).
Computing the delta, or the sum, in the module's bfloat16 instead is a quiet drift: the answers
stay plausible and differ.

**2. Install is all or nothing.** A partially installed adapter is a model nobody trained. Any
failure while registering removes every hook already registered and raises `AdapterInstallError`.

**3. One rank (addendum D-10).** The manifest carries one `rank` and one `alpha`, so one scale
`alpha / rank`. A module whose tensors have another rank would be served at a scale nobody recorded,
so it is refused (reason `rank`) rather than accepted.

**4. Targets match on a `.` boundary, never by regex (addendum D-17).** PEFT records target modules
as suffixes (`["q_proj", "v_proj"]`). Every planned module path must end in some entry on a `.`
boundary (a full path matches itself), and every entry must match at least one planned path. An
entry is a literal string: `".*q_proj"` matches nothing and is refused.

**5. Keys go through the model's own transformers renaming.** transformers 5 renames and fuses some
checkpoint keys at load time (`transformers/conversion_mapping.py`): MoE experts become one 3-D
`experts.gate_up_proj` parameter, multimodal checkpoints move under `model.language_model`. An
adapter key is mapped with the same `WeightRenaming`s, and a key that would be consumed by a
`WeightConverter` (a fused target) is refused with reason `fused_by_transformers`, because no
`nn.Linear` with that name exists in the live model. If that API is unavailable the keys are used
unrenamed — which is safe, because a module that was renamed then fails the existence check
(`missing_module`) instead of being matched to the wrong place. As in transformers' own loader, a
renamed key whose new name does not exist while its original name does is kept as it was: the key
already names the live module (an adapter trained on this transformers version).

Errors carry a `reason` and, for planning refusals, every offender. They are internal: the adapter
service maps them to the public error codes (`ADAPTER_SHAPE_MISMATCH`, `ADAPTER_INVALID`,
`ADAPTER_INSTALL_FAILED`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from millm.ml.layer_resolution import reset_dynamo_for_hook_change

logger = logging.getLogger(__name__)

#: PEFT's key prefix: `base_model.model.<module path>.lora_A.weight`.
PEFT_PREFIX = "base_model.model."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"

#: The only scaling rule schema v1 defines (`adapter.scaling`, default when absent).
SCALING_ALPHA_OVER_R = "alpha_over_r"

#: Base weight dtypes a hook may be installed over. Quantized linears (bitsandbytes `Linear4bit`,
#: `Linear8bitLt`) subclass `nn.Linear` but carry integer weights, so they fail this check.
SUPPORTED_WEIGHT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

Reason = Literal[
    "empty",
    "unsupported_tensor",
    "incomplete_pair",
    "duplicate_target",
    "missing_module",
    "unsupported_module",
    "fused_by_transformers",
    "rank",
    "shape",
    "dtype",
    "non_finite",
    "invalid_target_entry",
    "target_not_declared",
    "target_unused",
    "scaling",
    "install_failed",
]


class AdapterHookError(Exception):
    """Base of every refusal this module raises. `reason` is a stable machine-readable token."""

    def __init__(self, reason: str, message: str, *, offenders: Sequence[dict] = ()) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.offenders: tuple[dict, ...] = tuple(offenders)


class AdapterPlanError(AdapterHookError):
    """The adapter does not fit the resident model. Raised before any hook exists.

    `offenders` lists EVERY problem found (each a dict with `reason`, and `key` and/or `module`
    and `detail`), so a caller can report all of them at once; `reason` is the first offender's.
    """


class AdapterInstallError(AdapterHookError):
    """Registering the hooks failed part way. Every hook registered before the failure has been
    removed again, so the model is exactly as it was."""

    def __init__(self, message: str, *, installed_before_failure: int, target: str) -> None:
        super().__init__(
            "install_failed",
            message,
            offenders=({"reason": "install_failed", "module": target},),
        )
        self.installed_before_failure = installed_before_failure
        self.target = target


@dataclass(frozen=True)
class AdapterTarget:
    """One targeted linear module and the two tensor keys that drive it."""

    module_path: str  # as the LIVE model names it: "model.layers.3.self_attn.q_proj"
    a_key: str  # safetensors key of lora_A, (rank, in_features)
    b_key: str  # safetensors key of lora_B, (out_features, rank)
    rank: int
    in_features: int
    out_features: int


@dataclass
class PreparedAdapter:
    """Tensors placed and ready to hook. Holds the only references to A and B."""

    targets: tuple[AdapterTarget, ...]
    scaling: float
    #: module_path -> (A fp32, B fp32), each on that module's weight device.
    weights: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)
    #: module_path -> the module object resolved at build time. `install` refuses if the model
    #: no longer resolves the path to this exact object (the base changed in between).
    modules: dict[str, nn.Module] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------


def lora_scaling(alpha: float, rank: int, rule: Optional[str] = None) -> float:
    """The factor applied to `B·A·x`. Schema v1 defines only `alpha_over_r`, and `scaling` may be
    absent (it has a default the validator does not inject), so `None` means `alpha_over_r`.
    Any other rule is refused: the schema requires a reader to refuse rather than guess."""
    effective = SCALING_ALPHA_OVER_R if rule is None else rule
    if effective != SCALING_ALPHA_OVER_R:
        raise AdapterPlanError(
            "scaling",
            f"adapter scaling rule {effective!r} is not supported (only {SCALING_ALPHA_OVER_R!r})",
            offenders=({"reason": "scaling", "detail": str(effective)},),
        )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise AdapterPlanError("rank", f"adapter rank must be a positive integer, got {rank!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise AdapterPlanError("scaling", f"adapter alpha must be a finite number, got {alpha!r}")
    if alpha <= 0:
        raise AdapterPlanError("scaling", f"adapter alpha must be positive, got {alpha!r}")
    return float(alpha) / float(rank)


def parse_peft_key(key: str) -> Optional[tuple[str, Literal["A", "B"]]]:
    """`base_model.model.<module>.lora_A.weight` -> (`<module>`, "A"). Anything else -> None.

    The only format-specific function. A key that does not parse is refused by the planner
    (`unsupported_tensor`), never skipped: an ignored tensor is a partial adapter.
    """
    if not key.startswith(PEFT_PREFIX):
        return None
    rest = key[len(PEFT_PREFIX):]
    for suffix, which in ((LORA_A_SUFFIX, "A"), (LORA_B_SUFFIX, "B")):
        if rest.endswith(suffix):
            module = rest[: -len(suffix)]
            if not module or module.startswith(".") or module.endswith(".") or ".." in module:
                return None
            return module, which  # type: ignore[return-value]
    return None


def target_entry_matches(path: str, entry: str) -> bool:
    """True when `path` is `entry` or ends in `"." + entry`. Literal comparison, never a regex."""
    return path == entry or path.endswith("." + entry)


def check_target_modules(
    paths: Mapping[str, Iterable[str]], entries: Sequence[str]
) -> list[dict]:
    """Offenders for addendum D-17. `paths` maps each planned module path to every name it is
    known by (the key's own path and, if different, the live model's renamed path); a path is
    covered when any of its names matches any entry."""
    offenders: list[dict] = []
    valid: list[str] = []
    for entry in entries:
        if (
            not isinstance(entry, str)
            or not entry
            or entry.startswith(".")
            or entry.endswith(".")
            or ".." in entry
        ):
            offenders.append(
                {"reason": "invalid_target_entry", "detail": repr(entry)}
            )
        else:
            valid.append(entry)
    for path, names in paths.items():
        names = tuple(names)
        if not any(target_entry_matches(n, e) for n in names for e in valid):
            offenders.append(
                {
                    "reason": "target_not_declared",
                    "module": path,
                    "detail": "the weights file targets a module no target_modules entry names",
                }
            )
    for entry in valid:
        if not any(target_entry_matches(n, entry) for names in paths.values() for n in names):
            offenders.append(
                {
                    "reason": "target_unused",
                    "detail": f"target_modules entry {entry!r} matches no module in the weights file",
                }
            )
    return offenders


# ---------------------------------------------------------------------------------------------
# transformers renaming
# ---------------------------------------------------------------------------------------------


def _conversion_renamer(model: nn.Module) -> Optional[Callable[[str], tuple[str, Optional[str]]]]:
    """A function mapping a checkpoint key to (live key, converter pattern or None), built from
    the model's own transformers conversion mapping, or None when the API is unavailable."""
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import (
            WeightConverter,
            WeightRenaming,
            rename_source_key,
        )
    except ImportError as e:  # pragma: no cover - pinned transformers has it; tested by import
        logger.warning("adapter_key_renaming_unavailable: %s", e)
        return None
    try:
        transforms = get_model_conversion_mapping(model)
    except Exception as e:  # a model transformers cannot describe: use keys unrenamed
        logger.warning("adapter_key_renaming_unavailable: %s", e)
        return None
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]

    def rename(key: str) -> tuple[str, Optional[str]]:
        return rename_source_key(key, renamings, converters)

    return rename


def live_module_path(
    path: str, renamer: Optional[Callable[[str], tuple[str, Optional[str]]]]
) -> tuple[Optional[str], Optional[str]]:
    """(live path, None) or (None, why it cannot be mapped). The module's weight key
    `<path>.weight` is what transformers' patterns are written against, so that is what is
    renamed; the live module path is the renamed key without `.weight`."""
    if renamer is None:
        return path, None
    renamed, converter = renamer(path + ".weight")
    if converter is not None:
        return None, f"transformers fuses this module's weight at load ({converter!r} -> {renamed!r})"
    if not renamed.endswith(".weight"):
        return None, f"transformers renames this module's weight to {renamed!r}"
    return renamed[: -len(".weight")], None


# ---------------------------------------------------------------------------------------------
# Planning (shapes only: nothing is materialised)
# ---------------------------------------------------------------------------------------------


def _is_supported_linear(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return (
        isinstance(module, nn.Linear)
        and isinstance(weight, Tensor)
        and weight.dtype in SUPPORTED_WEIGHT_DTYPES
        and weight.dim() == 2
    )


def _has_module(model: nn.Module, path: str) -> bool:
    try:
        model.get_submodule(path)
    except AttributeError:
        return False
    return True


def plan_targets(
    model: nn.Module,
    shapes: Mapping[str, Sequence[int]],
    *,
    rank: int,
    target_modules: Sequence[str],
) -> tuple[AdapterTarget, ...]:
    """Map a weights file's keys (with their shapes) onto the resident model.

    Reads shapes only, so it runs before any tensor is materialised. Collects EVERY offender and
    raises one `AdapterPlanError` naming the first in its message and all in `offenders`.
    """
    offenders: list[dict] = []
    if not shapes:
        raise AdapterPlanError(
            "empty", "the adapter weights file contains no tensors",
            offenders=({"reason": "empty"},),
        )

    renamer = _conversion_renamer(model)
    # key path -> {"A": key, "B": key}
    pairs: dict[str, dict[str, str]] = {}
    for key in shapes:
        parsed = parse_peft_key(key)
        if parsed is None:
            offenders.append(
                {
                    "reason": "unsupported_tensor",
                    "key": key,
                    "detail": f"not a PEFT LoRA key ({PEFT_PREFIX}<module>{LORA_A_SUFFIX} or "
                    f"{LORA_B_SUFFIX})",
                }
            )
            continue
        path, which = parsed
        pairs.setdefault(path, {})[which] = key

    names_for_matching: dict[str, tuple[str, ...]] = {}
    live_for: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for path in pairs:
        live, why = live_module_path(path, renamer)
        if live is not None and live != path and not _has_module(model, live) and _has_module(
            model, path
        ):
            # transformers' own loader rule (`core_model_loading`: "renamed_key not in the model
            # and original_key in the model" -> keep the original): a key that already names a
            # live module — an adapter trained on this transformers — is not renamed away from it.
            live = path
        if live is None:
            offenders.append(
                {"reason": "fused_by_transformers", "module": path, "detail": why}
            )
            names_for_matching[path] = (path,)
            continue
        names_for_matching[path] = (path,) if live == path else (path, live)
        if live in claimed:
            offenders.append(
                {
                    "reason": "duplicate_target",
                    "module": live,
                    "detail": f"keys for {claimed[live]!r} and {path!r} both map to this module",
                }
            )
            continue
        claimed[live] = path
        live_for[path] = live

    offenders.extend(check_target_modules(names_for_matching, target_modules))

    targets: list[AdapterTarget] = []
    for path, pair in pairs.items():
        live = live_for.get(path)
        if live is None:
            continue
        if "A" not in pair or "B" not in pair:
            missing = "lora_A" if "A" not in pair else "lora_B"
            offenders.append(
                {"reason": "incomplete_pair", "module": live, "detail": f"{missing} is missing"}
            )
            continue
        try:
            module = model.get_submodule(live)
        except AttributeError:
            offenders.append(
                {"reason": "missing_module", "module": live, "detail": "no such module in the model"}
            )
            continue
        if not _is_supported_linear(module):
            weight = getattr(module, "weight", None)
            dtype = getattr(weight, "dtype", None)
            offenders.append(
                {
                    "reason": "unsupported_module",
                    "module": live,
                    "detail": f"{type(module).__name__} (weight dtype {dtype}) is not a "
                    "floating-point nn.Linear",
                }
            )
            continue
        a_shape = tuple(int(d) for d in shapes[pair["A"]])
        b_shape = tuple(int(d) for d in shapes[pair["B"]])
        in_f, out_f = int(module.in_features), int(module.out_features)
        if len(a_shape) != 2 or len(b_shape) != 2:
            offenders.append(
                {
                    "reason": "shape",
                    "module": live,
                    "detail": f"lora_A {a_shape} and lora_B {b_shape} must both be 2-D",
                }
            )
            continue
        if a_shape[0] != rank or b_shape[1] != rank:
            offenders.append(
                {
                    "reason": "rank",
                    "module": live,
                    "detail": f"lora_A {a_shape} / lora_B {b_shape} have rank "
                    f"{a_shape[0]}/{b_shape[1]}; the manifest's adapter.rank is {rank}",
                }
            )
            continue
        if a_shape[1] != in_f or b_shape[0] != out_f:
            offenders.append(
                {
                    "reason": "shape",
                    "module": live,
                    "detail": f"expected lora_A ({rank}, {in_f}) and lora_B ({out_f}, {rank}), "
                    f"got {a_shape} and {b_shape}",
                }
            )
            continue
        targets.append(
            AdapterTarget(
                module_path=live,
                a_key=pair["A"],
                b_key=pair["B"],
                rank=rank,
                in_features=in_f,
                out_features=out_f,
            )
        )

    if offenders:
        first = offenders[0]
        where = first.get("module") or first.get("key") or ""
        raise AdapterPlanError(
            first["reason"],
            f"adapter refused ({first['reason']}){': ' + where if where else ''}"
            f"{' — ' + first['detail'] if first.get('detail') else ''}"
            f"{f' (+{len(offenders) - 1} more)' if len(offenders) > 1 else ''}",
            offenders=offenders,
        )
    targets.sort(key=lambda t: t.module_path)
    return tuple(targets)


# ---------------------------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------------------------


def _make_hook(a: Tensor, b: Tensor, scaling: float) -> Callable[..., Tensor]:
    """`output + scaling · B·(A·x)`: the delta in float32, ADDED in float32, rounded once.

    The sum is formed in float32 and only the finished output is cast back to the module dtype,
    exactly as PEFT's `lora.Linear.forward` does (`result + lora_B(lora_A(x)) * scaling` promotes a
    bfloat16 `result` to float32, then `result.to(torch_result_dtype)`). Rounding the delta to
    bfloat16 first and adding in bfloat16 rounds twice: on the committed PEFT golden fixture that
    moved logits by 0.03 against an adapter effect of 0.65, while this form is bit-identical.
    """

    def hook(_module: nn.Module, args: tuple, kwargs: dict, output: Tensor) -> Tensor:
        x = args[0] if args else kwargs["input"]
        delta = F.linear(F.linear(x.to(a.device, torch.float32), a), b) * scaling
        return (output + delta.to(output.device)).to(output.dtype)

    return hook


class LoraHookSet:
    """Build, install and remove the hooks for one adapter."""

    @staticmethod
    def build(
        model: nn.Module,
        targets: Sequence[AdapterTarget],
        tensors: Mapping[str, Tensor],
        scaling: float,
    ) -> PreparedAdapter:
        """Place A and B in float32 on each module's weight device. Touches no module."""
        if not math.isfinite(scaling):
            raise AdapterPlanError("scaling", f"adapter scaling must be finite, got {scaling!r}")
        prepared = PreparedAdapter(targets=tuple(targets), scaling=float(scaling))
        for t in targets:
            module = model.get_submodule(t.module_path)
            device = module.weight.device
            placed = []
            for key, expected in ((t.a_key, (t.rank, t.in_features)), (t.b_key, (t.out_features, t.rank))):
                tensor = tensors.get(key)
                if tensor is None:
                    raise AdapterPlanError(
                        "incomplete_pair", f"tensor {key!r} is missing",
                        offenders=({"reason": "incomplete_pair", "key": key},),
                    )
                if tuple(tensor.shape) != expected:
                    raise AdapterPlanError(
                        "shape", f"tensor {key!r} has shape {tuple(tensor.shape)}, expected {expected}",
                        offenders=({"reason": "shape", "key": key},),
                    )
                if not tensor.is_floating_point():
                    raise AdapterPlanError(
                        "dtype", f"tensor {key!r} has non-floating dtype {tensor.dtype}",
                        offenders=({"reason": "dtype", "key": key},),
                    )
                placed_tensor = tensor.detach().to(device=device, dtype=torch.float32, copy=True)
                if not bool(torch.isfinite(placed_tensor).all()):
                    raise AdapterPlanError(
                        "non_finite", f"tensor {key!r} contains NaN or infinity",
                        offenders=({"reason": "non_finite", "key": key},),
                    )
                placed.append(placed_tensor.contiguous())
            prepared.weights[t.module_path] = (placed[0], placed[1])
            prepared.modules[t.module_path] = module
        return prepared

    @staticmethod
    def install(model: nn.Module, prepared: PreparedAdapter) -> list[RemovableHandle]:
        """Register every hook, or none. On failure, removes what was registered and raises
        `AdapterInstallError`."""
        handles: list[RemovableHandle] = []
        current = ""
        try:
            for t in prepared.targets:
                current = t.module_path
                module = model.get_submodule(t.module_path)
                if module is not prepared.modules.get(t.module_path):
                    raise RuntimeError(
                        f"{t.module_path} is no longer the module the adapter was built for"
                    )
                a, b = prepared.weights[t.module_path]
                handles.append(
                    module.register_forward_hook(
                        _make_hook(a, b, prepared.scaling), with_kwargs=True
                    )
                )
        except BaseException as exc:
            installed = len(handles)
            _remove_all(handles)
            if installed:
                reset_dynamo_for_hook_change()
            logger.error(
                "adapter_install_rolled_back installed_before_failure=%d target=%s cause=%s",
                installed, current, exc,
            )
            if isinstance(exc, Exception):
                raise AdapterInstallError(
                    f"installing the adapter hook on {current} failed: {exc}",
                    installed_before_failure=installed,
                    target=current,
                ) from exc
            raise
        reset_dynamo_for_hook_change()
        return handles

    @staticmethod
    def remove(handles: list[RemovableHandle]) -> None:
        """Remove every hook, then reset Dynamo. Never stops part way: each handle is attempted.

        A caller holding a lock should call `detach` under it and `reset_after_change` after
        releasing it (`AdapterState.take`): the reset can take long, and a lock held across it
        stalls every reader of that lock (033 review R1-A finding 7)."""
        LoraHookSet.detach(handles)
        LoraHookSet.reset_after_change()

    @staticmethod
    def detach(handles: list[RemovableHandle]) -> None:
        """Remove every hook WITHOUT resetting Dynamo. The caller owes `reset_after_change()`
        before the next forward that could run a compiled graph."""
        _remove_all(handles)

    @staticmethod
    def reset_after_change() -> None:
        """Reset TorchDynamo after a hook change, so compiled graphs re-trace. Never raises."""
        reset_dynamo_for_hook_change()


def _remove_all(handles: list[RemovableHandle]) -> None:
    errors: list[BaseException] = []
    for handle in handles:
        try:
            handle.remove()
        except Exception as e:  # pragma: no cover - RemovableHandle.remove does not raise
            errors.append(e)
    handles.clear()
    if errors:  # pragma: no cover
        logger.error("adapter_hook_remove_errors count=%d first=%s", len(errors), errors[0])


def adapter_shapes(tensors: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Shapes of an in-memory tensor mapping, for `plan_targets` in tests and callers that
    already hold tensors."""
    return {k: tuple(int(d) for d in v.shape) for k, v in tensors.items()}
