"""Finding a decoder layer in an arbitrary transformer, and which device it sits on.

Lifted verbatim out of `SAEHooker` (Feature 024 task 1.3). The behaviour is unchanged — the same
accessor patterns in the same order, the same fallbacks, the same error messages — so the existing
hooker tests pass against it without modification. `SAEHooker` now delegates here.

⚠ IT LIVES IN ITS OWN MODULE RATHER THAN IN `sae_hooker`, and that is an architectural statement
rather than tidiness. BR-002 says probes do not depend on SAE attachment; a probe hooker that had
to `from millm.ml.sae_hooker import ...` would make that claim false at the import level even while
it held at runtime, and the next person to read the imports would reasonably conclude a probe needs
an SAE. Resolving a layer index to a module has nothing to do with SAEs.

`reset_dynamo_for_hook_change` is here for the same reason. The FTID said to import it from
`SAEService` rather than duplicate it — but `SAEService` pulls in `AttachedSAEState`, the singleton
that task 4A.4 spies on to prove a probe never touches it, so importing it into probe code would
couple exactly what the design is trying to keep apart. Lifting it is the same move as the layer
helpers, and `SAEService._reset_dynamo_for_hook_change` now delegates here.
"""

from __future__ import annotations

import itertools
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def get_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """The decoder layer module at `layer_idx`.

    Supports several transformer layouts:
      - Gemma / Llama / Mistral:  model.model.layers[i]
      - GPT-2 / GPT-Neo:          model.transformer.h[i]
      - Some HF decoders:         model.model.decoder.layers[i]
      - Generic:                  model.layers[i], model.encoder.layer[i], model.decoder.layer[i]

    Raises ValueError when no pattern matches, naming the patterns tried — a hook installed on the
    wrong module is silent, so failing loudly here is the point.
    """
    layer_access_patterns = [
        lambda m: m.model.layers[layer_idx],
        lambda m: m.transformer.h[layer_idx],
        lambda m: m.model.decoder.layers[layer_idx],
        lambda m: m.layers[layer_idx],
        lambda m: m.encoder.layer[layer_idx],
        lambda m: m.decoder.layer[layer_idx],
    ]

    for accessor in layer_access_patterns:
        try:
            layer = accessor(model)
            logger.debug(f"Found layer {layer_idx} using accessor pattern")
            return layer
        except (AttributeError, IndexError, TypeError, KeyError):
            continue

    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > layer_idx:
            if "layer" in name.lower() or "block" in name.lower() or name == "h":
                logger.debug(f"Found layer via ModuleList search: {name}[{layer_idx}]")
                return module[layer_idx]

    raise ValueError(
        f"Could not find layer {layer_idx}. "
        f"Model architecture may not be supported. "
        f"Supported patterns: Llama/Gemma (model.model.layers), "
        f"GPT-2 (transformer.h), generic (layers). "
        f"Check model.named_modules() for layer structure."
    )


def layer_device(model: nn.Module, layer: int) -> torch.device:
    """The device holding `layer`'s weights.

    A model spread across GPUs (device_map="auto") keeps later layers on another card. Anything
    hooked there must live on that card, or every forward pass mixes devices.
    """
    target_layer = get_layer(model, layer)
    for tensor in itertools.chain(target_layer.parameters(), target_layer.buffers()):
        return tensor.device
    for tensor in model.parameters():
        return tensor.device
    return torch.device("cpu")


def get_layer_count(model: nn.Module) -> int:
    """Total number of decoder layers.

    The config is tried first because it is the only source that is right by construction; the
    structural fallbacks exist for models whose config omits the attribute.
    """
    if hasattr(model, "config"):
        config = model.config
        for attr in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
            if hasattr(config, attr):
                return getattr(config, attr)

    layer_access_patterns = [
        lambda m: len(m.model.layers),
        lambda m: len(m.transformer.h),
        lambda m: len(m.layers),
        lambda m: len(m.encoder.layer),
    ]

    for accessor in layer_access_patterns:
        try:
            count = accessor(model)
            if isinstance(count, int) and count > 0:
                return count
        except (AttributeError, TypeError):
            continue

    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first_child = list(module.children())[0] if len(list(module.children())) > 0 else None
            if first_child is not None and hasattr(first_child, "self_attn"):
                return len(module)

    raise ValueError(
        "Could not determine layer count. "
        "Model config should have num_hidden_layers or similar attribute."
    )


def reset_dynamo_for_hook_change() -> None:
    """Reset TorchDynamo so compiled graphs re-trace after a hook change.

    A hook added or removed after `torch.compile` may not be reflected in a cached graph, so the
    next forward pass must re-trace. Best-effort: never raises, because failing to clear a cache is
    not a reason to refuse to install a hook.
    """
    try:
        import torch._dynamo as _dynamo

        _dynamo.reset()
        logger.debug("dynamo_reset_after_hook_change")
    except Exception as e:
        logger.debug("dynamo_reset_skipped: %s", e)
