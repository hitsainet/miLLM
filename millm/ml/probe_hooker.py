"""The read-only forward hook a probe reads its layer through.

Three properties make this different from `SAEHooker`, and each is load-bearing.

**1. It is PREPENDED.** `register_forward_hook(fn, prepend=True)` puts this hook ahead of any SAE
steering hook on the same module, so a probe reads the **pre-steering** residual (BR-002, FR-24.5).
Without it, steering would change what the probe sees — and a monitor whose reading can be moved by
the thing it is monitoring is worse than no monitor. SAE hooks are registered without `prepend`
(`sae_hooker.py`), so prepending is sufficient to win the ordering.

⚠ `prepend=True` appears nowhere else in this repository. There is no existing shape to copy and no
existing test that would notice if it were dropped, so the mutation control in
`tests/unit/ml/test_probe_hooker.py` — remove `prepend=True`, watch a steered read go red — is the
only thing holding it. That test must steer the **same layer** the probe reads, or it passes for
the wrong reason.

**2. It never modifies the output.** The callback is handed the hidden states and its return value
is discarded; this hook returns `None`, which is how PyTorch is told "unchanged". A forward hook
that returns a value *replaces* the module's output, so an accidental `return output` here would
silently substitute whatever the probe last touched.

**3. One hook per layer, shared by every probe on it.** Installing one hook per armed probe would
multiply the per-pass cost by the number of probes and, worse, multiply the device-to-host copies.
The budget is one D2H per forward pass (FR-24.6); the callback receives the tensor once and every
probe on that layer reads from the same copy.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from millm.ml.layer_resolution import get_layer, reset_dynamo_for_hook_change

logger = logging.getLogger(__name__)

#: What the hook hands back: the layer's hidden states, exactly as the module produced them.
ProbeCallback = Callable[[Tensor], None]


def extract_hidden_states(output: Any) -> Optional[Tensor]:
    """The hidden-states tensor from whatever a decoder layer returned.

    HF decoder layers are inconsistent: most return `(hidden_states, ...)` with attention weights
    and cache entries trailing, some return the tensor bare, and a few return a dataclass-like
    object with `.last_hidden_state`. Returning `None` for a shape this does not recognise is
    deliberate — the caller marks the request unscored with a reason rather than guessing which
    element is the residual, because reading the wrong element is silent and plausible.
    """
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, Tensor):
            return first
        return None
    last = getattr(output, "last_hidden_state", None)
    if isinstance(last, Tensor):
        return last
    return None


class ProbeHooker:
    """Installs and removes the per-layer read-only hook."""

    def __init__(self) -> None:
        self.last_resolved_module_path: Optional[str] = None

    def install(self, model: nn.Module, layer: int, callback: ProbeCallback) -> RemovableHandle:
        """Install a read-only, prepended forward hook on `layer`.

        `callback` receives the layer's hidden states on every forward pass. Its return value is
        ignored and this hook returns `None`, so the model's output is untouched.
        """
        target = get_layer(model, layer)
        self.last_resolved_module_path = self._module_path(model, target)

        def hook_fn(_module: nn.Module, _inputs: Any, output: Any) -> None:
            # ⚠ Returns None ALWAYS. Returning `output` would work by accident today and replace
            # the module's output the moment anything upstream changed.
            try:
                hidden = extract_hidden_states(output)
                if hidden is None:
                    logger.warning(
                        "probe_hook_unrecognised_output layer=%s type=%s", layer, type(output).__name__
                    )
                    return None
                callback(hidden)
            except Exception:
                # A probe must never break generation. The callback records its own failure; here
                # we only guarantee the forward pass survives it.
                logger.exception("probe_hook_callback_failed layer=%s", layer)
            return None

        handle = target.register_forward_hook(hook_fn, prepend=True)
        reset_dynamo_for_hook_change()
        logger.info(
            "probe_hook_installed layer=%s module_path=%s prepend=True",
            layer,
            self.last_resolved_module_path,
        )
        return handle

    def remove(self, handle: RemovableHandle) -> None:
        """Remove a hook and let compiled graphs re-trace without it."""
        handle.remove()
        reset_dynamo_for_hook_change()
        logger.info("probe_hook_removed")

    @staticmethod
    def _module_path(model: nn.Module, target: nn.Module) -> Optional[str]:
        for name, module in model.named_modules():
            if module is target:
                return name
        return None


def is_single_row(hidden: Tensor) -> bool:
    """Whether this pass carries exactly one sequence.

    A probe scores one request. A batched pass (`n > 1`, `extra_messages`, or continuous batching)
    interleaves rows the request-scoped context cannot attribute, so the caller marks the request
    `not_scored` with a reason rather than scoring row 0 and labelling it as the request's verdict.
    """
    return hidden.dim() == 3 and hidden.shape[0] == 1
