"""
Model hook management for SAE attachment.

Implements direct residual stream steering (miStudio/Neuronpedia compatible).

Steering Formula:
    modified_activations = original_activations + Σ(strength_i × decoder_direction_i)

The hook applies steering uniformly to all token positions without full SAE reconstruction.
"""

import itertools
import logging
from typing import Callable, Tuple, Union

import torch
from torch import nn, Tensor
from torch.utils.hooks import RemovableHandle

from millm.ml.layer_resolution import get_layer as _resolve_layer
from millm.ml.layer_resolution import get_layer_count as _resolve_layer_count
from millm.ml.layer_resolution import layer_device as _resolve_layer_device
from millm.ml.sae_wrapper import LoadedSAE

logger = logging.getLogger(__name__)


class SAEHooker:
    """
    Manages PyTorch forward hooks for SAE attachment.

    Implements direct residual stream steering (miStudio/Neuronpedia compatible):
    - Steering is applied by adding decoder directions to hidden states
    - Applied uniformly to ALL token positions
    - No full SAE encode/decode for steering (lightweight)
    - Optional monitoring via SAE encoding

    Hook function signature:
        hook(module, input, output) -> modified_output

    Thread safety:
        Hook functions are called during forward pass.
        SAE steering/monitoring is thread-safe.

    Usage:
        hooker = SAEHooker()
        handle = hooker.install(model, layer=12, sae=loaded_sae)
        # ... use model with SAE active ...
        hooker.remove(handle)
    """

    def install(
        self,
        model: nn.Module,
        layer: int,
        sae: LoadedSAE,
    ) -> RemovableHandle:
        """
        Install forward hook at specified layer.

        Args:
            model: The loaded transformer model.
            layer: Target layer index (0-indexed).
            sae: Loaded SAE to apply.

        Returns:
            Hook handle for later removal.

        Raises:
            ValueError: If layer cannot be found in model.
        """
        # Get target layer
        target_layer = self._get_layer(model, layer)

        # Resolve the module's qualified name so operators can verify the hook
        # landed on the intended decoder layer (the accessor/ModuleList fallback
        # can pick a wrong module on exotic or multimodal architectures).
        self.last_resolved_module_path = self._resolve_module_path(
            model, target_layer, layer
        )

        # Create hook function
        hook_fn = self._create_hook_fn(sae)

        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)

        # NOTE: stdlib logger — no structlog kwargs here (a kwarg call made
        # EVERY attach 500 with 'Logger._log() got an unexpected keyword
        # argument'; pinned by tests/unit/test_logging_conventions.py)
        logger.info(
            "sae_hook_installed: layer=%s module_path=%s mode=direct_steering",
            layer,
            self.last_resolved_module_path,
        )
        return handle

    def layer_device(self, model: nn.Module, layer: int) -> torch.device:
        """The device holding `layer`'s weights.

        Delegates to `millm.ml.layer_resolution` (024 task 1.3). Kept as a method because
        callers and tests reach for it that way.
        """
        return _resolve_layer_device(model, layer)

    @staticmethod
    def _resolve_module_path(
        model: nn.Module, target: nn.Module, layer_idx: int
    ) -> str:
        """Return the dotted name of `target` within `model`, for observability."""
        try:
            for name, module in model.named_modules():
                if module is target:
                    return name
        except Exception:
            pass
        return f"<layer {layer_idx}>"

    def remove(self, handle: RemovableHandle) -> None:
        """
        Remove a previously installed hook.

        Args:
            handle: The handle returned from install().
        """
        handle.remove()
        logger.info("Removed SAE hook")

    def _create_hook_fn(self, sae: LoadedSAE) -> Callable:
        """
        Create the hook function for direct steering.

        The hook applies steering by adding decoder directions to hidden states,
        matching miStudio/Neuronpedia behavior.
        """

        def hook_fn(
            module: nn.Module,
            input: Tuple[Tensor, ...],
            output: Union[Tensor, Tuple[Tensor, ...]],
        ) -> Union[Tensor, Tuple[Tensor, ...]]:
            """
            Forward hook that applies direct residual stream steering.

            Handles the three output formats produced by HuggingFace transformers:
            - Single Tensor            — older/simple architectures
            - tuple[Tensor, ...]      — most common transformer layers
            - ModelOutput (dataclass) — e.g. CausalLMOutputWithPast, which is an
              OrderedDict subclass that also supports index access like a tuple
            """
            # ── Extract hidden states ──────────────────────────────────────────
            if isinstance(output, Tensor):
                hidden_states = output
            elif isinstance(output, tuple):
                hidden_states = output[0]
            else:
                # HF ModelOutput dataclasses (OrderedDict subclasses) support
                # index access: output[0] returns the first non-None value which
                # is always the hidden states for transformer layer outputs.
                try:
                    hidden_states = output[0]
                except (TypeError, KeyError, IndexError):
                    logger.warning(
                        "sae_hook_unsupported_output_type: %s",
                        type(output).__name__,
                    )
                    return output  # pass through unmodified

            if not isinstance(hidden_states, Tensor):
                # First element is not a tensor (None, metadata, etc.) — skip
                return output

            # ── Device ────────────────────────────────────────────────────────
            # The SAE must sit on this layer's device. Attach places it there;
            # this catches anything that moved it since. Encoding across devices
            # raised inside the forward pass (monitoring) or failed silently on
            # every pass (sensing), so move once and say so.
            if sae.W_enc.device != hidden_states.device:
                logger.warning(
                    "sae_moved_to_layer_device: from=%s to=%s",
                    sae.W_enc.device,
                    hidden_states.device,
                )
                sae.to_device(str(hidden_states.device))

            # ── Per-request activations, pre-steering read (Feature 27) ──────
            # Before monitoring and steering, and NOT gated on suppression: scoring mode reads its
            # activations with every SAE suppressed (FR-27.3). Owner-checked inside.
            if getattr(sae, "request_capture", None) is not None:
                sae.feed_request_capture(hidden_states, "pre")

            # ── Monitoring ────────────────────────────────────────────────────
            if sae.is_monitoring_enabled:
                with torch.no_grad():
                    x = hidden_states
                    if x.dtype != sae.W_enc.dtype:
                        x = x.to(sae.W_enc.dtype)
                    sae._capture_activations(sae.encode(x))

            # ── Co-activation sensing (Feature 11) ───────────────────────────
            # Sibling of monitoring (evaluates even when monitoring is off),
            # BEFORE apply_steering so positions reflect the pre-steer
            # residual read. _sense() respects suppressed() internally and
            # never raises into the forward pass.
            if sae.is_sensing_armed:
                with torch.no_grad():
                    sae._sense(hidden_states)

            # Feature 15 edge sensing — a SIBLING of the F11 branch above, not
            # nested under it: a deployment may run cluster sensing and circuit
            # edge sensing simultaneously, and each arms independently.
            if sae.is_edge_sensing_armed:
                with torch.no_grad():
                    sae._sense_edges(hidden_states)

            # ── Steering ──────────────────────────────────────────────────────
            modified = sae.apply_steering(hidden_states)

            # ── Per-request activations, post-steering read (Feature 27) ─────
            # What the model computed at this layer: after this layer's steering delta.
            if getattr(sae, "request_capture", None) is not None:
                sae.feed_request_capture(modified, "post")

            # ── Reconstruct output with same type ────────────────────────────
            if isinstance(output, Tensor):
                return modified
            elif isinstance(output, tuple):
                return (modified,) + output[1:]
            else:
                # HF ModelOutput: reconstruct from its own dict representation so
                # downstream code that pattern-matches on the type still works.
                try:
                    output_dict = dict(output)
                    first_key = next(iter(output_dict))
                    output_dict[first_key] = modified
                    return type(output)(**output_dict)
                except Exception:
                    # Last resort: return as a plain tuple (HF code handles this)
                    items = list(output)
                    items[0] = modified
                    return tuple(items)

        return hook_fn

    def _get_layer(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """The layer module at `layer_idx`. Delegates to `millm.ml.layer_resolution`."""
        return _resolve_layer(model, layer_idx)

    def get_layer_count(self, model: nn.Module) -> int:
        """Total number of layers. Delegates to `millm.ml.layer_resolution`."""
        return _resolve_layer_count(model)

    def validate_layer(self, model: nn.Module, layer: int) -> bool:
        """
        Validate that a layer index is valid for the model.

        Args:
            model: The transformer model.
            layer: Layer index to validate.

        Returns:
            True if layer is valid.
        """
        try:
            num_layers = self.get_layer_count(model)
            return 0 <= layer < num_layers
        except ValueError:
            # If we can't determine layer count, try to access the layer
            try:
                self._get_layer(model, layer)
                return True
            except (ValueError, IndexError):
                return False
