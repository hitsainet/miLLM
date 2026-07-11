"""
Model hook management for SAE attachment.

Implements direct residual stream steering (miStudio/Neuronpedia compatible).

Steering Formula:
    modified_activations = original_activations + Σ(strength_i × decoder_direction_i)

The hook applies steering uniformly to all token positions without full SAE reconstruction.
"""

import logging
from typing import Callable, Tuple, Union

import torch
from torch import nn, Tensor
from torch.utils.hooks import RemovableHandle

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

        logger.info(
            "sae_hook_installed",
            layer=layer,
            module_path=self.last_resolved_module_path,
            mode="direct_steering",
        )
        return handle

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

            # ── Monitoring ────────────────────────────────────────────────────
            if sae.is_monitoring_enabled:
                with torch.no_grad():
                    x = hidden_states
                    if x.dtype != sae.W_enc.dtype:
                        x = x.to(sae.W_enc.dtype)
                    sae._capture_activations(sae.encode(x))

            # ── Steering ──────────────────────────────────────────────────────
            modified = sae.apply_steering(hidden_states)

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
        """
        Get the layer module at specified index.

        Supports multiple transformer architectures:
        - Gemma/Llama: model.model.layers[layer_idx]
        - GPT-2: model.transformer.h[layer_idx]
        - Generic: model.layers[layer_idx]

        Args:
            model: The transformer model.
            layer_idx: Target layer index.

        Returns:
            The layer module.

        Raises:
            ValueError: If layer cannot be found.
        """
        # Architecture-specific layer access patterns
        layer_access_patterns = [
            # Gemma, Llama, Mistral style
            lambda m: m.model.layers[layer_idx],
            # GPT-2, GPT-Neo style
            lambda m: m.transformer.h[layer_idx],
            # Some HF models
            lambda m: m.model.decoder.layers[layer_idx],
            # Generic patterns
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

        # Fallback: search for ModuleList containing layers
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > layer_idx:
                # Check if this looks like a layer list
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

    def get_layer_count(self, model: nn.Module) -> int:
        """
        Get total number of layers in model.

        Args:
            model: The transformer model.

        Returns:
            Number of layers.

        Raises:
            ValueError: If layer count cannot be determined.
        """
        # Try config first (most reliable)
        if hasattr(model, "config"):
            config = model.config
            for attr in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
                if hasattr(config, attr):
                    return getattr(config, attr)

        # Try to find and count layers directly
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

        # Fallback: search for ModuleList
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                # Check if this looks like a layer list
                first_child = list(module.children())[0] if len(list(module.children())) > 0 else None
                if first_child is not None and hasattr(first_child, "self_attn"):
                    return len(module)

        raise ValueError(
            "Could not determine layer count. "
            "Model config should have num_hidden_layers or similar attribute."
        )

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
