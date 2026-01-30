"""
Model hook management for SAE attachment.

Installs PyTorch forward hooks to intercept and modify activations.
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

    Installs hooks that intercept layer outputs, apply SAE transformation,
    and return modified activations.

    Hook function signature:
        hook(module, input, output) -> modified_output

    Thread safety:
        Hook functions are called during forward pass.
        Ensure SAE forward is thread-safe.

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

        # Create hook function
        hook_fn = self._create_hook_fn(sae)

        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)

        logger.info(f"Installed SAE hook at layer {layer}")
        return handle

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
        Create the hook function for SAE.

        The hook intercepts layer output, applies SAE, returns modified output.
        """

        def hook_fn(
            module: nn.Module,
            input: Tuple[Tensor, ...],
            output: Union[Tensor, Tuple[Tensor, ...]],
        ) -> Union[Tensor, Tuple[Tensor, ...]]:
            """
            Forward hook that applies SAE.

            Handles different output formats:
            - Tuple (hidden_states, ...) - common in transformers
            - Single tensor
            """
            # Handle tuple output (common for transformer layers)
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Apply SAE
                modified = sae.forward(hidden_states)
                # Return with same structure
                return (modified,) + output[1:]
            else:
                # Single tensor output
                return sae.forward(output)

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
