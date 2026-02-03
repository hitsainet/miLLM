"""
LoadedSAE wrapper for inference.

Handles encoding, decoding, steering, and monitoring for attached SAEs.
"""

import logging
from typing import Optional

import torch
from torch import Tensor

from millm.ml.sae_config import SAEConfig

logger = logging.getLogger(__name__)


class LoadedSAE:
    """
    Loaded SAE with encoder and decoder weights.

    Implements the SAE forward pass with optional steering and monitoring.

    Thread-safety notes:
    - Forward pass is thread-safe (no mutation)
    - Steering modification should use external lock if concurrent
    - Monitoring capture creates new tensor (safe)

    Memory layout:
    - W_enc: (d_in, d_sae) - encoder weights
    - b_enc: (d_sae,) - encoder bias
    - W_dec: (d_sae, d_in) - decoder weights
    - b_dec: (d_in,) - decoder bias

    Attributes:
        W_enc: Encoder weight matrix.
        b_enc: Encoder bias vector.
        W_dec: Decoder weight matrix.
        b_dec: Decoder bias vector.
        config: SAE configuration.
        device: Current device (cpu/cuda).
        d_in: Input dimension.
        d_sae: SAE feature dimension.
    """

    def __init__(
        self,
        W_enc: Tensor,
        b_enc: Tensor,
        W_dec: Tensor,
        b_dec: Tensor,
        config: SAEConfig,
        device: str = "cpu",
    ) -> None:
        """
        Initialize LoadedSAE with weight tensors.

        Args:
            W_enc: Encoder weights (d_in, d_sae).
            b_enc: Encoder bias (d_sae,).
            W_dec: Decoder weights (d_sae, d_in).
            b_dec: Decoder bias (d_in,).
            config: SAE configuration.
            device: Target device.

        Raises:
            AssertionError: If tensor dimensions don't match config.
        """
        self.W_enc = W_enc.to(device)
        self.b_enc = b_enc.to(device)
        self.W_dec = W_dec.to(device)
        self.b_dec = b_dec.to(device)
        self.config = config
        self.device = device

        # Extract dimensions from weights
        self.d_in = W_enc.shape[0]
        self.d_sae = W_enc.shape[1]

        # Validate dimensions match config
        assert self.d_in == config.d_in, (
            f"d_in mismatch: weights have {self.d_in}, config has {config.d_in}"
        )
        assert self.d_sae == config.d_sae, (
            f"d_sae mismatch: weights have {self.d_sae}, config has {config.d_sae}"
        )

        # Steering state
        self._steering_values: dict[int, float] = {}
        self._steering_enabled: bool = False
        self._steering_vector: Optional[Tensor] = None

        # Monitoring state
        self._monitoring_enabled: bool = False
        self._monitored_features: Optional[list[int]] = None
        self._last_feature_acts: Optional[Tensor] = None

        logger.debug(
            f"LoadedSAE initialized: d_in={self.d_in}, d_sae={self.d_sae}, "
            f"device={device}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through SAE.

        Performs: encode -> (optional steering) -> decode

        Args:
            x: Input activations (batch, seq_len, d_in).

        Returns:
            Reconstructed activations (batch, seq_len, d_in).
        """
        # Store original dtype for output conversion
        original_dtype = x.dtype

        # Cast input to SAE weight dtype if different
        if x.dtype != self.W_enc.dtype:
            x = x.to(self.W_enc.dtype)

        # Encode: x @ W_enc + b_enc with ReLU
        feature_acts = torch.relu(x @ self.W_enc + self.b_enc)

        # Capture for monitoring
        if self._monitoring_enabled:
            self._capture_activations(feature_acts)

        # Apply steering
        if self._steering_enabled and self._steering_vector is not None:
            feature_acts = feature_acts + self._steering_vector

        # Decode: feature_acts @ W_dec + b_dec
        reconstructed = feature_acts @ self.W_dec + self.b_dec

        # Cast back to original dtype
        if reconstructed.dtype != original_dtype:
            reconstructed = reconstructed.to(original_dtype)

        return reconstructed

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode activations to feature space.

        Useful for monitoring and analysis without full reconstruction.

        Args:
            x: Input activations (batch, seq_len, d_in).

        Returns:
            Feature activations (batch, seq_len, d_sae).
        """
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, feature_acts: Tensor) -> Tensor:
        """
        Decode feature activations to input space.

        Args:
            feature_acts: Feature activations (batch, seq_len, d_sae).

        Returns:
            Reconstructed activations (batch, seq_len, d_in).
        """
        return feature_acts @ self.W_dec + self.b_dec

    # ==========================================================================
    # Steering Methods
    # ==========================================================================

    def set_steering(self, feature_idx: int, value: float) -> None:
        """
        Set steering value for a feature.

        Args:
            feature_idx: Feature index (0 to d_sae-1).
            value: Steering strength (positive=amplify, negative=suppress).

        Raises:
            ValueError: If feature_idx is out of range.
        """
        if not 0 <= feature_idx < self.d_sae:
            raise ValueError(
                f"Feature index {feature_idx} out of range [0, {self.d_sae})"
            )

        self._steering_values[feature_idx] = value
        self._rebuild_steering_vector()

    def set_steering_batch(self, steering: dict[int, float]) -> None:
        """
        Set multiple steering values at once.

        Args:
            steering: Dictionary mapping feature indices to steering values.

        Raises:
            ValueError: If any feature index is out of range.
        """
        for idx in steering.keys():
            if not 0 <= idx < self.d_sae:
                raise ValueError(f"Feature index {idx} out of range [0, {self.d_sae})")

        self._steering_values.update(steering)
        self._rebuild_steering_vector()

    def clear_steering(self, feature_idx: Optional[int] = None) -> None:
        """
        Clear steering for one or all features.

        Args:
            feature_idx: Specific feature to clear (None = clear all).
        """
        if feature_idx is None:
            self._steering_values.clear()
        elif feature_idx in self._steering_values:
            del self._steering_values[feature_idx]

        self._rebuild_steering_vector()

    def get_steering_values(self) -> dict[int, float]:
        """Get current steering values (copy)."""
        return dict(self._steering_values)

    def enable_steering(self, enabled: bool = True) -> None:
        """Enable or disable steering application."""
        self._steering_enabled = enabled

    @property
    def is_steering_enabled(self) -> bool:
        """Check if steering is enabled."""
        return self._steering_enabled

    def _rebuild_steering_vector(self) -> None:
        """Rebuild pre-computed steering vector from values."""
        if not self._steering_values:
            self._steering_vector = None
            return

        # Create sparse steering vector
        vector = torch.zeros(self.d_sae, device=self.device, dtype=self.W_enc.dtype)
        for idx, value in self._steering_values.items():
            vector[idx] = value

        self._steering_vector = vector

    # ==========================================================================
    # Monitoring Methods
    # ==========================================================================

    def enable_monitoring(
        self,
        enabled: bool = True,
        features: Optional[list[int]] = None,
    ) -> None:
        """
        Enable feature activation monitoring.

        Args:
            enabled: Whether to capture activations.
            features: Specific features to monitor (None = all).
                      Monitoring specific features reduces memory usage.
        """
        self._monitoring_enabled = enabled
        self._monitored_features = features

        if not enabled:
            self._last_feature_acts = None

    def get_last_feature_activations(self) -> Optional[Tensor]:
        """
        Get feature activations from last forward pass.

        Returns:
            Feature activations tensor or None if monitoring disabled.
        """
        return self._last_feature_acts

    @property
    def is_monitoring_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        return self._monitoring_enabled

    def _capture_activations(self, feature_acts: Tensor) -> None:
        """Capture activations for monitoring."""
        if self._monitored_features is not None:
            # Only capture selected features
            self._last_feature_acts = (
                feature_acts[..., self._monitored_features].detach().clone()
            )
        else:
            # Capture all (may be memory intensive for large SAEs)
            self._last_feature_acts = feature_acts.detach().clone()

    # ==========================================================================
    # Memory Management
    # ==========================================================================

    def estimate_memory_mb(self) -> float:
        """
        Estimate current GPU memory usage.

        Returns:
            Estimated memory in MB.
        """
        return self.config.estimate_memory_mb()

    def to_device(self, device: str) -> None:
        """
        Move all tensors to device.

        Args:
            device: Target device (e.g., "cuda", "cpu", "cuda:0").
        """
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_dec = self.b_dec.to(device)

        if self._steering_vector is not None:
            self._steering_vector = self._steering_vector.to(device)

        self.device = device
        logger.debug(f"LoadedSAE moved to {device}")

    def to_cpu(self) -> None:
        """Move all tensors to CPU (for cleanup)."""
        self.to_device("cpu")

    def __repr__(self) -> str:
        return (
            f"LoadedSAE(d_in={self.d_in}, d_sae={self.d_sae}, "
            f"device={self.device}, steering={self._steering_enabled})"
        )
