"""
Unit tests for LoadedSAE wrapper.
"""

import pytest
import torch

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE


@pytest.fixture
def small_sae():
    """Create a small SAE for testing."""
    d_in, d_sae = 64, 128
    config = SAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        model_name="test",
        hook_name="test",
        hook_layer=0,
    )
    return LoadedSAE(
        W_enc=torch.randn(d_in, d_sae),
        b_enc=torch.zeros(d_sae),
        W_dec=torch.randn(d_sae, d_in),
        b_dec=torch.zeros(d_in),
        config=config,
        device="cpu",
    )


class TestLoadedSAEForward:
    """Tests for SAE forward pass."""

    def test_forward_preserves_shape(self, small_sae):
        """Forward pass preserves input shape."""
        x = torch.randn(2, 10, 64)  # batch=2, seq=10, d_in=64
        out = small_sae.forward(x)
        assert out.shape == x.shape

    def test_encode_produces_features(self, small_sae):
        """Encode produces d_sae features."""
        x = torch.randn(1, 5, 64)
        features = small_sae.encode(x)
        assert features.shape == (1, 5, 128)  # d_sae=128

    def test_decode_restores_dimension(self, small_sae):
        """Decode restores d_in dimension."""
        features = torch.randn(1, 5, 128)
        output = small_sae.decode(features)
        assert output.shape == (1, 5, 64)  # d_in=64

    def test_features_are_non_negative(self, small_sae):
        """Feature activations are non-negative (ReLU)."""
        x = torch.randn(1, 5, 64)
        features = small_sae.encode(x)
        assert (features >= 0).all()


class TestLoadedSAESteering:
    """Tests for steering functionality."""

    def test_steering_modifies_output(self, small_sae):
        """Steering changes the output."""
        x = torch.randn(1, 5, 64)

        # Without steering
        small_sae.enable_steering(False)
        out_baseline = small_sae.forward(x.clone())

        # With steering
        small_sae.enable_steering(True)
        small_sae.set_steering(0, 10.0)
        out_steered = small_sae.forward(x.clone())

        assert not torch.allclose(out_baseline, out_steered)

    def test_set_steering_single(self, small_sae):
        """Can set steering for a single feature."""
        small_sae.set_steering(42, 5.0)
        values = small_sae.get_steering_values()
        assert values == {42: 5.0}

    def test_set_steering_batch(self, small_sae):
        """Can set steering for multiple features at once."""
        steering = {0: 1.0, 10: 2.0, 50: -1.5}
        small_sae.set_steering_batch(steering)
        values = small_sae.get_steering_values()
        assert values == steering

    def test_clear_steering_single(self, small_sae):
        """Can clear steering for a single feature."""
        small_sae.set_steering(0, 1.0)
        small_sae.set_steering(1, 2.0)
        small_sae.clear_steering(0)

        values = small_sae.get_steering_values()
        assert values == {1: 2.0}

    def test_clear_steering_all(self, small_sae):
        """Can clear all steering."""
        small_sae.set_steering(0, 1.0)
        small_sae.set_steering(1, 2.0)
        small_sae.clear_steering()

        values = small_sae.get_steering_values()
        assert values == {}

    def test_rejects_invalid_feature_index(self, small_sae):
        """Rejects feature index out of range."""
        with pytest.raises(ValueError):
            small_sae.set_steering(-1, 1.0)

        with pytest.raises(ValueError):
            small_sae.set_steering(128, 1.0)  # d_sae=128, max index is 127

    def test_steering_disabled_by_default(self, small_sae):
        """Steering is disabled by default."""
        assert not small_sae.is_steering_enabled


class TestLoadedSAEMonitoring:
    """Tests for monitoring functionality."""

    def test_monitoring_captures_activations(self, small_sae):
        """Monitoring captures feature activations."""
        x = torch.randn(1, 5, 64)

        small_sae.enable_monitoring(True)
        small_sae.forward(x)

        acts = small_sae.get_last_feature_activations()
        assert acts is not None
        assert acts.shape == (1, 5, 128)  # d_sae=128

    def test_monitoring_specific_features(self, small_sae):
        """Can monitor only specific features."""
        x = torch.randn(1, 5, 64)

        small_sae.enable_monitoring(True, features=[0, 1, 2])
        small_sae.forward(x)

        acts = small_sae.get_last_feature_activations()
        assert acts.shape == (1, 5, 3)  # Only 3 features

    def test_monitoring_disabled_returns_none(self, small_sae):
        """When disabled, get_last_feature_activations returns None."""
        x = torch.randn(1, 5, 64)

        small_sae.enable_monitoring(False)
        small_sae.forward(x)

        acts = small_sae.get_last_feature_activations()
        assert acts is None

    def test_monitoring_disabled_by_default(self, small_sae):
        """Monitoring is disabled by default."""
        assert not small_sae.is_monitoring_enabled


class TestLoadedSAEMemory:
    """Tests for memory management."""

    def test_estimate_memory(self, small_sae):
        """Can estimate memory usage."""
        memory_mb = small_sae.estimate_memory_mb()
        assert memory_mb > 0

    def test_to_device(self, small_sae):
        """Can move tensors to device."""
        small_sae.to_device("cpu")
        assert small_sae.device == "cpu"
        assert small_sae.W_enc.device.type == "cpu"

    def test_dimensions_match_config(self, small_sae):
        """Dimensions match the config."""
        assert small_sae.d_in == 64
        assert small_sae.d_sae == 128
        assert small_sae.d_in == small_sae.config.d_in
        assert small_sae.d_sae == small_sae.config.d_sae
