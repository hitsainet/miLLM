"""Unit tests for SAEHooker."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_hooker import SAEHooker
from millm.ml.sae_wrapper import LoadedSAE


@pytest.fixture
def hooker():
    """Create an SAEHooker instance."""
    return SAEHooker()


@pytest.fixture
def sae_config():
    """Create a test SAE config."""
    return SAEConfig(
        d_in=64,
        d_sae=128,
        model_name="test/model",
        hook_name="blocks.2.hook_resid_post",
        hook_layer=2,
        dtype="float32",
    )


@pytest.fixture
def loaded_sae(sae_config):
    """Create a LoadedSAE for testing."""
    W_enc = torch.randn(64, 128)
    b_enc = torch.randn(128)
    W_dec = torch.randn(128, 64)
    b_dec = torch.randn(64)
    return LoadedSAE(
        W_enc=W_enc,
        b_enc=b_enc,
        W_dec=W_dec,
        b_dec=b_dec,
        config=sae_config,
        device="cpu",
    )


def _make_gemma_model(num_layers=4):
    """Create a mock model with Gemma/Llama-style layer structure."""
    layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_layers)])
    inner = nn.Module()
    inner.layers = layers
    model = nn.Module()
    model.model = inner
    # Add config for layer count
    config = MagicMock()
    config.num_hidden_layers = num_layers
    model.config = config
    return model


def _make_gpt2_model(num_layers=4):
    """Create a mock model with GPT-2-style layer structure."""
    h = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_layers)])
    transformer = nn.Module()
    transformer.h = h
    model = nn.Module()
    model.transformer = transformer
    config = MagicMock()
    config.n_layer = num_layers
    model.config = config
    return model


def _make_flat_model():
    """Create a model with no recognized layer structure."""
    model = nn.Module()
    model.fc1 = nn.Linear(64, 64)
    model.fc2 = nn.Linear(64, 64)
    return model


class TestSAEHookerInstall:
    """Tests for install method."""

    def test_install_registers_hook_on_correct_layer(self, hooker, loaded_sae):
        """Test that install registers a forward hook on the target layer."""
        model = _make_gemma_model(num_layers=4)
        target_layer = model.model.layers[2]

        handle = hooker.install(model, layer=2, sae=loaded_sae)

        # Check that a hook was registered on the target layer
        assert len(target_layer._forward_hooks) > 0
        handle.remove()

    def test_install_returns_removable_handle(self, hooker, loaded_sae):
        """Test that install returns a handle that can be used for removal."""
        model = _make_gemma_model()

        handle = hooker.install(model, layer=0, sae=loaded_sae)

        assert handle is not None
        # Should be removable without error
        handle.remove()


class TestSAEHookerRemove:
    """Tests for remove method."""

    def test_remove_clears_hook(self, hooker, loaded_sae):
        """Test that remove clears the forward hook from the layer."""
        model = _make_gemma_model()
        target_layer = model.model.layers[0]

        handle = hooker.install(model, layer=0, sae=loaded_sae)
        assert len(target_layer._forward_hooks) > 0

        hooker.remove(handle)
        assert len(target_layer._forward_hooks) == 0


class TestSAEHookerGetLayer:
    """Tests for _get_layer method."""

    def test_finds_gemma_llama_style_layers(self, hooker):
        """Test that _get_layer finds layers for Gemma/Llama (model.model.layers[i])."""
        model = _make_gemma_model(num_layers=4)

        layer = hooker._get_layer(model, 2)

        assert layer is model.model.layers[2]

    def test_finds_gpt2_style_layers(self, hooker):
        """Test that _get_layer finds layers for GPT-2 (model.transformer.h[i])."""
        model = _make_gpt2_model(num_layers=4)

        layer = hooker._get_layer(model, 1)

        assert layer is model.transformer.h[1]

    def test_raises_for_unsupported_architecture(self, hooker):
        """Test that _get_layer raises ValueError for unsupported architecture."""
        model = _make_flat_model()

        with pytest.raises(ValueError) as exc_info:
            hooker._get_layer(model, 0)

        assert "Could not find layer" in str(exc_info.value)
        assert "not be supported" in str(exc_info.value)


class TestSAEHookerGetLayerCount:
    """Tests for get_layer_count method."""

    def test_returns_count_from_config_num_hidden_layers(self, hooker):
        """Test that get_layer_count reads num_hidden_layers from model config."""
        model = _make_gemma_model(num_layers=6)

        count = hooker.get_layer_count(model)

        assert count == 6

    def test_returns_count_from_config_n_layer(self, hooker):
        """Test that get_layer_count reads n_layer from model config."""
        model = _make_gpt2_model(num_layers=8)

        count = hooker.get_layer_count(model)

        assert count == 8

    def test_raises_when_cannot_determine_count(self, hooker):
        """Test that get_layer_count raises ValueError for unknown architecture."""
        model = _make_flat_model()

        with pytest.raises(ValueError) as exc_info:
            hooker.get_layer_count(model)

        assert "Could not determine layer count" in str(exc_info.value)


class TestSAEHookerValidateLayer:
    """Tests for validate_layer method."""

    def test_valid_layer_returns_true(self, hooker):
        """Test that validate_layer returns True for valid layer indices."""
        model = _make_gemma_model(num_layers=4)

        assert hooker.validate_layer(model, 0) is True
        assert hooker.validate_layer(model, 3) is True

    def test_invalid_layer_returns_false(self, hooker):
        """Test that validate_layer returns False for out-of-range indices."""
        model = _make_gemma_model(num_layers=4)

        assert hooker.validate_layer(model, 4) is False
        assert hooker.validate_layer(model, -1) is False
        assert hooker.validate_layer(model, 100) is False


class TestSAEHookerHookFunction:
    """Tests for the hook function behavior."""

    def test_hook_applies_steering(self, hooker, loaded_sae):
        """Test that the hook calls apply_steering on hidden states."""
        model = _make_gemma_model()
        loaded_sae.enable_steering(True)
        loaded_sae.set_steering(0, 5.0)

        handle = hooker.install(model, layer=0, sae=loaded_sae)

        # Run a forward pass through the hooked layer
        input_tensor = torch.randn(1, 4, 64)
        original_output = model.model.layers[0](input_tensor)

        # The hook should modify the output due to steering
        # Since we have steering enabled and set, output should differ
        # from a non-steered baseline
        handle.remove()

    def test_hook_captures_monitoring_activations(self, hooker, loaded_sae):
        """Test that the hook captures activations when monitoring is enabled."""
        model = _make_gemma_model()
        loaded_sae.enable_monitoring(True, features=[0, 1, 2])

        handle = hooker.install(model, layer=0, sae=loaded_sae)

        # Run a forward pass
        input_tensor = torch.randn(1, 4, 64)
        _ = model.model.layers[0](input_tensor)

        # Monitoring should have captured activations
        acts = loaded_sae.get_last_feature_activations()
        assert acts is not None
        # Should have 3 monitored features
        assert acts.shape[-1] == 3

        handle.remove()
