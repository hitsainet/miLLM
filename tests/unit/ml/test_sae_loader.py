"""Unit tests for SAELoader."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_loader import SAELoader


@pytest.fixture
def loader():
    """Create an SAELoader instance."""
    return SAELoader()


@pytest.fixture
def sae_config_dict():
    """Create a standard SAE config dictionary."""
    return {
        "d_in": 64,
        "d_sae": 128,
        "model_name": "test/model",
        "hook_name": "blocks.2.hook_resid_post",
        "hook_layer": 2,
        "dtype": "float32",
    }


@pytest.fixture
def sae_weights():
    """Create SAE weight tensors with matching dimensions."""
    d_in, d_sae = 64, 128
    return {
        "W_enc": torch.randn(d_in, d_sae),
        "b_enc": torch.randn(d_sae),
        "W_dec": torch.randn(d_sae, d_in),
        "b_dec": torch.randn(d_in),
    }


@pytest.fixture
def sae_dir_safetensors(tmp_path, sae_config_dict, sae_weights):
    """Create a directory with SafeTensors format SAE files."""
    sae_dir = tmp_path / "sae_safetensors"
    sae_dir.mkdir()

    # Write cfg.json
    config_path = sae_dir / "cfg.json"
    config_path.write_text(json.dumps(sae_config_dict))

    # Write safetensors weights file
    from safetensors.torch import save_file

    save_file(sae_weights, str(sae_dir / "sae_weights.safetensors"))

    return sae_dir


@pytest.fixture
def sae_dir_npz(tmp_path, sae_config_dict):
    """Create a directory with NPZ format SAE files."""
    sae_dir = tmp_path / "sae_npz"
    sae_dir.mkdir()

    # Write cfg.json
    config_path = sae_dir / "cfg.json"
    config_path.write_text(json.dumps(sae_config_dict))

    # Write NPZ weights
    d_in, d_sae = 64, 128
    np.savez(
        str(sae_dir / "params.npz"),
        W_enc=np.random.randn(d_in, d_sae).astype(np.float32),
        b_enc=np.random.randn(d_sae).astype(np.float32),
        W_dec=np.random.randn(d_sae, d_in).astype(np.float32),
        b_dec=np.random.randn(d_in).astype(np.float32),
    )

    return sae_dir


class TestSAELoaderLoadSafeTensors:
    """Tests for loading SAEs from SafeTensors format."""

    def test_loads_safetensors_format(self, loader, sae_dir_safetensors):
        """Test that load successfully loads from SafeTensors files."""
        sae = loader.load(sae_dir_safetensors, device="cpu")

        assert sae is not None
        assert sae.d_in == 64
        assert sae.d_sae == 128

    def test_loaded_sae_has_correct_weight_shapes(self, loader, sae_dir_safetensors):
        """Test that loaded SAE has correctly shaped weight tensors."""
        sae = loader.load(sae_dir_safetensors, device="cpu")

        assert sae.W_enc.shape == (64, 128)
        assert sae.b_enc.shape == (128,)
        assert sae.W_dec.shape == (128, 64)
        assert sae.b_dec.shape == (64,)


class TestSAELoaderLoadNPZ:
    """Tests for loading SAEs from NPZ format."""

    def test_loads_npz_format(self, loader, sae_dir_npz):
        """Test that load successfully loads from NPZ files."""
        sae = loader.load(sae_dir_npz, device="cpu")

        assert sae is not None
        assert sae.d_in == 64
        assert sae.d_sae == 128

    def test_npz_weights_are_torch_tensors(self, loader, sae_dir_npz):
        """Test that NPZ weights are properly converted to torch tensors."""
        sae = loader.load(sae_dir_npz, device="cpu")

        assert isinstance(sae.W_enc, torch.Tensor)
        assert isinstance(sae.b_enc, torch.Tensor)
        assert isinstance(sae.W_dec, torch.Tensor)
        assert isinstance(sae.b_dec, torch.Tensor)


class TestSAELoaderConfig:
    """Tests for config loading."""

    def test_loads_config_from_cfg_json(self, loader, sae_dir_safetensors):
        """Test that load_config finds and parses cfg.json."""
        config = loader.load_config(sae_dir_safetensors)

        assert config.d_in == 64
        assert config.d_sae == 128
        assert config.model_name == "test/model"
        assert config.hook_layer == 2

    def test_raises_for_missing_config(self, loader, tmp_path):
        """Test that load_config raises FileNotFoundError when no config exists."""
        empty_dir = tmp_path / "empty_sae"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_config(empty_dir)

        assert "No configuration source found" in str(exc_info.value)

    def test_loaded_sae_has_correct_dimensions_from_config(self, loader, sae_dir_safetensors):
        """Test that loaded SAE dimensions match config values."""
        config = loader.load_config(sae_dir_safetensors)
        sae = loader.load(sae_dir_safetensors, device="cpu")

        assert sae.d_in == config.d_in
        assert sae.d_sae == config.d_sae


class TestSAELoaderFindWeightsFile:
    """Tests for _find_weights_file method."""

    def test_finds_safetensors_file(self, loader, sae_dir_safetensors):
        """Test that _find_weights_file discovers safetensors file."""
        weights_path = loader._find_weights_file(sae_dir_safetensors)

        assert weights_path.suffix == ".safetensors"
        assert weights_path.exists()

    def test_finds_npz_file(self, loader, sae_dir_npz):
        """Test that _find_weights_file discovers npz file."""
        weights_path = loader._find_weights_file(sae_dir_npz)

        assert weights_path.suffix == ".npz"
        assert weights_path.exists()

    def test_raises_when_no_weights_found(self, loader, tmp_path):
        """Test that _find_weights_file raises FileNotFoundError."""
        empty_dir = tmp_path / "no_weights"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            loader._find_weights_file(empty_dir)

        assert "No weights file found" in str(exc_info.value)
