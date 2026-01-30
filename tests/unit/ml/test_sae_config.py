"""
Unit tests for SAE configuration parsing.
"""

import json
import tempfile
from pathlib import Path

import pytest

from millm.ml.sae_config import SAEConfig


class TestSAEConfigFromJson:
    """Tests for loading SAE config from JSON files."""

    def test_loads_standard_saelens_format(self, tmp_path):
        """Standard SAELens cfg.json format."""
        config_data = {
            "d_in": 2304,
            "d_sae": 16384,
            "model_name": "google/gemma-2-2b",
            "hook_name": "blocks.12.hook_resid_post",
            "hook_layer": 12,
            "dtype": "float32",
            "normalize_activations": "none",
        }

        (tmp_path / "cfg.json").write_text(json.dumps(config_data))

        config = SAEConfig.from_json(tmp_path)

        assert config.d_in == 2304
        assert config.d_sae == 16384
        assert config.model_name == "google/gemma-2-2b"
        assert config.hook_name == "blocks.12.hook_resid_post"
        assert config.hook_layer == 12
        assert config.dtype == "float32"

    def test_handles_alternative_config_names(self, tmp_path):
        """Falls back to config.json if cfg.json not found."""
        config_data = {
            "d_in": 64,
            "d_sae": 128,
            "model_name": "test",
            "hook_name": "layer.0",
            "hook_layer": 0,
        }

        (tmp_path / "config.json").write_text(json.dumps(config_data))

        config = SAEConfig.from_json(tmp_path)

        assert config.d_in == 64
        assert config.d_sae == 128

    def test_handles_dimension_variations(self, tmp_path):
        """Handles d_model/d_hidden naming variations."""
        config_data = {
            "d_model": 768,  # Alternative to d_in
            "d_hidden": 3072,  # Alternative to d_sae
            "model_name": "test",
            "hook_name": "blocks.0.mlp",
            "hook_layer": 0,
        }

        (tmp_path / "cfg.json").write_text(json.dumps(config_data))

        config = SAEConfig.from_json(tmp_path)

        assert config.d_in == 768
        assert config.d_sae == 3072

    def test_raises_on_missing_config(self, tmp_path):
        """Raises FileNotFoundError if no config file found."""
        with pytest.raises(FileNotFoundError, match="No configuration file found"):
            SAEConfig.from_json(tmp_path)

    def test_raises_on_missing_dimensions(self, tmp_path):
        """Raises ValueError if dimensions are missing."""
        config_data = {
            "model_name": "test",
            "hook_name": "layer.0",
        }

        (tmp_path / "cfg.json").write_text(json.dumps(config_data))

        with pytest.raises(ValueError, match="Config missing required dimensions"):
            SAEConfig.from_json(tmp_path)

    def test_extracts_layer_from_hook_name(self, tmp_path):
        """Extracts layer index from hook_name if hook_layer not provided."""
        config_data = {
            "d_in": 64,
            "d_sae": 128,
            "model_name": "test",
            "hook_name": "blocks.5.hook_resid_post",
            # hook_layer not provided
        }

        (tmp_path / "cfg.json").write_text(json.dumps(config_data))

        config = SAEConfig.from_json(tmp_path)

        assert config.hook_layer == 5


class TestSAEConfigMemoryEstimate:
    """Tests for memory estimation."""

    def test_estimates_float32_memory(self):
        """Estimates memory for float32 weights."""
        config = SAEConfig(
            d_in=64,
            d_sae=128,
            model_name="test",
            hook_name="test",
            hook_layer=0,
            dtype="float32",
        )

        # Memory = (encoder + decoder) * 4 bytes
        # encoder: 64*128 + 128 = 8320 params
        # decoder: 128*64 + 64 = 8256 params
        # Total: 16576 * 4 = 66304 bytes = ~0.063 MB
        memory_mb = config.estimate_memory_mb()

        assert 0.05 < memory_mb < 0.1

    def test_estimates_float16_memory(self):
        """Estimates memory for float16 weights (half of float32)."""
        config_fp32 = SAEConfig(
            d_in=64,
            d_sae=128,
            model_name="test",
            hook_name="test",
            hook_layer=0,
            dtype="float32",
        )
        config_fp16 = SAEConfig(
            d_in=64,
            d_sae=128,
            model_name="test",
            hook_name="test",
            hook_layer=0,
            dtype="float16",
        )

        memory_fp32 = config_fp32.estimate_memory_mb()
        memory_fp16 = config_fp16.estimate_memory_mb()

        # fp16 should be half the memory
        assert abs(memory_fp16 - memory_fp32 / 2) < 0.001


class TestSAEConfigFromDict:
    """Tests for creating config from dictionary."""

    def test_creates_from_dict(self):
        """Creates config from dictionary."""
        data = {
            "d_in": 256,
            "d_sae": 512,
            "model_name": "model",
            "hook_name": "hook",
            "hook_layer": 3,
        }

        config = SAEConfig.from_dict(data)

        assert config.d_in == 256
        assert config.d_sae == 512
        assert config.hook_layer == 3

    def test_to_dict_roundtrip(self):
        """Config can be serialized to dict and back."""
        original = SAEConfig(
            d_in=64,
            d_sae=128,
            model_name="test",
            hook_name="blocks.0",
            hook_layer=0,
            dtype="float16",
        )

        data = original.to_dict()
        restored = SAEConfig.from_dict(data)

        assert restored.d_in == original.d_in
        assert restored.d_sae == original.d_sae
        assert restored.dtype == original.dtype
