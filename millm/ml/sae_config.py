"""
SAE configuration parsing.

Supports SAELens format cfg.json files from HuggingFace repositories.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SAEConfig:
    """
    SAE configuration from SAELens format.

    Supports various config file formats and field naming conventions
    used by different SAE releases.

    Expected cfg.json structure (SAELens standard):
    {
        "d_in": 2304,
        "d_sae": 16384,
        "model_name": "google/gemma-2-2b",
        "hook_name": "blocks.12.hook_resid_post",
        "hook_layer": 12,
        "dtype": "float32",
        "normalize_activations": "none"
    }

    Attributes:
        d_in: Input dimension (model hidden size).
        d_sae: SAE feature dimension.
        model_name: Model the SAE was trained on.
        hook_name: Hook point name in the model.
        hook_layer: Layer index where SAE should be attached.
        dtype: Weight data type ("float32", "float16", "bfloat16").
        normalize_activations: Normalization mode ("none", "expected_average_only_in").
    """

    d_in: int
    d_sae: int
    model_name: str
    hook_name: str
    hook_layer: int
    dtype: str = "float32"
    normalize_activations: str = "none"

    @classmethod
    def from_json(cls, path: str | Path) -> "SAEConfig":
        """
        Load config from SAELens cfg.json.

        Handles variations in field naming across different SAE releases.

        Args:
            path: Path to SAE directory (containing cfg.json) or direct path to config file.

        Returns:
            Parsed SAEConfig.

        Raises:
            FileNotFoundError: If no config file is found.
            ValueError: If required fields are missing.
            json.JSONDecodeError: If config file is not valid JSON.
        """
        config_path = cls._find_config_file(path)

        logger.debug(f"Loading SAE config from {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        return cls._parse_config(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SAEConfig":
        """
        Create config from dictionary.

        Args:
            data: Config dictionary.

        Returns:
            Parsed SAEConfig.
        """
        return cls._parse_config(data)

    @classmethod
    def _find_config_file(cls, path: str | Path) -> Path:
        """Find the config file in the given path."""
        path = Path(path)

        # If path is a file, use it directly
        if path.is_file():
            return path

        # If path is a directory, search for config files
        if path.is_dir():
            config_names = ["cfg.json", "config.json", "sae_cfg.json", "sae_config.json"]
            for name in config_names:
                config_path = path / name
                if config_path.exists():
                    return config_path

        raise FileNotFoundError(
            f"No configuration file found in {path}. "
            f"Expected one of: cfg.json, config.json, sae_cfg.json"
        )

    @classmethod
    def from_npz(cls, npz_path: str | Path, dir_path: str | Path | None = None) -> "SAEConfig":
        """
        Infer SAE configuration from npz file contents and directory path.

        This is used for Gemma-Scope style SAEs that don't have separate cfg.json files.
        Config is inferred from:
        - Array shapes in the npz file (for d_in, d_sae)
        - Directory structure (for layer number)

        Args:
            npz_path: Path to the params.npz file.
            dir_path: Directory containing the SAE (for extracting layer info from path).

        Returns:
            Inferred SAEConfig.

        Raises:
            FileNotFoundError: If npz file doesn't exist.
            ValueError: If required arrays are missing from npz file.
        """
        import numpy as np
        import re

        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        logger.debug(f"Inferring SAE config from npz: {npz_path}")

        # Load npz file
        data = np.load(npz_path)

        # Get dimensions from weight matrices
        # W_enc shape is typically (d_in, d_sae) or (d_sae, d_in)
        # W_dec shape is typically (d_sae, d_in) or (d_in, d_sae)
        if "W_enc" in data:
            W_enc = data["W_enc"]
            # Assume W_enc is (d_in, d_sae)
            d_in, d_sae = W_enc.shape
            # Verify with b_enc if available
            if "b_enc" in data:
                b_enc = data["b_enc"]
                if b_enc.shape[0] != d_sae:
                    # W_enc might be transposed, swap dimensions
                    d_in, d_sae = d_sae, d_in
        elif "W_dec" in data:
            W_dec = data["W_dec"]
            # Assume W_dec is (d_sae, d_in)
            d_sae, d_in = W_dec.shape
        else:
            raise ValueError(
                f"NPZ file missing weight matrices. Expected W_enc or W_dec. "
                f"Found keys: {list(data.keys())}"
            )

        # Extract layer from directory path
        hook_layer = 0
        path_str = str(dir_path or npz_path.parent)
        layer_match = re.search(r"layer[_-]?(\d+)", path_str, re.IGNORECASE)
        if layer_match:
            hook_layer = int(layer_match.group(1))

        # Try to extract model name from path
        model_name = "unknown"
        if "gemma-scope-2b" in path_str.lower():
            model_name = "google/gemma-2-2b"
        elif "gemma-scope-9b" in path_str.lower():
            model_name = "google/gemma-2-9b"
        elif "gemma" in path_str.lower():
            model_name = "gemma"

        # Determine dtype from array
        dtype_str = "float32"
        if "W_enc" in data:
            dtype_str = str(data["W_enc"].dtype)

        config = cls(
            d_in=int(d_in),
            d_sae=int(d_sae),
            model_name=model_name,
            hook_name=f"blocks.{hook_layer}.hook_resid_post",
            hook_layer=hook_layer,
            dtype=dtype_str,
            normalize_activations="none",
        )

        logger.info(f"Inferred SAE config from npz: {config}")
        return config

    @classmethod
    def _parse_config(cls, data: dict[str, Any]) -> "SAEConfig":
        """Parse config dictionary with field name variations."""
        # Handle d_in variations
        d_in = (
            data.get("d_in")
            or data.get("d_model")
            or data.get("input_dim")
            or data.get("activation_dim")
        )

        # Handle d_sae variations
        d_sae = (
            data.get("d_sae")
            or data.get("d_hidden")
            or data.get("hidden_dim")
            or data.get("dict_size")
            or data.get("num_features")
        )

        if d_in is None or d_sae is None:
            raise ValueError(
                f"Config missing required dimensions. "
                f"Expected d_in and d_sae (or variants). "
                f"Found keys: {list(data.keys())}"
            )

        # Handle hook_layer variations
        hook_layer = data.get("hook_layer")
        if hook_layer is None:
            # Try to extract from hook_name
            hook_name = data.get("hook_name", "")
            hook_layer = cls._extract_layer_from_hook_name(hook_name)

        return cls(
            d_in=int(d_in),
            d_sae=int(d_sae),
            model_name=data.get("model_name", "unknown"),
            hook_name=data.get("hook_name", ""),
            hook_layer=int(hook_layer) if hook_layer is not None else 0,
            dtype=data.get("dtype", "float32"),
            normalize_activations=data.get("normalize_activations", "none"),
        )

    @staticmethod
    def _extract_layer_from_hook_name(hook_name: str) -> int | None:
        """
        Extract layer index from hook name.

        Examples:
            "blocks.12.hook_resid_post" -> 12
            "transformer.h.5.mlp" -> 5
            "model.layers.8.self_attn" -> 8

        Returns:
            Extracted layer index or None if not found.
        """
        import re

        # Common patterns for layer numbers
        patterns = [
            r"blocks\.(\d+)\.",
            r"layers\.(\d+)\.",
            r"\.h\.(\d+)\.",
            r"layer\.(\d+)\.",
            r"layer(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, hook_name)
            if match:
                return int(match.group(1))

        return None

    def estimate_memory_mb(self) -> float:
        """
        Estimate SAE memory usage in megabytes.

        Memory calculation:
        - Encoder: d_in × d_sae (weights) + d_sae (bias)
        - Decoder: d_sae × d_in (weights) + d_in (bias)
        - Total params: 2 × d_in × d_sae + d_in + d_sae

        Returns:
            Estimated memory in MB.
        """
        bytes_per_param = self._dtype_to_bytes(self.dtype)

        # Weight matrices
        encoder_weights = self.d_in * self.d_sae
        decoder_weights = self.d_sae * self.d_in

        # Biases
        encoder_bias = self.d_sae
        decoder_bias = self.d_in

        total_params = encoder_weights + decoder_weights + encoder_bias + decoder_bias
        total_bytes = total_params * bytes_per_param

        return total_bytes / (1024 * 1024)

    @staticmethod
    def _dtype_to_bytes(dtype: str) -> int:
        """Convert dtype string to bytes per parameter."""
        dtype_bytes = {
            "float32": 4,
            "fp32": 4,
            "float16": 2,
            "fp16": 2,
            "bfloat16": 2,
            "bf16": 2,
            "float64": 8,
            "fp64": 8,
        }
        return dtype_bytes.get(dtype.lower(), 4)  # Default to float32

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "dtype": self.dtype,
            "normalize_activations": self.normalize_activations,
        }

    def __repr__(self) -> str:
        return (
            f"SAEConfig(d_in={self.d_in}, d_sae={self.d_sae}, "
            f"model={self.model_name}, layer={self.hook_layer})"
        )
