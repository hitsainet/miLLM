"""
SAE weight loading.

Loads SAE weights from SafeTensors or NPZ files.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_file

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE

logger = logging.getLogger(__name__)


class SAELoader:
    """
    Loads SAE weights from disk.

    Supports:
    - SAELens SafeTensors format with cfg.json config
    - Gemma-Scope NPZ format (config inferred from array shapes)

    Usage:
        loader = SAELoader()
        config = loader.load_config("/path/to/sae")
        sae = loader.load("/path/to/sae", device="cuda")
    """

    def load_config(self, cache_path: str | Path) -> SAEConfig:
        """
        Load SAE configuration from cache path.

        Tries to load from cfg.json first, then falls back to inferring
        from NPZ file if no config file is found.

        Args:
            cache_path: Path to SAE directory.

        Returns:
            Parsed SAEConfig.
        """
        path = Path(cache_path)

        # Try loading from cfg.json first
        try:
            return SAEConfig.from_json(cache_path)
        except FileNotFoundError:
            pass

        # Fall back to inferring config from npz file
        npz_path = self._find_npz_file(path)
        if npz_path:
            logger.info(f"No cfg.json found, inferring config from {npz_path}")
            return SAEConfig.from_npz(npz_path, dir_path=path)

        # No config source found
        raise FileNotFoundError(
            f"No configuration source found in {path}. "
            f"Expected cfg.json or params.npz file."
        )

    def _find_npz_file(self, path: Path) -> Path | None:
        """Find NPZ weights file in directory."""
        if not path.is_dir():
            return None

        # Check common names
        common_names = ["params.npz", "weights.npz", "sae.npz"]
        for name in common_names:
            npz_path = path / name
            if npz_path.exists():
                return npz_path

        # Look for any .npz file
        npz_files = list(path.glob("*.npz"))
        if npz_files:
            return npz_files[0]

        return None

    def load(
        self,
        cache_path: str | Path,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> LoadedSAE:
        """
        Load SAE weights and create wrapper.

        Supports both SafeTensors (.safetensors) and NPZ (.npz) formats.

        Args:
            cache_path: Path to downloaded SAE directory.
            device: Target device (cuda/cpu).
            dtype: Override weight dtype (None = use config dtype).

        Returns:
            LoadedSAE wrapper ready for use.

        Raises:
            FileNotFoundError: If config or weights file not found.
            KeyError: If required weight tensors not found.
        """
        path = Path(cache_path)

        # Load config
        config = self.load_config(cache_path)

        # Try to find weights file (safetensors or npz)
        weights_path = self._find_weights_file(path)

        logger.info(f"Loading SAE from {weights_path}")

        # Load weights based on file type
        if weights_path.suffix == ".npz":
            W_enc, b_enc, W_dec, b_dec = self._load_npz_weights(weights_path)
        else:
            W_enc, b_enc, W_dec, b_dec = self._load_safetensors_weights(weights_path)

        # Convert dtype if specified
        target_dtype = dtype or self._str_to_dtype(config.dtype)

        W_enc = W_enc.to(target_dtype)
        b_enc = b_enc.to(target_dtype)
        W_dec = W_dec.to(target_dtype)
        b_dec = b_dec.to(target_dtype)

        # Validate shapes
        self._validate_shapes(W_enc, b_enc, W_dec, b_dec, config)

        # Create wrapper
        loaded_sae = LoadedSAE(
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=b_dec,
            config=config,
            device=device,
        )

        logger.info(
            f"Loaded SAE: d_in={config.d_in}, d_sae={config.d_sae}, "
            f"memory={loaded_sae.estimate_memory_mb():.1f}MB, device={device}"
        )

        return loaded_sae

    def _load_safetensors_weights(
        self, weights_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load weights from SafeTensors file."""
        state_dict = load_file(weights_path)

        W_enc = self._get_tensor(
            state_dict, ["W_enc", "encoder.weight", "W_e", "w_enc"]
        )
        b_enc = self._get_tensor(
            state_dict, ["b_enc", "encoder.bias", "b_e", "bias_enc"]
        )
        W_dec = self._get_tensor(
            state_dict, ["W_dec", "decoder.weight", "W_d", "w_dec"]
        )
        b_dec = self._get_tensor(
            state_dict, ["b_dec", "decoder.bias", "b_d", "bias_dec"]
        )

        return W_enc, b_enc, W_dec, b_dec

    def _load_npz_weights(
        self, weights_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load weights from NPZ file (Gemma-Scope format)."""
        data = np.load(weights_path)

        # Convert numpy arrays to torch tensors
        W_enc = torch.from_numpy(data["W_enc"].copy())
        b_enc = torch.from_numpy(data["b_enc"].copy())
        W_dec = torch.from_numpy(data["W_dec"].copy())
        b_dec = torch.from_numpy(data["b_dec"].copy())

        return W_enc, b_enc, W_dec, b_dec

    def _find_weights_file(self, path: Path) -> Path:
        """
        Find SAE weights file in directory.

        Supports both SafeTensors (.safetensors) and NPZ (.npz) formats.

        Args:
            path: Directory to search.

        Returns:
            Path to weights file.

        Raises:
            FileNotFoundError: If no weights file found.
        """
        # Check common safetensors names first
        safetensors_names = [
            "sae_weights.safetensors",
            "model.safetensors",
            "weights.safetensors",
            "sae.safetensors",
        ]

        for name in safetensors_names:
            weights_path = path / name
            if weights_path.exists():
                return weights_path

        # Look for any .safetensors file
        safetensors_files = list(path.glob("*.safetensors"))
        if safetensors_files:
            # Prefer files with "sae" or "weight" in name
            for f in safetensors_files:
                if "sae" in f.name.lower() or "weight" in f.name.lower():
                    return f
            return safetensors_files[0]

        # Check common npz names (Gemma-Scope format)
        npz_names = ["params.npz", "weights.npz", "sae.npz"]

        for name in npz_names:
            weights_path = path / name
            if weights_path.exists():
                return weights_path

        # Look for any .npz file
        npz_files = list(path.glob("*.npz"))
        if npz_files:
            return npz_files[0]

        raise FileNotFoundError(
            f"No weights file found in {path}. "
            f"Expected .safetensors or .npz file (e.g., sae_weights.safetensors, params.npz)."
        )

    def _get_tensor(self, state_dict: dict, names: list[str]) -> torch.Tensor:
        """
        Get tensor from state dict, trying multiple names.

        Args:
            state_dict: Loaded state dictionary.
            names: List of possible tensor names to try.

        Returns:
            The tensor.

        Raises:
            KeyError: If tensor not found under any name.
        """
        for name in names:
            if name in state_dict:
                return state_dict[name]

        raise KeyError(
            f"Could not find tensor. Tried: {names}. "
            f"Available keys: {list(state_dict.keys())}"
        )

    def _validate_shapes(
        self,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        config: SAEConfig,
    ) -> None:
        """
        Validate weight tensor shapes against config.

        Raises:
            ValueError: If shapes don't match config.
        """
        # W_enc should be (d_in, d_sae)
        if W_enc.shape != (config.d_in, config.d_sae):
            # Some SAEs have transposed weights
            if W_enc.shape == (config.d_sae, config.d_in):
                logger.warning(
                    "W_enc appears transposed, shape will be validated after transpose"
                )
            else:
                raise ValueError(
                    f"W_enc shape mismatch: expected ({config.d_in}, {config.d_sae}), "
                    f"got {tuple(W_enc.shape)}"
                )

        # b_enc should be (d_sae,)
        if b_enc.shape[0] != config.d_sae:
            raise ValueError(
                f"b_enc shape mismatch: expected ({config.d_sae},), "
                f"got {tuple(b_enc.shape)}"
            )

        # W_dec should be (d_sae, d_in)
        if W_dec.shape != (config.d_sae, config.d_in):
            if W_dec.shape == (config.d_in, config.d_sae):
                logger.warning(
                    "W_dec appears transposed, shape will be validated after transpose"
                )
            else:
                raise ValueError(
                    f"W_dec shape mismatch: expected ({config.d_sae}, {config.d_in}), "
                    f"got {tuple(W_dec.shape)}"
                )

        # b_dec should be (d_in,)
        if b_dec.shape[0] != config.d_in:
            raise ValueError(
                f"b_dec shape mismatch: expected ({config.d_in},), "
                f"got {tuple(b_dec.shape)}"
            )

    def _str_to_dtype(self, dtype_str: str) -> torch.dtype:
        """
        Convert string dtype to torch dtype.

        Args:
            dtype_str: String dtype (e.g., "float32", "fp16").

        Returns:
            Corresponding torch.dtype.
        """
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float64": torch.float64,
            "fp64": torch.float64,
        }
        return mapping.get(dtype_str.lower(), torch.float32)
