"""
Machine Learning module for miServer.

Contains model loading, downloading, and memory management utilities.
Also includes SAE (Sparse Autoencoder) components for interpretability.
"""

from miserver.ml.memory_utils import (
    estimate_memory_mb,
    get_available_memory_mb,
    get_total_memory_mb,
    get_used_memory_mb,
    parse_params,
    verify_memory_available,
)
from miserver.ml.model_downloader import ModelDownloader
from miserver.ml.model_loader import (
    LoadedModel,
    LoadedModelState,
    ModelLoadContext,
    ModelLoader,
)
from miserver.ml.sae_config import SAEConfig
from miserver.ml.sae_downloader import SAEDownloader
from miserver.ml.sae_hooker import SAEHooker
from miserver.ml.sae_loader import SAELoader
from miserver.ml.sae_wrapper import LoadedSAE

__all__ = [
    # Model operations
    "ModelDownloader",
    "ModelLoader",
    "LoadedModel",
    "LoadedModelState",
    "ModelLoadContext",
    # SAE operations
    "SAEConfig",
    "SAEDownloader",
    "SAEHooker",
    "SAELoader",
    "LoadedSAE",
    # Memory utilities
    "estimate_memory_mb",
    "get_available_memory_mb",
    "get_total_memory_mb",
    "get_used_memory_mb",
    "parse_params",
    "verify_memory_available",
]
