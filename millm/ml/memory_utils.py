"""
GPU memory utilities for miLLM.

Provides functions for estimating memory requirements and checking
available GPU memory for model loading.
"""

import re
from typing import Optional

import structlog

logger = structlog.get_logger()

# Bytes per parameter for different quantization types
BYTES_PER_PARAM = {
    "FP16": 2.0,
    "Q8": 1.0,
    "Q4": 0.5,
}

# Overhead factor for KV cache, activations, etc.
MEMORY_OVERHEAD_FACTOR = 1.2


def parse_params(params_str: Optional[str]) -> int:
    """
    Parse parameter string to number of parameters.

    Args:
        params_str: Parameter string (e.g., "2.5B", "350M", "7B")

    Returns:
        Number of parameters as integer, or 0 if parsing fails.

    Examples:
        >>> parse_params("2.5B")
        2500000000
        >>> parse_params("350M")
        350000000
        >>> parse_params("7B")
        7000000000
    """
    if not params_str or params_str.lower() == "unknown":
        return 0

    # Handle T (trillion) suffix as well
    match = re.match(r"^([\d.]+)\s*([TBMK]?)$", params_str.upper().strip())
    if not match:
        return 0

    try:
        value = float(match.group(1))
    except ValueError:
        return 0

    suffix = match.group(2)

    multipliers = {
        "T": 1e12,
        "B": 1e9,
        "M": 1e6,
        "K": 1e3,
        "": 1,
    }

    return int(value * multipliers.get(suffix, 1))


def estimate_memory_mb(params_str: Optional[str], quantization: str) -> int:
    """
    Estimate VRAM needed for loading a model.

    Formula:
    - FP16: params * 2 bytes
    - Q8: params * 1 byte
    - Q4: params * 0.5 bytes
    Plus ~20% overhead for KV cache, activations, and other runtime memory.

    Args:
        params_str: Parameter count string (e.g., "7B", "2.5B")
        quantization: Quantization type ("FP16", "Q8", "Q4")

    Returns:
        Estimated memory requirement in megabytes, or 0 if estimation fails.
    """
    params = parse_params(params_str)
    if params == 0:
        return 0

    bytes_per_param = BYTES_PER_PARAM.get(quantization.upper(), 2.0)
    base_bytes = params * bytes_per_param
    with_overhead = base_bytes * MEMORY_OVERHEAD_FACTOR

    return int(with_overhead / (1024 * 1024))


def get_available_memory_mb(device: int = 0) -> int:
    """
    Get available GPU memory in megabytes.

    Args:
        device: CUDA device index (default: 0)

    Returns:
        Available GPU memory in MB, or 0 if CUDA is not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0

        free, _ = torch.cuda.mem_get_info(device)
        return int(free / (1024 * 1024))

    except Exception as e:
        logger.warning("failed_to_get_gpu_memory", error=str(e))
        return 0


def get_total_memory_mb(device: int = 0) -> int:
    """
    Get total GPU memory in megabytes.

    Args:
        device: CUDA device index (default: 0)

    Returns:
        Total GPU memory in MB, or 0 if CUDA is not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0

        _, total = torch.cuda.mem_get_info(device)
        return int(total / (1024 * 1024))

    except Exception as e:
        logger.warning("failed_to_get_gpu_memory", error=str(e))
        return 0


def get_used_memory_mb(device: int = 0) -> int:
    """
    Get currently used GPU memory in megabytes.

    Args:
        device: CUDA device index (default: 0)

    Returns:
        Used GPU memory in MB, or 0 if CUDA is not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0

        return int(torch.cuda.memory_allocated(device) / (1024 * 1024))

    except Exception as e:
        logger.warning("failed_to_get_gpu_memory", error=str(e))
        return 0


def verify_memory_available(required_mb: int, device: int = 0) -> tuple[bool, int]:
    """
    Check if enough GPU memory is available.

    Args:
        required_mb: Required memory in megabytes
        device: CUDA device index (default: 0)

    Returns:
        Tuple of (is_available, available_mb)
    """
    available = get_available_memory_mb(device)
    return (available >= required_mb, available)


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def get_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        Number of CUDA devices, or 0 if CUDA is not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:
        return 0


def get_device_name(device: int = 0) -> str:
    """
    Get the name of a CUDA device.

    Args:
        device: CUDA device index (default: 0)

    Returns:
        Device name string, or "Unknown" if not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "No GPU"
        return torch.cuda.get_device_name(device)
    except Exception:
        return "Unknown"


def format_memory_mb(memory_mb: int) -> str:
    """
    Format memory in MB to a human-readable string.

    Args:
        memory_mb: Memory in megabytes

    Returns:
        Human-readable memory string (e.g., "8.0 GB", "512 MB")
    """
    if memory_mb >= 1024:
        return f"{memory_mb / 1024:.1f} GB"
    return f"{memory_mb} MB"
