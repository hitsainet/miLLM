"""
Model loader for miLLM.

Handles loading and unloading models from GPU memory with quantization support.
"""

import gc
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import structlog

from millm.core.errors import InsufficientMemoryError, ModelLoadError
from millm.ml.memory_utils import get_available_memory_mb

logger = structlog.get_logger()


@dataclass
class LoadedModel:
    """Represents a model loaded in GPU memory."""

    model_id: int
    model_name: str  # Human-readable model name (e.g., "gemma-2-2b")
    model: Any  # AutoModelForCausalLM
    tokenizer: Any  # AutoTokenizer
    loaded_at: datetime
    memory_used_mb: int = 0
    num_parameters: int = 0
    device: str = "unknown"
    dtype: str = "unknown"


class LoadedModelState:
    """
    Singleton managing the currently loaded model.

    Thread-safe for access from executor threads.
    Only one model can be loaded at a time to manage GPU memory.
    """

    _instance: Optional["LoadedModelState"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "LoadedModelState":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loaded: Optional[LoadedModel] = None
        return cls._instance

    @property
    def current(self) -> Optional[LoadedModel]:
        """Get the currently loaded model."""
        return self._loaded

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._loaded is not None

    @property
    def loaded_model_id(self) -> Optional[int]:
        """Get the ID of the currently loaded model."""
        return self._loaded.model_id if self._loaded else None

    def set(self, model: LoadedModel) -> None:
        """Set the currently loaded model."""
        with self._lock:
            self._loaded = model

    def clear(self) -> None:
        """Clear the currently loaded model and free GPU memory."""
        with self._lock:
            if self._loaded:
                try:
                    # Delete model and tokenizer references
                    if self._loaded.model is not None:
                        del self._loaded.model
                    if self._loaded.tokenizer is not None:
                        del self._loaded.tokenizer
                except Exception as e:
                    logger.warning("error_clearing_model", error=str(e))
                finally:
                    self._loaded = None

            # Force GPU memory release
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            # Force Python garbage collection
            gc.collect()


class ModelLoadContext:
    """
    Context manager for safe model loading.

    Ensures cleanup on any failure during the loading process.
    """

    def __init__(self, model_id: int, model_name: str) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.model: Any = None
        self.tokenizer: Any = None

    def __enter__(self) -> "ModelLoadContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None:
            logger.error(
                "model_load_failed",
                model_id=self.model_id,
                error=str(exc_val),
            )
            # Clean up on failure
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            gc.collect()

        return False  # Don't suppress exception

    def load(
        self,
        cache_path: str,
        quantization: str,
        trust_remote_code: bool = False,
        device: str = "cuda",
    ) -> LoadedModel:
        """
        Load model with quantization config.

        Args:
            cache_path: Path to the cached model files
            quantization: Quantization type ("FP16", "Q8", "Q4")
            trust_remote_code: Whether to trust remote code
            device: Device to load model on

        Returns:
            LoadedModel instance
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise ModelLoadError(
                "Required packages not installed. Install torch and transformers.",
                details={"missing_package": str(e)},
            )

        logger.info(
            "model_load_started",
            model_id=self.model_id,
            cache_path=cache_path,
            quantization=quantization,
        )

        # Configure quantization
        # Use bfloat16 instead of float16: same memory (2 bytes/param) but much larger
        # numeric range (max ~3.4e38 vs ~65504). Many modern models (Gemma 3, Llama 3, etc.)
        # are trained in bfloat16 and produce NaN/Inf logits when loaded in float16.
        quantization_config = None
        torch_dtype = torch.bfloat16

        if quantization == "Q4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "Q8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer first (small, quick)
        logger.debug("loading_tokenizer", model_id=self.model_id)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cache_path,
                trust_remote_code=trust_remote_code,
            )
        except ImportError as e:
            if trust_remote_code:
                logger.warning(
                    "tokenizer_trust_remote_code_fallback",
                    model_id=self.model_id,
                    error=str(e),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    cache_path,
                    trust_remote_code=False,
                )
            else:
                raise

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (large, slow)
        logger.debug("loading_model_weights", model_id=self.model_id)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                cache_path,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=trust_remote_code,
            )
        except ImportError as e:
            if trust_remote_code:
                # Custom model code may reference removed transformers internals.
                # Fall back to built-in transformers implementation.
                logger.warning(
                    "trust_remote_code_fallback",
                    model_id=self.model_id,
                    error=str(e),
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    cache_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=False,
                )
            else:
                raise

        # Get memory usage
        memory_used_mb = 0
        if torch.cuda.is_available():
            memory_used_mb = int(torch.cuda.memory_allocated() / (1024 * 1024))

        # Get model properties
        num_parameters = 0
        try:
            num_parameters = self.model.num_parameters()
        except Exception:
            pass

        # Get device info
        device_str = "unknown"
        try:
            if hasattr(self.model, "device"):
                device_str = str(self.model.device)
            elif hasattr(self.model, "hf_device_map"):
                devices = set(self.model.hf_device_map.values())
                device_str = ", ".join(str(d) for d in devices) if devices else "auto"
        except Exception:
            pass

        # Get dtype info
        dtype_str = "unknown"
        try:
            if hasattr(self.model, "dtype"):
                dtype_str = str(self.model.dtype).replace("torch.", "")
            elif hasattr(self.model, "config") and hasattr(self.model.config, "torch_dtype"):
                dtype_str = str(self.model.config.torch_dtype).replace("torch.", "")
        except Exception:
            pass

        logger.info(
            "model_load_complete",
            model_id=self.model_id,
            memory_used_mb=memory_used_mb,
            num_parameters=num_parameters,
            device=device_str,
            dtype=dtype_str,
        )

        return LoadedModel(
            model_id=self.model_id,
            model_name=self.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
            loaded_at=datetime.utcnow(),
            memory_used_mb=memory_used_mb,
            num_parameters=num_parameters,
            device=device_str,
            dtype=dtype_str,
        )


class ModelLoader:
    """
    High-level model loading operations.

    Manages the lifecycle of loading and unloading models,
    including memory verification and cleanup.
    """

    def __init__(self) -> None:
        self.state = LoadedModelState()

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.state.is_loaded

    @property
    def loaded_model_id(self) -> Optional[int]:
        """Get the ID of the currently loaded model."""
        return self.state.loaded_model_id

    @property
    def current_model(self) -> Optional[LoadedModel]:
        """Get the currently loaded model."""
        return self.state.current

    def load(
        self,
        model_id: int,
        model_name: str,
        cache_path: str,
        quantization: str,
        estimated_memory_mb: int,
        trust_remote_code: bool = False,
    ) -> LoadedModel:
        """
        Load a model into GPU memory.

        Verifies memory availability before loading.
        If another model is loaded, it should be unloaded first.

        Args:
            model_id: Database ID of the model
            model_name: Human-readable model name (e.g., "gemma-2-2b")
            cache_path: Path to the cached model files
            quantization: Quantization type ("FP16", "Q8", "Q4")
            estimated_memory_mb: Estimated memory requirement in MB
            trust_remote_code: Whether to trust remote code

        Returns:
            LoadedModel instance

        Raises:
            InsufficientMemoryError: If not enough GPU memory
            ModelLoadError: If loading fails
        """
        # Check if CUDA is available
        try:
            import torch

            if not torch.cuda.is_available():
                raise ModelLoadError(
                    "CUDA is not available. GPU required for model loading.",
                )
        except ImportError:
            raise ModelLoadError(
                "PyTorch is not installed. Install with CUDA support.",
            )

        # Check memory availability
        available_mb = get_available_memory_mb()
        if available_mb < estimated_memory_mb:
            raise InsufficientMemoryError(
                f"Not enough GPU memory. Need ~{estimated_memory_mb}MB, have {available_mb}MB",
                details={
                    "required_mb": estimated_memory_mb,
                    "available_mb": available_mb,
                },
            )

        # Load with context manager for cleanup on failure
        with ModelLoadContext(model_id, model_name) as ctx:
            loaded = ctx.load(
                cache_path=cache_path,
                quantization=quantization,
                trust_remote_code=trust_remote_code,
            )
            self.state.set(loaded)
            return loaded

    def unload(self) -> bool:
        """
        Unload current model and free GPU memory.

        Returns:
            True if a model was unloaded, False if no model was loaded.
        """
        if not self.state.is_loaded:
            return False

        model_id = self.state.loaded_model_id
        logger.info("unloading_model", model_id=model_id)

        self.state.clear()

        logger.info("model_unloaded", model_id=model_id)
        return True

    def get_memory_usage(self) -> int:
        """
        Get current GPU memory usage by the loaded model.

        Returns:
            Memory usage in MB, or 0 if no model is loaded.
        """
        if self.state.current:
            return self.state.current.memory_used_mb
        return 0
