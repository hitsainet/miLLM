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
from millm.ml.memory_utils import get_available_cpu_memory_mb, get_available_memory_mb

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
    attn_implementation: str = "unknown"
    quantization_method: str = "unknown"  # "bitsandbytes", "gptq", "awq", "none"


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
                    # Move model to CPU first to release GPU tensors before deleting.
                    # bitsandbytes models don't support .to("cpu"), so we skip on error.
                    if self._loaded.model is not None:
                        try:
                            self._loaded.model.to("cpu")
                        except Exception:
                            pass
                        del self._loaded.model
                    if self._loaded.tokenizer is not None:
                        del self._loaded.tokenizer
                except Exception as e:
                    logger.warning("error_clearing_model", error=str(e))
                finally:
                    self._loaded = None

            try:
                import torch

                if torch.cuda.is_available():
                    # Log memory before cleanup
                    free_before, total = torch.cuda.mem_get_info()
                    used_before = (total - free_before) / (1024 * 1024)

                    # Ensure all async CUDA operations are complete
                    torch.cuda.synchronize()

                    # GC first: Python must release bitsandbytes objects (which hold
                    # raw CUDA allocations via cudaMalloc) before empty_cache can
                    # reclaim them.  Multiple passes handle circular references.
                    gc.collect()
                    gc.collect()

                    # Now release PyTorch's cached memory blocks
                    torch.cuda.empty_cache()

                    # Release any IPC handles
                    torch.cuda.ipc_collect()

                    # Final GC pass for anything freed by empty_cache
                    gc.collect()

                    # If significant memory still held, try resetting CUDA state.
                    # reset_peak_memory_stats is safe and clears internal bookkeeping.
                    torch.cuda.reset_peak_memory_stats()

                    free_after, _ = torch.cuda.mem_get_info()
                    used_after = (total - free_after) / (1024 * 1024)
                    logger.info(
                        "gpu_memory_cleanup",
                        used_before_mb=int(used_before),
                        used_after_mb=int(used_after),
                        freed_mb=int(used_before - used_after),
                    )
            except ImportError:
                gc.collect()


def _get_auto_model_class(config: Any) -> Any:
    """
    Determine the appropriate Auto model class based on model config.

    Inspects the config's architectures field to pick the right class.
    Falls back to AutoModelForCausalLM -> AutoModel.

    Args:
        config: HuggingFace model config object

    Returns:
        The appropriate Auto model class.
    """
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    # Check architectures field for seq2seq indicators
    architectures = getattr(config, "architectures", []) or []
    model_type = getattr(config, "model_type", "")

    seq2seq_indicators = [
        "ConditionalGeneration",
        "Seq2Seq",
        "EncoderDecoder",
        "ForConditionalGeneration",
    ]
    seq2seq_model_types = {"t5", "bart", "mbart", "pegasus", "marian", "blenderbot"}

    for arch in architectures:
        if any(indicator in arch for indicator in seq2seq_indicators):
            logger.info("auto_model_class_seq2seq", architecture=arch)
            return AutoModelForSeq2SeqLM

    if model_type.lower() in seq2seq_model_types:
        logger.info("auto_model_class_seq2seq_by_type", model_type=model_type)
        return AutoModelForSeq2SeqLM

    # Default: causal LM (GPT-style)
    return AutoModelForCausalLM


def _patch_granite_hybrid_mamba_mask(model: Any) -> None:
    """
    Patch GraniteMoEHybrid models so _update_mamba_mask tolerates attention-only caches.

    The model class unconditionally calls has_previous_state() on the cache, which
    raises ValueError when no LinearAttention layers exist. Some granite-4.0-micro
    configs have layers_block_type empty and layer_types all "attention", yet still
    route through GraniteMoeHybridForCausalLM — making this path unreachable-but-taken.

    We wrap _update_mamba_mask to catch the ValueError and fall back to the
    attention_mask, which is the correct behavior when the model has no mamba layers.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    if "granitemoehybrid" not in model_type.lower():
        return

    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "_update_mamba_mask"):
        return

    original = inner._update_mamba_mask

    def _safe_update_mamba_mask(attention_mask, past_key_values):
        try:
            return original(attention_mask, past_key_values)
        except ValueError:
            # No LinearAttention layers in cache — this model instance has no mamba
            # layers despite the hybrid class. Return the attention_mask unmodified.
            return attention_mask

    inner._update_mamba_mask = _safe_update_mamba_mask
    logger.info("patched_granite_hybrid_mamba_mask", model_type=model_type)


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
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gc.collect()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()
            except ImportError:
                gc.collect()

        return False  # Don't suppress exception

    def load(
        self,
        cache_path: str,
        quantization: str,
        trust_remote_code: bool = False,
        device: str = "cuda",
        torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
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
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

        # Validate that quantization requiring CUDA actually has CUDA available
        if quantization.upper() in ("Q4", "Q8", "Q2") and not torch.cuda.is_available():
            raise ModelLoadError(
                f"Quantization type {quantization} requires CUDA, but no GPU is available.",
                details={"quantization": quantization},
            )

        # Detect best attention implementation
        attn_impl = "sdpa"  # PyTorch native SDPA (default in transformers 4.36+)
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            logger.info("flash_attention_available", version=getattr(flash_attn, "__version__", "unknown"))
        except ImportError:
            logger.info("flash_attention_not_available_using_sdpa")

        # Detect if model is already pre-quantized (GPTQ/AWQ/BitNet/etc.)
        quant_method = "none"
        is_pre_quantized = False
        config = None
        try:
            config = AutoConfig.from_pretrained(cache_path, trust_remote_code=trust_remote_code)
            pre_quant_config = getattr(config, "quantization_config", None)
            if pre_quant_config is not None:
                if isinstance(pre_quant_config, dict):
                    quant_method = pre_quant_config.get("quant_method", "unknown")
                else:
                    quant_method = getattr(pre_quant_config, "quant_method", "unknown")
                # Any model with a native quantization config should not have
                # bitsandbytes applied on top
                is_pre_quantized = True
                logger.info("pre_quantized_model_detected", quant_method=quant_method)
        except Exception as e:
            # Also check config.json directly as fallback
            import os, json as _json
            config_path = os.path.join(cache_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        raw_config = _json.load(f)
                    if "quantization_config" in raw_config:
                        quant_method = raw_config["quantization_config"].get("quant_method", "unknown")
                        is_pre_quantized = True
                        logger.info("pre_quantized_detected_from_json", quant_method=quant_method)
                except Exception:
                    pass
            logger.warning("config_load_for_quant_detection_failed", error=str(e))

        # Configure quantization
        # Use bfloat16 instead of float16: same memory (2 bytes/param) but much larger
        # numeric range (max ~3.4e38 vs ~65504). Many modern models (Gemma 3, Llama 3, etc.)
        # are trained in bfloat16 and produce NaN/Inf logits when loaded in float16.
        quantization_config = None
        torch_dtype = torch.bfloat16

        if is_pre_quantized:
            # Model is already quantized (GPTQ/AWQ) — skip bitsandbytes
            logger.info("skipping_bnb_for_pre_quantized", quant_method=quant_method)
        elif quantization == "Q4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            quant_method = "bitsandbytes"
        elif quantization == "Q8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            quant_method = "bitsandbytes"

        # Load tokenizer first (small, quick)
        logger.debug("loading_tokenizer", model_id=self.model_id)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cache_path,
                trust_remote_code=trust_remote_code,
            )
        except (ImportError, ValueError, OSError) as e:
            # Fall back for models with custom tokenizer classes (e.g. LiquidAI
            # TokenizersBackend) — use PreTrainedTokenizerFast if tokenizer.json exists
            import os
            tokenizer_json = os.path.join(cache_path, "tokenizer.json")
            if os.path.exists(tokenizer_json):
                from transformers import PreTrainedTokenizerFast
                logger.warning(
                    "tokenizer_fallback_to_fast",
                    model_id=self.model_id,
                    error=str(e),
                )
                self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=tokenizer_json,
                )
                # Load special tokens from tokenizer_config.json if available
                import json as _json
                tokenizer_config_path = os.path.join(cache_path, "tokenizer_config.json")
                if os.path.exists(tokenizer_config_path):
                    try:
                        with open(tokenizer_config_path) as f:
                            tok_config = _json.load(f)
                        special_token_keys = [
                            "bos_token", "eos_token", "unk_token",
                            "pad_token", "sep_token", "cls_token",
                            "mask_token",
                        ]
                        special_tokens = {}
                        for key in special_token_keys:
                            val = tok_config.get(key)
                            if val is not None:
                                # Value can be a string or a dict with "content" key
                                if isinstance(val, dict):
                                    val = val.get("content", None)
                                if val is not None:
                                    special_tokens[key] = val
                        if special_tokens:
                            self.tokenizer.add_special_tokens(special_tokens)
                            logger.info(
                                "loaded_special_tokens_from_config",
                                tokens=list(special_tokens.keys()),
                            )
                    except Exception as tok_err:
                        logger.warning(
                            "failed_to_load_special_tokens",
                            error=str(tok_err),
                        )
            elif trust_remote_code:
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

        # Validate eos_token is set (critical for generation)
        if self.tokenizer.eos_token is None:
            logger.warning(
                "tokenizer_missing_eos_token",
                model_id=self.model_id,
                msg="eos_token is None after loading — generation may not terminate properly",
            )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (large, slow)
        logger.debug("loading_model_weights", model_id=self.model_id, attn_impl=attn_impl)

        # For bitsandbytes quantization, set max_memory to ensure large models
        # are quantized in CPU RAM and only quantized weights are placed on GPU.
        # This prevents OOM when the full FP16 model exceeds GPU VRAM.
        load_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch_dtype,
            "device_map": "auto" if device == "cuda" else None,
            "trust_remote_code": trust_remote_code,
            "attn_implementation": attn_impl,
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None and device == "cuda" and torch.cuda.is_available():
            # Use 90% of free GPU memory instead of total minus a fixed offset
            free_gpu, _ = torch.cuda.mem_get_info(0)
            max_gpu_bytes = int(free_gpu * 0.9)
            max_gpu = f"{max_gpu_bytes // (1024**3)}GiB"

            # Derive CPU memory dynamically (leave ~4GB headroom for OS)
            cpu_avail_mb = get_available_cpu_memory_mb()
            if cpu_avail_mb > 0:
                cpu_headroom_mb = 4096  # 4 GB for OS
                usable_cpu_mb = max(cpu_avail_mb - cpu_headroom_mb, 1024)
                max_cpu = f"{usable_cpu_mb // 1024}GiB"
            else:
                max_cpu = "64GiB"  # Fallback if detection fails
                logger.warning("cpu_memory_detection_failed_using_fallback", fallback=max_cpu)

            load_kwargs["max_memory"] = {0: max_gpu, "cpu": max_cpu}
            logger.info("quantized_load_memory_map", max_gpu=max_gpu, max_cpu=max_cpu)

        # Auto-detect the appropriate model class
        ModelClass = AutoModelForCausalLM  # default
        if config is not None:
            try:
                ModelClass = _get_auto_model_class(config)
                if ModelClass is not AutoModelForCausalLM:
                    logger.info(
                        "using_auto_model_class",
                        model_class=ModelClass.__name__,
                    )
            except Exception as e:
                logger.warning("auto_model_class_detection_failed", error=str(e))

        try:
            self.model = ModelClass.from_pretrained(
                cache_path,
                **load_kwargs,
            )
        except (ImportError, OSError) as e:
            if trust_remote_code:
                # Custom model code may reference missing .py files or removed
                # transformers internals. Fall back to built-in implementation
                # (e.g. BitNet auto_map references local .py but class is now
                # built into transformers).
                logger.warning(
                    "trust_remote_code_fallback",
                    model_id=self.model_id,
                    error=str(e),
                )
                load_kwargs["trust_remote_code"] = False
                self.model = ModelClass.from_pretrained(
                    cache_path,
                    **load_kwargs,
                )
            else:
                raise
        except Exception as e:
            # If the chosen ModelClass fails, try AutoModelForCausalLM as fallback,
            # and then AutoModel as a last resort
            if ModelClass is not AutoModelForCausalLM:
                logger.warning(
                    "model_class_fallback_to_causal_lm",
                    original_class=ModelClass.__name__,
                    error=str(e),
                )
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cache_path,
                        **load_kwargs,
                    )
                except Exception:
                    from transformers import AutoModel
                    logger.warning(
                        "model_class_fallback_to_auto_model",
                        error=str(e),
                    )
                    self.model = AutoModel.from_pretrained(
                        cache_path,
                        **load_kwargs,
                    )
            else:
                raise

        # Workaround for GraniteMoEHybrid models whose config has no mamba layers
        # (layers_block_type empty, all layer_types == "attention") but whose model
        # class still unconditionally calls _update_mamba_mask during forward().
        # This fails because has_previous_state() on the DynamicCache raises when
        # no LinearAttention layers exist. Patch _update_mamba_mask on the inner
        # model so it tolerates attention-only caches.
        _patch_granite_hybrid_mamba_mask(self.model)

        # Get memory usage using mem_get_info for accuracy (includes bitsandbytes allocations)
        memory_used_mb = 0
        if torch.cuda.is_available():
            try:
                free_after, total = torch.cuda.mem_get_info(0)
                memory_used_mb = int((total - free_after) / (1024 * 1024))
            except Exception:
                # Fallback to memory_allocated if mem_get_info fails
                memory_used_mb = int(torch.cuda.memory_allocated() / (1024 * 1024))

        # Get model properties
        num_parameters = 0
        try:
            num_parameters = self.model.num_parameters()
        except Exception:
            pass

        # Get device info — check hf_device_map first since device_map="auto"
        # models always have model.device == "cpu" (the dispatch device), which is
        # misleading. hf_device_map shows where layers actually live.
        device_str = "unknown"
        try:
            if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
                devices = set(str(d) for d in self.model.hf_device_map.values())
                device_str = ", ".join(sorted(devices)) if devices else "auto"
            elif hasattr(self.model, "device"):
                device_str = str(self.model.device)
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

        # Apply torch.compile for faster decoding (skip for bitsandbytes which is incompatible)
        if torch_compile and quant_method != "bitsandbytes":
            try:
                logger.info("torch_compile_starting", mode=torch_compile_mode)
                self.model.forward = torch.compile(
                    self.model.forward,
                    mode=torch_compile_mode,
                    fullgraph=False,  # Allow hooks and dynamic control flow
                )
                logger.info("torch_compile_complete", mode=torch_compile_mode)
            except Exception as e:
                logger.warning("torch_compile_failed_continuing_without", error=str(e))

        logger.info(
            "model_load_complete",
            model_id=self.model_id,
            memory_used_mb=memory_used_mb,
            num_parameters=num_parameters,
            device=device_str,
            dtype=dtype_str,
            attn_implementation=attn_impl,
            quantization_method=quant_method,
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
            attn_implementation=attn_impl,
            quantization_method=quant_method,
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
        torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        is_pre_quantized: bool = False,
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
            torch_compile: Whether to apply torch.compile to model.forward
            torch_compile_mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
            is_pre_quantized: Whether the model is already pre-quantized (GPTQ/AWQ/etc.)

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
                # Quantized models absolutely require CUDA
                if quantization.upper() in ("Q4", "Q8", "Q2"):
                    raise ModelLoadError(
                        f"CUDA is not available. GPU required for {quantization} quantization.",
                    )
                raise ModelLoadError(
                    "CUDA is not available. GPU required for model loading.",
                )
        except ImportError:
            raise ModelLoadError(
                "PyTorch is not installed. Install with CUDA support.",
            )

        # Check memory availability (skip for quantized/offloadable models that use CPU offloading)
        available_mb = get_available_memory_mb()
        skip_mem_check = is_pre_quantized or quantization.upper() in ("Q4", "Q2")
        if not skip_mem_check and available_mb < estimated_memory_mb:
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
                torch_compile=torch_compile,
                torch_compile_mode=torch_compile_mode,
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
