"""
Model loader for miLLM.

Handles loading and unloading models from GPU memory with quantization support.
"""

import gc
import math
import threading
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

import structlog
import torch

# Defined BEFORE the optional imports below, because their failure handler logs.
# Without this the warning raises NameError inside an except block — turning a
# degraded feature into a failed module import, which is the exact outcome the
# handler exists to prevent.
logger = structlog.get_logger()

# Module-level imports allow @patch("millm.ml.model_loader.AutoTokenizer") etc.
# in tests. The actual load path also imports these inside the function body
# to preserve the informative ImportError message when transformers is absent.
try:
    from transformers import (  # noqa: F401  (re-exported for patching)
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError:
    AutoConfig = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    BitsAndBytesConfig = None  # type: ignore[assignment]

# Module-level so tests can @patch("millm.ml.model_loader.Llama"). Optional at
# import time on purpose: llama-cpp-python is an EXTRA, and a deployment that
# serves no GGUF must not fail to start because it is absent. The load path
# raises a clear error instead.
try:
    import llama_cpp as llama_cpp_module  # noqa: F401  (for pooling constants)
    from llama_cpp import Llama  # noqa: F401  (re-exported for patching)
except Exception as _llama_import_error:  # noqa: BLE001
    # NOT `except ImportError`. llama-cpp-python raises RUNTIMEERROR when its
    # bundled shared library will not load — verified in the running pod:
    #
    #   RuntimeError: Failed to load shared library '.../libllama.so':
    #   libcudart.so.13: cannot open shared object file
    #
    # In this image it resolves only because `import torch` above happens first
    # and pulls torch's bundled CUDA runtime into the process. That ordering is
    # an accident, not a contract. With ImportError alone, any environment
    # where it does not hold — a CPU-only container, a reordered import, a
    # mismatched wheel — makes THIS MODULE fail to import, and model_loader is
    # imported at startup, so the entire backend dies rather than one feature
    # being unavailable.
    Llama = None  # type: ignore[assignment]
    llama_cpp_module = None  # type: ignore[assignment]
    logger.warning(
        "llama_cpp_unavailable",
        error=str(_llama_import_error),
        error_type=type(_llama_import_error).__name__,
        detail="GGUF models will refuse to load; everything else serves normally",
    )

# Whether the installed llama.cpp WHEEL was built with GPU offload at all.
# A SEPARATE try on purpose: folding it into the block above would make a
# missing symbol set `Llama = None` and disable GGUF serving entirely, when the
# only thing lost is the ability to say which device the weights landed on.
try:
    from llama_cpp import llama_supports_gpu_offload  # noqa: F401
except Exception as _offload_probe_error:  # noqa: BLE001 - optional symbol
    # WARNS, does not merely pass. Without the probe `_gguf_device()` falls
    # back to `torch.cuda.is_available()` alone, which reports "cuda" on a
    # CUDA box running a CPU-only wheel — precisely the wrong answer that
    # function exists to prevent. A silent fallback there would have
    # /api/models/status assert a placement that never happened with nothing
    # anywhere saying the measurement was unavailable.
    llama_supports_gpu_offload = None  # type: ignore[assignment]
    logger.warning(
        "llama_cpp_offload_probe_unavailable",
        error=str(_offload_probe_error),
        error_type=type(_offload_probe_error).__name__,
        detail=(
            "GGUF device reporting falls back to torch.cuda.is_available(), "
            "which cannot see whether the llama.cpp build supports offload"
        ),
    )

from millm.core.config import parse_gguf_tensor_split
from millm.core.errors import (
    GgufTensorSplitError,
    InsufficientMemoryError,
    ModelLoadError,
    SplitNotHonouredError,
    UnsupportedQuantizationError,
)
from millm.ml.gguf_catalog import quant_label_from_path
from millm.ml.memory_utils import MEMORY_OVERHEAD_FACTOR
from millm.ml.gpu_placement import (
    ALL,
    BNB_MAX_MEMORY_FACTOR,
    MODE_CPU,
    MODE_SINGLE,
    OFF_GPU_LABELS,
    REASON_MOST_FREE,
    REASON_NO_GPU,
    REASON_NO_SINGLE_CARD,
    REASON_REQUESTED,
    REASON_REQUESTED_ALL,
    GpuInfo,
    GpuRequest,
    Placement,
    ShardRule,
    choose_gpu,
    cpu_placement,
    find_gpu,
    free_mb_by_index,
    gpu_indices_of,
    list_gpus,
    memory_used_by_device,
    model_device_labels,
    model_input_device,
    parse_gpu_request,
    plan_shard,
    refuse_cards_left_out_of_all,
    reported_free_mb_by_index,
    shard_refusal,
    transformers_shard_rule,
)


#: The transformers engine: a torch nn.Module tree, hookable, differentiable.
ENGINE_TRANSFORMERS = "transformers"
#: llama.cpp via llama-cpp-python: a ctypes handle onto a C++ graph. NO module
#: tree, so no forward hooks — every interpretability feature is structurally
#: impossible on it, not merely unimplemented.
ENGINE_LLAMACPP = "llamacpp"


@dataclass
class LoadedModel:
    """Represents a model loaded in GPU memory."""

    model_id: int
    model_name: str  # Human-readable model name (e.g., "gemma-2-2b")
    model: Any  # AutoModelForCausalLM, or llama_cpp.Llama
    tokenizer: Any  # AutoTokenizer, or None for llama.cpp (it tokenizes itself)
    loaded_at: datetime
    memory_used_mb: int = 0
    num_parameters: int = 0
    device: str = "unknown"
    dtype: str = "unknown"
    attn_implementation: str = "unknown"
    quantization_method: str = "unknown"  # "bitsandbytes", "gptq", "awq", "none"
    #: The context window this model was ACTUALLY loaded with.
    #:
    #: Recorded because it is not always the one that was asked for: a context
    #: that does not fit is retried smaller, and serving a 2048 window while the
    #: configuration says 8192 would truncate long prompts for reasons nothing
    #: on the system explains. 0 means "the model's full declared context".
    context_length: int = 0
    #: Whether THIS instance can serve /v1/embeddings.
    #:
    #: Not a config switch — a fact about the load. llama.cpp takes
    #: `embedding`/`pooling_type` at CONSTRUCTION only, and some architectures
    #: refuse MEAN pooling outright, so the model is loaded without it rather
    #: than not at all. A caller must be told which it got; discovering it from
    #: a confusing runtime failure is how the capability looks broken instead of
    #: absent.
    supports_embeddings: bool = False
    #: WHICH runtime holds this model. Defaults to transformers so every
    #: existing construction site keeps its meaning. Consumers branch on this
    #: rather than sniffing the object, because a duck-typed check would quietly
    #: pick the wrong path the day llama.cpp grows a `.config`.
    engine: str = ENGINE_TRANSFORMERS
    #: The CUDA indices this model actually holds memory on. Cleanup, KV-cache
    #: sizing and the placement report all read this rather than assuming GPU 0.
    #: Empty for a model with nothing on a card.
    gpu_indices: list[int] = field(default_factory=list)
    #: Memory the load consumed on each card ("cuda:N" -> MB), measured as the
    #: drop in free memory across the load.
    memory_by_device_mb: dict[str, int] = field(default_factory=dict)
    #: How the placement was decided (mode, reason, requested card, devices).
    placement: Optional[dict[str, Any]] = None

    @property
    def supports_hooks(self) -> bool:
        """Whether SAE attachment, steering and sensing are possible at all.

        Not a policy switch — a statement of fact about the runtime. llama.cpp
        exposes no `nn.Module`, so `register_forward_hook` has nothing to attach
        to and no per-layer residual tensor is reachable from Python.
        """
        return self.engine == ENGINE_TRANSFORMERS


def _release_cuda_memory(gpu_indices: list[int]) -> None:
    """Synchronize, collect and release cached memory on the cards a model used.

    This read and synchronized GPU 0 only. With a model on the 3090 (index 1)
    that waited on the wrong card's stream before `empty_cache`, and logged the
    3080 Ti's memory as what the unload freed. An empty list means no card was
    used, and no card is touched.

    `empty_cache` and `ipc_collect` act on every device's allocator already;
    `synchronize`, `reset_peak_memory_stats` and `mem_get_info` act on one
    device, so they take the index.
    """
    indices = sorted(set(gpu_indices))
    if not indices:
        # Nothing torch placed on any card. Touching a card here would create a
        # CUDA context on it just to clean up nothing, and that context's
        # memory is what other tenants of the node place against.
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return
    before: dict[int, tuple[int, int]] = {}
    for index in indices:
        before[index] = torch.cuda.mem_get_info(index)

    for index in indices:
        # Ensure all async CUDA operations on this card are complete.
        torch.cuda.synchronize(index)

    # GC first: Python must release bitsandbytes objects (which hold raw CUDA
    # allocations via cudaMalloc) before empty_cache can reclaim them.
    # Multiple passes handle circular references.
    gc.collect()
    gc.collect()

    # Now release PyTorch's cached memory blocks (every device's allocator).
    torch.cuda.empty_cache()

    # Release any IPC handles
    torch.cuda.ipc_collect()

    # Final GC pass for anything freed by empty_cache
    gc.collect()

    for index in indices:
        # reset_peak_memory_stats is safe and clears internal bookkeeping.
        torch.cuda.reset_peak_memory_stats(index)

    for index in indices:
        free_before, total = before[index]
        free_after, _ = torch.cuda.mem_get_info(index)
        used_before = (total - free_before) / (1024 * 1024)
        used_after = (total - free_after) / (1024 * 1024)
        logger.info(
            "gpu_memory_cleanup",
            device=f"cuda:{index}",
            used_before_mb=int(used_before),
            used_after_mb=int(used_after),
            freed_mb=int(used_before - used_after),
        )


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
            # Read before the slot is emptied: cleanup must act on the cards
            # this model used, and the record of which those were goes with it.
            # llama.cpp's memory is not torch's: torch calls on its cards would
            # only create a torch context there, so a GGUF model records none.
            gpu_indices = (
                list(self._loaded.gpu_indices)
                if self._loaded and self._loaded.engine == ENGINE_TRANSFORMERS
                else []
            )
            if self._loaded:
                try:
                    # Move model to CPU first to release GPU tensors before deleting.
                    # bitsandbytes models don't support .to("cpu"), so we skip on error.
                    if self._loaded.model is not None:
                        # llama.cpp holds its weights in a C++ context that
                        # Python's garbage collector cannot reach. It must be
                        # CLOSED explicitly, and nothing below would do it:
                        # `.to("cpu")` does not exist on a Llama (the except
                        # swallows it), `del` drops only the handle, and
                        # torch.cuda.empty_cache() knows nothing about an
                        # allocation torch never made. On a single shared 24 GB
                        # card a load/unload cycle that silently retains VRAM is
                        # the failure that takes the node down.
                        close = getattr(self._loaded.model, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception as e:  # noqa: BLE001
                                logger.warning("engine_close_failed", error=str(e))
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
                if torch.cuda.is_available():
                    _release_cuda_memory(gpu_indices)
            except ImportError:
                gc.collect()


def _is_offload_refusal(exc: BaseException) -> bool:
    """Whether transformers refused a bitsandbytes device map that leaves the GPU.

    Matched on its message because it is a plain ValueError
    (`quantizer_bnb_{4,8}bit.validate_environment`, transformers 5.15.1). It is
    raised while the device map is computed, before any weight is read.
    """
    return isinstance(exc, ValueError) and "dispatched on the CPU or the disk" in str(exc)


def checkpoint_quantization_config(cache_path: Optional[str]) -> Optional[dict[str, Any]]:
    """A checkpoint's own `quantization_config`, when its config.json holds one as an object.

    None for a missing or unreadable config.json, a null value, or anything that
    is not an object — the same reading as ModelLoadContext.load's config.json
    fallback.
    """
    if not cache_path:
        return None
    import json
    import os

    try:
        with open(os.path.join(cache_path, "config.json")) as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("quantization_config")
    return value if isinstance(value, dict) else None


def checkpoint_is_pre_quantized(cache_path: Optional[str]) -> bool:
    """Whether a checkpoint ships its own quantization (GPTQ, AWQ, BitNet, ...).

    Read from its config.json: a `quantization_config` OBJECT. This is what
    ModelLoadContext.load reads before it decides against bitsandbytes — its
    AutoConfig reading, and its config.json fallback, which treats a null or
    non-object value as not quantized — and the placement decision must read the
    SAME thing. Until review round 1
    (2026-09-14) no caller passed `is_pre_quantized` to it at all, so a GPTQ or
    AWQ checkpoint on a Q4 or Q8 row was planned with bitsandbytes' 0.9, which
    transformers never applies to it, and a split that fits was refused — before
    the unload and again at load.

    A path with no readable config.json is not pre-quantized: the load would
    apply the row's quantization to it.
    """
    return checkpoint_quantization_config(cache_path) is not None


def pre_quantized_max_memory_factor(quantization_config: Optional[dict[str, Any]]) -> float:
    """What transformers multiplies `max_memory` by to load a checkpoint with this quantization.

    Asked of the quantizer class transformers itself picks for the config
    (`AUTO_QUANTIZER_MAPPING`, keyed the way `AutoQuantizationConfig.from_dict`
    keys it). In transformers 5.15.1 the bitsandbytes, BitNet, torchao and quanto
    quantizers take 0.9 (`adjust_max_memory`); AWQ, GPTQ, FP8 and the rest leave
    it whole.

    Review round 1 planned EVERY pre-quantized checkpoint at 1.0, on the belief
    that only miLLM's own Q4/Q8 load meets bitsandbytes. A checkpoint uploaded
    already quantized by bitsandbytes (the `-bnb-4bit` repos) or BitNet gets the
    same quantizer class from its own config, so the plan promised 10% more than
    transformers would place, and a split it accepted could map to disk. Review
    round 2, 2026-09-14.

    The class is asked WITHOUT being constructed: constructing one checks for its
    kernel package (GPTQ needs optimum), which says nothing about memory. A
    method transformers does not know gets no quantizer and loads unquantized:
    1.0. A class that cannot answer is planned at bitsandbytes' 0.9 — refusing
    early is the cheaper mistake.
    """
    if not quantization_config:
        return 1.0
    try:
        from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
    except ImportError:
        return 1.0
    method = quantization_config.get("quant_method")
    if quantization_config.get("load_in_8bit") or quantization_config.get("load_in_4bit"):
        method = (
            "bitsandbytes_4bit" if quantization_config.get("load_in_4bit") else "bitsandbytes_8bit"
        )
    quantizer_class = AUTO_QUANTIZER_MAPPING.get(method) if isinstance(method, str) else None
    if quantizer_class is None:
        return 1.0
    probe = 1_000_000
    try:
        adjusted = quantizer_class.adjust_max_memory(object.__new__(quantizer_class), {0: probe})
        return float(adjusted[0]) / probe
    except Exception as e:  # noqa: BLE001 - a quantizer that needs its own state
        logger.warning(
            "pre_quantized_max_memory_factor_unknown", quant_method=method, error=str(e)[:200]
        )
        return BNB_MAX_MEMORY_FACTOR


def checkpoint_weights_mb(cache_path: Optional[str]) -> int:
    """The weights a checkpoint stores, in MB; 0 when none are found.

    From the safetensors (or .bin) index's `total_size` when there is one, else
    the size of the `model*.safetensors` files, else `pytorch_model*.bin`. Named
    patterns, not every weight file: some repos ship a second copy of the same
    tensors (`consolidated.safetensors`).
    """
    if not cache_path:
        return 0
    import json

    root = Path(cache_path)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            with open(root / index_name) as handle:
                total = json.load(handle).get("metadata", {}).get("total_size")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(total, (int, float)) and not isinstance(total, bool) and total > 0:
            return int(total / (1024 * 1024))
    for pattern in ("model*.safetensors", "pytorch_model*.bin"):
        try:
            sizes = [path.stat().st_size for path in root.glob(pattern) if path.is_file()]
        except OSError:
            sizes = []
        if sizes:
            return int(sum(sizes) / (1024 * 1024))
    return 0


#: Which part of a meta-device check failed decides how the failure is logged.
_STAGE_CHECKPOINT = "checkpoint"
_STAGE_ENGINE = "engine"


def _transformers_version() -> str:
    try:
        import transformers

        return str(getattr(transformers, "__version__", "unknown"))
    except ImportError:
        return "unavailable"


def _log_unverified(
    stage: str, error: BaseException, unverifiable_event: str, engine_event: str, **fields: Any
) -> None:
    """Log a meta-device check that could not run: the checkpoint's limit, or transformers'.

    The split preflight and the materialised sizing go through transformers
    PRIVATE functions (`_get_device_map`, `compute_module_sizes`,
    `get_hf_quantizer`). A checkpoint whose config no class builds on the meta
    device, or whose quantizer's package is missing, is a WARNING: that
    checkpoint cannot be verified here, and the load decides. A failure in
    transformers' own machinery, after the config, the class and the quantizer
    all built (an import that is gone, a call whose signature changed), is an
    ERROR with its own event. It happens to every checkpoint, so the check has
    gone dark for every load, and under the one shared warning it read as a run
    of odd checkpoints. Review round 4, 2026-09-14.
    """
    if stage == _STAGE_ENGINE:
        logger.error(
            engine_event,
            transformers_version=_transformers_version(),
            error_type=type(error).__name__,
            error=str(error)[:300],
            **fields,
        )
    else:
        logger.warning(unverifiable_event, error=str(error)[:300], **fields)


def checkpoint_materialised_mb(cache_path: Optional[str], trust_remote_code: bool = False) -> int:
    """What a pre-quantized checkpoint occupies once transformers has loaded it, in MB; 0 when unknown.

    The weights a checkpoint stores are not always what is loaded. transformers'
    quantizer decides in `validate_environment` whether to keep the stored
    precision, and some DEQUANTIZE to bf16: FineGrainedFP8 on a card below
    compute capability 8.9 (the RTX 3090 and 3080 Ti are 8.6), MXFP4 without the
    `kernels` package or Triton (transformers 5.15.1). So the checkpoint's config
    is built on the meta device with the quantizer the load constructs, and its
    modules are sized the way transformers sizes them for a device map
    (`compute_module_sizes`), with no weight read. The quantizer's own
    `validate_environment` reads the current card's compute capability, exactly
    as the load's does, and that initialises CUDA in this process.

    0 means unknown, never empty: no config, a method with no transformers
    quantizer, or a quantizer that cannot be constructed here (GPTQ without
    optimum — the load fails the same way). Review round 3, 2026-09-14. A
    failure in transformers' own machinery is logged apart (_log_unverified).
    """
    if not cache_path or AutoConfig is None:
        return 0
    stage = _STAGE_ENGINE  # the imports are transformers' private API
    try:
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.quantizers.auto import get_hf_quantizer

        stage = _STAGE_CHECKPOINT
        config = AutoConfig.from_pretrained(cache_path, trust_remote_code=trust_remote_code)
        hf_quantizer, config, device_map = get_hf_quantizer(config, None, "sequential", True, {})
        if hf_quantizer is None:
            return 0
        model = _meta_model(config, trust_remote_code)
        # Config, class and quantizer all built: from here a failure is transformers'.
        stage = _STAGE_ENGINE
        hf_quantizer.preprocess_model(
            model=model,
            dtype=torch.bfloat16,
            device_map=device_map,
            checkpoint_files=None,
            use_kernels=False,
        )
        sizes, _ = compute_module_sizes(model, hf_quantizer)
    except Exception as e:  # noqa: BLE001 - unknown here; what the checkpoint stores is the floor
        _log_unverified(
            stage,
            e,
            unverifiable_event="checkpoint_materialised_size_unknown",
            engine_event="checkpoint_materialised_engine_failed",
            cache_path=cache_path,
        )
        return 0
    return int(sizes.get("", 0) / (1024 * 1024))


def transformers_estimate_mb(
    row_estimate_mb: int,
    cache_path: Optional[str],
    pre_quantization: Optional[dict[str, Any]],
    trust_remote_code: bool = False,
) -> int:
    """The memory a transformers load of this checkpoint is planned for, in MB.

    A row's estimate is its parameter count at its quantization LABEL
    (memory_utils.estimate_memory_mb). A pre-quantized checkpoint loads at its
    own precision whatever the label says — and the troubleshooting guide tells
    operators to download GPTQ/AWQ checkpoints as FP16 — so it is sized from the
    weights it stores, with the same runtime overhead. Label-sized, a 4-bit GPTQ
    checkpoint of a 32B model on an FP16 row was planned at ~74 GB against its
    ~19 GB of weights, and refused on a node where one card holds it. Review
    round 2, 2026-09-14.

    What it stores is a FLOOR, not the answer: transformers dequantizes some
    methods on these cards (checkpoint_materialised_mb). An FP8 checkpoint of a
    14B model stores ~15 GB and loads as 28 GB of bf16; sized from its files it
    was planned whole onto the 3090 and would run out of memory mid-load, after
    the resident model was unloaded. It is sized at the larger of the two.
    Review round 3, 2026-09-14.

    Anything else, or a checkpoint that cannot be measured either way, keeps the
    row's estimate.
    """
    if pre_quantization is None:
        return int(row_estimate_mb or 0)
    weights_mb = max(
        checkpoint_weights_mb(cache_path),
        checkpoint_materialised_mb(cache_path, trust_remote_code),
    )
    if weights_mb <= 0:
        return int(row_estimate_mb or 0)
    return int(weights_mb * MEMORY_OVERHEAD_FACTOR)


def _bitsandbytes_config(quantization: str) -> Any:
    """The BitsAndBytesConfig miLLM loads a Q4 or Q8 checkpoint with; None otherwise.

    ONE definition for the load and the split preflight, which must compute the
    map with the quantizer the load will use.

    Q8 carries no `llm_int8_enable_fp32_cpu_offload`. That flag is what lets a
    bitsandbytes device map put modules on the CPU or disk; without it
    transformers refuses such a map before reading a weight
    (quantizer_bnb_8bit.validate_environment). A transformers model runs on GPUs
    only (operator decision 3, 2026-09-13), and the flag was set on every Q8 load
    while nothing needed it.
    """
    if quantization == "Q4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if quantization == "Q8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _meta_model(config: Any, trust_remote_code: bool) -> Any:
    """The skeleton the load would build, on the meta device: no weights, no GPU.

    The same class choice as ModelLoadContext.load: the class the config names,
    then AutoModelForCausalLM, then AutoModel.
    """
    from transformers import AutoModel

    candidates: list[Any] = []
    try:
        candidates.append(_get_auto_model_class(config))
    except Exception:  # noqa: BLE001 - fall through to the generic classes
        pass
    for generic in (AutoModelForCausalLM, AutoModel):
        if generic not in candidates:
            candidates.append(generic)
    last_error: Optional[BaseException] = None
    for model_class in candidates:
        try:
            with torch.device("meta"):
                return model_class.from_config(
                    config, dtype=torch.bfloat16, trust_remote_code=trust_remote_code
                )
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise last_error if last_error is not None else ModelLoadError("no model class built")


def preflight_split(
    model_name: str,
    cache_path: Optional[str],
    quantization: str,
    placement: Placement,
    trust_remote_code: bool = False,
) -> Optional[dict[str, int]]:
    """Refuse a split transformers would map partly off its GPUs, before any weight is read.

    The plan is sized in MB from an estimate. transformers then maps MODULES (its
    copy of accelerate's infer_auto_device_map): it holds back room for the
    largest layer on the lowest-index card and strands the tail of every card
    but the last. The estimate's 20% overhead usually absorbs that, and nothing
    guarantees it. Swept against that inference over meta-device models (review
    round 2, 2026-09-14: 6 shapes x FP16/Q8/Q4 x 19 depths x 7 card sets, 2,394
    plans per request type): an estimate 5% over the weights put a module on
    disk under 4 accepted Auto splits and 9 "all" splits, and the production
    estimate under one Q4 "all" split near capacity. plan_shard's "all" fix
    removed that one and 7 of the 9; the rest is what a plan in MB cannot see
    (with the exact weights as the estimate, 25 Auto and 16 "all"). Both surfaced only inside
    the load — an FP16 map to disk after every weight had loaded, a bitsandbytes
    one after the resident model was unloaded.

    So the map itself is computed here, the way from_pretrained computes it: the
    checkpoint's config on the meta device, the quantizer the load would use, and
    exactly the `device_map` and `max_memory` the placement passes, through
    transformers' own `_get_device_map`. Well under a second, with no weight read
    and no per-card memory query; a quantizer's own environment check may read
    the current card's compute capability, as the load's does.

    Returns:
        MB per device of the map ("cuda:N"), or None when it cannot be computed
        here: not a split, no readable config, a class that will not build on
        meta, a quantizer whose package is missing. None never refuses — the
        load checks where the model landed, off the GPU or (for "all") on
        fewer cards. Why it is None is logged: a warning for the checkpoint, an
        ERROR when transformers' own machinery failed (_log_unverified).

    Raises:
        InsufficientMemoryError: part of the model would map to the CPU or disk.
    """
    if not placement.is_shard or not placement.gpu_indices or not cache_path or AutoConfig is None:
        return None
    stage = _STAGE_ENGINE  # the imports are transformers' private API
    try:
        from transformers.integrations.accelerate import _get_device_map, compute_module_sizes
        from transformers.quantizers.auto import get_hf_quantizer

        stage = _STAGE_CHECKPOINT
        config = AutoConfig.from_pretrained(cache_path, trust_remote_code=trust_remote_code)
        quantization_config = (
            None if checkpoint_is_pre_quantized(cache_path) else _bitsandbytes_config(quantization)
        )
        device_map = placement.transformers_device_map()
        hf_quantizer, config, device_map = get_hf_quantizer(
            config, quantization_config, device_map, True, {}
        )
        model = _meta_model(config, trust_remote_code)
        # Config, class and quantizer all built: from here a failure is transformers'.
        stage = _STAGE_ENGINE
        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model,
                dtype=torch.bfloat16,
                device_map=device_map,
                checkpoint_files=None,
                use_kernels=False,
            )
        max_memory = placement.transformers_max_memory()
        mapped = _get_device_map(
            model, device_map, dict(max_memory) if max_memory else None, hf_quantizer
        )
        sizes, _ = compute_module_sizes(model, hf_quantizer, only_modules=False)
    except ValueError as e:
        if _is_offload_refusal(e):
            raise _off_gpu_refusal(
                model_name, placement, ["cpu or disk"], [], engine_message=str(e),
                before_loading=True,
            ) from e
        _log_unverified(
            stage,
            e,
            unverifiable_event="split_preflight_skipped",
            engine_event="split_preflight_engine_failed",
            model_name=model_name,
        )
        return None
    except Exception as e:  # noqa: BLE001 - unverifiable here; the load checks again
        _log_unverified(
            stage,
            e,
            unverifiable_event="split_preflight_skipped",
            engine_event="split_preflight_engine_failed",
            model_name=model_name,
        )
        return None

    mapped_mb: dict[str, int] = {}
    for name, device in mapped.items():
        label = f"cuda:{device}" if isinstance(device, int) and not isinstance(device, bool) else str(device)
        mapped_mb[label] = mapped_mb.get(label, 0) + int(sizes.get(name, 0) / (1024 * 1024))
    allowed = set(placement.device_labels)
    off_gpu = sorted(label for label in mapped_mb if label not in allowed)
    if off_gpu:
        raise _off_gpu_refusal(
            model_name, placement, off_gpu, sorted(mapped_mb),
            mapped_mb_by_device=mapped_mb, before_loading=True,
        )
    if placement.reason == REASON_REQUESTED_ALL:
        # "all" promises every card and is honoured or refused, never swapped.
        # The plan divides MB; transformers places whole layers in index order,
        # holding room for the largest on the lowest-index card, so a card's
        # share can hold none of them, or the first card all of them. Loaded, it
        # would have recorded "all" and run on fewer cards. Review round 3,
        # 2026-09-14. An Auto split that lands on fewer cards is fine: it only
        # ever promised to fit.
        unused = [label for label in placement.device_labels if label not in mapped_mb]
        if unused:
            raise _split_not_honoured(model_name, placement, unused, mapped_mb)
    logger.info("split_preflight_mapped", model_name=model_name, mapped_mb_by_device=mapped_mb)
    return mapped_mb


def _split_not_honoured(
    model_name: str,
    placement: Placement,
    unused: list[str],
    mapped_mb: Optional[dict[str, int]] = None,
    landed_on: Optional[list[str]] = None,
) -> SplitNotHonouredError:
    """The refusal for an "all" that leaves a card with none of the model.

    With `mapped_mb`, found by the preflight from the map before any weight was
    read; with `landed_on` instead, found where the loaded model actually is,
    because the preflight could not compute the map ahead (review round 4).
    """
    max_memory = placement.transformers_max_memory() or {}
    details: dict[str, Any] = {
        "requested": placement.requested,
        "unused_devices": unused,
        "max_memory": {f"cuda:{index}": value for index, value in sorted(max_memory.items())},
        "placement": placement.to_dict(),
        "before_loading": mapped_mb is not None,
    }
    why = (
        "A split places whole layers in index order and keeps room for the largest one free "
        "on the lowest-index card, so with this much free memory at least one card takes "
        "nothing."
    )
    if mapped_mb is not None:
        layout = ", ".join(f"{label} {mb} MB" for label, mb in sorted(mapped_mb.items()))
        details["mapped_mb_by_device"] = dict(sorted(mapped_mb.items()))
        message = (
            f"{model_name} cannot be split across every GPU as requested: transformers would "
            f"put none of it on {', '.join(unused)} ({layout}). {why} Nothing was unloaded. "
            "Choose Auto or a named card."
        )
    else:
        details["landed_on_devices"] = sorted(landed_on or [])
        message = (
            f"{model_name} was loaded to split across every GPU as requested, and transformers "
            f"put none of it on {', '.join(unused)} (it landed on "
            f"{', '.join(sorted(landed_on or [])) or 'no GPU'}). {why} The layout could not be "
            "worked out before loading, so the model served before this one was already "
            "unloaded; this load has been released. Choose Auto or a named card."
        )
    return SplitNotHonouredError(message, details=details)


def _off_gpu_refusal(
    model_name: str,
    placement: Placement,
    off_gpu: list[str],
    device_labels: list[str],
    engine_message: Optional[str] = None,
    mapped_mb_by_device: Optional[dict[str, int]] = None,
    before_loading: bool = False,
) -> InsufficientMemoryError:
    """The refusal for a transformers load that would run partly off the GPU.

    `before_loading` says the split preflight found it from the checkpoint's
    config, with no weight read and (at the pre-check) the served model still
    loaded; `mapped_mb_by_device` is the map it computed.
    """
    details: dict[str, Any] = {
        "required_mb": placement.required_mb,
        "available_mb": placement.budget_mb,
        "off_gpu": off_gpu,
        "devices": device_labels,
        "placement": placement.to_dict(),
        "before_loading": before_loading,
    }
    if mapped_mb_by_device is not None:
        details["mapped_mb_by_device"] = dict(sorted(mapped_mb_by_device.items()))
    if engine_message:
        details["engine_message"] = engine_message[:500]
    return InsufficientMemoryError(
        f"{model_name} does not fit on the GPU memory it was given "
        f"({', '.join(placement.device_labels) or 'no card'}): part of it would run from "
        f"{', '.join(off_gpu)}. A transformers model is never offloaded to the CPU or "
        "disk. Free memory on the cards, choose a smaller quantization, or serve it as GGUF.",
        details=details,
    )


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
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
    )

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

    # Encoder classifiers: NLI, zero-shot, sentiment, any *ForSequenceClassification.
    #
    # Without this branch these fall through to AutoModelForCausalLM and the load
    # dies with "Unrecognized configuration class ... for this kind of AutoModel:
    # AutoModelForCausalLM", followed by a 200-line dump of every config
    # transformers knows. Hit on p-christ/ModernBERT-large-nli, whose config says
    # architectures=['ModernBertForSequenceClassification'], num_labels=3,
    # id2label={0:'entailment',1:'neutral',2:'contradiction'}.
    #
    # These models DO NOT GENERATE. Loading them correctly is necessary but not
    # sufficient — the generation endpoints refuse them separately, by
    # architecture, so a chat request cannot reach a model with no lm_head.
    classification_indicators = ("ForSequenceClassification",
                                 "ForMultipleChoice",
                                 "ForTokenClassification")
    for arch in architectures:
        if any(ind in arch for ind in classification_indicators):
            logger.info("auto_model_class_sequence_classification", architecture=arch)
            return AutoModelForSequenceClassification

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
        #: The cards a failed load may have allocated on, for cleanup.
        self.gpu_indices: list[int] = []

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
                if torch.cuda.is_available():
                    _release_cuda_memory(self.gpu_indices)
            except Exception:
                gc.collect()

        return False  # Don't suppress exception

    def load(
        self,
        cache_path: str,
        quantization: str,
        trust_remote_code: bool = False,
        placement: Optional[Placement] = None,
        torch_compile: bool = False,
        # "default", NOT "reduce-overhead". reduce-overhead enables CUDA Graphs,
        # which broke this generate path in production on 2026-07-27: every
        # request after the first failed with "accessing tensor output of
        # CUDAGraphs that has been overwritten by a subsequent run".
        torch_compile_mode: str = "default",
    ) -> LoadedModel:
        """
        Load model with quantization config.

        Args:
            cache_path: Path to the cached model files
            quantization: Quantization type ("FP16", "Q8", "Q4")
            trust_remote_code: Whether to trust remote code
            placement: Where the model goes, from `choose_gpu`. None resolves
                it here (every card when CUDA is present, else the CPU), so no
                entry point can skip placement and land on GPU 0 by default.

        Returns:
            LoadedModel instance
        """
        if placement is None:
            placement = (
                choose_gpu(0) if torch.cuda.is_available() else cpu_placement()
            )
        self.gpu_indices = placement.gpu_indices
        # AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig are
        # imported at module level so test patches on 'millm.ml.model_loader.*'
        # apply correctly. If transformers is absent the module itself fails to load.
        if AutoTokenizer is None:
            raise ModelLoadError(
                "Required packages not installed. Install torch and transformers.",
                details={"missing_package": "transformers"},
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
        else:
            # Shared with preflight_split, which must compute the device map
            # with the quantizer this load uses.
            quantization_config = _bitsandbytes_config(quantization)
            if quantization_config is not None:
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

        # The device map comes from the placement decision, the same for every
        # quantization. A model that fits one card goes on that card whole
        # ({"": "cuda:N"}): device_map="auto" spread every model across both
        # cards, paying a cross-card copy on every forward pass for a model the
        # 3090 could hold alone.
        #
        # A split carries max_memory for EVERY quantization, GPU indices only.
        # Phase 1 passed "auto" with no max_memory for FP16, so accelerate took
        # every card and then the CPU; and bitsandbytes got a "cpu" entry,
        # which was never needed as staging (transformers 5.15.1 quantizes each
        # weight on its target device) and only let a model that did not fit
        # run from host RAM. See Placement.transformers_device_map.
        is_bitsandbytes = quantization_config is not None
        load_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch_dtype,
            "device_map": placement.transformers_device_map(),
            "trust_remote_code": trust_remote_code,
            "attn_implementation": attn_impl,
            "low_cpu_mem_usage": True,
        }
        max_memory = placement.transformers_max_memory()
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
            logger.info(
                "split_load_memory_map",
                model_id=self.model_id,
                device_map=load_kwargs["device_map"],
                max_memory={str(k): v for k, v in max_memory.items()},
                planned_mb_by_device=placement.to_dict()["planned_mb_by_device"],
                bitsandbytes=is_bitsandbytes,
            )

        free_before = free_mb_by_index(placement.gpu_indices)

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
                if _is_offload_refusal(e):
                    # The placement did not hold the model. Another model class
                    # computes the same device map and is refused the same way,
                    # so the fallbacks below would only repeat it twice.
                    raise
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
                    except Exception as causal_lm_error:
                        if _is_offload_refusal(causal_lm_error):
                            # The same refusal one class later: AutoModel would
                            # compute the same map. And when AutoModel fails for
                            # its own reason, that error replaced this one.
                            # Review round 1, 2026-09-14.
                            raise
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
        except ValueError as e:
            if _is_offload_refusal(e):
                raise _off_gpu_refusal(
                    self.model_name, placement, ["cpu or disk"], [], engine_message=str(e)
                ) from e
            raise

        # Workaround for GraniteMoEHybrid models whose config has no mamba layers
        # (layers_block_type empty, all layer_types == "attention") but whose model
        # class still unconditionally calls _update_mamba_mask during forward().
        # This fails because has_previous_state() on the DynamicCache raises when
        # no LinearAttention layers exist. Patch _update_mamba_mask on the inner
        # model so it tolerates attention-only caches.
        _patch_granite_hybrid_mamba_mask(self.model)

        # Where the model ACTUALLY landed, from its device map and its tensors.
        # hf_device_map comes first because device_map="auto" models always
        # have model.device == "cpu" (the dispatch device), which is misleading.
        device_labels = model_device_labels(self.model)
        gpu_indices = gpu_indices_of(device_labels)

        # REFUSE a transformers model that did not land entirely on GPUs. A
        # GPU-only max_memory does not guarantee it: accelerate always adds
        # "disk" as a last device (`_init_infer_auto_device_map`), and
        # transformers 5.15.1 serves safetensors "disk" entries straight from
        # the checkpoint with no offload folder and no error. The model would
        # load and answer, hours slower, with nothing saying why. bitsandbytes
        # refuses such a map itself (handled above); FP16 only shows it here.
        # The context manager's exit releases what was loaded.
        off_gpu = [label for label in device_labels if label in OFF_GPU_LABELS]
        if off_gpu and placement.mode != MODE_CPU:
            raise _off_gpu_refusal(self.model_name, placement, off_gpu, device_labels)

        # REFUSE an "all" that landed on fewer cards than it names. preflight_split
        # refuses that before any weight is read, but it SKIPS whenever it cannot
        # compute the map (no readable config, a class that will not build on the
        # meta device, a quantizer whose package is missing), and nothing here
        # checked for "all": such a load recorded "all" and ran on fewer cards.
        # Review round 4, 2026-09-14. Auto is not checked: it only promised to fit.
        if placement.reason == REASON_REQUESTED_ALL:
            unused = [label for label in placement.device_labels if label not in device_labels]
            if unused:
                raise _split_not_honoured(
                    self.model_name, placement, unused, landed_on=device_labels
                )

        # Memory used, per card: the drop in free memory (mem_get_info sees
        # bitsandbytes allocations too) on every card the model now occupies.
        # This read "total minus free" on GPU 0 only — the wrong card for a
        # model on the 3090, and it charged the model for whatever else the
        # card was holding.
        measured = sorted(set(gpu_indices) | set(placement.gpu_indices))
        memory_by_device_mb: dict[str, int] = {}
        if torch.cuda.is_available() and measured:
            try:
                free_after = free_mb_by_index(measured)
                memory_by_device_mb = memory_used_by_device(
                    free_before, free_after, gpu_indices or placement.gpu_indices
                )
            except Exception:
                # Fallback to the torch allocator's count on each card.
                memory_by_device_mb = {
                    f"cuda:{index}": int(torch.cuda.memory_allocated(index) / (1024 * 1024))
                    for index in (gpu_indices or placement.gpu_indices)
                }
        memory_used_mb = sum(memory_by_device_mb.values())

        # Get model properties
        num_parameters = 0
        try:
            num_parameters = self.model.num_parameters()
        except Exception:
            pass

        device_str = ", ".join(device_labels) if device_labels else "unknown"

        # Get dtype info
        dtype_str = "unknown"
        try:
            if hasattr(self.model, "dtype"):
                dtype_str = str(self.model.dtype).replace("torch.", "")
            elif hasattr(self.model, "config") and hasattr(self.model.config, "torch_dtype"):
                dtype_str = str(self.model.config.torch_dtype).replace("torch.", "")
        except Exception:
            pass

        # Apply torch.compile for faster decoding.
        # bitsandbytes quantization is incompatible with torch.compile (CUDA kernel
        # registration conflicts). The model_service auto-detection already avoids
        # passing torch_compile=True for bitsandbytes, but we guard here too.
        if torch_compile and quant_method == "bitsandbytes":
            logger.warning(
                "torch_compile_skipped_bitsandbytes_incompatible",
                quantization=quantization,
            )
        elif torch_compile and len(device_labels) > 1:
            # A model split across devices runs through accelerate's per-module
            # dispatch hooks, which move activations between cards inside the
            # forward pass. A compiled graph over that breaks at every hook and
            # gains nothing, and the split is only known here, after
            # from_pretrained — model_service resolves the setting before the
            # load and cannot see it. Single-card loads (the normal case now)
            # still compile.
            logger.warning(
                "torch_compile_skipped_model_split_across_devices",
                devices=device_labels,
                placement_mode=placement.mode,
            )
        elif torch_compile:
            try:
                # CRITICAL for SAE steering: TorchDynamo's default
                # skip_nnmodule_hook_guards=True means the compiled graph is NOT
                # re-guarded when a forward hook is later registered on a
                # submodule.  miLLM registers the SAE steering/monitoring hook on
                # an inner decoder layer *after* the model is compiled (at SAE
                # attach time), so with the default the cached graph would keep
                # running the un-hooked path and steering would silently no-op.
                # Setting this to False makes hook registration invalidate the
                # relevant cached graph (one-time ~recompile on the first forward
                # after attach/detach), guaranteeing the hook actually fires.
                try:
                    import torch._dynamo as _dynamo

                    _dynamo.config.skip_nnmodule_hook_guards = False
                    logger.info("dynamo_hook_guards_enabled_for_sae_steering")
                except Exception as _e:  # pragma: no cover - version drift
                    logger.warning(
                        "dynamo_hook_guard_config_failed",
                        error=str(_e),
                        hint="SAE steering may not take effect under torch.compile",
                    )

                logger.info("torch_compile_starting", mode=torch_compile_mode)
                # Kept so the soak below can UNDO the compile. Without a way
                # back, a compiled path that fails at generate time leaves the
                # server returning 500 for every request.
                _uncompiled_forward = self.model.forward
                self.model.forward = torch.compile(
                    self.model.forward,
                    mode=torch_compile_mode,
                    fullgraph=False,  # Allow hooks and dynamic control flow (SAE hooks)
                )
                logger.info("torch_compile_complete", mode=torch_compile_mode)

                # Eagerly trigger JIT compilation with a dummy decode-shaped
                # forward pass so the first real inference request doesn't pay
                # the ~20 s warmup cost.  seq_len=1 covers the hot decode path
                # (one new token per step).  Failure never blocks model loading.
                try:
                    import time as _time
                    logger.info("torch_compile_warmup_starting")
                    _t0 = _time.monotonic()

                    # Where the input embeddings live — the same answer the
                    # inference path uses. This kept its own shorter key list,
                    # which named no nested multimodal layout, so the two could
                    # disagree about one model.
                    _input_device = model_input_device(self.model) or (
                        device_labels[0] if device_labels else "cpu"
                    )

                    # SOAK, not a single call.
                    #
                    # One warmup pass is what let the CUDA-Graphs breakage ship
                    # looking clean on 2026-07-27: compile succeeded, the single
                    # warmup succeeded, and then EVERY subsequent request failed
                    # because the graph's output tensor had been overwritten by
                    # the following run. That class of fault is invisible until
                    # the second pass, so run several and read the outputs.
                    _SOAK_PASSES = 3
                    # Exercise the path that ACTUALLY broke: cached, multi-token
                    # generation. The original warmup was a single uncached
                    # seq_len=1 forward, which does not resemble decoding and
                    # would not have reproduced the CUDA-Graphs fault even if it
                    # had been run twice. A safety net that misses the failing
                    # path is not a safety net.
                    with torch.no_grad():
                        _prompt = torch.zeros(
                            1, 8, dtype=torch.long, device=_input_device
                        )
                        _mask = torch.ones_like(_prompt)
                        for _pass in range(_SOAK_PASSES):
                            _gen = self.model.generate(
                                input_ids=_prompt,
                                attention_mask=_mask,
                                max_new_tokens=4,
                                do_sample=False,
                                use_cache=True,
                            )
                            # Read the result back. A CUDA-Graphs violation
                            # raises on ACCESS, not on the call, so a soak that
                            # never touches the output still passes.
                            _ = int(_gen[0, -1].item())

                    logger.info(
                        "torch_compile_warmup_complete",
                        elapsed_s=round(_time.monotonic() - _t0, 1),
                        soak_passes=_SOAK_PASSES,
                    )
                except Exception as e:
                    # Do NOT keep serving a compiled path that just failed its
                    # soak — that is the 500-for-every-request outcome. Fall
                    # back to eager, which is slower and correct.
                    self.model.forward = _uncompiled_forward
                    logger.error(
                        "torch_compile_soak_failed_reverted_to_eager",
                        error=str(e),
                        mode=torch_compile_mode,
                        hint="serving uncompiled; compile is disabled for this load",
                    )

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
            # The transformers path computes embeddings from hidden states at
            # request time, so there is no construction-time flag to refuse and
            # no architecture that can decline it.
            supports_embeddings=True,
            gpu_indices=gpu_indices,
            memory_by_device_mb=memory_by_device_mb,
            placement={**placement.to_dict(), "devices": device_labels, "gpu_indices": gpu_indices},
        )


#: How many layers to offload to the GPU. -1 means "all of them". A GGUF file
#: is already quantized on disk, so the whole point is that it fits; partial
#: offload is a fallback we do not attempt to guess at.
GGUF_GPU_LAYERS = -1

#: The smallest context worth loading at. Below this a model is too cramped to
#: hold a real exchange, and failing is more useful than serving a window that
#: truncates the first message.
GGUF_MIN_CONTEXT = 2048


#: KV-cache precision names -> ggml type constants, resolved defensively: the
#: module may be absent (llama_cpp is an extra) and the constant names have
#: moved between versions. Values are the stable ggml enum numbers.
_KV_CACHE_TYPES: dict[str, int] = {
    "f16": getattr(llama_cpp_module, "GGML_TYPE_F16", 1) if llama_cpp_module else 1,
    "q8_0": getattr(llama_cpp_module, "GGML_TYPE_Q8_0", 8) if llama_cpp_module else 8,
    "q4_0": getattr(llama_cpp_module, "GGML_TYPE_Q4_0", 2) if llama_cpp_module else 2,
}


def _is_context_related(exc: Exception) -> bool:
    """Whether shrinking the context, or dropping a capability, could help.

    The ladder and the two capability fallbacks all exist for ONE failure:
    llama.cpp declining to create a *context*. They were applied to every
    exception, which turned a single unrelated error into fifteen attempts and
    three wrong diagnoses. Observed on 2026-09-07: a model whose WEIGHTS failed
    to load ("Failed to load model from file") was retried at five context
    lengths, then blamed on the KV cache type, then on the pooling type, over
    36 seconds — and the final message said no context could be created, which
    was true and beside the point.

    Matched on llama.cpp's own wording, positively: anything unrecognised is
    treated as NOT context-related and raised at once. Failing fast on an
    unknown error is the safe direction — the cost is one honest error surfacing
    immediately instead of after a pointless retry storm.
    """
    text = str(exc).lower()
    return "llama_context" in text or "context" in text and "create" in text


def _kv_cache_kwargs(kv_type: str, flash_attention: bool) -> dict[str, Any]:
    """llama.cpp kwargs for the KV cache, or {} for the plain F16 default.

    FLASH ATTENTION IS A HARD DEPENDENCY OF A QUANTIZED CACHE, not a tuning
    knob to pair with it. MEASURED on gemma-4-31b IQ4_XS: q8_0 WITHOUT flash
    attention fails to create a context at 8192, the same length it reaches
    12288 with it. Quantizing the cache while flash attention is off is
    therefore strictly worse than not quantizing at all, so it is refused
    rather than silently shipped.
    """
    name = (kv_type or "f16").strip().lower()
    if name not in _KV_CACHE_TYPES:
        logger.warning(
            "gguf_unknown_kv_cache_type",
            configured=kv_type,
            known=sorted(_KV_CACHE_TYPES),
            detail="falling back to f16, llama.cpp's own default",
        )
        name = "f16"

    if name == "f16":
        # Nothing to pass: this IS the default, and flash attention alone is
        # not worth forcing on when nothing depends on it.
        return {}

    if not flash_attention:
        logger.warning(
            "gguf_kv_quantization_needs_flash_attention",
            kv_cache_type=name,
            detail=(
                "GGUF_FLASH_ATTENTION is off, and a quantized KV cache without "
                "it fails to create a context at lengths it otherwise reaches. "
                "Using f16 instead — turn flash attention on to get the larger "
                "window."
            ),
        )
        return {}

    return {
        "flash_attn": True,
        "type_k": _KV_CACHE_TYPES[name],
        "type_v": _KV_CACHE_TYPES[name],
    }


def declared_context(path: str) -> int | None:
    """The context the model was TRAINED for, read from the file itself.

    Loading the model onto the CPU with mmap reads the hyper-parameters without
    reading the weights and without touching VRAM: MEASURED at 1.2 s and 0 MiB
    on a 7.8 GiB file. `vocab_only` is NOT usable here — it skips the hparams
    and reports `n_ctx_train = 0`.

    Why this exists: the ladder used to start from a global config default of
    8192, so a model trained for 262144 was served an 8192 window and nothing
    said so. The declared context is a property of the FILE and the only honest
    place to start from; a config default is a guess about a model it has never
    seen.

    Returns None on any failure — this is an optimisation, not a gate, and a
    model must never fail to load because its metadata could not be read.
    """
    if llama_cpp_module is None:
        return None
    try:
        llama_cpp_module.llama_backend_init()
        params = llama_cpp_module.llama_model_default_params()
        params.n_gpu_layers = 0
        params.use_mmap = True
        model = llama_cpp_module.llama_model_load_from_file(str(path).encode(), params)
        if not model:
            return None
        try:
            trained = int(llama_cpp_module.llama_model_n_ctx_train(model))
        finally:
            llama_cpp_module.llama_model_free(model)
        return trained if trained > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("gguf_declared_context_probe_failed", error=str(exc)[:200])
        return None


#: Bytes per element of a KV cache entry, by cache type. q8_0 and q4_0 carry a
#: scale per 32-element block, which the fractions include.
_KV_BYTES_PER_ELEMENT = {"f16": 2.0, "q8_0": 1.0 + 2 / 32, "q4_0": 0.5 + 2 / 32}

#: Everything on the card that is neither weights nor KV cache: the CUDA
#: context, llama.cpp's compute buffers, and cuBLAS workspaces. MEASURED at
#: ~2.0 GiB for gemma-4-31b on an RTX 3090 (20608 MiB in use with 16081 MiB of
#: weights and 2520 MiB of KV at 8192). It is not a constant of nature, but it
#: is stable enough to plan with, and the ladder verifies whatever it predicts.
_GGUF_RUNTIME_OVERHEAD_MB = 2048

#: Fraction of the card to plan against. The last few percent go to
#: fragmentation and to whatever else holds the device; planning to 100% picks
#: a context that computes as fitting and then fails to allocate.
_VRAM_PLANNING_FRACTION = 0.94


def gguf_kv_bytes_per_token(path: str, kv_cache_type: str) -> float | None:
    """KV-cache bytes one token costs this model at `kv_cache_type`. None if unknowable.

        bytes/token = 2 (K and V) x n_layer x n_head_kv x head_dim x bytes/element

    Read from the file's metadata with the same CPU-only mmap probe
    `declared_context` uses — no weights read, no VRAM touched.
    """
    if llama_cpp_module is None:
        return None
    try:
        llama_cpp_module.llama_backend_init()
        params = llama_cpp_module.llama_model_default_params()
        params.n_gpu_layers = 0
        params.use_mmap = True
        model = llama_cpp_module.llama_model_load_from_file(str(path).encode(), params)
        if not model:
            return None
        try:
            n_layer = int(llama_cpp_module.llama_model_n_layer(model))
            n_head_kv = int(llama_cpp_module.llama_model_n_head_kv(model))
            n_embd = int(llama_cpp_module.llama_model_n_embd(model))
            n_head = int(llama_cpp_module.llama_model_n_head(model))
        finally:
            llama_cpp_module.llama_model_free(model)

        if min(n_layer, n_head_kv, n_embd, n_head) <= 0:
            return None
        head_dim = n_embd // n_head
        per_element = _KV_BYTES_PER_ELEMENT.get(
            (kv_cache_type or "f16").strip().lower(), 2.0
        )
        return float(2 * n_layer * n_head_kv * head_dim * per_element)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gguf_kv_size_probe_failed", error=str(exc)[:200])
        return None


def predicted_max_context(
    path: str,
    kv_cache_type: str,
    free_vram_mb: int,
    weights_mb: int,
    bytes_per_token: float | None = None,
    n_cards: int = 1,
) -> int | None:
    """The largest context the arithmetic says will fit. None if unknowable.

    A KV cache has an exact size — there is no reason to discover it by loading
    the model repeatedly:

        bytes/token = 2 (K and V) x n_layer x n_head_kv x head_dim x bytes/element

    Every term comes from metadata the declared-context probe already reads.
    For gemma-4-31b that is 2 x 60 x 16 x 168 = 630 KB/token at f16, halved at
    q8_0 — which is why quantizing the cache doubled the usable window.

    VALIDATED against every measurement taken on the RTX 3090, all four
    predicted correctly:

        f16  @ 4096  -> 20.6 GiB   loaded (20608 MiB measured)
        f16  @ 8192  -> 23.1 GiB   failed
        q8_0 @ 12288 -> 21.9 GiB   loaded
        q8_0 @ 16384 -> 23.1 GiB   failed

    This SEEDS the ladder; it does not replace it. The overhead term is
    empirical and the compute buffer grows with batch size, so the number is a
    good starting point and not a guarantee — the load attempt is still what
    decides. Returning None simply means starting from the configured ceiling,
    as before.

    `free_vram_mb` must be the free memory of the card(s) the model will use —
    one card in single-GPU mode, their sum under a layer split — and `n_cards`
    how many cards that is. The runtime overhead is PER CARD: each card of a
    split holds its own CUDA context and compute buffers. Counting it once for a
    two-card split over-predicted the window by 2 GiB of cache, and the ladder
    then spent a full model load discovering that. `bytes_per_token` may be
    passed when the caller already probed it, saving a second metadata load.
    """
    try:
        if bytes_per_token is None:
            bytes_per_token = gguf_kv_bytes_per_token(path, kv_cache_type)
        if not bytes_per_token:
            return None

        budget_mb = (
            free_vram_mb * _VRAM_PLANNING_FRACTION
            - weights_mb
            - _GGUF_RUNTIME_OVERHEAD_MB * max(int(n_cards), 1)
        )
        if budget_mb <= 0:
            return None
        tokens = int(budget_mb * 1024 * 1024 / bytes_per_token)
        # Down to a 1024 boundary: the precision is not real, and a round
        # number is easier to reason about in a log line.
        return max(0, (tokens // 1024) * 1024) or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("gguf_context_prediction_failed", error=str(exc)[:200])
        return None


def _context_ladder(requested: int) -> list[int]:
    """Context lengths to try, largest first.

    Halving rather than a fixed list: the gap between "fits" and "does not" is
    model- and card-specific, and a ladder derived from what was asked for
    lands close to the largest that works.
    """
    if requested <= 0:
        # 0 means "the model's full declared context" — one attempt, no ladder,
        # because there is no meaningful halving of "whatever the file says".
        return [0]
    ladder = []
    n = int(requested)
    while n >= GGUF_MIN_CONTEXT:
        ladder.append(n)
        n //= 2
    return ladder or [GGUF_MIN_CONTEXT]


#: Context window fallback. See settings.GGUF_CONTEXT_LENGTH for why this is
#: not 0: asking for the model's full declared context OOMs on a large-context
#: model, and llama.cpp's own default of 512 truncates real conversations.
GGUF_CONTEXT_FROM_FILE = 0

#: llama.cpp's MEAN pooling constant, resolved defensively: the module may be
#: absent (it is an extra), and the constant's name has moved between versions.
#: 1 is LLAMA_POOLING_TYPE_MEAN.
_POOLING_MEAN = getattr(llama_cpp_module, "LLAMA_POOLING_TYPE_MEAN", 1) if llama_cpp_module else 1


#: llama.cpp split modes, resolved defensively like the pooling constant: the
#: module is an extra. 0 = LLAMA_SPLIT_MODE_NONE (one GPU, `main_gpu`),
#: 1 = LLAMA_SPLIT_MODE_LAYER (layers and KV spread across GPUs).
_SPLIT_MODE_NONE = (
    getattr(llama_cpp_module, "LLAMA_SPLIT_MODE_NONE", 0) if llama_cpp_module else 0
)
_SPLIT_MODE_LAYER = (
    getattr(llama_cpp_module, "LLAMA_SPLIT_MODE_LAYER", 1) if llama_cpp_module else 1
)


def _gguf_can_offload() -> bool:
    """Whether this process can put GGUF layers on a card at all.

    BOTH halves are needed. `torch.cuda.is_available()` answers "is there a
    card", which says nothing about the llama.cpp build: the wheel on PyPI is
    CPU-only, so a CUDA box that installed it offloads nothing while torch
    happily reports a GPU. `llama_supports_gpu_offload()` is the other half.
    """
    if GGUF_GPU_LAYERS == 0 or not torch.cuda.is_available():
        return False
    if llama_supports_gpu_offload is not None:
        try:
            if not llama_supports_gpu_offload():
                return False
        except Exception:  # noqa: BLE001 - a probe must never fail a load
            pass
    return True


def _gguf_device(placement: Placement) -> str:
    """Where the GGUF weights actually landed: "cuda:N", "cuda:0, cuda:1", or "cpu".

    Reported, not requested. Answering from `GGUF_GPU_LAYERS` alone would
    assert a placement that never happened on a CPU-only wheel or a box with
    no card; answering "cuda" named no card at all on a two-GPU node.
    """
    if not _gguf_can_offload() or not placement.gpu_indices:
        return "cpu"
    return ", ".join(placement.device_labels)


def _gguf_required_mb(weights_mb: int, bytes_per_token: float | None, n_ctx: int) -> int:
    """Free memory a card needs for these weights, overhead and a KV cache of `n_ctx`.

    Scaled by the planning fraction for the same reason `predicted_max_context`
    is: planning to 100% of free memory picks a card that then fails to allocate.
    """
    kv_mb = (bytes_per_token * n_ctx / (1024 * 1024)) if bytes_per_token and n_ctx > 0 else 0
    return int((weights_mb + _GGUF_RUNTIME_OVERHEAD_MB + kv_mb) / _VRAM_PLANNING_FRACTION)


def _gguf_shard_rule(weights_mb: int, bytes_per_token: float | None, n_ctx: int) -> ShardRule:
    """A llama.cpp layer split: weights plus KV cache, spread over cards.

    Each card offers its free memory at the planning fraction LESS its own
    runtime overhead — the same arithmetic as the single-card check
    (`_gguf_required_mb`), applied per card. The transformers rule's flat
    reserve does not describe llama.cpp, whose per-card overhead is measured.
    """
    kv_mb = (bytes_per_token * n_ctx / (1024 * 1024)) if bytes_per_token and n_ctx > 0 else 0
    return ShardRule(
        need_mb=int(weights_mb + kv_mb),
        limit_mb=lambda gpu: int(gpu.free_mb * _VRAM_PLANNING_FRACTION) - _GGUF_RUNTIME_OVERHEAD_MB,
    )


def plan_gguf_placement(
    weights_mb: int,
    bytes_per_token: float | None,
    target_ctx: int,
    requested: GpuRequest = None,
    gpus: Optional[list[GpuInfo]] = None,
) -> Placement:
    """Where a GGUF model goes: one card, a layer split across cards, or the CPU.

    Auto: the most-free card that holds the weights plus the KV cache at the
    context the loader is aiming for. When no card does, a layer split over the
    cards with the most free memory, as few as hold it, each counted with its own
    runtime overhead. When even every card together does not, the split still
    names every card with room, every layer is offloaded (GGUF_GPU_LAYERS=-1)
    and the context ladder shrinks the window; a model whose weights alone
    exceed the cards does not load. There is NO partial CPU offload: operator
    decision 3 allows one for GGUF and nothing implements it (this docstring said
    the model "spills" to the CPU — review round 2, 2026-09-14). When no card has
    any room at all, the whole model runs on the CPU.

    An explicit card, or "all", is sized at the SMALLEST usable context
    instead: the context ladder shrinks the window to what the cards hold, so
    refusing cards that can serve the model at a shorter context would refuse a
    load that works. Cards that cannot hold even that are refused, never
    swapped.

    `gpus` defaults to the live inventory; the pre-unload check passes a
    projection of what the cards will have once the resident model is gone.
    """
    wanted = parse_gpu_request(requested)
    if not _gguf_can_offload():
        if wanted is not None:
            # A named card on a box (or build) that offloads nothing cannot be
            # honoured; running on the CPU instead is a silent substitution.
            #
            # choose_gpu still runs first, so an unknown card is GpuNotFoundError
            # and a full one is refused with its figures. But a card that EXISTS
            # and HAS ROOM must be refused too: this used to return choose_gpu's
            # single-card placement, the load then ran with n_gpu_layers=0 or a
            # CPU-only wheel, and the model served from the CPU under a
            # placement that said "requested_card".
            choose_gpu(
                _gguf_required_mb(weights_mb, bytes_per_token, GGUF_MIN_CONTEXT),
                requested=wanted,
                gpus=(list_gpus() if torch.cuda.is_available() else []) if gpus is None else gpus,
                shard=_gguf_shard_rule(weights_mb, bytes_per_token, GGUF_MIN_CONTEXT),
            )
            raise InsufficientMemoryError(
                f"GPU {wanted!r} was requested, but llama.cpp here offloads no layers "
                f"(GGUF_GPU_LAYERS={GGUF_GPU_LAYERS}, or a CPU-only build), so the model "
                "would run on the CPU. The requested card is not swapped for the CPU; "
                "choose Auto to run on the CPU, or enable GPU offload.",
                details={
                    "requested": wanted,
                    "gguf_gpu_layers": GGUF_GPU_LAYERS,
                    "gpu_offload": False,
                },
            )
        return cpu_placement(REASON_NO_GPU)

    gpus = list_gpus() if gpus is None else gpus
    if not gpus and wanted is None:
        return cpu_placement(REASON_NO_GPU)
    if wanted is not None:
        return choose_gpu(
            _gguf_required_mb(weights_mb, bytes_per_token, GGUF_MIN_CONTEXT),
            requested=wanted,
            gpus=gpus,
            shard=_gguf_shard_rule(weights_mb, bytes_per_token, GGUF_MIN_CONTEXT),
        )
    target = target_ctx or GGUF_MIN_CONTEXT
    placement = choose_gpu(
        _gguf_required_mb(weights_mb, bytes_per_token, target),
        gpus=gpus,
        shard=_gguf_shard_rule(weights_mb, bytes_per_token, target),
    )
    if placement.is_shard and not placement.gpu_indices:
        # Not one card has room for its own runtime overhead. A split over
        # nothing would still be handed n_gpu_layers=-1 and fail at every
        # context; GGUF may run on the CPU, so it does.
        return cpu_placement(REASON_NO_GPU, required_mb=placement.required_mb)
    return placement


def _gguf_tensor_split(
    placement: Placement, configured: Optional[list[float]] = None
) -> list[float]:
    """llama.cpp's `tensor_split` for a layer split: one proportion per CUDA index.

    llama.cpp indexes the list by DEVICE, not by the cards a load uses, and a
    layer split with no list spreads over every visible card. So the list is as
    long as the highest card used, with 0 for every card the plan left out —
    that zero is what keeps a model two cards hold off a third.

    `configured` (GGUF_TENSOR_SPLIT) names one proportion per card USED, in
    index order; without it the proportions are the plan's shares, so the cards
    with the most free memory carry the most layers.

    Raises:
        ModelLoadError: `configured` does not have one value per card used. A
            misconfigured split is not guessed at: stretched or truncated, it
            would put layers on cards nobody chose.
    """
    used = placement.gpu_indices
    if not used:
        return []
    if configured is not None:
        if len(configured) != len(used):
            raise GgufTensorSplitError(
                f"GGUF_TENSOR_SPLIT has {len(configured)} value(s) but this load "
                f"splits across {len(used)} GPU(s) ({', '.join(placement.device_labels)}). "
                "Give one proportion per card used, in index order, or leave it empty "
                "to split by free memory.",
                details={
                    "tensor_split": configured,
                    "gpu_indices": used,
                    "placement": placement.to_dict(),
                },
            )
        weights = dict(zip(used, configured))
    else:
        weights = {index: float(mb) for index, mb in placement.planned_mb_by_index.items()}
    total = sum(weights.values())
    vector = [0.0] * (max(used) + 1)
    for index, weight in weights.items():
        vector[index] = round(weight / total, 4) if total > 0 else 0.0
    return vector


def _gguf_placement_kwargs(
    placement: Placement, tensor_split: Optional[list[float]] = None
) -> dict[str, Any]:
    """llama.cpp kwargs that put the model where `placement` says.

    One card: LLAMA_SPLIT_MODE_NONE with `main_gpu` = that card, so llama.cpp
    allocates nothing on the others. Its default (layer split with main_gpu 0)
    spread every GGUF model over both cards and put its scratch buffers on the
    3080 Ti.

    A split: LLAMA_SPLIT_MODE_LAYER with a `tensor_split` naming only the cards
    the plan took (see `_gguf_tensor_split`). `tensor_split` is the operator's
    GGUF_TENSOR_SPLIT, already parsed.
    """
    if placement.is_single:
        return {"split_mode": _SPLIT_MODE_NONE, "main_gpu": placement.index}
    if placement.gpu_indices:
        return {
            "split_mode": _SPLIT_MODE_LAYER,
            "tensor_split": _gguf_tensor_split(placement, tensor_split),
        }
    return {}


def load_gguf_model(
    model_id: int,
    model_name: str,
    cache_path: str,
    gguf_file: str,
    gpu: GpuRequest = None,
) -> LoadedModel:
    """Load one GGUF quantization through llama.cpp.

    Deliberately NOT part of ModelLoadContext: that context is a long sequence
    of transformers-specific steps — attn-implementation probing,
    BitsAndBytesConfig, dtype selection, device_map, torch.compile with a
    three-pass soak — and a GGUF load shares none of it. Threading a branch
    through all of that would leave every step reading as if it applied.

    The returned `LoadedModel` carries `engine=ENGINE_LLAMACPP`, which is what
    every consumer branches on. `tokenizer` is None: llama.cpp tokenizes
    internally and exposes no HuggingFace-shaped tokenizer, and returning a
    half-working stand-in would let code that needs a real one fail late and
    obscurely instead of at the boundary.
    """
    if Llama is None:
        raise ModelLoadError(
            "llama-cpp-python is not installed, so GGUF models cannot be served. "
            "Install the 'gguf' extra.",
            details={"model_id": model_id, "gguf_file": gguf_file},
        )

    path = Path(cache_path) / gguf_file
    if not path.is_file():
        # A directory that looks populated but lacks the chosen file is exactly
        # what a partial download of a SPLIT quantization leaves behind.
        raise ModelLoadError(
            f"GGUF file not found: {path}",
            details={"model_id": model_id, "cache_path": cache_path, "gguf_file": gguf_file},
        )

    logger.info(
        "gguf_load_started", model_id=model_id, model_name=model_name, path=str(path)
    )
    from millm.core.config import settings as _settings

    weights_mb = int(path.stat().st_size / (1024 * 1024))
    bytes_per_token = gguf_kv_bytes_per_token(str(path), _settings.GGUF_KV_CACHE_TYPE)
    declared = declared_context(str(path))
    # The context the loader aims for before anything is measured: the tighter
    # of policy and training. It sizes the job for placement below.
    target_ctx = min(
        [bound for bound in (_settings.GGUF_CONTEXT_LENGTH, declared) if bound and bound > 0],
        default=0,
    )

    # WHICH CARD. Decided OUTSIDE the try below, which relabels every exception
    # as a load failure: a refused or unknown card must reach the caller as
    # itself, with the per-card figures in its details.
    placement = plan_gguf_placement(weights_mb, bytes_per_token, target_ctx, requested=gpu)
    logger.info(
        "gguf_placement",
        model_id=model_id,
        mode=placement.mode,
        reason=placement.reason,
        devices=placement.device_labels,
        required_mb=placement.required_mb,
        capacity_mb=placement.capacity_mb,
    )
    # Also outside the try: a GGUF_TENSOR_SPLIT that does not match the cards
    # used is a configuration error, and must not read as a failed load.
    placement_kwargs = _gguf_placement_kwargs(
        placement, parse_gguf_tensor_split(_settings.GGUF_TENSOR_SPLIT)
    )
    # nvidia-smi, not torch: llama.cpp never creates a torch context, and
    # reading the card through torch would create one just to measure.
    free_before = reported_free_mb_by_index(placement.gpu_indices)

    try:
        # START FROM WHAT THE MODEL DECLARES, capped by configuration.
        #
        # GGUF_CONTEXT_LENGTH is a CEILING, not a target. Starting the ladder at
        # a fixed 8192 served an 8192 window to a model trained for 262144 and
        # said nothing — MEASURED on ByteOtter/Qwen3.8-27B-TAK-Reasoning-GGUF,
        # which declares 262144 and creates a context at 131072 on this 24 GiB
        # card. The declared value is a property of the file; a config default
        # is a guess about a model it has never seen.
        #
        # The cap is not timidity: llama.cpp allocates the whole KV cache at
        # context creation, so an unbounded window reserves the card, and this
        # GPU is shared with miStudio's extraction, training and steering work.
        ceiling = _settings.GGUF_CONTEXT_LENGTH

        # WHAT WILL ACTUALLY FIT, computed rather than discovered.
        #
        # The ladder halving down from a configured ceiling costs one full model
        # load per rung — six of them to find a number the arithmetic gives in
        # milliseconds. A KV cache has an exact size, and every term is in the
        # metadata already read above.
        #
        # Budgeted against the card(s) the model is placed on: the chosen card
        # alone in single-GPU mode, the cards the split actually uses otherwise
        # — each with its own runtime overhead. This read GPU 0, so a model
        # placed on the 3090 had its window sized by the 3080 Ti's free memory.
        predicted = None
        try:
            free_mb = placement.capacity_mb
            if free_mb > 0:
                predicted = predicted_max_context(
                    str(path),
                    _settings.GGUF_KV_CACHE_TYPE,
                    free_mb,
                    weights_mb,
                    bytes_per_token=bytes_per_token,
                    n_cards=len(placement.gpu_indices),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("gguf_context_prediction_skipped", error=str(exc)[:200])

        if ceiling <= 0:
            # 0 is the documented "whatever the file declares" escape hatch, and
            # llama.cpp reads n_ctx=0 that way itself. Stated explicitly rather
            # than falling out of min(declared, 0), which is the same answer for
            # the wrong reason and would not survive a refactor.
            start_ctx = 0
        else:
            # The smallest of: what the model was trained for, what policy
            # allows, and what the card can hold. Each is a real bound and the
            # tightest one governs.
            bounds = [ceiling]
            if declared:
                bounds.append(declared)
            if predicted:
                bounds.append(predicted)
            start_ctx = min(bounds)
        logger.info(
            "gguf_context_target",
            model_id=model_id,
            declared=declared,
            ceiling=ceiling,
            predicted_fit=predicted,
            starting_at=start_ctx,
        )
        kv_kwargs = _kv_cache_kwargs(
            _settings.GGUF_KV_CACHE_TYPE, _settings.GGUF_FLASH_ATTENTION
        )
        kwargs: dict[str, Any] = {
            "model_path": str(path),
            # A placement with no card offloads nothing. With CUDA and a
            # GPU-capable build but an empty inventory (nvidia-smi absent or
            # timed out), placement says CPU, and -1 here made llama.cpp spread
            # every layer over every card, main_gpu 0, while the load recorded
            # "cpu" with no per-card memory.
            "n_gpu_layers": GGUF_GPU_LAYERS if placement.gpu_indices else 0,
            "n_ctx": start_ctx,
            "verbose": False,
            **kv_kwargs,
            **placement_kwargs,
        }
        if _settings.GGUF_ENABLE_EMBEDDINGS:
            # MEAN pooling, matching what the transformers path does —
            # `hidden_states[-1].mean(dim=1)` in create_embeddings. Choosing the
            # same pooling strategy is what makes the two engines' vectors
            # comparable in method rather than merely both being "embeddings".
            #
            # VERIFIED on the RTX 3090 rather than assumed: one instance with
            # this set serves BOTH create_embedding and create_chat_completion.
            # llama.cpp logs "embeddings required but some input tokens were not
            # marked as outputs -> overriding" and adapts.
            kwargs["embedding"] = True
            kwargs["pooling_type"] = _POOLING_MEAN
        # Try the configured context, then progressively smaller ones.
        #
        # A context that does not fit is not a failure worth propagating: the
        # model loads perfectly at a smaller one, and refusing leaves the
        # operator with "Failed to create llama_context" and no indication that
        # a single number stands between them and a working model. MEASURED on
        # gemma-4-31b Q4_K_M (17.4 GiB) on a 24 GiB card: 8192 fails — with or
        # without flash attention — 4096 loads with 1.5 GiB free, 2048 loads
        # with 3.3 GiB. There is no universal default; there is only what fits.
        #
        # The context actually obtained is RECORDED and logged. Serving a 2048
        # window while the configuration says 8192 would truncate long prompts
        # for reasons nothing on the system explains.
        llm = None
        attempted: list[int] = []
        requested = int(kwargs["n_ctx"])
        embeddings_dropped = False

        def _try_ladder() -> Any:
            """Walk the context ladder with the CURRENT kwargs. None = all failed.

            Raises immediately on a failure that the ladder cannot help with —
            see `_is_context_related`. Retrying those turns one honest error
            into fifteen misleading ones.
            """
            for n_ctx in _context_ladder(requested):
                attempted.append(n_ctx)
                kwargs["n_ctx"] = n_ctx
                try:
                    return Llama(**kwargs)
                except Exception as attempt_error:  # noqa: BLE001
                    if not _is_context_related(attempt_error):
                        logger.error(
                            "gguf_load_failed_not_a_context_problem",
                            model_id=model_id,
                            n_ctx=n_ctx,
                            error=str(attempt_error)[:300],
                            detail=(
                                "the model itself did not load; smaller "
                                "contexts and capability fallbacks cannot help"
                            ),
                        )
                        raise
                    logger.warning(
                        "gguf_context_attempt_failed",
                        model_id=model_id,
                        n_ctx=n_ctx,
                        embeddings=bool(kwargs.get("embedding")),
                        error=str(attempt_error)[:200],
                    )
                    gc.collect()
            return None

        llm = _try_ladder()

        def _bisect_upward(low: int, high: int) -> Any:
            """Recover the window halving steps over.

            The ladder divides by two, so the answer is only ever within a
            FACTOR OF TWO of the true ceiling. MEASURED on gemma-4-31b IQ4_XS:
            the ladder settles at 8192 while 12288 loads on the same card —
            half the usable window thrown away for the sake of a tidy sequence.

            Bisects the gap between the last failure and the first success.
            Bounded to three probes: each is a real model load, and the
            remaining gain halves every time while the cost does not.
            """
            best = None
            for _ in range(3):
                mid = ((low + high) // 2 // 1024) * 1024
                if mid <= low or mid >= high:
                    break
                kwargs["n_ctx"] = mid
                attempted.append(mid)
                try:
                    candidate = Llama(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    if not _is_context_related(exc):
                        raise
                    high = mid
                    gc.collect()
                    continue
                if best is not None:
                    best.close()
                best = candidate
                low = mid
            return best, low

        # The ladder found `low` works and the rung above it does not; the true
        # ceiling is between them. Only worth probing when that gap is wide
        # enough to matter — a 1024-token gain does not justify a model load.
        if llm is not None and len(attempted) > 1:
            settled = int(kwargs["n_ctx"])
            failed_above = attempted[attempted.index(settled) - 1]
            if failed_above - settled > 2048:
                better, reached = _bisect_upward(settled, failed_above)
                if better is not None and reached > settled:
                    llm.close()
                    llm = better
                    logger.info(
                        "gguf_context_bisected_upward",
                        model_id=model_id,
                        ladder_settled_at=settled,
                        actually_reached=reached,
                        detail="halving alone would have served the smaller window",
                    )
                else:
                    reached = settled
                # RESTORE UNCONDITIONALLY. `_bisect_upward` assigns
                # kwargs["n_ctx"] before each probe and a failed probe leaves it
                # there, so the recorded context would be a length that DID NOT
                # LOAD — 5120 reported for a model actually serving 4096. That
                # is the same class of lie the ladder itself exists to prevent,
                # and the pre-existing ladder test caught it.
                kwargs["n_ctx"] = reached

        # EMBEDDINGS ARE OPTIONAL; SERVING THE MODEL IS NOT.
        #
        # `pooling_type=MEAN` is refused outright by some architectures —
        # llama.cpp logs "model default pooling_type is [-1], but [1] was
        # specified" and llama_context creation fails at EVERY context length,
        # because the context size was never the problem. Measured on
        # ByteOtter/Qwen3.8-27B-TAK-Reasoning-GGUF (7.8 GiB) on an otherwise
        # EMPTY 24 GiB card: fails at 8192/4096/2048 with the flag, loads at
        # 2048 without it.
        #
        # Before this, that model was unloadable and the operator was told "may
        # not fit on this GPU at all — try a smaller quantization", which is
        # advice that cannot work: the next quantization fails identically, and
        # the card was 98% free the whole time.
        #
        # So drop the capability, not the model. Which one is available is
        # recorded on the row rather than discovered by a caller getting a
        # confusing failure from /v1/embeddings.
        # A QUANTIZED KV CACHE THE BUILD WILL NOT TAKE MUST NOT COST THE MODEL.
        #
        # `type_k`/`type_v` and `flash_attn` depend on how llama.cpp was
        # compiled and on the architecture. When they are refused, the whole
        # ladder fails for a reason that has nothing to do with context length
        # — exactly the shape of the pooling_type defect below, which cost a
        # working model and sent an operator to download a smaller one.
        #
        # Falling back to f16 costs context (gemma-4-31b drops from 12288 to
        # 4096) and keeps the model servable, which is the right way round.
        if llm is None and kv_kwargs:
            logger.warning(
                "gguf_retrying_with_f16_kv_cache",
                model_id=model_id,
                attempted=list(attempted),
                kv_cache_type=_settings.GGUF_KV_CACHE_TYPE,
                detail=(
                    "no context could be created with a quantized KV cache; "
                    "this build or architecture refuses it. Retrying at f16, "
                    "which holds a SMALLER context for the same memory."
                ),
            )
            for key in ("flash_attn", "type_k", "type_v"):
                kwargs.pop(key, None)
            kv_kwargs = {}
            attempted = []
            llm = _try_ladder()

        if llm is None and kwargs.get("embedding"):
            logger.warning(
                "gguf_retrying_without_embeddings",
                model_id=model_id,
                attempted=list(attempted),
                detail=(
                    "no context could be created with embeddings enabled; this "
                    "architecture refuses the pooling type. Retrying WITHOUT "
                    "embeddings — chat and completions will work, /v1/embeddings "
                    "will not."
                ),
            )
            kwargs.pop("embedding", None)
            kwargs.pop("pooling_type", None)
            attempted = []
            llm = _try_ladder()
            embeddings_dropped = llm is not None

        if llm is None:
            raise ModelLoadError(
                "Could not create a llama.cpp context at any context length "
                f"(tried {attempted}), with embeddings disabled as a fallback. "
                "Check the backend log for llama.cpp's own diagnostic on each "
                "attempt — the cause is recorded there and is not always "
                "memory.",
                details={"model_id": model_id, "attempted": attempted},
            )

        if embeddings_dropped:
            logger.warning(
                "gguf_embeddings_unavailable",
                model_id=model_id,
                n_ctx=kwargs["n_ctx"],
                detail=(
                    "loaded WITHOUT embedding support: this architecture "
                    "refuses MEAN pooling. Chat and completions are unaffected."
                ),
            )
        if kwargs["n_ctx"] != requested:
            logger.warning(
                "gguf_context_reduced",
                model_id=model_id,
                requested=requested,
                actual=kwargs["n_ctx"],
                detail=(
                    "the configured context did not fit; long prompts will be "
                    "truncated at the smaller window"
                ),
            )
    except Exception as e:  # noqa: BLE001 - surfaced as a load failure
        raise ModelLoadError(
            f"Failed to load GGUF model: {e}",
            details={"model_id": model_id, "path": str(path)},
        ) from e

    size_mb = int(path.stat().st_size / (1024 * 1024))
    device = _gguf_device(placement)
    gguf_gpu_indices = placement.gpu_indices if device != "cpu" else []
    # nvidia-smi reads the driver's free memory, so unlike the torch allocator
    # it DOES see llama.cpp's weights and KV cache.
    memory_by_device_mb = memory_used_by_device(
        free_before, reported_free_mb_by_index(gguf_gpu_indices), gguf_gpu_indices
    )
    logger.info(
        "gguf_load_complete",
        model_id=model_id,
        size_mb=size_mb,
        device=device,
        memory_by_device_mb=memory_by_device_mb,
    )

    return LoadedModel(
        model_id=model_id,
        model_name=model_name,
        model=llm,
        tokenizer=None,
        loaded_at=datetime.utcnow(),
        # The file's size on disk, not a torch measurement: llama.cpp allocates
        # outside torch, so torch.cuda.mem_get_info would not attribute it here.
        memory_used_mb=size_mb,
        # What llama.cpp ACTUALLY used, not what we asked for: with
        # n_gpu_layers=-1 on a CPU-only build (or a box with no card) the
        # weights are in host RAM, and recording "cuda" there would make
        # /api/models/status assert a placement that never happened.
        #
        # BOTH halves are needed. `torch.cuda.is_available()` answers "is there
        # a card", which says nothing about the llama.cpp build: the wheel on
        # PyPI is CPU-only, so a CUDA box that installed it offloads nothing
        # while torch happily reports a GPU. `llama_supports_gpu_offload()` is
        # the other half, and it is asked of the library that did the loading.
        device=device,
        gpu_indices=gguf_gpu_indices,
        memory_by_device_mb=memory_by_device_mb,
        placement={
            **placement.to_dict(),
            "devices": device.split(", "),
            "gpu_indices": gguf_gpu_indices,
        },
        dtype="gguf",
        attn_implementation="llama.cpp",
        # The label comes from the catalogue's parser, which reads the
        # quantization token wherever it sits in the name. The previous
        # `stem.rsplit(".")[-1]` assumed a dot before it: correct for
        # "Model.Q4_K_M.gguf", but "qwen2.5-7b-instruct-q4_k_m.gguf" is a
        # perfectly ordinary name and it yielded "5-7b-instruct-q4_k_m".
        quantization_method=f"gguf:{quant_label_from_path(gguf_file) or 'unknown'}",
        engine=ENGINE_LLAMACPP,
        context_length=int(kwargs["n_ctx"]),
        supports_embeddings=bool(kwargs.get("embedding")),
    )


# =============================================================================
# Per-card fit of a transformers load (Decision 7, 2026-09-14)
# =============================================================================

#: The KV cache of a transformers load holds bfloat16 tensors, 2 bytes an element.
#: ModelLoadContext.load loads every model with torch_dtype=torch.bfloat16
#: (bitsandbytes computes in it too), and nothing on the transformers path
#: quantizes the cache: KV_CACHE_MODE is "dynamic" or "static", and
#: GGUF_KV_CACHE_TYPE (q8_0) applies to llama.cpp alone.
TRANSFORMERS_KV_BYTES = 2

#: Layer types as transformers' DynamicCache reads them (transformers 5.15.1,
#: cache_utils.get_layer_types_and_kwargs and DYNAMIC_LAYER_TYPE_MAPPING), with
#: the legacy names configuration_utils remaps ("attention", "mamba"). A type
#: not listed here is not sized: the load falls back to the slack.
_KV_FULL_LAYERS = frozenset({"full_attention", "attention", "hybrid"})
#: Sliding and chunked layers keep only their window of tokens
#: (DynamicSlidingWindowLayer); the value names the config field holding it.
_KV_WINDOWED_LAYERS = {
    "sliding_attention": "sliding_window",
    "hybrid_sliding": "sliding_window",
    "chunked_attention": "attention_chunk_size",
}
#: Layers with no per-token cache. Their convolution or recurrent state is fixed
#: in size and NOT counted: it cannot be derived exactly from every config.
_KV_FREE_LAYERS = frozenset({"conv", "linear_attention", "mamba", "moe", "mlp"})
#: Config fields meaning a token's cache is not 2 x key-value heads x head_dim a
#: layer: multi-head latent attention, layers reusing another's cache,
#: cross-attention to an encoder, and keys shared with values.
_KV_UNSIZED_FIELDS = (
    "kv_lora_rank",
    "num_kv_shared_layers",
    "cross_attention_layers",
    "attention_k_eq_v",
)


def refuse_unsupported_quantization(
    quantization: str, is_pre_quantized: bool, estimated_memory_mb: int
) -> None:
    """Refuse Q2 on a transformers checkpoint that is not already quantized.

    bitsandbytes has no 2-bit mode, so ModelLoadContext.load gives it no
    quantization config and it loads in bfloat16. See decide_transformers_placement.
    """
    if quantization.upper() == "Q2" and not is_pre_quantized:
        raise UnsupportedQuantizationError(
            "Q2 cannot be loaded as a transformers model: bitsandbytes has no 2-bit "
            "mode, so this checkpoint would load unquantized in bfloat16, eight times "
            "the memory its Q2 estimate assumes. Load it as Q4, Q8 or FP16, or serve a "
            "Q2 GGUF of the model.",
            details={
                "quantization": quantization,
                "estimated_memory_mb": estimated_memory_mb,
                "loads_as": "bfloat16",
            },
        )


def split_max_memory_factor(
    quantization: str, is_pre_quantized: bool, pre_quantized_factor: float
) -> float:
    """What transformers multiplies a split's `max_memory` by for this load.

    Only Q4 and Q8 get a BitsAndBytesConfig in ModelLoadContext.load, and only
    for a checkpoint that is not already quantized. A pre-quantized checkpoint
    gets the quantizer its own config names, with that quantizer's factor.
    """
    if is_pre_quantized:
        return pre_quantized_factor
    if quantization.upper() in ("Q4", "Q8"):
        return BNB_MAX_MEMORY_FACTOR
    return 1.0


def _architecture_name(config: Any) -> str:
    architectures = getattr(config, "architectures", None) or []
    if architectures and isinstance(architectures[0], str):
        return architectures[0]
    return str(getattr(config, "model_type", None) or type(config).__name__)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class KvCacheSpec:
    """What one token of KV cache costs each decoder layer, as transformers allocates it.

    `bytes_per_token[i]` is 2 x key-value heads x head_dim x TRANSFORMERS_KV_BYTES
    for a layer with a cache and 0 for one without; `token_cap[i]` is a sliding
    or chunked layer's window — all the tokens it keeps — or None.
    """

    architecture: str
    bytes_per_token: tuple[int, ...]
    token_cap: tuple[Optional[int], ...]
    #: The longest context the model serves: its text config's
    #: `max_position_embeddings`, which InferenceService._check_context_length
    #: enforces on every request. None when the config does not say.
    max_context: Optional[int] = None

    @property
    def num_layers(self) -> int:
        return len(self.bytes_per_token)

    def mb(self, layers: Iterable[int], tokens: int) -> int:
        """The cache of these decoder layers holding `tokens` tokens, in MB, rounded up."""
        total = 0
        for index in layers:
            cap = self.token_cap[index]
            total += self.bytes_per_token[index] * (min(tokens, cap) if cap else tokens)
        return math.ceil(total / (1024 * 1024))


def _layer_config(text: Any, index: int) -> Any:
    """Decoder layer `index`'s config with its own overrides (Gemma 4's global layers)."""
    if not getattr(text, "is_heterogeneous", False):
        return text
    return text.per_layer_config[index]


def kv_cache_spec(config: Any) -> tuple[Optional[KvCacheSpec], str]:
    """The KV cache a model's config says transformers allocates; (None, why) when unsure.

    Mirrors DynamicCache: `layer_types` when the config has it, else every layer
    sliding when `sliding_window` is set, chunked when `attention_chunk_size` is,
    full otherwise. An attention layer costs 2 x key-value heads x head_dim x 2
    bytes a token (keys and values, bfloat16), read per layer so a layer with
    its own head_dim or key-value heads is sized as it is. A sliding or chunked
    layer keeps at most its window. A linear-attention, Mamba or convolution
    layer contributes nothing per token.

    Not derivable, so (None, reason) and the load keeps the 20% slack: an
    encoder-decoder, a config with a field that changes what a token costs
    (_KV_UNSIZED_FIELDS), a layer type transformers may cache in a way this does
    not model, or fields missing.
    """
    architecture = _architecture_name(config)
    if getattr(config, "is_encoder_decoder", False):
        return None, "it is an encoder-decoder model"
    get_text_config = getattr(config, "get_text_config", None)
    text = get_text_config(decoder=True) if callable(get_text_config) else config
    for field_name in _KV_UNSIZED_FIELDS:
        if getattr(text, field_name, None):
            return None, (
                f"its config sets {field_name}, so a token's cache is not "
                "2 x key-value heads x head_dim a layer"
            )
    num_layers = getattr(text, "num_hidden_layers", None)
    if not _positive_int(num_layers):
        return None, "its config has no num_hidden_layers"
    layer_types = getattr(text, "layer_types", None)
    if layer_types is None:
        if getattr(text, "sliding_window", None) is not None:
            layer_types = ["sliding_attention"] * num_layers
        elif getattr(text, "attention_chunk_size", None) is not None:
            layer_types = ["chunked_attention"] * num_layers
        else:
            layer_types = ["full_attention"] * num_layers
    layer_types = list(layer_types)
    if len(layer_types) != num_layers:
        return None, f"its layer_types names {len(layer_types)} layers of {num_layers}"

    bytes_per_token: list[int] = []
    token_cap: list[Optional[int]] = []
    for index, layer_type in enumerate(layer_types):
        if layer_type in _KV_FREE_LAYERS:
            bytes_per_token.append(0)
            token_cap.append(None)
            continue
        if layer_type not in _KV_FULL_LAYERS and layer_type not in _KV_WINDOWED_LAYERS:
            return None, f"layer {index} is a {layer_type!r} layer, which miLLM does not size"
        layer = _layer_config(text, index)
        heads = getattr(layer, "num_attention_heads", None)
        kv_heads = getattr(layer, "num_key_value_heads", None) or heads
        head_dim = getattr(layer, "head_dim", None)
        if not _positive_int(head_dim):
            hidden = getattr(layer, "hidden_size", None)
            if not (_positive_int(hidden) and _positive_int(heads) and hidden % heads == 0):
                return None, (
                    f"layer {index} has no head_dim, and hidden_size is not a multiple "
                    "of num_attention_heads"
                )
            head_dim = hidden // heads
        if not _positive_int(kv_heads):
            return None, f"layer {index} has no num_key_value_heads or num_attention_heads"
        cap: Optional[int] = None
        if layer_type in _KV_WINDOWED_LAYERS:
            window_field = _KV_WINDOWED_LAYERS[layer_type]
            window = getattr(layer, window_field, None)
            if not _positive_int(window):
                return None, f"layer {index} is a {layer_type!r} layer with no {window_field}"
            cap = int(window)
        bytes_per_token.append(2 * int(kv_heads) * int(head_dim) * TRANSFORMERS_KV_BYTES)
        token_cap.append(cap)
    max_context = getattr(text, "max_position_embeddings", None)
    return KvCacheSpec(
        architecture,
        tuple(bytes_per_token),
        tuple(token_cap),
        max_context=int(max_context) if _positive_int(max_context) else None,
    ), ""


def admitted_context(spec: KvCacheSpec, configured: int) -> int:
    """The context a load's KV cache is sized at: TRANSFORMERS_MIN_CONTEXT, or less.

    Never more than the model serves. Every request is refused past the text
    config's `max_position_embeddings` (InferenceService._check_context_length),
    so a cache sized beyond it holds tokens no request can send. Sized at the
    setting, OLMo-2-13B and Vicuna-13B (4,096 each) were refused at 8,192 for
    memory their 8k cache would need, on cards that serve everything they can
    take. Review round 5, 2026-09-14.
    """
    if spec.max_context is not None:
        return min(int(configured), spec.max_context)
    return int(configured)


@dataclass(frozen=True)
class CardFit:
    """One card of a transformers load: its free memory, and what it must hold."""

    index: int
    name: str
    free_mb: int
    weights_mb: int
    kv_mb: int
    context_mb: int
    layers: int

    @property
    def need_mb(self) -> int:
        return self.weights_mb + self.kv_mb + self.context_mb

    @property
    def short_mb(self) -> int:
        return max(self.need_mb - self.free_mb, 0)

    @property
    def fits(self) -> bool:
        return self.need_mb <= self.free_mb

    def describe(self, tokens: int) -> str:
        verdict = (
            f"{self.short_mb} MiB short" if not self.fits
            else f"{self.free_mb - self.need_mb} MiB to spare"
        )
        return (
            f"cuda:{self.index} ({self.name}) has {self.free_mb} MiB free for {self.weights_mb} "
            f"MiB of weights, {self.kv_mb} MiB of KV cache at {tokens} tokens over its "
            f"{self.layers} layers and a {self.context_mb} MiB CUDA context — {verdict}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": f"cuda:{self.index}",
            "name": self.name,
            "free_mb": self.free_mb,
            "weights_mb": self.weights_mb,
            "kv_mb": self.kv_mb,
            "context_mb": self.context_mb,
            "layers": self.layers,
            "need_mb": self.need_mb,
            "short_mb": self.short_mb,
        }


@dataclass(frozen=True)
class SplitLayout:
    """Where transformers' own device map puts a split: MB and decoder layers per device."""

    weights_mb: dict[str, int]
    layers: dict[str, tuple[int, ...]]
    off_gpu: tuple[str, ...] = ()
    #: bitsandbytes' own refusal of a map leaving the GPU, when it raised one.
    engine_message: Optional[str] = None


def _device_map_label(device: Any) -> str:
    return f"cuda:{device}" if isinstance(device, int) and not isinstance(device, bool) else str(device)


def _mapped_device(name: str, mapped: dict[str, Any]) -> Any:
    """The device a module lands on: its own map entry, or its nearest mapped ancestor's."""
    best, device = -1, None
    for key, value in mapped.items():
        if key == "" or name == key or name.startswith(key + "."):
            if len(key) > best:
                best, device = len(key), value
    return device


def _decoder_layer_names(model: Any, num_layers: int) -> Optional[list[str]]:
    """The module names of a model's decoder layers, in order; None when they cannot be found."""
    names = {id(module): name for name, module in model.named_modules()}
    try:
        decoder = model.get_decoder()
    except Exception:  # noqa: BLE001 - fall back to the one "layers" list of that length
        decoder = None
    layers = getattr(decoder, "layers", None)
    if isinstance(layers, torch.nn.ModuleList) and len(layers) == num_layers and id(layers) in names:
        prefix = names[id(layers)]
    else:
        found = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.ModuleList)
            and name.rsplit(".", 1)[-1] == "layers"
            and len(module) == num_layers
        ]
        if len(found) != 1:
            return None
        prefix = found[0]
    return [f"{prefix}.{index}" for index in range(num_layers)]


@dataclass(frozen=True)
class TransformersFit:
    """A checkpoint sized for the per-card fit: its weights as loaded and its KV cache.

    `weights_mb` is what transformers materialises, measured on the meta device
    with the quantizer the load uses (bitsandbytes for Q4/Q8, the checkpoint's own
    for a pre-quantized one). `model`, `hf_quantizer` and `sizes` are kept to
    compute a split's layout the way from_pretrained does.
    """

    architecture: str
    weights_mb: int
    kv: KvCacheSpec
    min_context: int
    context_mb: int
    model: Any = field(repr=False, compare=False)
    hf_quantizer: Any = field(repr=False, compare=False)
    sizes: dict[str, int] = field(repr=False, compare=False)

    def kv_mb(self, layers: Optional[Iterable[int]] = None) -> int:
        return self.kv.mb(range(self.kv.num_layers) if layers is None else layers, self.min_context)

    @property
    def single_card_mb(self) -> int:
        """What one card holding the whole model needs."""
        return self.weights_mb + self.kv_mb() + self.context_mb

    def card(self, gpu: GpuInfo, weights_mb: int, layers: Iterable[int]) -> CardFit:
        on_card = tuple(layers)
        return CardFit(
            index=gpu.index,
            name=gpu.name,
            free_mb=gpu.free_mb,
            weights_mb=weights_mb,
            kv_mb=self.kv_mb(on_card),
            context_mb=self.context_mb,
            layers=len(on_card),
        )

    def layout(self, placement: Placement) -> Optional[SplitLayout]:
        """transformers' own device map for this split; None when it cannot be computed here."""
        names = _decoder_layer_names(self.model, self.kv.num_layers)
        if names is None:
            logger.error(
                "transformers_fit_layers_not_found",
                architecture=self.architecture,
                num_layers=self.kv.num_layers,
            )
            return None
        try:
            from transformers.integrations.accelerate import _get_device_map

            max_memory = placement.transformers_max_memory()
            mapped = _get_device_map(
                self.model,
                placement.transformers_device_map(),
                dict(max_memory) if max_memory else None,
                self.hf_quantizer,
            )
        except ValueError as e:
            if _is_offload_refusal(e):
                return SplitLayout({}, {}, ("cpu or disk",), engine_message=str(e))
            _log_unverified(
                _STAGE_ENGINE, e, unverifiable_event="transformers_fit_layout_unknown",
                engine_event="transformers_fit_engine_failed", architecture=self.architecture,
            )
            return None
        except Exception as e:  # noqa: BLE001 - judged by the slack instead
            _log_unverified(
                _STAGE_ENGINE, e, unverifiable_event="transformers_fit_layout_unknown",
                engine_event="transformers_fit_engine_failed", architecture=self.architecture,
            )
            return None
        weights: dict[str, int] = {}
        for name, device in mapped.items():
            label = _device_map_label(device)
            weights[label] = weights.get(label, 0) + int(self.sizes.get(name, 0) / (1024 * 1024))
        layers: dict[str, list[int]] = {}
        for index, name in enumerate(names):
            layers.setdefault(_device_map_label(_mapped_device(name, mapped)), []).append(index)
        allowed = set(placement.device_labels)
        return SplitLayout(
            weights_mb=weights,
            layers={label: tuple(indices) for label, indices in layers.items()},
            off_gpu=tuple(sorted(label for label in weights if label not in allowed)),
        )


def _fit_falls_back(
    architecture: str,
    reason: str,
    error: Optional[BaseException] = None,
    stage: Optional[str] = None,
) -> None:
    """Say, loudly, that a transformers load is judged by the 20% slack instead of per card."""
    fields: dict[str, Any] = {
        "architecture": architecture,
        "reason": reason,
        "slack_factor": MEMORY_OVERHEAD_FACTOR,
    }
    if error is not None:
        fields.update(stage=stage, error_type=type(error).__name__, error=str(error)[:300])
        if stage == _STAGE_ENGINE:
            fields["transformers_version"] = _transformers_version()
    logger.error("transformers_fit_falls_back_to_slack", **fields)


def transformers_fit(
    cache_path: Optional[str],
    quantization: str,
    is_pre_quantized: bool,
    trust_remote_code: bool = False,
) -> Optional[TransformersFit]:
    """Size a checkpoint for the per-card fit; None (logged loudly) to keep the slack.

    Reads TRANSFORMERS_MIN_CONTEXT and TRANSFORMERS_CUDA_CONTEXT_MB when called,
    so both call sites of a load judge it with the settings in force.
    """
    from millm.core.config import settings

    if not cache_path or AutoConfig is None:
        _fit_falls_back("unknown", "there is no checkpoint config to read")
        return None
    architecture = "unknown"
    stage = _STAGE_ENGINE  # the imports are transformers' private API
    try:
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.quantizers.auto import get_hf_quantizer

        stage = _STAGE_CHECKPOINT
        config = AutoConfig.from_pretrained(cache_path, trust_remote_code=trust_remote_code)
        architecture = _architecture_name(config)
        kv, reason = kv_cache_spec(config)
        if kv is None:
            _fit_falls_back(architecture, reason)
            return None
        quantization_config = None if is_pre_quantized else _bitsandbytes_config(quantization)
        hf_quantizer, config, device_map = get_hf_quantizer(
            config, quantization_config, "sequential", True, {}
        )
        if is_pre_quantized and hf_quantizer is None:
            _fit_falls_back(architecture, "its quantization method has no transformers quantizer")
            return None
        model = _meta_model(config, trust_remote_code)
        # Config, class and quantizer all built: from here a failure is transformers'.
        stage = _STAGE_ENGINE
        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model,
                dtype=torch.bfloat16,
                device_map=device_map,
                checkpoint_files=None,
                use_kernels=False,
            )
        sizes, _ = compute_module_sizes(model, hf_quantizer, only_modules=False)
    except Exception as e:  # noqa: BLE001 - unsized here; the slack judges it
        _fit_falls_back(
            architecture,
            "its model could not be built on the meta device here"
            if stage == _STAGE_CHECKPOINT
            else "transformers' own sizing failed",
            error=e,
            stage=stage,
        )
        return None
    weights_mb = int(sizes.get("", 0) / (1024 * 1024))
    if weights_mb <= 0:
        _fit_falls_back(architecture, "its model sizes to no weights")
        return None
    return TransformersFit(
        architecture=architecture,
        weights_mb=weights_mb,
        kv=kv,
        min_context=admitted_context(kv, int(settings.TRANSFORMERS_MIN_CONTEXT)),
        context_mb=int(settings.TRANSFORMERS_CUDA_CONTEXT_MB),
        model=model,
        hf_quantizer=hf_quantizer,
        sizes=dict(sizes),
    )


def per_card_fit_refusal(
    fit: TransformersFit,
    cards: list[CardFit],
    detail: str,
    requested: Any,
    gpus: Iterable[GpuInfo],
    placement: Optional[Placement] = None,
    off_gpu_mb: Optional[dict[str, int]] = None,
    mapped_mb_by_device: Optional[dict[str, int]] = None,
) -> InsufficientMemoryError:
    """The refusal of a transformers load that does not fit per card, naming every card."""
    off_gpu_mb = dict(sorted((off_gpu_mb or {}).items()))
    listing = "; ".join(card.describe(fit.min_context) for card in cards) or "no card takes any of it"
    details: dict[str, Any] = {
        "required_mb": sum(card.need_mb for card in cards) + sum(off_gpu_mb.values()),
        "available_mb": sum(card.free_mb for card in cards),
        "short_devices": [f"cuda:{card.index}" for card in cards if not card.fits],
        "per_card": [card.to_dict() for card in cards],
        "architecture": fit.architecture,
        "weights_mb": fit.weights_mb,
        "kv_mb": fit.kv_mb(),
        "min_context_tokens": fit.min_context,
        "model_max_context_tokens": fit.kv.max_context,
        "cuda_context_mb": fit.context_mb,
        "requested": requested,
        "gpus": [gpu.to_dict() for gpu in gpus],
        "before_loading": True,
    }
    if off_gpu_mb:
        details["off_gpu"] = sorted(off_gpu_mb)
        details["off_gpu_mb_by_device"] = off_gpu_mb
    if placement is not None:
        details["placement"] = placement.to_dict()
    if mapped_mb_by_device is not None:
        details["mapped_mb_by_device"] = dict(sorted(mapped_mb_by_device.items()))
    return InsufficientMemoryError(
        f"Not enough GPU memory for this {fit.architecture} model with a "
        f"{fit.min_context}-token context on each card it uses: {listing}. {detail}",
        details=details,
    )


def _check_split_fit(fit: TransformersFit, placement: Placement) -> Placement:
    """Accept a split only when transformers' own map leaves each used card room for its context."""
    if not placement.gpu_indices:
        raise shard_refusal(placement, fit.weights_mb, "No GPU has memory to spare for it.")
    layout = fit.layout(placement)
    if layout is None:
        # The map could not be computed here, and it was logged as an error:
        # this split is judged by the slack it replaced.
        need = int(fit.weights_mb * MEMORY_OVERHEAD_FACTOR)
        if placement.budget_mb < need:
            raise shard_refusal(
                placement,
                need,
                "Its layout could not be worked out here, so it was judged by the 20% slack. "
                "A transformers model is never offloaded to the CPU or disk.",
            )
        return placement
    if layout.engine_message is not None:
        raise _off_gpu_refusal(
            fit.architecture, placement, list(layout.off_gpu), [],
            engine_message=layout.engine_message, before_loading=True,
        )
    by_index = {gpu.index: gpu for gpu in placement.gpus}
    cards = [
        fit.card(by_index[index], layout.weights_mb[label], layout.layers.get(label, ()))
        for index in placement.gpu_indices
        if (label := f"cuda:{index}") in layout.weights_mb
    ]
    off_gpu_mb = {label: layout.weights_mb[label] for label in layout.off_gpu}
    if off_gpu_mb or not all(card.fits for card in cards):
        where = (
            f"{sum(off_gpu_mb.values())} MiB would be placed on {', '.join(off_gpu_mb)}, and a "
            "transformers model is never offloaded. " if off_gpu_mb else ""
        )
        raise per_card_fit_refusal(
            fit,
            cards,
            where + "Free memory on the cards, lower TRANSFORMERS_MIN_CONTEXT, choose a smaller "
            "quantization, or serve it as GGUF.",
            requested=placement.requested,
            gpus=placement.gpus,
            placement=placement,
            off_gpu_mb=off_gpu_mb,
            mapped_mb_by_device=layout.weights_mb,
        )
    logger.info(
        "transformers_fit_split_accepted",
        architecture=fit.architecture,
        per_card=[card.to_dict() for card in cards],
        min_context_tokens=fit.min_context,
    )
    return placement


def decide_transformers_fit(
    fit: TransformersFit,
    requested: GpuRequest,
    gpus: list[GpuInfo],
    max_memory_factor: float = 1.0,
) -> Placement:
    """Where a transformers load goes when every card can be judged on its own.

    Decision 7 (user, 2026-09-14) replaced the 20% slack on the weight estimate
    here. A load is accepted only when each card it uses has room, beside its
    weights, for its CUDA context (TRANSFORMERS_CUDA_CONTEXT_MB) and the KV cache
    of the layers it holds at TRANSFORMERS_MIN_CONTEXT tokens:

      * one card (Auto's choice or a named card): the whole model on that card;
      * a split (Auto's, when no card holds it, or "all"): the layout
        transformers' own device map computes from the plan's `max_memory`.

    The slack grew with the weights, not with what a card needs. On 11,500 and
    23,500 MB free it refused Qwen2.5-14B at FP16, whose map leaves both cards
    room for about 8k tokens, and accepted OLMo-2-13B, whose cuda:0 cannot hold
    an 8k cache — and it split models one card holds (a 7B at FP16 on a card with
    17 GB free: weights 14.5 GB x 1.2 > 17 GB). Nothing is re-planned to make a
    card fit: a card that is short is named, with its figures, in the refusal.
    """
    inventory = list(gpus)
    wanted = parse_gpu_request(requested)
    if not inventory:
        raise InsufficientMemoryError(
            "No GPU is visible to miLLM.",
            details={"required_mb": fit.single_card_mb, "available_mb": 0, "gpus": []},
        )
    everything = range(fit.kv.num_layers)
    rule = transformers_shard_rule(fit.weights_mb, max_memory_factor=max_memory_factor)

    if wanted == ALL:
        placement = plan_shard(
            inventory, rule, REASON_REQUESTED_ALL, fit.single_card_mb, requested=ALL,
            every_card=True,
        )
        refuse_cards_left_out_of_all(placement, inventory, rule)
        return _check_split_fit(fit, placement)

    if wanted is not None:
        card = find_gpu(inventory, wanted)
        judged = fit.card(card, fit.weights_mb, everything)
        if not judged.fits:
            raise per_card_fit_refusal(
                fit,
                [judged],
                "The requested card is not swapped for another one; choose a different "
                "card or Auto.",
                requested=wanted,
                gpus=inventory,
            )
        return Placement(
            mode=MODE_SINGLE,
            reason=REASON_REQUESTED,
            required_mb=judged.need_mb,
            gpus=tuple(inventory),
            index=card.index,
            requested=wanted,
        )

    # Most free first; on a tie the lower index, so the choice is stable.
    best = max(inventory, key=lambda gpu: (gpu.free_mb, -gpu.index))
    judged = fit.card(best, fit.weights_mb, everything)
    if judged.fits:
        return Placement(
            mode=MODE_SINGLE,
            reason=REASON_MOST_FREE,
            required_mb=judged.need_mb,
            gpus=tuple(inventory),
            index=best.index,
        )
    return _check_split_fit(
        fit, plan_shard(inventory, rule, REASON_NO_SINGLE_CARD, fit.single_card_mb)
    )


def decide_transformers_placement(
    estimated_memory_mb: int,
    quantization: str,
    requested: GpuRequest,
    gpus: list[GpuInfo],
    is_pre_quantized: bool = False,
    pre_quantized_max_memory_factor: float = 1.0,
) -> Placement:
    """Where a transformers load goes, or why it cannot go anywhere.

    Call sites that plan a real load go through `plan_transformers_load`, which
    reads every checkpoint-dependent argument here from the checkpoint itself.

    ONE decision for both checks that run it: the pre-unload check in
    ModelService (against a projection of the cards with the resident model's
    memory given back) and the authoritative check in ModelLoader.load (against
    live memory, after the unload). Two copies of this rule could disagree, and
    the pre-check would then refuse loads the loader accepts or wave through
    loads it refuses.

    A model that fits one card goes on the most-free such card (or the
    requested card, which choose_gpu refuses if it does not fit), so it fits by
    construction. A split is checked against its planned budgets — each card's
    free memory less SHARD_RESERVE_MB, and for bitsandbytes less the 0.9
    transformers applies itself — which are exactly the `max_memory` the load
    passes, so the check and the load cannot disagree about what a card holds.

    EVERY quantization is checked. Q4, Q2 and pre-quantized models used to skip
    the check because bitsandbytes could offload to the CPU; offload is gone
    (operator decision 3), so a skipped check only moved the refusal into the
    load, after the resident model had been unloaded.

    `is_pre_quantized` must be `checkpoint_is_pre_quantized(<the checkpoint>)` at
    BOTH call sites — the reading ModelLoadContext.load acts on — and
    `pre_quantized_max_memory_factor` what that checkpoint's own quantizer
    applies: 1.0 for GPTQ or AWQ, but 0.9 for a checkpoint quantized by
    bitsandbytes or BitNet, which meets the same quantizer class miLLM's own Q4/Q8
    load does (review round 2, 2026-09-14). `plan_transformers_load` reads both.

    Q2 on a checkpoint that is not already quantized is REFUSED (review round 1,
    2026-09-14). bitsandbytes has no 2-bit mode, so ModelLoadContext.load gives
    it no quantization config and it loads in bfloat16: 2 bytes a parameter
    against the 0.25 its estimate (memory_utils.BYTES_PER_PARAM) was sized for.
    Planned at 0.25 it went whole onto a card with an eighth of the room it
    needs and ran out of memory mid-load, after the unload. Planned at bf16 it
    would load — under a row label, a size estimate and a UI badge that all say
    2-bit. Neither is honest, so it stops here, before the unload. A
    pre-quantized checkpoint on a Q2 row (BitNet and the like) loads at its own
    precision and is not refused.

    Raises:
        GpuNotFoundError: the requested card is not visible.
        InsufficientMemoryError: the requested card lacks room, no GPU is
            visible, or no split across the cards holds the model.
        UnsupportedQuantizationError: Q2 on a checkpoint that is not pre-quantized.
    """
    refuse_unsupported_quantization(quantization, is_pre_quantized, estimated_memory_mb)
    max_memory_factor = split_max_memory_factor(
        quantization, is_pre_quantized, pre_quantized_max_memory_factor
    )
    placement = choose_gpu(
        estimated_memory_mb,
        requested=requested,
        gpus=gpus,
        shard=transformers_shard_rule(estimated_memory_mb, max_memory_factor=max_memory_factor),
    )
    if placement.is_shard and placement.required_mb > 0 and placement.budget_mb < estimated_memory_mb:
        raise shard_refusal(
            placement,
            estimated_memory_mb,
            "No single GPU holds it either. A transformers model is never "
            "offloaded to the CPU or disk: free memory on the cards, choose a "
            "smaller quantization, or serve it as GGUF.",
        )
    if placement.is_shard and not placement.gpu_indices:
        raise shard_refusal(
            placement,
            estimated_memory_mb,
            "No GPU has memory to spare for a model of unknown size.",
        )
    return placement


def plan_transformers_load(
    estimated_memory_mb: int,
    quantization: str,
    requested: GpuRequest,
    gpus: list[GpuInfo],
    cache_path: Optional[str],
    is_pre_quantized: bool = False,
    trust_remote_code: bool = False,
) -> Placement:
    """`decide_transformers_placement` for a real checkpoint, with what it needs read from it.

    THE entry point for both checks that plan a transformers load — the
    pre-unload check in ModelService and the authoritative one in
    ModelLoader.load — so they read the checkpoint the same way. Three readings
    go into the decision: whether it ships its own quantization
    (`checkpoint_is_pre_quantized`), the `max_memory` factor that quantization's
    transformers quantizer applies (`pre_quantized_max_memory_factor`), and the
    memory it is sized at (`transformers_estimate_mb`). Round 1 made the two
    call sites share the first; the other two would otherwise have needed that
    care twice more.

    `cache_path` must be the resolved path the load opens
    (ModelService.resolve_cache_path). `is_pre_quantized` lets a caller that
    already knows say so; the checkpoint is read either way.

    Fit is judged PER CARD (Decision 7, 2026-09-14; decide_transformers_fit):
    the checkpoint is sized as it loads and each card must hold its weights, its
    CUDA context and its layers' KV cache at TRANSFORMERS_MIN_CONTEXT. Only a
    checkpoint that cannot be sized that way — its KV cache not derivable from
    its config, or its model not buildable on the meta device — is judged by the
    20% slack (decide_transformers_placement), and that is logged as an error
    naming the architecture (transformers_fit_falls_back_to_slack).
    """
    pre_quantization = checkpoint_quantization_config(cache_path)
    pre_quantized = is_pre_quantized or pre_quantization is not None
    refuse_unsupported_quantization(quantization, pre_quantized, estimated_memory_mb)
    fit = transformers_fit(
        cache_path, quantization, pre_quantized, trust_remote_code=trust_remote_code
    )
    if fit is not None:
        return decide_transformers_fit(
            fit,
            requested=requested,
            gpus=gpus,
            max_memory_factor=split_max_memory_factor(
                quantization, pre_quantized, pre_quantized_max_memory_factor(pre_quantization)
            ),
        )
    return decide_transformers_placement(
        transformers_estimate_mb(
            estimated_memory_mb, cache_path, pre_quantization, trust_remote_code=trust_remote_code
        ),
        quantization,
        requested=requested,
        gpus=gpus,
        is_pre_quantized=pre_quantized,
        pre_quantized_max_memory_factor=pre_quantized_max_memory_factor(pre_quantization),
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

    @property
    def model_name(self) -> Optional[str]:
        """Human-readable name of the loaded model, or None.

        The /health route reads this to report which model is serving. It was
        never defined, so the read raised AttributeError inside the health
        check's try/except and the route reported
        `model_loader: unhealthy — "Component check failed"` with a null
        model_name for EVERY successfully loaded model. The load itself was
        fine; only the reporting was broken, which is the worst shape for a
        health signal — it cried wolf on a healthy runtime.
        """
        current = self.state.current
        return current.model_name if current is not None else None

    def load(
        self,
        model_id: int,
        model_name: str,
        cache_path: str,
        quantization: str,
        estimated_memory_mb: int,
        trust_remote_code: bool = False,
        torch_compile: bool = False,
        # See ModelLoadContext.load: "reduce-overhead" enables CUDA Graphs and
        # broke this generate path in production (2026-07-27).
        torch_compile_mode: str = "default",
        is_pre_quantized: bool = False,
        gguf_file: Optional[str] = None,
        gpu: GpuRequest = None,
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
            gguf_file: Repo-relative filename of a GGUF quantization. When set,
                the model is served by llama.cpp instead of transformers, and it
                CANNOT carry SAE attachment, steering or sensing.
            gpu: None or "auto" (the most-free card that fits), a CUDA index,
                or a GPU UUID. A named card that does not fit is refused.

        Returns:
            LoadedModel instance

        Raises:
            InsufficientMemoryError: If not enough GPU memory
            ModelLoadError: If loading fails
        """
        # A GGUF model takes a different runtime entirely. Branch BEFORE the
        # CUDA and memory checks below: those reason about torch allocations and
        # bitsandbytes quantization levels, neither of which describes a
        # llama.cpp context.
        if gguf_file:
            loaded = load_gguf_model(
                model_id=model_id,
                model_name=model_name,
                cache_path=cache_path,
                gguf_file=gguf_file,
                gpu=gpu,
            )
            self.state.set(loaded)
            return loaded

        # Check if CUDA is available
        try:
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

        # Decide the card, then check memory against THAT decision — the same
        # decision the pre-unload check made, now against live memory. This one
        # is authoritative: free memory can change between the two. Reading
        # GPU 0 alone refused a model needing more than 12 GB while the 3090
        # had 24 GB free.
        placement = plan_transformers_load(
            estimated_memory_mb,
            quantization,
            requested=gpu,
            gpus=list_gpus(),
            # The checkpoint's own answers, the ones ModelLoadContext.load acts
            # on. The pre-unload check reads the same file the same way.
            cache_path=cache_path,
            is_pre_quantized=is_pre_quantized,
            trust_remote_code=trust_remote_code,
        )
        logger.info(
            "model_placement",
            model_id=model_id,
            mode=placement.mode,
            reason=placement.reason,
            devices=placement.device_labels,
            required_mb=placement.required_mb,
            capacity_mb=placement.capacity_mb,
        )
        # A split's real device map, before a weight is read: an FP16 map to
        # disk was otherwise found only after every weight had loaded. Review
        # round 2, 2026-09-14.
        preflight_split(model_name, cache_path, quantization, placement, trust_remote_code)

        # Load with context manager for cleanup on failure
        with ModelLoadContext(model_id, model_name) as ctx:
            loaded = ctx.load(
                cache_path=cache_path,
                quantization=quantization,
                trust_remote_code=trust_remote_code,
                placement=placement,
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
