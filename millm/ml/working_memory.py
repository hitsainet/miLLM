"""What a transformers model needs on each card beyond its weights and its KV cache while it serves.

Hardware acceptance, 2026-09-14, items 4 and 10 (0xcc/reviews/multi_gpu_phase2_acceptance_2026-09-14.md).
The per-card fit (Decision 7) admitted a card when its free memory held the weights, the KV
cache at TRANSFORMERS_MIN_CONTEXT and TRANSFORMERS_CUDA_CONTEXT_MB (500). OLMo-2-13B at FP16
on Auto was admitted with 11 MiB to spare on cuda:0, and a 3,879 + 217-token request ran it
out of memory: "Tried to allocate 76.00 MiB ... 11.22 GiB allocated, 285 MiB reserved but
unallocated". The Q8 split did the same ("Tried to allocate 104.00 MiB"). A request needs,
beside its KV cache, the activations of its prefill and what torch's caching allocator
cannot hand back, and neither was counted.

THE ESTIMATE, per card, at the admitted context S:

    working_mb = T + ceil(ALLOCATOR_OVERHEAD_FRACTION x (T + KV))

  * T — the transient peak: the most memory alive at once, excluding the KV cache and the
    logits a prefill leaves behind, while any phase that runs on this card runs (a decoder
    layer it holds; the embeddings and generate's sampling if it holds the input; the final
    norm and lm_head if it holds the output). It is TRACED, not guessed: the model's own
    forward runs on the meta device at S tokens, as generate's prefill calls it (use_cache,
    logits_to_keep=1), and every storage it creates is followed from creation to release
    (trace_events). What the meta device cannot run is modelled allocation by allocation
    from the kernels that run on the node: SDPA as the FlashAttention-2 or memory-efficient
    kernel torch 2.10 dispatches on an Ampere card, and bitsandbytes 0.49.1's int8 matmul
    (an int32 product and a float16 dequantisation beside the output). What is alive at a
    layer's peak, read off the trace:
      OLMo-2-13B, 26 x hidden + 1,040 bytes a token: the embedding output (transformers
      keeps `inputs_embeds` for the whole forward), the layer's input, q, k and v in
      bfloat16, and four float32 rotary tensors — OLMo-2's RoPE returns float32 cos/sin, so
      q's sum and k's two products and sum are float32 at once (16 x hidden) — with cos/sin
      and the position ids.
      Qwen2.5-7B, 8 x hidden + 6 x intermediate + 528 bytes a token: four hidden-wide
      bfloat16 tensors (the embedding output, the layer input, the residual, the MLP's
      normalised input) and the MLP's act_fn(gate), up and their product at 18,944 wide.
  * The allocator term — what the caching allocator keeps reserved and cannot give to the
    next allocation: blocks rounded up, KV tensors carved out of freed activation segments,
    and DynamicCache reallocating every layer's keys and values at each decoded token.
    Calibrated by replaying traced requests (a prefill of S-16 tokens and 16 decoded, and
    of S-256 and 256) through a model of c10/cuda/CUDACachingAllocator.cpp's default policy
    (tests/support/caching_allocator.py) and finding the smallest room that serves each:
    across 90 cases (OLMo-2-13B, Qwen2.5-7B, Qwen2.5-14B, Vicuna-13B, phi-4; 2k-8k tokens;
    first, middle and last cards of splits) the excess over T never passed 0.338 x (T + KV).
    0.40 keeps a margin for architectures outside that sample.

VALIDATED against the node. The trace and the allocator model, fed OLMo-2-13B's cuda:0 (15
layers, 10,074 MiB landed), reproduce the 17:14:11Z out-of-memory error: a 75.76 MiB float32
allocation (torch: 76.00 MiB) in the rotary embedding of the 14th layer, with 11.21 GiB
allocated (torch: 11.22) and 310 MiB reserved but unallocated (torch: 285). The three
requests that succeeded afterwards, in order on one cache, peak at 10,969 / 11,613 / 12,213
MiB with the CUDA context (the poller: 10,924 / 11,626 / 12,106): +0.4%, -0.1%, +0.9%. The Q8
request's replay runs out of memory allocating 102.28 MiB (torch: 104.00).

THE FIT'S ALLOWANCE AGAINST THE NODE (card_mb, one figure per card, no replay). OLMo-2-13B's
cuda:0, 15 layers, holding the input. What the node measured beyond weights and the request's
KV cache includes the CUDA context, so the context the node measured after a generation (330
MiB) is added to the fit's working memory for the comparison:
    request tokens      1,073    2,077    3,083    4,096
    fit + 330 MiB         650      946    1,245    1,545
    node                  536      943    1,129   ~1,300 (extrapolated in the report)
                         +21%     +0.3%    +10%     +19%
At the 390-400 MiB the context had grown to by then, each is 60-70 MiB higher; with the
setting's 500 MiB, +18% to +53%. The fit is never below the node, and it is closest at 2k
tokens, where the node's figure includes cache the 1k request left reserved.
Qwen2.5-7B's cuda:0 (20 layers, the steered 3,893 + 88-token request): the fit's working memory
is 823 MiB, 1,153 with a 330 MiB context, against the report's ~525 MiB (+120%). That figure is
below the trace's own transient peak at that length (543 MiB) plus a context, and it was taken
with a 784 MB SAE attached on that card; the residual is unexplained. It errs toward refusing,
not toward running out; re-measure it with no SAE attached.

BITSANDBYTES LOAD STAGING. transformers 5.15.1 quantizes on the fly synchronously: each weight
is materialised on its card in bfloat16, Int8Params makes a float16 copy there and quantizes
it, and both copies are freed. The holes they leave are not given back. Replayed through the
allocator model, OLMo-2-13B at Q8 strands 670 / 507 MiB on its two cards (the node: landed
653 / 444 MiB above the map), of which empty_cache could return 136 MiB. It is 4.2-6.0% of
the bfloat16 size of the card's quantized weights; BNB_LOAD_OVERHEAD_FRACTION counts 7%. Not
measured for 4-bit, whose staging copy is four times what it stores.

A model the trace cannot run (a data-dependent op on the meta device: mixture-of-experts
routing, custom remote code) is sized by estimate_working_memory, a bound above every traced
model, and that is logged as an error naming the architecture.
"""

from __future__ import annotations

import inspect
import math
import time
import types
import weakref
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

from millm.core.logging import get_logger

logger = get_logger(__name__)

_MIB = 1024 * 1024

#: The caching allocator's reserve beyond the transient peak, as a fraction of (the card's
#: transient peak + its KV cache at the admitted context). Largest measured: 0.338.
ALLOCATOR_OVERHEAD_FRACTION = 0.40

#: What a bitsandbytes on-the-fly quantization leaves stranded on a card, as a fraction of
#: the bfloat16 size of the weights it quantized there. Measured 4.2-6.0% (OLMo-2-13B Q8).
BNB_LOAD_OVERHEAD_FRACTION = 0.07

#: The share of an int8 layer's input columns assumed to hold LLM.int8() outliers
#: (|x| >= 6.0). They are split out and multiplied in the input's dtype.
INT8_OUTLIER_COLUMN_FRACTION = 0.05

#: generate's float32 copies of one vocabulary row on the input card: the logits copy,
#: the logits processors' copies and the softmax.
SAMPLING_VOCAB_ROWS = 4

INPUT = "input"
OUTPUT = "output"
END = "end"


def _nbytes(shape: Iterable[int], dtype: torch.dtype) -> int:
    # dtype.itemsize, not a probe tensor: a tensor made here would be traced as an allocation.
    return math.prod(int(size) for size in shape) * dtype.itemsize


@dataclass(frozen=True)
class WorkingMemory:
    """A model's transient peaks at `tokens` tokens: per decoder layer, before them and after them."""

    tokens: int
    layer_bytes: tuple[int, ...]
    input_bytes: int
    output_bytes: int
    sampling_bytes: int
    method: str
    reason: str = ""

    def transient_mb(self, layers: Iterable[int], *, holds_input: bool, holds_output: bool) -> int:
        """The transient peak of a card holding these decoder layers (and the input or output)."""
        peaks = [self.layer_bytes[index] for index in layers]
        if holds_input:
            peaks.append(self.input_bytes + self.sampling_bytes)
        if holds_output:
            peaks.append(self.output_bytes)
        return math.ceil(max(peaks, default=0) / _MIB)

    def card_mb(
        self, layers: Iterable[int], *, holds_input: bool, holds_output: bool, kv_mb: int
    ) -> int:
        """The working memory a card needs beside its weights, its KV cache and its CUDA context."""
        transient = self.transient_mb(layers, holds_input=holds_input, holds_output=holds_output)
        return transient + math.ceil(ALLOCATOR_OVERHEAD_FRACTION * (transient + kv_mb))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "method": self.method,
            "reason": self.reason,
            "layer_peak_mb": math.ceil(max(self.layer_bytes, default=0) / _MIB),
            "input_peak_mb": math.ceil((self.input_bytes + self.sampling_bytes) / _MIB),
            "output_peak_mb": math.ceil(self.output_bytes / _MIB),
            "allocator_overhead_fraction": ALLOCATOR_OVERHEAD_FRACTION,
        }


class _StorageTrace:
    """Every storage a forward pass creates, from creation to release, with the phase it happened in."""

    def __init__(self) -> None:
        self.events: list[tuple[bool, int, int, Any]] = []  # (allocated?, id, bytes, phase)
        self._size: dict[int, int] = {}
        self._count: dict[int, int] = {}
        self._by_storage: dict[int, int] = {}
        self._next = 0
        self.phase: Any = INPUT

    def alloc(self, nbytes: int) -> int:
        sid = self._next
        self._next += 1
        self._size[sid] = nbytes
        self.events.append((True, sid, nbytes, self.phase))
        return sid

    def free(self, sid: int) -> None:
        self.events.append((False, sid, self._size[sid], self.phase))

    def live_ids(self) -> set[int]:
        return set(self._by_storage.values())

    def _released(self, storage: int) -> None:
        sid = self._by_storage.get(storage)
        if sid is None:
            return
        self._count[sid] -= 1
        if self._count[sid] == 0:
            del self._by_storage[storage]
            self.free(sid)

    def track(self, value: Any, inputs: set[int]) -> None:
        for tensor in _tensors(value):
            try:
                storage = tensor.untyped_storage()
            except Exception:  # noqa: BLE001 - a tensor with no storage holds no memory
                continue
            key = storage._cdata
            if key in self._by_storage:
                sid = self._by_storage[key]
            elif key in inputs:
                continue  # a view of something the trace did not create (a parameter)
            else:
                sid = self.alloc(storage.nbytes())
                self._by_storage[key] = sid
                self._count[sid] = 0
            self._count[sid] += 1
            weakref.finalize(tensor, self._released, key)

    def new(self, shape: Iterable[int], dtype: torch.dtype, device: Any) -> torch.Tensor:
        tensor = torch.empty(tuple(shape), dtype=dtype, device=device)
        self.track(tensor, set())
        return tensor


def _tensors(value: Any) -> list[torch.Tensor]:
    from torch.utils._pytree import tree_leaves

    return [leaf for leaf in tree_leaves(value) if isinstance(leaf, torch.Tensor)]


def _storages(value: Any) -> set[int]:
    found = set()
    for tensor in _tensors(value):
        try:
            found.add(tensor.untyped_storage()._cdata)
        except Exception:  # noqa: BLE001
            pass
    return found


# ---------------------------------------------------------------------------------------
# bitsandbytes 0.49.1 int8 kernels, as they allocate on CUDA (backends/cuda/ops.py and the
# default kernels in backends/default/ops.py). The meta device has no kernel for them.
# ---------------------------------------------------------------------------------------


def _int8_vectorwise_quant(trace: _StorageTrace, A: torch.Tensor, threshold: float = 0.0) -> Any:
    rows, cols = math.prod(A.shape[:-1]), A.shape[-1]
    row_stats = trace.new((rows,), torch.float32, A.device)
    out_row = trace.new(A.shape, torch.int8, A.device)
    outlier_cols = None
    if threshold > 0.0:
        magnitudes = trace.alloc(_nbytes(A.shape, A.dtype))  # A.abs()
        outliers = trace.alloc(_nbytes(A.shape, torch.bool))  # >= threshold
        trace.free(magnitudes)
        outlier_cols = trace.new(
            (max(1, math.ceil(cols * INT8_OUTLIER_COLUMN_FRACTION)),), torch.int64, A.device
        )
        trace.free(outliers)
    return out_row, row_stats, outlier_cols


def _int8_scaled_mm(
    trace: _StorageTrace, A: torch.Tensor, B: torch.Tensor, row_stats: Any, col_stats: Any,
    bias: Any = None, dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    shape = (*A.shape[:-1], B.shape[0])
    product = trace.alloc(_nbytes(shape, torch.int32))  # int8_linear_matmul
    half = trace.alloc(_nbytes(shape, torch.float16))  # int8_mm_dequant's output
    dtype = dtype or torch.float16
    if dtype != torch.float16:
        output = trace.new(shape, dtype, A.device)  # .to(dtype)
        trace.free(half)
    else:
        trace.free(half)
        output = trace.new(shape, dtype, A.device)
    trace.free(product)
    return output


def _int8_mixed_scaled_mm(
    trace: _StorageTrace, A: torch.Tensor, CA: torch.Tensor, CB: torch.Tensor, SCA: Any, SCB: Any,
    outlier_cols: Optional[torch.Tensor] = None, bias: Any = None,
) -> Any:
    rows, out_features = CA.shape[0], CB.shape[0]
    k = 0 if outlier_cols is None else int(outlier_cols.numel())
    sub_b = None
    if k:
        sub_a = trace.new((rows, k), A.dtype, A.device)
        columns = trace.alloc(_nbytes((out_features, k), torch.int8))
        scaled = trace.alloc(_nbytes((out_features, k), torch.float32))
        dequantised = trace.alloc(_nbytes((out_features, k), torch.float32))
        trace.free(scaled)
        trace.free(columns)
        sub_b = trace.alloc(_nbytes((out_features, k), A.dtype))
        trace.free(dequantised)
    else:
        sub_a = trace.new((0,), A.dtype, A.device)
    output = _int8_scaled_mm(trace, CA, CB, SCA, SCB, bias, A.dtype)
    if sub_b is not None:
        added = trace.new(output.shape, A.dtype, A.device)  # output.addmm(subA, subB)
        del output
        trace.free(sub_b)
        output = added
    return output, sub_a


def _int8_linear_matmul(trace: _StorageTrace, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return trace.new((*A.shape[:-1], B.shape[0]), torch.int32, A.device)


def _int8_mm_dequant(
    trace: _StorageTrace, A: torch.Tensor, row_stats: Any, col_stats: Any,
    dtype: Optional[torch.dtype] = None, bias: Any = None,
) -> torch.Tensor:
    dtype = dtype or torch.float16
    half = trace.alloc(_nbytes(A.shape, torch.float16))
    output = trace.new(A.shape, dtype, A.device)
    trace.free(half)
    return output


_KERNELS = {
    "bitsandbytes::int8_vectorwise_quant": _int8_vectorwise_quant,
    "bitsandbytes::int8_mixed_scaled_mm": _int8_mixed_scaled_mm,
    "bitsandbytes::int8_scaled_mm": _int8_scaled_mm,
    "bitsandbytes::int8_linear_matmul": _int8_linear_matmul,
    "bitsandbytes::int8_mm_dequant": _int8_mm_dequant,
}


def _nf4_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    """Linear4bit's prefill: the whole weight dequantised in the compute dtype, then one matmul."""
    dtype = getattr(self, "compute_dtype", None) or x.dtype
    weight = torch.empty((self.out_features, self.in_features), dtype=dtype, device=x.device)
    output = torch.nn.functional.linear(x.to(dtype), weight)
    del weight
    return output.to(x.dtype)


def _sdpa_kernel(trace: _StorageTrace, args: tuple, kwargs: dict) -> torch.Tensor:
    """torch.nn.functional.scaled_dot_product_attention as torch 2.10 runs it on an Ampere card.

    FlashAttention-2 for a half-precision call with no mask and head sizes up to 256: the
    output and a float32 logsumexp. Otherwise the memory-efficient kernel: a boolean mask is
    converted to an additive bias in the query's dtype and, with its last dimension padded
    to a multiple of 16, materialised at every head, beside a float32 logsumexp.
    """
    query, key, value = args[:3]
    mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
    batch, heads, q_len, head_dim = query.shape
    kv_len, value_dim = key.shape[-2], value.shape[-1]
    temporaries = []
    half = query.dtype in (torch.float16, torch.bfloat16)
    if mask is None and half and max(head_dim, value_dim) <= 256:
        temporaries.append(trace.alloc(_nbytes((batch, heads, q_len), torch.float32)))
        # The output contiguous in (batch, heads, length, dim), as cuDNN's attention kernel
        # allocates it, so transformers' transpose(1, 2).contiguous() copies it once more.
        # Which fused kernel torch picks on the node was not observed, and the two layouts
        # differ in exactly that copy; the node's own error decides it. With this layout the
        # replay of OLMo-2-13B's cuda:0 fails allocating 75.76 MiB with 11.21 GiB allocated
        # (torch: "Tried to allocate 76.00 MiB", 11.22 GiB); with a (batch, length, heads, dim)
        # buffer it fails on a 102.28 MiB MLP tensor torch never reported. It is also the
        # larger of the two.
        output = torch.empty((batch, heads, q_len, value_dim), dtype=query.dtype, device=query.device)
        for sid in temporaries:
            trace.free(sid)
        return output
    else:
        if mask is not None:
            if mask.dtype == torch.bool:
                temporaries.append(trace.alloc(_nbytes(mask.shape, query.dtype)))
            temporaries.append(
                trace.alloc(_nbytes((batch, heads, q_len, 16 * math.ceil(kv_len / 16)), query.dtype))
            )
        temporaries.append(
            trace.alloc(_nbytes((batch, heads, 32 * math.ceil(q_len / 32)), torch.float32))
        )
    output = torch.empty(
        (batch, q_len, heads, value_dim), dtype=query.dtype, device=query.device
    ).transpose(1, 2)
    for sid in temporaries:
        trace.free(sid)
    return output


def trace_events(
    model: Any, layer_names: list[str], tokens: int, decode_steps: int = 0
) -> tuple[list[tuple[bool, int, int, Any]], set[int]]:
    """Every allocation and release of a request on the meta-device model, and what the prefill left alive.

    A prefill of `tokens` tokens, as generate calls it, then `decode_steps` one-token steps on
    its cache. Each event carries the phase it happened in: INPUT, a decoder layer's index,
    OUTPUT, or END once the request's cache is dropped. The second value holds the storages
    alive right after the prefill (its KV cache and logits).

    Thread-safe with real inference running beside it: the dispatch and function modes are
    thread-local, and every change to the model — hooks, int8 state, 4-bit forwards — is on
    this meta model's own modules and undone before returning.
    """
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    trace = _StorageTrace()

    class _Dispatch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):  # noqa: N805
            kwargs = kwargs or {}
            inputs = _storages((args, kwargs))
            schema = getattr(func, "_schema", None)
            kernel = _KERNELS.get(getattr(schema, "name", ""))
            result = kernel(trace, *args, **kwargs) if kernel else func(*args, **kwargs)
            trace.track(result, inputs)
            return result

    class _Functions(TorchFunctionMode):
        def __torch_function__(self, func, types_, args=(), kwargs=None):  # noqa: N805
            kwargs = kwargs or {}
            if getattr(func, "__name__", "") == "scaled_dot_product_attention":
                return _sdpa_kernel(trace, args, kwargs)
            return func(*args, **kwargs)

    handles = []
    int8_states: list[tuple[Any, Any, Any]] = []
    nf4_modules: list[Any] = []
    try:
        for index, name in enumerate(layer_names):
            layer = model.get_submodule(name)
            handles.append(layer.register_forward_pre_hook(
                lambda module, args, index=index: setattr(trace, "phase", index)
            ))
            handles.append(layer.register_forward_hook(
                lambda module, args, output: setattr(trace, "phase", OUTPUT)
            ))
        for module in model.modules():
            kind = type(module).__name__
            if kind == "Linear8bitLt":
                state = module.state
                int8_states.append((state, state.CB, state.SCB))
                # As loaded: the int8 weight and its row scales are already on the card.
                state.CB = module.weight.data
                state.SCB = torch.empty((module.weight.shape[0],), dtype=torch.float32, device="meta")
            elif kind == "Linear4bit":
                module.forward = types.MethodType(_nf4_forward, module)
                nf4_modules.append(module)
        keep_logits = "logits_to_keep" in inspect.signature(model.forward).parameters

        def step(input_ids: torch.Tensor, cache: Any) -> Any:
            kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": True}
            if cache is not None:
                kwargs["past_key_values"] = cache
            if keep_logits:
                kwargs["logits_to_keep"] = 1
            return model(**kwargs)

        with torch.no_grad(), _Functions(), _Dispatch():
            outputs = step(torch.zeros((1, tokens), dtype=torch.long, device="meta"), None)
            kept = trace.live_ids()
            cache = getattr(outputs, "past_key_values", None)
            del outputs
            for _ in range(decode_steps):
                trace.phase = INPUT
                outputs = step(torch.zeros((1, 1), dtype=torch.long, device="meta"), cache)
                cache = getattr(outputs, "past_key_values", None)
                del outputs
            trace.phase = END
            del cache
    finally:
        for handle in handles:
            handle.remove()
        for state, cb, scb in int8_states:
            state.CB, state.SCB = cb, scb
        for module in nf4_modules:
            del module.forward
    return trace.events, kept


def trace_working_memory(model: Any, layer_names: list[str], tokens: int, vocab_size: int) -> WorkingMemory:
    """The transient peaks of a prefill of `tokens` tokens on the meta-device model (trace_events)."""
    events, kept = trace_events(model, layer_names, tokens)
    live = 0
    peaks: dict[Any, int] = {}
    for allocated, sid, nbytes, phase in events:
        if sid in kept:
            continue
        live += nbytes if allocated else -nbytes
        peaks[phase] = max(peaks.get(phase, 0), live)
    return WorkingMemory(
        tokens=tokens,
        layer_bytes=tuple(peaks.get(index, 0) for index in range(len(layer_names))),
        input_bytes=peaks.get(INPUT, 0),
        output_bytes=peaks.get(OUTPUT, 0),
        sampling_bytes=int(vocab_size) * 4 * SAMPLING_VOCAB_ROWS,
        method="traced",
    )


def _text_config(config: Any) -> Any:
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            return get_text_config(decoder=True)
        except Exception:  # noqa: BLE001
            pass
    return config


def estimate_working_memory(
    config: Any, num_layers: int, tokens: int, quantization: str, reason: str
) -> WorkingMemory:
    """A bound for a model the trace cannot run: above every traced model, per token.

    32 x hidden + 8 x intermediate bytes a token a layer (traced: OLMo-2-13B 134,160 against
    this bound's 274,432; Qwen2.5-7B 142,864 against 266,240), plus 10 x intermediate for
    int8 (traced OLMo-2-13B Q8: about 177,000 against 412,672) and the whole dequantised MLP
    weight for 4-bit. A mixture of experts counts the routed experts' width.

    Raises:
        ValueError: the config has no hidden size.
    """
    text = _text_config(config)
    hidden = getattr(text, "hidden_size", None)
    if not isinstance(hidden, int) or hidden <= 0:
        raise ValueError("its config has no hidden_size")
    intermediate = max(
        int(getattr(text, "intermediate_size", 0) or 0),
        int(getattr(text, "moe_intermediate_size", 0) or 0)
        * int(getattr(text, "num_experts_per_tok", 1) or 1),
    ) or 4 * hidden
    vocab = int(getattr(text, "vocab_size", 0) or 0)
    per_token = 32 * hidden + 8 * intermediate
    fixed = 0
    if quantization.upper() == "Q8":
        per_token += 10 * intermediate
    elif quantization.upper() == "Q4":
        fixed = 2 * intermediate * hidden
    return WorkingMemory(
        tokens=tokens,
        layer_bytes=(per_token * tokens + fixed,) * num_layers,
        input_bytes=4 * hidden * tokens,
        output_bytes=12 * hidden * tokens + 2 * vocab * tokens,
        sampling_bytes=vocab * 4 * SAMPLING_VOCAB_ROWS,
        method="estimated",
        reason=reason,
    )


def size_working_memory(
    model: Any,
    config: Any,
    layer_names: Optional[list[str]],
    tokens: int,
    quantization: str,
    architecture: str,
) -> WorkingMemory:
    """The working memory of a load: traced when the model runs on the meta device, estimated (loudly) when not."""
    vocab = int(getattr(_text_config(config), "vocab_size", 0) or 0)
    started = time.perf_counter()
    if layer_names is None:
        reason, error = "its decoder layers could not be found", None
    else:
        try:
            working = trace_working_memory(model, layer_names, tokens, vocab)
        except Exception as e:  # noqa: BLE001 - estimated instead, and said so below
            reason, error = "its forward pass could not be traced on the meta device", e
        else:
            logger.info(
                "transformers_fit_working_memory_traced",
                architecture=architecture,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                **working.to_dict(),
            )
            return working
    working = estimate_working_memory(config, len(layer_names or ()) or 1, tokens, quantization, reason)
    fields: dict[str, Any] = {"architecture": architecture, **working.to_dict()}
    if error is not None:
        fields.update(error_type=type(error).__name__, error=str(error)[:300])
    logger.error("transformers_fit_working_memory_estimated", **fields)
    return working


def bnb_quantized_bytes_by_module(model: Any, hf_quantizer: Any) -> dict[str, int]:
    """The bfloat16 bytes of each weight bitsandbytes quantizes during the load, by module name.

    Empty unless the load quantizes on the fly (a pre-quantized checkpoint is not staged).
    """
    if hf_quantizer is None or getattr(hf_quantizer, "pre_quantized", True):
        return {}
    found: dict[str, int] = {}
    for name, module in model.named_modules():
        if type(module).__name__ in ("Linear8bitLt", "Linear4bit"):
            found[name] = int(module.in_features) * int(module.out_features) * 2
    return found


def bnb_staging_mb(quantized_bytes: Iterable[int]) -> int:
    """What a card keeps stranded by the load's staging copies of these weights."""
    return math.ceil(BNB_LOAD_OVERHEAD_FRACTION * sum(quantized_bytes) / _MIB)
