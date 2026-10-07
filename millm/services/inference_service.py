"""
Inference service for OpenAI-compatible generation.

Provides the core generation logic for chat completions, text completions,
and embeddings. Handles streaming via TextIteratorStreamer.

Implementation notes:
1. Thread-based streaming (Transformers generate() is blocking)
2. TextIteratorStreamer bridges generate() to async iteration
3. Request queue prevents GPU memory conflicts
4. Steering integration is transparent to API layer
"""

import asyncio
import dataclasses
import contextlib
import contextvars
import gc
import math
from dataclasses import dataclass
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from threading import Event, Thread
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Iterator,
    Optional,
    TypeVar,
)

import torch

from millm.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatLogprobAlternative,
    ChatLogprobs,
    ChatLogprobToken,
    ChatMessage,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    CompletionLogprobs,
    TextCompletionChoice,
    TextCompletionRequest,
    TextCompletionResponse,
    Usage,
)
from millm.core.errors import (
    ContextLengthExceededError,
    EmbeddingInputTooLongError,
    EmbeddingVectorInvalidError,
    EngineUnsupportedError,
    FieldNotHonouredError,
    InvalidScoringRequestError,
    NoChatTemplateError,
    ResponseFormatUnsupportedError,
    ScoringNumericalError,
    GenerationOutOfMemoryError,
    MiLLMError,
    ModelBusyError,
)
from millm.core.logging import get_logger
from millm.ml.constrained_decoding import (
    CompiledConstraint,
    GrammarCache,
    JsonConstraintProcessor,
    constrained_header,
    schema_name,
    stop_token_ids,
    validate_output,
)
from millm.ml.embedding_pooling import (
    EmbeddingOptions,
    NonFiniteEmbeddingError,
    finalize_vector,
    pool_hidden,
)
from millm.ml.generation_config import GenerationConfig
from millm.ml.model_loader import LoadedModelState
from millm.services.request_queue import RequestQueue

if TYPE_CHECKING:
    from millm.services.model_service import ModelService
    from millm.services.monitoring_service import MonitoringService
from millm.services.async_iter import aiter_blocking
from millm.services.reasoning_split import (
    OPEN as THINK_OPEN,
    StreamingReasoningSplitter,
    split_reasoning,
)


logger = get_logger(__name__)

#: The result type of `InferenceService.run_model_work`'s callable.
_T = TypeVar("_T")



def _draft_torch_dtype(draft_model_id: str) -> "torch.dtype":
    """The draft model's OWN precision, by the shared rule (`ml/native_dtype.py`) over its config.

    Not the target's: forcing a bfloat16-native draft to float16 because the target is float16
    reintroduces the overflow risk the rule exists to avoid, and an FP32 target would double the
    draft's memory with no sizing for it. The draft only proposes tokens; the target verifies them.
    """
    from millm.ml.native_dtype import resolve_for_config

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(draft_model_id)
    except Exception as exc:  # noqa: BLE001 - nothing readable: the rule's default (bfloat16)
        # SAID, not silent: a custom-architecture or uncached draft then loads at bfloat16 even
        # if its checkpoint is float16.
        logger.warning("draft_dtype_config_unreadable", draft_model_id=draft_model_id, error=str(exc))
        config = None
    return resolve_for_config("FP16", config).torch_dtype

def _return_cached_draft_memory() -> None:
    """Give a discarded draft's memory back to the card. Never raises.

    Dropping the last reference returns a draft's tensors to torch's caching
    allocator, not to the card: nvidia-smi — what every placement, the pre-unload
    check and the other tenants of the node read — still counts them as used
    until the cache is emptied. A draft discarded because the model changed
    finishes AFTER the unload that emptied the cache, so nothing else would.
    Review round 4, 2026-09-14.
    """
    gc.collect()
    try:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 - a cleanup must not turn off speculation
        logger.warning("draft_memory_release_failed", error=str(e))

#: torch names the card in its message: "... GPU 1 has a total capacity of ...".
_OOM_GPU = re.compile(r"\bGPU (\d+)\b")


def _generation_oom_error(exc: BaseException, generation_kwargs: dict) -> GenerationOutOfMemoryError:
    """The typed refusal for a CUDA out-of-memory error raised by generate(): which card, and what to change.

    Holds only strings and numbers, never the exception, so raising it does not
    keep the failed pass's traceback (and the tensors its frames hold) alive.
    """
    text = str(exc)
    match = _OOM_GPU.search(text)
    device = f"cuda:{match.group(1)}" if match else None
    name = None
    if match is not None:
        try:
            name = torch.cuda.get_device_name(int(match.group(1)))
        except Exception:  # noqa: BLE001 - the index alone still names the card
            name = None
    shape = getattr(generation_kwargs.get("input_ids"), "shape", None)
    prompt_tokens = int(shape[-1]) if shape is not None and len(shape) >= 1 else None
    rows = int(shape[0]) if shape is not None and len(shape) >= 2 else None
    max_new_tokens = generation_kwargs.get("max_new_tokens")
    where = f"{device} ({name})" if device and name else (device or "a GPU")
    size = f"{prompt_tokens} prompt tokens" if prompt_tokens is not None else "its prompt"
    if rows and rows > 1:
        size += f" in each of {rows} rows"
    return GenerationOutOfMemoryError(
        f"Generation ran out of memory on {where}: this request ({size}, up to "
        f"{max_new_tokens} new tokens) needed more room for its KV cache than the card "
        "had left beside the model. Its memory has been released and the server keeps "
        "serving. Send a shorter prompt or conversation, or ask for fewer max_tokens.",
        details={
            "device": device,
            "device_name": name,
            "prompt_tokens": prompt_tokens,
            "batch_rows": rows,
            "max_new_tokens": max_new_tokens,
            "torch_message": text[:500],
        },
    )


def _scoring_oom_error(exc: BaseException, inputs: Any) -> GenerationOutOfMemoryError:
    """The out-of-memory refusal for a scoring pass, worded for scoring (review round 2, L1).

    Generation's message blames the KV cache and suggests fewer max_tokens; a scoring pass has no
    cache (`use_cache=False`) and must send max_tokens=1, so that advice would be wrong.
    """
    base = _generation_oom_error(exc, {**dict(inputs), "max_new_tokens": 1})
    tokens = base.details.get("prompt_tokens")
    size = f"{tokens} prompt tokens" if tokens is not None else "its prompt"
    device, name = base.details.get("device"), base.details.get("device_name")
    where = f"{device} ({name})" if device and name else (device or "a GPU")
    return GenerationOutOfMemoryError(
        f"Scoring ran out of memory on {where}: one forward pass over {size} needed more room than "
        "the card had left beside the model. Its memory has been released and the server keeps "
        "serving. Send a shorter prompt.",
        details={**base.details, "max_new_tokens": None, "scoring": True},
    )


def _release_generation_memory() -> None:
    """Give a failed generation's memory back: collect what its frames held, then empty torch's cache. Never raises.

    Called only after the except block that caught the out-of-memory error has
    exited: inside it, the traceback still references the frames holding the
    partial KV cache and activations, so emptying the cache there frees nothing.
    """
    gc.collect()
    try:
        torch.cuda.empty_cache()  # a no-op when CUDA was never initialised
    except Exception as e:  # noqa: BLE001 - a cleanup must not replace the refusal
        logger.warning("generation_oom_release_failed", error=str(e))


def _release_cached_gpu_memory(indices: list[int]) -> dict[str, int]:
    """Return torch's unused cached blocks to the cards, reporting what each of these got back (MiB). Never raises.

    torch.cuda.empty_cache acts on every card's allocator and creates no CUDA
    context; the before/after reads touch only the model's own cards.
    """
    try:
        before = {index: int(torch.cuda.memory_reserved(index)) for index in indices}
        torch.cuda.empty_cache()
        return {
            f"cuda:{index}": max(before[index] - int(torch.cuda.memory_reserved(index)), 0)
            // (1024 * 1024)
            for index in indices
        }
    except Exception as e:  # noqa: BLE001 - a release is housekeeping, never a failure
        logger.warning("idle_cache_release_failed", error=str(e))
        return {}


def _stream_error_event(exc: MiLLMError) -> str:
    """The SSE error event for a refusal raised after a stream's 200 was committed.

    The same envelope /v1 answers with when it can still choose the status: the
    error's own OpenAI type when it names one, otherwise invalid_request_error for
    a request to change (4xx) and server_error for the rest.
    """
    import json

    from millm.api.routes.openai.errors import ERROR_STATUS_MAP
    from millm.core.backpressure import retry_after_for

    error_type = getattr(exc, "openai_error_type", None) or (
        "invalid_request_error" if exc.status_code < 500 else "server_error"
    )
    error: dict[str, Any] = {"message": exc.message, "type": error_type, "code": exc.code.lower()}
    # Feature 29 (FR-29.6.5): the 200 is committed, so no Retry-After HEADER is possible. A
    # refusal that /v1 answers with 503 carries the same number in the event instead.
    if ERROR_STATUS_MAP.get(exc.code, (exc.status_code, ""))[0] == 503:
        error["retry_after"] = retry_after_for(exc.code, exc.details)
    body = {"error": error}
    return f"data: {json.dumps(body)}\n\n"


def _served_max_context(config: Any) -> Optional[int]:
    """The longest prompt + generation a transformers model is asked to serve; None when unknown.

    Read from the TEXT config. A multimodal checkpoint (Gemma3ForConditionalGeneration
    and its kind) keeps `max_position_embeddings` in its text config only, so the
    top-level read found nothing and every request was accepted at any length,
    to run out of memory or past the model's positions. The per-card fit sizes
    the KV cache against the same field (model_loader.admitted_context). Review
    round 5, 2026-09-14.
    """
    if config is None:
        return None
    candidates = []
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            candidates.append(get_text_config(decoder=True))
        except Exception:  # noqa: BLE001 - fall back to the config itself
            pass
    candidates.append(config)
    for candidate in candidates:
        value = getattr(candidate, "max_position_embeddings", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _attention_mask_of(encoded: Any) -> torch.Tensor:
    """The attention mask of one tokenizer encoding; all ones when the tokenizer returns none
    (one unpadded sequence has no padding to mask)."""
    input_ids = encoded["input_ids"]
    try:
        mask = encoded["attention_mask"]
    except KeyError:
        return torch.ones_like(input_ids)
    return mask if isinstance(mask, torch.Tensor) else torch.ones_like(input_ids)


def _encode_embedding(vector: list[float], encoding_format: str) -> list[float] | str:
    """One embedding in the requested encoding: the floats, or little-endian float32 base64."""
    if encoding_format != "base64":
        return vector
    import base64
    import struct

    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


#: Per-request memo for "which circuit is actually steering". A ContextVar
#: because the InferenceService is a process singleton (see _steering_circuit).
#: Reset at the top of each chat request by reset_steering_memo().
_MEMO_UNSET: Any = object()
_STEERING_CIRCUIT_MEMO: "contextvars.ContextVar[Any]" = contextvars.ContextVar(
    "millm_steering_circuit_memo", default=_MEMO_UNSET
)


#: Set when a per-request circuit dial FAILED to apply, so the rung echo can be
#: retracted. F18 R3-01: the header is computed at request entry and the apply
#: happens later inside generation, so an apply failure left the response
#: advertising `X-miLLM-Circuit-Rung: 2; language="causally validated (edge)"`
#: for an intervention that provably did not run. `_steering_circuit`'s own
#: docstring names that hazard — the R1 fix closed it for the LOOKUP path and
#: left the apply-failure path open. Same ContextVar discipline as the memo:
#: the service is a process singleton, so per-request state cannot live on it.
_CIRCUIT_APPLY_FAILED: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "millm_circuit_apply_failed", default=False
)


#: This request's probe verdicts, published by `_probe_finish` and read by the chat route when it
#: writes `X-miLLM-Probe-Verdicts` (non-streaming) or the terminal chunk (streaming).
#:
#: ⚠ Same ContextVar discipline as the memos above, for the same reason: the InferenceService is a
#: process singleton, so per-request state cannot live on it. And the same explicit reset — an ASGI
#: server may reuse a context across requests, so a stale list would attach one request's verdicts
#: to another's response. That is not a cosmetic leak: it would report a concept as detected in a
#: conversation that never contained it.
_PROBE_VERDICTS: "contextvars.ContextVar[list]" = contextvars.ContextVar(
    "millm_probe_verdicts", default=[]
)


def _verdict_payload(verdict: Any) -> dict:
    """One verdict, as it travels on the wire.

    `not_scored` entries carry their reason and nothing else numeric — reporting a score of 0 for
    a request nobody scored would be a measurement that was never taken.
    """
    # ⚠ `probe_id` and `window` are both here now. This emitted neither, so three verdicts from
    # one probe reached a streaming consumer as three entries distinguishable only by their
    # numbers — and `provisional` is the difference between a calibrated alert and one fired
    # against a bar that was never cut for that window.
    if not verdict.scored:
        return {
            "name": verdict.name,
            "probe_id": verdict.probe_id,
            "window": verdict.window,
            "provisional": bool(verdict.provisional),
            "scored": False,
            "reason": verdict.not_scored_reason,
            "rung": verdict.rung,
            "rung_language": verdict.rung_language,
        }
    return {
        "name": verdict.name,
        "probe_id": verdict.probe_id,
        "window": verdict.window,
        "provisional": bool(verdict.provisional),
        "scored": True,
        "score": verdict.score,
        "threshold": verdict.threshold,
        "verdict": verdict.fires,
        "rung": verdict.rung,
        "rung_language": verdict.rung_language,
    }


def _request_n(request: Any) -> int:
    """`n` as an int; anything that is not an int (absent, a stand-in) reads as 1."""
    n = getattr(request, "n", 1)
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else 1


def _request_seed(request: Any) -> Optional[int]:
    """The request's seed, or None when none was sent (T-60: miLLM never chooses one)."""
    seed = getattr(request, "seed", None)
    return seed if isinstance(seed, int) and not isinstance(seed, bool) else None


def _request_extra_messages(request: Any) -> list[Any]:
    extra = getattr(request, "extra_messages", None)
    return extra if isinstance(extra, list) else []


def _constraint_type(response_format: Any) -> Optional[str]:
    """`json_object` / `json_schema` when a request asks for a constraint; None for none or text."""
    if response_format is None:
        return None
    kind = (
        response_format.get("type") if isinstance(response_format, dict)
        else getattr(response_format, "type", None)
    )
    return kind if kind in ("json_object", "json_schema") else None


from millm.services.batch.state import BATCH_ROW  # noqa: E402  (light: enums + a ContextVar)

#: The asyncio TASK holding the request-queue slot, set by `_admit` while the slot is held
#: (Feature 26, FTDD §7 constraint 1). `_admit` re-enters ONLY when this is the current task.
_SLOT_OWNER: "contextvars.ContextVar[Optional[asyncio.Task]]" = contextvars.ContextVar(
    "millm_slot_owner", default=None
)

#: Feature 25: what this request's generation actually did, for the route's headers — the seed
#: scope that was promised (X-miLLM-Seed) and the constraint that was applied
#: (X-miLLM-Constrained). Request-scoped like the probe verdicts above; reset with them.
_REQUEST_OUTCOME: "contextvars.ContextVar[Optional[dict[str, Any]]]" = contextvars.ContextVar(
    "millm_request_outcome", default=None
)


def reset_request_outcome() -> None:
    _REQUEST_OUTCOME.set({})


def note_request_outcome(**values: Any) -> None:
    """Record part of this request's outcome — always a new dict, never mutated in place."""
    _REQUEST_OUTCOME.set({**(_REQUEST_OUTCOME.get() or {}), **values})


def get_request_outcome() -> dict[str, Any]:
    return dict(_REQUEST_OUTCOME.get() or {})


#: Seed scopes (FR-25.13.6, FR-25.14). A response never claims a wider scope than measured.
SEED_SCOPE_REQUEST = "request"
SEED_SCOPE_BATCH_SHAPE = "batch-shape"
SEED_SCOPE_BEST_EFFORT = "best-effort"


def _rng_devices() -> list[int]:
    """Every CUDA device whose generator `torch.manual_seed` reseeds — all of them, once CUDA is
    initialised. Forking only the model's own cards would leave the others permanently reseeded
    (manual_seed seeds every device), so every initialised device is saved and restored."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        return list(range(torch.cuda.device_count()))
    return []


@contextlib.contextmanager
def seeded_rng(seed: Optional[int]) -> Iterator[None]:
    """Sampling inside this block draws from a stream seeded with `seed`; outside it the global
    generators are exactly as they were (FR-25.13.2). A no-op when `seed` is None.

    ⚠ BOTH LINES ARE LOAD-BEARING. Without `fork_rng` an unseeded request after a seeded one
    replays the seeded stream (the state is left at the seed); without `manual_seed` the seed is
    echoed and never applied. Each has a mutation control.
    """
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=_rng_devices()):
        torch.manual_seed(seed)
        yield


def _seed_kwargs(gen_config: Any) -> dict[str, Any]:
    """`{"seed": n}` for a seeded request, else `{}` — so an unseeded call is exactly the call it
    was before Feature 25 (stand-ins that take one positional argument keep working)."""
    seed = getattr(gen_config, "seed", None)
    return {"seed": seed} if seed is not None else {}


def reset_probe_verdicts() -> None:
    """Drop any probe verdicts left over from an earlier request in this context."""
    _PROBE_VERDICTS.set([])


def set_probe_verdicts(verdicts: list) -> None:
    _PROBE_VERDICTS.set(list(verdicts))


def get_probe_verdicts() -> list:
    """This request's verdicts. Empty when nothing was armed."""
    return list(_PROBE_VERDICTS.get())


def reset_steering_memo() -> None:
    """Drop any memoised steering-circuit verdict for this context.

    Called at the top of each chat request. An ASGI server may reuse a context
    across requests, so the reset is explicit — assuming a fresh context per
    request would repeat the very "it's request-scoped" mistake that made the
    previous memo process-wide.

    Also clears the apply-failure flag, for the identical reason: a stale True
    would suppress the rung header on an unrelated later request that steered
    perfectly well.
    """
    _STEERING_CIRCUIT_MEMO.set(_MEMO_UNSET)
    _CIRCUIT_APPLY_FAILED.set(False)
    # Probe verdicts reset here too, deliberately: a route that has to remember TWO resets is a
    # route that will one day remember one. Same context, same lifetime, same hazard.
    _PROBE_VERDICTS.set([])
    _REQUEST_OUTCOME.set({})


def note_circuit_apply_failed() -> None:
    """Record that this request's circuit dial did not apply."""
    _CIRCUIT_APPLY_FAILED.set(True)


def circuit_apply_failed() -> bool:
    """True if this request's circuit dial failed to apply.

    The rung echo MUST consult this before emitting a header: a rung phrase
    describes evidence for an intervention, and no intervention ran.
    """
    return _CIRCUIT_APPLY_FAILED.get()


@dataclass(frozen=True)
class ScoreSpec:
    """One prompt to score and its options (Feature 26's packed path; one per batch row)."""

    text: str
    add_special_tokens: bool
    allowed: Optional[list[int]]
    temperature: float
    top_k: int


@dataclass(frozen=True)
class PackedScore:
    scores: Any
    prompt_tokens: int


class LoadedModelInfo:
    """Information about the currently loaded model."""

    def __init__(self, name: str, model_id: int, loaded_at: datetime) -> None:
        self.name = name
        self.model_id = model_id
        self.loaded_at = loaded_at


def _make_event_stopping_criteria(event: "Event"):
    """Build a transformers StoppingCriteria that halts generate() when `event`
    is set.

    Used by the streaming path so that when the consumer stops early (a stop
    sequence matched, or the client disconnected) the background generate()
    thread ends promptly instead of running to max_new_tokens while holding the
    GPU and the request-queue slot.  Returns None if transformers' stopping
    criteria API is unavailable.
    """
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        return None

    class _EventStoppingCriteria(StoppingCriteria):
        def __init__(self, ev: "Event") -> None:
            self._ev = ev

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return self._ev.is_set()

    return StoppingCriteriaList([_EventStoppingCriteria(event)])


from dataclasses import dataclass


@dataclass(frozen=True)
class SensingRequestContext:
    """Begin-time snapshot carried through a sensed request (R3 #10: the
    positional 3-tuple had six touch points and a test fixture had already
    drifted off its contract). Frozen: the whole point is that a
    mid-request re-arm cannot rewrite it."""

    sae: Any
    profile_id: Optional[str]
    config: Any  # SensingConfig snapshot


def _make_id_capture_criteria():
    """Zero-copy token-id capture for streaming sensing context (Feature 11).

    Stopping criteria run every generation step with the full input_ids
    tensor; storing the reference survives early stops. Returns None when
    transformers' stopping-criteria API is unavailable.
    """
    try:
        from transformers import StoppingCriteria
    except Exception:
        return None

    class _IdCapture(StoppingCriteria):
        def __init__(self) -> None:
            self.latest_ids = None

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            self.latest_ids = input_ids
            return False

    return _IdCapture()



def _render_chat_template(
    template: str, messages: list[dict], *, bos: str, eos: str
) -> str:
    """Render a GGUF chat template with the generation prompt OPEN.

    `add_generation_prompt=True` is the whole point: it emits the header that
    starts the assistant's turn and stops, so the caller can append a partial
    answer and have the model resume it.

    The globals mirror what real templates reach for — `raise_exception` is used
    by most HuggingFace-derived templates to reject unsupported role orders, and
    a missing one turns a clear template error into an obscure UndefinedError.
    """
    import jinja2

    # LENIENT undefined, deliberately. Real templates reference variables that
    # only a tool-calling request supplies — the gemma-4-31b template fails
    # outright on `tools` under StrictUndefined. A chat template is arbitrary
    # third-party code shipped inside a model file; the standard variables are
    # passed explicitly below, and anything else it reaches for renders empty
    # rather than costing the caller their continuation.
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True
    )

    def _raise_exception(message: str):
        raise ValueError(message)

    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: __import__("datetime").datetime.now().strftime(fmt)

    return env.from_string(template).render(
        messages=messages,
        add_generation_prompt=True,
        bos_token=bos,
        eos_token=eos,
        # The tool-calling variables templates branch on. Absent, gemma-4-31b
        # raises before emitting anything.
        tools=None,
        tool_choice=None,
        documents=None,
    )


def _completion_as_chat(raw: dict) -> dict:
    """Reshape a raw completion into the chat-completion envelope.

    The continuation path calls `create_completion`, which returns
    `choices[].text`; every consumer downstream reads `choices[].message.content`.
    Translating here keeps that difference inside the one function that causes
    it, rather than teaching each caller about two shapes.
    """
    choices = []
    for choice in raw.get("choices") or []:
        choices.append(
            {
                "index": choice.get("index", 0),
                "message": {
                    "role": "assistant",
                    "content": choice.get("text") or "",
                },
                "finish_reason": choice.get("finish_reason"),
            }
        )
    out = dict(raw)
    out["choices"] = choices
    return out


def _completion_chunks_as_chat(stream):
    """The streaming counterpart: `text` deltas become `delta.content` deltas."""
    for chunk in stream:
        choices = []
        for choice in chunk.get("choices") or []:
            choices.append(
                {
                    "index": choice.get("index", 0),
                    "delta": {"content": choice.get("text") or ""},
                    "finish_reason": choice.get("finish_reason"),
                }
            )
        out = dict(chunk)
        out["choices"] = choices
        yield out


class InferenceService:
    """
    Handles inference for OpenAI-compatible endpoints.

    Thread safety notes:
    - One generation at a time via request queue
    - Model/tokenizer access is thread-safe for inference
    - Steering values applied via hooks (not thread-local)

    Attributes:
        request_queue: The request queue for managing concurrency
    """

    def __init__(
        self,
        model_service: Optional["ModelService"] = None,
        max_concurrent: int = 1,
        max_pending: int = 5,
        kv_cache_mode: str = "dynamic",
        speculative_model: Optional[str] = None,
        speculative_num_tokens: int = 5,
        enable_cbm: bool = False,
        cbm_config: Optional[dict] = None,
        cbm_force_serial_monitoring: bool = False,
    ) -> None:
        """
        Initialize the inference service.

        Args:
            model_service: Reference to ModelService for model info
            max_concurrent: Maximum concurrent GPU operations
            max_pending: Maximum pending requests in queue
            kv_cache_mode: KV cache mode ("static" or "dynamic")
            speculative_model: HF model ID for draft model (speculative decoding)
            speculative_num_tokens: Number of tokens for draft model to propose
            enable_cbm: Whether to enable continuous batching backend
            cbm_config: Configuration dict for CBM backend
            cbm_force_serial_monitoring: When True, route requests with SAE
                monitoring enabled through the serial path for accurate
                per-request activation attribution instead of CBM batching.
        """
        self._model_service = model_service
        self._request_queue = RequestQueue(max_concurrent, max_pending)
        if max_concurrent > 1:
            # The serial queue is a correctness boundary, not just a perf
            # knob: per-request steering overrides, monitoring attribution,
            # and sensing all assume exactly one generation mutates the
            # global SAE state at a time (011 R1). CBM is the supported
            # concurrency path.
            logger.warning(
                "max_concurrent_above_one_breaks_request_isolation",
                max_concurrent=max_concurrent,
                detail="per-request steering/monitoring/sensing require 1; "
                       "use the CBM backend for batching",
            )
        # Only the answer when NO model is loaded. Where inputs go is read from
        # the loaded model in _get_input_device; a bare "cuda" here meant GPU 0,
        # which on a multi-GPU node is often not where the model was placed.
        self._device = "cpu"
        self._model_state = LoadedModelState()
        self._kv_cache_mode = kv_cache_mode
        self._speculative_model_id = speculative_model
        # What the operator configured, kept apart from the live id above: a
        # draft that fails to load disables speculation, and a model reloaded
        # on another card must be able to try again.
        self._configured_speculative_model_id = speculative_model
        self._speculative_num_tokens = speculative_num_tokens
        # Lazy-loaded on first use. Thread-safety note: with max_concurrent=1
        # only one generate call is active at a time, so the double-init race
        # (two requests both seeing None and both loading the draft model) is
        # practically impossible. If max_concurrent is ever raised above 1, add
        # a threading.Lock here before reading/writing _draft_model.
        self._draft_model: Any = None
        # Set while a model is unloading, cleared when the next one is loaded:
        # no draft is loaded, or kept, in between (see on_model_unloading).
        self._draft_suspended = False
        # Advanced by every unload (on_model_unloading), which every model change
        # goes through. A draft whose load began under another epoch was placed
        # for another model and is not kept (see _get_draft_model). Review round
        # 3, 2026-09-14.
        self._model_epoch = 0
        # Feature 25: (model identity, GrammarCache) for the loaded model; dropped on unload.
        self._grammar_cache_entry: Optional[tuple[Any, GrammarCache]] = None
        # Advanced by every admitted request; an idle cache release scheduled under an
        # older value is void (_schedule_idle_cache_release).
        self._idle_release_generation = 0
        self._cbm_force_serial_monitoring = cbm_force_serial_monitoring

        # Continuous Batching backend. Initialised once in __init__ when
        # enable_cbm=True, then started in on_model_loaded(). The start() call
        # itself is not thread-safe but on_model_loaded() is only ever called
        # from the model-load worker thread, so no race exists in practice.
        self._cbm_backend: Any = None
        if enable_cbm:
            from millm.services.cbm_backend import ContinuousBatchingBackend

            self._cbm_backend = ContinuousBatchingBackend(**(cbm_config or {}))

    @property
    def request_queue(self) -> RequestQueue:
        """Get the request queue."""
        return self._request_queue

    def _unloading_refusal(self) -> Optional[ModelBusyError]:
        """The refusal for a request while the loaded model is being unloaded; None otherwise."""
        state = getattr(self, "_model_state", None)
        if getattr(state, "is_unloading", False) is not True:
            return None
        current = state.current
        return ModelBusyError(
            f"The model '{getattr(current, 'model_name', None)}' is being unloaded; retry once "
            "the unload finishes.",
            details={"model_id": getattr(current, "model_id", None), "unloading": True},
        )

    def refuse_if_unloading(self) -> None:
        """Refuse a request for the loaded model while it is being unloaded.

        Raises:
            ModelBusyError: the model is being unloaded (503 model_busy on /v1).
        """
        refusal = self._unloading_refusal()
        if refusal is not None:
            raise refusal

    @asynccontextmanager
    async def _admit(
        self, raise_refusal: bool = True, background: bool = False
    ) -> AsyncIterator[Optional[ModelBusyError]]:
        """A request-queue slot for work on the loaded model, or a refusal while that model unloads.

        THE way work takes a slot (test_unload_admission asserts no other caller
        of the queue's acquire). Hardware acceptance, 2026-09-14, item 11: an
        unload moves the weights to the CPU while the model still reports as
        loaded, and a request arriving 1-3 s in ran on the half-moved model and
        answered 500 ("index is on cuda:0, different from other tensors on cpu").

        Checked twice. Before queueing, so a request arriving during an unload
        is told to retry at once instead of waiting behind the requests the
        unload is draining. And again once the slot is held, because an unload
        can begin while a request waits: ModelService.unload_model marks the
        model before it drains the queue, and the drain waits for this slot, so
        a request that finds no mark here runs before any weight moves.

        With `raise_refusal` False the refusal is yielded instead of raised, for
        a stream whose 200 is already committed.

        Feature 26 (026 FTDD §7):

        * `background=True` takes the slot through `RequestQueue.acquire_background()` — a batch
          chunk, never counted as pending, always behind a waiting interactive request.
        * ⚠ RE-ENTRANT FOR THE TASK THAT HOLDS THE SLOT, AND ONLY THAT TASK. A batch chunk holds
          the one slot (MAX_CONCURRENT_REQUESTS=1) and runs each row through the unchanged
          synchronous service method, which enters `_admit()` itself; without re-entry that call
          waits forever for a slot its own task holds. The owner is a TASK, read from
          `_SLOT_OWNER`, and compared with `asyncio.current_task()`: a child task created inside
          the slot inherits the context variable but is a DIFFERENT task, so it queues normally
          and can never run concurrently with its parent by mistake. Re-entry still refuses
          while the model unloads, and does no bookkeeping (the outer acquisition owns it).
        """
        owner = _SLOT_OWNER.get()
        current = asyncio.current_task()
        # ⚠ `owner is current`, not merely "the variable is set": a child task inherits the
        # variable and must NOT inherit the slot (mutation control M3).
        if owner is not None and owner is current:
            refusal = self._unloading_refusal()
            if refusal is not None and raise_refusal:
                raise refusal
            yield refusal
            return
        refusal = self._unloading_refusal()
        if refusal is None:
            admitted = False
            try:
                slot = (
                    self._request_queue.acquire_background()
                    if background
                    else self._request_queue.acquire()
                )
                async with slot:
                    refusal = self._unloading_refusal()
                    if refusal is None or not raise_refusal:
                        if refusal is None:
                            admitted = True
                            # Work is running: a cache release scheduled before it is void.
                            self._idle_release_generation = (
                                getattr(self, "_idle_release_generation", 0) + 1
                            )
                        token = _SLOT_OWNER.set(current)
                        try:
                            yield refusal
                        finally:
                            _SLOT_OWNER.reset(token)
                        return
            finally:
                if admitted:
                    # The slot is free again (the `async with` has exited).
                    self._schedule_idle_cache_release()
        if raise_refusal:
            raise refusal
        yield refusal

    async def run_model_work(self, fn: Callable[[], _T]) -> _T:
        """Model work that is not generation, inside ONE admission slot, with every SAE suppressed.

        THE seam for probe scoring, probe parity and the parity forward at arm time (FR-27.6,
        T-73). Before Feature 27 neither the parity route nor the arm route took a slot at all, so
        a parity forward ran concurrently with a generation on the same model; and neither
        suppressed steering, so a profile steering an EARLIER layer moved the residual parity
        compared against miStudio's recorded scores.

        ⚠ The slot is taken through `_admit()` — the only way work takes one
        (`test_every_request_queue_slot_is_taken_through_admission`) — so a model being unloaded
        refuses before `fn` runs. Suppression is entered in `_unsteered_call`, INSIDE the worker
        thread, because it is per-thread: entered around the `await` it would suppress nothing in
        the thread that runs the forward.
        """
        async with self._admit():
            return await asyncio.to_thread(self._unsteered_call, fn)

    def _unsteered_call(self, fn: Callable[[], _T]) -> _T:
        """Run `fn` with every attached SAE suppressed — in THIS thread, the worker's."""
        with self._unsteered():
            return fn()

    def _schedule_idle_cache_release(self) -> None:
        """Once the queue has stayed idle for TRANSFORMERS_IDLE_CACHE_RELEASE_S, give the model's cards torch's unused cache back.

        torch keeps the blocks a finished request freed, and nvidia-smi — what
        miStudio's placement and every other tenant of the node read — counts
        them as used: after three requests the RTX 3080 Ti sat at 12,004 MiB used
        with 155 MiB free until the model was unloaded (hardware acceptance,
        2026-09-14). Only the out-of-memory path emptied the cache.

        Released only when idle, and never during a request (_release_idle_cache
        takes the queue's slot). The cost falls on the next request, which
        reserves those segments again: replaying a 2,000 + 64-token request on
        OLMo-2-13B's cuda:0 through the allocator model, 29 cudaMalloc calls for
        1,070 MiB; on Qwen2.5-7B, 13 for 428 MiB — against seconds of generation.
        Waiting first keeps back-to-back requests (a labeling run) from paying it
        on every request. A GGUF model's memory is llama.cpp's, not torch's.

        Not while continuous batching runs: its manager generates without a queue
        slot (the queue reads idle during CBM generation), so a release timed by
        the queue could run in the middle of a CBM request. A serial request —
        sampling parameters CBM cannot serve fall back to it — used to schedule
        one anyway.
        """
        from millm.core.config import settings

        # Called from _admit's `finally`: anything raised here would replace the
        # request's own answer (a 400 became an AttributeError in a test with a
        # stand-in queue). Housekeeping never does that.
        try:
            if self._engine_is_llamacpp() or self._use_cbm() or not self._loaded_gpu_indices():
                return
            delay = float(settings.TRANSFORMERS_IDLE_CACHE_RELEASE_S)
            # `occupied_count`, not `pending_count`: a batch chunk waiting for or holding the slot
            # is work too (026 FTDD §7), and a release must not be timed around it.
            if delay < 0 or getattr(self._request_queue, "occupied_count", 0):
                return
            generation = getattr(self, "_idle_release_generation", 0)
            loop = asyncio.get_running_loop()
            tasks = self.__dict__.setdefault("_idle_release_tasks", set())

            def start() -> None:
                task = loop.create_task(self._release_idle_cache(generation))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            loop.call_later(delay, start)
        except Exception as e:  # noqa: BLE001 - never replace the request's outcome
            logger.warning("idle_cache_release_not_scheduled", error=str(e))

    async def _release_idle_cache(self, generation: int) -> None:
        """Empty torch's cache on the model's cards, unless work has been admitted since `generation`.

        The checks and the slot come with no await between them, so no request can
        be admitted in between; holding the slot keeps any that arrive during the
        release from starting until it is done. Checked again here, not only when
        scheduled: continuous batching may have started since, and it holds no slot.
        """
        if generation != getattr(self, "_idle_release_generation", 0):
            return
        if self._request_queue.occupied_count or self._use_cbm():
            return
        indices = self._loaded_gpu_indices()
        if not indices or self._engine_is_llamacpp():
            return
        try:
            async with self._request_queue.acquire():
                freed = await asyncio.to_thread(_release_cached_gpu_memory, indices)
        except Exception as e:  # noqa: BLE001 - a full queue, a closed loop: nothing to release now
            logger.warning("idle_cache_release_skipped", error=str(e))
            return
        logger.info("idle_cache_released", freed_mb_by_device=freed)

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_state.is_loaded

    def get_loaded_model_info(self) -> Optional[LoadedModelInfo]:
        """
        Get info about the currently loaded model.

        Returns:
            LoadedModelInfo if a model is loaded, None otherwise
        """
        if not self._model_state.is_loaded:
            return None

        loaded = self._model_state.current
        if loaded is None:
            return None

        return LoadedModelInfo(
            name=loaded.model_name,
            model_id=loaded.model_id,
            loaded_at=loaded.loaded_at,
        )

    @property
    def _model(self) -> Any:
        """Get the loaded model."""
        if not self._model_state.is_loaded:
            raise RuntimeError("No model is loaded")
        return self._model_state.current.model

    @property
    def _tokenizer(self) -> Any:
        """Get the loaded tokenizer."""
        if not self._model_state.is_loaded:
            raise RuntimeError("No model is loaded")
        return self._model_state.current.tokenizer

    def _get_input_device(self) -> str:
        """
        Return the device where model inputs (input_ids) should be placed.

        A split model's `model.device` names its first parameter's device, which
        under accelerate dispatch need not be the embedding's. The answer comes
        from `gpu_placement.model_input_device`, shared with the loader's compile
        warm-up: the model's own `get_input_embeddings()` first, which knows a
        nested multimodal layout (gemma-4 keeps its text stack at
        `model.language_model`), then `hf_device_map`. This list named only flat
        layouts, so a split gemma-4 sent its inputs to whichever card the map
        listed first.
        """
        if not self._model_state.is_loaded:
            return self._device
        try:
            from millm.ml.gpu_placement import model_input_device

            device = model_input_device(self._model_state.current.model)
            if device is None:
                raise LookupError("the model names no input device")
            return device
        except Exception:
            # The loader recorded which cards the model is on; the first of
            # them beats a device the model does not live on.
            indices = getattr(self._model_state.current, "gpu_indices", None) or []
            if indices:
                return f"cuda:{indices[0]}"
            return self._device

    def _use_cbm(self) -> bool:
        """Whether to use continuous batching for generation."""
        return self._cbm_backend is not None and self._cbm_backend.is_running

    def cbm_enabled(self) -> bool:
        """The continuous batching manager is configured in this process (running or not).
        Structured output is refused while it is (FR-25.11.3)."""
        return self._cbm_backend is not None

    def _logits_width(self) -> int:
        """The model's logits width — not `len(tokenizer)`, which differs when a head pads its
        vocabulary; an xgrammar bitmask narrower than the logits would leave the tail unmasked."""
        model = self._model
        try:
            head = model.get_output_embeddings()
            if head is not None and getattr(head, "weight", None) is not None:
                return int(head.weight.shape[0])
        except Exception:  # noqa: BLE001 - fall back to the config
            pass
        config = getattr(model, "config", None)
        width = getattr(config, "vocab_size", None) or getattr(
            getattr(config, "text_config", None), "vocab_size", None
        )
        return int(width or len(self._tokenizer))

    def _grammar_cache(self) -> GrammarCache:
        """The current model's grammar cache, built on first use after a load (keyed by model id
        and `loaded_at`, so a reload never reuses a stale TokenizerInfo)."""
        from millm.core.config import settings

        loaded = self._model_state.current
        key = (getattr(loaded, "model_id", None), getattr(loaded, "loaded_at", None))
        entry = getattr(self, "_grammar_cache_entry", None)
        if entry is None or entry[0] != key:
            cache = GrammarCache(
                self._tokenizer,
                self._logits_width(),
                stop_token_ids(self._model, self._tokenizer),
                int(settings.STRUCTURED_OUTPUT_GRAMMAR_CACHE),
            )
            entry = (key, cache)
            self._grammar_cache_entry = entry
        return entry[1]

    async def _compile_constraint(self, request: Any) -> Optional[CompiledConstraint]:
        """Compile a request's `response_format` BEFORE the admission slot, off the event loop:
        a cold compile is 0.1-0.7 s on the served tokenizers and must never hold the GPU slot."""
        response_format = getattr(request, "response_format", None)
        if _constraint_type(response_format) is None:
            return None
        cache = self._grammar_cache()
        grammar = await asyncio.to_thread(cache.compile, response_format)
        return CompiledConstraint(
            response_format=response_format,
            grammar=grammar,
            vocab_size=cache.vocab_size,
            stop_ids=list(cache.stop_ids),
            header=constrained_header(response_format),
        )

    def _finish_constrained(
        self, constraint: CompiledConstraint, generated_ids: Any, text: str
    ) -> str:
        """`finish_reason` for a constrained generation (FR-25.12): complete iff the LAST
        generated token is a stop token — never matcher state, which is not consulted after the
        final token. A complete output is validated; a budget-ended one is "length", whether or
        not its partial text happens to parse.

        Raises:
            ConstrainedOutputInvalidError: a complete output that does not validate (500).
        """
        last = int(generated_ids[-1]) if len(generated_ids) > 0 else None
        if last is None or last not in constraint.stop_ids:
            return "length"
        try:
            validate_output(text, constraint.response_format)
        except MiLLMError as exc:
            logger.error(
                "constrained_output_invalid",
                schema_name=schema_name(constraint.response_format),
                length=len(text or ""),
                reason=exc.message,
            )
            raise
        return "stop"

    def _log_constrained(self, constraint: CompiledConstraint, tokens: int, finish: str) -> None:
        logger.info(
            "constrained_generation",
            format=constraint.header,
            schema_name=schema_name(constraint.response_format),
            tokens=tokens,
            finish_reason=finish,
            mask_ms=round(sum(p.mask_ms for p in constraint.processors), 3),
        )

    def _seed_scope(self, base: str) -> str:
        """The scope a seed can promise on this path: `base`, unless the continuous batching
        manager runs in this process — its thread draws from the same global generator, so the
        promise is only best-effort (FR-25.14.4: never claim wider than measured)."""
        return SEED_SCOPE_BEST_EFFORT if self._use_cbm() else base

    def seed_scope_for(self, request: Any) -> str:
        """The scope a request's seed will get, decided before it runs — for a streaming
        response, whose headers are sent before the generator body."""
        batched = bool(_request_extra_messages(request)) and not (
            isinstance(request, ChatCompletionRequest) and request.wants_scores()
        )
        return self._seed_scope(SEED_SCOPE_BATCH_SHAPE if batched else SEED_SCOPE_REQUEST)

    def loaded_model(self) -> Any:
        """The resident LoadedModel (engine, dtype, ...), or None. Read by the fingerprint."""
        return self._model_state.current if self._model_state.is_loaded else None

    @property
    def backend_name(self) -> str:
        """Active inference backend identifier for observability headers."""
        if self._engine_is_llamacpp():
            return "llamacpp"
        return "cbm" if self._use_cbm() else "serial"

    def _engine_is_llamacpp(self) -> bool:
        """Whether the resident model is served by llama.cpp.

        Reads the engine recorded at load time rather than sniffing the object:
        a duck-typed check ("does it have .config?") picks the wrong path the
        day either runtime grows an attribute the other has.
        """
        current = self._model_state.current
        return current is not None and not current.supports_hooks

    def get_backend_info(self) -> dict:
        """
        Return a description of the active inference backend and its capabilities.

        Used by the GET /api/health/inference endpoint so operators and
        clients
        can understand which path is serving requests and what its limitations are.
        """
        if self._engine_is_llamacpp():
            # Must still come FIRST, though no longer because of `streaming`:
            # this branch differs from serial on `per_request_profile_override`
            # and on `limitations`, and reporting the serial defaults would
            # advertise steering that the very next request is refused for.
            return {
                "backend": "llamacpp",
                "description": "llama.cpp (GGUF file served through llama-cpp-python)",
                "capabilities": {
                    "streaming": True,
                    "per_request_sampling_params": True,
                    "per_request_profile_override": False,
                    "speculative_decoding": False,
                },
                "context_length": getattr(
                    self._model_state.current, "context_length", 0
                ),
                "limitations": [
                    "batched conversations and n > 1 are not supported on "
                    "this engine",
                    "chat_template_kwargs is accepted but IGNORED: llama.cpp "
                    "applies the template baked into the GGUF file and exposes "
                    "no way to pass variables into it",
                    "no PyTorch module tree, so SAE attachment, steering and "
                    "sensing are impossible rather than merely unimplemented",
                    "reasoning_content is not populated for a model whose "
                    "chat template opens <think>: llama.cpp applies the "
                    "template internally, so the prompt that would reveal it "
                    "is never visible to miLLM. The trace is still delivered, "
                    "inline in content, and a client that parses think tags "
                    "itself (Open WebUI can) renders it correctly",
                ],
            }

        if self._use_cbm():
            backend: dict = {
                "backend": "cbm",
                "description": "ContinuousBatchingManager (high-throughput batching)",
                "capabilities": {
                    "streaming": True,
                    "per_request_sampling_params": False,
                    "per_request_profile_override": False,
                    "speculative_decoding": False,
                },
                "cbm_config": {
                    "default_temperature": getattr(
                        self._cbm_backend, "_default_temperature", None
                    ),
                    "default_top_p": getattr(
                        self._cbm_backend, "_default_top_p", None
                    ),
                    "max_queue_size": getattr(
                        self._cbm_backend, "_max_queue_size", None
                    ),
                },
                "limitations": [
                    "temperature and top_p are fixed at manager creation; "
                    "requests with different values fall back to the serial path",
                    "requests with a profile override fall back to the serial path",
                    "CBM_FORCE_SERIAL_MONITORING=true routes monitored requests "
                    "to the serial path for accurate activation attribution",
                ],
            }
        else:
            backend = {
                "backend": "serial",
                "description": "Serial request queue (one generation at a time)",
                "capabilities": {
                    "streaming": True,
                    "per_request_sampling_params": True,
                    "per_request_profile_override": True,
                    "speculative_decoding": self._speculative_model_id is not None,
                },
                "queue": {
                    "max_concurrent": self._request_queue.max_concurrent,
                    "max_pending": self._request_queue.max_pending,
                    "current_pending": self._request_queue.pending_count,
                },
                "limitations": [
                    "one generation active at a time; concurrent requests queue",
                ],
            }
        return backend

    @staticmethod
    def _has_steering_override(request: Any) -> bool:
        """
        True when the request carries a per-request steering override
        (profile and/or intensity dial) — such requests must route through
        the serial path: they mutate the process-global SAE steering state,
        which CBM-batched rows would share. getattr-based so schemas without
        the extension fields (text completions, embeddings) answer False.
        """
        return (
            bool(getattr(request, "profile", None))
            or getattr(request, "steering_intensity", None) is not None
        )

    def _cbm_route_kwargs(self, request: Any) -> dict[str, Any]:
        """Everything the CBM gate must see about a request, read in ONE place.

        Feature 25 (FR-25.3.6): the manager returns one choice and has no per-request seed or
        logits processor, so a request carrying `n > 1`, a `seed`, a `response_format`
        constraint or a frequency/presence penalty (its GenerationConfig is fixed at start-up)
        would be served with that field DROPPED. Each routes to the serial path instead.
        """
        return {
            "temperature": getattr(request, "temperature", None),
            "top_p": getattr(request, "top_p", None),
            "has_steering_override": self._has_steering_override(request),
            # FR-27.1g: a CBM batch row cannot be attributed to one request's positions.
            "wants_activations": getattr(request, "return_sae_activations", None) is not None,
            "n": _request_n(request),
            "seeded": _request_seed(request) is not None,
            "constrained": _constraint_type(getattr(request, "response_format", None)) is not None,
            "penalised": any(
                isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0
                for v in (getattr(request, "frequency_penalty", 0.0),
                          getattr(request, "presence_penalty", 0.0))
            ),
        }

    def _use_cbm_for_request(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        has_steering_override: bool = False,
        n: int = 1,
        seeded: bool = False,
        constrained: bool = False,
        penalised: bool = False,
        wants_activations: bool = False,
    ) -> bool:
        """
        Whether to route this specific request through the CBM backend.

        ContinuousBatchingManager uses a fixed GenerationConfig (temperature, top_p
        are baked in at manager creation). Requests with different sampling params
        must fall back to the serial path to preserve correctness.

        Requests carrying a per-request ``profile`` steering override must also
        fall back to serial: CBM does not run the per-request profile
        apply/restore logic (that lives in the serial path inside the request
        queue), so serving such a request via CBM would silently use the global
        steering state instead of the requested profile — the wrong causal
        influence with no client-visible signal.

        When cbm_force_serial_monitoring is True and SAE monitoring is active,
        requests are also routed to the serial path so that captured activations
        can be accurately attributed to this specific request (batch position ≠
        request ID in CBM, so monitoring data would be inexact otherwise).
        """
        if not self._use_cbm():
            return False
        if BATCH_ROW.get() is not None:
            # Feature 26 (FTASKS 5.3): a batch row runs INSIDE its chunk's admission slot, and the
            # manager generates with no slot at all — so a row routed there would run outside the
            # slot, beside whatever the next chunk or an interactive request does.
            logger.info("cbm_routing_fallback_to_serial", reason="batch_row")
            return False
        if self._unloading_refusal() is not None:
            # The manager holds no queue slot, so nothing there refuses a model
            # being unloaded: the serial path does (_admit).
            return False
        matches = self._cbm_backend.sampling_params_match(temperature, top_p)
        if not matches:
            # Elevated to INFO so operators can correlate latency jitter with
            # requests that silently fell back from CBM to the serial path.
            logger.info(
                "cbm_routing_fallback_to_serial",
                reason="sampling_params_mismatch",
                request_temperature=temperature,
                request_top_p=top_p,
                cbm_temperature=getattr(self._cbm_backend, "_default_temperature", None),
                cbm_top_p=getattr(self._cbm_backend, "_default_top_p", None),
            )
            return False
        if has_steering_override:
            logger.info(
                "cbm_routing_fallback_to_serial",
                reason="per_request_steering_override",
            )
            return False
        # Feature 25 (FR-25.3.6, FR-25.13.7): fields the manager cannot honour route serial.
        for flag, reason in (
            (n > 1, "n_gt_1"),
            (seeded, "seeded_request"),
            (constrained, "constrained_output"),
            (penalised, "repetition_penalty"),
            (wants_activations, "return_sae_activations"),
        ):
            if flag:
                logger.info("cbm_routing_fallback_to_serial", reason=reason)
                return False
        from millm.core.config import settings as _settings

        if _settings.SENSING_FORCE_SERIAL:
            # Armed sensing forces serial routing: CBM batch rows cannot be
            # attributed to requests (Feature 11 / SEN-S1). With forcing
            # off, CBM requests simply go unsensed (begin is never called).
            from millm.services.sae_service import AttachedSAEState

            _sae = AttachedSAEState().attached_sae
            if _sae is not None and _sae.is_sensing_armed:
                logger.info(
                    "cbm_routing_fallback_to_serial",
                    reason="sensing_armed",
                )
                return False

        # Feature 15: the same rule for circuit edge sensing. Asked of the
        # SERVICE, not the SAE registry — a circuit's armed state spans layers
        # and AttachedSAEState.attached_sae is only the FIRST entry, so a
        # circuit armed on layers 10+13 would go undetected if 10 were absent.
        if _settings.CIRCUIT_SENSING_FORCE_SERIAL:
            import millm.api.dependencies as _deps

            _circ_sensing = _deps._circuit_sensing_service
            if _circ_sensing is not None and _circ_sensing.is_armed:
                logger.info(
                    "cbm_routing_fallback_to_serial",
                    reason="circuit_sensing_armed",
                )
                return False
        # Probes (Feature 24). Asks the runtime registry, not the SAE registry — a probe may sit
        # on any layer and needs no attached SAE, so the sensing clause's shape would miss it.
        # Same reasoning as the circuit clause above.
        if _settings.PROBE_FORCE_SERIAL:
            from millm.services.probe_runtime import ProbeRuntimeState

            if ProbeRuntimeState().has_armed():
                logger.info("cbm_routing_fallback_to_serial", reason="probes_armed")
                return False
        if self._cbm_force_serial_monitoring and self._is_monitoring_enabled():
            logger.info(
                "cbm_routing_fallback_to_serial",
                reason="force_serial_monitoring_active",
            )
            return False
        return True

    def _release_draft_model(self) -> None:
        """Forget the speculative draft, so the next request loads it beside the NEW model.

        The draft is loaded once, on the input device of whatever model was
        loaded then, and was never dropped. After an unload and a load on
        another card — or a load that is now split — generation handed the old
        card's draft to the new model and every proposed token crossed cards,
        or failed on a device mismatch. Also re-arms a draft that failed to
        load, since the failure may have been the old card's lack of room.

        Never raises. ModelService calls both hooks inside a bare
        `except Exception: pass`, so a failure here would also silently skip
        starting continuous batching, with nothing logged.
        """
        configured = getattr(
            self, "_configured_speculative_model_id", getattr(self, "_speculative_model_id", None)
        )
        if getattr(self, "_draft_model", None) is not None:
            logger.info("draft_model_released", model_id=configured)
        self._draft_model = None
        self._speculative_model_id = configured

    def on_model_loaded(self) -> None:
        """Called after model is loaded. Drops any old draft; starts CBM if enabled."""
        self._draft_suspended = False
        self._release_draft_model()
        if self._cbm_backend is not None and self._model_state.is_loaded:
            # Continuous batching is transformers' ContinuousBatchingManager. It
            # would be handed a llama.cpp ctypes handle and a None tokenizer.
            # This is the highest-value early guard in the file: everything
            # downstream routes on _use_cbm(), and a manager that started on the
            # wrong object would fail deep inside generation instead of here.
            if not self._model_state.current.supports_hooks:
                logger.info(
                    "cbm_skipped",
                    reason="engine_has_no_module_tree",
                    engine=self._model_state.current.engine,
                )
                return
            gpu_indices = list(getattr(self._model_state.current, "gpu_indices", None) or [])
            if len(gpu_indices) > 1:
                # transformers' PagedAttentionCache puts every layer's KV blocks
                # on ONE device, `model.device` (transformers 5.15.1
                # continuous_api.py:1001, cache.py:257). The layers of a split
                # model that live on another card would write and read a cache
                # that is not on their card, so every request through the
                # manager fails inside its background thread. The serial path
                # runs through accelerate's dispatch hooks and serves a split
                # model correctly. Review round 1, 2026-09-14.
                logger.info(
                    "cbm_skipped", reason="model_split_across_gpus", gpu_indices=gpu_indices
                )
                return
            try:
                model = self._model_state.current.model
                tokenizer = self._model_state.current.tokenizer
                self._cbm_backend.start(model, tokenizer)
            except Exception as e:
                logger.warning("cbm_start_failed", error=str(e))

    def on_model_unloading(self) -> None:
        """Called before model unload. Drops the draft and keeps it dropped; stops CBM if running.

        The draft stays suspended until the next model is loaded. ModelService
        drains pending requests for up to five seconds AFTER this call, and each
        of them asked for the draft: released here, it was loaded again beside
        the model being unloaded and held its memory through the next load,
        whose placement read those cards as free. Review round 2, 2026-09-14.
        """
        self._model_epoch = getattr(self, "_model_epoch", 0) + 1
        self._draft_suspended = True
        self._release_draft_model()
        # A compiled grammar belongs to the unloaded tokenizer (Feature 25).
        self._grammar_cache_entry = None
        if self._cbm_backend is not None and self._cbm_backend.is_running:
            self._cbm_backend.stop()

    def _draft_device(self) -> str:
        """The card the speculative draft goes on, whole.

        A model on one card: its input device, beside the embeddings. A model
        split across cards: the one of its cards with the most free memory, read
        live. The planner fills a split's cards in index order, each but the
        last to its whole budget, and the lowest-index card holds the input
        embeddings — so the input device is the FULLEST card of the split. A
        bf16 draft there either failed to load, silently switching speculation
        off, or took the room kept back for that card's activations and KV
        cache. Assisted generation moves token ids between the draft's card and
        the model's (transformers generation/utils.py, candidate_generator.py).
        Review round 2, 2026-09-14.

        Free memory is read only on the model's own cards (free_mb_by_index
        creates a CUDA context on each card it asks); a card that cannot be read
        is not chosen, and when none can be, the input device.
        """
        indices = self._loaded_gpu_indices() if self._model_state.is_loaded else []
        if len(indices) > 1:
            from millm.ml.gpu_placement import free_mb_by_index

            free = free_mb_by_index(indices)
            if free:
                # On a tie, the lower index: max() keeps the first it meets.
                best = max(sorted(free), key=lambda index: free[index])
                return f"cuda:{best}"
        return self._get_input_device()

    def _is_sae_attached(self) -> bool:
        """Check if an SAE is currently attached (steering active)."""
        try:
            from millm.services.sae_service import AttachedSAEState
            return AttachedSAEState().is_attached
        except Exception:
            return False

    def _get_attached_sae(self) -> Any:
        """Return the currently attached LoadedSAE, or None."""
        try:
            from millm.services.sae_service import AttachedSAEState
            return AttachedSAEState().attached_sae
        except Exception:
            return None

    def count_prompt_tokens(self, request: Any, *, chat: bool) -> int:
        """The request's prompt length, tokenized as the path that will serve it does — for the
        pre-generation activation cap (FR-27.2f). Scoring renders without special tokens for chat
        (a template carries its own BOS) and honours `add_special_tokens` for text."""
        scoring = bool(getattr(request, "wants_scores", lambda: False)())
        if chat:
            text = self._format_chat_messages(request.messages, request.chat_template_kwargs)
            special = not scoring
        else:
            prompt = request.prompt
            text = prompt[0] if isinstance(prompt, list) else prompt
            special = bool(getattr(request, "add_special_tokens", True)) if scoring else True
        return len(self._tokenizer(text, add_special_tokens=special)["input_ids"])

    def _activations_begin(self, request: Any, n_prompt: int, read_point: Optional[str] = None):
        """Open this request's activation capture, if it asked for one. Never raises.

        Validation that should REFUSE a request (no matching SAE, shape, caps) runs in the route,
        before the slot (`request_activations.validate_request`); this seam only observes. The
        owner token goes into this task's context, which `asyncio.to_thread` copies into the
        worker that runs the forward — the SAE hook feeds the capture only from that context.
        """
        spec = getattr(request, "return_sae_activations", None)
        if spec is None:
            return None
        try:
            from millm.core.config import settings as _settings
            from millm.services.request_activations import (
                CAPTURE_OWNER,
                RequestActivationCapture,
                select_sae,
            )
            from millm.services.sae_service import AttachedSAEState

            entry = select_sae(spec, AttachedSAEState().entries())
            capture = RequestActivationCapture(
                spec=spec, n_prompt=int(n_prompt), sae=entry.sae, sae_id=entry.sae_id,
                layer=entry.layer, read_point=read_point or spec.read_point,
                encode_chunk=int(_settings.SAE_ACTIVATIONS_ENCODE_CHUNK),
            )
            entry.sae.begin_request_capture(capture)
            CAPTURE_OWNER.set(capture.owner)
            return capture
        except Exception as exc:  # noqa: BLE001 - an observer never fails a request
            logger.warning("sae_activations_begin_failed", error=str(exc))
            return None

    def _activations_close(self, capture) -> None:
        """Close `capture` if it is still open on its SAE. Idempotent; never raises."""
        if capture is None:
            return
        try:
            from millm.services.request_activations import CAPTURE_OWNER

            if capture.sae.request_capture is capture:
                capture.sae.end_request_capture()
            if CAPTURE_OWNER.get() == capture.owner:
                CAPTURE_OWNER.set(None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sae_activations_close_failed", error=str(exc))

    def _activations_finish(self, capture, full_ids) -> Optional[Any]:
        """Close the capture and build the response's `millm` object. Never raises."""
        if capture is None:
            return None
        self._activations_close(capture)
        try:
            from millm.api.schemas.millm_extension import MillmExtension

            block = capture.build(full_ids)
            logger.debug(
                "sae_activations", sae=capture.sae_id, positions=len(block["positions"]),
                entries=sum(len(p["features"]) for p in block["positions"]),
            )
            return MillmExtension(sae_activations=block)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sae_activations_finish_failed", error=str(exc))
            return None

    def _close_request_captures(self, reason: str) -> int:
        """Close any open per-request activation capture on every attached SAE. Never raises.

        Returns how many were open. Used by the hung-generation guard: a thread that outlives its
        request must not feed the next request's capture (FTID §12).
        """
        closed = 0
        try:
            from millm.services.sae_service import AttachedSAEState

            for entry in AttachedSAEState().entries():
                end = getattr(entry.sae, "end_request_capture", None)
                if callable(end) and end() is not None:
                    closed += 1
        except Exception as exc:  # noqa: BLE001 - a guard must not raise
            logger.warning("request_capture_close_failed", error=str(exc))
        if closed:
            logger.warning("request_captures_closed", reason=reason, count=closed)
        return closed

    @contextlib.contextmanager
    def _unsteered(self) -> Iterator[None]:
        """Every attached SAE inert for forward passes run IN THIS THREAD while the context is open.

        ⚠ EVERY ENTRY, NOT THE FIRST. Embeddings used `_get_attached_sae().suppressed()`, which is the
        FIRST attached SAE only, while a multi-layer circuit attaches one per layer — so "embeddings
        are never steered" was false whenever a circuit was serving (found 2026-10-04 reviewing the
        scoring path, which needs the same guarantee).

        ⚠ ENTER IT IN THE THREAD THAT RUNS THE FORWARD. Suppression is per-thread
        (`LoadedSAE._suppressed`), so it reaches exactly the pass it wraps and never a concurrent
        generation in another thread — entered around an `await asyncio.to_thread(...)` it would
        suppress nothing in the worker.
        """
        try:
            from millm.services.sae_service import AttachedSAEState

            entries = AttachedSAEState().entries()
        except Exception:  # noqa: BLE001 - no SAE service means nothing to suppress
            entries = []
        with contextlib.ExitStack() as stack:
            for entry in entries:
                stack.enter_context(entry.sae.suppressed())
            yield

    @staticmethod
    def _intensity_range_of(profile: Any) -> Optional[tuple[float, float]]:
        """The profile's declared intensity_range via the SHARED parser
        (millm.core.steering_range.declared_intensity_range) so the /v1 dial,
        management API, and import warnings interpret the document
        identically (010 R2 find)."""
        from millm.core.steering_range import declared_intensity_range

        if profile is None:
            return None
        return declared_intensity_range(getattr(profile, "cluster_meta", None))

    @classmethod
    def _plan_effective_intensity(
        cls,
        *,
        raw: "float | str | None",
        profile: Any,
        explicit: bool,
        steering_enabled: bool,
        has_live_values: bool,
    ) -> Optional[float]:
        """
        Pure decision core shared by _apply_request_steering and the echo
        header: resolves the raw dial value, caps it, and returns the
        effective lambda this request will run under — or None when apply
        will leave steering untouched (no-op). Symbolic resolution and the
        ceiling cap live IN here (010 R3: duplicating them at the two
        consumers was exactly how echo/apply drift survived R2). Keyword-
        only: three bool-ish params invite silent transposition otherwise.

        0.0 means "steering will be disabled for this request".
        """
        lam = cls._resolve_intensity(raw, profile)
        # Cap a numeric dial at the authored ceiling; cluster rows WITHOUT a
        # declared range cap at the config envelope the management API
        # enforces (010 R3: /v1 must never exceed what an authenticated
        # set_intensity would accept). Manual profiles keep the schema's
        # [0, 2] as their documented envelope.
        if lam is not None and profile is not None:
            rng = cls._intensity_range_of(profile)
            if rng is not None:
                hi: Optional[float] = rng[1]
            elif getattr(profile, "source_kind", None) == "cluster":
                from millm.core.config import settings

                hi = settings.CLUSTER_INTENSITY_MAX
            else:
                hi = None
            if hi is not None and lam > hi:
                lam = hi

        if profile is None and lam is None:
            return None
        if lam == 0.0:
            # Request-level "off" applies to whatever is running.
            return 0.0 if steering_enabled else None
        if profile is not None and profile.steering:
            if not explicit and not steering_enabled:
                return None  # dial-only never enables disabled steering
            effective = (lam if lam is not None
                         else profile.intensity if profile.intensity is not None
                         else 1.0)
            if effective == 0.0:
                # Stored intensity 0 with no dial: uniform disable semantics —
                # NOT an all-zero-enabled batch (010 R3: zero tensors still
                # fire apply_steering per token and report steering as on).
                return 0.0 if steering_enabled else None
            return effective
        if explicit and profile is not None:
            return None  # named profile with no steering — nothing to override
        if lam is None:
            return None
        if not has_live_values or not steering_enabled:
            return None  # nothing to scale; never enable unconfigured steering
        return lam

    @classmethod
    def _resolve_intensity(
        cls, raw: Optional[Any], profile: Any
    ) -> Optional[float]:
        """
        Resolve the request's steering_intensity to a numeric lambda.

        Numeric values pass through. Symbolic values resolve against the
        profile's declared budget.intensity_range (cluster rows), falling back
        to the configured envelope: "off" -> 0.0, "min" -> low, "max" -> high.
        None means "field absent - leave steering untouched".
        """
        if raw is None:
            return None
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        from millm.core.config import settings

        rng = cls._intensity_range_of(profile)
        lo, hi = (rng if rng is not None
                  else (settings.CLUSTER_INTENSITY_MIN, settings.CLUSTER_INTENSITY_MAX))
        return {"off": 0.0, "min": lo, "max": hi}[raw]

    async def ensure_profile_exists(self, profile_name: str) -> None:
        """
        Raise ProfileNotFoundError when the named profile doesn't exist.

        Used by the streaming route BEFORE committing a 200: apply-time
        validation runs inside the response generator, after headers are
        sent, so a bad profile name would otherwise abort the stream instead
        of returning the documented 404 (010 R2 find). Mirrors the existing
        pre-stream QueueFullError check.
        """
        from millm.core.errors import ProfileNotFoundError
        from millm.db.base import async_session_factory
        from millm.db.repositories.profile_repository import ProfileRepository

        async with async_session_factory() as session:
            repo = ProfileRepository(session)
            if await repo.get_by_name(profile_name) is None:
                raise ProfileNotFoundError(
                    f"Profile '{profile_name}' not found",
                    details={"profile": profile_name},
                )

    async def resolve_request_intensity(
        self,
        request: ChatCompletionRequest,
        *,
        ensure_named_profile: bool = False,
    ) -> Optional[float]:
        """
        Effective lambda for a request (for the X-miLLM-Steering-Intensity
        echo header). Best-effort by design: the header must never lie
        loudly nor fail a request over an observability nicety, so this
        returns None (no header) when nothing can apply — no SAE attached,
        a named profile that doesn't exist (apply will 404) — or when the
        DB read for a symbolic value fails. A concurrent profile switch
        between this pre-queue resolution and apply-time inside the
        semaphore can still skew a symbolic echo; that residual window is
        documented in the API reference.

        ensure_named_profile=True raises ProfileNotFoundError instead of
        suppressing when the request names a missing profile — the
        streaming route uses this so the 404 fires BEFORE the 200 commits,
        without a second profile read.
        """
        raw = getattr(request, "steering_intensity", None)
        if raw is None:
            return None
        try:
            from millm.services.sae_service import AttachedSAEState

            # Feature 14: mirror apply's ordering — a dial-only request over an
            # ACTIVE CIRCUIT resolves against the circuit's envelope, not the
            # profile's. Resolving it here (rather than only at apply) is what
            # keeps the echo header from drifting away from what actually runs,
            # which is exactly the class of bug Feature 10 R3 fixed by making
            # ONE decision core serve both.
            if not getattr(request, "profile", None):
                circuit_lam = await self._resolve_active_circuit_intensity(raw)
                if circuit_lam is not None:
                    return circuit_lam

            sae = AttachedSAEState().attached_sae
            if sae is None:
                return None  # apply will no-op; an echoed lambda would lie

            from millm.db.base import async_session_factory
            from millm.db.repositories.profile_repository import ProfileRepository

            profile_name = getattr(request, "profile", None)
            async with async_session_factory() as session:
                repo = ProfileRepository(session)
                profile = (await repo.get_by_name(profile_name)
                           if profile_name else await repo.get_active())
            if profile_name and profile is None:
                if ensure_named_profile:
                    from millm.core.errors import ProfileNotFoundError

                    raise ProfileNotFoundError(
                        f"Profile '{profile_name}' not found",
                        details={"profile": profile_name},
                    )
                return None  # apply will raise ProfileNotFound; don't echo first

            # Same decision core as apply (resolution + cap + no-op rules
            # all inside): None means apply will no-op — emit no header.
            return self._plan_effective_intensity(
                raw=raw,
                profile=profile,
                explicit=bool(profile_name),
                steering_enabled=sae.is_steering_enabled,
                has_live_values=bool(sae.get_steering_values()),
            )
        except MiLLMError:
            raise  # ensure_named_profile contract — not an echo failure
        except Exception as exc:
            # No exc_info: this fires per dialed request on an
            # unauthenticated endpoint — a DB outage must not become a
            # traceback-per-request log flood (010 R3).
            logger.warning(
                "intensity_echo_resolution_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None

    # F18: `_circuit_serving_members` and `_sae_service_for_dial` were
    # DELETED here.
    #
    # The first forwarded to `CircuitService._serving_members`; both are now
    # `CircuitSteeringEngine`. The second built an SAEService via `__new__`,
    # leaving four fields and two collections unset — a partially-constructed
    # object on the inference hot path that worked only because the dial
    # happened to touch none of them. `SAEService.for_registry()` constructs it
    # totally.

    async def _active_full_circuit(self) -> Optional[Any]:
        """The active circuit when it is serving in FULL multi-SAE mode.

        A slice-fallback circuit is steered by a cluster profile, so the
        ordinary profile path owns it — returning it here would double-apply.
        Best-effort: a DB hiccup must not fail a chat request.
        """
        try:
            from millm.db.base import async_session_factory
            from millm.db.repositories.circuit_repository import CircuitRepository

            async with async_session_factory() as session:
                actives = await CircuitRepository(session).list_active()

            # F19 R3-06: with SEVERAL circuits serving, no single one describes
            # the response.
            #
            # This read `get_active()`, which returns the most recently updated
            # row — so the dial, the intensity resolution and the rung header
            # all described ONE of two serving circuits while the response
            # carried both circuits' summed steering. An operator dialling
            # "the active circuit" changed a different circuit than the one the
            # header named.
            #
            # Same rule as composition, for the same reason: return None rather
            # than name one arbitrarily. A per-circuit dial is future work
            # (recorded in the FTASKS); until then, refusing to guess is the
            # only honest answer.
            full = [
                c for c in actives
                if getattr(c, "serving_mode", None) == "full"
            ]
            if not full:
                return None
            if len(full) > 1:
                logger.info(
                    "circuit_dial_ambiguous_several_serving",
                    circuit_ids=[getattr(c, "id", None) for c in full],
                    detail=(
                        "several circuits are serving, so no single circuit's "
                        "dial or evidence describes the response — the "
                        "per-request dial and the rung header are both "
                        "suppressed"
                    ),
                )
                return None
            return full[0]
        except Exception as e:
            # F18 R3-14: returning None here is indistinguishable from "no
            # circuit is active", which is the NORMAL case and is logged
            # nowhere. So during a Postgres blip every dialled request silently
            # degrades to unsteered AND drops the rung header, and an operator
            # watching the logs cannot tell "nothing is active" from "we could
            # not find out". `error=str(e)` alone loses the type and the
            # traceback that would say which it was.
            logger.warning(
                "active_circuit_lookup_failed",
                error=str(e),
                error_type=type(e).__name__,
                detail=(
                    "could not determine whether a circuit is active — this "
                    "request served UNSTEERED and dropped its rung header; "
                    "this is NOT the same as no circuit being active"
                ),
                exc_info=True,
            )
            return None

    @staticmethod
    def _circuit_definition(circuit: Any) -> Optional[Any]:
        """Parse a circuit row's stored ``circuit-definition/v1`` document."""
        from millm.api.schemas.circuit import CircuitDefinitionV1

        try:
            return CircuitDefinitionV1.model_validate(circuit.circuit_meta)
        except Exception:
            # F18 R3-13: this returned None with NO LOG ANYWHERE. A corrupt
            # `circuit_meta` therefore made both the dial and the rung echo
            # degrade to "nothing is steering" with zero operator-visible
            # signal — no warning, no counter, no header. The circuit still
            # reads ACTIVE in the management API and steers nothing, forever,
            # and the only way to discover it is to notice the model stopped
            # behaving differently.
            #
            # Going quietly dark is the failure mode this codebase treats as
            # worse than raising. Say it, once per call, with the reason.
            logger.warning(
                "circuit_definition_unparseable",
                circuit_id=getattr(circuit, "id", None),
                detail=(
                    "the stored circuit document no longer validates against "
                    "the v1 contract — this circuit reads active but cannot "
                    "steer; re-import it from miStudio"
                ),
                exc_info=True,
            )
            return None

    async def _steering_circuit(self) -> Optional[Any]:
        """The active circuit IF it is genuinely steering right now.

        The single predicate behind all three surfaces — the apply, the λ echo,
        and the rung echo. R1 fixed the λ echo's copy of these rules and left
        the rung echo's, so a response could still advertise
        ``X-miLLM-Circuit-Rung: 2`` while nothing was steering. Any surface that
        answers "what is steering" must ask THIS, never re-derive it.

        Memoised in a CONTEXTVAR, not on ``self``. R2 cached this on the
        service "which is request-scoped" — it is not: ``get_inference_service``
        is ``@lru_cache``'d and its own docstring reads "Singleton inference
        service", so the memo was written once per PROCESS and never
        invalidated. That advertised a deactivated circuit's rung header
        forever after the first request, and in the negative case permanently
        suppressed the rung disclosure while steering was live — resurrecting
        the exact overclaim R2 was written to kill. A contextvar cannot outlive
        the request that set it.
        """
        cached = _STEERING_CIRCUIT_MEMO.get()
        if cached is not _MEMO_UNSET:
            return cached
        result = await self._steering_circuit_uncached()
        _STEERING_CIRCUIT_MEMO.set(result)
        return result

    async def _steering_circuit_uncached(self) -> Optional[Any]:
        circuit = await self._active_full_circuit()
        if circuit is None:
            return None
        definition = self._circuit_definition(circuit)
        if definition is None:
            return None
        # F18: one derivation. `is_serveable` asks exactly the question this
        # predicate asked by hand — are there members, and is at least one of
        # their layers attached — from the SAME plan the apply drives. An
        # echoed rung header on a circuit that is not steering would attach an
        # evidence claim to an intervention that never happened.
        from millm.ml.circuit_steering import CircuitSteeringEngine
        from millm.services.sae_service import AttachedSAEState

        plan = CircuitSteeringEngine(AttachedSAEState()).plan_for(definition, circuit)
        if not plan.is_serveable:
            return None
        return circuit

    async def _resolve_active_circuit_intensity(
        self, raw: "float | str | None"
    ) -> Optional[float]:
        """Echo-side twin of the apply-side circuit resolution (same core)."""
        circuit = await self._steering_circuit()
        if circuit is None:
            return None
        return self._resolve_circuit_intensity(raw, circuit)

    async def _any_layer_composed(self) -> bool:
        """True if ANY live claim is composed (F19).

        Fails OPEN — deliberately, and this is a real trade-off rather than an
        oversight. An unreadable claim table reports NOT composed, so the rung
        header still describes a single circuit.

        F19 R1-07: this docstring previously claimed the opposite ("fails
        CLOSED"), as did a comment in `active_circuit_rung`, while the code
        below returned False. Two of the three statements were wrong, and a
        reader auditing this for honesty would have read the prose. Whichever
        behaviour is chosen, they must agree — a docstring that lies about a
        safety property is worse than either choice.

        The reasoning for fail-open: composition requires an explicit operator
        override and is rare; an unreachable claims table is comparatively
        common (a Postgres blip) and already degrades the rest of this path.
        Suppressing on every DB error would silently delete the rung disclosure
        for every request during a blip — losing an honesty signal far more
        often than it prevents a wrong one, and losing it in the direction that
        tells the operator LESS.

        The residual risk is stated, not hidden: during a blip WITH a live
        composition, a response carries a rung header describing one circuit
        when two contributed. The warning logged on that path says so
        explicitly, and the claim gate is what keeps composition rare.
        """
        try:
            from millm.db.base import async_session_factory
            from millm.services.circuit_claim_registry import CircuitClaimRegistry

            async with async_session_factory() as session:
                claims = await CircuitClaimRegistry(session).live_claims()
            return any(c.composed for c in claims)
        except Exception as e:
            # NOT fail-closed, deliberately, and this is a real trade-off.
            #
            # Composition requires an explicit operator override and is rare;
            # an unreachable claims table is comparatively common (a Postgres
            # blip) and already degrades the rest of this path. Suppressing on
            # every DB error would silently delete the rung disclosure for
            # every request during a blip — losing an honesty signal far more
            # often than it prevents a wrong one, and losing it in the
            # direction that tells the operator LESS.
            #
            # So: report not-composed, and say loudly that the answer is
            # unverified. The claim gate is what keeps composition rare; this
            # is a read of it, not the gate itself.
            logger.warning(
                "circuit_claims_unreadable_assuming_uncomposed",
                error=str(e),
                error_type=type(e).__name__,
                detail=(
                    "could not determine whether any layer is composed — "
                    "assuming not, so the rung header still describes a single "
                    "circuit; if a composition IS live this header understates "
                    "what produced the response"
                ),
            )
            return False

    async def active_circuit_rung(self) -> Optional[tuple[int, str]]:
        """`(rung, rung_language)` of the active full-serving circuit, or None.

        Feeds the ``X-miLLM-Circuit-Rung`` echo so a dial client can show what
        it is steering with. The phrase is rendered from the evidence ladder —
        never composed here — so the header can never overclaim.
        """
        circuit = await self._steering_circuit()
        if circuit is None:
            return None

        # F19: SUPPRESS the header when any served layer is COMPOSED. The rung
        # describes ONE circuit's evidence; when two circuits sum on a layer,
        # no single rung describes what the user actually received, and
        # emitting either one would overclaim. Same rule that already omits the
        # header for slice-fallback.
        #
        # An unreadable claims table reports NOT composed (see
        # `_any_layer_composed` — fail-OPEN, with the trade-off argued there).
        # The residual risk is a rung header describing one circuit during a DB
        # blip that hides a live composition; that path logs the ambiguity.
        if await self._any_layer_composed():
            logger.info(
                "circuit_rung_header_suppressed_composed",
                circuit_id=getattr(circuit, "id", None),
                detail=(
                    "a served layer carries more than one circuit, so no "
                    "single circuit's evidence describes the response"
                ),
            )
            return None

        from millm.core.circuit_evidence import rung_language

        # R3: an unguarded int() on a NULL/garbage rung column raised, and the
        # route swallows it with a bare except — silently disabling the rung
        # disclosure with nothing in the logs. Degrade DOWNWARD to MINED
        # instead, matching _coerce, and say so loudly.
        try:
            rung = int(circuit.rung)
        except (TypeError, ValueError):
            logger.warning(
                "circuit_rung_uncoercible_degraded_to_mined",
                circuit_id=getattr(circuit, "id", None),
                raw_rung=repr(getattr(circuit, "rung", None)),
            )
            rung = 0
        return rung, rung_language(rung)

    async def _apply_request_circuit_steering(
        self,
        intensity_raw: "float | str | None",
        request_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Per-request dial over an ACTIVE CIRCUIT (Feature 14).

        A circuit spans layers, so one global λ scales EVERY member together —
        each through its own layer's SAE. This is why the circuit dial cannot
        reuse the single-SAE path above: that one saves and restores exactly
        one SAE, which would leave the other layers permanently dialled.

        Returns the per-layer saved state for ``_restore_request_profile``, or
        None when there is no active circuit to dial (the caller then falls
        through to the profile/live-values path unchanged).

        Only ``serving_mode="full"`` is dialled here. A slice-fallback circuit
        is steered by a cluster PROFILE, which the ordinary profile path
        already handles correctly — dialling it here would double-apply.
        """
        from millm.api.schemas.circuit import CircuitDefinitionV1
        from millm.services.sae_service import AttachedSAEState

        circuit = await self._active_full_circuit()
        if circuit is None:
            return None

        lam = self._resolve_circuit_intensity(intensity_raw, circuit)
        if lam is None:
            return None

        # R2: derive the participating layers from the DEFINITION, the same
        # source the apply below uses. Keying the snapshot on circuit.layers
        # (the DB column) while applying to the definition's member layers let
        # any layer present in one and not the other be dialled but never
        # restored — a per-request override leaking permanently into global
        # state. The two must not be allowed to drift.
        definition = self._circuit_definition(circuit)
        if definition is None:
            return None
        # F18: ONE derivation. The snapshot below is keyed on
        # `plan.claimed_layers`, which is DEFINED as the layers of
        # `plan.members` — the same list the apply drives. F14-R2-01 was the
        # gap between the DB column and those member layers; making them the
        # same object closes it structurally rather than by agreement.
        from millm.ml.circuit_steering import CircuitSteeringEngine
        from millm.services.sae_service import SAEService

        state = AttachedSAEState()
        plan = CircuitSteeringEngine(state).plan_for(definition, circuit, intensity=lam)
        members = plan.members
        if not members:
            logger.info("circuit_dial_noop_no_serving_members",
                        circuit_id=circuit.id)
            return None
        # R2-06: `member_layers` was DELETED here — R1-08 replaced its only
        # consumer with `plan.claimed_entries` and left the assignment behind.
        # The claim set now travels with the plan, filtered into the entries.

        # R1-08: use the plan's OWN attachment snapshot rather than re-reading
        # the registry. `plan_for` already read it; a second read is both pure
        # overhead on the hot path and a drift window — a detach landing
        # between them meant the snapshot the plan reports and the entries this
        # request saves and restores disagree. A narrower version of exactly
        # the drift F18 exists to close.
        # R1-08: the entries the PLAN read, not a second registry read. A
        # detach between the two reads meant the snapshot the plan reports and
        # the entries this request saves and restores disagree — a narrower
        # version of exactly the drift F18 exists to close.
        entries = list(plan.claimed_entries)
        if not entries:
            logger.info("circuit_dial_noop_no_attached_layers",
                        circuit_id=circuit.id)
            return None

        # Feature 16 R1: capture the epoch HERE, with the snapshot it belongs
        # to — not at the return. Reading it after the apply absorbed any
        # operator write that landed during the apply window, so the restore
        # compared equal and reverted them: the exact defect F16 exists to fix
        # (TID §3.2 forbids the late read by name).
        saved_epoch = state.steering_epoch

        # Save EVERY participating layer before touching any of them, so the
        # restore is complete even if a later layer fails.
        saved_layers: list[dict] = [
            {
                "sae_id": e.sae_id,
                "layer": e.layer,
                "values": e.sae.get_steering_values(),
                "enabled": e.sae.is_steering_enabled,
            }
            for e in entries
        ]

        if lam == 0.0:
            # Clear as well as disable: set_circuit_steering (the λ>0 path)
            # clears each target SAE first, so disabling alone would leave the
            # previous values resident behind a false flag — visible to
            # get_steering_values and re-armed by any later enable.
            for e in entries:
                e.sae.clear_steering()
                e.sae.enable_steering(False)
            logger.info("circuit_dial_disabled", circuit_id=circuit.id,
                        layers=[e.layer for e in entries])
            return {"circuit": True, "epoch": saved_epoch,
                    "request_id": request_id,
                    "layers": saved_layers}

        # Re-derive from the AUTHORED basis rather than rescaling the live
        # values. Dividing live values by a stored λ cannot recover the basis:
        # (a) activation CLAMPS each member at ±200, and the overflow is gone —
        #     authored 150 at λ=2 stores clamp(300)=200, so 200/2×1 = 100, not
        #     the correct 150; and
        # (b) _serve_full applies `definition.budget.intensity`, which is a
        #     DIFFERENT field from `circuit.intensity` (the DB dial column), so
        #     the divisor was wrong for any circuit whose document declares a
        #     non-1.0 budget intensity.
        # Re-serving from the stored definition is the same path set_intensity
        # uses, so the dial and the management API agree by construction.
        #
        # `definition` and `members` were parsed above to derive the snapshot
        # layers. R3: this block used to re-parse and re-flatten them, leaving
        # two unreachable failure branches whose log events could never fire —
        # so an operator grepping for `circuit_dial_definition_unparseable` to
        # debug a silent no-op would wrongly conclude the document parsed.
        # R1-09: construct OUTSIDE the try. `for_registry` inside it meant a
        # construction fault would surface as `circuit_dial_apply_failed` with
        # an AttributeError string — an apply failure that never reached the
        # apply. That is precisely how R1-05's missing-attribute bug would have
        # presented, and how the two NameErrors during implementation did.
        dial_service = SAEService.for_registry()
        try:
            outcome = dial_service.set_circuit_steering(
                members,
                lam,
                edges=[e.model_dump(mode="json") for e in definition.edges],
                # A per-request apply is NOT authoritative: bumping here would
                # make this request supersede its own restore.
                authoritative=False,
            )
        except Exception as e:
            # The dial must never fail a chat request: restore what we saved
            # and fall through unsteered-by-this-dial.
            # F18 R3-01: retract the rung echo. The header was computed at
            # request entry, before this apply ran, so without this the
            # response advertises `X-miLLM-Circuit-Rung: 2; language="causally
            # validated (edge)"` for an intervention that provably did not
            # happen — an evidence claim about nothing, on the one surface a
            # dial client actually reads. `_steering_circuit`'s docstring names
            # this hazard; R1 closed it for the LOOKUP path and left the
            # apply-failure path open.
            note_circuit_apply_failed()
            # R3-05: `error=str(e)` alone cannot distinguish a real
            # misconfiguration (SAESetIncompleteError — a member's SAE is gone)
            # from a transient GPU hiccup, so an operator sees one undifferentiated
            # WARN either way. Name the type and keep the traceback.
            logger.warning(
                "circuit_dial_apply_failed",
                circuit_id=circuit.id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            self._restore_request_profile(
                {"circuit": True, "epoch": saved_epoch,
                 "request_id": request_id, "layers": saved_layers}
            )
            return None

        # R3: the dial discarded set_circuit_steering's result entirely, so a
        # dialled λ=2 could compound cross-layer hazards and clamp every member
        # while PUT /api/circuits/active/intensity — the same operation through
        # the management API — reports both. Two paths to one intervention, one
        # of them silent. The dial cannot put warnings in an OpenAI-shaped
        # response body, but it must not swallow them.
        hazards = list(getattr(outcome, "hazards", None) or [])
        clamps = list(getattr(outcome, "clamp_warnings", None) or [])
        logger.info(
            "circuit_dial_applied",
            circuit_id=circuit.id,
            intensity=lam,
            layers=[e.layer for e in entries],
            hazard_count=len(hazards),
            clamp_count=len(clamps),
        )
        if hazards or clamps:
            logger.warning(
                "circuit_dial_hazards",
                circuit_id=circuit.id,
                intensity=lam,
                hazards=[str(h) for h in hazards],
                clamp_warnings=[str(c) for c in clamps],
            )
        return {"circuit": True, "epoch": saved_epoch,
                "request_id": request_id,
                "layers": saved_layers}

    @classmethod
    def _resolve_circuit_intensity(
        cls, raw: "float | str | None", circuit: Any
    ) -> Optional[float]:
        """Resolve a dial value against the CIRCUIT's intensity envelope.

        Symbolic values resolve against the circuit's authored range when the
        stored document declares one, else the configured circuit envelope.
        A numeric dial is capped at the same ceiling so /v1 can never exceed
        what an authenticated ``PUT /api/circuits/active/intensity`` accepts.
        """
        from millm.core.config import settings

        if raw is None:
            return None

        lo = float(settings.CIRCUIT_INTENSITY_MIN)
        hi = float(settings.CIRCUIT_INTENSITY_MAX)
        # R3: the configured envelope is operator-set and unvalidated. Inverted
        # bounds would invert the dial itself ("max" → the floor), so normalise
        # here the way sae_service already does for its own envelope.
        if lo > hi:
            lo, hi = hi, lo
        budget = ((circuit.circuit_meta or {}).get("budget") or {})
        declared = budget.get("intensity_range")
        if isinstance(declared, list) and len(declared) == 2:
            try:
                d_lo, d_hi = float(declared[0]), float(declared[1])
                if d_lo > d_hi:
                    d_lo, d_hi = d_hi, d_lo
                # Intersect with the config envelope — an authored range must
                # not smuggle overdrive past the dial's own bounds.
                lo, hi = max(lo, d_lo), min(hi, d_hi)
                if lo > hi:
                    lo, hi = float(settings.CIRCUIT_INTENSITY_MIN), float(
                        settings.CIRCUIT_INTENSITY_MAX
                    )
            except (TypeError, ValueError):
                pass

        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            lam = float(raw)
            # R3: NaN and +inf both survive max(lo, min(hi, x)) and resolve to
            # the CEILING — a garbage dial silently producing the most
            # aggressive intervention available. Reject rather than fail open.
            if not math.isfinite(lam):
                return None
            # Dialling to 0 (off) is ALWAYS allowed, even below an authored floor.
            if lam == 0.0:
                return 0.0
            # Clamp to BOTH ends of the intersected envelope. R2: this capped
            # at `hi` but ignored `lo`, so a numeric dial could sit below an
            # authored floor that "min" itself refuses to go below — the
            # symbolic and numeric paths disagreeing about the same envelope.
            return max(lo, min(hi, lam))
        return {"off": 0.0, "min": lo, "max": hi}.get(raw)

    async def _apply_request_steering(
        self,
        profile_name: Optional[str],
        intensity_raw: "float | str | None" = None,
        request_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Apply per-request steering override: a named profile, an intensity
        dial (Feature 10), or both.

        Must be called INSIDE the request-queue semaphore so that only one
        request can mutate the global steering state at a time.  Saves the
        current steering state and returns it so _restore_request_profile can
        undo the override after generation completes.

        Semantics (010_FTDD):
        - profile_name set → that profile's λ=1-basis values are the base.
        - profile_name None + dial set → the ACTIVE profile is the base; with
          no active profile, the live steering values are treated as a λ=1
          base (never enabling steering that wasn't already enabled).
        - A request λ OVERRIDES the stored intensity (absolute dial, not a
          multiplier); λ absent (None) falls back to the stored intensity.
        - Effective λ == 0 disables steering for this request only.

        Returns None when there is nothing to override (no SAE attached, or
        nothing to scale) — no restore is needed and generation proceeds under
        the current global steering.

        Raises a MiLLMError subclass when the requested profile genuinely cannot
        be applied (profile not found, out-of-range feature index). Out-of-range
        VALUES no longer reject — they clamp to the steering range at apply time
        (Feature 8 / PADR v1.1: cluster strengths scaled by the intensity dial
        may legitimately exceed the range).  Raising rather than silently falling through is
        deliberate: the client explicitly asked for this profile's causal
        influence, so serving a response under the *wrong* steering would be a
        silent correctness failure.  The error propagates to the API layer and
        becomes a 4xx response.
        """
        from millm.services.sae_service import AttachedSAEState
        from millm.db.base import async_session_factory
        from millm.db.repositories.profile_repository import ProfileRepository
        from millm.core.errors import (
            InvalidFeatureIndexError,
            ProfileNotFoundError,
        )

        # Feature 14: an ACTIVE CIRCUIT is the base for a dial-only request —
        # its members span layers, so scaling them needs the multi-SAE path
        # (this function's single-SAE base would only ever reach layer[0]).
        # A named profile still wins: the client asked for that profile
        # explicitly.
        if not profile_name and intensity_raw is not None:
            circuit_saved = await self._apply_request_circuit_steering(
                intensity_raw, request_id=request_id
            )
            if circuit_saved is not None:
                return circuit_saved

        sae = AttachedSAEState().attached_sae
        if sae is None:
            # No SAE attached — a profile cannot steer anything.  This is not an
            # error (the base model still answers); log and proceed unsteered.
            logger.info(
                "request_profile_no_sae_attached",
                profile=profile_name,
            )
            return None

        async with async_session_factory() as session:
            repo = ProfileRepository(session)
            if profile_name:
                profile = await repo.get_by_name(profile_name)
                if not profile:
                    raise ProfileNotFoundError(
                        f"Profile '{profile_name}' not found",
                        details={"profile": profile_name},
                    )
            else:
                # Dial-only request: the base is the active profile (the
                # running cluster), or the live steering values if none.
                profile = await repo.get_active()

        explicit = bool(profile_name)

        from millm.core.steering_range import clamp_steering

        # Cluster gate parity (round-2 find): the per-request path must apply
        # the same declared-feature-space check as every other activation
        # path — index bounds alone can pass by coincidence on a mismatched
        # SAE, silently applying meaningless steering. Runs before ANY other
        # decision (pre-010 ordering) so that even an empty-membership
        # cluster authored for a different SAE refuses instead of falling
        # through to a live-values base.
        if profile is not None and getattr(profile, "source_kind", None) == "cluster":
            declared = ((profile.cluster_meta or {}).get("sae") or {}).get("n_features")
            if declared is not None and int(declared) != sae.d_sae:
                raise InvalidFeatureIndexError(
                    f"Profile '{profile.name}' is a cluster authored for an SAE "
                    f"with {declared} features; the attached SAE has "
                    f"{sae.d_sae} — steering would be meaningless.",
                    details={"profile": profile.name,
                             "declared_n_features": declared,
                             "d_sae": sae.d_sae},
                )

        # ONE decision core for "what will this request run under" — shared
        # with the echo header so the two can never drift (R2/R3 finds):
        # symbolic resolution and the ceiling cap happen inside the planner.
        # All no-op decisions live there too; the raises (gate above, index
        # validation below) stay here.
        effective = self._plan_effective_intensity(
            raw=intensity_raw,
            profile=profile,
            explicit=explicit,
            steering_enabled=sae.is_steering_enabled,
            has_live_values=bool(sae.get_steering_values()),
        )
        if effective is None:
            logger.info(
                "steering_intensity_noop",
                profile=profile.name if profile else None,
                explicit=explicit,
                intensity=intensity_raw,
                steering_enabled=sae.is_steering_enabled,
            )
            return None
        if (isinstance(intensity_raw, (int, float))
                and 0.0 < effective < float(intensity_raw)):
            # Numeric dial was capped at the authored/config ceiling —
            # observable for operators correlating dial requests (EC-10.2).
            logger.info(
                "request_intensity_capped_at_authored_max",
                requested=float(intensity_raw),
                applied=effective,
                profile=profile.name if profile else None,
            )

        if effective == 0.0:
            # Effective λ 0 disables steering for this request only —
            # uniformly, whatever the base would have been. NOTE: this
            # deliberately skips per-feature index validation (nothing is
            # applied), so a profile that would 400 at λ=0.01 succeeds at
            # λ=0 — pinned by test, documented in the API reference.
            saved: dict = {
                "values": sae.get_steering_values(),
                "enabled": True,
                "epoch": AttachedSAEState().steering_epoch,
                "request_id": request_id,
            }
            sae.enable_steering(False)
            logger.info(
                "request_steering_disabled",
                profile=profile.name if profile else None,
                base="profile" if (profile is not None and profile.steering)
                     else "live",
                intensity=0.0,
            )
            return saved

        if profile is not None and profile.steering:
            # The request dial is ABSOLUTE: it overrides the stored intensity
            # rather than multiplying it (010 pitfall 1); the planner already
            # folded the stored-λ fallback into `effective`.
            #
            # Parse and validate the profile's steering before mutating any
            # state, so a bad value fails cleanly without leaving partial
            # steering applied. Values are stored at lambda=1 basis (Feature
            # 8): scale by λ and CLAMP to the steering range rather than
            # reject — imported cluster strengths (contract ±300) times λ (≤2)
            # legitimately exceed ±200, and the documented semantics are
            # clamp-at-apply (PADR v1.1). Out-of-range indices still reject:
            # they are meaningless for the attached SAE.
            steering: dict[int, float] = {}
            for k, v in profile.steering.items():
                idx = int(k)
                if not 0 <= idx < sae.d_sae:
                    raise InvalidFeatureIndexError(
                        f"Profile '{profile.name}' references feature {idx}, "
                        f"out of range [0, {sae.d_sae}) for the attached SAE.",
                        details={"profile": profile.name, "feature_idx": idx,
                                 "d_sae": sae.d_sae},
                    )
                steering[idx] = clamp_steering(float(v) * effective)
        else:
            # Dial over live steering: the planner guaranteed live values
            # exist and steering is enabled (it returns None otherwise —
            # never enabling unconfigured steering, never falling through
            # for a named-but-empty profile).
            live = sae.get_steering_values()
            steering = {int(i): clamp_steering(float(v) * effective)
                        for i, v in live.items()}

        # Save the state we are about to overwrite
        saved = {
            "values": sae.get_steering_values(),
            "enabled": sae.is_steering_enabled,
            "epoch": AttachedSAEState().steering_epoch,
            "request_id": request_id,
        }

        # set_steering_batch MERGES into the live dict (sae_wrapper) — clear
        # first so the request runs under EXACTLY its base, not the union of
        # the base and whatever live steering existed (010 R3: a named
        # profile was silently superimposed on operator-set values; restore
        # already clears, apply didn't).
        sae.clear_steering()
        sae.set_steering_batch(steering)
        sae.enable_steering(True)

        logger.info(
            "request_steering_applied",
            profile=profile.name if profile else None,
            intensity=effective,
            features=len(steering),
        )
        return saved

    def _restore_request_profile(self, saved: Optional[dict]) -> None:
        """
        Restore SAE steering to the state it was in before this request's
        profile override.  Always called in a finally block.

        If saved is None (_apply_request_steering found nothing to override)
        this is a no-op.
        """
        if saved is None:
            return
        try:
            from millm.services.sae_service import AttachedSAEState

            state = AttachedSAEState()

            # Feature 16: an authoritative writer (an operator activating,
            # deactivating or re-dialling; an attach or detach) may have landed
            # between our save and now. Restoring the pre-request snapshot would
            # silently undo them — and set_intensity would already have told
            # them it succeeded. The later authoritative writer wins.
            #
            # The guard sits ABOVE both branches so a saved shape added later
            # inherits it by default rather than by someone remembering. A
            # snapshot with no "epoch" key (older state) proceeds as before.
            #
            # R3 finding 1: the apply-failure rollback used to rely on that
            # same absence, which conflated "deliberate exemption" with "old
            # state" — and the exemption was WRONG. `set_circuit_steering` can
            # raise arbitrarily late, so an operator write landing during a
            # failing apply was silently reverted by a rollback that always
            # proceeded. R2 deleted the revert ledger arguing "once the guard
            # works, an in-flight restore CANNOT revert an operator"; this path
            # was the counterexample. The rollback now carries its epoch like
            # every other caller and is exempt from nothing.
            saved_epoch = saved.get("epoch")
            current_epoch = state.steering_epoch
            if saved_epoch is not None and saved_epoch != current_epoch:
                logger.info(
                    "request_restore_skipped_superseded",
                    saved_epoch=saved_epoch,
                    current_epoch=current_epoch,
                    path="circuit" if saved.get("circuit") else "profile",
                    # FR-16.3: without this a skip cannot be correlated to the
                    # request that caused it in a concurrent log stream.
                    request_id=saved.get("request_id"),
                    # R1: name the layers left holding this request's transient
                    # values, since skipping means they are NOT restored.
                    layers_left_dialled=[
                        lay.get("layer") for lay in (saved.get("layers") or [])
                    ] or None,
                )
                return

            # Feature 14: a circuit dial saved EVERY participating layer.
            # Restoring only the first would leave the other layers dialled
            # for every subsequent request — a per-request override leaking
            # into global state.
            if saved.get("circuit"):
                for entry_state in saved.get("layers", []):
                    # Each layer restores INDEPENDENTLY: without this, one
                    # failing layer aborted the loop and left the remaining
                    # layers permanently dialled — a per-request override
                    # leaking into global state, the exact thing restore exists
                    # to prevent.
                    try:
                        entry = state.get(entry_state["sae_id"], entry_state["layer"])
                        if entry is None or entry.sae is None:
                            # Detached mid-request; nothing to restore there.
                            continue
                        entry.sae.clear_steering()
                        if entry_state["values"]:
                            entry.sae.set_steering_batch(entry_state["values"])
                        entry.sae.enable_steering(entry_state["enabled"])
                    except Exception as layer_error:
                        logger.warning(
                            "request_circuit_layer_restore_failed",
                            layer=entry_state.get("layer"),
                            error=str(layer_error),
                        )
                logger.debug(
                    "request_circuit_steering_restored",
                    layers=[s["layer"] for s in saved.get("layers", [])],
                )
                return

            sae = state.attached_sae
            if sae is None:
                return

            sae.clear_steering()
            if saved["values"]:
                sae.set_steering_batch(saved["values"])
            sae.enable_steering(saved["enabled"])

            logger.debug("request_profile_steering_restored")
        except Exception as e:
            logger.warning("request_profile_restore_failed", error=str(e))

    def _is_monitoring_enabled(self) -> bool:
        """Check if SAE feature monitoring is currently enabled."""
        try:
            from millm.services.sae_service import AttachedSAEState
            sae_state = AttachedSAEState()
            sae = sae_state.attached_sae
            return sae is not None and sae.is_monitoring_enabled
        except Exception:
            return False

    def _get_draft_model(self) -> Any:
        """
        Lazy-load the draft model for speculative decoding.

        Returns the draft model if configured, None if not configured or load failed.

        SAE steering is compatible with speculative decoding: the SAE hook fires on
        the main model's verification pass (where it applies correctly), not on the
        draft model. The draft model proposes tokens without knowledge of steering,
        so acceptance rate is lower than baseline, but output correctness is
        preserved — every accepted token was verified by the steered main model.
        Monitoring captures real main-model activations from verification passes.
        """
        if self._speculative_model_id is None:
            return None
        if getattr(self, "_draft_suspended", False):
            # The model is unloading. A request drained during the unload would
            # otherwise load the draft again, beside the model being removed,
            # and hold its memory while the next load reads the cards as free.
            # Generation without a draft is the same text, only slower.
            return None

        if self._draft_model is None:
            try:
                from transformers import AutoModelForCausalLM

                epoch = getattr(self, "_model_epoch", 0)
                device = self._draft_device()
                logger.info(
                    "loading_draft_model",
                    model_id=self._speculative_model_id,
                    device=device,
                )
                # Whole, on one card. device_map="auto" spread the draft over
                # every card, so each proposed token crossed cards before the
                # main model could verify it.
                # At the draft checkpoint's OWN precision (`ml/native_dtype.py`). This was
                # bfloat16 whatever the checkpoint recorded.
                draft = AutoModelForCausalLM.from_pretrained(
                    self._speculative_model_id,
                    torch_dtype=_draft_torch_dtype(self._speculative_model_id),
                    device_map={"": device},
                )
                if getattr(self, "_model_epoch", 0) != epoch:
                    # The model changed while this draft was loading: an unload
                    # began, or an unload AND the next load both finished. Round
                    # 2 checked the suspension flag, which that next load had
                    # already cleared, so a draft that outlasted the whole switch
                    # (one downloaded from the Hub on first use takes minutes)
                    # was kept: placed for the old model's card, holding memory
                    # the new model's placement had read as free. Review round
                    # 3, 2026-09-14.
                    logger.info("draft_model_discarded_model_changed")
                    del draft
                    _return_cached_draft_memory()
                    return None
                draft.eval()
                self._draft_model = draft
                if getattr(self, "_model_epoch", 0) != epoch:
                    # ...and again once it is KEPT. No lock guards the draft: the
                    # unload runs on the event loop while this runs in a
                    # generation thread, and one landing between the check above
                    # and the assignment advanced the epoch, found no draft to
                    # release, and this kept the draft for the model being
                    # removed. Publishing first and checking after closes that:
                    # whichever runs second sees the other. Review round 4.
                    if self._draft_model is draft:
                        self._draft_model = None
                    logger.info("draft_model_discarded_model_changed")
                    del draft
                    _return_cached_draft_memory()
                    return None
                logger.info("draft_model_loaded", model_id=self._speculative_model_id)
            except Exception as e:
                logger.warning(
                    "draft_model_load_failed",
                    error=str(e),
                    model_id=self._speculative_model_id,
                )
                self._speculative_model_id = None  # Disable future attempts
                return None

        return self._draft_model

    # =========================================================================
    # Generation Helpers
    # =========================================================================

    def _slice_generated(self, sequence, prompt_len: int):
        """Return only the newly generated tokens, for EITHER model family.

        A decoder-only model returns [prompt..., generated...], so the prompt
        must be sliced off. An ENCODER-DECODER model returns only the decoder
        output — the prompt never appears in it — and slicing it discards the
        answer.

        Verified on Falconsai/text_summarization (T5-small, is_encoder_decoder
        True): a 30-token prompt produced a 23-token summary, so
        `outputs[0][30:]` returned an empty string. Every seq2seq request came
        back HTTP 200 with content "" and no error anywhere — the failure was
        completely silent.
        """
        try:
            if bool(getattr(self._model.config, "is_encoder_decoder", False)):
                return sequence
        except Exception:
            pass
        return sequence[prompt_len:]

    def _build_generate_kwargs(
        self, gen_config: GenerationConfig, inputs: dict
    ) -> dict:
        """
        Build kwargs for model.generate() from GenerationConfig.

        Uses to_generate_kwargs() for proper penalty mapping, then adds
        tokenizer-specific pad/eos tokens and KV cache mode.
        """
        # Inject cache mode from server config if not already set
        if gen_config.cache_implementation is None and self._kv_cache_mode == "static":
            # `replace`, not a field-by-field copy, so the seed and the constraint survive.
            gen_config = dataclasses.replace(gen_config, cache_implementation="static")
        kwargs = gen_config.to_generate_kwargs()

        # Newer transformers pre-allocates the KV cache before _prefill via _init_cache.
        # Hybrid/mamba models (GraniteMoEHybrid, etc.) require cache_implementation="hybrid"
        # — a DynamicCache or StaticCache causes _update_mamba_mask to raise ValueError.
        # Detection priority (hybrid/mamba check must come FIRST because it's the strongest
        # architectural constraint — a "static" server-level or generation_config setting
        # is wrong for these architectures and must be overridden):
        model_type = getattr(getattr(self._model, "config", None), "model_type", "")
        if "hybrid" in model_type.lower() or "mamba" in model_type.lower():
            kwargs["cache_implementation"] = "hybrid"
        elif "cache_implementation" not in kwargs or kwargs.get("cache_implementation") is None:
            model_cache_impl = getattr(
                getattr(self._model, "generation_config", None),
                "cache_implementation",
                None,
            )
            if model_cache_impl:
                kwargs["cache_implementation"] = model_cache_impl
        kwargs.update({k: v.to(self._get_input_device()) for k, v in inputs.items()})
        # `or` is wrong here: a pad_token_id of 0 is FALSY, so a model that
        # legitimately uses id 0 as its pad token silently got eos as the pad
        # filler instead. gemma-4-12B-it is exactly that case
        # (generation_config.json: pad_token_id 0, eos_token_id [1, 106, 50]).
        #
        # Harmless while every request is a single sequence — nothing is padded,
        # so the value is never written. It stops being harmless the moment a
        # batch is generated: transformers fills finished rows with this id each
        # step, so the wrong value lands in every early-finishing row.
        _pad = self._tokenizer.pad_token_id
        kwargs["pad_token_id"] = (
            _pad if _pad is not None else self._tokenizer.eos_token_id
        )

        # Do NOT replace the model's EOS list with the tokenizer's single id.
        #
        # `tokenizer.eos_token_id` is a scalar. Many chat models stop on SEVERAL
        # tokens and declare them in generation_config.json — gemma-4-12B-it
        # ships `eos_token_id: [1, 106, 50]`, where 106 is <end_of_turn>.
        # Assigning the scalar REPLACES that list, so the model closes its turn,
        # the closing token is not honoured, and generation runs on to
        # max_new_tokens. Observed against gemma-4-12B-it: valid JSON, then a
        # bare "thought" (a vocab token that survives skip_special_tokens=True),
        # then the same JSON again, repeating until the cap. It cost ~1.7x the
        # tokens of a correct stop on every single request.
        #
        # Union instead: keep everything the model declares, and add the
        # tokenizer's id only if it is missing. A model that declares nothing
        # still falls back to the tokenizer, which is what this line was for.
        # Only INTEGER ids are accepted. A generation_config carrying anything
        # else is treated as declaring nothing and we fall back to the
        # tokenizer, rather than forwarding a value generate() cannot use.
        raw_eos = getattr(
            getattr(self._model, "generation_config", None), "eos_token_id", None
        )
        if isinstance(raw_eos, int) and not isinstance(raw_eos, bool):
            declared = [raw_eos]
        elif isinstance(raw_eos, (list, tuple)):
            declared = [i for i in raw_eos if isinstance(i, int) and not isinstance(i, bool)]
        else:
            declared = []

        tok_eos = self._tokenizer.eos_token_id
        if isinstance(tok_eos, int) and not isinstance(tok_eos, bool) and tok_eos not in declared:
            declared.append(tok_eos)

        if declared:
            kwargs["eos_token_id"] = declared[0] if len(declared) == 1 else declared
        else:
            kwargs["eos_token_id"] = tok_eos

        # Make the OpenAI `stop` parameter actually STOP generation.
        #
        # It was previously honoured only as post-generation string truncation,
        # so a request with `stop` still generated every token up to
        # max_new_tokens and merely had the tail trimmed off — measured against
        # gemma-4-12B-it, passing `stop` flipped finish_reason to "stop" while
        # completion_tokens and latency were unchanged. transformers supports
        # this natively via `stop_strings`, which additionally requires the
        # tokenizer to be handed to generate().
        #
        # The caller's post-hoc truncation stays as-is: it is still needed for
        # the streaming path, and it makes the boundary exact when a stop string
        # spans a token.
        stop_sequences = getattr(gen_config, "stop_sequences", None)
        if stop_sequences:
            kwargs["stop_strings"] = list(stop_sequences)
            kwargs["tokenizer"] = self._tokenizer

        draft_model = self._get_draft_model()
        if draft_model is not None:
            kwargs["assistant_model"] = draft_model
            kwargs["num_assistant_tokens"] = self._speculative_num_tokens
            if self._is_sae_attached():
                # Draft model is unsteered; acceptance rate is lower but output
                # correctness is maintained — all accepted tokens are verified by
                # the steered main model.
                logger.debug("speculative_decoding_with_sae_attached_lower_acceptance_rate_expected")

        # Structured output (Feature 25, FR-25.10.5): a FRESH processor per generate() call —
        # matchers are stateful, and the serial n-loop and each batched chunk call this anew.
        constraint = getattr(gen_config, "constraint", None)
        if constraint is not None:
            if not isinstance(constraint, CompiledConstraint):
                # A path that never compiled the constraint must not generate unconstrained.
                raise ResponseFormatUnsupportedError(
                    "response_format reached generation without a compiled constraint on this "
                    "path; it is refused rather than ignored.",
                    details={"param": "response_format"},
                )
            from transformers import LogitsProcessorList

            processor = JsonConstraintProcessor(
                constraint.grammar, int(inputs["input_ids"].shape[0]), constraint.vocab_size
            )
            constraint.processors.append(processor)
            processors = kwargs.get("logits_processor") or LogitsProcessorList()
            processors.append(processor)
            kwargs["logits_processor"] = processors
            if kwargs.pop("assistant_model", None) is not None:
                # Assisted generation proposes tokens the processor never sees.
                kwargs.pop("num_assistant_tokens", None)
                logger.info("speculative_disabled_for_constraint", format=constraint.header)

        return kwargs

    # ── Probe monitors (Feature 24) ───────────────────────────────────────────
    #
    # Three calls, mirroring the sensing lifecycle: `_probe_begin` at the request boundary,
    # `_probe_finish` BEFORE the response is committed (the verdict has to exist before the
    # header or the terminal chunk is written), and `_probe_record` in a `finally` so an event
    # is written even when generation failed.
    #
    # ⚠ NONE OF THEM MAY RAISE. A probe is an observer; taking a generation down because a
    # monitor failed would make arming one strictly worse than not.

    def _probe_begin(self, request_id: str):
        """Open the probe request boundary. Returns the context, or None when nothing is armed."""
        try:
            from millm.services.probe_runtime import ProbeRuntimeState

            state = ProbeRuntimeState()
            if not state.has_armed():
                return None
            return state.begin_request(request_id)
        except Exception as exc:
            logger.warning("probe_begin_failed", error=str(exc))
            return None

    def _probe_begin_detached(self, request_id: str, reason: str):
        """A probe context for a path that never scores, marked with `reason`. Never raises.

        ⚠ DETACHED: built over the armed probes and NEVER registered with `ProbeRuntimeState`
        (FR-27.8h). The runtime holds one request context, and `begin_request` refuses a second
        ("already open"), which `_probe_begin` turns into `None` — so with `PROBE_FORCE_SERIAL`
        off, the second of two concurrent continuous-batching requests recorded nothing at all,
        and `_probe_record`'s `end_request()` could close the OTHER request's context. A path that
        cannot score has no use for the shared slot: the hook has nothing to observe for it, and
        the event only has to say why.

        Returns None when nothing is armed, so an unarmed server is unchanged.
        """
        try:
            from millm.services.probe_runtime import ProbeRequestContext, ProbeRuntimeState

            state = ProbeRuntimeState()
            if not state.has_armed():
                return None
            context = ProbeRequestContext(request_id, state.armed())
            context.mark_not_scored(reason)
            return context
        except Exception as exc:
            logger.warning("probe_begin_detached_failed", error=str(exc))
            return None

    def _probe_note_prompt_length(self, context, n_prompt_tokens: int) -> None:
        """Tell the open probe context where the prompt ends. Safe when nothing is armed.

        ⚠ EXPLICIT RATHER THAN INFERRED. The context could take the first forward pass's length
        as the prompt length, which is right for an ordinary generation and silently wrong under
        chunked prefill or speculative decoding. The failure mode is a `prompt`-scoped probe
        scoring the model's own output as though it were the user's words — the precise confusion
        the role mask exists to prevent — so the boundary is stated by whoever tokenized the
        prompt, and a context never told reports `prompt_boundary_unknown` instead of guessing.
        """
        if context is None:
            return
        try:
            context.set_prompt_length(int(n_prompt_tokens))
        except Exception as exc:  # a probe must never break generation
            logger.warning("probe_prompt_length_failed", error=str(exc))

    def _probe_note_last_user_span(self, context, messages, served_ids, template_kwargs) -> None:
        """Tell the open probe context where the newest user message sits (`last_user`).

        Computed by rendering the same prefixes miStudio calibrated the window over
        (`probe_turns.last_user_token_span`); every failure is a stated reason on that window's
        verdict, never a guessed span. Safe when nothing is armed.
        """
        if context is None:
            return
        # Only when an armed probe reads it: three template renders and tokenizations per request
        # are not free on a long conversation (review round 1, M3).
        if not any("last_user" in (p.windows or ()) for p in getattr(context, "probes", ())):
            return
        try:
            from millm.services.probe_turns import last_user_token_span

            ids = served_ids[0] if hasattr(served_ids, "dim") and served_ids.dim() == 2 else served_ids
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            span, reason = last_user_token_span(
                self._tokenizer,
                [{"role": m.role, "content": m.content} for m in messages],
                ids,
                template_kwargs,
            )
            context.set_last_user_span(span, reason)
        except Exception as exc:  # a probe must never break generation
            logger.warning("probe_last_user_span_failed", error=str(exc))
            context.set_last_user_span(None, "last_user_span_unresolved")

    def _probe_mark_not_scored(self, reason: str) -> None:
        """Record why the request in flight cannot be scored. Safe when nothing is armed."""
        try:
            from millm.services.probe_runtime import ProbeRuntimeState

            context = ProbeRuntimeState().current_request()
            if context is not None:
                context.mark_not_scored(reason)
        except Exception as exc:
            logger.warning("probe_mark_not_scored_failed", error=str(exc))

    def _probe_finish(self, context):
        """Compute the verdicts and publish them for this request's response.

        Called explicitly before the response is written — NOT from a `finally`, because by the
        time a `finally` runs on the streaming path the terminal chunk has already been yielded
        and there is nothing left to attach a verdict to.
        """
        if context is None:
            return []
        try:
            verdicts = context.finish()
            set_probe_verdicts(verdicts)
            return verdicts
        except Exception as exc:
            logger.warning("probe_finish_failed", error=str(exc))
            return []

    async def _probe_stream_chunk(self, verdicts, completion_id, created, model_name):
        """Yield the terminal probe chunk, between the final chunk and `[DONE]` (FR-24.7).

        Shaped like OpenAI's usage chunk: `choices: []` plus one extension field. Spike 0.2
        measured both clients tolerating it — the OpenAI SDK consumes it cleanly and exposes the
        extension through `model_extra`, and Open WebUI guards empty choices explicitly at
        `utils/middleware.py:4966` — so it is emitted unconditionally and the proposed
        `X-miLLM-Probe-Stream: 1` opt-in was dropped.

        Yields nothing when nothing is armed, so an unarmed server's stream is byte-identical to
        what it was before this feature existed.
        """
        if not verdicts:
            return
        try:
            import json as _probe_json

            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [],
                "millm_probe_verdicts": [_verdict_payload(v) for v in verdicts],
            }
            yield f"data: {_probe_json.dumps(payload)}\n\n"
        except Exception as exc:
            # A monitor must not truncate a stream. The header and the event still carry it.
            logger.warning("probe_stream_chunk_failed", error=str(exc))

    async def _probe_record(
        self, context, verdicts=None, full_ids=None, *, detached: bool = False
    ) -> None:
        """Persist one event per armed probe, and emit them. Never raises.

        `full_ids` is the request's token ids, used to build each verdict's decoded context
        window. Every caller passes it; see `probe_context` for why it was missing.

        `detached=True` for a context from `_probe_begin_detached`: it was never registered, so
        the runtime's slot is not this request's to close — `end_request()` would close whatever
        context ANOTHER request has open (FR-27.8h).
        """
        if context is None:
            return
        try:
            from millm.services.probe_runtime import ProbeRuntimeState

            if not detached:
                ProbeRuntimeState().end_request()
            if verdicts is None:
                verdicts = context.finish()

            import millm.api.dependencies as deps

            # ⚠ THE SERVICE MUST NOT DEPEND ON A ROUTE HAVING RUN FIRST. This read used to be
            # `service = getattr(deps, "_probe_event_service", None)` followed by
            # `if service is None: return` — and that global is populated by exactly ONE thing in
            # the product, the `ProbeEventServiceDep` on `GET /api/probes/status`. So after every
            # restart, a probe could be armed and serving verdicts in the response header while
            # every event was dropped, with no log line and no counter, until something happened
            # to poll status. A deploy restarts the pod, and a caller using the OpenAI API with no
            # admin UI open never polls it at all — so the events were lost indefinitely.
            # Found on hardware 2026-09-30: verdicts in the header, `events_recorded: 0`, and the
            # identical request recorded once `/status` had been called in between.
            #
            # ⚠ AND THE SESSION IS OURS. Rebinding is not optional even when the singleton exists:
            # its repositories belong to whichever REQUEST last resolved the dependency, and that
            # request has finished. It happened to still work, which is luck, not a guarantee.
            # The INSTANCE is still reused, because the socket throttle and the dropped-event
            # counter live on it — a fresh service per request is the same as having no throttle.
            from millm.db.base import async_session_factory
            from millm.db.repositories.probe_repository import (
                ProbeEventRepository,
                ProbeRepository,
            )
            from millm.services.probe_event_service import ProbeEventService
            # ⚠ THE OVERHEAD IS PASSED. `note_request_overhead` existed, was unit-tested by
            # direct call, and had NO production caller — so `GET /api/probes/status` reported
            # `last_request_overhead_ms: null` on every request ever served, the
            # above-threshold warning could never fire, and SC-4 was unmeasurable from the
            # product. Found on hardware, by trying to measure it.
            #
            # ⚠ AND SO IS `contexts`, WHICH HAD THE SAME DEFECT AS THE ARGUMENT BESIDE IT.
            # `record()` has always accepted it, the column, the detail route, the modal and
            # `PROBE_EVENT_CONTEXT_TOKENS` all existed — and nothing passed it, so
            # `context_text` was NULL on every event ever recorded and the UI's prompt window
            # opened empty. Reported 2026-09-28. The comment above records the identical
            # omission being fixed for `overhead_ms` in the same review round.
            from millm.core.config import settings as _settings
            from millm.services.probe_context import contexts_for

            contexts = contexts_for(
                verdicts,
                full_ids,
                _settings.PROBE_EVENT_CONTEXT_TOKENS,
                self._tokenizer if self.is_model_loaded() else None,
                # The window boundary, so each context shows only the tokens its window read.
                prompt_length=getattr(context, "prompt_length", None),
                last_user_span=getattr(context, "last_user_span", None),
            )
            async with async_session_factory() as session:
                service = getattr(deps, "_probe_event_service", None)
                if service is None:
                    service = ProbeEventService(
                        ProbeRepository(session), ProbeEventRepository(session)
                    )
                    deps._probe_event_service = service
                else:
                    service.repository = ProbeRepository(session)
                    service.events = ProbeEventRepository(session)
                await service.record(
                    context.request_id,
                    verdicts,
                    overhead_ms=context.overhead_ms,
                    # The denominator. Without it the service has a total and no rate, and the
                    # budget goes back to judging long answers.
                    n_passes=context.n_passes,
                    contexts=contexts or None,
                    # Feature 26 (T-70): a batch row's events are marked, capped apart from live
                    # traffic, deduplicated per (probe, line, window), and never emitted live.
                    batch_row=BATCH_ROW.get(),
                )
        except Exception as exc:
            logger.warning("probe_record_failed", error=str(exc))

    def _sensing_begin(self, request_id: str):
        """Open a sensing request boundary (serial paths only). Returns
        (sae, profile_id) for the armed SAE, or None when the request will
        not be sensed.

        The profile_id is SNAPSHOTTED here (011 R1): a re-arm to a different
        cluster while this request generates must not let the flush persist
        these hits under the new profile.

        Speculative decoding is excluded: verification passes advance the
        offset by the whole candidate block and rejected tokens re-run, so
        absolute positions diverge from real token indices — such requests
        go unsensed rather than mis-attributed (documented v1 limitation).
        """
        try:
            from millm.services.sae_service import AttachedSAEState

            sae = AttachedSAEState().attached_sae
            if sae is None or not sae.is_sensing_armed:
                return None
            if self._speculative_model_id:
                logger.info(
                    "sensing_skipped",
                    reason="speculative_decoding_active",
                    request_id=request_id,
                )
                return None
            sae.begin_sensing_request(request_id)
            # Snapshot profile AND config at begin (011 R1 + enh R1): a
            # mid-request re-arm must not lend the flush the NEW cluster's
            # context window size or member count.
            config = sae._sensing
            profile_id = config.profile_id if config else None
            return SensingRequestContext(sae=sae, profile_id=profile_id,
                                         config=config)
        except Exception:
            logger.warning("sensing_begin_failed", exc_info=False)
        return None

    def _circuit_sensing_layer_saes(self) -> dict:
        """layer -> LoadedSAE for the layers a circuit could be armed on.

        Resolved ONCE per call. ``by_layer`` returns None when a layer is
        ambiguous (zero or more than one SAE attached) so a caller can never
        silently pick the wrong basis; re-resolving per edge inside a loop is
        the TOCTOU wrong-basis risk set_circuit_steering warns about.
        """
        from millm.services.sae_service import AttachedSAEState

        state = AttachedSAEState()
        out: dict = {}
        for entry in state.entries():
            resolved = state.by_layer(entry.layer)
            if resolved is not None:
                out[entry.layer] = resolved.sae
        return out

    def _circuit_sensing_begin(self, request_id: str):
        """Open an edge-sensing boundary across the circuit's SAEs.

        Returns the layer->SAE map used, or None when not sensing. Excludes
        speculative decoding for the same reason Feature 11 does: verification
        passes advance the offset by a whole candidate block and rejected
        tokens re-run, so the absolute positions the ring matches on diverge.
        """
        try:
            import millm.api.dependencies as deps

            service = deps._circuit_sensing_service
            if service is None or not service.is_armed:
                return None
            # R1-06: every skip must reach the operator. These paths returned
            # None silently (one of them merely logged), so a deployment with
            # `speculative_model` set senses NOTHING, FOREVER, while status
            # reports `armed: true, paused_reason: null, events_recorded: 0` —
            # indistinguishable from quiet traffic. That is the "armed but
            # silently dark" mode F15 R1-01 existed to kill, surviving on the
            # skip path because the skip lives here and the status lives there.
            if self._speculative_model_id:
                logger.info(
                    "circuit_sensing_skipped",
                    reason="speculative_decoding_active",
                    request_id=request_id,
                )
                service.note_paused("speculative_decoding")
                return None
            layer_saes = self._circuit_sensing_layer_saes()
            if not layer_saes:
                logger.info(
                    "circuit_sensing_skipped",
                    reason="no_layer_saes",
                    request_id=request_id,
                )
                service.note_paused("no_attached_saes")
                return None
            if not service.begin_request(request_id, layer_saes):
                # begin_request records its own, more specific reason
                # (concurrent_request / layer_unavailable) — do not overwrite it.
                return None
            # Observing normally: clear any stale reason from a PREVIOUS
            # request, or the operator keeps seeing why sensing was paused
            # after it has resumed.
            #
            # R2-02: this cleared unconditionally, which wiped the reason
            # `begin_request` had just set for THIS request. `begin_request`
            # returns True when SOME layers began, so a partially dark circuit
            # succeeded here and its `layer_unavailable` reason was erased —
            # R1-06's fix (say why sensing is degraded) deleted by R1-02's
            # (say which layers are dark). Verified: reason went to None while
            # layer 13 was dark.
            service.clear_stale_pause()
            return layer_saes
        except Exception:
            logger.warning("circuit_sensing_begin_failed", exc_info=False)
        return None

    async def _notify_circuit_sensing(self, layer_saes, full_ids) -> None:
        """Drain and persist the request's observed edges, then CLOSE the
        boundary. Never raises.

        F17 task 4.2: closing lives here because this is the one place every
        generation path already reaches in a `finally`. Before this, the only
        `close_request()` in the codebase was inside the hung-thread handler —
        so the two normal completion paths drained their edges and left the
        context (and its rings) alive past the end of the request. Verified by
        grep: three `_circuit_sensing_begin` call sites, one `close_request`.

        The close is in a `finally` because this method has two early returns
        (no service, nothing sensed) and the quiet path — a request that
        observed nothing — is the common one. Closing only when edges were
        found would leak the context on exactly the requests that look fine.
        """
        if not layer_saes:
            return
        service = None
        try:
            import millm.api.dependencies as deps

            service = deps._circuit_sensing_service
            if service is None:
                return
            request_id, edges, truncated = service.collect_edges(layer_saes)
            if not request_id or not edges:
                return
            await service.record(
                request_id,
                edges,
                truncated,
                full_ids,
                self._tokenizer if self.is_model_loaded() else None,
            )
        except Exception:
            logger.exception("circuit_sensing_flush_failed")
        finally:
            if service is not None:
                try:
                    service.close_request()
                except Exception:
                    logger.exception("circuit_sensing_close_failed")

    def _sensing_mark_history(self, sensing_ctx, prompt_ids) -> None:
        """Set the history-dedup boundary for this request (goal item 2):
        positions inside the longest common prefix with the previous sensed
        request were already reported when they first occurred. Called right
        after tokenization; never raises."""
        if sensing_ctx is None or prompt_ids is None:
            return
        try:
            import millm.api.dependencies as deps

            service = deps._sensing_service
            if service is None:
                return
            ids = prompt_ids[0] if prompt_ids.dim() == 2 else prompt_ids
            boundary = service.history_boundary([int(i) for i in ids.tolist()])
            if boundary > 0:
                sensing_ctx.sae.set_sensing_report_from(boundary)
        except Exception:
            logger.warning("sensing_history_boundary_failed", exc_info=False)

    async def _notify_sensing(self, sensing_ctx, full_ids) -> None:
        """Collect + record this request's sensing hits (post-generation,
        off the hot path). Sibling of _notify_monitoring; never raises.

        sensing_ctx is the (sae, profile_id) pair from _sensing_begin — the
        profile id was snapshotted at begin time so a mid-request re-arm
        cannot mis-attribute the flush (011 R1)."""
        if sensing_ctx is None:
            return
        sensing_sae = sensing_ctx.sae
        profile_id = sensing_ctx.profile_id
        config_snapshot = sensing_ctx.config
        try:
            import millm.api.dependencies as deps

            request_id, hits, truncated = sensing_sae.collect_sensing_hits()
            service = deps._sensing_service
            if service is None:
                return
            service.note_request_overhead(sensing_sae._sensing_overhead_ms)
            if not request_id:
                # Empty id = the boundary was destroyed mid-request (a
                # same-profile re-arm reset the buffer). The dropped hits
                # were never reported — writing this sequence into history
                # would suppress them FOREVER (enh R2 #2).
                return
            # History advances on EVERY sensed request — the next request's
            # dedup boundary needs this sequence even when nothing fired.
            # Capped requests stop history at the last REPORTED position
            # (capped-away moments were never reported and must re-read
            # next turn); the profile guard skips post-disarm races.
            reported_through = None
            if truncated:
                reported_through = (hits[-1].pos_end + 1) if hits else 0
            service.note_request_ids(full_ids, profile_id=profile_id,
                                     reported_through=reported_through)
            if not hits:
                return
            ambient = self._ambient_counts(sensing_sae, hits)
            await service.record(
                request_id,
                hits,
                truncated,
                full_ids,
                self._tokenizer if self.is_model_loaded() else None,
                ambient_counts=ambient,
                profile_id=profile_id,
                config_snapshot=config_snapshot,
            )
        except Exception:
            logger.exception("sensing_flush_failed")

    @staticmethod
    def _ambient_counts(sae, hits) -> Optional[dict[int, int]]:
        """Best-effort alone-vs-within signal (FTID pitfall 4): full-SAE
        fired count, ONLY when un-compacted monitoring co-ran and only for
        spans that include the last captured position (monitoring keeps the
        last pass only). Anything else stays None — never estimated."""
        try:
            if (not sae.is_monitoring_enabled
                    or sae._monitored_features is not None):
                return None
            acts = sae.get_feature_activations_for_item(0)
            if acts is None:
                return None
            last_abs = sae._sensing_token_offset - 1
            counts: dict[int, int] = {}
            for i, hit in enumerate(hits):
                if hit.pos_end == last_abs:
                    counts[i] = int((acts[-1] > 0).sum().item())
            return counts or None
        except Exception:
            return None

    def _notify_monitoring(self, request_id: Optional[str] = None) -> None:
        """
        Forward captured activations to the monitoring service.

        Reads per-batch-item activations from the attached SAE and routes them
        to the MonitoringService.

        Serial path (batch_size == 1): the single item's activations are tagged
        with the request_id for accurate per-request attribution.

        CBM path (batch_size > 1): each batch item is emitted as a separate
        event tagged "<request_id>:batch_<idx>" since the mapping from batch
        position to request ID is not available from inside the hook.
        Set CBM_FORCE_SERIAL_MONITORING=true to avoid this and get accurate
        per-request data at the cost of disabling batching for monitored requests.
        """
        try:
            from millm.services.sae_service import AttachedSAEState
            import millm.api.dependencies as deps

            sae_state = AttachedSAEState()
            sae = sae_state.attached_sae
            if sae is None or not sae.is_monitoring_enabled:
                return

            batch_size = sae.get_last_batch_size()
            if batch_size == 0:
                logger.warning(
                    "monitoring_no_activations",
                    sae_id=sae_state.attached_sae_id,
                    monitoring_enabled=sae.is_monitoring_enabled,
                )
                return

            monitoring_service = deps._monitoring_service
            if monitoring_service is None:
                # MonitoringService is a singleton initialized by get_monitoring_service()
                # when the monitoring API is first used.  If it hasn't been initialized
                # yet, the SAE activations were captured but there is no recipient to
                # forward them to — just skip rather than trying to construct a
                # MonitoringService here (which would create a DB session inside a
                # synchronous post-generation path and use a broken SAEService with
                # no cache_dir).  Activations will be forwarded once monitoring is
                # configured via the /api/monitoring endpoint.
                return

            if batch_size == 1:
                # Serial path: single request — accurate attribution
                activations = sae.get_feature_activations_for_item(0)
                if activations is not None:
                    monitoring_service.on_activation(activations, request_id=request_id)
            else:
                # CBM batch: emit each item with a position-tagged request_id.
                # Batch position ≠ request_id; set CBM_FORCE_SERIAL_MONITORING=true
                # for accurate per-request attribution.
                logger.debug(
                    "monitoring_cbm_batch_attribution_approximate",
                    batch_size=batch_size,
                    request_id=request_id,
                )
                for item_idx in range(batch_size):
                    activations = sae.get_feature_activations_for_item(item_idx)
                    if activations is not None:
                        item_request_id = (
                            f"{request_id}:batch_{item_idx}"
                            if request_id
                            else f"batch_{item_idx}"
                        )
                        monitoring_service.on_activation(
                            activations, request_id=item_request_id
                        )
        except Exception as e:
            # Never let monitoring errors affect inference
            logger.warning("monitoring_notification_failed", error=str(e))

    def _check_context_length(self, prompt_tokens: int, max_new_tokens: int) -> None:
        """
        Validate that prompt + generation fits within model context.

        Raised as ContextLengthExceededError (400 context_length_exceeded on
        /v1). It was a bare ValueError from 2026-02-07, which reached a
        non-streaming client as a 500 — an invitation to retry a request that
        can never succeed — and cut a stream off after its 200 with no error
        event and no [DONE]. Hardware acceptance, 2026-09-14 (item 7).

        Raises:
            ContextLengthExceededError: If context length would be exceeded.
        """
        max_length = _served_max_context(getattr(self._model, "config", None))
        if max_length is None:
            return  # Can't validate without config

        total = prompt_tokens + max_new_tokens
        if total > max_length:
            asked = (
                f"{prompt_tokens} in the prompt, {max_new_tokens} for the completion"
                if max_new_tokens
                else f"{prompt_tokens} in the input"
            )
            raise ContextLengthExceededError(
                f"This model's maximum context length is {max_length} tokens. However, "
                f"you requested {total} tokens ({asked}). Shorten the prompt or ask for "
                "fewer max_tokens.",
                details={
                    "max_context_tokens": max_length,
                    "requested_tokens": total,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": max_new_tokens,
                },
            )

    def check_stream_admission(self, request: ChatCompletionRequest) -> None:
        """Refuse, before a stream's 200 is committed, a chat request its generator would refuse.

        A streaming response commits its status and headers before the
        generator runs, so a refusal found there can only be an error event.
        The route calls this first, so a prompt past the model's context is a
        400 with the error envelope, as it is without streaming. The generator
        checks again (the model can change in between) and answers with an
        error event and [DONE]. Hardware acceptance, 2026-09-14 (item 7): the
        stream was cut off after its 200 with neither.

        llama.cpp is skipped: it measures its own window and its refusal is
        translated where it is raised.

        A model being unloaded is refused here too (503 model_busy), before its
        tokenizer is touched: an unload deletes it (hardware acceptance, item 11).

        Raises:
            ModelBusyError: the model is being unloaded.
            ContextLengthExceededError: prompt tokens plus max_tokens exceed the
                model's context.
        """
        self.refuse_if_unloading()
        if not self._model_state.is_loaded or self._engine_is_llamacpp():
            return
        prompt = self._format_chat_messages(request.messages, request.chat_template_kwargs)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        self._check_context_length(
            int(inputs["input_ids"].shape[1]),
            GenerationConfig.from_request(request).max_new_tokens,
        )

    def _determine_finish_reason(
        self,
        generated_token_count: int,
        max_new_tokens: int,
        last_token_id: Optional[int] = None,
    ) -> str:
        """
        Determine finish_reason per OpenAI spec.

        Returns "length" if generation hit max_tokens, "stop" otherwise.

        The OpenAI spec uses "stop" for both model-initiated stops (EOS token)
        and user-supplied stop sequences.  We log the internal stop mechanism at
        DEBUG level so operators can distinguish the two without changing the
        API-visible value.

        Args:
            generated_token_count: Number of tokens generated.
            max_new_tokens: The max_tokens limit for this request.
            last_token_id: Optional last token ID for EOS detection (non-streaming
                path only — TextIteratorStreamer does not expose individual IDs).
        """
        if generated_token_count >= max_new_tokens:
            logger.debug("finish_reason_length", count=generated_token_count)
            return "length"

        if last_token_id is not None:
            try:
                eos_id = getattr(self._tokenizer, "eos_token_id", None)
                if eos_id is not None and last_token_id == eos_id:
                    logger.debug("finish_reason_eos_token")
                else:
                    logger.debug(
                        "finish_reason_stop_other",
                        last_token_id=last_token_id,
                    )
            except Exception:
                pass  # Tokenizer not available during testing

        return "stop"

    def _apply_stop_sequences(
        self, text: str, stop_sequences: Optional[list[str]]
    ) -> tuple[str, bool]:
        """
        Truncate text at the first occurrence of any stop sequence.

        Returns:
            Tuple of (truncated_text, was_stopped).
        """
        if not stop_sequences:
            return text, False

        earliest_pos = len(text)
        found = False
        for seq in stop_sequences:
            pos = text.find(seq)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                found = True

        if found:
            return text[:earliest_pos], True
        return text, False

    # =========================================================================
    # Chat Completions
    # =========================================================================

    async def _generate_batch_chunk(
        self,
        prompts: list[str],
        gen_config: Any,
        completion_id: str,
        chunk_start: int,
    ) -> list[dict]:
        """Generate one chunk as a single batched forward pass.

        Returns one dict per prompt, in input order.
        """
        # LEFT PADDING IS LOAD-BEARING AND ITS FAILURE IS SILENT.
        #
        # transformers defaults to RIGHT padding. For a decoder-only model that
        # puts the pad tokens BETWEEN the prompt and the first generated token,
        # so every row shorter than the longest continues from padding and
        # produces fluent garbage — while the longest row, being unpadded, looks
        # perfect. Nothing raises. Left padding keeps every row's prompt flush
        # against the generation boundary.
        #
        # Passed per-call, never by assigning self._tokenizer.padding_side: that
        # object is shared with the streaming path, embeddings, chat formatting
        # and stop_strings, and a global mutation here would reach all of them.
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(self._get_input_device())

        padded_width = inputs["input_ids"].shape[1]
        # One check on the padded width — that is the width the model actually
        # runs — rather than per prompt.
        self._check_context_length(padded_width, gen_config.max_new_tokens)

        generate_kwargs = self._build_generate_kwargs(gen_config, inputs)

        # transformers raises "assisted generate is only supported for
        # batch_size = 1". Drop the draft model for this pass rather than fail;
        # the batch speedup is far larger than the speculative one anyway.
        if generate_kwargs.pop("assistant_model", None) is not None:
            logger.info(
                "batch_speculative_disabled", batch_size=len(prompts),
                request_id=completion_id,
            )

        outputs = await asyncio.to_thread(
            self._generate_sync, generate_kwargs, **_seed_kwargs(gen_config)
        )

        self._notify_monitoring(request_id=f"{completion_id}:batch_{chunk_start}")

        attention_mask = inputs.get("attention_mask")
        pad_id = generate_kwargs.get("pad_token_id")
        eos_ids = generate_kwargs.get("eos_token_id")
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_set = set(eos_ids or [])

        results: list[dict] = []
        for row_idx in range(len(prompts)):
            # True prompt length is the unpadded count, not the padded width —
            # billing the pad volume would over-report usage on every short row.
            if attention_mask is not None:
                prompt_tokens = int(attention_mask[row_idx].sum())
            else:
                prompt_tokens = padded_width

            generated_ids = self._slice_generated(outputs[row_idx], padded_width)

            # generate() runs until EVERY row finishes, filling rows that
            # stopped early with pad tokens. Without trimming here, a row that
            # stopped at 20 tokens reports the batch's length and inherits the
            # batch's finish_reason — so one long row would make every row in
            # the batch claim "length".
            trimmed = generated_ids
            for pos in range(generated_ids.shape[0]):
                tok = int(generated_ids[pos])
                if tok in eos_set or (pad_id is not None and tok == pad_id):
                    trimmed = generated_ids[:pos + 1]
                    break

            completion_text = self._tokenizer.decode(
                trimmed, skip_special_tokens=True
            )
            completion_tokens = int(trimmed.shape[0])

            completion_text, stopped_by_sequence = self._apply_stop_sequences(
                completion_text, gen_config.stop_sequences
            )

            constraint = getattr(gen_config, "constraint", None)
            if isinstance(constraint, CompiledConstraint):
                finish_reason = self._finish_constrained(constraint, trimmed, completion_text)
                self._log_constrained(constraint, completion_tokens, finish_reason)
            elif stopped_by_sequence:
                finish_reason = "stop"
            else:
                last_token_id = (
                    int(trimmed[-1]) if completion_tokens > 0 else None
                )
                finish_reason = self._determine_finish_reason(
                    completion_tokens,
                    gen_config.max_new_tokens,
                    last_token_id=last_token_id,
                )

            results.append({
                "text": completion_text,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })

        return results

    # Batch 8 is the shipped default: 5.59x throughput with 4.8 GB of headroom
    # on a 24 GB card serving gemma-4-12B-it at Q8. Batch 12 reaches 7.31x but
    # leaves 1.7 GB, and 16 OOMed outright — too close to the edge for an
    # unattended run. This ceiling is a safety net under that default, not a
    # substitute for it.
    MAX_BATCH_ROWS = 8

    def _project_kv_bytes(self, rows: int, total_len: int) -> Optional[int]:
        """Bytes of KV cache a `rows x total_len` batch would need, or None.

        None means the projection could not be made (an unknown config shape),
        and the caller must then fall back to the row cap rather than to an
        unbounded batch — an unmeasurable batch is not a safe batch.
        """
        try:
            cfg = self._model.config
            layers = getattr(cfg, "num_hidden_layers", None)
            hidden = getattr(cfg, "hidden_size", None)
            heads = getattr(cfg, "num_attention_heads", None)
            kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
            if not all(isinstance(v, int) and v > 0
                       for v in (layers, hidden, heads, kv_heads)):
                return None
            head_dim = getattr(cfg, "head_dim", None) or (hidden // heads)
            # key + value, 2 bytes per element at fp16/bf16 KV.
            return 2 * 2 * rows * total_len * layers * kv_heads * head_dim
        except Exception:
            return None

    def _loaded_gpu_indices(self) -> list[int]:
        """The cards the loaded model holds memory on; empty when none."""
        if not self._model_state.is_loaded:
            return []
        return list(getattr(self._model_state.current, "gpu_indices", None) or [])

    @staticmethod
    def _kv_fits(
        need_mb: int, gpu_indices: list[int], shares: Optional[dict[int, float]] = None
    ) -> tuple[bool, int]:
        """Whether a KV cache of `need_mb` fits on the cards the model lives on.

        This checked GPU 0 whatever card the model was on, so a batch for a
        model on the 3090 was sized by the 3080 Ti's free memory.

        The cache is allocated beside each attention layer, so each card holds
        the share of it that it holds of the layers (`shares`, from the model's
        device map — `gpu_placement.layer_share_by_index`). A split planned
        largest-free-first is deliberately uneven: an even 1/N share asked the
        3080 Ti for as much cache as the 3090 while it held a fraction of the
        layers. A card with no layers is not asked. Without shares, an even
        split.

        Returns:
            (fits, least free MB among the cards asked)
        """
        from millm.ml.memory_utils import verify_memory_available

        if shares and any(shares.get(index, 0) > 0 for index in gpu_indices):
            asked = {
                index: math.ceil(int(need_mb) * shares[index])
                for index in gpu_indices
                if shares.get(index, 0) > 0
            }
        else:
            even_mb = -(-int(need_mb) // len(gpu_indices))  # ceiling division
            asked = {index: even_mb for index in gpu_indices}
        fits, least_free = True, None
        for index, share_mb in asked.items():
            ok, free_mb = verify_memory_available(share_mb, device=index)
            fits = fits and ok
            least_free = free_mb if least_free is None else min(least_free, free_mb)
        return fits, int(least_free or 0)

    def _chunk_batch_for_memory(
        self, prompts: list[str], max_new_tokens: int
    ) -> list[tuple[int, list[str]]]:
        """Split a batch into chunks that fit, yielding (start_index, chunk).

        Chunking rather than refusing: the caller asked for N conversations and
        gets N back either way, so the API contract does not depend on how much
        VRAM happened to be free. A slow answer beats a 500.
        """
        rows = min(len(prompts), self.MAX_BATCH_ROWS)

        try:
            gpu_indices = self._loaded_gpu_indices()
            if gpu_indices and prompts:
                from millm.ml.gpu_placement import layer_share_by_index

                shares = layer_share_by_index(self._model, gpu_indices)
                longest = max(len(self._tokenizer.encode(p)) for p in prompts)
                total_len = longest + max(int(max_new_tokens or 0), 0)
                while rows > 1:
                    projected = self._project_kv_bytes(rows, total_len)
                    if projected is None:
                        break  # unmeasurable -> keep the row cap, do not grow
                    need_mb = int(projected / (1024 * 1024) * 1.2)  # +20% slack
                    ok, available_mb = self._kv_fits(need_mb, gpu_indices, shares)
                    if ok:
                        break
                    rows -= 1
                    logger.info(
                        "batch_chunk_reduced", rows=rows,
                        needed_mb=need_mb, available_mb=available_mb,
                    )
        except Exception:
            logger.warning("batch_memory_projection_failed", exc_info=True)

        rows = max(1, rows)
        return [
            (i, prompts[i:i + rows]) for i in range(0, len(prompts), rows)
        ]

    # Batch position IS conversation index in this path: row i of the batch is
    # prompts[i], which is `messages` at 0 and `extra_messages[i-1]` after. The
    # ":batch_i" monitoring tag below therefore names the conversation. (CBM
    # borrows the same suffix for a different quantity; see its docstring.)
    async def _create_batched_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate every conversation in the request in ONE forward pass.

        The weights are read once and amortised across the batch — the vLLM
        mechanism. Measured 5.59x aggregate throughput at batch 8 on
        gemma-4-12B-it. Running the same N as independent concurrent requests
        does NOT do this: each re-reads the full weights, which is why this is
        a batch and why MAX_CONCURRENT_REQUESTS stays at 1.

        The whole batch is ONE request holding ONE queue slot, so the steering
        isolation that the concurrency limit provides is untouched. Steering
        applies uniformly to every row (the delta is expanded over the batch
        dimension in sae_wrapper), which is correct for a single request.

        NOT BIT-REPRODUCIBLE AGAINST SERIAL, and this is inherent rather than a
        defect to be fixed. Measured on gemma-4-12B-it at int8 (2026-08-30):
        a prompt that is the LONGEST in its batch — and therefore receives no
        padding at all — still produces different greedy text at batch 1, 2 and
        4. Each shape is individually deterministic (repeat a shape, get the
        same bytes), so the cause is the batched GEMM's reduction order under
        bitsandbytes dequantisation: tiny FP differences flip a near-tie argmax
        and greedy decoding diverges from that token on.

        Quality is unaffected — over 8 realistic labeling prompts, 5/8 labels
        were identical and the other 3 differed only in wording between equally
        good answers ("physical floor covering" vs "household floor covering"),
        with zero parse failures on either path.

        The consequence that matters: BATCH COMPOSITION IS AN INPUT. For bulk
        labeling that is harmless. For a labeling TRIAL, where the template is
        supposed to be the only variable, it is not — vary the batching and the
        template stops being the only thing that changed. Trials must hold the
        batch size and the panel order fixed, or run serially.
        """
        conversations = [request.messages] + list(request.extra_messages or [])
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # CBM has no batched path. Falling through to it would silently drop
        # every conversation after the first, so serve serially instead: slower
        # than a batch, identical in result.
        if self._use_cbm_for_request(**self._cbm_route_kwargs(request)):
            logger.info(
                "batch_serialised", reason="cbm_active",
                batch_size=len(conversations), request_id=completion_id,
            )
            return await self._serial_chat_fallback(
                request, conversations, completion_id, created
            )

        prompts = [
            self._format_chat_messages(c, request.chat_template_kwargs)
            for c in conversations
        ]
        gen_config = GenerationConfig.from_request(request)
        constraint = await self._compile_constraint(request)
        if constraint is not None:
            # One matcher per row, built per chunk by _build_generate_kwargs (FR-25.10.10).
            gen_config = dataclasses.replace(gen_config, constraint=constraint)

        choices: list[ChatCompletionChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with self._admit():
            # Probes (FR-27.8a–c). A batch has no single row to score, so the request says so —
            # `not_scored: batched_request`, as the `n > 1` path does — instead of reaching
            # generation with nothing open, which left the header, the chunk and the event all
            # absent while a probe was armed. Detached: a path that never scores does not use the
            # runtime's shared slot (FR-27.8h). Recorded in the `finally`, so a failed generation
            # still leaves an event (FR-27.8c).
            _probe_ctx = self._probe_begin_detached(completion_id, "batched_request")
            _probe_verdicts = None
            try:
                _saved_steering = None
                if request.profile or request.steering_intensity is not None:
                    _saved_steering = await self._apply_request_steering(
                        request.profile, request.steering_intensity,
                        request_id=completion_id,
                    )

                # Sensing is refused for a batch: hit positions are absolute within
                # a row, and there is no way to attribute them back to a
                # conversation once the rows are padded to a common width. It goes
                # UNSENSED rather than mis-attributed — and it says so, because a
                # sensing path that goes quietly dark while /api/sensing/status
                # still reports armed is the failure this project has shipped
                # before.
                try:
                    from millm.services.sae_service import AttachedSAEState as _S

                    _armed = _S().attached_sae
                    if _armed is not None and _armed.is_sensing_armed:
                        logger.info(
                            "sensing_skipped", reason="batched_request",
                            batch_size=len(prompts), request_id=completion_id,
                        )
                except Exception:  # pragma: no cover - never fail a request on this
                    logger.warning("sensing_skip_log_failed", exc_info=True)

                try:
                    for chunk_start, chunk in self._chunk_batch_for_memory(
                        prompts, gen_config.max_new_tokens
                    ):
                        rows = await self._generate_batch_chunk(
                            chunk, gen_config, completion_id, chunk_start
                        )
                        for offset, row in enumerate(rows):
                            choices.append(
                                ChatCompletionChoice(
                                    index=chunk_start + offset,
                                    message=(
                                        ChatMessage(role="assistant", content=row["text"])
                                        if constraint is not None
                                        else self._assistant_message(row["text"], chunk[offset])
                                    ),
                                    finish_reason=row["finish_reason"],
                                )
                            )
                            total_prompt_tokens += row["prompt_tokens"]
                            total_completion_tokens += row["completion_tokens"]
                finally:
                    self._restore_request_profile(_saved_steering)
                if constraint is not None:
                    note_request_outcome(constrained=constraint.header)
                if gen_config.seed is not None:
                    # Batched rows are deterministic per batch SHAPE only (FR-25.14.1): the batched
                    # GEMM's reduction order depends on the shape, and padding on the other rows.
                    note_request_outcome(seed_scope=self._seed_scope(SEED_SCOPE_BATCH_SHAPE))
                # ⚠ BEFORE the method returns: the chat route reads the ContextVar for
                # `X-miLLM-Probe-Verdicts` straight after it does (FR-27.8b).
                _probe_verdicts = self._probe_finish(_probe_ctx)
            finally:
                await self._probe_record(
                    _probe_ctx, _probe_verdicts, full_ids=None, detached=True
                )

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    async def _serial_chat_fallback(
        self,
        request: ChatCompletionRequest,
        conversations: list,
        completion_id: str,
        created: int,
    ) -> ChatCompletionResponse:
        """One conversation at a time, assembled into one batched-shaped response.

        Used when a batch cannot be served as a batch (CBM active, or
        speculative decoding attached — transformers rejects assisted generation
        for batch_size > 1). The contract the caller sees is identical; only the
        throughput differs.
        """
        choices: list[ChatCompletionChoice] = []
        p_tokens = 0
        c_tokens = 0
        for i, conv in enumerate(conversations):
            sub = request.model_copy(
                update={"messages": conv, "extra_messages": None, "n": 1}
            )
            resp = await self.create_chat_completion(sub)
            inner = resp.choices[0] if resp.choices else None
            choices.append(
                ChatCompletionChoice(
                    index=i,
                    message=(
                        inner.message
                        if inner
                        else ChatMessage(role="assistant", content="")
                    ),
                    finish_reason=(inner.finish_reason if inner else "stop"),
                )
            )
            if resp.usage:
                p_tokens += resp.usage.prompt_tokens
                c_tokens += resp.usage.completion_tokens

        model_info = self.get_loaded_model_info()
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_info.name if model_info else "unknown",
            choices=choices,
            usage=Usage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=p_tokens + c_tokens,
            ),
        )

    async def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Create non-streaming chat completion.

        Supports n > 1 for multiple completions per request.

        Args:
            request: The chat completion request

        Returns:
            ChatCompletionResponse with generated text

        Raises:
            RuntimeError: If no model is loaded
        """
        # SCORING FIRST (FR-25.5.8), before llama.cpp, the batched path and the CBM — as
        # create_text_completion does. Otherwise an `extra_messages` scoring request would be
        # GENERATED by the batched path with its scores dropped.
        if isinstance(request, ChatCompletionRequest) and request.wants_scores():
            return await self._score_chat_completion(request)

        # llama.cpp first: everything below this point reaches through
        # self._model as a torch object — tensors moved to a device, a
        # `.config`, a HuggingFace tokenizer, GenerationConfig kwargs — and a
        # GGUF model has none of it.
        if self._engine_is_llamacpp():
            return await self._llamacpp_chat_completion(request)

        # Batched extension: every conversation in ONE forward pass. Checked
        # before the CBM delegation because that path has no batch support and
        # would silently drop all but the first conversation.
        if getattr(request, "extra_messages", None):
            return await self._create_batched_chat_completion(request)

        # Delegate to CBM if active and sampling params are compatible
        if self._use_cbm_for_request(**self._cbm_route_kwargs(request)):
            return await self._cbm_chat_completion(request)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # Structured output: compiled BEFORE the slot, off the event loop (FR-25.10).
        constraint = await self._compile_constraint(request)

        # Format messages to prompt
        prompt = self._format_chat_messages(
            request.messages, request.chat_template_kwargs
        )
        n = getattr(request, "n", 1) or 1

        choices: list[ChatCompletionChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with self._admit():
            # Per-request profile override: applied inside the semaphore so that
            # concurrent requests cannot race on the global steering state.
            # The previous state is restored in the finally block below.
            _saved_steering = None
            if request.profile or request.steering_intensity is not None:
                _saved_steering = await self._apply_request_steering(
                    request.profile, request.steering_intensity,
                    request_id=completion_id,
                )

            # Sensing boundary (Feature 11): n==1 only — with n>1 the
            # absolute-position accounting would concatenate independent
            # generations (documented v1 limitation; such requests go
            # unsensed rather than mis-attributed).
            _sensing_sae = self._sensing_begin(completion_id) if n == 1 else None
            _circuit_sensing = (self._circuit_sensing_begin(completion_id)
                                if n == 1 else None)
            # Probes open on EVERY request, including n>1 — unlike sensing, which simply skips.
            # A skipped probe is a silent probe, and the request has to carry a reason.
            _probe_ctx = self._probe_begin(completion_id)
            _probe_verdicts = None
            if _probe_ctx is not None and n > 1:
                _probe_ctx.mark_not_scored("batched_request")
            if n > 1:
                from millm.services.sae_service import AttachedSAEState as _S

                _armed_sae = _S().attached_sae
                if _armed_sae is not None and _armed_sae.is_sensing_armed:
                    logger.info(
                        "sensing_skipped", reason="n_gt_1", n=n,
                        request_id=completion_id,
                    )
            _sensing_full_ids = None
            _activations = None
            _millm = None

            try:
                # Tokenize input
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._get_input_device())
                prompt_tokens = inputs.input_ids.shape[1]
                self._probe_note_prompt_length(_probe_ctx, prompt_tokens)
                self._probe_note_last_user_span(
                    _probe_ctx, request.messages, inputs.input_ids, request.chat_template_kwargs
                )
                _sensing_full_ids = inputs.input_ids  # prefill-only fallback
                self._sensing_mark_history(_sensing_sae, inputs.input_ids)
                # Feature 27: this request's own activations (n == 1; the route refuses n > 1).
                _activations = self._activations_begin(request, prompt_tokens)

                # Build generation config
                gen_config = GenerationConfig.from_request(request)
                if constraint is not None:
                    gen_config = dataclasses.replace(gen_config, constraint=constraint)
                self._check_context_length(prompt_tokens, gen_config.max_new_tokens)

                # The seed covers the WHOLE n-loop: per-call seeding would make the n choices
                # identical. RNG state is process-global, so the fork made here (inside the slot)
                # spans every generate() of this request and nothing else's.
                with seeded_rng(gen_config.seed):
                    for i in range(n):
                        # Generate - offload to thread to avoid blocking the event loop
                        generate_kwargs = self._build_generate_kwargs(gen_config, inputs)

                        outputs = await asyncio.to_thread(
                            self._generate_sync, generate_kwargs
                        )
                        _sensing_full_ids = outputs[0]

                        # Notify monitoring after generation
                        self._notify_monitoring(request_id=completion_id)

                        # Decode output
                        generated_ids = self._slice_generated(outputs[0], prompt_tokens)
                        completion_text = self._tokenizer.decode(
                            generated_ids, skip_special_tokens=True
                        )
                        completion_tokens = len(generated_ids)

                        # Apply stop sequences
                        completion_text, stopped_by_sequence = self._apply_stop_sequences(
                            completion_text, gen_config.stop_sequences
                        )

                        # Determine finish reason.
                        # Pass last_token_id for EOS detection — available only in
                        # the non-streaming path where we have the full output IDs.
                        if constraint is not None:
                            finish_reason = self._finish_constrained(
                                constraint, generated_ids, completion_text
                            )
                            self._log_constrained(constraint, completion_tokens, finish_reason)
                        elif stopped_by_sequence:
                            logger.debug("finish_reason_stop_sequence")
                            finish_reason = "stop"
                        else:
                            last_token_id = (
                                int(generated_ids[-1]) if len(generated_ids) > 0 else None
                            )
                            finish_reason = self._determine_finish_reason(
                                completion_tokens,
                                gen_config.max_new_tokens,
                                last_token_id=last_token_id,
                            )

                        choices.append(
                            ChatCompletionChoice(
                                index=i,
                                # A constrained document is the answer, whole: never split into
                                # reasoning, even when the template opened a <think> block.
                                message=(
                                    ChatMessage(role="assistant", content=completion_text)
                                    if constraint is not None
                                    else self._assistant_message(completion_text, prompt)
                                ),
                                finish_reason=finish_reason,
                            )
                        )

                        total_prompt_tokens += prompt_tokens
                        total_completion_tokens += completion_tokens

                if gen_config.seed is not None:
                    note_request_outcome(seed_scope=self._seed_scope(SEED_SCOPE_REQUEST))
                if constraint is not None:
                    note_request_outcome(constrained=constraint.header)

                # ⚠ BEFORE the `finally`, and before the route reads the ContextVar to write
                # `X-miLLM-Probe-Verdicts`. A verdict computed in the `finally` would exist only
                # after the response had already been handed back.
                _probe_verdicts = self._probe_finish(_probe_ctx)
                _millm = self._activations_finish(_activations, _sensing_full_ids)
            finally:
                self._activations_close(_activations)
                # Restore steering to its pre-request state regardless of success/failure.
                self._restore_request_profile(_saved_steering)
                # Flush sensing hits (post-generation, inside the semaphore
                # so the boundary can't interleave with the next request)
                await self._notify_sensing(_sensing_sae, _sensing_full_ids)
                await self._notify_circuit_sensing(_circuit_sensing, _sensing_full_ids)
                # `_probe_finish` ran before the response was built (below); this only persists.
                await self._probe_record(_probe_ctx, _probe_verdicts, full_ids=_sensing_full_ids)

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
            millm=_millm,
        )

    @staticmethod
    def _translate_llamacpp_error(exc: Exception) -> Exception:
        """Turn llama.cpp's bare ValueErrors into errors that mean something.

        It signals an oversized prompt as `ValueError: Requested tokens (4703)
        exceed context window of 4096`, which propagates as an unhandled
        exception and reaches the client as a 500. That is the wrong answer in
        the way that matters: a 500 invites a retry, and an oversized prompt
        will fail identically every time.
        """
        text = str(exc)
        if "exceed context window" in text or "exceeds context window" in text:
            return ContextLengthExceededError(
                f"{text}. The model is serving a smaller context than its file "
                "declares because the larger one did not fit in VRAM — see the "
                "context_length in GET /api/health/inference. Shorten the "
                "prompt, raise GGUF_CONTEXT_LENGTH, or use a smaller "
                "quantization."
            )
        return exc

    def _llamacpp_continuation_prompt(self, messages: list[dict]) -> Optional[str]:
        """A prompt that CONTINUES a partial assistant turn, or None.

        THE BUG THIS FIXES. An OpenAI client that offers "continue" — Open WebUI
        does — resends the conversation with the truncated answer as a trailing
        assistant message. `create_chat_completion` applies the GGUF's baked-in
        template to that list, and every such template CLOSES the last turn:

            no trailing assistant : ...<|turn>model\n<|channel>thought\n<channel|>
            trailing assistant    : ...<|turn>model\nPARTIAL<turn|>\n

        Sealed with `<turn|>`, the model cannot do anything but start a new
        answer. Observed on both GGUF models here: gemma restated its whole
        explanation from the top ("It looks like your previous message had a
        technical glitch... Let's start fresh"), and the Qwen reasoning model
        re-opened and restarted its <think> block three times, because the
        template re-opens the thought channel on every fresh turn.

        The fix is the same one HuggingFace calls `continue_final_message`:
        render the prompt for everything BEFORE the partial with a generation
        prompt, then append the partial text raw. The turn stays open and the
        model resumes mid-sentence.

        Returns None whenever continuation does not apply or cannot be done
        safely — an ordinary request, an empty partial, or a model whose
        template is unreadable. The caller then takes the normal path, so a
        failure here costs the continuation feature and never the request.
        """
        if not messages or messages[-1].get("role") != "assistant":
            return None
        partial = messages[-1].get("content") or ""
        if not partial.strip():
            # An empty trailing assistant turn is how some clients ask for a
            # FRESH answer. Continuing it would append to nothing.
            return None

        try:
            model = self._model
            template = (getattr(model, "metadata", None) or {}).get(
                "tokenizer.chat_template"
            )
            if not template:
                return None

            # Token TEXT, not ids: the template interpolates the strings.
            inner = model._model
            eos = inner.token_get_text(model.token_eos())
            bos = inner.token_get_text(model.token_bos())

            # Rendered with jinja2 DIRECTLY rather than through
            # llama_cpp.llama_chat_format.Jinja2ChatFormatter. That class does
            # the same thing, but importing it ties this behaviour to
            # llama-cpp-python internals that have moved between versions, and
            # makes the whole feature untestable anywhere the wheel is not
            # installed — which is every developer machine here, though not CI.
            # A capability that can only be exercised in CI is one nobody can
            # iterate on.
            return _render_chat_template(
                template, list(messages[:-1]), bos=bos, eos=eos
            ) + partial
        except Exception as exc:  # noqa: BLE001
            # Never let this break a request. A model whose template does not
            # render is served the ordinary way — it restarts, which is the old
            # behaviour, rather than failing outright.
            logger.warning(
                "llamacpp_continuation_prompt_failed",
                error=str(exc)[:200],
                detail="serving as a fresh turn; the answer will not continue",
            )
            return None

    def _llamacpp_sync(self, messages: list[dict], params: dict) -> dict:
        """Blocking llama.cpp call, for asyncio.to_thread.

        `create_chat_completion` applies the chat template baked into the GGUF
        file. We deliberately do NOT pre-format the prompt with
        `_format_chat_messages` first: that would apply a template twice, once
        from a HuggingFace tokenizer this model does not have and once inside
        llama.cpp, producing doubled control tokens.
        """
        prompt = self._llamacpp_continuation_prompt(messages)
        if prompt is not None:
            # RAW completion, deliberately. create_chat_completion would
            # re-template and re-seal the turn we are trying to keep open.
            raw = self._model.create_completion(prompt=prompt, **params)
            return _completion_as_chat(raw)
        return self._model.create_chat_completion(messages=messages, **params)

    def _refuse_unsupported_llamacpp_request(
        self, request: ChatCompletionRequest
    ) -> None:
        """Refuse the chat features this engine cannot honour.

        SHARED BY BOTH the streaming and non-streaming paths, and that sharing is
        the point. These guards lived inside `_llamacpp_chat_completion` alone,
        so adding a streaming generator beside it bypassed every one of them —
        and each bypass is SILENT. A steered streaming request would have served
        unsteered output rather than refusing, which is the exact failure this
        codebase has recorded twice.

        Raising rather than degrading: a quietly unsteered answer, or a response
        carrying one choice where three were asked for, looks like success.
        """
        if getattr(request, "extra_messages", None):
            raise EngineUnsupportedError(
                "Batched conversations are not supported on the llama.cpp engine."
            )
        if request.profile or request.steering_intensity is not None:
            raise EngineUnsupportedError(
                "Steering requires forward hooks on a PyTorch module tree, which "
                "the llama.cpp engine does not have. Load a transformers-served "
                "model to steer."
            )
        # chat_template_kwargs is IGNORED here, not refused, and the difference
        # matters more than the principle it bends.
        #
        # miStudio's labeling service sends {"enable_thinking": False} on EVERY
        # request (openai_labeling_service.py:270), on the documented premise
        # that "a template that does not reference the variable ignores it".
        # That premise held for every transformers-served model and is false for
        # this engine, so refusing turned "labeling with a GGUF judge" into a
        # 400 on every call — not a degraded result, a total failure.
        #
        # Ignoring is defensible here in a way it would not be for steering,
        # because nothing downstream is silently wrong: the caller asked to
        # suppress a reasoning block, we cannot, and the reasoning arrives in
        # the completion where miStudio's own _strip_think already handles the
        # exact shape a template-opened GGUF produces (its case (c),
        # openai_labeling_service.py:1287 — reasoning with a CLOSING tag only).
        # Steering ignored would serve unsteered text that looks steered; this
        # is visible in the output and already handled.
        #
        # Logged, never silent, so an unhonoured instruction is discoverable.
        if getattr(request, "chat_template_kwargs", None):
            logger.info(
                "chat_template_kwargs_ignored",
                engine="llamacpp",
                keys=sorted(request.chat_template_kwargs),
                detail=(
                    "llama.cpp applies the template baked into the GGUF file "
                    "and exposes no way to pass variables into it"
                ),
            )
        # Feature 25 defence in depth: the request policy refuses these from the model ROW before
        # any load; a direct caller (Feature 26) or a row/engine mismatch must not reach here and
        # be served with the field dropped.
        if _constraint_type(getattr(request, "response_format", None)) is not None:
            raise ResponseFormatUnsupportedError(
                "Structured output on a GGUF model is refused in this release.",
                details={"param": "response_format"},
            )
        if (getattr(request, "n", 1) or 1) > 1:
            # The transformers path documents and honours n > 1. This one builds
            # exactly one choice, so an unguarded n=3 returned a single-choice
            # response with no error — the quiet degradation every other branch
            # here refuses. A client demultiplexing on `index` silently loses
            # two thirds of what it asked for.
            raise EngineUnsupportedError(
                "n > 1 is not supported on the llama.cpp engine in this "
                "release: it returns one completion per request. Send n "
                "separate requests, or use a transformers-served model."
            )

    def _llamacpp_params(self, gen_config: Any, request: ChatCompletionRequest) -> dict:
        """Map a request onto create_chat_completion's keyword arguments.

        Shared so the two paths cannot sample differently for the same request.
        """
        params: dict[str, Any] = {
            "max_tokens": gen_config.max_new_tokens,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            # The transformers path HONOURS these (see _build_generate_kwargs),
            # so dropping them here would make the same request sample
            # differently depending on which engine happens to hold the model —
            # a quiet degradation of exactly the kind this path refuses
            # elsewhere. llama.cpp takes both under the same names.
            "frequency_penalty": gen_config.frequency_penalty,
            "presence_penalty": gen_config.presence_penalty,
        }
        # gen_config already normalised str-or-list into a list; re-deriving it
        # from request.stop was a second implementation of the same rule.
        if gen_config.stop_sequences:
            params["stop"] = list(gen_config.stop_sequences)
        # T-61, measured 2026-10-06 on the 3090 (LFM2.5-1.2B-Instruct Q4_K_M, temperature 1.0):
        # seed 7 twice in one instance AND in a fresh instance gave byte-identical text, and
        # seed 8 differed. So the seed is forwarded, not refused (025_FTASKS 6.6).
        if gen_config.seed is not None:
            params["seed"] = gen_config.seed
        return params

    def _llamacpp_messages(self, request: ChatCompletionRequest) -> list[dict]:
        """The message list llama.cpp templates internally."""
        return [{"role": m.role, "content": m.content or ""} for m in request.messages]

    async def _llamacpp_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat through llama.cpp.

        The streaming counterpart is `_llamacpp_stream_chat_completion`; both
        share the guards and the parameter mapping so one request cannot be
        refused on one path and quietly served on the other.

        Anything this engine cannot honour REFUSES rather than degrading quietly
        — a silently unsteered answer, or an embedding computed by a different
        method than the caller expects, is worse than a clear error.
        """
        self._refuse_unsupported_llamacpp_request(request)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        gen_config = GenerationConfig.from_request(request)

        params = self._llamacpp_params(gen_config, request)
        messages = self._llamacpp_messages(request)

        async with self._admit():
            # FR-27.8f: llama.cpp exposes no hook, so a probe cannot score here — and says so
            # rather than reaching generation silently. Arming refuses on this engine, so in
            # production nothing is armed and this is None; it lets the path guard hold on every
            # path with no exemption (FR-27.9).
            _probe_ctx = self._probe_begin_detached(completion_id, "engine_unsupported")
            _probe_verdicts = None
            try:
                try:
                    raw = await asyncio.to_thread(self._llamacpp_sync, messages, params)
                except Exception as exc:  # noqa: BLE001
                    raise self._translate_llamacpp_error(exc) from exc
                _probe_verdicts = self._probe_finish(_probe_ctx)
            finally:
                # llama.cpp tokenizes internally; there are no served ids to give a context window.
                await self._probe_record(
                    _probe_ctx, _probe_verdicts, full_ids=None, detached=True
                )

        choice_raw = (raw.get("choices") or [{}])[0]
        text = (choice_raw.get("message") or {}).get("content") or ""
        usage_raw = raw.get("usage") or {}

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=self._model_state.current.model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    # llama.cpp reports its own reason; fall back to "stop"
                    # rather than inventing a length claim we did not observe.
                    finish_reason=choice_raw.get("finish_reason") or "stop",
                )
            ],
            usage=Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
                completion_tokens=int(usage_raw.get("completion_tokens", 0)),
                total_tokens=int(usage_raw.get("total_tokens", 0)),
            ),
        )

    async def _llamacpp_stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming chat through llama.cpp, as SSE strings.

        Yields the SAME framed strings as the transformers path — role-only
        first chunk, content chunks, a final chunk carrying finish_reason and
        usage, then `data: [DONE]`. The route does no framing of its own, so
        matching that sequence exactly is the whole contract.

        MUCH THINNER than the transformers path, and deliberately so. llama.cpp
        already yields OpenAI-shaped `chat.completion.chunk` dicts, so there is
        no token reassembly; and its generator is PULL-based, so there is no
        producer thread, no `stop_event`, no `StoppingCriteria` and no
        five-second join. Copying that machinery would be theatre around a
        mechanism that does not exist here.
        """
        self._refuse_unsupported_llamacpp_request(request)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        model_name = self._model_state.current.model_name
        gen_config = GenerationConfig.from_request(request)

        # `prompt_opened_think=False`, and it cannot be otherwise here.
        # reasoning_split's contract is that this is "knowable exactly — it is
        # the string the template produced", which holds only because the
        # transformers path produces that string itself. llama.cpp applies the
        # template INTERNALLY, so miLLM never sees it.
        #
        # Consequence, stated precisely: for a model whose template opens
        # `<think>`, `reasoning_content` stays empty and the trace is delivered
        # inline in `content`. NOTHING IS LOST — a client that parses think tags
        # itself renders it correctly, and Open WebUI has a setting for exactly
        # this. What is lost is the wire-level split, which a client relying on
        # the `reasoning_content` convention would want.
        #
        # This is reasoning_split's own safe default: a visible trace beats an
        # answer moved into reasoning_content and looking like data loss.
        splitter = StreamingReasoningSplitter(False)

        params = self._llamacpp_params(gen_config, request)
        messages = self._llamacpp_messages(request)

        def _open_stream():
            # SAME continuation branch as the non-streaming path. Open WebUI
            # streams, so a fix applied only to the blocking path would leave
            # the actual user-facing case broken — which is exactly how the
            # llama.cpp guards diverged once before.
            prompt = self._llamacpp_continuation_prompt(messages)
            if prompt is not None:
                return _completion_chunks_as_chat(
                    self._model.create_completion(
                        prompt=prompt, stream=True, **params
                    )
                )
            return self._model.create_chat_completion(
                messages=messages, stream=True, **params
            )

        # HELD FOR THE WHOLE GENERATOR, not just the first chunk. `Llama` is not
        # thread-safe and owns a single C++ context; releasing between chunks
        # would let a second request interleave into it.
        async with self._admit(raise_refusal=False) as refusal:
            if refusal is not None:
                yield _stream_error_event(refusal)
                yield "data: [DONE]\n\n"
                return
            stream = None
            token_count = 0
            finish_reason = "stop"
            # FR-27.8f, as in `_llamacpp_chat_completion`. Recorded in the `finally` below.
            _probe_ctx = self._probe_begin_detached(completion_id, "engine_unsupported")
            _probe_verdicts = None
            try:
                try:
                    stream = await asyncio.to_thread(_open_stream)
                except Exception as exc:  # noqa: BLE001
                    raise self._translate_llamacpp_error(exc) from exc

                yield self._sse(
                    ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(role="assistant"),
                                finish_reason=None,
                            )
                        ],
                    )
                )

                async for raw in aiter_blocking(stream):
                    choice = (raw.get("choices") or [{}])[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    piece = (choice.get("delta") or {}).get("content")
                    if not piece:
                        continue
                    token_count += 1

                    reasoning, content = splitter.feed(piece)
                    if reasoning is None and content is None:
                        # Withheld: a `</think>` may be splitting across chunks.
                        continue
                    yield self._sse(
                        ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(
                                        content=content, reasoning_content=reasoning
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                    )

                flushed_reasoning, flushed_content = splitter.flush()
                if flushed_reasoning is not None or flushed_content is not None:
                    yield self._sse(
                        ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(
                                        content=flushed_content,
                                        reasoning_content=flushed_reasoning,
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                    )

                yield self._sse(
                    ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(),
                                finish_reason=finish_reason,
                            )
                        ],
                        usage=Usage(
                            prompt_tokens=self._llamacpp_prompt_tokens(messages),
                            completion_tokens=token_count,
                        ),
                    )
                )
                _probe_verdicts = self._probe_finish(_probe_ctx)
                async for _probe_extra in self._probe_stream_chunk(
                    _probe_verdicts, completion_id, created, model_name
                ):
                    yield _probe_extra
                yield "data: [DONE]\n\n"

            except Exception as e:  # noqa: BLE001
                # The status and headers are long since committed, so this
                # cannot become an HTTP error: it has to go out as an SSE event
                # and still close the stream, exactly as the transformers path
                # does for a crashed generation thread.
                logger.exception("llamacpp_streaming_error")
                yield (
                    'data: {"error":{"message":"An internal server error '
                    'occurred during streaming","type":"server_error",'
                    '"code":"streaming_error"}}\n\n'
                )
                yield "data: [DONE]\n\n"
            finally:
                # THIS is the abort. There is no flag to set and no thread to
                # join: llama.cpp's generator is pull-based, so closing it is
                # what actually stops sampling. On a client disconnect Starlette
                # closes this async generator, GeneratorExit lands at a yield,
                # and without this the C++ decode loop would keep running on a
                # stream nobody is reading.
                if stream is not None:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:  # noqa: BLE001
                            logger.warning("llamacpp_stream_close_failed")
                # llama.cpp tokenizes internally; there are no served ids to give a context window.
                await self._probe_record(
                    _probe_ctx, _probe_verdicts, full_ids=None, detached=True
                )

    def _llamacpp_prompt_tokens(self, messages: list[dict]) -> int:
        """Best-effort prompt token count for the final chunk's usage.

        llama.cpp never puts `usage` on a stream chunk, so this is measured
        here rather than reported. It tokenizes the concatenated message text,
        which is close but NOT the templated prompt llama.cpp actually consumed
        — the template adds control tokens this cannot see. Approximate and
        honest beats absent; zero would be a false measurement.
        """
        try:
            text = "\n".join(m.get("content") or "" for m in messages)
            return len(self._model.tokenize(text.encode("utf-8")))
        except Exception:  # noqa: BLE001 - usage must never fail a stream
            return 0

    @staticmethod
    def _sse(chunk: Any) -> str:
        """Frame a chunk as an SSE `data:` line, matching the transformers path."""
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    async def _llamacpp_text_completion(self, request: Any) -> Any:
        """Plain text completion through llama.cpp.

        Ollama serves `/v1/completions` from a GGUF file, so refusing it here
        made miLLM strictly less capable as a general-purpose offline server for
        no reason a caller could act on — `Llama.create_completion` is right
        there. The only genuine gap was that the transformers path reaches
        through `self._tokenizer`, which is None on this engine.
        """
        from millm.api.schemas.openai import (
            TextCompletionChoice,
            TextCompletionResponse,
        )

        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        gen_config = GenerationConfig.from_request(request)
        params = self._llamacpp_params(gen_config, request)

        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        choices: list[Any] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        def _complete(text: str) -> dict:
            return self._model.create_completion(prompt=text, **params)

        async with self._admit():
            # FR-27.8f, as in `_llamacpp_chat_completion`.
            _probe_ctx = self._probe_begin_detached(completion_id, "engine_unsupported")
            _probe_verdicts = None
            try:
                for index, prompt_text in enumerate(prompts):
                    raw = await asyncio.to_thread(_complete, prompt_text)
                    choice_raw = (raw.get("choices") or [{}])[0]
                    usage_raw = raw.get("usage") or {}
                    total_prompt_tokens += int(usage_raw.get("prompt_tokens", 0))
                    total_completion_tokens += int(usage_raw.get("completion_tokens", 0))
                    choices.append(
                        TextCompletionChoice(
                            index=index,
                            text=choice_raw.get("text") or "",
                            # llama.cpp's own reason, not a default we did not observe.
                            finish_reason=choice_raw.get("finish_reason") or "stop",
                        )
                    )

                _probe_verdicts = self._probe_finish(_probe_ctx)
            finally:
                # llama.cpp tokenizes internally; there are no served ids to give a context window.
                await self._probe_record(
                    _probe_ctx, _probe_verdicts, full_ids=None, detached=True
                )

        return TextCompletionResponse(
            id=completion_id,
            created=created,
            model=self._model_state.current.model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    async def _refuse_on_llamacpp(self, what: str) -> None:
        """Refuse an operation the llama.cpp engine cannot perform.

        Explicit refusal, never a quiet degradation. A streamed response that
        silently arrives in one chunk, or an embedding computed by a different
        method than the caller expects, is a wrong answer wearing a right
        answer's shape.
        """
        raise EngineUnsupportedError(
            f"{what} is not supported on the llama.cpp engine in this release."
        )

    async def stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion via SSE.

        Yields SSE-formatted strings: "data: {json}\\n\\n"
        First chunk has role, middle chunks have content, last has finish_reason.
        Always ends with "data: [DONE]\\n\\n".

        Args:
            request: The chat completion request

        Yields:
            SSE-formatted strings for streaming

        Raises:
            FieldNotHonouredError: `n > 1` or `extra_messages` — this path builds one choice
                from one conversation (FR-25.3.5). The schema refuses both for HTTP callers;
                this guard is for direct callers (Feature 26's batch runner).
        """
        if _constraint_type(getattr(request, "response_format", None)) is not None:
            raise ResponseFormatUnsupportedError(
                "response_format with stream=true is not supported in this release (T-59).",
                details={"param": "response_format"},
            )
        if _request_n(request) > 1 or _request_extra_messages(request):
            field_name = "n" if _request_n(request) > 1 else "extra_messages"
            raise FieldNotHonouredError(
                f"'{field_name}' is not honoured on a streaming chat completion: the stream "
                "carries one choice for one conversation",
                details={"param": field_name},
            )
        if self._engine_is_llamacpp():
            # Same shape as the CBM delegation below: a second engine's
            # streaming generator yielding the same SSE strings.
            async for chunk in self._llamacpp_stream_chat_completion(request):
                yield chunk
            return

        # Delegate to CBM if active and sampling params are compatible
        if self._use_cbm_for_request(**self._cbm_route_kwargs(request)):
            async for chunk in self._cbm_stream_chat_completion(request):
                yield chunk
            return

        from transformers import TextIteratorStreamer

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        # Format messages to prompt
        prompt = self._format_chat_messages(
            request.messages, request.chat_template_kwargs
        )
        # Routes tokens to reasoning_content until the think block closes.
        # Seeded from the PROMPT because granite-style templates open the tag
        # there, so the completion never contains an opening tag to detect.
        _splitter = StreamingReasoningSplitter(
            self._prompt_opened_think(prompt)
        )

        async with self._admit(raise_refusal=False) as refusal:
            if refusal is not None:
                # The model is being unloaded. The route refuses this before the
                # 200 (check_stream_admission); an unload that began since ends
                # the stream with the refusal and [DONE].
                yield _stream_error_event(refusal)
                yield "data: [DONE]\n\n"
                return
            # Per-request profile override (same logic as non-streaming path)
            _saved_steering = None
            if request.profile or request.steering_intensity is not None:
                try:
                    _saved_steering = await self._apply_request_steering(
                        request.profile, request.steering_intensity,
                        request_id=completion_id,
                    )
                except MiLLMError as exc:
                    # The 200 + headers are already committed (route-level
                    # pre-checks catch the 404 case, but gate/index errors
                    # and pre-check TOCTOUs land here) — emit an OpenAI-style
                    # error event instead of aborting the stream (010 R3).
                    logger.info(
                        "stream_steering_error_event",
                        code=exc.code,
                        profile=request.profile,
                    )
                    import json as _sse_json

                    error_event = _sse_json.dumps({
                        "error": {
                            "message": exc.message,
                            "type": "invalid_request_error",
                            "code": exc.code.lower(),
                        }
                    })
                    yield f"data: {error_event}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            # Probe boundary (Feature 24) — serial streaming path
            _probe_ctx = self._probe_begin(completion_id)
            _probe_verdicts = None

            # Sensing boundary (Feature 11) — serial streaming path
            _sensing_sae = self._sensing_begin(completion_id)
            _circuit_sensing = self._circuit_sensing_begin(completion_id)
            _id_capture = None
            _activations = None

            # Setup runs BEFORE the try/finally below that restores the
            # per-request steering, so any exception in this window
            # (tokenization, the context-length check, thread start) must
            # restore-and-reraise here — otherwise the dial/profile override
            # leaks into the global steering state (review R1, top finding).
            try:
                # Tokenize
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._get_input_device())
                prompt_tokens = inputs["input_ids"].shape[1]
                self._probe_note_prompt_length(_probe_ctx, prompt_tokens)
                self._probe_note_last_user_span(
                    _probe_ctx, request.messages, inputs["input_ids"], request.chat_template_kwargs
                )
                self._sensing_mark_history(_sensing_sae, inputs["input_ids"])
                _activations = self._activations_begin(request, prompt_tokens)

                # Set up streamer
                streamer = TextIteratorStreamer(
                    self._tokenizer, skip_prompt=True, skip_special_tokens=True
                )

                # Build generation kwargs
                gen_config = GenerationConfig.from_request(request)
                prompt_tokens = inputs["input_ids"].shape[1]
                self._check_context_length(prompt_tokens, gen_config.max_new_tokens)
                generation_kwargs = self._build_generate_kwargs(gen_config, inputs)
                generation_kwargs["streamer"] = streamer

                # Early-stop signal: set when the consumer stops reading (stop
                # sequence matched or client disconnected) so generate() halts
                # promptly instead of running to max_new_tokens while holding the
                # GPU and the queue slot.
                stop_event = Event()
                stopping_criteria = _make_event_stopping_criteria(stop_event)
                if stopping_criteria is not None:
                    generation_kwargs["stopping_criteria"] = stopping_criteria

                # Token-id capture for event context (Feature 11): criteria
                # run every step; storing the reference is zero-copy and
                # survives early stops (client disconnect, stop sequence).
                #
                # ⚠ FOR PROBES AND CIRCUIT SENSING TOO, not only SAE sensing. Installed for
                # sensing alone, a streamed request with probes armed handed the recorder the
                # PROMPT ids, so every response-window verdict's top position fell past their end
                # and its event stored no context — found on a real chat 2026-10-03, where the
                # non-streamed title and tags requests had context and the person's own did not.
                if (
                    _sensing_sae is not None
                    or _circuit_sensing is not None
                    or _probe_ctx is not None
                    or _activations is not None
                ) and stopping_criteria is not None:
                    _id_capture = _make_id_capture_criteria()
                    if _id_capture is not None:
                        stopping_criteria.append(_id_capture)

                # Start generation thread with error capture
                thread_error: list[Exception] = []
                thread = Thread(
                    target=self._generate_in_thread,
                    args=(generation_kwargs, thread_error),
                    # A plain Thread does not copy the caller's context, so the capture's owner
                    # is handed over explicitly (Feature 27): without it the hook would see no
                    # owner and this request's activations would never be recorded.
                    kwargs={**_seed_kwargs(gen_config),
                            **({"capture_owner": _activations.owner}
                               if _activations is not None else {})},
                )
                thread.start()
            except BaseException as setup_error:
                self._activations_close(_activations)
                self._restore_request_profile(_saved_steering)
                # Close the sensing boundary too — a stale open boundary
                # would let later non-begin passes sense with garbage
                # offsets (011 R1).
                if _sensing_sae is not None:
                    _sensing_sae.sae.collect_sensing_hits()
                if not isinstance(setup_error, MiLLMError):
                    raise
                # A refusal found here — a prompt past the context, which the
                # route's check_stream_admission refuses first unless the model
                # changed in between — arrives after the 200 is committed.
                # Raised, it cut the stream off with no error event and no
                # [DONE] (hardware acceptance, 2026-09-14, item 7).
                logger.info(
                    "stream_refused_before_generation",
                    code=setup_error.code,
                    completion_id=completion_id,
                )
                yield _stream_error_event(setup_error)
                yield "data: [DONE]\n\n"
                return

            try:
                # Send first chunk with role
                first_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant"),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

                # Stream tokens with stop sequence checking
                token_count = 0
                accumulated_text = ""
                stop_sequences = gen_config.stop_sequences
                stopped_by_sequence = False

                # TextIteratorStreamer.__next__ BLOCKS on a queue. Iterating
                # it directly inside an async generator pins the event loop for
                # the entire generation, so nothing already yielded can be
                # flushed to the socket and the client receives the whole
                # response in one burst at the end -- measured 2026-09-02:
                # 16 chunks all arriving at 6.12s, spread 0.00s, including the
                # role chunk that is yielded BEFORE generation starts. It also
                # starves every other request on the loop, health checks
                # included. Awaiting each token in a worker thread hands the
                # loop back between tokens.
                async for token in aiter_blocking(streamer):
                    if not token:
                        continue

                    # Count every token emitted, including a partial stop-sequence
                    # token.  Previously token_count += 1 appeared after the break
                    # and was unreachable on the stop-sequence path, making
                    # _determine_finish_reason compare a count one short of the
                    # actual generation length.
                    token_count += 1

                    if stop_sequences:
                        accumulated_text += token
                        truncated, found = self._apply_stop_sequences(
                            accumulated_text, stop_sequences
                        )
                        if found:
                            # Yield only the portion before the stop sequence
                            remaining = truncated[len(accumulated_text) - len(token):]
                            if remaining:
                                chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    created=created,
                                    model=model_name,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            index=0,
                                            delta=ChatCompletionChunkDelta(
                                                content=remaining
                                            ),
                                            finish_reason=None,
                                        )
                                    ],
                                )
                                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                            stopped_by_sequence = True
                            # Signal the generate() thread to stop instead of
                            # running to max_new_tokens after we stop reading.
                            stop_event.set()
                            break

                    _r, _c = _splitter.feed(token)
                    if _r is None and _c is None:
                        # Held back: a closing tag may be splitting across
                        # tokens. Emitting now would leak `</th` to the client.
                        continue
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    content=_c, reasoning_content=_r
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                # Check for thread errors before notifying monitoring — if the
                # thread crashed, its captured activations may be incomplete.
                if thread_error:
                    import json as _json
                    failure = thread_error[0]
                    error_msg = str(failure)
                    logger.error(
                        "generation_failed_during_stream",
                        error=error_msg,
                        completion_id=completion_id,
                    )
                    # Signal the client with an SSE error event followed by [DONE].
                    # The HTTP status is already 200 at this point; this is the
                    # standard approach for signalling mid-stream errors over SSE.
                    if isinstance(failure, GenerationOutOfMemoryError):
                        # The same envelope the non-streaming route answers with.
                        error_body = {
                            "message": failure.message,
                            "type": failure.openai_error_type,
                            "code": failure.code.lower(),
                        }
                    else:
                        error_body = {
                            "message": "Generation failed during streaming. "
                                       "See server logs for details.",
                            "type": "server_error",
                            "code": "generation_error",
                        }
                    error_event = _json.dumps({"error": error_body})
                    yield f"data: {error_event}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Notify monitoring after successful generation
                self._notify_monitoring(request_id=completion_id)

                # Determine finish reason
                if stopped_by_sequence:
                    finish_reason = "stop"
                else:
                    finish_reason = self._determine_finish_reason(
                        token_count, gen_config.max_new_tokens
                    )

                # Send final chunk with finish_reason and token usage.
                # Intermediate chunks omit `usage` (exclude_none=True strips it).
                # Emit anything still withheld by the split-tag guard.
                _fr, _fc = _splitter.flush()
                if _fr is not None or _fc is not None:
                    yield (
                        "data: "
                        + ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(
                                        content=_fc, reasoning_content=_fr
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        ).model_dump_json(exclude_none=True)
                        + "\n\n"
                    )

                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=token_count,
                    ),
                )
                # ⚠ FINISH BEFORE THE FINAL CHUNK, NOT IN THE `finally`.
                # By the time a `finally` runs here the stream is already closed and there is
                # nothing left to attach a verdict to. FTDD §8 names this precise hazard.
                _probe_verdicts = self._probe_finish(_probe_ctx)

                yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                async for _probe_extra in self._probe_stream_chunk(
                    _probe_verdicts, completion_id, created, model_name
                ):
                    yield _probe_extra
                # T-76: the activations, in ONE `choices: []` chunk AFTER the probe chunk and
                # before [DONE]. Read from the captured ids, which include generated tokens.
                if _activations is not None:
                    _act_ids = (_id_capture.latest_ids
                                if _id_capture is not None and _id_capture.latest_ids is not None
                                else inputs["input_ids"])
                    _millm = self._activations_finish(_activations, _act_ids)
                    if _millm is not None:
                        import json as _act_json

                        yield "data: " + _act_json.dumps({
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name, "choices": [],
                            "millm": _millm.model_dump(),
                        }) + "\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("streaming_error", error=str(e))
                # Try to send error in SSE format
                import json

                try:
                    error_event = json.dumps(
                        {
                            "error": {
                                "message": "An internal server error occurred during streaming",
                                "type": "server_error",
                                "code": "streaming_error",
                            }
                        }
                    )
                    yield f"data: {error_event}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception:
                    pass
            finally:
                # Always signal the generate() thread to stop — whether we
                # exited on a stop sequence, EOS, an exception, or a client
                # disconnect.  Without this an early exit would leave generate()
                # running to max_new_tokens, holding the GPU and the queue slot
                # (and delaying the steering restore below into the next
                # request's window).
                stop_event.set()
                thread.join(timeout=5.0)
                self._activations_close(_activations)
                if thread.is_alive():
                    # The generation thread did not finish within 5 seconds.  This
                    # typically means model.generate() is stuck (CUDA deadlock, OOM
                    # pending, or infinite loop in a stopping criterion).  Python
                    # cannot forcibly terminate threads, so the stuck thread will
                    # continue occupying GPU memory.  Signal the streamer to
                    # unblock any waiting iterators, log an error, and let the
                    # request queue release so subsequent requests can proceed —
                    # they may OOM, but at least the server remains responsive.
                    # If this happens repeatedly, restarting the server is required.
                    try:
                        streamer.on_finalize(None, None)  # unblock iterator
                    except Exception:
                        pass
                    logger.error(
                        "generation_thread_hung_after_5s",
                        completion_id=completion_id,
                        hint="GPU may be stuck. Restart the server if this recurs.",
                    )
                    # A hung generate thread can wake up later and keep
                    # calling _sense into the NEXT request's freshly-begun
                    # buffer (011 R1). Disarm: better to lose sensing until
                    # the cluster is re-activated than to mis-attribute.
                    if _sensing_sae is not None:
                        try:
                            import millm.api.dependencies as _deps

                            _deps.get_sensing_service().disarm(_sensing_sae.sae)
                        except Exception:
                            logger.warning("sensing_disarm_after_hang_failed")
                    # F15: same hazard, LARGER blast radius. The edge ring is
                    # SHARED across the circuit's layers, so a woken hung
                    # thread writes stale absolute positions into the next
                    # request's ring and corrupts EVERY layer's coordinates,
                    # not one self-contained buffer.
                    if _circuit_sensing:
                        try:
                            import millm.api.dependencies as _deps

                            _cs = _deps._circuit_sensing_service
                            if _cs is not None:
                                _cs.disarm(_circuit_sensing)
                                _cs.close_request()
                        except Exception:
                            logger.warning("circuit_sensing_disarm_after_hang_failed")
                    # F24: identical hazard for probes. A woken hung thread's forward pass would
                    # call the probe hook into the NEXT request's context, scoring one
                    # conversation's activations and reporting the verdict against another's id.
                    # Disarming with a recorded reason is better than a verdict about the wrong
                    # request — and the reason is what stops it reading as "nothing detected".
                    try:
                        from millm.services.probe_runtime import ProbeRuntimeState

                        _pstate = ProbeRuntimeState()
                        if _pstate.has_armed():
                            _pstate.disarm_all("generation_thread_hung")
                            # ⚠ This read a `dependencies._probe_arming_service` global that was
                            # never defined, so the rows were never marked (found 2026-10-04).
                            from millm.db.base import async_session_factory
                            from millm.services.probe_arming import (
                                HANG_REASON,
                                mark_armed_rows_disarmed,
                            )

                            await mark_armed_rows_disarmed(
                                async_session_factory, HANG_REASON, event="probes_disarmed_by_hang"
                            )
                        _pstate.end_request()
                    except Exception:
                        logger.warning("probe_disarm_after_hang_failed")
                        _circuit_sensing = None
                    # F27 (FTID §12): the same hazard for a per-request activation capture. A
                    # woken hung thread's forward pass would feed this request's capture — or,
                    # once the next request opens one, THAT request's — so every attached SAE's
                    # capture is closed here, beside the probe disarm.
                    self._close_request_captures("generation_thread_hung")
                # Restore steering to its pre-request state (Fix #1: steering race)
                self._restore_request_profile(_saved_steering)
                # Flush sensing hits: captured ids when any step ran, else
                # the prompt ids (prefill-only events still get context)
                _full_ids = (_id_capture.latest_ids
                             if _id_capture is not None
                             and _id_capture.latest_ids is not None
                             else inputs["input_ids"])
                await self._notify_sensing(_sensing_sae, _full_ids)
                await self._notify_circuit_sensing(_circuit_sensing, _full_ids)
                # Persist only — `_probe_finish` already ran before the terminal chunk. If
                # generation raised before reaching it, `_probe_verdicts` is None and
                # `_probe_record` computes them here so the event still exists.
                await self._probe_record(_probe_ctx, _probe_verdicts, full_ids=_full_ids)

    # =========================================================================
    # Text Completions
    # =========================================================================

    async def create_text_completion(
        self, request: TextCompletionRequest
    ) -> TextCompletionResponse:
        """
        Create non-streaming text completion.

        Args:
            request: The text completion request

        Returns:
            TextCompletionResponse with generated text
        """
        # SCORING MODE FIRST: neither llama.cpp nor the continuous-batching manager below can
        # return a per-token distribution, and routing a scoring request to either would generate
        # text with no logprobs — the request's whole point silently dropped.
        if request.wants_scores():
            return await self._score_text_completion(request)

        # llama.cpp first, for the same reason as the chat path: everything
        # below reaches through `self._tokenizer`, which is None on this engine.
        # Unguarded, the first line of the loop is
        # `None(prompt_text, return_tensors="pt")` -> "'NoneType' object is not
        # callable" as a 500, deep inside generation instead of at the boundary.
        if self._engine_is_llamacpp():
            return await self._llamacpp_text_completion(request)

        # Delegate to CBM if active and sampling params are compatible
        if self._use_cbm_for_request(**self._cbm_route_kwargs(request)):
            return await self._cbm_text_completion(request)

        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # Handle prompt as string or list
        prompts = (
            request.prompt
            if isinstance(request.prompt, list)
            else [request.prompt]
        )

        choices: list[TextCompletionChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with self._admit():
            gen_config = GenerationConfig.from_request(request)

            # Sensing boundary (011 R1: this endpoint was silently unsensed
            # while status said armed). Single-prompt only — multiple
            # prompts would concatenate position accounting, like n>1.
            _sensing_ctx = (self._sensing_begin(completion_id)
                            if len(prompts) == 1 else None)
            _circuit_sensing = (self._circuit_sensing_begin(completion_id)
                                if len(prompts) == 1 else None)
            _probe_ctx = self._probe_begin(completion_id)
            _probe_verdicts = None
            if _probe_ctx is not None and len(prompts) > 1:
                _probe_ctx.mark_not_scored("batched_request")
            _sensing_full_ids = None
            _activations = None
            _millm = None

            try:
                for i, prompt_text in enumerate(prompts):
                    # Tokenize input
                    inputs = self._tokenizer(prompt_text, return_tensors="pt").to(
                        self._get_input_device()
                    )
                    prompt_tokens = inputs.input_ids.shape[1]
                    self._probe_note_prompt_length(_probe_ctx, prompt_tokens)
                    if _probe_ctx is not None:
                        # A raw-text completion has no roles, so no user turn to read.
                        _probe_ctx.set_last_user_span(None, "text_completion_has_no_user_turn")
                    self._sensing_mark_history(_sensing_ctx, inputs.input_ids)
                    self._check_context_length(prompt_tokens, gen_config.max_new_tokens)
                    # Feature 27: one prompt only (the route refuses several, T-76).
                    if i == 0:
                        _activations = self._activations_begin(request, prompt_tokens)

                    # Generate - offload to thread to avoid blocking the event loop
                    generate_kwargs = self._build_generate_kwargs(
                        gen_config, inputs
                    )
                    # Seeded per prompt (FTID I7): each prompt is its own generate(), so prompt
                    # i's output does not depend on prompt i-1's length.
                    outputs = await asyncio.to_thread(
                        self._generate_sync, generate_kwargs, **_seed_kwargs(gen_config)
                    )
                    _sensing_full_ids = outputs[0]

                    # Notify monitoring after generation
                    self._notify_monitoring(request_id=completion_id)

                    # Decode output
                    generated_ids = self._slice_generated(outputs[0], prompt_tokens)
                    completion_text = self._tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    completion_tokens = len(generated_ids)

                    # Apply stop sequences
                    completion_text, stopped_by_sequence = (
                        self._apply_stop_sequences(
                            completion_text, gen_config.stop_sequences
                        )
                    )

                    # Determine finish reason with EOS logging
                    if stopped_by_sequence:
                        logger.debug("finish_reason_stop_sequence")
                        finish_reason = "stop"
                    else:
                        last_token_id = (
                            int(generated_ids[-1]) if len(generated_ids) > 0 else None
                        )
                        finish_reason = self._determine_finish_reason(
                            completion_tokens,
                            gen_config.max_new_tokens,
                            last_token_id=last_token_id,
                        )

                    choices.append(
                        TextCompletionChoice(
                            index=i,
                            text=completion_text,
                            finish_reason=finish_reason,
                        )
                    )

                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens

                if gen_config.seed is not None:
                    note_request_outcome(seed_scope=self._seed_scope(SEED_SCOPE_REQUEST))
                _probe_verdicts = self._probe_finish(_probe_ctx)
                _millm = self._activations_finish(_activations, _sensing_full_ids)
            finally:
                self._activations_close(_activations)
                await self._notify_sensing(_sensing_ctx, _sensing_full_ids)
                await self._notify_circuit_sensing(_circuit_sensing, _sensing_full_ids)
                await self._probe_record(_probe_ctx, _probe_verdicts, full_ids=_sensing_full_ids)

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return TextCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
            millm=_millm,
        )

    async def _score_text_completion(
        self, request: TextCompletionRequest
    ) -> TextCompletionResponse:
        """One forward pass per prompt; the next token's log-probabilities in OpenAI's shape.

        What a typed-decision judge reads its answer from (see `next_token_scores`). Probe and
        sensing hooks see no open request here, so they record nothing — a judge's prompt is not
        user traffic.

        ⚠ UNSTEERED, like embeddings (review round 1, M1). A judge's whole output is a probability;
        a steering profile left on the model would bias every verdict and nothing in the response
        would say so. Scoring runs with every attached SAE suppressed.
        """
        if self._engine_is_llamacpp():
            raise EngineUnsupportedError(
                "Scoring mode (logprobs / allowed_token_ids) needs the transformers engine; the "
                "loaded model runs on llama.cpp, which exposes no per-token distribution here."
            )
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

        async with self._admit():
            scored = await self._score_prompts(
                prompts,
                add_special_tokens=request.add_special_tokens,
                allowed=request.allowed_token_ids,
                temperature=request.temperature,
                top_k=request.logprobs or 0,
                activations_request=request,
                millm_out=(_millm_out := []),
                pack_size=1,
            )
            response = self._text_scoring_response(request, prompts, scored, _millm_out)
        return response

    def _text_scoring_response(
        self, request: TextCompletionRequest, prompts: list[str], scored: list[tuple[Any, int]],
        millm_out: list,
    ) -> TextCompletionResponse:
        """The `/v1/completions` scoring body for `scored` — ONE builder for the synchronous path
        and a packed batch row (Feature 26), so the two cannot drift. Called inside the slot (it
        decodes with the tokenizer an unload deletes)."""
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        choices: list[TextCompletionChoice] = []
        prompt_total = 0

        def key(token_id: int) -> str:
            if request.return_tokens_as_token_ids:
                return f"token_id:{token_id}"
            return self._tokenizer.decode([token_id])

        for index, (prompt_text, (scores, prompt_tokens)) in enumerate(zip(prompts, scored, strict=True)):
            choices.append(
                TextCompletionChoice(
                    index=index,
                    text=self._tokenizer.decode([scores.chosen_id]),
                    finish_reason="length",
                    # `allowed_token_ids` alone constrains the token without asking for
                    # scores; vLLM then returns `logprobs: null`, and so does this.
                    logprobs=None if request.logprobs is None else CompletionLogprobs(
                        tokens=[key(scores.chosen_id)],
                        token_logprobs=[scores.chosen_logprob],
                        top_logprobs=[{key(t): lp for t, lp in scores.top}],
                        text_offset=[len(prompt_text)],
                    ),
                )
            )
            prompt_total += prompt_tokens

        if request.seed is not None:
            note_request_outcome(seed_scope=SEED_SCOPE_REQUEST)  # scoring is deterministic
        model_info = self.get_loaded_model_info()
        _millm = next((m for m in millm_out if m is not None), None)
        return TextCompletionResponse(
            millm=_millm,
            id=completion_id,
            created=created,
            model=model_info.name if model_info else "unknown",
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_total,
                completion_tokens=len(choices),
                total_tokens=prompt_total + len(choices),
            ),
        )

    async def _score_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Chat scoring (FR-25.5 – FR-25.9): render each conversation's chat template with the
        generation prompt, then score it through `_score_prompts` — the scorer /v1/completions
        uses — with `add_special_tokens=False`, because a rendered template already carries its
        BOS (a double BOS would shift every score while looking plausible).

        Unsteered and unrecorded like completion scoring (FR-25.7, X-09): no probe, sensing or
        circuit-sensing context is opened, no steering is applied (the table refuses steering
        fields on a scoring request), and every attached SAE is suppressed in the worker thread.

        One choice per conversation, `index` in input order (index 0 is `messages`), scored one
        at a time under ONE admission slot (FR-25.9).
        """
        if self._engine_is_llamacpp():
            raise EngineUnsupportedError(
                "Chat scoring (logprobs / allowed_token_ids) needs the transformers engine; the "
                "loaded model runs on llama.cpp, which exposes no per-token distribution here."
            )
        conversations = [request.messages] + list(request.extra_messages or [])
        top_n = request.top_logprobs or 0

        async with self._admit():
            texts = self._chat_scoring_texts(request, conversations)
            scored = await self._score_prompts(
                texts,
                add_special_tokens=False,
                allowed=request.allowed_token_ids,
                temperature=request.temperature,
                top_k=top_n,
                label="conversation",
                activations_request=request,
                millm_out=(_millm_out := []),
                pack_size=1,
            )
            response = self._chat_scoring_response(request, scored, _millm_out)
        return response

    def _chat_scoring_texts(self, request: ChatCompletionRequest, conversations: list) -> list[str]:
        """Render each conversation's chat template with the generation prompt. Inside the slot:
        an unload deletes the tokenizer (check_stream_admission's note)."""
        if not getattr(self._tokenizer, "chat_template", None):
            model_info = self.get_loaded_model_info()
            name = model_info.name if model_info else "the loaded model"
            raise NoChatTemplateError(
                f"'{name}' has no chat template, so chat scoring would score a generic "
                "format the model was never trained on. Score the rendered prompt on "
                "/v1/completions instead (T-55).",
                details={"param": "model"},
            )
        return [
            self._format_chat_messages(conversation, request.chat_template_kwargs)
            for conversation in conversations
        ]

    def _chat_scoring_response(
        self, request: ChatCompletionRequest, scored: list[tuple[Any, int]], millm_out: list
    ) -> ChatCompletionResponse:
        """The chat scoring body for `scored` — one builder for the synchronous path and a packed
        batch row (Feature 26)."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        top_n = request.top_logprobs or 0
        choices: list[ChatCompletionChoice] = []
        prompt_total = 0

        def key(token_id: int, decoded: str) -> str:
            return f"token_id:{token_id}" if request.return_tokens_as_token_ids else decoded

        for index, (scores, prompt_tokens) in enumerate(scored):
            decoded = self._tokenizer.decode([scores.chosen_id])
            logprobs = None
            if request.logprobs is True:
                alternatives = []
                for token_id, lp in scores.top[:top_n]:
                    text = self._tokenizer.decode([token_id])
                    alternatives.append(ChatLogprobAlternative(
                        token=key(token_id, text), logprob=lp,
                        bytes=list(text.encode("utf-8")),
                    ))
                logprobs = ChatLogprobs(content=[ChatLogprobToken(
                    token=key(scores.chosen_id, decoded),
                    logprob=scores.chosen_logprob,
                    bytes=list(decoded.encode("utf-8")),
                    top_logprobs=alternatives,
                )])
            choices.append(ChatCompletionChoice(
                index=index,
                message=ChatMessage(role="assistant", content=decoded),
                finish_reason="length",
                logprobs=logprobs,
            ))
            prompt_total += prompt_tokens

        if request.seed is not None:
            note_request_outcome(seed_scope=SEED_SCOPE_REQUEST)  # scoring is deterministic
        model_info = self.get_loaded_model_info()
        _millm = next((m for m in millm_out if m is not None), None)
        return ChatCompletionResponse(
            millm=_millm,
            id=completion_id,
            created=created,
            model=model_info.name if model_info else "unknown",
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_total,
                completion_tokens=len(choices),
                total_tokens=prompt_total + len(choices),
            ),
        )

    async def _score_prompts(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        allowed: Optional[list[int]],
        temperature: float,
        top_k: int,
        label: str = "prompt",
        activations_request: Any = None,
        millm_out: Optional[list] = None,
        pack_size: int = 1,
    ) -> list[tuple[Any, int]]:
        """THE scorer: one unsteered forward pass per text, in order; `(NextTokenScores,
        prompt_tokens)` for each. Chat and completion scoring BOTH call this, so the arithmetic
        exists once and the two agree by construction (PADR §10 "One scoring path"; FR-25.5.7).

        Must be called INSIDE `_admit()` (an unload deletes the tokenizer). Never calls
        `generate()` and never opens a probe, sensing or circuit-sensing context, so Feature 27's
        discovery test (FR-27.9) can classify it as a path that never reaches generation
        (FR-25.7.4). Feature 27 (R-04.26) attaches `return_sae_activations` capture HERE, in
        scoring mode, which is why this stays one function.

        One input at a time, never packed: bfloat16 is not batch-invariant (PADR §10 "Packed
        scoring"), and packing would break chat-vs-completion equality (FR-25.9.2).

        A failure names its index (`details["index"]`, and the message), and no partial result
        is returned (FR-25.9.4).

        Feature 26 (FTASKS 6.1): `pack_size` > 1 scores the texts in right-padded packs through
        `_score_specs_packed` — the Batch API's packed path. Every synchronous caller passes
        `pack_size=1`, which is this loop, unchanged and bit-identical. Packing is refused with
        activation capture (one request's positions cannot be attributed inside a pack).
        """
        from millm.services.next_token_scores import next_token_scores

        if pack_size > 1 and activations_request is None:
            specs = [
                ScoreSpec(text, add_special_tokens, allowed, temperature, top_k) for text in texts
            ]
            outcomes = await self._score_specs_packed(specs, max_rows=pack_size)
            for index, outcome in enumerate(outcomes):
                if isinstance(outcome, MiLLMError):
                    if isinstance(outcome.details, dict):
                        outcome.details.setdefault("index", index)
                    raise outcome
            return [(o.scores, o.prompt_tokens) for o in outcomes]  # type: ignore[union-attr]

        results: list[tuple[Any, int]] = []
        for index, text in enumerate(texts):
            try:
                inputs = self._tokenizer(
                    text, return_tensors="pt", add_special_tokens=add_special_tokens
                ).to(self._get_input_device())
                prompt_tokens = int(inputs.input_ids.shape[1])
                if prompt_tokens == 0:
                    raise InvalidScoringRequestError(
                        f"{label} {index} tokenises to nothing, so there is no position to score"
                    )
                self._check_context_length(prompt_tokens, 1)
                # Feature 27 (FR-27.3): activations in scoring mode are read UNSTEERED — every SAE
                # is suppressed in the worker, and the capture still records under suppression.
                # `last` is the last prompt position, the one whose distribution is scored.
                capture = (
                    self._activations_begin(activations_request, prompt_tokens, "unsteered")
                    if activations_request is not None and index == 0 else None
                )
                try:
                    # Suppression is entered INSIDE the worker thread
                    # (`_unsteered_next_token_logits`); around this await it would suppress
                    # nothing there.
                    logits = await asyncio.to_thread(self._unsteered_next_token_logits, inputs)
                    if capture is not None and millm_out is not None:
                        millm_out.append(self._activations_finish(capture, inputs.input_ids))
                finally:
                    self._activations_close(capture)
                # The vocabulary is what the model actually scores — its logits — not a config
                # field or an embedding matrix some wrappers do not expose (review round 1).
                vocab = int(logits.shape[-1])
                if allowed and max(allowed) >= vocab:
                    raise InvalidScoringRequestError(
                        f"allowed_token_ids contains {max(allowed)}, outside the loaded model's "
                        f"vocabulary of {vocab}"
                    )
                # NaN or +inf poisons the normaliser — over the whole vocabulary when
                # unrestricted, over the allowed ids only when restricted (log-softmax runs over
                # that set alone, so a NaN elsewhere cannot touch the answer; review round 3). A
                # -inf is how some heads mask padded vocabulary, so it is refused only where it
                # would be REPORTED.
                checked = (
                    logits if allowed is None
                    else logits[torch.tensor(allowed, dtype=torch.long)]
                )
                if bool(torch.isnan(checked).any()) or bool(torch.isposinf(checked).any()):
                    raise ScoringNumericalError(
                        "The model produced NaN or infinite logits for this prompt, so no "
                        "probability can be reported for it."
                    )
                scores = next_token_scores(
                    logits, allowed=allowed, temperature=temperature, top_k=top_k
                )
                if not all(math.isfinite(lp) for _, lp in scores.top):
                    raise ScoringNumericalError(
                        "A reported log-probability is not finite: a requested token's logit is "
                        "-inf under this model, so it has no probability to report."
                    )
            except MiLLMError as exc:
                if isinstance(exc.details, dict) and "index" not in exc.details:
                    exc.details["index"] = index
                    if len(texts) > 1 and f"{label} {index}" not in exc.message:
                        exc.message = f"{exc.message} ({label} {index})"
                        exc.args = (exc.message,)
                raise
            results.append((scores, prompt_tokens))
        return results

    async def _score_specs_packed(
        self, specs: list["ScoreSpec"], *, max_rows: int, max_tokens: Optional[int] = None
    ) -> list[Any]:
        """Score many prompts in right-padded packs; one outcome per spec, in order (Feature 26).

        Each outcome is a `PackedScore` or the `MiLLMError` that spec alone earned — a too-long
        prompt fails ITS row, not its neighbours. Must be called inside `_admit()`.

        ⚠ RIGHT padding, gathered at each row's own last real token (FTDD TD5). Padding after a
        causal row cannot change any earlier position, for attention, convolution and recurrent
        mixers alike; left padding would shift position ids and is unsafe for a convolution mixer
        (LFM2). The tokenizer's own `padding_side` is never touched (the reason is at the batched
        generation path). Unsteered, like the single path.

        Bounds: at most `max_rows` rows and `max_tokens` padded tokens per forward. A CUDA
        out-of-memory error halves the pack and retries; at one row it becomes that row's error
        (`_chunk_batch_for_memory` already prefers a slow answer to a 500).
        """
        from millm.core.config import settings
        from millm.services.next_token_scores import next_token_scores

        budget = int(max_tokens if max_tokens is not None else settings.BATCH_PACK_MAX_TOKENS)
        outcomes: list[Any] = [None] * len(specs)
        ready: list[tuple[int, list[int]]] = []
        for index, spec in enumerate(specs):
            try:
                ids = list(self._tokenizer(
                    spec.text, add_special_tokens=spec.add_special_tokens
                )["input_ids"])
                if not ids:
                    raise InvalidScoringRequestError(
                        "the prompt tokenises to nothing, so there is no position to score"
                    )
                self._check_context_length(len(ids), 1)
                ready.append((index, ids))
            except MiLLMError as exc:
                outcomes[index] = exc

        packs: list[list[tuple[int, list[int]]]] = []
        current: list[tuple[int, list[int]]] = []
        for item in ready:
            longest = max([len(item[1])] + [len(i[1]) for i in current])
            if current and (len(current) >= max_rows or longest * (len(current) + 1) > budget):
                packs.append(current)
                current = []
            current.append(item)
        if current:
            packs.append(current)

        async def run(pack: list[tuple[int, list[int]]]) -> None:
            try:
                rows = await asyncio.to_thread(
                    self._unsteered_call, lambda: self._packed_next_token_logits(
                        [ids for _, ids in pack]
                    )
                )
            except GenerationOutOfMemoryError as exc:
                if len(pack) == 1:
                    outcomes[pack[0][0]] = exc
                    return
                half = len(pack) // 2
                await run(pack[:half])
                await run(pack[half:])
                return
            for (index, ids), logits in zip(pack, rows, strict=True):
                spec = specs[index]
                try:
                    outcomes[index] = PackedScore(
                        self._scores_from_logits(logits, spec, next_token_scores), len(ids)
                    )
                except MiLLMError as exc:
                    outcomes[index] = exc

        for pack in packs:
            await run(pack)
        return outcomes

    def _scores_from_logits(self, logits: torch.Tensor, spec: "ScoreSpec", scorer: Any) -> Any:
        """The single path's checks and arithmetic over one row's logits (unchanged rules)."""
        vocab = int(logits.shape[-1])
        allowed = spec.allowed
        if allowed and max(allowed) >= vocab:
            raise InvalidScoringRequestError(
                f"allowed_token_ids contains {max(allowed)}, outside the loaded model's "
                f"vocabulary of {vocab}"
            )
        checked = logits if allowed is None else logits[torch.tensor(allowed, dtype=torch.long)]
        if bool(torch.isnan(checked).any()) or bool(torch.isposinf(checked).any()):
            raise ScoringNumericalError(
                "The model produced NaN or infinite logits for this prompt, so no probability "
                "can be reported for it."
            )
        scores = scorer(logits, allowed=allowed, temperature=spec.temperature, top_k=spec.top_k)
        if not all(math.isfinite(lp) for _, lp in scores.top):
            raise ScoringNumericalError(
                "A reported log-probability is not finite: a requested token's logit is -inf "
                "under this model, so it has no probability to report."
            )
        return scores

    def _packed_next_token_logits(self, rows: list[list[int]]) -> list[torch.Tensor]:
        """ONE forward over right-padded rows; each row's logits at ITS last real token.

        `logits_to_keep` is passed as the index tensor of the distinct last positions (0.5: the
        served architectures accept a tensor); a class that rejects it falls back to full
        logits, bounded by the pack token budget.
        """
        device = self._get_input_device()
        lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
        width = int(lengths.max())
        pad = self._tokenizer.pad_token_id
        if pad is None:
            pad = self._tokenizer.eos_token_id if self._tokenizer.eos_token_id is not None else 0
        input_ids = torch.full((len(rows), width), int(pad), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
        last = lengths - 1
        keep = torch.unique(last)  # sorted
        inputs = {"input_ids": input_ids.to(device), "attention_mask": mask.to(device)}
        failure: Optional[GenerationOutOfMemoryError] = None
        try:
            with torch.no_grad():
                try:
                    outputs = self._model(**inputs, use_cache=False, logits_to_keep=keep.to(device))
                    # ⚠ Row i's scored position is the index of last[i] WITHIN `keep` — never -1,
                    # which is a padding position for every row shorter than the longest
                    # (mutation control M13).
                    where = torch.searchsorted(keep, last)
                except TypeError as exc:
                    if "logits_to_keep" not in str(exc):
                        raise
                    outputs = self._model(**inputs, use_cache=False)
                    where = last
            logits = outputs.logits
            return [logits[i, int(where[i])].float().cpu() for i in range(len(rows))]
        except torch.cuda.OutOfMemoryError as exc:
            failure = _scoring_oom_error(exc, {"input_ids": input_ids})
        _release_generation_memory()
        logger.error("scoring_out_of_memory", packed_rows=len(rows), **failure.details)
        raise failure

    def _unsteered_next_token_logits(self, inputs: Any) -> torch.Tensor:
        """`_next_token_logits` with every attached SAE suppressed — in the worker thread itself,
        where the forward hooks run (review round 2, M-A)."""
        with self._unsteered():
            return self._next_token_logits(inputs)

    def _next_token_logits(self, inputs: Any) -> torch.Tensor:
        """The last position's logits as float32 on the CPU, from one forward pass.

        ⚠ ONLY THE LAST POSITION (review round 1, H1). A plain forward computes logits for every
        prompt position — ~2 GB at 4k tokens with a 248k vocabulary, ~16 GB at 32k — which
        `generate()` avoids by asking for one. A CUDA out-of-memory error is the same typed,
        memory-releasing refusal generation gives, not a bare 500.
        """
        failure: Optional[GenerationOutOfMemoryError] = None
        try:
            with torch.no_grad():
                try:
                    outputs = self._model(**inputs, use_cache=False, logits_to_keep=1)
                except TypeError as exc:
                    if "logits_to_keep" not in str(exc):
                        raise
                    outputs = self._model(**inputs, use_cache=False)
            return outputs.logits[0, -1].float().cpu()
        except torch.cuda.OutOfMemoryError as exc:
            failure = _scoring_oom_error(exc, inputs)
        _release_generation_memory()
        logger.error("scoring_out_of_memory", **failure.details)
        raise failure

    # =========================================================================
    # Embeddings
    # =========================================================================

    #: At most this many over-limit inputs are named in one refusal; the rest are counted.
    EMBEDDING_MAX_LISTED = 16

    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Create embeddings for input text (Feature 30).

        The last hidden layer, pooled over real tokens by `request.pooling` (`mean` default,
        `last`, `cls`) and L2-normalised when `request.normalize` is true. Every input is
        tokenised WITHOUT truncation and measured before any forward pass; an input over the
        model's limit is refused naming its index (FR-30.3). Float and base64 encodings. Always
        unsteered, and no probe or sensing event is recorded.
        """
        if self._engine_is_llamacpp():
            return await self._llamacpp_embeddings(request)

        texts = request.input if isinstance(request.input, list) else [request.input]
        options = EmbeddingOptions(pooling=request.pooling, normalize=request.normalize)
        encoding_format = getattr(request, "encoding_format", "float") or "float"

        async with self._admit():
            vectors, counts = self._embed_inputs(
                texts, options, param_for_string=isinstance(request.input, str)
            )

        embeddings_data = [
            EmbeddingData(index=i, embedding=_encode_embedding(vector, encoding_format))
            for i, vector in enumerate(vectors)
        ]
        total_tokens = sum(counts)
        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        return EmbeddingResponse(
            data=embeddings_data,
            model=model_name,
            usage=Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            ),
        )

    def _embed_inputs(
        self,
        texts: list[str],
        options: EmbeddingOptions,
        *,
        param_for_string: bool = False,
    ) -> tuple[list[list[float]], list[int]]:
        """Embed `texts` on the transformers engine: (vectors, full token counts).

        THE embedding body, and the entry point Feature 26's batch executor calls. Synchronous
        and takes NO slot: the caller holds one (`create_embeddings` enters `_admit()`).

        Measure, then refuse, then run. Every input is tokenised first, with `truncation=False`
        stated explicitly — before Feature 30 it was `truncation=True`, so an over-long input was
        embedded from its first N tokens and returned a 200 — and the whole set is checked
        before ANY forward. The check used to run inside the loop, so input 0 was embedded
        before input 1 was refused.

        Each forward runs under `torch.no_grad()` and `_unsteered()`, entered on THIS thread,
        the one running the forward: suppression is per-thread, and every attached SAE (a
        circuit attaches one per layer) must be inert (2026-10-04 fix). No probe context is
        opened, so the probe hook records nothing (FR-30.2.9).
        """
        tokenizer = self._tokenizer
        encoded = [tokenizer(text, return_tensors="pt", truncation=False) for text in texts]
        counts = [int(enc["input_ids"].shape[1]) for enc in encoded]
        self._check_embedding_lengths(
            counts, self._embedding_limit(), param_for_string=param_for_string
        )

        device = self._get_input_device()
        vectors: list[list[float]] = []
        for index, enc in enumerate(encoded):
            enc = enc.to(device)
            with torch.no_grad(), self._unsteered():
                outputs = self._model(**enc, output_hidden_states=True)
            pooled = pool_hidden(
                outputs.hidden_states[-1], _attention_mask_of(enc), options.pooling
            )[0]
            try:
                vectors.append(finalize_vector(pooled, options.normalize))
            except NonFiniteEmbeddingError as exc:
                raise self._embedding_vector_invalid(
                    index, exc, param_for_string=param_for_string
                ) from exc
        return vectors, counts

    def _embedding_limit(self) -> int | None:
        """The longest input, in tokens, this model embeds without truncation; None if unknown.

        transformers: the served context (`_served_max_context`), the same limit generation uses.
        llama.cpp: `min(n_ctx(), n_batch)`. llama-cpp-python's `embed()` cuts each input to
        `n_batch` tokens by default (its `truncate=True`), and `create_embedding` does not expose
        the flag, so an input longer than `n_batch` would be embedded from a prefix. Bounding by
        both refuses it instead. (030_FTASKS 0.1 confirms the names and the truncation on the
        backend image; FTDD TD15.)
        """
        if self._engine_is_llamacpp():
            handle = self._model
            return min(int(handle.n_ctx()), int(handle.n_batch))
        return _served_max_context(getattr(self._model, "config", None))

    def _check_embedding_lengths(
        self, counts: list[int], limit: int | None, *, param_for_string: bool
    ) -> None:
        """Refuse when any input exceeds `limit`, naming every over-limit index (bounded).

        `limit` None means the model states no limit: inputs are served untruncated and
        unchecked (FTDD R3). Logs one `embedding_refused` warning carrying indices and counts,
        never input text.
        """
        if limit is None:
            return
        over = [(i, n) for i, n in enumerate(counts) if n > limit]
        if not over:
            return
        listed = over[: self.EMBEDDING_MAX_LISTED]
        omitted = len(over) - len(listed)
        parts = ", ".join(f"input {i} has {n:,} tokens" for i, n in listed)
        more = f" ({omitted} more over the limit not listed)" if omitted else ""
        param = "input" if param_for_string else f"input[{over[0][0]}]"
        logger.warning(
            "embedding_refused",
            reason="input_too_long",
            indices=[i for i, _ in listed],
            tokens=[n for _, n in listed],
            omitted=omitted,
            limit=limit,
        )
        raise EmbeddingInputTooLongError(
            f"{parts[0].upper()}{parts[1:]}{more}; this model's limit is {limit:,} tokens. "
            "Inputs are never truncated. Shorten or split them.",
            details={
                "param": param,
                "max_context_tokens": limit,
                "over_limit": [{"index": i, "tokens": n} for i, n in listed],
                "omitted": omitted,
            },
        )

    @staticmethod
    def _embedding_vector_invalid(
        index: int, exc: Exception, *, param_for_string: bool
    ) -> EmbeddingVectorInvalidError:
        param = "input" if param_for_string else f"input[{index}]"
        logger.warning("embedding_refused", reason="vector_invalid", indices=[index])
        return EmbeddingVectorInvalidError(
            f"Input {index}: {exc}", details={"param": param, "index": index}
        )

    async def _llamacpp_embeddings(self, request: Any) -> Any:
        """Embeddings from a GGUF model.

        Refused in the first increment on the reasoning that llama.cpp needs
        `embedding=True` at CONSTRUCTION and pools internally, so parity would
        mean quietly returning differently-computed vectors. Both halves of that
        turned out to be wrong, and measuring settled it:

          * one instance serves BOTH. Loaded with `embedding=True` and MEAN
            pooling, the same handle answered create_embedding AND
            create_chat_completion on the RTX 3090.
          * the pooling is not different. The transformers path mean-pools the
            last hidden layer; LLAMA_POOLING_TYPE_MEAN is the same strategy, so
            the vectors are comparable in METHOD rather than merely both being
            called embeddings.

        Feature 30: pooling is fixed at construction, so only `mean` is served (the request
        policy refuses `last` and `cls` before any load; the guard here is defence in depth).
        `normalize` is applied here, after the engine. Every input is tokenised with the
        instance's own tokenizer and checked against `_embedding_limit()` before the first
        `create_embedding` call, so the engine never truncates. Tokenising runs inside the
        slot: an unload frees the model the tokenizer belongs to.

        The load-time flag is `settings.GGUF_ENABLE_EMBEDDINGS` (default on;
        measured cost 6.7% of generation throughput). When it is off the model
        was built without embedding support and llama.cpp raises — surfaced here
        as a clear refusal naming the setting, rather than the library's error.
        """
        from millm.api.schemas.openai import EmbeddingData, EmbeddingResponse
        from millm.core.config import settings as _settings

        texts = request.input if isinstance(request.input, list) else [request.input]
        param_for_string = isinstance(request.input, str)
        options = EmbeddingOptions(pooling=request.pooling, normalize=request.normalize)
        encoding_format = getattr(request, "encoding_format", "float") or "float"
        if options.pooling != "mean":
            raise EngineUnsupportedError(
                f"pooling '{options.pooling}' is not served on a GGUF model: llama.cpp fixes "
                "pooling when the model is loaded, and miLLM loads GGUF models with mean pooling.",
                details={"param": "pooling"},
            )

        embeddings_data: list[Any] = []
        total_tokens = 0

        def _embed(text: str) -> tuple[list[float], int]:
            raw = self._model.create_embedding(text)
            vector = raw["data"][0]["embedding"]
            # With pooling enabled this is a flat vector; with pooling NONE it
            # would be per-token. Flatten defensively rather than emit a nested
            # list that a client would read as a batch.
            if vector and isinstance(vector[0], list):
                vector = [sum(col) / len(col) for col in zip(*vector)]
            used = int((raw.get("usage") or {}).get("prompt_tokens", 0))
            return vector, used

        async with self._admit():
            handle = self._model
            counts = [len(handle.tokenize(text.encode("utf-8"))) for text in texts]
            self._check_embedding_lengths(
                counts, self._embedding_limit(), param_for_string=param_for_string
            )
            for index, text in enumerate(texts):
                try:
                    vector, used = await asyncio.to_thread(_embed, text)
                except Exception as exc:  # noqa: BLE001
                    if not _settings.GGUF_ENABLE_EMBEDDINGS:
                        raise EngineUnsupportedError(
                            "This GGUF model was loaded without embedding "
                            "support. Set GGUF_ENABLE_EMBEDDINGS=true and "
                            "reload the model: llama.cpp can only enable it at "
                            "construction."
                        ) from exc
                    raise

                try:
                    values = finalize_vector(vector, options.normalize)
                except NonFiniteEmbeddingError as exc:
                    raise self._embedding_vector_invalid(
                        index, exc, param_for_string=param_for_string
                    ) from exc
                total_tokens += used
                embeddings_data.append(
                    EmbeddingData(index=index, embedding=_encode_embedding(values, encoding_format))
                )

        model_info = self.get_loaded_model_info()
        return EmbeddingResponse(
            data=embeddings_data,
            model=model_info.name if model_info else "unknown",
            usage=Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            ),
        )

    # =========================================================================
    # CBM Generation Methods (Continuous Batching)
    # =========================================================================

    async def _cbm_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Chat completion via ContinuousBatchingManager."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # FR-27.8d: the continuous-batching manager scores nothing per request, so the request
        # says `not_scored: continuous_batching`, as the streaming CBM path already did. Detached
        # (FR-27.8h): concurrent CBM requests cannot share the runtime's single context.
        probe_ctx = self._probe_begin_detached(completion_id, "continuous_batching")
        probe_verdicts = None
        input_ids: list[int] = []
        try:
            prompt = self._format_chat_messages(
                request.messages, request.chat_template_kwargs
            )
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
            gen_config = GenerationConfig.from_request(request)
            # The serial path's check; this delegation had none (hardware acceptance,
            # 2026-09-14, item 7).
            self._check_context_length(len(input_ids), gen_config.max_new_tokens)

            generated_ids, finish_reason = await self._cbm_backend.generate(
                input_ids=input_ids,
                max_new_tokens=gen_config.max_new_tokens,
                request_id=completion_id,
            )

            self._notify_monitoring(request_id=completion_id)

            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            text, stopped = self._apply_stop_sequences(text, gen_config.stop_sequences)
            if stopped:
                finish_reason = "stop"

            model_info = self.get_loaded_model_info()
            model_name = model_info.name if model_info else "unknown"
            # Before the return: the chat route reads the verdicts straight afterwards.
            probe_verdicts = self._probe_finish(probe_ctx)

            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=self._assistant_message(text, prompt),
                        finish_reason=finish_reason,
                    )
                ],
                usage=Usage(
                    prompt_tokens=len(input_ids),
                    completion_tokens=len(generated_ids),
                    total_tokens=len(input_ids) + len(generated_ids),
                ),
            )
        finally:
            await self._probe_record(probe_ctx, probe_verdicts, full_ids=input_ids, detached=True)

    async def _cbm_stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion via CBM."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # ⚠ PROBES CANNOT SCORE HERE, AND MUST SAY SO (task 5.8, BR-006).
        #
        # `PROBE_FORCE_SERIAL` normally keeps continuous batching out while anything is armed —
        # but it is a SETTING, and this path does no sensing at all (its only observability call
        # is `_notify_monitoring`). Sensing gets away with relying on its flag alone; a probe
        # cannot, because "a probe never goes silently quiet" is this feature's governing
        # invariant. With the flag off and a probe armed, a verdict would simply be ABSENT — no
        # header, no chunk, no event, no reason — which is the failure the feature exists to
        # prevent, wearing the costume of a normal response.
        #
        # ⚠ DETACHED (FR-27.8h). This used `_probe_begin`, which REGISTERS the context with the
        # runtime's single slot — so a second concurrent CBM request found it occupied, got None,
        # and recorded nothing, and the first request's `_probe_record` could close the second's.
        probe_ctx = self._probe_begin_detached(completion_id, "continuous_batching")
        probe_verdicts = None
        input_ids: list[int] = []

        try:
            model_info = self.get_loaded_model_info()
            model_name = model_info.name if model_info else "unknown"

            prompt = self._format_chat_messages(
                request.messages, request.chat_template_kwargs
            )
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
            self._probe_note_prompt_length(probe_ctx, len(input_ids))
            gen_config = GenerationConfig.from_request(request)
            try:
                # The route's check_stream_admission refuses this first; after the
                # 200 a refusal is an error event and [DONE], never a cut-off stream.
                self._check_context_length(len(input_ids), gen_config.max_new_tokens)
            except ContextLengthExceededError as refusal:
                yield _stream_error_event(refusal)
                yield "data: [DONE]\n\n"
                return
            _splitter = StreamingReasoningSplitter(
                self._prompt_opened_think(prompt)
            )

            # First chunk: role
            first_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(role="assistant"),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

            # Stream tokens from CBM
            token_count = 0
            async for new_token_ids in self._cbm_backend.generate_stream(
                input_ids=input_ids,
                max_new_tokens=gen_config.max_new_tokens,
                request_id=completion_id,
            ):
                text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
                if text:
                    token_count += len(new_token_ids)
                    _r, _c = _splitter.feed(text)
                    if _r is None and _c is None:
                        continue        # withheld: a closing tag may be splitting
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    content=_c, reasoning_content=_r
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

            _fr, _fc = _splitter.flush()
            if _fr is not None or _fc is not None:
                yield (
                    "data: "
                    + ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    content=_fc, reasoning_content=_fr
                                ),
                                finish_reason=None,
                            )
                        ],
                    ).model_dump_json(exclude_none=True)
                    + "\n\n"
                )

            self._notify_monitoring(request_id=completion_id)

            # Final chunk with finish_reason
            finish_reason = self._determine_finish_reason(
                token_count, gen_config.max_new_tokens
            )
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(),
                        finish_reason=finish_reason,
                    )
                ],
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
            # The verdict says `not_scored: continuous_batching` rather than nothing at all.
            probe_verdicts = self._probe_finish(probe_ctx)
            async for extra in self._probe_stream_chunk(
                probe_verdicts, completion_id, created, model_name
            ):
                yield extra
            yield "data: [DONE]\n\n"
        finally:
            # In the `finally`, so a failed or abandoned stream still records (FR-27.8c).
            # `input_ids` is a plain list here; `context_window` accepts either shape. Verdicts on
            # this path are `not_scored: continuous_batching` and so carry no top position, but
            # the ids are passed rather than dropped so this site does not become the one that
            # silently stops producing context if that ever changes.
            await self._probe_record(
                probe_ctx, probe_verdicts, full_ids=input_ids, detached=True
            )

    async def _cbm_text_completion(
        self, request: TextCompletionRequest
    ) -> TextCompletionResponse:
        """Text completion via ContinuousBatchingManager."""
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        prompts = (
            request.prompt
            if isinstance(request.prompt, list)
            else [request.prompt]
        )

        choices: list[TextCompletionChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        gen_config = GenerationConfig.from_request(request)

        # FR-27.8d, as in `_cbm_chat_completion`. Detached (FR-27.8h).
        probe_ctx = self._probe_begin_detached(completion_id, "continuous_batching")
        probe_verdicts = None
        input_ids: list[int] = []
        try:
            for i, prompt_text in enumerate(prompts):
                input_ids = self._tokenizer.encode(
                    prompt_text, return_tensors="pt"
                )[0].tolist()
                prompt_tokens = len(input_ids)
                self._check_context_length(prompt_tokens, gen_config.max_new_tokens)

                generated_ids, finish_reason = await self._cbm_backend.generate(
                    input_ids=input_ids,
                    max_new_tokens=gen_config.max_new_tokens,
                    request_id=f"{completion_id}-{i}",
                )

                self._notify_monitoring(request_id=completion_id)

                completion_text = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                completion_tokens = len(generated_ids)

                completion_text, stopped = self._apply_stop_sequences(
                    completion_text, gen_config.stop_sequences
                )
                if stopped:
                    finish_reason = "stop"

                choices.append(
                    TextCompletionChoice(
                        index=i,
                        text=completion_text,
                        finish_reason=finish_reason,
                    )
                )

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

            model_info = self.get_loaded_model_info()
            model_name = model_info.name if model_info else "unknown"
            # Before the return: the completions route reads the verdicts straight afterwards.
            probe_verdicts = self._probe_finish(probe_ctx)

            return TextCompletionResponse(
                id=completion_id,
                created=created,
                model=model_name,
                choices=choices,
                usage=Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                ),
            )
        finally:
            await self._probe_record(probe_ctx, probe_verdicts, full_ids=input_ids, detached=True)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _generate_sync(self, generation_kwargs: dict, seed: Optional[int] = None) -> Any:
        """
        Run model.generate() synchronously (for use with asyncio.to_thread).

        This keeps the blocking GPU computation off the async event loop,
        allowing FastAPI to continue serving health checks, WebSocket
        connections, and other requests during inference.

        A CUDA out-of-memory error is raised as GenerationOutOfMemoryError, after
        the failed pass's memory is released (review round 6, 2026-09-14). It
        reached clients as a bare 500, and the cache stayed held by the traceback
        until the error handler had finished with it.
        """
        try:
            # The seed is applied HERE, in the worker thread, inside the request's admission slot
            # (FR-25.13.2): no other request's sampling can consume the seeded stream.
            with seeded_rng(seed), torch.no_grad():
                return self._model.generate(**generation_kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            refusal = _generation_oom_error(exc, generation_kwargs)
        # Outside the except block: the traceback holding the failed pass's
        # tensors is gone before the cache is emptied.
        _release_generation_memory()
        logger.error("generation_out_of_memory", **refusal.details)
        raise refusal

    def _generate_in_thread(
        self,
        generation_kwargs: dict,
        errors: Optional[list] = None,
        seed: Optional[int] = None,
        capture_owner: Optional[str] = None,
    ) -> None:
        """
        Run generation in thread for streaming.

        Must be called in separate thread because generate() is blocking.
        Errors are captured in the errors list so the caller can check them.

        On ANY failure the streamer is ended, after the error is recorded.
        generate() ends the streamer only when it finishes (transformers 5.15.1
        generation/utils.py:2944, not in a finally), and TextIteratorStreamer
        waits with no timeout: a generation that raised left the consumer
        blocked forever — no error event, no [DONE], the request queue slot held
        until the client gave up, and one executor thread stranded for good.
        Review round 6, 2026-09-14. A CUDA out-of-memory error is recorded as
        GenerationOutOfMemoryError, after its memory is released.
        """
        failure: Optional[Exception] = None
        if capture_owner is not None:
            # This plain Thread starts with an empty context; the request's capture owner is set
            # here so the SAE hook may feed that request's activations (Feature 27).
            from millm.ml.sae_wrapper import CAPTURE_OWNER

            CAPTURE_OWNER.set(capture_owner)
        try:
            with seeded_rng(seed), torch.no_grad():
                self._model.generate(**generation_kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            failure = _generation_oom_error(exc, generation_kwargs)
        except Exception as e:
            logger.error("generation_thread_error", error=str(e))
            failure = e
        if failure is None:
            return
        if isinstance(failure, GenerationOutOfMemoryError):
            _release_generation_memory()
            logger.error("generation_out_of_memory", **failure.details)
        if errors is not None:
            errors.append(failure)
        streamer = generation_kwargs.get("streamer")
        if streamer is not None:
            try:
                streamer.end()
            except Exception as e:  # noqa: BLE001 - the error is already recorded
                logger.warning("streamer_end_after_failure_failed", error=str(e))


    @staticmethod
    def _prompt_opened_think(prompt: Optional[str]) -> bool:
        """Did the chat template leave a `<think>` block open?

        Knowable exactly -- it is the string the template produced. This is the
        positive evidence `split_reasoning` needs before it will treat a
        completion as reasoning, which is what stops a non-reasoning model's
        answer being moved into `reasoning_content`.
        """
        return bool(prompt) and prompt.rstrip().endswith(THINK_OPEN)

    def _assistant_message(
        self, text: Optional[str], prompt: Optional[str] = None
    ) -> ChatMessage:
        """Build the assistant message, splitting any reasoning trace out."""
        reasoning, content = split_reasoning(
            text, self._prompt_opened_think(prompt)
        )
        return ChatMessage(
            role="assistant", content=content, reasoning_content=reasoning
        )

    def _format_chat_messages(
        self,
        messages: list[ChatMessage],
        template_kwargs: Optional[dict] = None,
    ) -> str:
        """
        Format chat messages into prompt string.

        Uses model's chat template if available, otherwise falls back
        to Gemma-style format with turn markers.

        Args:
            messages: List of chat messages

        Returns:
            Formatted prompt string
        """
        # Log incoming messages for debugging template issues
        for i, m in enumerate(messages):
            logger.debug(
                "chat_message",
                index=i,
                role=m.role,
                content_preview=m.content[:200] if m.content else "",
            )

        template_kwargs = dict(template_kwargs or {})

        # Prefer model's built-in chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                # Check if chat_template is actually set
                if self._tokenizer.chat_template:
                    formatted = self._tokenizer.apply_chat_template(
                        [{"role": m.role, "content": m.content} for m in messages],
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_kwargs,
                    )
                    logger.debug(
                        "formatted_prompt",
                        length=len(formatted),
                        preview=formatted[:500],
                        template_kwargs=sorted(template_kwargs) or None,
                    )
                    return formatted
            except Exception as e:
                # FAIL LOUDLY when the caller asked for something specific.
                #
                # The fallback below is a generic Gemma-style format. Reaching
                # it after an explicit chat_template_kwargs request is doubly
                # wrong: the model is formatted for the wrong family AND the
                # request is discarded, and the caller still gets a 200. For
                # enable_thinking=False that means reasoning stays on and the
                # deliberation lands in their parsed output looking like an
                # answer. A 500 they can see beats a wrong answer they cannot.
                if template_kwargs:
                    raise ValueError(
                        "chat template rejected "
                        f"{sorted(template_kwargs)}: {e}"
                    ) from e
                logger.warning(
                    "chat_template_failed_using_fallback", error=str(e)
                )

        if template_kwargs:
            raise ValueError(
                "chat_template_kwargs was requested "
                f"({sorted(template_kwargs)}) but this model has no chat "
                "template, so the generic fallback format would silently "
                "ignore it"
            )

        # Fallback: Gemma-style format with turn markers
        # This format works well with Gemma 2 and similar models
        parts = []
        pending_system = None
        for msg in messages:
            if msg.role == "system":
                # Buffer system message to prepend to next user turn
                pending_system = msg.content
            elif msg.role == "user":
                if pending_system:
                    parts.append(
                        f"<start_of_turn>user\n{pending_system}\n\n{msg.content}<end_of_turn>"
                    )
                    pending_system = None
                else:
                    parts.append(f"<start_of_turn>user\n{msg.content}<end_of_turn>")
            elif msg.role == "assistant":
                parts.append(f"<start_of_turn>model\n{msg.content}<end_of_turn>")

        # If there's a dangling system message with no user turn after it
        if pending_system:
            parts.append(f"<start_of_turn>user\n{pending_system}<end_of_turn>")

        # Add generation prompt
        parts.append("<start_of_turn>model")
        return "\n".join(parts)
