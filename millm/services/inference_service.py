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
import contextvars
import math
import uuid
from datetime import datetime
from threading import Event, Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

import torch

from millm.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    TextCompletionChoice,
    TextCompletionRequest,
    TextCompletionResponse,
    Usage,
)
from millm.core.errors import MiLLMError
from millm.core.logging import get_logger
from millm.ml.generation_config import GenerationConfig
from millm.ml.model_loader import LoadedModelState
from millm.services.request_queue import RequestQueue

if TYPE_CHECKING:
    from millm.services.model_service import ModelService
    from millm.services.monitoring_service import MonitoringService
from millm.services.reasoning_split import (
    OPEN as THINK_OPEN,
    StreamingReasoningSplitter,
    split_reasoning,
)


logger = get_logger(__name__)

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


def note_circuit_apply_failed() -> None:
    """Record that this request's circuit dial did not apply."""
    _CIRCUIT_APPLY_FAILED.set(True)


def circuit_apply_failed() -> bool:
    """True if this request's circuit dial failed to apply.

    The rung echo MUST consult this before emitting a header: a rung phrase
    describes evidence for an intervention, and no intervention ran.
    """
    return _CIRCUIT_APPLY_FAILED.get()


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
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_state = LoadedModelState()
        self._kv_cache_mode = kv_cache_mode
        self._speculative_model_id = speculative_model
        self._speculative_num_tokens = speculative_num_tokens
        # Lazy-loaded on first use. Thread-safety note: with max_concurrent=1
        # only one generate call is active at a time, so the double-init race
        # (two requests both seeing None and both loading the draft model) is
        # practically impossible. If max_concurrent is ever raised above 1, add
        # a threading.Lock here before reading/writing _draft_model.
        self._draft_model: Any = None
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

    @staticmethod
    def _normalize_device(d: object) -> str:
        """
        Convert an accelerate device_map value to a valid PyTorch device string.

        accelerate stores device_map values as integers (0, 1, ...) for CUDA
        devices, or as the strings "cpu" / "disk".  PyTorch's .to() only accepts
        proper device strings like "cuda:0", so integers must be converted.
        """
        if isinstance(d, int):
            return f"cuda:{d}"
        s = str(d)
        if s == "cpu" or s.startswith("cuda"):
            return s
        # Bare integer stored as string (shouldn't happen, but be safe)
        try:
            return f"cuda:{int(s)}"
        except ValueError:
            return s

    def _get_input_device(self) -> str:
        """
        Return the device where model inputs (input_ids) should be placed.

        For device_map="auto" models, model.device returns "cpu" (a dispatch
        device), which doesn't reflect where the embedding layer actually lives.
        We inspect hf_device_map first, then fall back to the first parameter's
        device, and finally to self._device.
        """
        if not self._model_state.is_loaded:
            return self._device
        try:
            hf_model = self._model_state.current.model
            # device_map models expose hf_device_map; find where embeddings live
            device_map = getattr(hf_model, "hf_device_map", None)
            if device_map:
                for key in ("", "model.embed_tokens", "transformer.wte",
                            "model.embedding", "model.shared", "model.embed"):
                    if key in device_map:
                        d = self._normalize_device(device_map[key])
                        # Skip "disk" (offloaded to disk, not a valid .to() target)
                        if d != "disk":
                            return d
                # Fall back to the device of the first non-disk layer
                for val in device_map.values():
                    d = self._normalize_device(val)
                    if d not in ("disk", "cpu"):
                        return d
                # All layers on CPU or disk — return cpu
                return "cpu"
            # Non-device_map model: use first parameter device
            return str(next(hf_model.parameters()).device)
        except Exception:
            return self._device

    def _use_cbm(self) -> bool:
        """Whether to use continuous batching for generation."""
        return self._cbm_backend is not None and self._cbm_backend.is_running

    @property
    def backend_name(self) -> str:
        """Active inference backend identifier for observability headers."""
        return "cbm" if self._use_cbm() else "serial"

    def get_backend_info(self) -> dict:
        """
        Return a description of the active inference backend and its capabilities.

        Used by the /api/inference/status endpoint so operators and clients
        can understand which path is serving requests and what its limitations are.
        """
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

    def _use_cbm_for_request(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        has_steering_override: bool = False,
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
        if self._cbm_force_serial_monitoring and self._is_monitoring_enabled():
            logger.info(
                "cbm_routing_fallback_to_serial",
                reason="force_serial_monitoring_active",
            )
            return False
        return True

    def on_model_loaded(self) -> None:
        """Called after model is loaded. Starts CBM if enabled."""
        if self._cbm_backend is not None and self._model_state.is_loaded:
            try:
                model = self._model_state.current.model
                tokenizer = self._model_state.current.tokenizer
                self._cbm_backend.start(model, tokenizer)
            except Exception as e:
                logger.warning("cbm_start_failed", error=str(e))

    def on_model_unloading(self) -> None:
        """Called before model unload. Stops CBM if running."""
        if self._cbm_backend is not None and self._cbm_backend.is_running:
            self._cbm_backend.stop()

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

        if self._draft_model is None:
            try:
                from transformers import AutoModelForCausalLM

                logger.info(
                    "loading_draft_model",
                    model_id=self._speculative_model_id,
                )
                self._draft_model = AutoModelForCausalLM.from_pretrained(
                    self._speculative_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                self._draft_model.eval()
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
            gen_config = GenerationConfig(
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                do_sample=gen_config.do_sample,
                stop_sequences=gen_config.stop_sequences,
                frequency_penalty=gen_config.frequency_penalty,
                presence_penalty=gen_config.presence_penalty,
                cache_implementation="static",
            )
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

        return kwargs

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

        Raises:
            ValueError: If context length would be exceeded.
        """
        max_length = getattr(
            getattr(self._model, "config", None), "max_position_embeddings", None
        )
        if max_length is None:
            return  # Can't validate without config

        total = prompt_tokens + max_new_tokens
        if total > max_length:
            from millm.api.routes.openai.errors import context_length_exceeded_error
            raise ValueError(
                f"Context length exceeded: {prompt_tokens} prompt + "
                f"{max_new_tokens} max_tokens = {total} > {max_length}"
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

        outputs = await asyncio.to_thread(self._generate_sync, generate_kwargs)

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

            if stopped_by_sequence:
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
            from millm.ml.memory_utils import is_cuda_available, verify_memory_available

            if is_cuda_available() and prompts:
                longest = max(len(self._tokenizer.encode(p)) for p in prompts)
                total_len = longest + max(int(max_new_tokens or 0), 0)
                while rows > 1:
                    projected = self._project_kv_bytes(rows, total_len)
                    if projected is None:
                        break  # unmeasurable -> keep the row cap, do not grow
                    need_mb = int(projected / (1024 * 1024) * 1.2)  # +20% slack
                    ok, available_mb = verify_memory_available(need_mb)
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
        if self._use_cbm_for_request(
            temperature=getattr(request, "temperature", None),
            top_p=getattr(request, "top_p", None),
            has_steering_override=self._has_steering_override(request),
        ):
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

        choices: list[ChatCompletionChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with self._request_queue.acquire():
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
                                message=self._assistant_message(
                                    row["text"], chunk[offset]
                                ),
                                finish_reason=row["finish_reason"],
                            )
                        )
                        total_prompt_tokens += row["prompt_tokens"]
                        total_completion_tokens += row["completion_tokens"]
            finally:
                self._restore_request_profile(_saved_steering)

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
        # Batched extension: every conversation in ONE forward pass. Checked
        # before the CBM delegation because that path has no batch support and
        # would silently drop all but the first conversation.
        if getattr(request, "extra_messages", None):
            return await self._create_batched_chat_completion(request)

        # Delegate to CBM if active and sampling params are compatible
        if self._use_cbm_for_request(
            temperature=getattr(request, "temperature", None),
            top_p=getattr(request, "top_p", None),
            has_steering_override=self._has_steering_override(request),
        ):
            return await self._cbm_chat_completion(request)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        # Format messages to prompt
        prompt = self._format_chat_messages(
            request.messages, request.chat_template_kwargs
        )
        n = getattr(request, "n", 1) or 1

        choices: list[ChatCompletionChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with self._request_queue.acquire():
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
            if n > 1:
                from millm.services.sae_service import AttachedSAEState as _S

                _armed_sae = _S().attached_sae
                if _armed_sae is not None and _armed_sae.is_sensing_armed:
                    logger.info(
                        "sensing_skipped", reason="n_gt_1", n=n,
                        request_id=completion_id,
                    )
            _sensing_full_ids = None

            try:
                # Tokenize input
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._get_input_device())
                prompt_tokens = inputs.input_ids.shape[1]
                _sensing_full_ids = inputs.input_ids  # prefill-only fallback
                self._sensing_mark_history(_sensing_sae, inputs.input_ids)

                # Build generation config
                gen_config = GenerationConfig.from_request(request)
                self._check_context_length(prompt_tokens, gen_config.max_new_tokens)

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
                        ChatCompletionChoice(
                            index=i,
                            message=self._assistant_message(
                                completion_text, prompt
                            ),
                            finish_reason=finish_reason,
                        )
                    )

                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens

            finally:
                # Restore steering to its pre-request state regardless of success/failure.
                self._restore_request_profile(_saved_steering)
                # Flush sensing hits (post-generation, inside the semaphore
                # so the boundary can't interleave with the next request)
                await self._notify_sensing(_sensing_sae, _sensing_full_ids)
                await self._notify_circuit_sensing(_circuit_sensing, _sensing_full_ids)

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
        """
        # Delegate to CBM if active and sampling params are compatible
        if self._use_cbm_for_request(
            temperature=getattr(request, "temperature", None),
            top_p=getattr(request, "top_p", None),
            has_steering_override=self._has_steering_override(request),
        ):
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

        async with self._request_queue.acquire():
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

            # Sensing boundary (Feature 11) — serial streaming path
            _sensing_sae = self._sensing_begin(completion_id)
            _circuit_sensing = self._circuit_sensing_begin(completion_id)
            _id_capture = None

            # Setup runs BEFORE the try/finally below that restores the
            # per-request steering, so any exception in this window
            # (tokenization, the context-length check, thread start) must
            # restore-and-reraise here — otherwise the dial/profile override
            # leaks into the global steering state (review R1, top finding).
            try:
                # Tokenize
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._get_input_device())
                prompt_tokens = inputs["input_ids"].shape[1]
                self._sensing_mark_history(_sensing_sae, inputs["input_ids"])

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

                # Token-id capture for sensing context (Feature 11): criteria
                # run every step; storing the reference is zero-copy and
                # survives early stops (client disconnect, stop sequence).
                if _sensing_sae is not None and stopping_criteria is not None:
                    _id_capture = _make_id_capture_criteria()
                    if _id_capture is not None:
                        stopping_criteria.append(_id_capture)

                # Start generation thread with error capture
                thread_error: list[Exception] = []
                thread = Thread(
                    target=self._generate_in_thread,
                    args=(generation_kwargs, thread_error),
                )
                thread.start()
            except BaseException:
                self._restore_request_profile(_saved_steering)
                # Close the sensing boundary too — a stale open boundary
                # would let later non-begin passes sense with garbage
                # offsets (011 R1).
                if _sensing_sae is not None:
                    _sensing_sae.sae.collect_sensing_hits()
                raise

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

                for token in streamer:
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
                    error_msg = str(thread_error[0])
                    logger.error(
                        "generation_failed_during_stream",
                        error=error_msg,
                        completion_id=completion_id,
                    )
                    # Signal the client with an SSE error event followed by [DONE].
                    # The HTTP status is already 200 at this point; this is the
                    # standard approach for signalling mid-stream errors over SSE.
                    error_event = _json.dumps({
                        "error": {
                            "message": "Generation failed during streaming. "
                                       "See server logs for details.",
                            "type": "server_error",
                            "code": "generation_error",
                        }
                    })
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
                yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
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
                        _circuit_sensing = None
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
        # Delegate to CBM if active and sampling params are compatible
        if self._use_cbm_for_request(
            temperature=getattr(request, "temperature", None),
            top_p=getattr(request, "top_p", None),
            has_steering_override=self._has_steering_override(request),
        ):
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

        async with self._request_queue.acquire():
            gen_config = GenerationConfig.from_request(request)

            # Sensing boundary (011 R1: this endpoint was silently unsensed
            # while status said armed). Single-prompt only — multiple
            # prompts would concatenate position accounting, like n>1.
            _sensing_ctx = (self._sensing_begin(completion_id)
                            if len(prompts) == 1 else None)
            _circuit_sensing = (self._circuit_sensing_begin(completion_id)
                                if len(prompts) == 1 else None)
            _sensing_full_ids = None

            try:
                for i, prompt_text in enumerate(prompts):
                    # Tokenize input
                    inputs = self._tokenizer(prompt_text, return_tensors="pt").to(
                        self._get_input_device()
                    )
                    prompt_tokens = inputs.input_ids.shape[1]
                    self._sensing_mark_history(_sensing_ctx, inputs.input_ids)
                    self._check_context_length(prompt_tokens, gen_config.max_new_tokens)

                    # Generate - offload to thread to avoid blocking the event loop
                    generate_kwargs = self._build_generate_kwargs(
                        gen_config, inputs
                    )
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
            finally:
                await self._notify_sensing(_sensing_ctx, _sensing_full_ids)
                await self._notify_circuit_sensing(_circuit_sensing, _sensing_full_ids)

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
        )

    # =========================================================================
    # Embeddings
    # =========================================================================

    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Create embeddings for input text.

        Uses the model's last hidden layer with mean pooling.
        Supports float and base64 encoding formats.

        Args:
            request: The embedding request

        Returns:
            EmbeddingResponse with embeddings
        """
        import base64
        import struct

        # Normalize input to list
        inputs = (
            request.input if isinstance(request.input, list) else [request.input]
        )

        encoding_format = getattr(request, "encoding_format", "float") or "float"

        embeddings_data: list[EmbeddingData] = []
        total_tokens = 0

        # Embeddings must reflect the *unsteered* model: an attached SAE hook
        # would otherwise perturb the hidden states these embeddings are pooled
        # from, and the pass would clobber the last-captured monitoring
        # activations.  Suppress the hook for the duration of the embedding
        # forward passes.
        import contextlib

        attached_sae = self._get_attached_sae()

        async with self._request_queue.acquire():
            for i, text in enumerate(inputs):
                # Tokenize
                encoded = self._tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                ).to(self._get_input_device())
                total_tokens += encoded.input_ids.shape[1]

                # Get embeddings from last hidden layer
                suppress_ctx = (
                    attached_sae.suppressed()
                    if attached_sae is not None
                    else contextlib.nullcontext()
                )
                with torch.no_grad(), suppress_ctx:
                    outputs = self._model(
                        **encoded, output_hidden_states=True
                    )

                # Extract last hidden layer and mean pool
                last_hidden = outputs.hidden_states[-1]
                embedding = last_hidden.mean(dim=1).squeeze().cpu().tolist()

                # Ensure embedding is a list
                if isinstance(embedding, float):
                    embedding = [embedding]

                # Encode as base64 if requested
                if encoding_format == "base64":
                    packed = struct.pack(f"<{len(embedding)}f", *embedding)
                    embedding = base64.b64encode(packed).decode("ascii")

                embeddings_data.append(EmbeddingData(index=i, embedding=embedding))

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

    # =========================================================================
    # CBM Generation Methods (Continuous Batching)
    # =========================================================================

    async def _cbm_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Chat completion via ContinuousBatchingManager."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        prompt = self._format_chat_messages(
            request.messages, request.chat_template_kwargs
        )
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        gen_config = GenerationConfig.from_request(request)

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

    async def _cbm_stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion via CBM."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())

        model_info = self.get_loaded_model_info()
        model_name = model_info.name if model_info else "unknown"

        prompt = self._format_chat_messages(
            request.messages, request.chat_template_kwargs
        )
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        gen_config = GenerationConfig.from_request(request)
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
        yield "data: [DONE]\n\n"

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

        for i, prompt_text in enumerate(prompts):
            input_ids = self._tokenizer.encode(
                prompt_text, return_tensors="pt"
            )[0].tolist()
            prompt_tokens = len(input_ids)

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

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _generate_sync(self, generation_kwargs: dict) -> Any:
        """
        Run model.generate() synchronously (for use with asyncio.to_thread).

        This keeps the blocking GPU computation off the async event loop,
        allowing FastAPI to continue serving health checks, WebSocket
        connections, and other requests during inference.
        """
        with torch.no_grad():
            return self._model.generate(**generation_kwargs)

    def _generate_in_thread(
        self, generation_kwargs: dict, errors: Optional[list] = None
    ) -> None:
        """
        Run generation in thread for streaming.

        Must be called in separate thread because generate() is blocking.
        Errors are captured in the errors list so the caller can check them.
        """
        try:
            with torch.no_grad():
                self._model.generate(**generation_kwargs)
        except Exception as e:
            logger.error("generation_thread_error", error=str(e))
            if errors is not None:
                errors.append(e)


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
