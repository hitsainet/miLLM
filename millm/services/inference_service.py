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
from millm.core.logging import get_logger
from millm.ml.generation_config import GenerationConfig
from millm.ml.model_loader import LoadedModelState
from millm.services.request_queue import RequestQueue

if TYPE_CHECKING:
    from millm.services.model_service import ModelService
    from millm.services.monitoring_service import MonitoringService

logger = get_logger(__name__)


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
        """
        The profile's declared budget.intensity_range as an ordered float
        pair, or None when absent/malformed. cluster_meta stores the RAW
        imported document (lossless storage), so nothing about the shape or
        ordering of the range can be assumed here — a swapped or
        non-numeric range degrades to None rather than inverting the dial
        or 500ing every symbolic request.
        """
        if profile is None or not getattr(profile, "cluster_meta", None):
            return None
        budget = (profile.cluster_meta or {}).get("budget") or {}
        candidate = budget.get("intensity_range")
        if not isinstance(candidate, list) or len(candidate) != 2:
            return None
        try:
            lo, hi = float(candidate[0]), float(candidate[1])
        except (TypeError, ValueError):
            return None
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

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
        if isinstance(raw, (int, float)):
            return float(raw)
        from millm.core.config import settings

        rng = cls._intensity_range_of(profile)
        lo, hi = (rng if rng is not None
                  else (settings.CLUSTER_INTENSITY_MIN, settings.CLUSTER_INTENSITY_MAX))
        return {"off": 0.0, "min": lo, "max": hi}[raw]

    async def resolve_request_intensity(
        self, request: ChatCompletionRequest
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
        """
        raw = getattr(request, "steering_intensity", None)
        if raw is None:
            return None

        from millm.services.sae_service import AttachedSAEState

        if AttachedSAEState().attached_sae is None:
            return None  # apply will no-op; an echoed lambda would be a lie
        if isinstance(raw, (int, float)):
            return float(raw)
        try:
            from millm.db.base import async_session_factory
            from millm.db.repositories.profile_repository import ProfileRepository

            async with async_session_factory() as session:
                repo = ProfileRepository(session)
                profile = (await repo.get_by_name(request.profile)
                           if getattr(request, "profile", None)
                           else await repo.get_active())
        except Exception:
            logger.warning("intensity_echo_resolution_failed", exc_info=True)
            return None
        if getattr(request, "profile", None) and profile is None:
            return None  # apply will raise ProfileNotFound; don't echo first
        return self._resolve_intensity(raw, profile)

    async def _apply_request_steering(
        self,
        profile_name: Optional[str],
        intensity_raw: "float | str | None" = None,
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

        lam = self._resolve_intensity(intensity_raw, profile)
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

        # A numeric dial is clamped into the cluster's authored
        # intensity_range (the declared safe envelope the management API
        # already enforces — /v1 is unauthenticated and must not become the
        # overdrive loophole). Dial-to-0 stays always allowed, matching
        # set_intensity semantics; symbolic values resolve inside the range
        # by construction.
        if lam is not None and lam != 0.0 and profile is not None:
            rng = self._intensity_range_of(profile)
            if rng is not None:
                bounded = min(max(lam, rng[0]), rng[1])
                if bounded != lam:
                    logger.info(
                        "request_intensity_clamped_to_authored_range",
                        requested=lam,
                        applied=bounded,
                        profile=profile.name,
                    )
                    lam = bounded

        if lam == 0.0:
            # Effective λ 0 disables steering for this request only —
            # uniformly, whatever the base would have been.
            if not sae.is_steering_enabled:
                return None  # already off; nothing to restore
            saved: dict = {
                "values": sae.get_steering_values(),
                "enabled": True,
            }
            sae.enable_steering(False)
            logger.info(
                "request_steering_disabled",
                profile=profile.name if profile else None,
                intensity=0.0,
            )
            return saved

        if profile is not None and profile.steering:
            if not explicit and not sae.is_steering_enabled:
                # Dial-only requests must never turn ON steering the
                # operator turned off (010 pitfall 3) — only an explicitly
                # named profile may enable.
                logger.info(
                    "steering_intensity_noop",
                    reason="steering disabled; dial-only request will not enable it",
                    intensity=lam,
                )
                return None

            # The request dial is ABSOLUTE: it overrides the stored intensity
            # rather than multiplying it (010 pitfall 1). Field absent falls
            # back to the profile's stored λ.
            if lam is None:
                lam = profile.intensity if profile.intensity is not None else 1.0

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
                steering[idx] = clamp_steering(float(v) * lam)
        elif explicit and profile is not None:
            # Named profile with no steering values — nothing to override,
            # WITH or WITHOUT a dial (pre-010 semantics preserved; falling
            # through to the live-values base would steer the request by a
            # profile the caller never named).
            logger.info(
                "steering_intensity_noop",
                reason="named profile has no steering values",
                profile=profile.name,
                intensity=lam,
            )
            return None
        else:
            # Dial over live steering (no profile carries a base). Never
            # enable steering that wasn't already configured and enabled
            # (010 pitfall 3): with steering disabled or empty, a dial has
            # nothing to scale.
            if lam is None:
                return None
            live = sae.get_steering_values()
            if not live or not sae.is_steering_enabled:
                logger.info(
                    "steering_intensity_noop",
                    reason="no live steering to scale",
                    intensity=lam,
                )
                return None
            # Live values are the λ=1 base for this request.
            steering = {int(i): clamp_steering(float(v) * lam)
                        for i, v in live.items()}

        # Save the state we are about to overwrite
        saved = {
            "values": sae.get_steering_values(),
            "enabled": sae.is_steering_enabled,
        }

        sae.set_steering_batch(steering)
        sae.enable_steering(True)

        logger.info(
            "request_steering_applied",
            profile=profile.name if profile else None,
            intensity=lam,
            features=len(steering),
        )
        return saved

    def _restore_request_profile(self, saved: Optional[dict]) -> None:
        """
        Restore SAE steering to the state it was in before this request's
        profile override.  Always called in a finally block.

        If saved is None (apply_request_profile found nothing to override)
        this is a no-op.
        """
        if saved is None:
            return
        try:
            from millm.services.sae_service import AttachedSAEState

            sae = AttachedSAEState().attached_sae
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
        kwargs["pad_token_id"] = (
            self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        )
        kwargs["eos_token_id"] = self._tokenizer.eos_token_id

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
        prompt = self._format_chat_messages(request.messages)
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
                    request.profile, request.steering_intensity
                )

            try:
                # Tokenize input
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._get_input_device())
                prompt_tokens = inputs.input_ids.shape[1]

                # Build generation config
                gen_config = GenerationConfig.from_request(request)
                self._check_context_length(prompt_tokens, gen_config.max_new_tokens)

                for i in range(n):
                    # Generate - offload to thread to avoid blocking the event loop
                    generate_kwargs = self._build_generate_kwargs(gen_config, inputs)

                    outputs = await asyncio.to_thread(
                        self._generate_sync, generate_kwargs
                    )

                    # Notify monitoring after generation
                    self._notify_monitoring(request_id=completion_id)

                    # Decode output
                    generated_ids = outputs[0][prompt_tokens:]
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
                            message=ChatMessage(role="assistant", content=completion_text),
                            finish_reason=finish_reason,
                        )
                    )

                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens

            finally:
                # Restore steering to its pre-request state regardless of success/failure.
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
        prompt = self._format_chat_messages(request.messages)

        async with self._request_queue.acquire():
            # Per-request profile override (same logic as non-streaming path)
            _saved_steering = None
            if request.profile or request.steering_intensity is not None:
                _saved_steering = await self._apply_request_steering(
                    request.profile, request.steering_intensity
                )

            # Setup runs BEFORE the try/finally below that restores the
            # per-request steering, so any exception in this window
            # (tokenization, the context-length check, thread start) must
            # restore-and-reraise here — otherwise the dial/profile override
            # leaks into the global steering state (review R1, top finding).
            try:
                # Tokenize
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._get_input_device())
                prompt_tokens = inputs["input_ids"].shape[1]

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

                # Start generation thread with error capture
                thread_error: list[Exception] = []
                thread = Thread(
                    target=self._generate_in_thread,
                    args=(generation_kwargs, thread_error),
                )
                thread.start()
            except BaseException:
                self._restore_request_profile(_saved_steering)
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

                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=token),
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
                # Restore steering to its pre-request state (Fix #1: steering race)
                self._restore_request_profile(_saved_steering)

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

            for i, prompt_text in enumerate(prompts):
                # Tokenize input
                inputs = self._tokenizer(prompt_text, return_tensors="pt").to(
                    self._get_input_device()
                )
                prompt_tokens = inputs.input_ids.shape[1]
                self._check_context_length(prompt_tokens, gen_config.max_new_tokens)

                # Generate - offload to thread to avoid blocking the event loop
                generate_kwargs = self._build_generate_kwargs(
                    gen_config, inputs
                )
                outputs = await asyncio.to_thread(
                    self._generate_sync, generate_kwargs
                )

                # Notify monitoring after generation
                self._notify_monitoring(request_id=completion_id)

                # Decode output
                generated_ids = outputs[0][prompt_tokens:]
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

        prompt = self._format_chat_messages(request.messages)
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
                    message=ChatMessage(role="assistant", content=text),
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

        prompt = self._format_chat_messages(request.messages)
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        gen_config = GenerationConfig.from_request(request)

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
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=text),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

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

    def _format_chat_messages(self, messages: list[ChatMessage]) -> str:
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

        # Prefer model's built-in chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                # Check if chat_template is actually set
                if self._tokenizer.chat_template:
                    formatted = self._tokenizer.apply_chat_template(
                        [{"role": m.role, "content": m.content} for m in messages],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    logger.debug(
                        "formatted_prompt",
                        length=len(formatted),
                        preview=formatted[:500],
                    )
                    return formatted
            except Exception as e:
                logger.warning(
                    "chat_template_failed_using_fallback", error=str(e)
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
