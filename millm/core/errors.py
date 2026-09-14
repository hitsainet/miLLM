"""
Custom exception hierarchy for miLLM.

All application errors inherit from MiLLMError, which provides
consistent error codes and HTTP status codes for API responses.
"""

from typing import Any, Optional


class MiLLMError(Exception):
    """Base exception for all miLLM errors."""

    code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


# =============================================================================
# Model Errors
# =============================================================================


class SensingEventNotFoundError(MiLLMError):
    """Sensing event does not exist (pruned, cleared, or never existed)."""

    code = "SENSING_EVENT_NOT_FOUND"
    status_code = 404


class EngineUnsupportedError(MiLLMError):
    """The resident inference engine cannot perform the requested operation.

    A LIMIT OF THE RUNTIME, not of the request: llama.cpp exposes no PyTorch
    module tree, so streaming, embeddings, batched conversations, steering and
    chat_template_kwargs are refused rather than quietly degraded.

    Declared as a class, because `code` and `status_code` on MiLLMError are
    CLASS attributes — `MiLLMError(msg, code=..., status_code=...)` is a
    TypeError, and every refusal raised that way surfaced as a 500 instead of
    the 400 it meant to be.
    """

    code = "ENGINE_UNSUPPORTED"
    status_code = 400


class ContextLengthExceededError(MiLLMError):
    """The request does not fit in the model's context window.

    A LIMIT OF THE REQUEST, not a server fault, and the distinction is not
    cosmetic: llama.cpp raises a bare ValueError for this, which reached the
    client as a 500 "An internal server error occurred". A 500 means "try
    again" — miStudio's labeling run duly retried each oversized prompt three
    times, burning a model call each time on a request that could never
    succeed, and reported nothing an operator could act on.

    OpenAI returns 400 `context_length_exceeded` here, and clients know it.
    """

    code = "CONTEXT_LENGTH_EXCEEDED"
    status_code = 400


class AmbiguousModelNameError(MiLLMError):
    """A bare repo name matches several quantizations.

    Since a repository's quantizations can coexist, `foo-GGUF` may be any of
    `foo-GGUF:Q4_K_M`, `foo-GGUF:IQ4_XS`… Picking one silently would make the
    served model depend on insertion order, and nothing on the wire would say
    which answered. The caller is told which tags exist instead.
    """

    code = "AMBIGUOUS_MODEL_NAME"
    status_code = 400


class ModelNotFoundError(MiLLMError):
    """Raised when a requested model does not exist."""

    code = "MODEL_NOT_FOUND"
    status_code = 404


class ModelAlreadyExistsError(MiLLMError):
    """Raised when attempting to create a model that already exists."""

    code = "MODEL_ALREADY_EXISTS"
    status_code = 409


class ModelLoadError(MiLLMError):
    """Raised when model loading fails."""

    code = "MODEL_LOAD_FAILED"
    status_code = 500


class GgufTensorSplitError(ModelLoadError):
    """GGUF_TENSOR_SPLIT does not name one proportion per card a GGUF split uses.

    A configuration error, not a failed load, and only the setting fixes it.
    Raised as MODEL_LOAD_FAILED, the Admin UI replaced its message with that
    code's generic text — "Please check that you have sufficient VRAM available"
    — and the refusal round 1 moved before the unload read as a memory problem.
    A subclass of ModelLoadError, so every handler of that still catches it.
    Review round 2, 2026-09-14.
    """

    code = "INVALID_GGUF_TENSOR_SPLIT"
    status_code = 500


class UnsupportedQuantizationError(MiLLMError):
    """Raised when a load asks for a quantization its engine cannot apply.

    Q2 on a transformers checkpoint that is not already quantized: bitsandbytes
    has no 2-bit mode, so the load ran unquantized in bfloat16 — eight times the
    memory its 2-bit estimate planned for, under a label that still said Q2.
    A wrong request, not a failed load, so it is a 400.
    """

    code = "UNSUPPORTED_QUANTIZATION"
    status_code = 400


class ModelNotLoadedError(MiLLMError):
    """Raised when operation requires a loaded model but none is loaded."""

    code = "MODEL_NOT_LOADED"
    status_code = 400


class ModelAlreadyLoadedError(MiLLMError):
    """Raised when attempting to load a model that is already loaded."""

    code = "MODEL_ALREADY_LOADED"
    status_code = 400


class ModelBusyError(MiLLMError):
    """Raised when model is busy with another operation.

    On /v1 it is 503 server_error `model_busy`: the request is fine and succeeds
    once the other operation — a load, or an unload of the model it names —
    finishes, so a client is told to retry, not to change it. The management API
    answers 409.
    """

    code = "MODEL_BUSY"
    status_code = 409
    #: The OpenAI envelope's `type` (exception_handlers and a stream's error event read it).
    openai_error_type = "server_error"


class ModelLockedError(MiLLMError):
    """Raised when a model operation is blocked because a model is locked for steering."""

    code = "MODEL_LOCKED"
    status_code = 409


# =============================================================================
# Resource Errors
# =============================================================================


class InsufficientMemoryError(MiLLMError):
    """Raised when there's not enough GPU memory."""

    code = "INSUFFICIENT_MEMORY"
    status_code = 507


class GenerationOutOfMemoryError(InsufficientMemoryError):
    """A request ran a card out of memory while generating.

    Raised in place of torch's OutOfMemoryError, which reached a non-streaming
    client as a bare 500 "An internal server error occurred." and left a
    streaming one waiting forever (review round 6, 2026-09-14). The load's
    per-card fit keeps each card room for a KV cache at TRANSFORMERS_MIN_CONTEXT;
    a longer prompt, a larger batch or another tenant on the card can still
    exhaust it. That is the size of the request, not a fault to retry unchanged,
    so /v1 types it invalid_request_error (status 503, INSUFFICIENT_MEMORY's
    row); the management API answers 507.
    """

    #: The OpenAI envelope's `type` for this error (exception_handlers and the
    #: streaming error event both read it).
    openai_error_type = "invalid_request_error"


class GpuNotFoundError(MiLLMError):
    """Raised when a load names a GPU (index or UUID) that is not visible.

    Separate from InsufficientMemoryError: a card that does not exist is a
    wrong request, not a full card, and the fix is different.
    """

    code = "GPU_NOT_FOUND"
    status_code = 404


class SplitNotHonouredError(MiLLMError):
    """A split across every GPU ("all") would leave a card with none of the model.

    transformers fills the cards of a split in index order with whole layers and
    keeps room for the largest layer free on the lowest-index card, so a card's
    share can hold nothing, or the first card can hold everything. The load would
    then record "all" and run on fewer cards. Measured on transformers' own map
    inference (review round 3, 2026-09-14): Qwen2.5-7B at Q4 on both cards idle
    mapped entirely onto the 3090, and gemma-3-1b at FP16 with the 3080 Ti busy
    entirely onto the 3080 Ti. "all" is honoured or refused, never swapped, so it
    is refused — before anything is unloaded.

    Not an INSUFFICIENT_MEMORY: the cards may have room to spare; it is the
    request that cannot be met as asked.

    Also raised (review round 4, 2026-09-14) when a visible card is too full to
    take any share — the plan used to leave it out and split over the rest — and
    by the load itself when the preflight could not compute the map ahead and the
    model landed on fewer cards (`details.before_loading` False).
    """

    code = "SPLIT_NOT_HONOURED"
    status_code = 409


class InsufficientDiskError(MiLLMError):
    """Raised when there's not enough disk space."""

    code = "INSUFFICIENT_DISK"
    status_code = 507


# =============================================================================
# Download Errors
# =============================================================================


class DownloadFailedError(MiLLMError):
    """Raised when model download fails."""

    code = "DOWNLOAD_FAILED"
    status_code = 502


class DownloadCancelledError(MiLLMError):
    """Raised when download is cancelled by user."""

    code = "DOWNLOAD_CANCELLED"
    status_code = 499  # Client Closed Request


class RepoNotFoundError(MiLLMError):
    """Raised when HuggingFace repository is not found."""

    code = "REPO_NOT_FOUND"
    status_code = 404


class GatedModelError(MiLLMError):
    """Raised when accessing a gated model without proper authentication."""

    code = "GATED_MODEL_NO_TOKEN"
    status_code = 401


class InvalidTokenError(MiLLMError):
    """Raised when HuggingFace token is invalid."""

    code = "INVALID_HF_TOKEN"
    status_code = 401


# =============================================================================
# Path Errors
# =============================================================================


class InvalidLocalPathError(MiLLMError):
    """Raised when local path is invalid or doesn't exist."""

    code = "INVALID_LOCAL_PATH"
    status_code = 400


# =============================================================================
# SAE Errors
# =============================================================================


class SAENotFoundError(MiLLMError):
    """Raised when a requested SAE does not exist."""

    code = "SAE_NOT_FOUND"
    status_code = 404


class SAENotAttachedError(MiLLMError):
    """Raised when operation requires an attached SAE but none is attached."""

    code = "SAE_NOT_ATTACHED"
    status_code = 400


class SAEAlreadyAttachedError(MiLLMError):
    """Raised when attempting to attach an SAE when one is already attached."""

    code = "SAE_ALREADY_ATTACHED"
    status_code = 409


class SAEIncompatibleError(MiLLMError):
    """Raised when SAE is incompatible with the loaded model."""

    code = "SAE_INCOMPATIBLE"
    status_code = 400


class SAELoadError(MiLLMError):
    """Raised when SAE loading fails."""

    code = "SAE_LOAD_FAILED"
    status_code = 500


class InvalidFeatureIndexError(MiLLMError):
    """Raised when a feature index is outside the SAE's valid range [0, d_sae)."""

    code = "INVALID_FEATURE_INDEX"
    status_code = 400


class SAESetIncompleteError(MiLLMError):
    """A circuit member's layer has no (unique) attached SAE.

    Feature 12: a cross-layer circuit is only serveable when every member's
    layer resolves to exactly one attached SAE. If any member's layer has no
    attached SAE — or an ambiguous one, or the member index is out of that
    layer's range — serving is refused rather than steering through the wrong
    basis. The offenders list names each ``{feature_idx, layer, sae_id?,
    reason?}`` so the caller can fall back to the per-layer cluster slice.
    """

    code = "SAE_SET_INCOMPLETE"
    status_code = 422

    def __init__(self, offenders: list[dict[str, Any]]) -> None:
        self.offenders = offenders
        super().__init__(
            f"SAE set incomplete: {len(offenders)} member(s) have no attached "
            f"SAE for their layer",
            details={"offenders": offenders},
        )


# =============================================================================
# Profile Errors
# =============================================================================


class ProfileNotFoundError(MiLLMError):
    """Raised when a requested profile does not exist."""

    code = "PROFILE_NOT_FOUND"
    status_code = 404


class ProfileAlreadyExistsError(MiLLMError):
    """Raised when attempting to create a profile that already exists."""

    code = "PROFILE_ALREADY_EXISTS"
    status_code = 409


class ProfileCompatibilityError(MiLLMError):
    """Raised when profile is incompatible with current configuration."""

    code = "PROFILE_INCOMPATIBLE"
    status_code = 400


class InvalidProfileFormatError(MiLLMError):
    """Raised when profile import format is invalid."""

    code = "INVALID_PROFILE_FORMAT"
    status_code = 400


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(MiLLMError):
    """Raised when request validation fails."""

    code = "VALIDATION_ERROR"
    status_code = 422


# =============================================================================
# Circuit Errors (Feature 13)
# =============================================================================


class CircuitNotFoundError(MiLLMError):
    """Raised when a requested circuit does not exist."""

    code = "CIRCUIT_NOT_FOUND"
    status_code = 404


class UnvalidatedCircuitError(MiLLMError):
    """Activating a circuit whose evidence rung is below CAUSALLY_VALIDATED (2)
    without an explicit acknowledgement.

    The evidence ladder forbids describing such a circuit as causal; steering
    live traffic with one is allowed, but only deliberately — the caller must
    re-send with ``acknowledge_unvalidated=true``.
    """

    code = "UNVALIDATED_CIRCUIT"
    status_code = 200  # house style: handler-level refusal in the envelope


#: The measurement behind the default refusal. Carried IN the refusal payload
#: because §6.2 of the contention model makes it a binding retention condition:
#: an operator who overrides has been told what happened last time. The caveat
#: is part of the data, not a footnote — it is one model and one fixture, and
#: stating it as more would be the same overclaim the evidence ladder exists to
#: prevent.
CONTENTION_MEASURED_HAZARD: dict[str, Any] = {
    "source": "GPU close-out 2026-07-20, LFM2.5-1.2B-Instruct",
    "one_layer_at_strength_5": "coherent, indistinguishable from baseline",
    "two_layers_at_strength_5": "degenerate output (repeated tokens)",
    "note": "one model, one fixture — indicative, not exhaustive",
}


class CircuitLayerContentionError(MiLLMError):
    """Activating a circuit whose layers another active circuit already holds.

    Refused BY DEFAULT rather than composed, because composition on a layer is
    additive and unbounded in aggregate: the ±200 clamp bounds each member
    individually and nothing bounds the sum. The GPU close-out measured two
    steered layers at strength 5 destroying generation entirely, two orders of
    magnitude below that clamp.

    The refusal NAMES THE INCUMBENT so the operator's next action is obvious
    (deactivate it, or edit one circuit's layers), and carries the measurement
    so an override is an informed act rather than a guess. A refusal that
    states only the fact of contention does not satisfy BR-011.

    A same-key COLLISION uses this same code but is never overridable — see
    `colliding_keys` in the details.
    """

    code = "CIRCUIT_LAYER_CONTENTION"
    status_code = 200  # house style: handler-level refusal in the envelope

    def __init__(
        self,
        *,
        contended_layers: Any,
        incumbent_id: Optional[str] = None,
        incumbent_name: Optional[str] = None,
        requested_id: Optional[str] = None,
        requested_name: Optional[str] = None,
        colliding_keys: Any = (),
        all_incumbents: Any = (),
        detail: Optional[str] = None,
    ) -> None:
        layers = sorted(contended_layers or [])
        who = f"circuit '{incumbent_name}'" if incumbent_name else "another active circuit"
        if incumbent_id:
            who += f" ({incumbent_id})"

        if colliding_keys:
            pairs = ", ".join(
                f"L{layer}/feature {idx}" for layer, idx, _cid in colliding_keys
            )
            message = (
                f"{pairs} are steered by BOTH this circuit and {who}. "
                "Composition merges into one steering dict, so one strength "
                "would silently overwrite the other and the served value would "
                "belong to neither author. This cannot be overridden — edit "
                "one circuit's members."
            )
        else:
            message = (
                f"Layers {layers} are already served by {who}. Overriding "
                "composes both circuits additively on those layers. In "
                "close-out testing, TWO steered layers at individually-harmless "
                "strength (5) destroyed generation entirely — two orders of "
                "magnitude below the per-member clamp. Pass "
                "allow_layer_overlap=true only if you intend a compounding "
                "study; the circuit-rung header is omitted while any layer is "
                "composed, because no single circuit's evidence describes the "
                "response."
            )
        if detail:
            message = f"{message} ({detail})"

        super().__init__(
            message,
            details={
                "contended_layers": layers,
                "incumbent": {"id": incumbent_id, "name": incumbent_name},
                "requested": {"id": requested_id, "name": requested_name},
                # Absent for a collision: naming an override parameter that
                # cannot help would be an invitation to try it.
                **(
                    {}
                    if colliding_keys
                    else {
                        "override_param": "allow_layer_overlap",
                        "rung_header_suppressed_if_overridden": True,
                    }
                ),
                "colliding_keys": [
                    {"layer": layer, "feature_idx": idx, "incumbent": cid}
                    for layer, idx, cid in (colliding_keys or ())
                ],
                # R2-12: every incumbent, not just the one the dialog can
                # offer to deactivate. With two circuits holding two contended
                # layers, naming one sent the operator to deactivate it, retry,
                # and be refused again with no hint the second existed.
                "all_incumbents": list(all_incumbents or []),
                "overridable": not bool(colliding_keys),
                "measured_hazard": CONTENTION_MEASURED_HAZARD,
            },
        )


class NoActiveCircuitError(MiLLMError):
    """An operation needing an active circuit was called with none serving."""

    code = "NO_ACTIVE_CIRCUIT"
    status_code = 200  # house style: handler-level refusal in the envelope


class CircuitSensingEventNotFoundError(MiLLMError):
    """A circuit edge sensing event id that does not exist (Feature 15)."""

    code = "CIRCUIT_SENSING_EVENT_NOT_FOUND"
    status_code = 404
