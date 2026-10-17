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
    #: Response headers the handler adds to the error response (both envelopes). A class
    #: attribute, like `code`: `WEIGHTS_UNAUTHORIZED` carries `WWW-Authenticate: Bearer`.
    headers: Optional[dict[str, str]] = None

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


class InvalidScoringRequestError(MiLLMError):
    """A scoring-mode request (`logprobs` / `allowed_token_ids`) the loaded model cannot answer —
    e.g. a token id outside its vocabulary. A fault of the request, so 400, never a 500."""

    code = "INVALID_SCORING_REQUEST"
    status_code = 400


# =============================================================================
# Request validation, chat scoring and structured output (Feature 25)
#
# Every one of these answers a request that miLLM would otherwise have served with a
# field silently dropped. `details["param"]` names the field; the /v1 handler copies
# it into the OpenAI envelope's `param`.
# =============================================================================


class InvalidParameterError(MiLLMError):
    """A request parameter or header with a value miLLM does not accept, e.g. an
    `X-miLLM-Strict` value other than true/1/false/0 (FR-25.2.2). The code has had an
    ERROR_STATUS_MAP row for a long time; this is its first class."""

    code = "INVALID_PARAMETER"
    status_code = 400


class FieldNotHonouredError(MiLLMError):
    """An output-changing field this endpoint and engine cannot honour (FR-25.3).

    Raised whether or not the client asked for strict mode: ignoring such a field
    returns a 200 whose output is not the output the request described."""

    code = "FIELD_NOT_HONOURED"
    status_code = 400


class UnusedFieldsRefusedError(MiLLMError):
    """`X-miLLM-Strict: true` and at least one field the request path would not use
    (FR-25.2). `details["fields"]` lists every location, not only the first."""

    code = "UNUSED_FIELDS_REFUSED"
    status_code = 400


class ResponseFormatUnsupportedError(MiLLMError):
    """`response_format` cannot be honoured here: a schema keyword outside the declared
    subset, a GGUF model, the continuous batching manager, or a combination (stop,
    stream, scoring) that would report truncated JSON as complete (FR-25.11)."""

    code = "RESPONSE_FORMAT_UNSUPPORTED"
    status_code = 400


class NoChatTemplateError(MiLLMError):
    """Chat scoring on a model without a chat template (T-55, FR-25.5.9).

    Generation keeps its generic fallback format; scoring refuses, because a score of a
    prompt the model was never trained on looks exactly like a real score."""

    code = "NO_CHAT_TEMPLATE"
    status_code = 400


class ConstrainedOutputInvalidError(MiLLMError):
    """A COMPLETE constrained generation failed miLLM's own validation (FR-25.10.6).

    A server fault, never the client's: the constraint should have made it impossible.
    Reported as an error instead of a 200 carrying invalid JSON."""

    code = "CONSTRAINED_OUTPUT_INVALID"
    status_code = 500


class ScoringNumericalError(MiLLMError):
    """The model produced NaN or infinite logits, so no probability can be reported. A fault of the
    model or its weights, not of the request — and reported, never serialised as `NaN`."""

    code = "NON_FINITE_LOGITS"
    status_code = 500


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


class EmbeddingInputTooLongError(ContextLengthExceededError):
    """One or more `/v1/embeddings` inputs exceed the model's limit (Feature 30, FR-30.3).

    Inputs are NEVER truncated. Before Feature 30 the tokenizer was called with
    `truncation=True`, so an over-long input was embedded from its first N tokens and returned
    a 200 for text the caller never sent whole. Every input is measured before any forward pass
    and the refusal names each over-limit index. `details["param"]` is `input[i]` for the first
    (or `input` for a string input); the live handler forwards it as the OpenAI `param`.
    """


class EmbeddingVectorInvalidError(MiLLMError):
    """A pooled embedding vector is non-finite, or has a zero or non-finite norm under
    `normalize` (FR-30.2.6). The request was valid, so this is a 500, and it names the input's
    index rather than return NaN."""

    code = "EMBEDDING_VECTOR_INVALID"
    status_code = 500


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
# Model lease (Feature 29)
#
# A lease pins the resident model for a named holder until its TTL. It sits BESIDE
# `locked` (checkpoint decision C8): ModelLeasedError must never subclass
# ModelLockedError, because the /v1 routes catch ModelLockedError first and would
# answer `model_locked` — a code that names no holder and no expiry.
# =============================================================================


class ModelLeasedError(MiLLMError):
    """A load, unload or swap refused because another holder leases the resident model.

    `details` names holder, reason, `expires_at` and the leased model so the caller knows
    who to wait for and until when. It never carries the lease ID, its digest or its ref.
    """

    code = "MODEL_LEASED"
    status_code = 409

    @classmethod
    def for_lease(
        cls,
        *,
        model_id: int,
        model_name: str,
        holder: str,
        reason: str,
        expires_at: str,
        operation: str,
        target_model_id: Optional[int],
    ) -> "ModelLeasedError":
        """The refusal for `operation` on `target_model_id` under the given live lease."""
        message = (
            f"Model '{model_name}' is leased by '{holder}' until {expires_at} ({reason}); "
            f"{operation} of model {target_model_id} refused."
        )
        return cls(
            message,
            details={
                "holder": holder,
                "reason": reason,
                "expires_at": expires_at,
                "leased_model_id": model_id,
                "leased_model_name": model_name,
                "operation": operation,
                "target_model_id": target_model_id,
            },
        )


class ModelNotResidentError(MiLLMError):
    """A lease asked for a model that is not the resident, LOADED one (T-85)."""

    code = "MODEL_NOT_RESIDENT"
    status_code = 409


class LeaseNotFoundError(MiLLMError):
    """An unknown lease ID, or one that belongs to a different model than the route names."""

    code = "LEASE_NOT_FOUND"
    status_code = 404


class LeaseExpiredError(MiLLMError):
    """A lease ID this process has seen, which has ended (`details.end_reason`)."""

    code = "LEASE_EXPIRED"
    status_code = 409


class InvalidLeaseRequestError(MiLLMError):
    """A lease request field outside its bounds; `details.param` names it and the limit."""

    code = "INVALID_LEASE_REQUEST"
    status_code = 400


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
        overridable: Optional[bool] = None,
    ) -> None:
        layers = sorted(contended_layers or [])
        # A collision is never overridable; neither is contention while CIRCUIT_ALLOW_CONCURRENT
        # is off, whose caller passes False. The Circuits page offers "Compose anyway" from this
        # field alone, so a flag-off refusal saying `true` offered a button the server refused
        # again (found on hardware 2026-10-16).
        if overridable is None:
            overridable = not bool(colliding_keys)
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
                    if not overridable
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
                "overridable": bool(overridable),
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

# ── Probe monitors (Feature 24) ───────────────────────────────────────────────
#
# ⚠ PROBES REFUSE WHERE CIRCUITS BIND. A circuit may be bound to a model whose identity does not
# match, because a circuit is an authored artifact a human can reason about. A probe cannot: its
# weights are a direction in one specific model's residual space, and read in another model's
# space they produce numbers that are plausible, stable, and about nothing.


class ProbeNotFoundError(MiLLMError):
    """No probe with that id."""

    code = "PROBE_NOT_FOUND"
    status_code = 404


class ProbeModelMismatchError(MiLLMError):
    """The definition was fitted on a different model than the one loaded.

    ``details["mismatches"]`` names every field that differs — all of them, not the first — so an
    operator sees whether they loaded the wrong model or imported the wrong probe.
    """

    code = "PROBE_MODEL_MISMATCH"
    status_code = 409


class ProbeDtypeMismatchError(ProbeModelMismatchError):
    """The probe was fitted at a different PRECISION than this server loaded the model at.

    Same model, same layer, same weights — and still a different distribution: the residual
    stream a probe reads depends on the precision the model ran at. Re-scoring a float16-fitted
    probe at bfloat16 moved its combined scores by up to 0.25 (2026-10-03), more than parity's
    whole tolerance. Split out from `ProbeModelMismatchError` because "fitted on a different model"
    is the wrong sentence for it and points an operator at the wrong fix.

    A subclass, so every handler of a model mismatch still catches it.
    """

    code = "PROBE_DTYPE_MISMATCH"
    status_code = 409


class ProbeScopeUnverifiableError(MiLLMError):
    """The probe's scope cannot be checked against its recorded scores at all.

    ⚠ **NOT THE SAME AS FAILING PARITY, AND SAYING SO MATTERS.** Only `scope: all` is exactly
    reproducible. For `prompt` and `response`, miStudio scored under a narrower internal role
    mask and `mistudio.probe-definition/v1` carries no record of WHICH positions those were —
    only the token ids. So nothing can be compared, and the probe is refused rather than armed
    unverified.

    This used to surface as `PROBE_PARITY_FAILED` — *"this build does not reproduce the scores
    miStudio recorded"* — over a report where **zero of sixteen vectors were comparable and zero
    tokens were scored**. An operator reading that goes and debugs their build, their model and
    their precision, none of which is involved. This estate already has the rule, written after
    the mirror-image defect: a parity check that reports "incorrect" against a correct
    implementation is worse than none, because it is believed the first time.

    The fix is on the PRODUCER side — export the probe with `scope: all`, or extend the contract
    to record the mask — so the message says that instead of implying a numerical disagreement.
    """

    code = "PROBE_SCOPE_UNVERIFIABLE"
    status_code = 409


class ProbeParityFailedError(MiLLMError):
    """This build does not reproduce the scores miStudio recorded for the definition's vectors.

    Refusing to arm is the whole point: the alternative is a probe that scores subtly differently
    from the one whose AUROC was measured, reporting under that measurement's authority.
    """

    code = "PROBE_PARITY_FAILED"
    status_code = 409


class UnvalidatedProbeError(MiLLMError):
    """Arming below rung 2 without an explicit acknowledgement.

    Returned in the envelope with a 200, following the circuit house style: this is a refusal the
    caller can resolve by asserting intent, not a malformed request.
    """

    code = "UNVALIDATED_PROBE"
    status_code = 200


class ProbeLimitError(MiLLMError):
    """More probes armed than ``PROBE_MAX_ARMED`` allows."""

    code = "PROBE_LIMIT"
    status_code = 409


class ProbeSaeMissingError(MiLLMError):
    """A k-sparse probe's SAE is not downloaded here. Names the repo and path to fetch."""

    code = "PROBE_SAE_MISSING"
    status_code = 409


class ProbeSaeMismatchError(MiLLMError):
    """The downloaded SAE is not the one the probe was fitted against."""

    code = "PROBE_SAE_MISMATCH"
    status_code = 409


class ProbeNoModelLoadedError(MiLLMError):
    """Nothing is loaded, or its width and depth cannot be read from its config.

    ⚠ Raised rather than defaulted. A fabricated ``d_model`` compares cleanly against a
    definition's model block, so every identity gate would report a match while the probe read a
    residual space it was never fitted in.
    """

    code = "PROBE_NO_MODEL_LOADED"
    status_code = 409


class ProbeRecalibrationMismatchError(MiLLMError):
    """The incoming cut does not describe the probe whose bar it would move.

    ⚠ THE WHOLE CARVE-OUT RESTS ON THIS REFUSAL. Moving a bar in place is permitted only because
    the detector underneath is provably unchanged, and `provenance.probe_id` plus
    `provenance.run_id` are what prove it. A definition carrying neither cannot be verified, so it
    refuses rather than defaulting to yes — that probe goes disarm -> delete -> import like any
    other rebuild.
    """

    code = "PROBE_RECALIBRATION_MISMATCH"
    status_code = 409


class ProbeThresholdUncalibratedError(MiLLMError):
    """A bar nobody calibrated, or one the runtime's own parser would discard.

    Two cases, one refusal. A `threshold` with no `target_fpr` and no `threshold_source` is an
    opinion wearing a measurement's clothes, and every surface downstream would present it as the
    latter. A per-window or per-length entry the arming parsers DROP would mean a 200 that moved
    no bar — the import-time tolerance that is correct there and inverted here.
    """

    code = "PROBE_THRESHOLD_UNCALIBRATED"
    status_code = 409


class ProbeHookUnsupportedError(MiLLMError):
    """The loaded runtime exposes no module tree to hook.

    A statement of fact about llama.cpp, not a policy: ``register_forward_hook`` has nothing to
    attach to and no per-layer residual tensor is reachable from Python.
    """

    code = "PROBE_HOOK_UNSUPPORTED"
    status_code = 409


class ProbeScoreRequestError(MiLLMError):
    """A `POST /api/probes/score` request this server refuses as a whole, before any forward:
    no inputs, too many inputs or probes, an input with two kinds, `prompt_tokens` past the input,
    a token id outside the vocabulary, every probe skipped, or `text` while the T-49 switch
    (`probe_scoring.TEXT_INPUT_VERIFIED`) is off (Feature 27, FR-27.4). A fault of the request,
    so 400."""

    code = "INVALID_PROBE_SCORE_REQUEST"
    status_code = 400


class SaeActivationsRefusedError(MiLLMError):
    """A `return_sae_activations` request refused before generation (Feature 27, FR-27.1i,
    FR-27.2): a shape with no single position axis (`n > 1`, `extra_messages`, several prompts),
    an ambiguous SAE, a feature index past the SAE's width, `top_k` over the cap, or a worst-case
    entry count over the cap. Refused rather than answered partly — 400, never after generating."""

    code = "SAE_ACTIVATIONS_REFUSED"
    status_code = 400


# =============================================================================
# Batch API (Feature 26)
#
# Every refusal names the field, limit, model or batch that caused it (FPRD §4). On `/v1` they
# render through the OpenAI envelope with `param` from `details["param"]`.
# =============================================================================


class InvalidBatchRequestError(MiLLMError):
    """A files/batches request field outside its bounds (purpose, endpoint, window, …)."""

    code = "INVALID_BATCH_REQUEST"
    status_code = 400


class BatchFileLimitError(MiLLMError):
    """An upload over BATCH_MAX_FILE_BYTES or BATCH_MAX_ROWS (FR-26.8.2). Nothing is stored."""

    code = "BATCH_FILE_LIMIT"
    status_code = 400


class BatchFileNotFoundError(MiLLMError):
    code = "FILE_NOT_FOUND"
    status_code = 404


class BatchFileExpiredError(MiLLMError):
    """The file's bytes were pruned by retention (FR-26.9.3); the record remains."""

    code = "FILE_EXPIRED"
    status_code = 404


class BatchFileDeletedError(MiLLMError):
    code = "FILE_DELETED"
    status_code = 404


class BatchFileInUseError(MiLLMError):
    """DELETE of a file a non-terminal batch references (FR-26.6.10); names the batch."""

    code = "FILE_IN_USE"
    status_code = 409


class BatchNotFoundError(MiLLMError):
    code = "BATCH_NOT_FOUND"
    status_code = 404


class BatchStateConflictError(MiLLMError):
    """A transition the batch's current status does not allow (FR-26.3.5); names the status."""

    code = "BATCH_STATE_CONFLICT"
    status_code = 409


# =============================================================================
# Attachable weights (Features 31, 32, 33)
#
# Contract: 0xcc/reviews/attachable_weights_impl_addendum_2026-10-14.md §2.3. Every refusal
# names a reason or the figures in `details`, and NONE carries a token, an attachment ID or
# a path outside the checkpoint root: `millm_error_handler` logs `details` on every refusal.
# Helper constructors are classmethods, so every class keeps MiLLMError's (message, details).
# =============================================================================

# --- Feature 31: weight attachment and the step barrier ----------------------------------


class AlreadyAttachedError(MiLLMError):
    """An operation refused because an attachment (active or stale) holds the weights.

    On `/v1` it answers the auto-load refusal: 409 `invalid_request_error`, the caller must
    wait for the trainer to detach or ask an operator to clear a stale attachment.
    """

    code = "ALREADY_ATTACHED"
    status_code = 409

    @classmethod
    def for_attachment(
        cls,
        *,
        role: str,
        label: str,
        state: str,
        attached_at: str,
        operation: str,
    ) -> "AlreadyAttachedError":
        """The refusal for `operation` while `label` (role `role`) holds an attachment."""
        return cls(
            f"The resident weights are attached by '{label}' ({role}, {state}) since "
            f"{attached_at}; {operation} refused until it detaches.",
            details={
                "role": role,
                "label": label,
                "state": state,
                "attached_at": attached_at,
                "operation": operation,
            },
        )


class NotAttachableError(MiLLMError):
    """The resident model cannot be attached. `details.reason` is one of `gguf`, `quantized`,
    `dtype`, `not_on_gpu`, `continuous_batching_running`, `allocator` (not an attachable load,
    addendum D-03), `concurrency` (more than one admission slot, D-04), `model_transition`
    (a load, an unload or an internal write is in progress) or `weights_writing` (a commit left
    open for the watchdog by clear-stale, D-40); `details.parameter` names the first offending
    tensor where one exists."""

    code = "NOT_ATTACHABLE"
    status_code = 400

    @classmethod
    def for_reason(cls, reason: str, parameter: Optional[str] = None) -> "NotAttachableError":
        message = f"The resident model cannot be attached: {reason}"
        if parameter:
            message += f" (first offending parameter: {parameter})"
        details: dict[str, Any] = {"reason": reason}
        if parameter:
            details["parameter"] = parameter
        return cls(message + ".", details=details)


class WeightsNotCleanError(MiLLMError):
    """An operation that needs `weights_state == clean` (a save, a pause, an internal write)
    found a commit open (`writing`) or the weights `torn`. `details.weights_state` says which."""

    code = "WEIGHTS_NOT_CLEAN"
    status_code = 409


class WeightsTornError(MiLLMError):
    """The resident weights were left mid-write and are not served (R-01.23).

    503 everywhere, with `Retry-After: RETRY_AFTER_WEIGHTS_TORN_S`: a late `commit-end` from
    the writer, or a weights load, restores serving.
    """

    code = "WEIGHTS_TORN"
    status_code = 503


class WeightsUnsavedError(MiLLMError):
    """A `/v1` auto-load refused because the resident weights carry commits not saved to a
    checkpoint (addendum D-45, review R2-A F-A).

    "Unsaved" is derived, never stored: the barrier's `steps_since_load` minus the resident
    identity's `commits_at_identity` (a save sets that to the counter at the save; a load resets
    the counter). Swapping the model out would discard those commits on the strength of a name a
    client typed, so the auto-load is refused until a save records them or an explicit operator
    load (`POST /api/weights/load`, the management load) replaces them. `details`: `operation`,
    `resident` (the served name), `resident_model_id`, `unsaved_commits`, `requested_model`,
    `requested_model_id`, `recovery`.
    """

    code = "WEIGHTS_UNSAVED"
    status_code = 409


class WeightsUnauthorizedError(MiLLMError):
    """A gated weights call without a valid bearer token. `details.reason`: `missing`,
    `invalid_token`, `no_tokens_configured` or `tokens_malformed`. Never names the token."""

    code = "WEIGHTS_UNAUTHORIZED"
    status_code = 401
    headers = {"WWW-Authenticate": "Bearer"}

    #: Appended to the `no_tokens_configured` message on an adapter route (addendum D-07): an
    #: operator serving adapters may run no trainer at all, and still needs a token.
    ADAPTER_NO_TOKENS_GUIDANCE = (
        "Serving an adapter needs an operator token even with no trainer; see the configuration "
        "reference (WEIGHTS_TOKENS, manual/docs/reference/configuration.md)."
    )

    @classmethod
    def for_reason(
        cls, reason: str, operation: Optional[str] = None
    ) -> "WeightsUnauthorizedError":
        """The refusal for `reason`. `operation` (the route name, e.g. `adapter/load`) makes the
        `no_tokens_configured` guidance specific to the call that was refused."""
        messages = {
            "missing": "This weights operation needs an 'Authorization: Bearer <token>' header.",
            "invalid_token": "The bearer token is not one this deployment accepts.",
            "no_tokens_configured": (
                "No weights tokens are configured (WEIGHTS_TOKENS is empty), so every gated "
                "weights operation is refused. An operator must configure a trainer or "
                "operator token."
            ),
            "tokens_malformed": (
                "WEIGHTS_TOKENS is malformed, so every gated weights operation is refused. "
                "The server log names the faulty entry."
            ),
        }
        message = messages.get(reason, "Unauthorized.")
        details: dict[str, Any] = {"reason": reason}
        if (
            reason == "no_tokens_configured"
            and isinstance(operation, str)
            and operation.startswith("adapter/")
        ):
            message = f"{message} {cls.ADAPTER_NO_TOKENS_GUIDANCE}"
            # Also on its own: the management handler replaces `message` with the code's generic
            # sentence (`error_messages.ERROR_MESSAGES`), keeping this one only as
            # `technical_message`; `details` reaches the client whole.
            details["guidance"] = cls.ADAPTER_NO_TOKENS_GUIDANCE
        return cls(message, details=details)


class WeightsForbiddenError(MiLLMError):
    """A valid token whose role may not perform this call (`details.required_roles`)."""

    code = "WEIGHTS_FORBIDDEN"
    status_code = 403

    @classmethod
    def for_roles(cls, role: str, required: tuple[str, ...]) -> "WeightsForbiddenError":
        return cls(
            f"A '{role}' token may not perform this weights operation; it needs one of: "
            f"{', '.join(required)}.",
            details={"role": role, "required_roles": list(required)},
        )


class NotAttachmentHolderError(MiLLMError):
    """A holder-only call whose `X-miLLM-Attachment` is missing or names another attachment,
    or whose token label is not the label that attached."""

    code = "NOT_ATTACHMENT_HOLDER"
    status_code = 403


class NotAttachedError(MiLLMError):
    """A holder-only call while nothing is attached."""

    code = "NOT_ATTACHED"
    status_code = 409


class NotQuiescedError(MiLLMError):
    """`commit-begin` while the barrier is serving (or the holder's quiesce has not returned),
    or `resume` while the holder's quiesce is still draining in-flight work (addendum D-39,
    `details.reason` `quiesce_in_progress`). A `resume` while serving is a no-op, not this."""

    code = "NOT_QUIESCED"
    status_code = 409


class AlreadyQuiescedError(MiLLMError):
    """`quiesce` while the barrier already holds."""

    code = "ALREADY_QUIESCED"
    status_code = 409


class NoOpenCommitError(MiLLMError):
    """`commit-end` with no commit open. A `commit-end` from someone other than the commit's
    owner is refused `NOT_ATTACHMENT_HOLDER` instead (FR-31.12.2)."""

    code = "NO_OPEN_COMMIT"
    status_code = 409


class AttachmentNotStaleError(MiLLMError):
    """`clear-stale` against an attachment whose heartbeat is current."""

    code = "ATTACHMENT_NOT_STALE"
    status_code = 409


class PauseInProgressError(MiLLMError):
    """A serving pause (a save snapshot, an adapter load or unload) is in progress, so a
    `commit-begin`, a second pause or a concurrent save is refused. `details.operation` names
    the pause. Renamed from SAVE_IN_PROGRESS (addendum D-06); a 409, so no `Retry-After`."""

    code = "PAUSE_IN_PROGRESS"
    status_code = 409

    @classmethod
    def for_operation(cls, active: str, refused: str) -> "PauseInProgressError":
        return cls(
            f"A serving pause for '{active}' is in progress; {refused} refused. Retry once "
            "it finishes.",
            details={"operation": active, "refused_operation": refused},
        )


class WeightsBarrierUnavailableError(MiLLMError):
    """The step barrier cannot hold serving, so a pause or internal write is refused.

    `details.reason`: `cbm` (the continuous batching manager generates without the admission
    slot, addendum D-13) or `concurrency` (more than one admission slot, D-04). Both are
    deployment settings; a retry does not help.
    """

    code = "BARRIER_UNAVAILABLE"
    status_code = 409

    @classmethod
    def for_reason(cls, reason: str, operation: str) -> "WeightsBarrierUnavailableError":
        why = {
            "cbm": "continuous batching is enabled (ENABLE_CONTINUOUS_BATCHING) and generates "
            "without the admission slot",
            "concurrency": "MAX_CONCURRENT_REQUESTS is above 1, so holding one slot stops "
            "nothing",
        }.get(reason, reason)
        return cls(
            f"{operation} refused: the weights barrier cannot pause serving because {why}.",
            details={"reason": reason, "operation": operation},
        )


# --- Feature 32: checkpoints, load and lineage --------------------------------------------


class InvalidRunIdError(MiLLMError):
    """A `run_id` outside the save rule (strict) or the load rule (safe segment), or too long
    for the carrier row's name (addendum D-22). `details`: `run_id`, `pattern`."""

    code = "INVALID_RUN_ID"
    status_code = 400


class ManifestFieldNotCallersError(MiLLMError):
    """A save named manifest fields the writer owns (`details.fields`)."""

    code = "MANIFEST_FIELD_NOT_CALLERS"
    status_code = 400


class InvalidManifestError(MiLLMError):
    """A checkpoint manifest that fails the vendored schema (`details.errors[]` with
    `json_path` and `message`), or, with `details.reason = "schema_ownership"`, any 032 save or
    load while the vendored schema itself has a field whose owner cannot be derived (D-35)."""

    code = "INVALID_MANIFEST"
    status_code = 400


class CheckpointExistsError(MiLLMError):
    """The step directory already exists (`details.path`, relative to the root)."""

    code = "CHECKPOINT_EXISTS"
    status_code = 409


class CheckpointAmbiguousError(MiLLMError):
    """A `run_id` present under more than one checkpoint root (addendum D-21); `details.roots`."""

    code = "CHECKPOINT_AMBIGUOUS"
    status_code = 409


class InsufficientStorageError(MiLLMError):
    """Free disk below the checkpoint estimate times the headroom factor (`details`:
    `need_bytes`, `free_bytes`, `factor`), refused before quiescing."""

    code = "INSUFFICIENT_STORAGE"
    status_code = 507


class CheckpointWriteFailedError(MiLLMError):
    """Writing a checkpoint failed after the snapshot (`details`: `stage`, `error`)."""

    code = "CHECKPOINT_WRITE_FAILED"
    status_code = 500


class InvalidCheckpointPathError(MiLLMError):
    """A checkpoint path resolving outside every checkpoint root. `details.input` is what the
    caller sent, never the resolved outside path."""

    code = "INVALID_CHECKPOINT_PATH"
    status_code = 400


class CheckpointNotFoundError(MiLLMError):
    """No checkpoint by that name or path (`details.name` or `details.path`)."""

    code = "CHECKPOINT_NOT_FOUND"
    status_code = 404


class CheckpointHashMismatchError(MiLLMError):
    """The weights file's SHA-256 differs from the manifest (`expected`, `actual`) or changed
    after hashing (`changed_since_hash: true`). Refused before any write."""

    code = "CHECKPOINT_HASH_MISMATCH"
    status_code = 422


class CheckpointNotLoadableError(MiLLMError):
    """A verified checkpoint miLLM will not load as it stands. `details.reason`: `dtype` (weights
    not 16-bit, never cast silently, addendum D-25) or `remote_code` (a config with `auto_map`
    and no operator `trust_remote_code: true`, D-23)."""

    code = "CHECKPOINT_NOT_LOADABLE"
    status_code = 400


class WeightsLoadFailedError(MiLLMError):
    """A weights load failed after its first write: the weights are `torn` (`details`:
    `stage`, `error`, `weights_state`)."""

    code = "WEIGHTS_LOAD_FAILED"
    status_code = 500


class NotInstantiableError(MiLLMError):
    """Instantiate refused (`details.reason`). Instantiate is DEFERRED (addendum scope); the
    class exists so the code is reserved and mapped."""

    code = "NOT_INSTANTIABLE"
    status_code = 400


class ModelNameTakenError(MiLLMError):
    """A carrier row's name is held by another row (`details.name`)."""

    code = "MODEL_NAME_TAKEN"
    status_code = 409


class LocalPathRegisteredError(MiLLMError):
    """A checkpoint directory already registered as a plain LOCAL model row (addendum D-26):
    delete or re-register that row (`details.model_id`). Without it `uq_local_path` would 500."""

    code = "LOCAL_PATH_REGISTERED"
    status_code = 409


class LoadThroughWeightsApiError(MiLLMError):
    """A weights-origin row named by the management load or `/v1`; it loads only through
    `POST /api/weights/load`, which verifies its hash (`details`: `model_id`,
    `weights_origin`)."""

    code = "LOAD_THROUGH_WEIGHTS_API"
    status_code = 409


# --- Feature 33: adapter serving ------------------------------------------------------------


class AdapterNotServableError(MiLLMError):
    """The resident model cannot serve an adapter. `details.reason`: `engine`, `quantization`
    or `cbm` (continuous batching enabled, addendum D-13)."""

    code = "ADAPTER_NOT_SERVABLE"
    status_code = 400


class AdapterInvalidError(MiLLMError):
    """The adapter directory or its weights file is unusable. `details.reason`:

    - the path: `bad_path`, `not_found`, `outside_root`, `ambiguous` (in more than one root);
    - a file: `missing_file`, `not_regular_file`, `unreadable` (cannot be opened, not
      safetensors, or an OS error while preparing — `details.error` names the exception type),
      `too_large` (above the bound the targets' shapes allow, or grew while read);
    - the tensors (from `adapter_hooks`, with `offenders` and `first`): `unsupported_tensor` (a
      key that is not LoRA's A or B), `dtype`, `non_finite`, `empty`, `scaling`.

    The path is never echoed beyond the root."""

    code = "ADAPTER_INVALID"
    status_code = 400


class AdapterManifestInvalidError(MiLLMError):
    """`adapter_manifest.json` fails the vendored schema; each failing field is named."""

    code = "ADAPTER_MANIFEST_INVALID"
    status_code = 400


class AdapterHashMismatchError(MiLLMError):
    """The adapter weights' SHA-256 differs from the manifest (`expected`, `actual`)."""

    code = "ADAPTER_HASH_MISMATCH"
    status_code = 400


class AdapterBaseMismatchError(MiLLMError):
    """The manifest names another base (name or manifest hash); both identities in `details`."""

    code = "ADAPTER_BASE_MISMATCH"
    status_code = 400


class AdapterShapeMismatchError(MiLLMError):
    """The adapter does not fit the resident model. `details.reason`: `rank`, `shape`,
    `missing_module`, `unsupported_module` (not an unquantized `nn.Linear`),
    `fused_by_transformers`, `incomplete_pair`, `duplicate_target`, `invalid_target_entry`,
    `target_not_declared`, `target_unused` (a `target_modules` entry, or the whole list, matches
    no module). `details.offenders` lists every offender and `details.first` the first; a tensor
    outside LoRA's A and B is `ADAPTER_INVALID` (`unsupported_tensor`), not this."""

    code = "ADAPTER_SHAPE_MISMATCH"
    status_code = 400


class AdapterSchemaNotVendoredError(MiLLMError):
    """The adapter manifest schema is not vendored, so every load is refused."""

    code = "ADAPTER_SCHEMA_NOT_VENDORED"
    status_code = 409


class AdapterAlreadyLoadedError(MiLLMError):
    """Another adapter is applied (`details` name it); unload it first."""

    code = "ADAPTER_ALREADY_LOADED"
    status_code = 409


class NoAdapterLoadedError(MiLLMError):
    """Adapter unload with nothing applied."""

    code = "NO_ADAPTER_LOADED"
    status_code = 409


class AdapterBaseChangedError(MiLLMError):
    """The base was replaced between verification and the pause; nothing was installed."""

    code = "ADAPTER_BASE_CHANGED"
    status_code = 409


class AdapterInstallFailedError(MiLLMError):
    """Nothing was installed. `details.reason`: `install_failed` (a hook install failed partway;
    every installed hook was removed) or `prepare_failed` (an unexpected torch / safetensors
    error while placing the tensors, before the pause). `details.error` names the exception
    type, never its message."""

    code = "ADAPTER_INSTALL_FAILED"
    status_code = 500


class AdapterAppliedError(MiLLMError):
    """An operation that compares against the base (probe parity) refused while an adapter is
    applied (addendum D-14); `details.adapter` names it."""

    code = "ADAPTER_APPLIED"
    status_code = 409
