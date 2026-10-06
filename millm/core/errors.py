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
    a token id outside the vocabulary, every probe skipped, or `text` before its render is
    verified (Feature 27, FR-27.4, T-49). A fault of the request, so 400."""

    code = "INVALID_PROBE_SCORE_REQUEST"
    status_code = 400


class SaeActivationsRefusedError(MiLLMError):
    """A `return_sae_activations` request refused before generation (Feature 27, FR-27.1i,
    FR-27.2): a shape with no single position axis (`n > 1`, `extra_messages`, several prompts),
    an ambiguous SAE, a feature index past the SAE's width, `top_k` over the cap, or a worst-case
    entry count over the cap. Refused rather than answered partly — 400, never after generating."""

    code = "SAE_ACTIVATIONS_REFUSED"
    status_code = 400
