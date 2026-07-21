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


class ModelNotLoadedError(MiLLMError):
    """Raised when operation requires a loaded model but none is loaded."""

    code = "MODEL_NOT_LOADED"
    status_code = 400


class ModelAlreadyLoadedError(MiLLMError):
    """Raised when attempting to load a model that is already loaded."""

    code = "MODEL_ALREADY_LOADED"
    status_code = 400


class ModelBusyError(MiLLMError):
    """Raised when model is busy with another operation."""

    code = "MODEL_BUSY"
    status_code = 409


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


# =============================================================================
# Error code to class mapping for lookup
# =============================================================================

class CircuitSensingEventNotFoundError(MiLLMError):
    """A circuit edge sensing event id that does not exist (Feature 15)."""

    code = "CIRCUIT_SENSING_EVENT_NOT_FOUND"
    status_code = 404


ERROR_CLASSES: dict[str, type[MiLLMError]] = {
    "CIRCUIT_LAYER_CONTENTION": CircuitLayerContentionError,
    "CIRCUIT_SENSING_EVENT_NOT_FOUND": CircuitSensingEventNotFoundError,
    "INTERNAL_ERROR": MiLLMError,
    "MODEL_NOT_FOUND": ModelNotFoundError,
    "MODEL_ALREADY_EXISTS": ModelAlreadyExistsError,
    "MODEL_LOAD_FAILED": ModelLoadError,
    "MODEL_NOT_LOADED": ModelNotLoadedError,
    "MODEL_ALREADY_LOADED": ModelAlreadyLoadedError,
    "MODEL_BUSY": ModelBusyError,
    "INSUFFICIENT_MEMORY": InsufficientMemoryError,
    "INSUFFICIENT_DISK": InsufficientDiskError,
    "DOWNLOAD_FAILED": DownloadFailedError,
    "DOWNLOAD_CANCELLED": DownloadCancelledError,
    "REPO_NOT_FOUND": RepoNotFoundError,
    "GATED_MODEL_NO_TOKEN": GatedModelError,
    "INVALID_HF_TOKEN": InvalidTokenError,
    "INVALID_LOCAL_PATH": InvalidLocalPathError,
    "SAE_NOT_FOUND": SAENotFoundError,
    "SAE_NOT_ATTACHED": SAENotAttachedError,
    "SAE_ALREADY_ATTACHED": SAEAlreadyAttachedError,
    "SAE_INCOMPATIBLE": SAEIncompatibleError,
    "SAE_LOAD_FAILED": SAELoadError,
    "INVALID_FEATURE_INDEX": InvalidFeatureIndexError,

    "PROFILE_NOT_FOUND": ProfileNotFoundError,
    "PROFILE_ALREADY_EXISTS": ProfileAlreadyExistsError,
    "PROFILE_INCOMPATIBLE": ProfileCompatibilityError,
    "INVALID_PROFILE_FORMAT": InvalidProfileFormatError,
    "VALIDATION_ERROR": ValidationError,
    "SAE_SET_INCOMPLETE": SAESetIncompleteError,
    "CIRCUIT_NOT_FOUND": CircuitNotFoundError,
    "UNVALIDATED_CIRCUIT": UnvalidatedCircuitError,
    "NO_ACTIVE_CIRCUIT": NoActiveCircuitError,
}
