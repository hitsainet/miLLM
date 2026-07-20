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
