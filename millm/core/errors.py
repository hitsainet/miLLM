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


# =============================================================================
# Steering Errors
# =============================================================================


class SteeringError(MiLLMError):
    """Base error for steering-related issues."""

    code = "STEERING_ERROR"
    status_code = 400


class InvalidFeatureIndexError(MiLLMError):
    """Raised when feature index is out of range."""

    code = "INVALID_FEATURE_INDEX"
    status_code = 400


class InvalidSteeringValueError(MiLLMError):
    """Raised when steering value is out of range."""

    code = "INVALID_STEERING_VALUE"
    status_code = 400


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
# Error code to class mapping for lookup
# =============================================================================

ERROR_CLASSES: dict[str, type[MiLLMError]] = {
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
    "STEERING_ERROR": SteeringError,
    "INVALID_FEATURE_INDEX": InvalidFeatureIndexError,
    "INVALID_STEERING_VALUE": InvalidSteeringValueError,
    "PROFILE_NOT_FOUND": ProfileNotFoundError,
    "PROFILE_ALREADY_EXISTS": ProfileAlreadyExistsError,
    "PROFILE_INCOMPATIBLE": ProfileCompatibilityError,
    "INVALID_PROFILE_FORMAT": InvalidProfileFormatError,
    "VALIDATION_ERROR": ValidationError,
}
