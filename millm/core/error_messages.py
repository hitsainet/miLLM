"""
User-friendly error messages for all error scenarios.

Maps error codes to human-readable messages for display in the frontend.
"""

from typing import Optional

# User-friendly error messages mapped by error code
ERROR_MESSAGES: dict[str, str] = {
    # Model errors
    "MODEL_NOT_FOUND": "The requested model could not be found. It may have been deleted or never existed.",
    "MODEL_ALREADY_EXISTS": "A model with the same repository and quantization already exists. You can load the existing model instead.",
    "MODEL_LOAD_FAILED": "Failed to load the model into GPU memory. Please check that you have sufficient VRAM available.",
    "MODEL_NOT_LOADED": "No model is currently loaded. Please load a model first before performing this operation.",
    "MODEL_ALREADY_LOADED": "This model is already loaded and ready to use.",
    "MODEL_BUSY": "The model is currently busy with another operation. Please wait and try again.",

    # Resource errors
    "INSUFFICIENT_MEMORY": "Not enough GPU memory available to load this model. Try a smaller model or quantization level (Q4/Q8), or free up memory by unloading other models.",
    "INSUFFICIENT_DISK": "Not enough disk space available to download this model. Free up some disk space and try again.",

    # Download errors
    "DOWNLOAD_FAILED": "Model download failed. Please check your internet connection and try again.",
    "DOWNLOAD_CANCELLED": "The download was cancelled.",
    "REPO_NOT_FOUND": "The HuggingFace repository was not found. Please verify the repository ID is correct (format: owner/model-name).",
    "GATED_MODEL_NO_TOKEN": "This model requires authentication. Please provide a valid HuggingFace token to access this gated model.",
    "INVALID_HF_TOKEN": "The HuggingFace token is invalid or expired. Please check your token and try again.",

    # Path errors
    "INVALID_LOCAL_PATH": "The specified local path is invalid or does not exist. Please provide a valid path to the model files.",

    # SAE errors
    "SAE_NOT_FOUND": "The requested SAE (Sparse Autoencoder) could not be found.",
    "SAE_NOT_ATTACHED": "No SAE is currently attached to the model. Please attach an SAE first.",
    "SAE_ALREADY_ATTACHED": "An SAE is already attached to this model. Detach it first before attaching a new one.",
    "SAE_INCOMPATIBLE": "This SAE is not compatible with the currently loaded model. Please use a matching SAE.",
    "SAE_LOAD_FAILED": "Failed to load the SAE. Please check that the SAE files are valid.",

    # Steering errors
    "STEERING_ERROR": "An error occurred while applying feature steering.",
    "INVALID_FEATURE_INDEX": "The feature index is out of range for this SAE.",
    "INVALID_STEERING_VALUE": "The steering value is out of the valid range (-10 to +10).",

    # Profile errors
    "PROFILE_NOT_FOUND": "The requested profile could not be found.",
    "PROFILE_INCOMPATIBLE": "This profile is not compatible with the current model/SAE configuration.",
    "INVALID_PROFILE_FORMAT": "The profile file format is invalid. Please use a valid JSON profile.",

    # Validation errors
    "VALIDATION_ERROR": "The request contains invalid data. Please check your input and try again.",

    # Generic error
    "INTERNAL_ERROR": "An unexpected error occurred. Please try again or contact support if the problem persists.",
}


def get_user_friendly_message(
    error_code: str,
    default_message: Optional[str] = None,
) -> str:
    """
    Get a user-friendly error message for an error code.

    Args:
        error_code: The error code to look up
        default_message: Optional default message if code not found

    Returns:
        User-friendly error message
    """
    return ERROR_MESSAGES.get(
        error_code,
        default_message or ERROR_MESSAGES["INTERNAL_ERROR"]
    )


def format_error_with_details(
    error_code: str,
    technical_message: str,
    details: Optional[dict] = None,
) -> dict:
    """
    Format an error with both user-friendly and technical details.

    Args:
        error_code: The error code
        technical_message: The technical error message
        details: Optional additional details

    Returns:
        Dict with user_message, technical_message, and details
    """
    return {
        "user_message": get_user_friendly_message(error_code, technical_message),
        "technical_message": technical_message,
        "details": details or {},
    }
