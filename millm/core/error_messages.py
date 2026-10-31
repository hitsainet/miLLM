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
    # INSUFFICIENT_MEMORY deliberately has NO generic entry, for the reason
    # VALIDATION_ERROR has none: every raise site crafts its message with the
    # figures and the fix — which card, how much each card of a split can hold,
    # "choose a different card or Auto", "serve it as GGUF" — and this sentence
    # replaced all of it in the Admin UI's toast with advice ("unloading other
    # models") that cannot apply to a one-model server. Review round 2, 2026-09-14.
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

    # Profile errors
    "PROFILE_NOT_FOUND": "The requested profile could not be found.",
    "PROFILE_INCOMPATIBLE": "This profile is not compatible with the current model/SAE configuration.",
    "INVALID_PROFILE_FORMAT": "The profile file format is invalid. Please use a valid JSON profile.",

    # Validation errors
    # VALIDATION_ERROR deliberately has NO generic entry: gate messages
    # (cluster feature-space mismatch, intensity range, index bounds) are
    # crafted for users and were being replaced by a generic sentence
    # (review find) — the handler's default (the exception's own message)
    # flows through instead.
    "INVALID_FEATURE_INDEX": "The feature index is outside the valid range for the attached SAE.",

    # Attachable weights (Features 31, 32, 33). Each names the next step; the reason and the
    # figures stay in `details` (and `technical_message`), which the handler keeps beside this.
    "ALREADY_ATTACHED": "A trainer holds the resident weights. Wait for it to detach, or ask an operator to clear a stale attachment.",
    "NOT_ATTACHABLE": "The resident model cannot be attached as loaded. See the reason in the details.",
    "WEIGHTS_NOT_CLEAN": "The resident weights are mid-commit or torn, so this operation is refused. Wait for the commit to end, or load a checkpoint to restore serving.",
    "WEIGHTS_UNSAVED": "The resident weights carry commits that were never saved, so a request may not swap the model out. Save them (POST /api/weights/save, as the attachment holder), or replace them deliberately with POST /api/weights/load or a management load.",
    "WEIGHTS_TORN": "The resident weights were left mid-write and are not served. A late commit-end from the writer or a load through POST /api/weights/load restores serving.",
    # WEIGHTS_UNAUTHORIZED deliberately has NO generic entry, for the reason VALIDATION_ERROR and
    # INSUFFICIENT_MEMORY have none: every refusal crafts its message per reason (malformed token
    # set, none configured, missing header, invalid token) and, on adapter routes, names the
    # operator-token requirement (addendum D-07). A generic sentence here replaced all of it.
    "WEIGHTS_FORBIDDEN": "This token's role may not perform this weights operation.",
    "NOT_ATTACHMENT_HOLDER": "Only the process holding the attachment may do this. Send its X-miLLM-Attachment header with the token that attached.",
    "NOT_ATTACHED": "Nothing is attached to the resident weights. Attach first.",
    # NOT_QUIESCED covers commit-begin before quiesce AND a resume or commit-begin sent while
    # the holder's quiesce is still draining (D-39, `reason: quiesce_in_progress`). "Quiesce
    # first" alone told a holder whose quiesce was in flight to quiesce again (ALREADY_QUIESCED).
    "NOT_QUIESCED": "Serving is not quiesced by this attachment yet. Quiesce first, or, if a quiesce is still draining in-flight work (reason quiesce_in_progress), wait for it to return before resuming or beginning a commit.",
    "ALREADY_QUIESCED": "Serving is already paused.",
    "NO_OPEN_COMMIT": "No commit is open for this attachment.",
    "ATTACHMENT_NOT_STALE": "The attachment is still sending heartbeats, so it cannot be cleared.",
    "PAUSE_IN_PROGRESS": "A serving pause (a save or an adapter change) is in progress. Retry once it finishes.",
    "BARRIER_UNAVAILABLE": "Serving cannot be paused under this deployment's settings (continuous batching on, or more than one request slot). See the reason in the details.",
    "INVALID_RUN_ID": "The run_id is not a valid checkpoint run name.",
    "MANIFEST_FIELD_NOT_CALLERS": "The request sets manifest fields that miLLM writes itself. Remove them.",
    "INVALID_MANIFEST": "The checkpoint manifest does not match the schema. The details name each failing field.",
    "CHECKPOINT_EXISTS": "A checkpoint already exists at that step.",
    "CHECKPOINT_AMBIGUOUS": "That run exists under more than one checkpoint root. Remove or rename one copy.",
    "INSUFFICIENT_STORAGE": "Not enough free disk for this checkpoint. Free some space and try again.",
    "CHECKPOINT_WRITE_FAILED": "Writing the checkpoint failed; nothing was published. See the server logs.",
    "INVALID_CHECKPOINT_PATH": "The path is outside the checkpoint roots.",
    "CHECKPOINT_NOT_FOUND": "No checkpoint by that name or path.",
    "CHECKPOINT_HASH_MISMATCH": "The checkpoint's weights do not match its manifest hash; nothing was loaded.",
    "CHECKPOINT_NOT_LOADABLE": "miLLM will not load this checkpoint as it stands. See the reason in the details.",
    "WEIGHTS_LOAD_FAILED": "The weights load failed partway and the weights are torn. Load a checkpoint to restore serving.",
    "NOT_INSTANTIABLE": "This model cannot be instantiated.",
    "MODEL_NAME_TAKEN": "Another model already uses that name.",
    "LOCAL_PATH_REGISTERED": "That checkpoint directory is already registered as a local model. Delete or re-register that model first.",
    "LOAD_THROUGH_WEIGHTS_API": "This model is a checkpoint and loads only through POST /api/weights/load, which verifies its hash.",
    "ADAPTER_NOT_SERVABLE": "The resident model cannot serve an adapter. See the reason in the details.",
    "ADAPTER_INVALID": "The adapter directory is invalid or incomplete.",
    "ADAPTER_MANIFEST_INVALID": "The adapter manifest does not match the schema. The details name each failing field.",
    "ADAPTER_HASH_MISMATCH": "The adapter weights do not match the manifest hash; nothing was applied.",
    "ADAPTER_BASE_MISMATCH": "The adapter was trained on a different base model than the one resident.",
    "ADAPTER_SHAPE_MISMATCH": "The adapter does not fit the resident model. The details name the first mismatch.",
    "ADAPTER_SCHEMA_NOT_VENDORED": "The adapter manifest schema is not installed, so adapters cannot be loaded.",
    "ADAPTER_ALREADY_LOADED": "An adapter is already applied. Unload it first.",
    "NO_ADAPTER_LOADED": "No adapter is applied.",
    "ADAPTER_BASE_CHANGED": "The base model changed while the adapter was being verified; nothing was applied. Try again.",
    "ADAPTER_INSTALL_FAILED": "Applying the adapter failed and was rolled back. See the server logs.",
    "ADAPTER_APPLIED": "An adapter is applied, so this comparison against the base model is refused. Unload the adapter first.",

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
