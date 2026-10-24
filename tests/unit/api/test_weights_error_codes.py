"""Every Feature 31/32/33 error code is declared once, mapped, explained and documented.

The full code set comes from the three features' error tables plus the addendum (§2.3). Each
must be a MiLLMError class with the status the documents give, an ERROR_STATUS_MAP row of the
same status (a 4xx never typed server_error — test_error_map_complete covers that generally),
a friendly message, and a row in manual/docs/reference/error-codes.md.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from millm.api.routes.openai.errors import ERROR_STATUS_MAP
from millm.core import errors as errors_module
from millm.core.error_messages import ERROR_MESSAGES
from millm.core.errors import MiLLMError

ROOT = Path(__file__).resolve().parents[3]

#: (code, class, status, feature). "addendum" marks codes the addendum adds or W0a adds for it.
WEIGHTS_CODES = [
    ("ALREADY_ATTACHED", "AlreadyAttachedError", 409, "031"),
    ("NOT_ATTACHABLE", "NotAttachableError", 400, "031"),
    ("WEIGHTS_NOT_CLEAN", "WeightsNotCleanError", 409, "031"),
    ("WEIGHTS_TORN", "WeightsTornError", 503, "031"),
    ("WEIGHTS_UNAUTHORIZED", "WeightsUnauthorizedError", 401, "031"),
    ("WEIGHTS_FORBIDDEN", "WeightsForbiddenError", 403, "031"),
    ("NOT_ATTACHMENT_HOLDER", "NotAttachmentHolderError", 403, "031"),
    ("NOT_ATTACHED", "NotAttachedError", 409, "031"),
    ("NOT_QUIESCED", "NotQuiescedError", 409, "031"),
    ("ALREADY_QUIESCED", "AlreadyQuiescedError", 409, "031"),
    ("NO_OPEN_COMMIT", "NoOpenCommitError", 409, "031"),
    ("ATTACHMENT_NOT_STALE", "AttachmentNotStaleError", 409, "031"),
    ("PAUSE_IN_PROGRESS", "PauseInProgressError", 409, "031"),
    ("BARRIER_UNAVAILABLE", "WeightsBarrierUnavailableError", 409, "addendum"),
    ("WEIGHTS_UNSAVED", "WeightsUnsavedError", 409, "addendum"),
    ("INVALID_RUN_ID", "InvalidRunIdError", 400, "032"),
    ("MANIFEST_FIELD_NOT_CALLERS", "ManifestFieldNotCallersError", 400, "032"),
    ("INVALID_MANIFEST", "InvalidManifestError", 400, "032"),
    ("CHECKPOINT_EXISTS", "CheckpointExistsError", 409, "032"),
    ("CHECKPOINT_AMBIGUOUS", "CheckpointAmbiguousError", 409, "addendum"),
    ("INSUFFICIENT_STORAGE", "InsufficientStorageError", 507, "032"),
    ("CHECKPOINT_WRITE_FAILED", "CheckpointWriteFailedError", 500, "032"),
    ("INVALID_CHECKPOINT_PATH", "InvalidCheckpointPathError", 400, "032"),
    ("CHECKPOINT_NOT_FOUND", "CheckpointNotFoundError", 404, "032"),
    ("CHECKPOINT_HASH_MISMATCH", "CheckpointHashMismatchError", 422, "032"),
    ("CHECKPOINT_NOT_LOADABLE", "CheckpointNotLoadableError", 400, "addendum"),
    ("WEIGHTS_LOAD_FAILED", "WeightsLoadFailedError", 500, "032"),
    ("NOT_INSTANTIABLE", "NotInstantiableError", 400, "032"),
    ("MODEL_NAME_TAKEN", "ModelNameTakenError", 409, "032"),
    ("LOCAL_PATH_REGISTERED", "LocalPathRegisteredError", 409, "addendum"),
    ("LOAD_THROUGH_WEIGHTS_API", "LoadThroughWeightsApiError", 409, "032"),
    ("ADAPTER_NOT_SERVABLE", "AdapterNotServableError", 400, "033"),
    ("ADAPTER_INVALID", "AdapterInvalidError", 400, "033"),
    ("ADAPTER_MANIFEST_INVALID", "AdapterManifestInvalidError", 400, "033"),
    ("ADAPTER_HASH_MISMATCH", "AdapterHashMismatchError", 400, "033"),
    ("ADAPTER_BASE_MISMATCH", "AdapterBaseMismatchError", 400, "033"),
    ("ADAPTER_SHAPE_MISMATCH", "AdapterShapeMismatchError", 400, "033"),
    ("ADAPTER_SCHEMA_NOT_VENDORED", "AdapterSchemaNotVendoredError", 409, "033"),
    ("ADAPTER_ALREADY_LOADED", "AdapterAlreadyLoadedError", 409, "033"),
    ("NO_ADAPTER_LOADED", "NoAdapterLoadedError", 409, "033"),
    ("ADAPTER_BASE_CHANGED", "AdapterBaseChangedError", 409, "033"),
    ("ADAPTER_INSTALL_FAILED", "AdapterInstallFailedError", 500, "033"),
    ("ADAPTER_APPLIED", "AdapterAppliedError", 409, "addendum"),
]

IDS = [row[0] for row in WEIGHTS_CODES]


@pytest.mark.parametrize("code, cls_name, status, feature", WEIGHTS_CODES, ids=IDS)
def test_the_class_declares_code_and_status(code, cls_name, status, feature):
    cls = getattr(errors_module, cls_name)
    assert issubclass(cls, MiLLMError)
    assert (cls.code, cls.status_code) == (code, status)


@pytest.mark.parametrize("code, cls_name, status, feature", WEIGHTS_CODES, ids=IDS)
def test_the_map_row_keeps_the_status(code, cls_name, status, feature):
    mapped_status, error_type = ERROR_STATUS_MAP[code]
    assert mapped_status == status
    assert error_type == ("server_error" if status >= 500 else
                          "authentication_error" if status == 401 else "invalid_request_error")


#: Codes whose raise sites craft the message; a generic friendly sentence would replace it.
CRAFTED_MESSAGE_CODES = {"WEIGHTS_UNAUTHORIZED"}


@pytest.mark.parametrize("code", IDS)
def test_a_friendly_message_and_a_manual_row(code):
    if code in CRAFTED_MESSAGE_CODES:
        assert code not in ERROR_MESSAGES, f"{code} must keep its crafted message"
    else:
        assert ERROR_MESSAGES.get(code)
    text = (ROOT / "manual/docs/reference/error-codes.md").read_text()
    assert f"| `{code}` |" in text


def test_save_in_progress_is_gone():
    """D-06: renamed PAUSE_IN_PROGRESS; no class or row keeps the old name."""
    assert "SAVE_IN_PROGRESS" not in ERROR_STATUS_MAP
    assert all(getattr(c, "code", None) != "SAVE_IN_PROGRESS" for c in vars(errors_module).values())


def _handle(path: str, exc: MiLLMError):
    from millm.api.exception_handlers import millm_error_handler

    request = MagicMock()
    request.url.path = path
    request.method = "POST"
    return asyncio.run(millm_error_handler(request, exc))


def test_weights_unauthorized_carries_www_authenticate_on_both_envelopes():
    from millm.core.errors import WeightsUnauthorizedError

    for path in ("/api/weights/attach", "/v1/chat/completions"):
        response = _handle(path, WeightsUnauthorizedError.for_reason("missing"))
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


def test_weights_torn_carries_retry_after_and_no_other_class_header():
    from millm.core.errors import WeightsTornError

    response = _handle("/api/weights/status", WeightsTornError("torn"))
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert "www-authenticate" not in response.headers


def test_helper_constructors_name_their_reason():
    from millm.core.errors import (
        AlreadyAttachedError,
        NotAttachableError,
        PauseInProgressError,
        WeightsBarrierUnavailableError,
        WeightsForbiddenError,
        WeightsUnauthorizedError,
    )

    assert NotAttachableError.for_reason("dtype", "lm_head.weight").details == {
        "reason": "dtype", "parameter": "lm_head.weight"}
    assert NotAttachableError.for_reason("allocator").details == {"reason": "allocator"}
    exc = AlreadyAttachedError.for_attachment(role="trainer", label="miforge", state="active",
                                              attached_at="t", operation="model_load")
    assert exc.details["label"] == "miforge" and exc.details["operation"] == "model_load"
    for reason in ("missing", "invalid_token", "no_tokens_configured", "tokens_malformed"):
        assert WeightsUnauthorizedError.for_reason(reason).details == {"reason": reason}
    assert WeightsForbiddenError.for_roles("trainer", ("operator",)).details == {
        "role": "trainer", "required_roles": ["operator"]}
    assert PauseInProgressError.for_operation("save", "commit-begin").details == {
        "operation": "save", "refused_operation": "commit-begin"}
    assert WeightsBarrierUnavailableError.for_reason("cbm", "save").details == {
        "reason": "cbm", "operation": "save"}
    assert json.dumps(NotAttachableError.for_reason("gguf").details)
