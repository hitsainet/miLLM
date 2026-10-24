"""A server fault never sends exception text to a client, at ANY log level (ENH-003, 2026-10-20).

`generic_exception_handler` returned `str(exc)` as `details.technical_message` whenever its logger
was enabled for DEBUG, and the exception's class name always. The management path of
`millm_error_handler` sent every 5xx `MiLLMError`'s message and details whole (`ModelLoadError`
wraps llama.cpp's text, `DownloadFailedError` a Hugging Face client error under `details.error`),
and an unmapped 5xx code's friendly message fell back to that same text. The SSE error event sent
`exc.message` for a 5xx too.

Now every one of them logs the text (with the exception) under an `error_id` and answers with the
stable code, a fixed sentence and the same `error_id`. Each test runs at DEBUG and at INFO, through
the handlers as `create_app` registers them, and checks BOTH halves: the body lacks the text, the
log line has it under the id the body names.
"""

from __future__ import annotations

import json
import logging
import re

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from millm.core.errors import (
    DownloadFailedError,
    GgufTensorSplitError,
    InsufficientMemoryError,
    ModelLoadError,
    ModelNotFoundError,
    ScoringNumericalError,
)

SECRET = "/data/model_cache/private/llama.gguf: password=hunter2 llama_model_load failed"
ERROR_ID = re.compile(r"[0-9a-f]{32}")


class SecretLeakError(Exception):
    """A class name a client must not see either."""


@pytest.fixture(params=[logging.DEBUG, logging.INFO], ids=["DEBUG", "INFO"])
def log_level(request):
    """The root logger AND the handler module's logger at the level under test.

    The old gate asked `logging.getLogger("millm.api.exception_handlers").isEnabledFor(DEBUG)`,
    which follows the root level when the module logger has none of its own.
    """
    root = logging.getLogger()
    module = logging.getLogger("millm.api.exception_handlers")
    saved = (root.level, module.level)
    root.setLevel(request.param)
    module.setLevel(logging.NOTSET)
    assert module.isEnabledFor(logging.DEBUG) is (request.param == logging.DEBUG)
    yield request.param
    root.setLevel(saved[0])
    module.setLevel(saved[1])


@pytest.fixture(scope="module")
def client():
    """The real app's handler registrations (main.create_app), with routes that raise."""
    from millm.main import create_app

    app = create_app()
    raising = {
        "/api/_fault/unexpected": SecretLeakError(SECRET),
        "/v1/_fault/unexpected": SecretLeakError(SECRET),
        "/api/_fault/load": ModelLoadError(
            f"Failed to load GGUF model: {SECRET}", details={"model_id": 7, "error": SECRET}
        ),
        "/api/_fault/download": DownloadFailedError(
            f"Download failed: {SECRET}", details={"repo_id": "o/r", "circuit_error": SECRET}
        ),
        # No ERROR_MESSAGES entry: the friendly sentence fell back to the message itself.
        "/api/_fault/unmapped": ScoringNumericalError(f"non-finite: {SECRET}"),
        "/api/_fault/not-found": ModelNotFoundError("Model 'nope' not found"),
        "/api/_fault/memory": InsufficientMemoryError(
            "GPU 0 has 100 MB free; the model needs ~900 MB.", details={"required_mb": 900}
        ),
        "/api/_fault/split": GgufTensorSplitError(
            "GGUF_TENSOR_SPLIT has 3 value(s) but this load splits across 2 GPU(s).",
            details={"tensor_split": [1, 1, 1]},
        ),
        "/v1/_fault/split": GgufTensorSplitError(
            "GGUF_TENSOR_SPLIT has 3 value(s) but this load splits across 2 GPU(s).",
        ),
    }
    for path, exc in raising.items():
        async def endpoint(_exc=exc):
            raise _exc

        app.add_api_route(path, endpoint, methods=["GET"])
    return TestClient(app, raise_server_exceptions=False)


def _assert_no_leak(text: str) -> None:
    assert SECRET not in text
    assert "hunter2" not in text and "llama_model_load" not in text
    assert "SecretLeakError" not in text and "Traceback" not in text


def _one(logs, event):
    found = [e for e in logs if e.get("event") == event]
    assert len(found) == 1, f"expected one {event!r} log line, got {[e.get('event') for e in logs]}"
    return found[0]


class TestUnexpectedException:
    def test_management_body_has_the_code_and_id_and_the_log_has_the_text(self, client, log_level):
        with capture_logs() as logs:
            r = client.get("/api/_fault/unexpected")
        assert r.status_code == 500
        _assert_no_leak(r.text)
        error = r.json()["error"]
        assert error["code"] == "INTERNAL_ERROR"
        error_id = error["details"]["error_id"]
        assert ERROR_ID.fullmatch(error_id)
        assert error_id in error["details"]["technical_message"]
        assert set(error["details"]) == {"error_id", "technical_message"}

        logged = _one(logs, "unhandled_exception")
        assert logged["error_id"] == error_id
        assert SECRET in logged["error_message"]
        assert isinstance(logged["exc_info"], SecretLeakError), "the traceback must be logged"

    def test_v1_body_has_the_id_and_the_log_has_the_text(self, client, log_level):
        with capture_logs() as logs:
            r = client.get("/v1/_fault/unexpected")
        assert r.status_code == 500
        _assert_no_leak(r.text)
        error = r.json()["error"]
        assert error["code"] == "server_error"
        logged = _one(logs, "unhandled_exception")
        assert logged["error_id"] in error["message"]
        assert SECRET in logged["error_message"]


class TestManagementServerFault:
    @pytest.mark.parametrize(
        "path, code, kept",
        [
            ("/api/_fault/load", "MODEL_LOAD_FAILED", {"model_id": 7}),
            ("/api/_fault/download", "DOWNLOAD_FAILED", {"repo_id": "o/r"}),
            ("/api/_fault/unmapped", "NON_FINITE_LOGITS", {}),
        ],
    )
    def test_the_text_is_logged_never_sent(self, client, log_level, path, code, kept):
        with capture_logs() as logs:
            r = client.get(path)
        assert r.status_code >= 500
        _assert_no_leak(r.text)
        error = r.json()["error"]
        assert error["code"] == code
        details = error["details"]
        error_id = details["error_id"]
        assert ERROR_ID.fullmatch(error_id) and error_id in details["technical_message"]
        for key, value in kept.items():
            assert details[key] == value, "structured, non-text details stay"
        assert "error" not in details and "circuit_error" not in details

        logged = _one(logs, "api_server_fault")
        assert logged["error_id"] == error_id
        assert SECRET in logged["error_message"]
        assert logged["exc_info"] is not None

    def test_a_refusal_keeps_its_composed_text(self, client, log_level):
        r = client.get("/api/_fault/not-found")
        assert r.status_code == 404
        assert r.json()["error"]["details"]["technical_message"] == "Model 'nope' not found"
        assert "error_id" not in r.json()["error"]["details"]

    def test_a_capacity_refusal_keeps_its_figures(self, client, log_level):
        r = client.get("/api/_fault/memory")
        assert r.status_code == 507
        assert "900 MB" in r.text and r.json()["error"]["details"]["required_mb"] == 900

    @pytest.mark.parametrize("path", ["/api/_fault/split", "/v1/_fault/split"])
    def test_a_class_marked_client_safe_keeps_its_guidance(self, client, log_level, path):
        r = client.get(path)
        assert r.status_code == 500
        assert "GGUF_TENSOR_SPLIT has 3 value(s)" in r.text


class TestStreamEvent:
    def test_a_server_fault_event_carries_the_id_not_the_text(self, log_level):
        from millm.services.inference_service import _stream_error_event

        with capture_logs() as logs:
            event = _stream_error_event(ModelLoadError(f"Failed to load GGUF model: {SECRET}"))
        _assert_no_leak(event)
        body = json.loads(event.removeprefix("data: ").strip())
        logged = _one(logs, "v1_stream_server_fault")
        assert logged["error_id"] in body["error"]["message"]
        assert SECRET in logged["error_message"]

    def test_a_refusal_event_keeps_its_text(self, log_level):
        from millm.services.inference_service import _stream_error_event

        event = _stream_error_event(ModelNotFoundError("Model 'nope' not found"))
        assert "Model 'nope' not found" in event


class TestHealthBlocks:
    def test_an_unreadable_inference_block_names_an_id_not_the_text(self, log_level):
        from millm.api.routes.system.health import read_inference_state

        class Broken:
            @property
            def request_queue(self):
                raise SecretLeakError(SECRET)

        with capture_logs() as logs:
            state = read_inference_state(Broken())
        _assert_no_leak(state.error)
        logged = _one(logs, "health_inference_state_unavailable")
        assert logged["error_id"] in state.error and SECRET in logged["error"]
