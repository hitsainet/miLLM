"""A /v1 server fault never sends internal exception text to the client (CodeQL alert 9).

Mirror alert 9 (py/stack-trace-exposure, 2026-10-17) traced the caught load exception in the chat,
completions and embeddings routes into the OpenAI error body. The GGUF loader wraps llama.cpp's own
exception text — file paths included — as `ModelLoadError(f"Failed to load GGUF model: {e}")`, and
the auto-load path returns the background loader's raw error text; both reached the caller.

Now a 500 answers with its code and logs the text server-side (`client_safe_message`); refusals
(4xx, and 503 "retry later") keep the message miLLM composed for the client. Checked on every
route that catches a load error, and on the global handler for an error raised past the route.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from millm.core.errors import ModelLoadError, ModelNotFoundError

SECRET = "/data/model_cache/private/llama.gguf: llama_model_load: error loading model"


@pytest.fixture
def wrapped_error():
    try:
        try:
            raise ValueError(SECRET)
        except ValueError as e:
            raise ModelLoadError(f"Failed to load GGUF model: {e}") from e
    except ModelLoadError as exc:
        return exc


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"])
def test_a_load_failure_reaches_the_client_as_its_code_only(path, wrapped_error):
    from tests.unit.api.test_openai_load_refusal_status import BODIES, _client

    client, _ = _client(wrapped_error)
    with capture_logs() as logs:
        response = client.post(path, json={"model": "wanted", **BODIES[path]})
    assert response.status_code == 500
    assert SECRET not in response.text and "llama_model_load" not in response.text
    assert "MODEL_LOAD_FAILED" in response.json()["error"]["message"]
    logged = [e for e in logs if e.get("event") == "v1_model_load_failed"]
    assert len(logged) == 1 and SECRET in logged[0]["error_message"], "the text must be logged"


def test_the_global_handler_hides_a_500_and_keeps_a_refusal():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from millm.api.exception_handlers import millm_error_handler
    from millm.core.errors import MiLLMError

    app = FastAPI()
    app.add_exception_handler(MiLLMError, millm_error_handler)  # as main.py registers it

    @app.get("/v1/fault")
    async def fault():
        raise ModelLoadError(f"Failed to load GGUF model: {SECRET}")

    @app.get("/v1/refusal")
    async def refusal():
        raise ModelNotFoundError("Model 'nope' not found")

    client = TestClient(app)
    with capture_logs() as logs:
        fault_response = client.get("/v1/fault")
    assert fault_response.status_code == 500
    assert SECRET not in fault_response.text
    assert "MODEL_LOAD_FAILED" in fault_response.json()["error"]["message"]
    assert any(e.get("event") == "v1_server_fault" and SECRET in e["error_message"] for e in logs)

    refused = client.get("/v1/refusal")
    assert refused.json()["error"]["message"] == "Model 'nope' not found"
