"""A load an OpenAI request triggers is refused with the refusal's own status and type.

Review round 3, 2026-09-14. The chat, completions and embeddings routes load the
model a request names (`load_model_and_wait`) and answered every MiLLMError from
it as a 500 `server_error`. Two consequences, both on the path Open WebUI uses to
switch models:

  * The ERROR_STATUS_MAP rows for load refusals were unreachable there. Review
    round 2 added UNSUPPORTED_QUANTIZATION -> 400 invalid_request_error "because a
    request naming a Q2 checkpoint auto-loads it" and pinned the row by calling
    millm_error_handler directly (test_exception_handlers.py) — a handler this
    route never reaches, since it catches the exception itself. A split no card
    set holds went out as 500, not the 503 the table gives INSUFFICIENT_MEMORY.
  * A load that FAILED in the background came back from load_model_and_wait as
    ModelBusyError, which the routes answer "Another model load is already in
    progress; retry once it finishes" — for a load that had finished and failed.

Everything here drives the real routes through the app; only the model service
and the inference service are stubbed.

MUTATION CONTROLS (review round 3, 2026-09-14; mutate.py, restored and sha256-verified):
  R3-M1  chat.py answers a load refusal with server_error again
         -> the three /v1/chat/completions cases
  R3-M1b completions.py answers it with server_error again -> the three /v1/completions cases
  R3-M1c embeddings.py answers it with server_error again  -> the three /v1/embeddings cases
  R3-M1d load_refused_error ignores ERROR_STATUS_MAP (always 500 server_error)
         -> the 400 and 503 cases on every endpoint
  R3-M2  load_model_and_wait raises ModelBusyError for a failed load again
         -> test_a_load_that_failed_in_the_background_is_not_reported_as_busy
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from millm.core.errors import (
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadError,
    UnsupportedQuantizationError,
)
from millm.db.models.model import ModelStatus
from millm.main import create_app
from millm.services.model_service import ModelService
from tests.support.factories import make_model

BODIES = {
    "/v1/chat/completions": {"messages": [{"role": "user", "content": "hi"}]},
    "/v1/completions": {"prompt": "hi"},
    "/v1/embeddings": {"input": "hi"},
}

REFUSALS = [
    pytest.param(
        UnsupportedQuantizationError("Q2 cannot be loaded as a transformers model: bitsandbytes has no 2-bit mode"),
        400, "invalid_request_error", "unsupported_quantization",
        id="q2-is-the-callers-to-change",
    ),
    pytest.param(
        InsufficientMemoryError(
            "Not enough GPU memory. Need ~40000 MB; split across GPUs this can hold 31952 MB "
            "(GPU 0: 9976 MB, GPU 1: 21976 MB)."
        ),
        503, "server_error", "insufficient_memory",
        id="no-split-holds-it",
    ),
    pytest.param(
        ModelLoadError("Model failed to load: CUDA out of memory"),
        500, "server_error", "model_load_failed",
        id="a-failed-load-stays-a-server-fault",
    ),
]


def _client(error: Exception) -> tuple[TestClient, MagicMock]:
    from millm.api.dependencies import get_inference_service, get_model_service

    row = make_model(id=3, name="wanted", status=ModelStatus.READY)
    svc = MagicMock()
    svc.find_model_by_name = AsyncMock(return_value=row)
    svc.get_locked_model = AsyncMock(return_value=None)
    svc.load_model_and_wait = AsyncMock(side_effect=error)
    inference = MagicMock()
    # Another model resident, so every route goes on to load the one it names.
    other = MagicMock()
    other.name = "some-other-model"
    inference.get_loaded_model_info = lambda: other
    inference.request_queue = MagicMock(pending_count=0, max_pending=10)

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_inference_service] = lambda: inference
    return TestClient(app), svc


@pytest.mark.parametrize("path", sorted(BODIES))
@pytest.mark.parametrize("error, status, error_type, code", REFUSALS)
def test_a_load_refusal_keeps_its_status_and_type(path, error, status, error_type, code):
    client, svc = _client(error)

    response = client.post(path, json={"model": "wanted", **BODIES[path]})

    assert svc.load_model_and_wait.await_count == 1, "the route must reach the load it answers for"
    assert svc.load_model_and_wait.await_args.args == (3,)
    assert response.status_code == status, response.text
    body = response.json()["error"]
    assert body["type"] == error_type
    assert body["code"] == code
    assert body["message"] == f"Could not load 'wanted': {error.message}", "the refusal's figures reach the caller"


@pytest.mark.asyncio
async def test_a_load_that_failed_in_the_background_is_not_reported_as_busy(monkeypatch):
    """The row went READY -> ERROR while load_model_and_wait polled it."""
    import millm.services.model_service as model_service_module

    ready = make_model(id=3, name="wanted", status=ModelStatus.READY)
    failed = make_model(
        id=3, name="wanted", status=ModelStatus.ERROR,
        error_message="GPU 1 (RTX 3090) ran out of memory while loading layer 31",
    )
    repo = MagicMock()
    repo.get_by_id = AsyncMock(side_effect=[ready, failed])
    repo.get_locked_model = AsyncMock(return_value=None)
    loader = MagicMock()
    loader.loaded_model_id = None
    svc = ModelService(repository=repo, downloader=MagicMock(), loader=loader, emitter=None)
    svc.load_model = AsyncMock(return_value=ready)

    async def _no_wait(_seconds):
        return None

    monkeypatch.setattr(model_service_module.asyncio, "sleep", _no_wait)

    with pytest.raises(ModelLoadError) as raised:
        await svc.load_model_and_wait(3, timeout=5)

    assert not isinstance(raised.value, ModelBusyError)
    assert raised.value.code == "MODEL_LOAD_FAILED"
    assert "ran out of memory while loading layer 31" in raised.value.message
    assert svc.load_model.await_count == 1


def test_the_route_does_not_tell_the_caller_to_wait_for_a_load_that_failed():
    client, _ = _client(ModelLoadError("Model failed to load: GPU 1 ran out of memory"))

    response = client.post("/v1/chat/completions", json={"model": "wanted", **BODIES["/v1/chat/completions"]})

    assert "already in progress" not in response.text
    assert "GPU 1 ran out of memory" in response.json()["error"]["message"]
