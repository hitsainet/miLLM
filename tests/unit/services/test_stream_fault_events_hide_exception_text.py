"""Mid-stream error events never carry a server fault's text (ENH-003 follow-up, 2026-10-20).

Once a stream's 200 is committed, an error goes out as an SSE event. The per-request steering
branch of `stream_chat_completion` sent `exc.message` for ANY `MiLLMError`, so an SAE that failed
to load (`SAELoadError`, 500, wrapping the loader's text) reached the client verbatim. Driven
through the real streaming generator on a tiny real model.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import pytest
from structlog.testing import capture_logs

from millm.core.errors import SAELoadError, SAENotAttachedError
from tests.support.generation_entry_points import _chat
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

SECRET = "/data/sae_cache/private/sae.safetensors: password=hunter2 header too large"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    @asynccontextmanager
    async def no_db():
        raise RuntimeError("no database here")
        yield  # pragma: no cover

    monkeypatch.setattr("millm.db.base.async_session_factory", no_db)
    yield
    clear_loaded()


@pytest.fixture(params=[logging.DEBUG, logging.INFO], ids=["DEBUG", "INFO"])
def log_level(request):
    root = logging.getLogger()
    saved = root.level
    root.setLevel(request.param)
    yield
    root.setLevel(saved)


async def _stream_with_steering_error(monkeypatch, exc):
    svc = make_service(word_model(), word_tokenizer())

    async def refuse(request, request_id=None):
        raise exc

    monkeypatch.setattr(svc, "_dispatch_request_steering", refuse)
    return [chunk async for chunk in svc.stream_chat_completion(_chat(stream=True))]


def _error(chunks):
    events = [json.loads(c.removeprefix("data: ").strip()) for c in chunks
              if c.startswith("data: {") and '"error"' in c]
    assert len(events) == 1 and chunks[-1] == "data: [DONE]\n\n"
    return events[0]["error"]


async def test_a_server_fault_sends_its_id_and_logs_its_text(monkeypatch, log_level):
    with capture_logs() as logs:
        chunks = await _stream_with_steering_error(monkeypatch, SAELoadError(f"load: {SECRET}"))
    text = "".join(chunks)
    assert SECRET not in text and "hunter2" not in text
    error = _error(chunks)
    assert error["code"] == "sae_load_failed"
    logged = [e for e in logs if e.get("event") == "v1_stream_server_fault"]
    assert len(logged) == 1 and SECRET in logged[0]["error_message"]
    assert logged[0]["error_id"] in error["message"]


async def test_a_refusal_keeps_its_text(monkeypatch, log_level):
    chunks = await _stream_with_steering_error(monkeypatch, SAENotAttachedError("No SAE is attached"))
    assert _error(chunks)["message"] == "No SAE is attached"
