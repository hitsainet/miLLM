"""The per-request steering dial runs while its request holds the request-queue slot, on every path that applies it.

F18 R3-16 (tests/integration/test_single_serving_derivation.py::TestR3TheDialSerialisationDEPENDENCYIsPinned):
the dial saves, applies and restores global steering state. That is safe only because
MAX_CONCURRENT_REQUESTS is 1 AND the dial runs inside the request queue's semaphore. The second
half was pinned by scraping create_chat_completion's source for the text
"self._request_queue.acquire()". ee20b11 (2026-09-14) moved admission into InferenceService._admit,
the text disappeared, and the guard failed with "substring not found" while the dial was still
serialised. Integration tests do not run in CI, so nothing noticed.

This checks the behaviour instead: a stand-in queue records whether its slot is held at the moment
_apply_request_steering is awaited, and the request stops there, before anything is generated. An
AST walk keeps the list of paths honest: a new caller of _apply_request_steering fails
test_every_path_that_applies_the_dial_is_covered_here until it is added below.

MUTATION CONTROLS (2026-09-15; each restored, sha256 verified, git diff clean):
  DIAL-M1 _admit enters an unrelated asyncio.Lock instead of the queue's acquire
          -> red: chat, batched and streamed
  DIAL-M2 create_chat_completion enters an unrelated lock instead of _admit -> red: chat
  DIAL-M3 _create_batched_chat_completion does the same                    -> red: batched
  DIAL-M4 stream_chat_completion does the same                             -> red: streamed
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from contextlib import asynccontextmanager

import pytest

from millm.api.schemas.openai import ChatCompletionRequest
from millm.services.inference_service import InferenceService

APPLY_PATHS = {"create_chat_completion", "_create_batched_chat_completion", "stream_chat_completion"}


class _Queue:
    """The request queue, reduced to whether its one slot is held."""

    def __init__(self) -> None:
        self.held = False
        self.pending_count = 0
        self.max_concurrent = 1
        self.max_pending = 5

    @asynccontextmanager
    async def acquire(self):
        self.held = True
        try:
            yield
        finally:
            self.held = False


class _StopAtApply(Exception):
    """Raised by the stand-in dial so the request ends before generation."""


@pytest.fixture
def service(monkeypatch):
    svc = InferenceService()
    queue = _Queue()
    svc._request_queue = queue
    held_at_apply: list[bool] = []

    async def apply(profile, intensity, request_id=None):
        held_at_apply.append(queue.held)
        raise _StopAtApply

    monkeypatch.setattr(svc, "_apply_request_steering", apply)
    monkeypatch.setattr(svc, "_engine_is_llamacpp", lambda: False)
    monkeypatch.setattr(svc, "_use_cbm_for_request", lambda **kwargs: False)
    monkeypatch.setattr(svc, "_has_steering_override", lambda request: True)
    monkeypatch.setattr(svc, "_format_chat_messages", lambda messages, kwargs=None: "prompt")
    monkeypatch.setattr(svc, "_prompt_opened_think", lambda prompt: False)
    monkeypatch.setattr(svc, "get_loaded_model_info", lambda: None)
    monkeypatch.setattr(svc, "_schedule_idle_cache_release", lambda: None)
    return svc, queue, held_at_apply


def _request(**extra) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4,
        steering_intensity=0.5,
        **extra,
    )


@pytest.mark.asyncio
async def test_chat_completion_applies_the_dial_inside_the_slot(service):
    svc, queue, held_at_apply = service
    with pytest.raises(_StopAtApply):
        await svc.create_chat_completion(_request())
    assert held_at_apply == [True]
    assert queue.held is False


@pytest.mark.asyncio
async def test_batched_chat_completion_applies_the_dial_inside_the_slot(service):
    svc, queue, held_at_apply = service
    with pytest.raises(_StopAtApply):
        await svc.create_chat_completion(
            _request(extra_messages=[[{"role": "user", "content": "hello"}]])
        )
    assert held_at_apply == [True]
    assert queue.held is False


@pytest.mark.asyncio
async def test_streamed_chat_completion_applies_the_dial_inside_the_slot(service):
    svc, queue, held_at_apply = service
    with pytest.raises(_StopAtApply):
        async for _chunk in svc.stream_chat_completion(_request(stream=True)):
            pass
    assert held_at_apply == [True]
    assert queue.held is False


def test_every_path_that_applies_the_dial_is_covered_here():
    tree = ast.parse(textwrap.dedent(inspect.getsource(InferenceService)))
    callers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_apply_request_steering"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                callers.add(fn.name)
    assert callers == APPLY_PATHS, (
        "the set of methods that apply the per-request steering dial changed; add a behaviour "
        f"test above for each new one. Found: {sorted(callers)}"
    )
