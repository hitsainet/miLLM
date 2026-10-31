"""Streaming results are truthful about interruption; client bytes are unchanged (FR-38.9, AT04).

Driven through `InferenceService.stream_chat_completion` — the generator the `/v1/chat/completions`
streaming route hands to Starlette — on a tiny real Llama with a real armed probe and a real SQLite
database behind `async_session_factory`:

1. completed → `final`;
2. the client goes away AFTER the terminal probe chunk → still `final`;
3. the client goes away mid-stream → `interrupted` / `partial` over the passes observed;
4. the client goes away before any forward pass → `interrupted` / `unavailable`.

In every case the bytes the client received are the completed stream's own prefix: nothing new is
sent, and an interrupted result never reaches the client (D38.11).

`test_record_survives_cancellation` cancels the consumer the way Starlette does — an anyio cancel
scope, level-triggered, so every later `await` in the generator's `finally` is cancelled too. The
record is a SHIELDED, TRACKED task (D38.33), so the event is still written. Control M24: await
`_probe_record` unshielded in that `finally` — red. Control M9: mark every result `final` — red.
"""

from __future__ import annotations

import json
import threading

import anyio
import pytest
import torch
from sqlalchemy import select

import millm.api.dependencies as deps
from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage
from millm.db.models.probe import ProbeEvent
from millm.ml.probe_head import ProbeHead
from millm.services.inference_service import InferenceService
from millm.services.probe_runtime import ArmedProbe, ProbeRuntimeState
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.f28_fixtures import db, no_cuda  # noqa: F401 - fixtures

from tests.support.telemetry_mode import _stub_exports, telemetry_mode  # noqa: E402,F401 - fixtures

# 038 FR-38.23: every test here runs with export off AND with every signal on against the stub.
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("telemetry_mode")]


@pytest.fixture
async def served(db, monkeypatch):  # noqa: F811
    from millm.db.repositories.probe_repository import ProbeRepository
    from millm.services.probe_service import ProbeService
    from tests.unit.score_fixtures import tiny_definition

    ProbeRuntimeState.reset_for_tests()
    monkeypatch.setattr(deps, "_probe_event_service", None, raising=False)
    model, tokenizer = word_model(), word_tokenizer()
    svc = make_service(model, tokenizer)
    async with db() as session:
        row = await ProbeService(ProbeRepository(session)).import_definition(
            tiny_definition("probe-s", layer=0, threshold=0.0))
    ProbeRuntimeState().arm(ArmedProbe(
        probe_id=row.id, name=row.name, head=ProbeHead(weight=torch.ones(16), bias=0.0, layer=0),
        rule="mean", scope="all", layer=0, rung=2, rung_language="detects on unseen tasks",
        threshold=0.0, windows=("all",),
    ), model)
    yield svc, db
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="tiny", messages=[ChatMessage(role="user", content="w1 w2")],
                                 max_tokens=6, temperature=0.0, stream=True)


async def _events(factory) -> list[ProbeEvent]:
    await InferenceService.drain_probe_records(5.0)
    async with factory() as session:
        return list((await session.execute(select(ProbeEvent))).scalars().all())


def _payloads(chunks: list[str]) -> list:
    out = []
    for chunk in chunks:
        data = chunk.removeprefix("data: ").strip()
        out.append(data if data == "[DONE]" else
                   {k: v for k, v in json.loads(data).items() if k not in ("id", "created")})
    return out


async def _stream(svc, stop_after=None) -> list[str]:
    """Consume the stream; close it (a client disconnect) once `stop_after(chunk)` says so."""
    chunks: list[str] = []
    agen = svc.stream_chat_completion(_request())
    with no_cuda():
        async for chunk in agen:
            chunks.append(chunk)
            if stop_after is not None and stop_after(chunks):
                await agen.aclose()
                break
    return chunks


async def _completed(svc) -> list[str]:
    return await _stream(svc)


async def test_a_completed_stream_is_final(served):
    svc, factory = served
    chunks = await _completed(svc)
    assert chunks[-1].strip() == "data: [DONE]"
    [event] = await _events(factory)
    assert event.status == "final" and event.outcome == "completed"
    probe_chunk = next(c for c in chunks if "millm_probe_verdicts" in c)
    sent = json.loads(probe_chunk.removeprefix("data: "))["millm_probe_verdicts"][0]
    assert sent["result_id"] == event.result_id, "the client's object IS the stored one"


async def test_a_disconnect_after_the_terminal_chunk_stays_final(served):
    svc, factory = served
    await _stream(svc, lambda cs: "millm_probe_verdicts" in cs[-1])
    [event] = await _events(factory)
    assert event.status == "final" and event.outcome == "completed"


async def test_disconnect_is_partial(served):
    svc, factory = served
    full = _payloads(await _completed(svc))
    received = await _stream(svc, lambda cs: len(cs) >= 3)
    events = await _events(factory)
    interrupted = [e for e in events if e.outcome == "interrupted"]
    assert len(interrupted) == 1, [(e.outcome, e.status) for e in events]
    event = interrupted[0]
    assert event.status == "partial" and event.reason == "client_disconnected"
    assert event.score is not None and event.verdict is None
    assert event.result["status"] == "partial" and event.result["n_scored_tokens"] >= 1
    # The client's bytes are the completed stream's own prefix: nothing new, no verdict chunk.
    assert _payloads(received) == full[:len(received)]
    assert not any("millm_probe_verdicts" in c for c in received)


async def test_a_disconnect_before_any_pass_is_unavailable(served, monkeypatch):
    svc, factory = served
    gate = threading.Event()

    def no_forward(generation_kwargs, errors=None, **_kw):
        gate.wait(5.0)
        generation_kwargs["streamer"].end()  # the thread ends without one forward pass

    monkeypatch.setattr(svc, "_generate_in_thread", no_forward)
    chunks: list[str] = []
    agen = svc.stream_chat_completion(_request())
    with no_cuda():
        chunks.append(await agen.__anext__())   # the role chunk, before generation
        gate.set()
        await agen.aclose()
    [event] = await _events(factory)
    assert event.outcome == "interrupted" and event.status == "unavailable"
    assert event.score is None and event.reason == "client_disconnected"
    assert not any("millm_probe_verdicts" in c for c in chunks)


async def test_record_survives_cancellation(served):
    """Starlette cancels the response task with an anyio cancel scope: LEVEL-triggered, so every
    `await` in the generator's `finally` raises again. The record is a tracked, shielded task."""
    svc, factory = served
    with no_cuda():
        with anyio.CancelScope() as scope:
            n = 0
            async for _chunk in svc.stream_chat_completion(_request()):
                n += 1
                if n == 3:
                    scope.cancel()
    assert scope.cancelled_caught or scope.cancel_called
    events = await _events(factory)
    assert len(events) == 1, "the cancelled stream's event was lost"
    # ⚠ EXACTLY `cancelled` (review R1): a cancel scope raises CancelledError inside the
    # generator, and that path names itself. Accepting either code let the mutation that drops
    # `_probe_interrupt = "cancelled"` survive.
    assert events[0].outcome == "interrupted" and events[0].reason == "cancelled"


async def test_a_failed_record_task_is_logged_by_class_name_only():
    import asyncio

    from structlog.testing import capture_logs

    from millm.services.inference_service import _probe_record_done

    async def fails():
        raise RuntimeError("row values: secret-prompt-text")

    task = asyncio.get_running_loop().create_task(fails())
    InferenceService._probe_record_tasks.add(task)
    await asyncio.wait({task})
    with capture_logs() as logs:
        _probe_record_done(task)
    assert task not in InferenceService._probe_record_tasks
    assert [(e["event"], e["error"]) for e in logs] == [("probe_record_task_failed", "RuntimeError")]
    assert "secret-prompt-text" not in str(logs)


# ── generation raised (review R1) ──────────────────────────────────────────────────
# Mutations that survived the build's suite: dropping `_probe_interrupt = "generation_error"` from
# the streaming `except`, and `_probe_record` calling `context.finish()` instead of
# `finish(interrupted=…)` — the second makes a request whose generation RAISED record a `final`
# result, indistinguishable from a completed one (the U8 defect 038 exists to close).


async def test_a_stream_whose_generation_raised_is_interrupted_generation_error(served,
                                                                                monkeypatch):
    svc, factory = served

    def fails(generation_kwargs, errors=None, **_kw):
        errors.append(RuntimeError("generation exploded"))
        generation_kwargs["streamer"].end()

    monkeypatch.setattr(svc, "_generate_in_thread", fails)
    chunks = await _completed(svc)
    [event] = await _events(factory)
    assert event.outcome == "interrupted" and event.reason == "generation_error", (
        event.outcome, event.reason)
    assert event.status == "unavailable" and event.verdict is None
    assert not any("millm_probe_verdicts" in c for c in chunks)


async def test_a_non_streaming_request_whose_generation_raised_is_never_final(served,
                                                                              monkeypatch):
    svc, factory = served

    def fails(*_a, **_kw):
        raise RuntimeError("generation exploded")

    monkeypatch.setattr(svc, "_generate_sync", fails)
    request = _request().model_copy(update={"stream": False})
    with no_cuda(), pytest.raises(Exception):
        await svc.create_chat_completion(request)
    [event] = await _events(factory)
    assert event.outcome == "interrupted" and event.reason == "generation_error", (
        event.outcome, event.reason)
    assert event.status in ("partial", "unavailable") and event.result["status"] != "final"


async def test_a_stream_that_raised_after_generation_is_interrupted_generation_error(served,
                                                                                    monkeypatch):
    """The generator's own `except Exception` branch (not the thread-error branch above)."""
    svc, factory = served

    def boom(**_kw):
        raise RuntimeError("monitoring exploded")

    monkeypatch.setattr(svc, "_notify_monitoring", boom)
    await _completed(svc)
    [event] = await _events(factory)
    assert event.outcome == "interrupted" and event.reason == "generation_error", (
        event.outcome, event.reason)


async def test_a_spawned_record_is_tracked_and_the_drain_waits_for_it():
    """Review R1: removing `_probe_record_tasks.add(task)` survived — the lifespan drain then
    returns at once and a shutdown drops records in flight. Pins the WIRING the drain reads."""
    import asyncio

    svc = object.__new__(InferenceService)
    written: list[str] = []

    async def slow_record(*_a, **_kw):
        await asyncio.sleep(0.2)
        written.append("row")

    svc._probe_record = slow_record
    task = svc._spawn_probe_record(None)
    assert task in InferenceService._probe_record_tasks, "the record task is not tracked"
    assert await InferenceService.drain_probe_records(5.0) == 0
    assert written == ["row"], "the drain returned before the record was written"
    assert task not in InferenceService._probe_record_tasks, "a finished task is never forgotten"
