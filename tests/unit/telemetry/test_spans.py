"""Spans (FR-38.15; FTDD §6.3; FTASKS 8.4), read from the receiver fed by the real export path.

* ONE inference span per request, named by its GenAI operation (`chat`, `text_completion`,
  `millm.probe_score`), with a `millm.queue.wait` child and ONE `millm.probe.evaluate` child whose
  span id is every result's `span_id`;
* the serial chat fallback (several conversations, continuous batching active) still makes one;
* `ERROR` status on an exception and on a cancelled stream, with the error's CLASS only and no
  status message, no events;
* only allowlisted attributes, ever.

Control M15: remove the probe span (`child_span` in `_probe_finish`); add a second inference span
(delete the one-per-request guard in `inference_span`).
"""

from __future__ import annotations

from collections import Counter

import pytest

from millm.api.schemas.openai import ChatCompletionRequest
from millm.telemetry import allowlist
from tests.support import capture_pre038_goldens as cap
from tests.unit.telemetry.conftest import flush

OK, ERROR = 0, 2  # OTLP status codes (UNSET is 0; miLLM never sets OK)


def _by_trace(spans):
    out = {}
    for s in spans:
        out.setdefault(s["trace_id"], []).append(s)
    return out


async def test_one_inference_span_with_its_children_and_the_results_ids(stub, telemetry_on):
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA, cap.BETA])
        r = h.post_chat(cap.BODY, {})
        assert r.status_code == 200
        results = r.json()["millm_probe_verdicts"]
        flush()
    traces = _by_trace(stub.spans)
    assert len(traces) == 1
    spans = next(iter(traces.values()))
    names = Counter(s["name"] for s in spans)
    assert names == {"chat": 1, "millm.queue.wait": 1, "millm.probe.evaluate": 1}, names
    root = next(s for s in spans if s["name"] == "chat")
    evaluate = next(s for s in spans if s["name"] == "millm.probe.evaluate")
    wait = next(s for s in spans if s["name"] == "millm.queue.wait")
    assert evaluate["parent_span_id"] == root["span_id"] == wait["parent_span_id"]
    assert root["parent_span_id"] == ""
    assert {(e["trace_id"], e["span_id"]) for e in results} == {(root["trace_id"],
                                                                evaluate["span_id"])}
    assert root["attributes"]["gen_ai.operation.name"] == "chat"
    assert root["attributes"]["gen_ai.request.model"] == "tiny"
    assert root["attributes"]["millm.stream"] is False
    assert evaluate["attributes"]["millm.probe.results.completed"] == 3
    assert isinstance(wait["attributes"]["millm.queue.position_at_entry"], int)
    for s in spans:
        assert set(s["attributes"]) <= set(allowlist.SPAN), s["attributes"]
        assert s["status_code"] != ERROR and s["events"] == 0


async def test_a_stream_and_a_text_completion_and_a_score_each_open_one(stub, telemetry_on):
    telemetry_on()
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA])
        text = h.stream_chat(cap.BODY, {})
        chunk = cap._stream_chunk(text)["millm_probe_verdicts"]
        t = h.client.post("/v1/completions", json={"model": "tiny", "prompt": "w1 w2",
                                                    "max_tokens": 2, "temperature": 0.0})
        assert t.status_code == 200, t.text
        s = h.client.post("/api/probes/score", json={
            "probe_ids": [rows[0].id],
            "inputs": [{"token_ids": [0, 2, 6, 7, 8], "prompt_tokens": 3}]})
        assert s.status_code == 200, s.text
        flush()
    roots = Counter(sp["name"] for sp in stub.spans if sp["parent_span_id"] == "")
    assert roots == {"chat": 1, "text_completion": 1, "millm.probe_score": 1}, roots
    stream_root = next(sp for sp in stub.spans if sp["name"] == "chat")
    assert stream_root["attributes"]["millm.stream"] is True
    assert stream_root["status_code"] != ERROR
    assert {e["trace_id"] for e in chunk} == {stream_root["trace_id"]}
    assert Counter(sp["name"] for sp in stub.spans
                   if sp["trace_id"] == stream_root["trace_id"])["millm.probe.evaluate"] == 1
    # 12.5a F2: each root states its engine and mode (the score route's: STATELESS_EXECUTION,
    # which its results state too). Control F2c: delete `note_execution` in `score`.
    for root in (sp for sp in stub.spans if sp["parent_span_id"] == ""):
        attrs = root["attributes"]
        assert (attrs.get("millm.engine"), attrs.get("millm.execution_mode")) == (
            "transformers", "serial"), (root["name"], attrs)


async def test_the_serial_fallback_is_still_one_inference_span(stub, telemetry_on):
    telemetry_on()
    body = {**cap.BODY, "extra_messages": [[{"role": "user", "content": "w2 w3"}]]}
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        with cap._cbm(h.svc):
            r = h.post_chat(body, {})
        assert r.status_code == 200, r.text
        flush()
    names = Counter(s["name"] for s in stub.spans)
    assert names["chat"] == 1, names
    # 12.5a R1: the one span states the path that served the conversations — each through the
    # continuous-batching manager, which is also what each conversation's own results state.
    root = next(s for s in stub.spans if s["name"] == "chat")
    assert (root["attributes"].get("millm.engine"),
            root["attributes"].get("millm.execution_mode")) == ("cbm", "continuous_batching")


async def test_a_scoring_request_states_its_engine_and_mode(stub, telemetry_on):
    """12.5a R1: `/v1` scoring opens no probe context, so `_probe_begin` never stated its span's
    engine and mode; both scoring entries state them (FR-38.15: every span). Controls R1-S1
    (completion scoring) and R1-S2 (chat scoring): delete `note_execution` in each."""
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        t = h.client.post("/v1/completions", json={"model": "tiny", "prompt": "w1 w2",
                                                    "max_tokens": 1, "logprobs": 2})
        c = h.client.post("/v1/chat/completions", json={**cap.BODY, "logprobs": True,
                                                         "max_tokens": 1})
        assert t.status_code == 200 and c.status_code == 200, (t.text, c.text)
        assert t.json()["choices"][0]["logprobs"] and c.json()["choices"][0]["logprobs"]
        flush()
    roots = {s["name"]: s["attributes"] for s in stub.spans if s["parent_span_id"] == ""}
    assert set(roots) == {"text_completion", "chat"}, roots
    for name, attrs in roots.items():
        assert (attrs.get("millm.engine"), attrs.get("millm.execution_mode")) == (
            "transformers", "serial"), (name, attrs)


async def test_an_exception_ends_the_span_in_error_with_the_class_only(stub, telemetry_on,
                                                                       monkeypatch):
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])

        def boom(*a, **k):
            raise RuntimeError("SENTINEL-EXCEPTION-TEXT")

        monkeypatch.setattr(h.svc, "_format_chat_messages", boom)
        from fastapi.testclient import TestClient

        h._fresh_socket_throttle()
        client = TestClient(h.client.app, raise_server_exceptions=False)
        r = client.post("/v1/chat/completions", json=cap.BODY)
        assert r.status_code >= 500
        flush()
    root = next(s for s in stub.spans if s["name"] == "chat")
    assert root["status_code"] == ERROR and root["status_message"] == ""
    assert root["attributes"]["error.type"] == "RuntimeError"
    assert root["events"] == 0
    assert b"SENTINEL-EXCEPTION-TEXT" not in stub.all_bytes()


async def test_a_cancelled_stream_ends_its_span_in_error(stub, telemetry_on):
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        from millm.telemetry.context import _CURRENT_SPAN

        seen = []
        real = h.svc._record_probe_shielded

        async def spy(*a, **k):
            seen.append(_CURRENT_SPAN.get())
            return await real(*a, **k)

        h.svc._record_probe_shielded = spy
        request = ChatCompletionRequest(**{**cap.BODY, "stream": True, "max_tokens": 3})
        agen = h.svc.stream_chat_completion(request)
        first = await agen.__anext__()
        assert first.startswith("data: ")
        await agen.aclose()
        await h.svc.drain_probe_records(10)
        flush()
    root = next(s for s in stub.spans if s["name"] == "chat")
    assert root["status_code"] == ERROR
    assert root["attributes"]["error.type"] in ("GeneratorExit", "CancelledError")
    # Review: the stream's own cleanup (the interrupted record) ran INSIDE the span.
    assert seen and seen[0] is not None and seen[0].span_id == root["span_id"], seen


async def test_off_the_stream_entry_returns_the_generator_itself():
    """Telemetry off: the method returns the INNER async generator itself — no wrapper layer
    (off means unchanged). On, it returns the tracing wrapper around it."""
    from millm import telemetry
    from millm.services.inference_service import InferenceService

    for name in ("create_chat_completion", "stream_chat_completion", "create_text_completion"):
        assert getattr(InferenceService, name).__millm_inference_span__ in ("chat",
                                                                           "text_completion")
    inner = InferenceService.stream_chat_completion.__wrapped__
    request = ChatCompletionRequest(**{**cap.BODY, "stream": True})
    assert telemetry._PIPELINE is None
    off = InferenceService.stream_chat_completion(object(), request)
    try:
        assert off.ag_code is inner.__code__, "telemetry off wrapped the stream"
    finally:
        await off.aclose()


async def test_on_the_stream_entry_returns_the_tracing_wrapper(stub, telemetry_on):
    from millm.services.inference_service import InferenceService

    telemetry_on()
    inner = InferenceService.stream_chat_completion.__wrapped__
    request = ChatCompletionRequest(**{**cap.BODY, "stream": True})
    on = InferenceService.stream_chat_completion(object(), request)
    try:
        assert on.ag_code is not inner.__code__
    finally:
        await on.aclose()


async def test_a_client_model_name_that_is_not_the_resident_never_leaves(stub, telemetry_on):
    """Review: `request.model` is client text. A path that reaches an entry point without
    resolving the name exports no model attribute at all."""
    from millm.services.inference_service import InferenceService

    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        request = ChatCompletionRequest(**{**cap.BODY, "model": "my card is SENTINEL MODEL"})
        # The service refuses the name itself (one guard); the span must not carry it either.
        with pytest.raises(Exception) as refused:
            await InferenceService.create_chat_completion(h.svc, request)
        assert type(refused.value).__name__ == "ModelNotFoundError"
        await h.svc.drain_probe_records(10)
        flush()
    root = next(s for s in stub.spans if s["name"] == "chat")
    assert root["attributes"]["error.type"] == "ModelNotFoundError"
    assert "gen_ai.request.model" not in root["attributes"]
    assert b"SENTINEL" not in stub.all_bytes()
