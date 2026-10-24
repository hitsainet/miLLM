"""The request's trace (FR-38.14; FTDD §7.2; FTID X-01; FTASKS 8.1 – 8.3).

AT02's three cases (a known, a missing and a malformed `traceparent`) through the real chat route
on the plain-thread path and the continuous-batching path; interleaved and concurrent requests
never share a trace; every Batch row gets its own trace, never the trace of the request that
created the batch task; and a live chat KEEPS the client's `trace_id` (the X-01 guard).

Controls: M14 delete the capture line in `_attach_probe_meta` (plain and CBM tests red); M25 delete
one Batch reset; X-01 mutation: reset `_REQUEST_TRACE` in `pre_generation` (the live case red).
"""

from __future__ import annotations

import contextlib

import pytest

from millm.telemetry.context import _REQUEST_TRACE, TraceRef, parse_traceparent
from tests.support import capture_pre038_goldens as cap
from tests.unit.telemetry.conftest import flush

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT = "00f067aa0ba902b7"
KNOWN = f"00-{TRACE}-{PARENT}-01"
MALFORMED = [
    "00-zzzz-00f067aa0ba902b7-01", f"00-{'0' * 32}-{PARENT}-01", f"00-{TRACE}-{'0' * 16}-01",
    f"ff-{TRACE}-{PARENT}-01", "garbage",
]


def _cbm(h):
    return cap._cbm(h.svc)


PATHS = {"plain": lambda h: contextlib.nullcontext(), "cbm": _cbm}


def _traces(h, response) -> set:
    body = {e.get("trace_id") for e in response.json()["millm_probe_verdicts"]}
    return body


@pytest.mark.parametrize("path", sorted(PATHS))
async def test_at02_known_traceparent_is_kept(path):
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA, cap.BETA])
        with PATHS[path](h):
            r = h.post_chat(cap.BODY, {"traceparent": KNOWN})
        assert r.status_code == 200, r.text
        assert _traces(h, r) == {TRACE}
        rows = await h.event_rows(r.json()["id"])
        assert {row["trace_id"] for row in rows} == {TRACE}
        # telemetry off: miLLM opened no span, so no span id is claimed.
        assert {e.get("span_id") for e in r.json()["millm_probe_verdicts"]} == {None}


@pytest.mark.parametrize("path", sorted(PATHS))
async def test_at02_missing_traceparent_is_null_while_off(path):
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        with PATHS[path](h):
            r = h.post_chat(cap.BODY, {})
        assert r.status_code == 200
        assert _traces(h, r) == {None}


@pytest.mark.parametrize("value", MALFORMED)
async def test_at02_malformed_traceparent_is_never_an_error(value):
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        r = h.post_chat(cap.BODY, {"traceparent": value})
        assert r.status_code == 200, r.text
        assert _traces(h, r) == {None}


async def test_interleaved_requests_never_share_a_trace():
    """Alternating clients, then one with no header: each request reads only its own context.
    (Concurrent requests are serialised by the one admission slot; the middleware sets the
    variable per request and resets it, so nothing survives a request.)"""
    ids = [f"{i:032x}" for i in range(1, 3)]
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        seen = []
        for tid in (ids[0], ids[1], ids[0], None):
            headers = {"traceparent": f"00-{tid}-{PARENT}-01"} if tid else {}
            seen.append(_traces(h, h.post_chat(cap.BODY, headers)))
        assert seen == [{ids[0]}, {ids[1]}, {ids[0]}, {None}]
        assert _REQUEST_TRACE.get() is None


async def test_batch_rows_get_their_own_trace_never_the_creators():
    """M25: delete one Batch reset → the row carries the creating request's trace."""
    outer = TraceRef(TRACE, parent_span_id=PARENT)
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        token = _REQUEST_TRACE.set(outer)
        try:
            lines = [await h.batch_line(cap.BODY) for _ in range(2)]
        finally:
            _REQUEST_TRACE.reset(token)
    for line in lines:
        verdicts = line["response"]["body"]["millm_probe_verdicts"]
        assert verdicts and {v.get("trace_id") for v in verdicts} == {None}, verdicts


async def test_batch_rows_each_mint_a_trace_when_export_is_on(stub, telemetry_on):
    telemetry_on()
    outer = TraceRef(TRACE, parent_span_id=PARENT)
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        token = _REQUEST_TRACE.set(outer)
        try:
            lines = [await h.batch_line(cap.BODY) for _ in range(2)]
        finally:
            _REQUEST_TRACE.reset(token)
        flush()
    traces = [{v["trace_id"] for v in line["response"]["body"]["millm_probe_verdicts"]}
              for line in lines]
    assert all(len(t) == 1 for t in traces)
    first, second = (next(iter(t)) for t in traces)
    assert first != second and TRACE not in (first, second)
    roots = [s for s in stub.spans if s["name"] == "chat"]
    assert len(roots) == 2 and all(s["parent_span_id"] == "" for s in roots)
    assert {s["attributes"].get("millm.batch.id") for s in roots} == {"batch_0123456789abcdef"}


async def test_with_traces_on_the_client_span_is_the_parent(stub, telemetry_on):
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        r = h.post_chat(cap.BODY, {"traceparent": KNOWN, "tracestate": "vendor=SENTINEL-TS"})
        assert r.status_code == 200
        flush()
    root = next(s for s in stub.spans if s["name"] == "chat")
    assert root["trace_id"] == TRACE and root["parent_span_id"] == PARENT
    assert "SENTINEL-TS" not in str(stub.spans), "the client's tracestate is not exported"


def test_the_parser_returns_the_client_context():
    ref = parse_traceparent(KNOWN, "a=b")
    assert (ref.trace_id, ref.span_id, ref.parent_span_id, ref.sampled) == (TRACE, None, PARENT,
                                                                           True)
    assert parse_traceparent(f"00-{TRACE}-{PARENT}-00", None).sampled is False
    for value in MALFORMED + [None, ""]:
        assert parse_traceparent(value, None) is None


async def test_packed_rows_each_run_with_their_own_trace(monkeypatch):
    """M25 (packing.py, both sites): a packed scoring row and a packed embedding row run with
    `_REQUEST_TRACE` reset, never the creating request's. Read at the row's provenance call."""
    import json
    from types import SimpleNamespace

    import millm.api.provenance as provenance
    from millm.services.batch.executors import RowExecutor
    from millm.services.batch.packing import run_packed_embeddings, run_packed_scoring
    from tests.unit.f25_fixtures import model_row

    seen: list = []
    real_post = provenance.post_generation

    def recording(*args, **kwargs):
        seen.append(_REQUEST_TRACE.get())
        return real_post(*args, **kwargs)

    monkeypatch.setattr(provenance, "post_generation", recording)
    outer = TraceRef(TRACE, parent_span_id=PARENT)
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])

        def executor(endpoint, bodies):
            lines = {i: json.dumps({"custom_id": f"r{i}", "method": "POST", "url": endpoint,
                                    "body": body}).encode() for i, body in bodies.items()}

            async def read_line(row):
                return lines[row.line_no]

            return RowExecutor(batch_id="batch_0123456789abcdef", endpoint=endpoint,
                               inference=h.svc, model_row=model_row(), read_line=read_line,
                               session_factory=h.factory)

        rows = [SimpleNamespace(line_no=i, custom_id=f"r{i}", kind="scoring") for i in (1, 2)]
        scoring = executor("/v1/completions", {i: {"model": "tiny", "prompt": "w1 w2",
                                                   "max_tokens": 1, "logprobs": 1}
                                               for i in (1, 2)})
        embedding = executor("/v1/embeddings", {i: {"model": "tiny", "input": "w1 w2"}
                                                for i in (1, 2)})

        async def fake_embed(requests, max_rows):
            return [([[0.0, 1.0]], [2]) for _ in requests]

        monkeypatch.setattr(h.svc, "_embed_requests_packed", fake_embed)
        monkeypatch.setattr(h.svc, "_embedding_response",
                            lambda request, vectors, counts: SimpleNamespace(
                                model_dump=lambda mode="json": {"data": vectors}))
        token = _REQUEST_TRACE.set(outer)
        try:
            scored = await run_packed_scoring(scoring, rows)
            embedded = await run_packed_embeddings(
                embedding, [SimpleNamespace(line_no=i, custom_id=f"r{i}", kind="embedding")
                            for i in (1, 2)])
        finally:
            _REQUEST_TRACE.reset(token)
    assert scored and len(embedded) == 2, "both packed paths ran"
    assert seen == [None, None, None, None], seen


async def test_two_traceparent_headers_are_no_context():
    """W3C §3.2 (review R1 of S2): more than one `traceparent` is malformed, so the request has NO
    context — never the first one. Driven through the middleware itself (an HTTP client's header
    dict cannot repeat a name). Control: take `parent[0]` whenever any is present → red."""
    from millm.telemetry.context import TraceContextMiddleware

    seen = []

    async def app(scope, receive, send):
        seen.append(_REQUEST_TRACE.get())

    other = f"00-{'a' * 32}-{PARENT}-01"
    middleware = TraceContextMiddleware(app)
    for headers in ([(b"traceparent", KNOWN.encode())],
                    [(b"traceparent", KNOWN.encode()), (b"traceparent", other.encode())]):
        await middleware({"type": "http", "headers": headers}, None, None)
    assert seen[0] is not None and seen[0].trace_id == TRACE
    assert seen[1] is None, seen[1]
    assert _REQUEST_TRACE.get() is None
