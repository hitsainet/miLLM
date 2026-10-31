"""The request's trace (FR-38.14; D38.19 – D38.21; FTDD §7.2).

* `TraceContextMiddleware` (pure ASGI, like `millm/api/retry_after.py`) reads the client's W3C
  `traceparent` / `tracestate` with the OpenTelemetry API's propagator — API only, never the SDK —
  into `_REQUEST_TRACE`. A missing or malformed header is never an error and never borrows another
  request's context: the variable is simply `None`.
* `_CURRENT_SPAN` is miLLM's own open inference (or probe-evaluate) span for this request, set by
  `millm.telemetry.inference_span` / `child_span`.
* `capture_trace()` is THE ONE CAPTURE POINT a probe context's trace comes from (called once, in
  `InferenceService._attach_probe_meta`).

⚠ CLIENT-SUPPLIED IDS AUTHORIZE NOTHING. They are carried only as identifiers matching the W3C
pattern (validated by `TraceRef`).
"""

from __future__ import annotations

import contextvars
import re
import secrets
from dataclasses import dataclass
from typing import Any

_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True)
class TraceRef:
    """A W3C trace reference: 32 and 16 lowercase hex characters, never all zeros.

    `span_id` is MILLM'S span (the one a result was evaluated in), or `None` when miLLM opened
    none. `parent_span_id`, `sampled` and `tracestate` describe the CLIENT's incoming context.
    """

    trace_id: str
    span_id: str | None = None
    parent_span_id: str | None = None
    sampled: bool = True
    tracestate: str | None = None

    def __post_init__(self) -> None:
        if not _TRACE_ID.match(self.trace_id) or set(self.trace_id) == {"0"}:
            raise ValueError("trace_id must be 32 lowercase hex characters, not all zero")
        for value in (self.span_id, self.parent_span_id):
            if value is not None and (not _SPAN_ID.match(value) or set(value) == {"0"}):
                raise ValueError("span_id must be 16 lowercase hex characters, not all zero")


@dataclass
class SpanRef:
    """miLLM's open span: its ids, and the SDK span when traces export (else `None`)."""

    trace_id: str
    span_id: str | None
    sampled: bool = True
    sdk_span: Any = None
    #: `inference` for the request's one inference span (`inference_span` never nests one).
    kind: str = "span"
    #: Attributes set when the span ends (the probe-evaluate span's result counts).
    end_attrs: dict[str, Any] | None = None


#: The trace of the request being served, from the client's `traceparent` (or `None`).
#: Request-scoped like every per-request ContextVar in `inference_service`; a Batch row resets it
#: beside `BATCH_ROW.set` (FTID X-01), never in `pre_generation`, which the live routes also call.
_REQUEST_TRACE: contextvars.ContextVar[TraceRef | None] = contextvars.ContextVar(
    "millm_request_trace", default=None
)

#: miLLM's own innermost open span for this request (or `None`).
_CURRENT_SPAN: contextvars.ContextVar[SpanRef | None] = contextvars.ContextVar(
    "millm_current_span", default=None
)


def new_trace_id() -> str:
    while True:
        value = secrets.token_hex(16)
        if set(value) != {"0"}:
            return value


def capture_trace() -> TraceRef | None:
    """THE ONE CAPTURE POINT for a probe context's trace (called in `_attach_probe_meta`).

    In order: the open miLLM span's ids; else the client's incoming trace (no span of ours);
    else, when any signal is on, a local trace id minted once for this request; else `None`
    (D38.21: with every switch off and no parent, a result's `trace_id` stays null).
    """
    span = _CURRENT_SPAN.get()
    if span is not None:
        return TraceRef(span.trace_id, span.span_id, sampled=span.sampled)
    incoming = _REQUEST_TRACE.get()
    if incoming is not None:
        return incoming
    from millm import telemetry

    if telemetry.any_signal_on():
        local = TraceRef(new_trace_id())
        _REQUEST_TRACE.set(local)
        return local
    return None


def parse_traceparent(traceparent: str | None, tracestate: str | None) -> TraceRef | None:
    """The client's context through the API's W3C propagator, or `None`. Never raises."""
    if not traceparent:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        carrier = {"traceparent": traceparent}
        if tracestate:
            carrier["tracestate"] = tracestate
        ctx = TraceContextTextMapPropagator().extract(carrier)
        sc = trace.get_current_span(ctx).get_span_context()
        if not sc.is_valid:
            return None
        state = sc.trace_state.to_header() if sc.trace_state else None
        return TraceRef(
            trace_id=format(sc.trace_id, "032x"),
            parent_span_id=format(sc.span_id, "016x"),
            sampled=bool(sc.trace_flags & 0x01),
            tracestate=state or None,
        )
    except Exception:  # noqa: BLE001 - a malformed header is never an error
        return None


class TraceContextMiddleware:
    """Pure ASGI: parse `traceparent` / `tracestate` into `_REQUEST_TRACE` for the request.

    ⚠ ALWAYS SET, EVEN TO `None`. A request with no (or a malformed) header must not inherit a
    context set earlier in the same task; the variable is set for every HTTP request and reset
    when it returns. SSE streams pass through untouched (nothing wraps `send`).
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        parent: list[str] = []
        state: list[str] = []
        for name, value in scope.get("headers") or ():
            if name == b"traceparent":
                parent.append(value.decode("latin-1"))
            elif name == b"tracestate":
                state.append(value.decode("latin-1"))
        # Two `traceparent` headers are malformed (W3C §3.2): no context, never one of them.
        ref = parse_traceparent(parent[0], ",".join(state) or None) if len(parent) == 1 else None
        token = _REQUEST_TRACE.set(ref)
        try:
            await self.app(scope, receive, send)
        finally:
            _REQUEST_TRACE.reset(token)
