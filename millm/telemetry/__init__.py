"""miLLM telemetry (Feature 038): probe coverage counters always; OpenTelemetry export OFF by default.

⚠ OFF MEANS FREE, BY CONSTRUCTION (D38.17). This package imports only `settings`, `context`,
`allowlist`, `counters`, `health` and `lifecycle`, none of which imports `opentelemetry.sdk*` or
an exporter. `pipeline` and `logs` (the SDK users) are imported inside `configure()` and only when
a signal is switched on AND its configuration is accepted. With every switch off each entry point
below returns at its first line (`_PIPELINE is None`). T-S2-01 checks `sys.modules` in a
subprocess after a chat, a stream and a score.

⚠ NOTHING LEAVES UNLESS THE OPERATOR TURNS IT ON (D31, D-08). Three switches
(`TELEMETRY_TRACES|METRICS|LOGS_ENABLED`), an endpoint the operator names, environment only, never
changeable by a route. And never user text or secrets: every attribute passes `allowlist.filter`
before it is queued (FR-38.21).

Public API, all no-ops when off: `configure`, `shutdown`, `capture_trace`, `inference_span`,
`child_span`, `record_span`, `emit_results`, `emit_lifecycle`, `observe`, plus the decorators
`traced_entry` / `traced_stream_entry` for the three `InferenceService` entry points.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from millm.telemetry import allowlist, counters, health
from millm.telemetry.context import (
    _CURRENT_SPAN,
    _REQUEST_TRACE,
    SpanRef,
    TraceContextMiddleware,
    TraceRef,
    capture_trace,
)
from millm.telemetry.lifecycle import error_code, record_for
from millm.telemetry.settings import TelemetrySettings, load

logger = logging.getLogger(__name__)

#: The semantic-conventions schema this export follows (OT-D1; the pinned 0.66b1 release).
SEMCONV_SCHEMA_URL = "https://opentelemetry.io/schemas/1.44.0"

__all__ = [
    "TraceContextMiddleware", "TraceRef", "SpanRef", "capture_trace", "configure", "shutdown",
    "inference_span", "child_span", "note_execution", "record_span", "emit_results", "emit_lifecycle", "observe",
    "traced_entry", "traced_stream_entry", "current_settings", "any_signal_on",
]

_SETTINGS: TelemetrySettings | None = None
#: The running pipeline, or None. ⚠ None whenever every signal is off: the one test every entry
#: point makes first.
_PIPELINE: Any = None


def current_settings() -> TelemetrySettings:
    global _SETTINGS
    if _SETTINGS is None:
        from millm.core.config import settings

        _SETTINGS = load(settings)
    return _SETTINGS


def any_signal_on() -> bool:
    return _PIPELINE is not None


def sdk_version() -> str | None:
    """The installed SDK's version, from its metadata (never by importing it)."""
    try:
        from importlib.metadata import version

        return version("opentelemetry-sdk")
    except Exception:  # noqa: BLE001
        return None


# ── lifecycle ────────────────────────────────────────────────────────────────────


def configure(source: Any = None) -> None:
    """Resolve the settings and, only if a signal is on and accepted, start the pipeline.

    Called first thing in `lifespan`. Never raises: a failure leaves the enabled signals
    `unavailable` with the error's CLASS (`telemetry_configure_failed` in the log).
    """
    global _SETTINGS, _PIPELINE
    shutdown(0.0)
    from millm import __version__

    if source is None:
        from millm.core.config import settings as source
    cfg = load(source)
    _SETTINGS = cfg
    counters.STORE.series_cap = cfg.series_cap
    for name, state in health.SIGNALS.items():
        on = bool(cfg.switches.get(name))
        state.reset(enabled=on, refusal=cfg.refusal if on else None,
                    capacity=cfg.queue_size if on else 0)
    if not any(cfg.enabled(s) for s in cfg.switches):
        if cfg.refusal is not None:
            logger.warning("telemetry_configuration_refused reason=%s", cfg.refusal)
        return
    try:
        from millm.telemetry.pipeline import Pipeline

        pipeline = Pipeline(cfg, __version__, SEMCONV_SCHEMA_URL)
        pipeline.start()
        _PIPELINE = pipeline
        logger.info("telemetry_export_on signals=%s", ",".join(pipeline.signals))
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry_configure_failed error=%s", type(exc).__name__)
        for name, state in health.SIGNALS.items():
            if cfg.switches.get(name):
                state.reset(enabled=True, refusal=type(exc).__name__, capacity=0)


def shutdown(deadline_s: float | None = None) -> None:
    """Stop the pipeline: drain until the deadline (`TELEMETRY_DRAIN_DEADLINE_S`), count the
    rest as dropped. Called in `lifespan` AFTER the probe-record tasks are drained."""
    global _PIPELINE
    pipeline, _PIPELINE = _PIPELINE, None
    if pipeline is None:
        return
    try:
        pipeline.shutdown(pipeline.cfg.drain_deadline_s if deadline_s is None else deadline_s)
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry_shutdown_failed error=%s", type(exc).__name__)


# ── spans ────────────────────────────────────────────────────────────────────────


def _batch_id() -> str | None:
    try:
        from millm.services.batch.state import BATCH_ROW

        row = BATCH_ROW.get()
        return row[0] if row is not None else None
    except Exception:  # noqa: BLE001
        return None


@contextlib.contextmanager
def inference_span(operation: str, model: str | None = None,
                   **attrs: Any) -> Iterator[SpanRef | None]:
    """THE inference span of one request (FTDD §6.3). No content attributes, ever.

    ⚠ ONE PER REQUEST: inside an open inference span (the serial chat fallback calls
    `create_chat_completion` per conversation) this yields the open one instead of a second.
    Ends with `ERROR` status on an exception and on cancellation (`GeneratorExit`,
    `CancelledError`).
    """
    pipeline = _PIPELINE
    if pipeline is None:
        yield None
        return
    current = _CURRENT_SPAN.get()
    if current is not None and getattr(current, "kind", None) == "inference":
        yield current
        return
    ref = None
    try:
        span_attrs = {"gen_ai.operation.name": operation, "gen_ai.request.model": model,
                      "millm.batch.id": _batch_id(), **attrs}
        ref = pipeline.start_span(operation, span_attrs, _REQUEST_TRACE.get())
        ref.kind = "inference"
    except Exception as exc:  # noqa: BLE001 - an observer never breaks a request
        logger.warning("telemetry_span_failed error=%s", type(exc).__name__)
    if ref is None:
        yield None
        return
    _CURRENT_SPAN.set(ref)
    error: str | None = None
    try:
        yield ref
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        # ⚠ `set(previous)`, not a token reset: a stream's span may close in a different task
        # than the one that opened it (sse-starlette), where a token reset raises.
        _CURRENT_SPAN.set(current)
        with contextlib.suppress(Exception):
            pipeline.end_span(ref, error=error)


@contextlib.contextmanager
def child_span(name: str, **attrs: Any) -> Iterator[SpanRef | None]:
    """A child of the open inference span (`millm.probe.evaluate`). Yields its ref, whose ids
    every result built inside carries; `None` when traces are off or no span is open."""
    pipeline = _PIPELINE
    parent = _CURRENT_SPAN.get() if pipeline is not None else None
    if pipeline is None or parent is None or getattr(pipeline, "tracer", None) is None:
        yield None
        return
    ref = None
    try:
        ref = pipeline.start_span(name, attrs, parent)
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry_span_failed error=%s", type(exc).__name__)
    if ref is None:
        yield None
        return
    error: str | None = None
    try:
        yield ref
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        with contextlib.suppress(Exception):
            pipeline.end_span(ref, error=error, attrs=getattr(ref, "end_attrs", None))


def note_execution(execution: Any) -> None:
    """Put how the request is served (`millm.engine`, `millm.execution_mode`) on the open
    INFERENCE span (FR-38.15; 12.5a F2). Called by `_probe_begin` / `_probe_begin_detached`,
    which every generation path calls with its own `ExecutionFacts` whether or not a probe is
    armed, so the span says it on every path. A later call on the same span (the serial chat
    fallback) overwrites the earlier one. Never raises; values pass the span allowlist."""
    pipeline = _PIPELINE
    if pipeline is None or execution is None or getattr(pipeline, "tracer", None) is None:
        return
    ref = _CURRENT_SPAN.get()
    if ref is None or getattr(ref, "kind", None) != "inference" or ref.sdk_span is None:
        return
    try:
        attrs = {"millm.engine": getattr(execution, "engine", None),
                 "millm.execution_mode": getattr(execution, "mode", None)}
        kept, _ = allowlist.filter("span", attrs)
        ref.sdk_span.set_attributes(kept)
    except Exception as exc:  # noqa: BLE001 - an observer never breaks a request
        logger.warning("telemetry_span_failed error=%s", type(exc).__name__)


def record_span(name: str, start_ns: int, end_ns: int, **attrs: Any) -> None:
    """A finished child span with explicit times (`millm.queue.wait`). Never raises."""
    pipeline = _PIPELINE
    if pipeline is None or getattr(pipeline, "tracer", None) is None:
        return
    parent = _CURRENT_SPAN.get()
    if parent is None:
        return
    try:
        ref = pipeline.start_span(name, attrs, parent, start_time=start_ns)
        pipeline.end_span(ref, end_time=end_ns)
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry_span_failed error=%s", type(exc).__name__)


def _resident_name(requested: Any) -> str | None:
    """The request's model name only when it NAMES the resident model (review): `request.model`
    is client text, and a path that reached an entry point without resolving it must not export
    whatever the client wrote. Anything else is omitted, never sent."""
    if not isinstance(requested, str):
        return None
    try:
        from millm.ml.model_loader import LoadedModelState

        current = LoadedModelState().current
        name = getattr(current, "model_name", None)
        return requested if name is not None and requested == name else None
    except Exception:  # noqa: BLE001
        return None


def _span_entry(operation: str, request: Any) -> Any:
    model = _resident_name(getattr(request, "model", None))
    return inference_span(operation, model,
                          **{"millm.stream": bool(getattr(request, "stream", False))})


def traced_entry(operation: str) -> Any:
    """Decorator for a coroutine entry point: the request runs inside ONE inference span.

    Off, the wrapper calls the method directly (one attribute read). The paths guard asserts by
    AST that every discovered entry point carries this decorator or `traced_stream_entry`.
    """
    def decorate(fn: Any) -> Any:
        @functools.wraps(fn)
        async def wrapper(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            if _PIPELINE is None:
                return await fn(self, request, *args, **kwargs)
            with _span_entry(operation, request):
                return await fn(self, request, *args, **kwargs)

        wrapper.__millm_inference_span__ = operation  # type: ignore[attr-defined]
        return wrapper

    return decorate


def traced_stream_entry(operation: str) -> Any:
    """Decorator for the streaming entry point (an async generator).

    ⚠ OFF, THE GENERATOR ITSELF IS RETURNED, unwrapped, so an unarmed, telemetry-off stream is
    the same object it was before 038. On, the span covers the stream until its `finally`; the
    inner generator is CLOSED (`aclose`) when the outer one is, so its `finally` — the probe
    record and the interrupted marking — still runs on a client disconnect.
    """
    def decorate(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            agen = fn(self, request, *args, **kwargs)
            if _PIPELINE is None:
                return agen
            return _traced_stream(agen, _span_entry(operation, request))

        wrapper.__millm_inference_span__ = operation  # type: ignore[attr-defined]
        return wrapper

    return decorate


async def _traced_stream(agen: Any, span_cm: Any) -> Any:
    # ⚠ The inner generator is closed INSIDE the span: its `finally` (interrupted marking, the
    # probe record) runs while the span is still current, and the span's duration includes it.
    with span_cm:
        try:
            async for chunk in agen:
                yield chunk
        finally:
            await agen.aclose()


# ── counters ─────────────────────────────────────────────────────────────────────


def observe(results: Any, admitted_ids: set[str] | None = None) -> None:
    """Count minted results (called by the builder). With metrics on, also queue the duration
    and margin measurements for the export thread. Never raises."""
    try:
        results = list(results)
        counters.STORE.observe(results, admitted_ids)
        pipeline = _PIPELINE
        if pipeline is None or "metrics" not in pipeline.signals:
            return
        items = []
        for r in results:
            if getattr(r, "kind", "probe") != "probe":
                continue  # 039 S2: a rule result has no duration or margin; counted above
            pid, window, engine = r.probe_id, r.window, r.engine
            if admitted_ids is not None and pid not in admitted_ids:
                pid = None
            dims = {"probe_id": pid, "window": window}
            if r.evaluation_ms is not None:
                items.append(("millm.probe.evaluation.duration", float(r.evaluation_ms),
                              {**dims, "engine": engine}))
            if r.outcome == "completed" and r.threshold is not None and r.score is not None:
                items.append(("millm.probe.score_margin", float(r.score - r.threshold), dims))
        pipeline.enqueue_measurements(items)
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry_observe_failed error=%s", type(exc).__name__)


# ── log records ──────────────────────────────────────────────────────────────────


def _ns(rfc3339: str | None) -> int:
    try:
        moment = datetime.strptime(rfc3339 or "", "%Y-%m-%dT%H:%M:%S.%fZ")
        return int(moment.replace(tzinfo=UTC).timestamp() * 1_000_000) * 1000
    except ValueError:
        return time.time_ns()


def emit_results(results: Any, *, stateless_permitted: bool = False) -> int:
    """One `mi.probe.result` log record per RECORDED result (FR-38.16). Returns how many were
    queued. Live and batch results pass `TELEMETRY_RESULT_LOG_ORIGINS`; a stateless result leaves
    only when the caller passes the FR-38.17 gate (`stateless_permitted`). Unsampled (D38.15)."""
    pipeline = _PIPELINE
    if pipeline is None or "logs" not in pipeline.signals:
        return 0
    queued = 0
    try:
        from millm.api.schemas.probe_result import wire

        origins = pipeline.cfg.result_log_origins
        for result in results:
            data = result if isinstance(result, dict) else wire(result)
            origin = data.get("origin")
            if origin == "stateless":
                if not stateless_permitted:
                    continue
            elif origin not in origins:
                continue
            trace = None
            if data.get("trace_id"):
                # ⚠ `sampled=False`: a result does not know whether its trace was sampled, and a
                # record must not CLAIM a sampled trace the receiver may never have got (review).
                trace = TraceRef(data["trace_id"], data.get("span_id"), sampled=False)
            if isinstance(data.get("set"), dict):
                # 039: the `set` block's lists are not attributes (ids and rule-set refs); its
                # counts and digest are. Removed here, not dropped-and-counted by the filter.
                data = {**data, "set": {k: v for k, v in data["set"].items()
                                        if not isinstance(v, (list, tuple))}}
            attrs = allowlist.flatten(data, allowlist.RESULT_PREFIX)
            if pipeline.enqueue_log("mi.probe.result", "log_result", attrs, trace,
                                    _ns(data.get("event_time"))):
                queued += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry_emit_results_failed error=%s", type(exc).__name__)
    return queued


def emit_lifecycle(probe: Any, action: str, **fields: Any) -> None:
    """One `mi.probe.lifecycle` record (FR-38.20). Never raises; reads no free text."""
    pipeline = _PIPELINE
    if pipeline is None or "logs" not in pipeline.signals:
        return
    try:
        attrs = record_for(probe, action, **fields)
        pipeline.enqueue_log("mi.probe.lifecycle", "log_lifecycle", attrs, None, time.time_ns())
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry_emit_lifecycle_failed error=%s", type(exc).__name__)


def lifecycle_refused(probe: Any, action: str, exc: BaseException) -> None:
    """A refused (or failed) lifecycle step: the error's CODE, never its message."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    outcome = "refused" if isinstance(status, int) and status < 500 else "failed"
    emit_lifecycle(probe, action, outcome=outcome, code=error_code(exc))
