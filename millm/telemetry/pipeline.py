"""The export pipeline (FTDD §7.4, D38.35). Imported ONLY when a signal is switched on (D38.17).

* **Private providers, never the global ones.** A `TracerProvider` (sampler
  `ParentBased(TraceIdRatioBased(ratio))`) and a `MeterProvider` built here and never passed to
  `set_tracer_provider` / `set_meter_provider` / `set_logger_provider`, so FastAPI's native
  telemetry and the `mcp` package's built-in instrumentation stay dark (FTDD §2.4 U6, T-S2-02).
* **Filter, then a bounded non-blocking queue.** Every span (in `_QueueingProcessor.on_end`),
  every log record (`enqueue_log`) and every metric dimension (the counter store) passes the
  allowlist BEFORE `put_nowait`. A full queue drops the record and counts it (`dropped`); the
  request never waits (control M22).
* **One daemon export thread** drains the queues in batches of `TELEMETRY_BATCH_SIZE`, collects
  metrics every `TELEMETRY_METRIC_INTERVAL_S`, encodes with the SDK's OTLP protobuf encoders and
  POSTs over its own `requests.Session`, retrying a retryable failure with backoff until
  `TELEMETRY_RETRY_HORIZON_S`. A retry resends the same records, so `result_id` is stable.

⚠ WHY NOT THE SDK's `OTLP*Exporter` CLASSES (FTID discrepancy, recorded in the S2 controls).
In 1.45.1 they consult `OTEL_EXPORTER_OTLP_*` environment variables even when given explicit
arguments: `_resolve_headers` MERGES `OTEL_EXPORTER_OTLP_HEADERS` into the explicit headers, the
certificate, client key and compression fall back to the environment when an argument is `None`,
and a credential-provider session is loaded from the environment. FTDD §7.4 requires that no
`OTEL_EXPORTER_*` variable is consulted. They also report a 401 only as a log line, and health must
say `unavailable` on 401/403. So the pipeline uses the same package's ENCODERS (the wire format)
and sends the bytes itself, with every argument explicit and `trust_env = False` (no proxy, CA
bundle or netrc picked up from the environment).

⚠ NO MESSAGE EVER LEAVES THROUGH HEALTH. Failures are recorded by class (`ConnectionError`,
`HTTP401`), never by text.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from opentelemetry import context as otel_context
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.exporter.otlp.proto.common.metrics_encoder import encode_metrics
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.metrics import NoOpMeterProvider, Observation
from opentelemetry.sdk.metrics import AlwaysOffExemplarFilter, MeterProvider
from opentelemetry.sdk.metrics._internal.measurement import Measurement
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.metrics.view import ExponentialBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from millm.telemetry import allowlist, counters, health
from millm.telemetry.context import SpanRef, TraceRef, new_trace_id
from millm.telemetry.settings import TelemetrySettings

logger = logging.getLogger(__name__)

PATHS = {"traces": "/v1/traces", "metrics": "/v1/metrics", "logs": "/v1/logs"}
RETRYABLE = frozenset({408, 429, 502, 503, 504})
SCOPE = "millm"


def signal_url(endpoint: str, signal: str) -> str:
    """The base URL with `/v1/<signal>` appended to its PATH (review R1 of S2). A plain string
    append put the suffix after a query (`…/base?tenant=a` → `…/base?tenant=a/v1/logs`), so a
    receiver keyed by query got every signal at the base path with a corrupted query. The query
    is kept, the fragment (never sent anyway) dropped; userinfo is left to `requests`."""
    parts = urlsplit(endpoint.strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + PATHS[signal],
                       parts.query, ""))


def _hex_trace(value: int) -> str:
    return format(value, "032x")


def _hex_span(value: int) -> str:
    return format(value, "016x")


#: The largest response body read (raw, never decoded) so that the connection can be reused.
MAX_RESPONSE_BODY = 4096


def _discard(response: Any) -> None:
    """Close a response without ever decoding its body. A small body with a declared length is
    read RAW (`decode_content=False`) so the pooled connection is reused; anything else (larger,
    chunked, undeclared) closes the connection instead of reading it."""
    try:
        length = int(response.headers.get("Content-Length", "-1"))
    except (TypeError, ValueError):
        length = -1
    try:
        if 0 <= length <= MAX_RESPONSE_BODY:
            response.raw.read(length + 1, decode_content=False)
    except Exception:  # noqa: BLE001 - the status was already read; closing is all that is left
        pass
    finally:
        response.close()


def _record_signed(hist: Any, value: float, attrs: dict[str, Any]) -> None:
    """`hist.record`, except that a NEGATIVE value is kept (review R1 of S2).

    The SDK's `Histogram.record` (1.45.1) logs and DROPS any amount below zero, but
    `millm.probe.score_margin` is signed by definition (D38.41: score − threshold, and most
    results sit BELOW their threshold): recorded through `record`, the histogram held only the
    crossings. The base-2 exponential aggregation keeps negative buckets, so a finite negative
    value goes to the instrument's measurement consumer directly, with an EMPTY context (never the
    global one; exemplars are off regardless). Pinned by
    `test_a_negative_margin_is_exported_not_dropped`; the lockstep pin guards the private names.
    """
    if value >= 0 or not math.isfinite(value):
        hist.record(value, attributes=attrs)
        return
    hist._measurement_consumer.consume_measurement(
        Measurement(value, time.time_ns(), hist, otel_context.Context(), attrs)
    )


class _QueueingProcessor(SpanProcessor):
    """`on_end` filters the finished span and enqueues it. No SDK batch processor: it drops
    silently, and a drop here must be counted."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def on_start(self, span: Any, parent_context: Any = None) -> None:  # noqa: ARG002
        return None

    def on_end(self, span: ReadableSpan) -> None:
        if span.name not in allowlist.SPAN_NAMES:
            allowlist._count(1)
            return
        kept, _dropped = allowlist.filter("span", dict(span.attributes or {}))
        # ⚠ Rebuilt, not passed through: no events (an exception event carries its message), no
        # links, and a status with NO description (the SDK puts exception text there).
        clean = ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=kept,
            events=(),
            links=(),
            kind=span.kind,
            status=Status(span.status.status_code),
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )
        self.pipeline.enqueue("traces", clean)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG001,ARG002,ARG004 - the SDK interface
        return True


class Pipeline:
    """Providers, queues, the export thread and the sender, for the enabled signals only."""

    def __init__(self, cfg: TelemetrySettings, version: str, schema_url: str) -> None:
        self.cfg = cfg
        self.version = version
        self.schema_url = schema_url
        self.signals = tuple(s for s in ("traces", "metrics", "logs") if cfg.enabled(s))
        attrs = {
            "service.name": cfg.service_name,
            "service.version": version,
            "service.instance.id": cfg.service_instance_id,
            "deployment.environment.name": cfg.deployment_environment,
        }
        kept, _ = allowlist.filter("resource", attrs)
        # ⚠ `Resource(attrs)`, never `Resource.create`, which merges OTEL_RESOURCE_ATTRIBUTES and
        # runs the process detector (the process detector exports the command line).
        self.resource = Resource(kept, schema_url=schema_url)
        self.queues: dict[str, queue.Queue[Any]] = {
            s: queue.Queue(maxsize=cfg.queue_size) for s in self.signals
        }
        self._accepting = True
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._drain_until: float | None = None
        self._retired = False
        #: Per signal, a batch waiting to be retried: (body, n, deadline, attempt, next_at). A
        #: failing signal waits HERE, so it never holds the one thread from the other signals.
        self._retry: dict[str, tuple[bytes, int, float, int, float] | None] = {}
        self.session = requests.Session()
        self.session.trust_env = False
        self.tracer: Any = None
        self.tracer_provider: Any = None
        self.meter_provider: Any = None
        self._reader: Any = None
        self._histograms: dict[str, Any] = {}
        self._hist_series: dict[str, set[tuple[Any, ...]]] = {}
        if "traces" in self.signals:
            self.tracer_provider = TracerProvider(
                sampler=ParentBased(TraceIdRatioBased(cfg.sampler_ratio)),
                resource=self.resource,
                shutdown_on_exit=False,
                meter_provider=NoOpMeterProvider(),
            )
            self.tracer_provider.add_span_processor(_QueueingProcessor(self))
            self.tracer = self.tracer_provider.get_tracer(SCOPE, version, schema_url=schema_url)
        if "metrics" in self.signals:
            self._reader = InMemoryMetricReader()
            self.meter_provider = MeterProvider(
                metric_readers=[self._reader],
                resource=self.resource,
                shutdown_on_exit=False,
                # ⚠ No exemplars: a trace-based exemplar would attach trace and span ids to a
                # metric point (FR-18: no request/result/trace ids in metrics).
                exemplar_filter=AlwaysOffExemplarFilter(),
                views=[
                    View(instrument_name="millm.probe.score_margin",
                         aggregation=ExponentialBucketHistogramAggregation(max_scale=20)),
                ],
            )
            meter = self.meter_provider.get_meter(SCOPE, version, schema_url=schema_url)
            for instrument in counters.INSTRUMENTS:
                meter.create_observable_counter(
                    instrument, callbacks=[self._counter_callback(instrument)], unit="1"
                )
            meter.create_observable_counter(
                counters.OVERFLOW_INSTRUMENT, callbacks=[self._overflow_callback], unit="1"
            )
            self._histograms = {
                "millm.probe.evaluation.duration": meter.create_histogram(
                    "millm.probe.evaluation.duration", unit="ms"),
                "millm.probe.score_margin": meter.create_histogram(
                    "millm.probe.score_margin", unit="1"),
            }
            self._hist_series = {name: set() for name in self._histograms}
        self._thread = threading.Thread(
            target=self._run, name="millm-telemetry-export", daemon=True
        )

    # ── lifecycle ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        for signal in self.signals:
            q = self.queues[signal]
            health.SIGNALS[signal].queue_used = q.qsize
        self._thread.start()

    def shutdown(self, deadline_s: float) -> None:
        """Stop accepting, drain until the deadline, count the rest as dropped — and LOG what
        was lost (12.5a F6): the counters live in memory and die with the process, so a loss at
        shutdown that is only counted is a loss nobody can ever read."""
        before = {s: (health.SIGNALS[s].dropped, health.SIGNALS[s].failed) for s in self.queues}
        self._accepting = False
        self._drain_until = time.monotonic() + max(0.0, deadline_s)
        self._stop.set()
        self._wake.set()
        # One POST in flight can take the export timeout itself, so the join allows for it.
        self._thread.join(timeout=max(0.0, deadline_s) + self.cfg.export_timeout_s + 1.0)
        if self._thread.is_alive():
            # ⚠ RETIRED: whatever the still-running thread does from here reports to a detached
            # counter, never to the live health of a NEW configuration, and an item it takes
            # after this point was already counted dropped below (no double count).
            self._retired = True
        for signal, q in self.queues.items():
            left = q.qsize()
            if left:
                health.SIGNALS[signal].drop(left)
        # ⚠ Counts and signal names only: nothing of a lost record is ever written.
        for signal, (dropped0, failed0) in before.items():
            dropped = health.SIGNALS[signal].dropped - dropped0
            failed = health.SIGNALS[signal].failed - failed0
            if dropped or failed:
                logger.warning("telemetry_shutdown_lost signal=%s dropped=%d failed=%d",
                               signal, dropped, failed)
        try:
            if self.tracer_provider is not None:
                self.tracer_provider.shutdown()
            if self.meter_provider is not None:
                self.meter_provider.shutdown(timeout_millis=100)
        except Exception:  # noqa: BLE001
            pass
        self.session.close()

    # ── the enqueue (event loop) ─────────────────────────────────────────────────

    def enqueue(self, signal: str, item: Any) -> bool:
        """`put_nowait` or count a drop. ⚠ NEVER BLOCKS (FR-38.22, control M22)."""
        q = self.queues.get(signal)
        if q is None:
            return False
        if not self._accepting:
            health.SIGNALS[signal].drop(1)
            return False
        try:
            q.put_nowait(item)
        except queue.Full:
            health.SIGNALS[signal].drop(1)
            return False
        self._wake.set()
        return True

    # ── spans ────────────────────────────────────────────────────────────────────

    def _parent_context(self, parent: Any) -> Any:
        if isinstance(parent, SpanRef) and parent.sdk_span is not None:
            return trace_api.set_span_in_context(parent.sdk_span, otel_context.Context())
        if isinstance(parent, TraceRef) and parent.parent_span_id is not None:
            remote = SpanContext(
                trace_id=int(parent.trace_id, 16),
                span_id=int(parent.parent_span_id, 16),
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED if parent.sampled else 0),
                # ⚠ The client's `tracestate` is NOT carried into exported spans: it is opaque
                # vendor data the client controls, and nothing here can allowlist its values.
            )
            return trace_api.set_span_in_context(NonRecordingSpan(remote), otel_context.Context())
        # ⚠ An EMPTY context, never `get_current()`: the global API context is not ours.
        return otel_context.Context()

    def start_span(self, name: str, attrs: dict[str, Any], parent: Any,
                   start_time: int | None = None) -> SpanRef:
        if self.tracer is None:
            trace_id = getattr(parent, "trace_id", None) or new_trace_id()
            return SpanRef(trace_id, None, sampled=getattr(parent, "sampled", True))
        kept, _ = allowlist.filter("span", attrs)
        span = self.tracer.start_span(
            name, context=self._parent_context(parent), kind=SpanKind.INTERNAL,
            attributes=kept, start_time=start_time,
            record_exception=False, set_status_on_exception=False,
        )
        sc = span.get_span_context()
        return SpanRef(_hex_trace(sc.trace_id), _hex_span(sc.span_id),
                       sampled=bool(sc.trace_flags.sampled), sdk_span=span)

    @staticmethod
    def end_span(ref: SpanRef, *, error: str | None = None,
                 attrs: dict[str, Any] | None = None, end_time: int | None = None) -> None:
        span = ref.sdk_span
        if span is None:
            return
        if attrs:
            kept, _ = allowlist.filter("span", attrs)
            span.set_attributes(kept)
        if error is not None:
            kept, _ = allowlist.filter("span", {"error.type": error})
            span.set_attributes(kept)
            # ⚠ No description: the SDK's description is where exception text would go.
            span.set_status(Status(StatusCode.ERROR))
        span.end(end_time=end_time)

    # ── logs and histograms (event loop) ──────────────────────────────────────────

    def enqueue_log(self, event_name: str, signal_table: str, attrs: dict[str, Any],
                    trace: TraceRef | None, timestamp_ns: int) -> bool:
        """Filter on the loop (FR-38.21: before enqueue); the record is built on the thread."""
        if event_name not in allowlist.EVENT_NAMES:
            return False
        kept, _ = allowlist.filter(signal_table, attrs)
        item = (event_name, kept, getattr(trace, "trace_id", None), getattr(trace, "span_id", None),
                bool(getattr(trace, "sampled", False)), timestamp_ns, time.time_ns())
        return self.enqueue("logs", item)

    def enqueue_measurements(self, items: list[tuple[str, float, dict[str, Any]]]) -> None:
        # Each item its own `put_nowait`: every one that does not fit is counted as a drop.
        for item in items:
            self.enqueue("metrics", item)

    def _health(self, signal: str) -> health.SignalHealth:
        """The live health of `signal`, or a detached one once this pipeline is retired."""
        if self._retired:
            return health.SignalHealth(signal)
        return health.SIGNALS[signal]

    # ── the export thread ────────────────────────────────────────────────────────

    def _run(self) -> None:
        next_metrics = time.monotonic() + self.cfg.metric_interval_s
        while True:
            stopping = self._stop.is_set()
            if stopping:
                # ⚠ No retry once shutdown began: a dead receiver cannot hold the pod's
                # termination grace, and is never hammered.
                for signal, pending in list(self._retry.items()):
                    if pending is not None:
                        self._health(signal).failure("ShutdownBeforeRetry", pending[1])
                        self._retry[signal] = None
                if self._drain_until is not None and time.monotonic() >= self._drain_until:
                    return
            busy = False
            for signal in ("traces", "logs"):
                if signal in self.queues and self._service(signal):
                    busy = True
            if "metrics" in self.queues:
                self._record_measurements()
                if self._retry.get("metrics") is not None:
                    busy = self._service("metrics") or busy
                elif not stopping and time.monotonic() >= next_metrics:
                    self._export_metrics()
                    next_metrics = time.monotonic() + self.cfg.metric_interval_s
            if stopping and not busy:
                if "metrics" in self.queues:
                    self._export_metrics()  # the final collection, once
                return
            if not busy:
                waits = [p[4] - time.monotonic() for p in self._retry.values() if p is not None]
                self._wake.wait(timeout=max(0.01, min([0.25, *waits])))
                self._wake.clear()

    def _service(self, signal: str) -> bool:
        """One step for one signal: its due retry, or its next batch. True when it sent."""
        pending = self._retry.get(signal)
        if pending is not None:
            if time.monotonic() < pending[4]:
                return False
            body, n, deadline, attempt, _ = pending
            self._retry[signal] = None
            self._attempt(signal, body, n, deadline, attempt)
            return True
        if signal == "metrics":
            return False
        batch = self._take(self.queues[signal])
        if not batch:
            return False
        self._export(signal, batch)
        return True

    def _take(self, q: queue.Queue[Any]) -> list[Any]:
        batch: list[Any] = []
        while len(batch) < self.cfg.batch_size:
            try:
                batch.append(q.get_nowait())
            except queue.Empty:
                break
        return batch

    def _record_measurements(self) -> None:
        q = self.queues["metrics"]
        while True:
            try:
                name, value, attrs = q.get_nowait()
            except queue.Empty:
                return
            hist = self._histograms.get(name)
            if hist is None:
                continue
            kept, dropped = allowlist.filter("metric", attrs)
            key = tuple(sorted(kept.items()))
            series = self._hist_series[name]
            if dropped or (key not in series and len(series) >= self.cfg.series_cap):
                counters.STORE.count_overflow()
                kept = dict(counters.OVERFLOW_ATTRS)
            else:
                series.add(key)
            _record_signed(hist, value, kept)

    def _export_metrics(self) -> None:
        try:
            data = self._reader.get_metrics_data()
        except Exception as exc:  # noqa: BLE001
            self._health("metrics").failure(type(exc).__name__, 0)
            return
        if data is None:
            return
        points = sum(len(m.data.data_points) for rm in data.resource_metrics
                     for sm in rm.scope_metrics for m in sm.metrics)
        if not points:
            return
        try:
            body = encode_metrics(data).SerializeToString()
        except Exception as exc:  # noqa: BLE001
            self._health("metrics").failure(type(exc).__name__, points)
            return
        self._send("metrics", body, points)

    def _export(self, signal: str, batch: list[Any]) -> None:
        try:
            if signal == "traces":
                body = encode_spans(batch).SerializeToString()
            else:
                from millm.telemetry.logs import build_records

                body = encode_logs(build_records(batch, self.resource, self.version,
                                                 self.schema_url)).SerializeToString()
        except Exception as exc:  # noqa: BLE001
            self._health(signal).failure(type(exc).__name__, len(batch))
            return
        self._send(signal, body, len(batch))

    def _send(self, signal: str, body: bytes, n: int) -> None:
        self._attempt(signal, body, n, time.monotonic() + self.cfg.retry_horizon_s, 0)

    def _attempt(self, signal: str, body: bytes, n: int, deadline: float, attempt: int) -> None:
        """ONE POST. A retryable failure inside the horizon is parked in `_retry` with its
        backoff (the same bytes are resent, so `result_id` is stable); otherwise the records are
        counted `failed`. Counts by CLASS only."""
        cfg = self.cfg
        url = signal_url(cfg.endpoint, signal)  # type: ignore[arg-type]
        headers = {"Content-Type": "application/x-protobuf", **cfg.headers}
        cert = ((cfg.client_cert_file, cfg.client_key_file)
                if cfg.client_cert_file and cfg.client_key_file else None)
        verify: Any = cfg.ca_file or True
        state = self._health(signal)
        auth = False
        try:
            # ⚠ `stream=True` (review R1 of S2): only the status matters, and `requests` would
            # otherwise read and DECODE the whole body, so a huge or compressed (gzip-bomb)
            # response could hold the export thread and the server's memory (`_discard`).
            # `allow_redirects=False`: a redirect is a failure, never a second POST of the records
            # and the credential headers to another host.
            response = self.session.post(
                url, data=body, headers=headers, timeout=cfg.export_timeout_s,
                verify=verify, cert=cert, allow_redirects=False, stream=True,
            )
            code = response.status_code
            _discard(response)
            if 200 <= code < 300:
                state.success(n)
                return
            error, auth, retryable = f"HTTP{code}", code in (401, 403), code in RETRYABLE
        except requests.exceptions.RequestException as exc:
            error, retryable = type(exc).__name__, True
        except Exception as exc:  # noqa: BLE001
            error, retryable = type(exc).__name__, False
        now = time.monotonic()
        if not retryable or now >= deadline or self._stop.is_set():
            state.failure(error, n, auth=auth, final=True)
            return
        state.failure(error, n, auth=auth, final=False)
        pause = min(0.5 * 2 ** min(attempt, 8), 5.0, max(0.0, deadline - now))
        self._retry[signal] = (body, n, deadline, attempt + 1, now + pause)

    # ── callbacks ────────────────────────────────────────────────────────────────

    @staticmethod
    def _counter_callback(instrument: str) -> Any:
        def callback(options: Any) -> Any:  # noqa: ARG001,ARG002,ARG004 - the SDK interface
            for attrs, value in counters.STORE.snapshot()[instrument].items():
                yield Observation(value, dict(attrs))
        return callback

    @staticmethod
    def _overflow_callback(options: Any) -> Any:  # noqa: ARG001,ARG002,ARG004 - the SDK interface
        yield Observation(counters.STORE.series_overflow, {})

