"""An in-process OTLP/HTTP receiver for the telemetry tests (038 D38.51).

`OtlpStub` serves `POST /v1/traces|metrics|logs` on 127.0.0.1 from a background thread, decodes
each body with `opentelemetry-proto` and keeps:

* `raw` — every request body as received (the privacy tests search these BYTES for sentinels, so a
  value that slipped past decoding is still found);
* `spans`, `logs`, `metric_points` — decoded, flattened views;
* `scopes` — the instrumentation scope name of every span / log / metric received.

Modes: `healthy` (200), `slow` (sleeps `slow_s` then 200), `401`, `fail_first` (503 to the first
request of each signal, then 200), `fail_logs` (503 to every logs request, 200 to the rest),
`absent` (the socket is closed: connection refused), and two hostile receivers (review R1 of S2):
`redirect` (307 to `redirect_to`, which a sender must never follow) and `huge` (200 declaring a
1 GB body, of which only a few bytes arrive before the connection is held for `slow_s`).
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


def _value(any_value: Any) -> Any:
    kind = any_value.WhichOneof("value")
    return getattr(any_value, kind) if kind else None


def _attrs(kvs: Any) -> dict[str, Any]:
    return {kv.key: _value(kv.value) for kv in kvs}


class OtlpStub:
    def __init__(self, mode: str = "healthy", slow_s: float = 0.5, tls: Any = None) -> None:
        #: An `ssl.SSLContext` (server side) to serve HTTPS instead of HTTP.
        self.tls = tls
        self.mode = mode
        self.slow_s = slow_s
        #: Where `redirect` mode points the client (another stub's endpoint).
        self.redirect_to = ""
        self.lock = threading.Lock()
        self.raw: list[tuple[str, bytes, dict[str, str]]] = []
        self.spans: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []
        self.metric_points: list[dict[str, Any]] = []
        self.scopes: list[str] = []
        self.resources: list[dict[str, Any]] = []
        self.requests_by_path: dict[str, int] = {}
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port = 0

    @property
    def endpoint(self) -> str:
        return f"{'https' if self.tls else 'http'}://127.0.0.1:{self.port}"

    # ── lifecycle ────────────────────────────────────────────────────────────────

    def start(self) -> "OtlpStub":
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # quiet
                return None

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                path = self.path
                with stub.lock:
                    seen = stub.requests_by_path.get(path, 0)
                    stub.requests_by_path[path] = seen + 1
                    stub.raw.append((path, body, {k.lower(): v for k, v in self.headers.items()}))
                mode = stub.mode
                if mode == "401":
                    self.send_response(401)
                    self.end_headers()
                    return
                if mode == "fail_logs" and path == "/v1/logs":
                    self.send_response(503)
                    self.end_headers()
                    return
                if mode == "fail_first" and seen == 0:
                    self.send_response(503)
                    self.end_headers()
                    return
                if mode == "redirect":
                    self.send_response(307)
                    self.send_header("Location", stub.redirect_to + path)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if mode == "huge":
                    stub._decode(path, body)
                    self.send_response(200)
                    self.send_header("Content-Length", str(10**9))
                    self.end_headers()
                    self.wfile.write(b"\x00" * 16)
                    self.wfile.flush()
                    time.sleep(stub.slow_s)
                    return
                if mode == "slow":
                    time.sleep(stub.slow_s)
                stub._decode(path, body)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._server.daemon_threads = True
        if self.tls is not None:
            self._server.socket = self.tls.wrap_socket(self._server.socket, server_side=True)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def go_absent(self) -> None:
        """Close the socket but keep the port number: connections are refused."""
        self.stop()

    # ── decoding ─────────────────────────────────────────────────────────────────

    def _decode(self, path: str, body: bytes) -> None:
        with self.lock:
            if path == "/v1/traces":
                req = ExportTraceServiceRequest.FromString(body)
                for rs in req.resource_spans:
                    self.resources.append(_attrs(rs.resource.attributes))
                    for ss in rs.scope_spans:
                        for span in ss.spans:
                            self.scopes.append(ss.scope.name)
                            self.spans.append({
                                "name": span.name, "scope": ss.scope.name,
                                "trace_id": span.trace_id.hex(), "span_id": span.span_id.hex(),
                                "parent_span_id": span.parent_span_id.hex(),
                                "attributes": _attrs(span.attributes),
                                "status_code": span.status.code,
                                "status_message": span.status.message,
                                "events": len(span.events), "trace_state": span.trace_state,
                            })
            elif path == "/v1/logs":
                req = ExportLogsServiceRequest.FromString(body)
                for rl in req.resource_logs:
                    self.resources.append(_attrs(rl.resource.attributes))
                    for sl in rl.scope_logs:
                        for rec in sl.log_records:
                            self.scopes.append(sl.scope.name)
                            self.logs.append({
                                "event_name": rec.event_name, "scope": sl.scope.name,
                                "trace_id": rec.trace_id.hex(), "span_id": rec.span_id.hex(),
                                "time_unix_nano": rec.time_unix_nano, "flags": rec.flags,
                                "attributes": _attrs(rec.attributes),
                                "body": _value(rec.body) if rec.HasField("body") else None,
                            })
            elif path == "/v1/metrics":
                req = ExportMetricsServiceRequest.FromString(body)
                for rm in req.resource_metrics:
                    self.resources.append(_attrs(rm.resource.attributes))
                    for sm in rm.scope_metrics:
                        for metric in sm.metrics:
                            self.scopes.append(sm.scope.name)
                            kind = metric.WhichOneof("data")
                            for point in getattr(metric, kind).data_points:
                                value = (point.as_int if hasattr(point, "as_int")
                                         and point.WhichOneof("value") == "as_int"
                                         else getattr(point, "as_double", None)
                                         if hasattr(point, "as_double") else point.count)
                                self.metric_points.append({
                                    "name": metric.name, "kind": kind, "value": value,
                                    "attributes": _attrs(point.attributes),
                                    "exemplars": len(getattr(point, "exemplars", ())),
                                })

    # ── helpers ──────────────────────────────────────────────────────────────────

    def wait_for(self, predicate: Any, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if predicate(self):
                    return True
            time.sleep(0.02)
        return False

    def all_bytes(self) -> bytes:
        with self.lock:
            return b"".join(body for _p, body, _h in self.raw)
