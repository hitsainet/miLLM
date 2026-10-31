"""A fake miLLM bench pod and a fake Collector, for `otel_acceptance.py --dry-run` (FTASKS 13.1).

The whole harness flow — preflight, arming, the arm switch and its health check, A-B-B-A blocks
with a backlog, streaming, the relay's arrival timing, the outage and the AT checks — runs against
this in CI with no GPU. It answers the routes the harness calls with the shapes the real server
uses (`/api/health/detailed`'s `telemetry` block, the `ApiResponse` envelope, `PageResponse` for
`/api/probes` and its events, the bench routes of `otel_bench_server.py`), and when an arm is on
it exports real OTLP/HTTP protobuf to its configured endpoint (the relay), as miLLM's pipeline
does. It is NOT a model of miLLM's correctness: the harness's checks are proven against it by
switching on one `defects` entry at a time (`tests/unit/hardware/test_otel_acceptance_harness.py`).

Defects (each makes one harness check fail): `arm_ignored` (the bench route does not change the
telemetry state), `no_enqueue` (the enqueue timer records nothing), `leak` (a user message is
exported as an attribute value), `wrong_trace` (results ignore the caller's traceparent),
`stateless_leak` (the stateless route exports without permission), `export_fails` (every export
POST is reported failed although the receiver is up). Added by review R1 (2026-10-25), one per
check that no defect exercised: `outage_fails` (a chat fails while the last export failed: export
breaking inference), `chat_fails` (every chat fails), `disconnect_completed` (a client disconnect
is recorded `completed`), `sampler_ignored` (sampler 0 still exports spans), `refusal_wrong_code`
(the refusal's lifecycle record carries another code than the API's), `drop_sentinel_requests`
(nothing is exported for a request carrying a sentinel: an empty capture), `metric_ids` (a
result id as a metric dimension), `mcr_4` (MAX_CONCURRENT_REQUESTS 4), `endpoint_elsewhere`
(telemetry points at the Collector, not the relay), `wrong_model` (responses name another
model), `stream_error` (a stream ends in an SSE error event, then `[DONE]`), `stream_truncated`
(a stream stops with no `[DONE]`), `leak_stream` (only a streamed chat's text is exported),
`leak_score` (only the score route's permitted export carries its input text), `auth_leak` (the
Authorization header is exported), `outcome_mismatch` (a disconnect is stored `interrupted` but
exported `completed`), `stateless_partial` (a permitted stateless export sends only its first
result), `score_never_exports` (the score route exports nothing even when export is permitted
and requested).
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

FAKE_HF_ID = "fake-org/fake-model"
FAKE_CARD = "GPU-00000000-0000-4000-8000-00000000fake"
TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")
WINDOWS = ("all", "prompt", "response")
SIGNALS = ("traces", "metrics", "logs")


def _otlp_logs(records: list[dict[str, Any]]) -> bytes:
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord

    req = ExportLogsServiceRequest()
    scope = req.resource_logs.add().scope_logs.add()
    scope.scope.name = "millm"
    for r in records:
        rec = LogRecord(time_unix_nano=r["time_ns"], event_name=r["event_name"])
        if r.get("trace_id"):
            rec.trace_id = bytes.fromhex(r["trace_id"])
        for k, v in r["attributes"].items():
            av = AnyValue(bool_value=v) if isinstance(v, bool) else (
                AnyValue(int_value=v) if isinstance(v, int) else AnyValue(string_value=str(v)))
            rec.attributes.append(KeyValue(key=k, value=av))
        scope.log_records.append(rec)
    return req.SerializeToString()


def _otlp_spans(n: int, trace_id: str) -> bytes:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    req = ExportTraceServiceRequest()
    ss = req.resource_spans.add().scope_spans.add()
    for _ in range(n):
        span = ss.spans.add()
        span.name = "chat"
        span.trace_id = bytes.fromhex(trace_id)
        span.span_id = uuid.uuid4().bytes[:8]
    return req.SerializeToString()


def _otlp_metrics(points: list[dict[str, Any]]) -> bytes:
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
    )
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

    req = ExportMetricsServiceRequest()
    sm = req.resource_metrics.add().scope_metrics.add()
    metric = sm.metrics.add()
    metric.name = "millm.probe.evaluations"
    for p in points:
        dp = metric.sum.data_points.add()
        dp.as_int = 1
        for k, v in p.items():
            dp.attributes.append(KeyValue(key=k, value=AnyValue(string_value=str(v))))
    return req.SerializeToString()


class FakeCollector:
    """Accepts OTLP POSTs and keeps the bodies (the relay must forward them byte-identically)."""

    def __init__(self) -> None:
        self.bodies: list[tuple[str, bytes]] = []
        self.lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self.port = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeCollector:
        coll = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # noqa: ARG002 - quiet
                return None

            def do_POST(self) -> None:  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                with coll.lock:
                    coll.bodies.append((self.path, body))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


class FakeMillm:
    def __init__(self, model: str, probe_names: list[str], bench: bool = True,
                 defects: frozenset[str] = frozenset(), token_ms: float = 0.2,
                 on_overhead: float = 0.02) -> None:
        self.model = model
        self.bench = bench
        self.defects = defects
        self.token_ms = token_ms
        self.on_overhead = on_overhead
        self.lock = threading.Lock()
        self.serve_lock = threading.Lock()  # MAX_CONCURRENT_REQUESTS = 1
        self.models = [{"id": 1, "name": model, "status": "loaded"},
                       {"id": 2, "name": "fake-other-model", "status": "ready"}]
        self.resident = model
        self.auto_loaded: list[str] = []
        self.served = 0
        unload = [d for d in defects if d.startswith("unload_after_")]
        self.unload_after = int(unload[0].rsplit("_", 1)[1]) if unload else None
        self.v1_requests = 0
        self.v1_without_policy = 0
        self.env_overrides: dict[str, Any] = {"MAX_CONCURRENT_REQUESTS": 4} if "mcr_4" in defects else {}
        self.last_export_ok = True
        self.env_changes: list[tuple[str, str, Any]] = []
        self.probes: dict[str, dict[str, Any]] = {}
        for name in probe_names:
            self._import({"name": name, "model": {"hf_id": FAKE_HF_ID}})
        self.events: list[dict[str, Any]] = []
        self.preset = "off"
        self.endpoint: str | None = None
        self.health = {s: {"exported": 0, "failed": 0, "dropped": 0} for s in SIGNALS}
        self.pending = 0
        self.enqueue_us: list[float] = []
        self.enqueue_calls = 0
        self._server: ThreadingHTTPServer | None = None
        self.port = 0

    # ── state ──────────────────────────────────────────────────────────────────

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _import(self, doc: dict[str, Any]) -> dict[str, Any]:
        pid = "pr_" + uuid.uuid4().hex[:12]
        row = {"id": pid, "name": doc["name"], "armed": False,
               "hf_id": (doc.get("model") or {}).get("hf_id"), "definition": doc}
        self.probes[pid] = row
        return row

    def _summary(self, row: dict[str, Any]) -> dict[str, Any]:
        return {k: row[k] for k in ("id", "name", "armed", "hf_id")}

    def telemetry_block(self) -> dict[str, Any]:
        on = self.preset != "off" if "arm_ignored" not in self.defects else False
        return {
            "enabled": on,
            "config": {"endpoint": ("http://otel-collector.millm-otel.svc:4318"
                                    if "endpoint_elsewhere" in self.defects else self.endpoint),
                       "metric_interval_s": 0.2},
            "signals": {s: {"state": "healthy" if on else "disabled", **self.health[s],
                            "queue": {"used": self.pending if on else 0, "capacity": 8192}}
                        for s in SIGNALS},
        }

    def _results(self, request_id: str, trace_id: str, origin: str, outcome: str,
                 n_windows: int = 3) -> list[dict[str, Any]]:
        out = []
        for row in self.probes.values():
            if not row["armed"] and origin == "live":
                continue
            for window in WINDOWS[:n_windows]:
                out.append({"schema": "mi.probe.result/v1", "result_id": str(uuid.uuid4()),
                            "probe_id": row["id"], "probe_name": row["name"], "window": window,
                            "origin": origin, "outcome": outcome, "request_id": request_id,
                            "trace_id": trace_id, "score": 0.5, "verdict": False,
                            "event_time_ns": time.time_ns()})
        return out

    def _export(self, results: list[dict[str, Any]], *, trace_id: str,
                lifecycle: list[dict[str, Any]] | None = None, text: str | None = None,
                leak: bool = False) -> None:
        """What miLLM's pipeline does when an arm is on: enqueue (timed), then POST later."""
        if self.preset == "off" or not self.endpoint:
            return
        if "drop_sentinel_requests" in self.defects and text and "qz" in text:
            return
        t0 = time.perf_counter_ns()
        records = []
        for r in results:
            attrs = {f"mi.probe.{k}": v for k, v in r.items()
                     if k not in ("event_time_ns", "score", "verdict")}
            if ("leak" in self.defects or leak) and text:
                attrs["mi.probe.reason"] = text
            records.append({"event_name": "mi.probe.result", "time_ns": r["event_time_ns"],
                            "trace_id": r["trace_id"], "attributes": attrs})
        for lc in lifecycle or []:
            records.append({"event_name": "mi.probe.lifecycle", "time_ns": time.time_ns(),
                            "attributes": lc})
        if "no_enqueue" not in self.defects and records:
            per = (time.perf_counter_ns() - t0) / 1000.0 / len(records)
            with self.lock:
                self.enqueue_us.extend([per] * len(records))
                self.enqueue_calls += len(records)
                self.pending += 1
        preset, endpoint = self.preset, self.endpoint
        trace_on = preset != "on_sampler0" or "sampler_ignored" in self.defects

        def send() -> None:
            import httpx

            time.sleep(0.02)
            posts = []
            if records:
                posts.append(("logs", "/v1/logs", _otlp_logs(records)))
            if trace_on and results:
                posts.append(("traces", "/v1/traces", _otlp_spans(1, trace_id)))
            if results:
                posts.append(("metrics", "/v1/metrics", _otlp_metrics(
                    [{"probe_id": r["probe_id"], "outcome": r["outcome"],
                      **({"result_id": r["result_id"]} if "metric_ids" in self.defects else {})}
                     for r in results])))
            for signal, path, body in posts:
                try:
                    r = httpx.post(endpoint + path, content=body, timeout=2.0,
                                   headers={"Content-Type": "application/x-protobuf"})
                    ok = r.status_code < 300 and "export_fails" not in self.defects
                except Exception:  # noqa: BLE001
                    ok = False
                with self.lock:
                    self.health[signal]["exported" if ok else "failed"] += 1
                    self.last_export_ok = ok
            with self.lock:
                self.pending = max(0, self.pending - 1)

        threading.Thread(target=send, daemon=True).start()

    # ── server ─────────────────────────────────────────────────────────────────

    def start(self) -> FakeMillm:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # noqa: ARG002 - quiet
                return None

            def _json(self, status: int, obj: Any) -> None:
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> Any:
                # Read once per request (route() reads it first): an unread body on a keep-alive
                # connection would be parsed as the next request line.
                if not hasattr(self, "_cached_body"):
                    n = int(self.headers.get("Content-Length") or 0)
                    self._cached_body = json.loads(self.rfile.read(n) or b"null") if n else None
                return self._cached_body

            def do_GET(self) -> None:  # noqa: N802
                fake.route(self, "GET")

            def do_POST(self) -> None:  # noqa: N802
                fake.route(self, "POST")

            def do_PUT(self) -> None:  # noqa: N802
                fake.route(self, "PUT")

            def do_DELETE(self) -> None:  # noqa: N802
                fake.route(self, "DELETE")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def route(self, h: Any, method: str) -> None:  # noqa: C901 - one table of fake routes
        url = urlparse(h.path)
        path, q = url.path, parse_qs(url.query)
        # One handler instance serves every request of a keep-alive connection: forget the last
        # request's body, then read this one's.
        h.__dict__.pop("_cached_body", None)
        h._body()
        ok = lambda data: h._json(200, {"success": True, "data": data, "error": None})  # noqa: E731
        if path.startswith("/__otel_bench"):
            if not self.bench:
                return h._json(404, {"detail": "Not Found"})
            if path.endswith("/arm"):
                preset = (h._body() or {}).get("preset")
                if "arm_ignored" not in self.defects:
                    self.preset = preset
                self.enqueue_us, self.enqueue_calls = [], 0
                return h._json(200, {"preset": preset})
            if path.endswith("/reset"):
                with self.lock:
                    self.enqueue_us, self.enqueue_calls = [], 0
                return h._json(200, {"reset": True})
            with self.lock:
                samples = list(self.enqueue_us) if q.get("samples", ["1"])[0] != "0" else None
                calls = self.enqueue_calls
            return h._json(200, {
                "preset": self.preset, "telemetry": self.telemetry_block(),
                "counter_store": {"series": {"millm.probe.evaluations": len(self.probes)},
                                  "series_overflow": 0},
                "rss_kib": 1000,
                "enqueue": {"calls": {"enqueue_log": calls},
                            **({"samples_us": {"enqueue_log": samples}} if samples is not None
                               else {})}})
        if path == "/api/health/detailed":
            return h._json(200, {"status": "healthy", "version": "0.0.0-fake",
                                 "model_loaded": self.resident is not None,
                                 "model_name": self.resident,
                                 "telemetry": self.telemetry_block()})
        if path == "/api/health/metrics/prometheus":
            body = f"millm_probe_evaluations_total {len(self.events)}\n".encode()
            h.send_response(200)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return None
        if path == "/api/models":
            return ok(self.models)
        if path == "/api/environment" and method == "GET":
            value = self.env_overrides.get("MAX_CONCURRENT_REQUESTS", 1)
            return ok({"settings": [{"key": "MAX_CONCURRENT_REQUESTS", "value": value},
                                    *[{"key": k, "value": v} for k, v in self.env_overrides.items()
                                      if k != "MAX_CONCURRENT_REQUESTS"]]})
        if path.startswith("/api/environment/"):
            key = path.rsplit("/", 1)[1]
            if method == "PUT":
                self.env_overrides[key] = (h._body() or {}).get("value")
            else:
                self.env_overrides.pop(key, None)
            self.env_changes.append((method, key, self.env_overrides.get(key)))
            return ok({"key": key})
        if path == "/api/probes" and method == "GET":
            return h._json(200, {"success": True, "error": None, "meta": {"next": None},
                                 "data": [self._summary(r) for r in self.probes.values()]})
        if path == "/api/probes/import":
            doc = h._body() or {}
            if any(r["name"] == doc.get("name") for r in self.probes.values()) and \
                    q.get("on_conflict", ["rename"])[0] == "fail":
                return h._json(409, {"success": False, "error": {"code": "PROBE_NAME_TAKEN"}})
            row = self._import(doc)
            self._export([], trace_id="", lifecycle=[{"mi.lifecycle.action": "import",
                                                      "mi.lifecycle.outcome": "ok",
                                                      "mi.lifecycle.probe_id": row["id"]}])
            return ok(self._summary(row))
        if path == "/api/probes/events":
            rid = q.get("request_id", [None])[0]
            limit = int(q.get("limit", ["100"])[0])
            rows = [e for e in reversed(self.events) if rid is None or e["request_id"] == rid]
            return h._json(200, {"success": True, "error": None, "meta": {"next": None},
                                 "data": rows[:limit]})
        if path == "/api/probes/score":
            return self._score(h, ok)
        m = re.fullmatch(r"/api/probes/([^/]+)(?:/(arm|disarm))?", path)
        if m:
            row = self.probes.get(m.group(1))
            if row is None:
                return h._json(404, {"success": False, "error": {"code": "PROBE_NOT_FOUND"}})
            if m.group(2) == "arm":
                if row["hf_id"] != FAKE_HF_ID:
                    self._export([], trace_id="", lifecycle=[{
                        "mi.lifecycle.action": "arm_refused", "mi.lifecycle.outcome": "refused",
                        "mi.lifecycle.code": ("PROBE_HOOK_UNSUPPORTED"
                                              if "refusal_wrong_code" in self.defects
                                              else "PROBE_MODEL_MISMATCH"),
                        "mi.lifecycle.probe_id": row["id"]}])
                    return h._json(409, {"success": False,
                                         "error": {"code": "PROBE_MODEL_MISMATCH"}})
                row["armed"] = "arm_noop" not in self.defects
                return ok(self._summary(row))
            if m.group(2) == "disarm":
                row["armed"] = False
                return ok(self._summary(row))
            if method == "DELETE":
                self.probes.pop(row["id"])
                self._export([], trace_id="", lifecycle=[{"mi.lifecycle.action": "delete",
                                                          "mi.lifecycle.outcome": "ok",
                                                          "mi.lifecycle.probe_id": row["id"]}])
                return ok({"deleted": row["id"]})
            return ok(self._summary(row))
        if path == "/v1/chat/completions":
            return self._chat(h)
        return h._json(404, {"detail": "Not Found"})

    def _score(self, h: Any, ok: Any) -> None:
        body = h._body() or {}
        results = []
        for _inp in body.get("inputs") or []:
            results.extend(self._results("score-" + uuid.uuid4().hex[:8], uuid.uuid4().hex,
                                         "stateless", "completed", n_windows=1))
        permitted = self.preset == "on_stateless" or "stateless_leak" in self.defects
        text = " ".join(str(i.get("text") or i.get("messages")) for i in body.get("inputs") or [])
        if body.get("export_results") and permitted and "score_never_exports" not in self.defects:
            self._export(results[:1] if "stateless_partial" in self.defects else results,
                         trace_id=uuid.uuid4().hex, text=text,
                         leak="leak_score" in self.defects)
        if "leak" in self.defects:
            self._export(results[:1], trace_id=uuid.uuid4().hex, text=text)
        return ok({"results": [{k: v for k, v in r.items() if k != "event_time_ns"}
                               for r in results],
                   "export": {"requested": bool(body.get("export_results")),
                              "permitted": permitted}})

    def _chat(self, h: Any) -> None:
        body = h._body() or {}
        self.v1_requests += 1
        policy = (h.headers.get("X-miLLM-Load-Policy") or "auto").lower()
        if policy != "refuse":
            self.v1_without_policy += 1
        model = body.get("model")
        if "chat_fails" in self.defects or (
                "outage_fails" in self.defects and not self.last_export_ok):
            return h._json(500, {"error": {"code": "internal_error"}})
        with self.serve_lock:
            if model != self.resident:
                if policy == "refuse":
                    return h._json(409, {"error": {"code": "model_not_resident"}})
                self.auto_loaded.append(model)  # what the real server would do: a swap, ungated
                self.resident = model
            try:
                return self._serve(h, body)
            finally:
                self.served += 1
                if self.unload_after is not None and self.served >= self.unload_after:
                    self.resident = None

    def _serve(self, h: Any, body: dict[str, Any]) -> None:
        m = TRACEPARENT.match(h.headers.get("traceparent") or "")
        trace_id = m.group(1) if (m and "wrong_trace" not in self.defects) else uuid.uuid4().hex
        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        n_out = int(body.get("max_tokens") or 16)
        text = " ".join(str(x.get("content")) for x in body.get("messages") or [])
        n_prompt = max(1, round(len(text.split()) * 1.35))
        factor = 1.0 + (self.on_overhead if self.preset != "off" else 0.0)
        served_model = "fake-other-model" if "wrong_model" in self.defects else self.model
        if not body.get("stream"):
            time.sleep(self.token_ms * (n_out + n_prompt / 50) * factor / 1000.0)
            results = self._results(request_id, trace_id, "live", "completed")
            self._store(results)
            auth = h.headers.get("Authorization") if "auth_leak" in self.defects else None
            self._export(results, trace_id=trace_id, text=auth or text, leak=bool(auth))
            return h._json(200, {
                "id": request_id, "object": "chat.completion", "model": served_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                          "total_tokens": n_prompt + n_out},
                "millm_probe_verdicts": [{k: v for k, v in r.items() if k != "event_time_ns"}
                                         for r in results]})
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Connection", "close")
        h.end_headers()
        h.close_connection = True
        outcome = "completed"
        try:
            for i in range(n_out):
                time.sleep(self.token_ms * factor / 1000.0)
                chunk = {"id": request_id, "model": served_model,
                         "choices": [{"index": 0, "delta": {"content": f"t{i} "}}]}
                h.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                h.wfile.flush()
                if i == 1 and "stream_error" in self.defects:
                    h.wfile.write(b'data: {"error":{"message":"An internal server error occurred '
                                  b'during streaming","type":"server_error",'
                                  b'"code":"streaming_error"}}\n\ndata: [DONE]\n\n')
                    h.wfile.flush()
                    return None
                if i == 1 and "stream_truncated" in self.defects:
                    h.wfile.flush()
                    return None
            final = {"id": request_id, "model": self.model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                     "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out}}
            h.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            results = self._results(request_id, trace_id, "live", "completed")
            verdicts = {"id": request_id, "choices": [], "millm_probe_verdicts": [
                {k: v for k, v in r.items() if k != "event_time_ns"} for r in results]}
            h.wfile.write(f"data: {json.dumps(verdicts)}\n\ndata: [DONE]\n\n".encode())
            h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            outcome = "completed" if "disconnect_completed" in self.defects else "interrupted"
            results = self._results(request_id, trace_id, "live", outcome)
        self._store(results)
        if outcome == "interrupted" and "outcome_mismatch" in self.defects:
            results = [{**r, "outcome": "completed"} for r in results]
        self._export(results, trace_id=trace_id, text=text, leak="leak_stream" in self.defects)
        return None

    def _store(self, results: list[dict[str, Any]]) -> None:
        with self.lock:
            for r in results:
                self.events.append({"id": len(self.events) + 1, "result_id": r["result_id"],
                                    "request_id": r["request_id"], "outcome": r["outcome"],
                                    "origin": r["origin"]})


class FakeStack:
    """One or two fake servers and a fake Collector. With `second`, `base` is the pre-038 server
    (no bench routes) and `base_b` this image's bench server, as `--arm-switch servers` expects."""

    def __init__(self, model: str, probe_names: list[str] | None = None, second: bool = False,
                 defects: frozenset[str] = frozenset()) -> None:
        names = probe_names or []
        self.collector = FakeCollector()
        self.a = FakeMillm(model, names, bench=not second and "no_bench" not in defects,
                           defects=defects)
        self.b = FakeMillm(model, names, bench=True, defects=defects) if second else None
        self.card = FAKE_CARD

    def start(self) -> FakeStack:
        self.collector.start()
        self.a.start()
        if self.b is not None:
            self.b.start()
        return self

    def stop(self) -> None:
        for s in (self.a, self.b, self.collector):
            if s is not None:
                s.stop()

    @property
    def base(self) -> str:
        return self.a.base

    @property
    def base_b(self) -> str | None:
        return self.b.base if self.b is not None else None

    @property
    def collector_url(self) -> str:
        return self.collector.url

    def point_telemetry_at(self, url: str) -> None:
        for s in (self.a, self.b):
            if s is not None and s.bench:
                s.endpoint = url

    def gpu_memory(self, card: str | None) -> int | None:
        return 2048 if card == FAKE_CARD else None
