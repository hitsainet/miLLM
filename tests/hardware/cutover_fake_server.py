"""A fake miServer for `cutover_checks.py` (040 FR-040.29, FR-040.43; TID §3.10; T26).

It answers the routes the cutover script reads with the shapes the real server uses (the
`ApiResponse` envelope, `PageResponse` with `meta.next`, OpenAI's list for `/v1/batches`,
`/api/health/detailed`'s `inference`, `lease`, `weights`, `telemetry` and `legacy_names` blocks, the
probe import's refusals). It is NOT a model of the server's correctness: the script's checks are
proven against it by switching on one condition or defect at a time
(`tests/unit/hardware/test_cutover_checks.py`).

Like `otel_fake_millm.py` it is a standard-library HTTP server on 127.0.0.1 (the TID names a
Starlette app; the pattern the repo actually uses is this one, and it puts the script's real HTTP
client under test). The OLD address is the same listener reached as `localhost`: the fake tells
the two apart by the `Host` header, as the real host middleware does.

`busy` (work in flight, each blocks the preflight): `download`, `load`, `executing_approval`,
`non_terminal_batch`, `lease`, `attached_trainer`; and `pending_approval` (a warning, or a block
under `--require-no-pending`).

`defects` (each makes one check fail): `approvals_500`, `weights_error` (preflight); and for the
post-window checks `headers_renamed`, `headers_missing`, `hf_token_unreadable`, `old_alias_down`,
`new_host_down`, `telemetry_missing`, `legacy_host_not_recorded`, `probes_not_rearmed`.

`reimport`: how `POST /api/probes/import?on_conflict=fail` answers: `name_conflict` (the right
answer) or one of `REIMPORT_REFUSALS`.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

NEW_HOST = "127.0.0.1"
OLD_HOST = "localhost"
MODEL = "fake-org/fake-model"
PLACEMENT = {"device_map": {"": "cuda:0"}, "card_uuids": ["GPU-00000000-0000-4000-8000-00000000fake"]}
SAES = ["sae_a", "sae_b"]
CIRCUITS = ["circ_1"]
ARMED_PROBES = ["pr_000000000001", "pr_000000000002", "pr_000000000003"]
ARMED_BEHAVIOURS = ["deception"]
PROFILE = {"id": "prof_1", "name": "calm"}
STORED_DEFINITION = {"kind": "mistudio.probe-definition/v1", "name": "fake probe", "version": 1}

#: busy condition → the preflight check it must fail.
PREFLIGHT_CONDITIONS = {
    "download": "downloads",
    "load": "loads",
    "executing_approval": "executing_approvals",
    "non_terminal_batch": "batches",
    "lease": "lease",
    "attached_trainer": "attached_trainer",
}
#: busy condition → the item the FAIL line must name.
WHAT = {
    "download": "dl-model",
    "load": "ld-model",
    "executing_approval": "apr_" + "e" * 32,
    "non_terminal_batch": "batch_busy",
    "lease": "miforge-trainer",
    "attached_trainer": "attached",
}


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"success": False, "data": None, "error": {"code": code, "message": message, "details": details or {}}}


#: every refusal that is NOT "already here", and a 2xx: the re-import check must FAIL on each.
REIMPORT_REFUSALS: dict[str, tuple[int, Any]] = {
    "unknown_kind": (422, _error("VALIDATION_ERROR", "Unknown kind 'x'",
                                 {"kind": "x", "code": "UNKNOWN_KIND"})),
    "schema": (422, _error("VALIDATION_ERROR", "Probe definition does not conform to mistudio.probe-definition/v1",
                           {"errors": [{"loc": ["head"], "msg": "Field required"}]})),
    "identity": (409, _error("PROBE_IDENTITY_MISMATCH", "carried identity does not match")),
    "catalogue_limit": (409, _error("CATALOGUE_LIMIT", "catalogue full")),
    "2xx": (200, {"success": True, "data": {"id": "pr_new", "name": "fake probe (2)"}, "error": None}),
}
NAME_CONFLICT = (422, _error("VALIDATION_ERROR", f"A probe named '{STORED_DEFINITION['name']}' already exists",
                             {"name": STORED_DEFINITION["name"], "code": "NAME_CONFLICT"}))


class FakeServer:
    def __init__(self) -> None:
        self.busy: set[str] = set()
        self.defects: set[str] = set()
        self.reimport = "name_conflict"
        self.batch_page_size = 100
        self.calls: Counter[str] = Counter()
        self.chat_headers: list[dict[str, str]] = []
        self.imports: list[dict[str, Any]] = []
        self.old_host_seen: datetime | None = None
        self.lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self.port = 0

    @property
    def base(self) -> str:
        return f"http://{NEW_HOST}:{self.port}"

    @property
    def old_base(self) -> str:
        return f"http://{OLD_HOST}:{self.port}"

    # ── answers ────────────────────────────────────────────────────────────────────────────

    def _models(self) -> list[dict[str, Any]]:
        rows = [{"id": 1, "name": MODEL, "status": "loaded"}, {"id": 2, "name": "spare", "status": "ready"}]
        if "download" in self.busy:
            rows.append({"id": 3, "name": "dl-model", "status": "downloading"})
        if "load" in self.busy:
            rows.append({"id": 4, "name": "ld-model", "status": "loading"})
        return rows

    def _approvals(self) -> list[dict[str, Any]]:
        rows = [{"approval_id": "apr_" + "d" * 32, "action": "model.load", "status": "executed"}]
        if "executing_approval" in self.busy:
            rows.append({"approval_id": WHAT["executing_approval"], "action": "model.load", "status": "executing"})
        if "pending_approval" in self.busy:
            rows.append({"approval_id": "apr_" + "p" * 32, "action": "model.delete", "status": "pending"})
        return rows

    def _batches(self) -> list[dict[str, Any]]:
        rows = [{"id": f"batch_{n}", "object": "batch", "status": s}
                for n, s in enumerate(["completed", "failed", "cancelled", "expired", "completed"])]
        if "non_terminal_batch" in self.busy:
            rows.append({"id": "batch_busy", "object": "batch", "status": "in_progress"})
        return rows

    def _detailed(self) -> dict[str, Any]:
        lease = None
        if "lease" in self.busy:
            lease = {"model_id": 1, "model_name": MODEL, "holder": WHAT["lease"], "reason": "eval",
                     "acquired_at": "2026-11-10T00:00:00Z", "expires_at": "2026-11-10T01:00:00Z",
                     "ttl_seconds": 3600, "seconds_remaining": 1800}
        weights: dict[str, Any] = {"weights_state": "base", "barrier": "open",
                                   "attached": "attached_trainer" in self.busy,
                                   "attachment_state": "active" if "attached_trainer" in self.busy else None,
                                   "error": None}
        if "weights_error" in self.defects:
            weights = {"error": "unavailable (error id 7); see the server log"}
        telemetry = None if "telemetry_missing" in self.defects else {
            "enabled": True, "config": {"endpoint": "http://otel:4318", "stateless_export_permitted": False},
            "signals": {}, "error": None}
        in_use = []
        if self.old_host_seen is not None and "legacy_host_not_recorded" not in self.defects:
            in_use.append({"kind": "host", "old": OLD_HOST, "new": NEW_HOST,
                           "last_seen": self.old_host_seen.isoformat().replace("+00:00", "Z")})
        return {
            "status": "healthy", "version": "0.0.0-fake", "timestamp": _now(), "components": [],
            "circuit_breakers": [], "model_loaded": True, "model_name": MODEL, "model_placement": PLACEMENT,
            "sae_attached": True, "sae_id": SAES[0], "sae_count": len(SAES), "sae_ids": SAES,
            "inference": {"backend": "queue", "in_flight": 0, "queue_waiting": 0, "queue_pending": 0,
                          "batch_backlog_rows": None},
            "lease": lease, "active_profile": {**PROFILE, "source_kind": "manual", "intensity": 1.0,
                                               "sensing_enabled": False},
            "weights": weights, "telemetry": telemetry,
            "legacy_names": {"in_use": in_use, "also_set": [], "conflicts": [], "retired_but_set": []},
            "legacy_names_meta": {"observing_since": None, "last_flush_at": None, "watching": {},
                                  "in_use_window_hours": 24, "transition_hosts": [OLD_HOST],
                                  "unobserved_hosts": [], "new_host_for": {}, "retire_on": None,
                                  "error": None},
        }

    # ── the server ─────────────────────────────────────────────────────────────────────────

    def start(self) -> FakeServer:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # noqa: ARG002 - quiet
                return None

            def _send(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
                body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _route(self, method: str) -> None:
                url = urlparse(self.path)
                query = parse_qs(url.query)
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
                body: Any = None
                if method == "POST":
                    raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    body = json.loads(raw) if raw else None
                with fake.lock:
                    fake.calls[f"{method} {url.path}"] += 1
                    if host == OLD_HOST:
                        fake.old_host_seen = datetime.now(UTC)
                    status, obj, headers = fake.answer(method, url.path, query, host, body, dict(self.headers))
                self._send(status, obj, headers)

            def do_GET(self) -> None:  # noqa: N802
                self._route("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._route("POST")

        self._server = ThreadingHTTPServer((NEW_HOST, 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        server = self._server
        threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def answer(self, method: str, path: str, query: dict[str, list[str]], host: str, body: Any,
               headers: dict[str, str]) -> tuple[int, Any, dict[str, str] | None]:
        ok = lambda data: (200, {"success": True, "data": data, "error": None}, None)  # noqa: E731
        if method == "GET" and path == "/api/health":
            if host == OLD_HOST and "old_alias_down" in self.defects:
                return 404, "<html>404 Not Found</html>", None
            if host == NEW_HOST and "new_host_down" in self.defects:
                return 503, {"status": "unhealthy", "version": "0.0.0-fake"}, None
            return 200, {"status": "healthy", "version": "0.0.0-fake", "timestamp": _now()}, None
        if method == "GET" and path == "/api/health/detailed":
            return 200, self._detailed(), None
        if method == "GET" and path == "/api/models":
            return ok(self._models())
        if method == "GET" and path == "/api/approvals":
            if "approvals_500" in self.defects:
                return 500, _error("INTERNAL_ERROR", "boom"), None
            want = (query.get("status") or [None])[0]
            rows = [r for r in self._approvals() if want is None or r["status"] == want]
            pending = sum(r["status"] == "pending" for r in self._approvals())
            return ok({"approvals": rows, "count": len(rows), "pending": pending})
        if method == "GET" and path == "/v1/batches":
            rows = self._batches()
            after = (query.get("after") or [None])[0]
            if after is not None:
                rows = rows[[r["id"] for r in rows].index(after) + 1:]
            page = rows[: self.batch_page_size]
            return 200, {"object": "list", "data": page, "first_id": page[0]["id"] if page else None,
                         "last_id": page[-1]["id"] if page else None,
                         "has_more": len(rows) > len(page)}, None
        if method == "GET" and path == "/api/probes":
            if (query.get("armed") or [""])[0] == "true":
                ids = [] if "probes_not_rearmed" in self.defects else ARMED_PROBES
                start = int((query.get("cursor") or ["0"])[0])
                page = ids[start:start + 2]
                nxt = str(start + 2) if start + 2 < len(ids) else None
                return 200, {"success": True, "data": [{"id": i, "armed": True} for i in page],
                             "error": None, "meta": {"next": nxt, "limit": 2}}, None
            return 200, {"success": True, "data": [{"id": ARMED_PROBES[0], "name": STORED_DEFINITION["name"]}],
                         "error": None, "meta": {"next": None, "limit": 1}}, None
        if method == "GET" and path == "/api/probes/behaviours":
            items = [{"key": "deception", "armed": 2, "armed_probe_ids": ARMED_PROBES[:2]},
                     {"key": "idle", "armed": 0, "armed_probe_ids": []}]
            if "probes_not_rearmed" in self.defects:
                items = [{**i, "armed": 0, "armed_probe_ids": []} for i in items]
            return 200, {"success": True, "data": items, "error": None, "meta": {"next": None, "limit": 200}}, None
        if method == "GET" and path == "/api/circuits/active":
            return ok([{"id": c, "name": c, "steering": True} for c in CIRCUITS])
        if method == "GET" and path == f"/api/probes/{ARMED_PROBES[0]}":
            return ok({"id": ARMED_PROBES[0], "name": STORED_DEFINITION["name"], "definition": STORED_DEFINITION})
        if method == "GET" and path == "/api/settings/hf-token":
            unreadable = "hf_token_unreadable" in self.defects
            return ok({"configured": True, "source": "settings", "stored": True,
                       "stored_error": "key_mismatch" if unreadable else None,
                       "encryption_available": True, "env_configured": False})
        if method == "POST" and path == "/api/probes/import":
            self.imports.append({"query": query, "body": body})
            status, obj = NAME_CONFLICT if self.reimport == "name_conflict" else REIMPORT_REFUSALS[self.reimport]
            return status, obj, None
        if method == "POST" and path == "/v1/chat/completions":
            self.chat_headers.append({k.lower(): v for k, v in headers.items()})
            out = {"X-miLLM-Backend": "queue", "X-miLLM-Steering": "none"}
            if "headers_renamed" in self.defects:
                out = {"X-miServer-Backend": "queue", "X-miServer-Steering": "none"}
            if "headers_missing" in self.defects:
                out = {}
            return 200, {"id": "chatcmpl-1", "object": "chat.completion", "model": MODEL,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                      "finish_reason": "stop"}]}, out
        return 404, _error("NOT_FOUND", f"no route {method} {path}"), None


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
