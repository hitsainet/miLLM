"""The `rule-runner` worker: operator code, outside the API process (039 S3; FR-39.26; TD39-33).

Run as ``python -m millm.rules.runner_worker`` — the sidecar container of the `millm-backend` pod
(`k8s/base/backend.yaml`) and the `rule-runner` Compose service. It serves the framed protocol of
`runner_protocol` on one Unix socket and runs each call on a FRESH instance of the published guest
snapshot (`runner_host`), under the per-call limits the caller sends, never below the contract
floor.

What it holds, and what it never touches:

- configuration from ``RULE_RUNNER_*`` environment variables ONLY (never `Settings`, a `.env`, a
  database or a secret); the container is given none of the backend's secrets;
- code-rule documents in an in-memory LRU keyed by their content hash, filled by ``load`` frames;
  a restarted worker answers ``unknown_module`` and the client re-sends the bytes once;
- the guest snapshot, verified by sha256 against the vendored contract at start-up — a mismatch
  leaves the worker up answering every call ``rule_worker_unavailable`` (it never runs an
  unverified guest);
- its OWN seccomp filter (`runner_seccomp`, suite decision D55), installed by :func:`start` BEFORE
  the snapshot is loaded: x86-64 only, `AF_UNIX` sockets only, no io_uring, no namespaces, no
  mount API. If it cannot be installed the worker FAILS CLOSED — it loads no guest, health says
  ``seccomp_unavailable`` and every call is ``rule_worker_unavailable`` (code rules stay pending;
  the backend is unaffected).

⚠ NOTHING THE GUEST SEES OR SAYS IS LOGGED: no text, no source, no parameter, no reply byte. A log
line names the operation, the code hash and an error class at most.

Image build helpers (the Dockerfile): ``python -m millm.rules.runner_worker fetch <dir>`` downloads
the snapshot from the contract's `guest.bundle.snapshot.url` and keeps it only if its sha256 is the
contract's; ``precompile <dir>`` writes the `.cwasm` for this Wasmtime build; ``seccomp-check``
builds the worker's seccomp filter without loading it (an image whose libseccomp cannot build it
never ships).
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import socketserver
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from millm.rules import runner_contract as rc
from millm.rules import runner_protocol as proto

log = logging.getLogger("millm.rule_runner")

WORKER_UNAVAILABLE = "rule_worker_unavailable"
INPUT_TOO_LARGE = "rule_input_too_large"
#: Health status of a worker whose own seccomp filter could not be installed (D55).
SECCOMP_UNAVAILABLE = "seccomp_unavailable"


def _int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class WorkerConfig:
    socket_path: Path
    bundle_dir: Path
    slots: int
    module_cache: int
    max_source_bytes: int
    #: Ceilings: the most a caller may ask for (the FLOOR comes from the contract, never here).
    max_call_fuel: int
    max_load_fuel: int
    max_prelude_fuel: int
    max_memory_bytes: int
    max_input_code_points: int
    max_output_bytes: int
    max_wall_ms: int

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> "WorkerConfig":
        e = dict(os.environ if env is None else env)
        return cls(
            socket_path=Path(e.get("RULE_RUNNER_SOCKET", "/run/millm-rules/runner.sock")),
            bundle_dir=Path(e.get("RULE_RUNNER_BUNDLE_DIR", "/app/runner-bundle")),
            slots=_int(e, "RULE_RUNNER_SLOTS", 2),
            module_cache=_int(e, "RULE_RUNNER_MODULE_CACHE", 64),
            max_source_bytes=_int(e, "RULE_RUNNER_MAX_SOURCE_BYTES", 1 << 20),
            max_call_fuel=_int(e, "RULE_RUNNER_MAX_CALL_FUEL", 1 << 35),
            max_load_fuel=_int(e, "RULE_RUNNER_MAX_LOAD_FUEL", 1 << 31),
            max_prelude_fuel=_int(e, "RULE_RUNNER_MAX_PRELUDE_FUEL", 1 << 31),
            max_memory_bytes=_int(e, "RULE_RUNNER_MAX_MEMORY_BYTES", 512 << 20),
            max_input_code_points=_int(e, "RULE_RUNNER_MAX_INPUT_CODE_POINTS", 1 << 20),
            max_output_bytes=_int(e, "RULE_RUNNER_MAX_OUTPUT_BYTES", 1 << 16),
            max_wall_ms=_int(e, "RULE_RUNNER_MAX_WALL_MS", 60_000),
        )


_CEILING = {
    "call_fuel": "max_call_fuel", "load_fuel": "max_load_fuel",
    "prelude_fuel": "max_prelude_fuel", "memory_bytes": "max_memory_bytes",
    "input_code_points": "max_input_code_points", "output_bytes": "max_output_bytes",
    "wall_ms": "max_wall_ms",
}


class Worker:
    """Everything one worker process holds: config, contract, host, module cache, slots."""

    def __init__(self, config: WorkerConfig, *, refused: Optional[str] = None) -> None:
        self.config = config
        self.contract = rc.contract()
        self.floor = rc.floor()
        self.request_limits: tuple[str, ...] = tuple(self.contract["request_limits"])
        self.engine: Any = None
        self.host: Any = None
        if refused is not None:
            # Fail closed (D55): no engine, no guest; every call answers `rule_worker_unavailable`.
            self.status = refused
        else:
            from millm.rules import runner_host

            self.engine = runner_host.Host.engine_for(self.contract)
            loaded = runner_host.load_bundle(self.engine, config.bundle_dir, self.contract)
            self.status = loaded.status
            self.host = (runner_host.Host(self.engine, loaded.module, self.contract)
                         if loaded.module is not None else None)
            if self.status != "ok":
                log.error("rule_runner_bundle_refused detail=%s", loaded.detail)
        self._documents: OrderedDict[str, rc.CodeDocument] = OrderedDict()
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(config.slots)

    # ── requests ─────────────────────────────────────────────────────────────────────────

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        op = message.get("op")
        if op == proto.OP_HEALTH:
            return self.health()
        if op == proto.OP_LOAD:
            return self._load(message)
        if op == proto.OP_CALL:
            return self._call(message)
        return {"error": "request_invalid"}

    def health(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "pid": os.getpid(),
            "contract": self.contract["format"],
            "contract_version": self.contract["contract_version"],
            "bundle_digest": self.contract["guest"]["bundle"]["bundle_digest"],
            "cached_modules": len(self._documents),
        }

    def _load(self, message: dict[str, Any]) -> dict[str, Any]:
        sha = message.get("code_sha256")
        try:
            doc = rc.parse_code_document(sha, message.get("canonical_utf8"),
                                         max_source_bytes=self.config.max_source_bytes)
        except rc.CodeDocumentRefused as exc:
            log.info("rule_runner_load_refused code=%s", exc.code)
            return {"error": exc.code}
        with self._lock:
            self._documents[doc.code_sha256] = doc
            self._documents.move_to_end(doc.code_sha256)
            while len(self._documents) > self.config.module_cache:
                self._documents.popitem(last=False)
        return {"ok": True}

    def _limits(self, raw: Any) -> Any:
        from millm.rules.runner_host import Limits

        if not isinstance(raw, dict) or set(raw) != set(self.request_limits):
            return "request_invalid"
        values: dict[str, int] = {}
        for name in self.request_limits:
            value = raw[name]
            if type(value) is not int or value < 1:
                return "request_invalid"
            floor = self.floor.get(name)
            if floor is not None and value < floor:
                return "limit_below_floor"
            if value > getattr(self.config, _CEILING[name]):
                return "limit_above_ceiling"
            values[name] = value
        return Limits(**values)

    def _call(self, message: dict[str, Any]) -> dict[str, Any]:
        from millm.rules.runner_host import guest_request

        limits = self._limits(message.get("limits"))
        if isinstance(limits, str):
            return {"error": limits}
        sha, function = message.get("code_sha256"), message.get("function")
        params, text = message.get("params"), message.get("text")
        if not isinstance(params, dict) or not isinstance(text, str) or not isinstance(function, str):
            return {"error": "request_invalid"}
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return {"error": "text_invalid"}
        if len(text) > limits.input_code_points:
            return {"reason": INPUT_TOO_LARGE}
        with self._lock:
            doc = self._documents.get(sha) if isinstance(sha, str) else None
            if doc is not None:
                self._documents.move_to_end(sha)
        if doc is None:
            return {"error": proto.UNKNOWN_MODULE}
        if function not in doc.functions:
            return {"error": "function_not_declared"}
        if self.status != "ok" or self.host is None:
            return {"reason": WORKER_UNAVAILABLE}
        if not self._slots.acquire(timeout=limits.wall_ms / 1000):
            return {"reason": WORKER_UNAVAILABLE}
        try:
            outcome = self.host.run_one(
                guest_request(source=doc.source, function=function, params=params, text=text),
                limits)
        finally:
            self._slots.release()
        reduced = outcome.reduced
        if reduced.count is not None:
            return {"count": reduced.count}
        return {"reason": reduced.reason or "rule_output_invalid"}

    def close(self) -> None:
        if self.host is not None:
            self.host.close()


# ── the socket server ────────────────────────────────────────────────────────────────────


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        worker: Worker = self.server.worker  # type: ignore[attr-defined]
        sock: socket.socket = self.request
        while True:
            try:
                message = proto.read_frame(sock)
            except (proto.ProtocolError, OSError):
                return
            try:
                reply = worker.handle(message)
            except Exception as exc:  # noqa: BLE001 - the class only (FR-39.28)
                log.error("rule_runner_request_failed error=%s", type(exc).__name__)
                reply = {"error": "runner_internal"}
            try:
                proto.write_frame(sock, reply)
            except OSError:
                return


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def start(config: WorkerConfig) -> Worker:
    """The worker, with its own seccomp filter installed FIRST (D55): the filter is in force
    before the guest snapshot is read or any operator code is loaded. If it cannot be installed,
    a worker that refuses every call (fail closed)."""
    from millm.rules import runner_seccomp

    try:
        runner_seccomp.install()
    except runner_seccomp.SeccompUnavailable as exc:
        log.error("rule_runner_seccomp_refused code=%s", exc.code)
        return Worker(config, refused=SECCOMP_UNAVAILABLE)
    log.info("rule_runner_seccomp_installed filters=%s", runner_seccomp.installed_filters())
    return Worker(config)


class _Terminated(SystemExit):
    """Raised by the SIGTERM handler: leaves `serve_forever` like SIGINT's KeyboardInterrupt."""


def _on_sigterm(signum: int, frame: Any) -> None:
    raise _Terminated(0)


def install_sigterm_handler() -> bool:
    """Exit cleanly on SIGTERM, as on SIGINT (039 19.7, found live). In the sidecar the worker is
    PID 1, and the kernel delivers to PID 1 only the signals it HANDLES: with the default
    disposition every rollout waited the full grace period for the force-kill. Only the main
    thread may set a handler (the in-process test harness serves from a thread: nothing set)."""
    if threading.current_thread() is not threading.main_thread():
        return False
    signal.signal(signal.SIGTERM, _on_sigterm)
    return True


def serve(config: WorkerConfig, *, ready: Optional[threading.Event] = None) -> None:
    # Before `start`: a SIGTERM while the filter is installed or the snapshot loads exits too.
    install_sigterm_handler()
    worker = start(config)
    path = config.socket_path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    server = _Server(str(path), _Handler)
    server.worker = worker  # type: ignore[attr-defined]
    os.chmod(path, 0o660)
    log.info("rule_runner_listening status=%s", worker.status)
    if ready is not None:
        ready.set()
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        worker.close()


# ── image build helpers ──────────────────────────────────────────────────────────────────


def fetch(bundle_dir: Path) -> Path:
    """Download the published snapshot and keep it only if its sha256 is the contract's."""
    import urllib.request

    from millm.rules.runner_host import file_sha256

    snap = rc.snapshot()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    part = bundle_dir / "snapshot.wasm.part"
    with urllib.request.urlopen(snap["url"], timeout=300) as response, part.open("wb") as fh:  # noqa: S310 - the contract's https url
        while block := response.read(1 << 20):
            fh.write(block)
    if file_sha256(part) != snap["sha256"] or part.stat().st_size != int(snap["size"]):
        part.unlink()
        raise SystemExit("the downloaded snapshot is not the published guest (sha256 or size)")
    final = bundle_dir / "snapshot.wasm"
    part.replace(final)
    return final


def main(argv: Optional[list[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "serve"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if command == "fetch":
        print(fetch(Path(args[1])))
        return
    if command == "precompile":
        from millm.rules.runner_host import precompile

        print(precompile(Path(args[1]), rc.contract()))
        return
    if command == "seccomp-check":
        from millm.rules import runner_seccomp

        runner_seccomp.check()
        print("seccomp filter builds")
        return
    if command != "serve":
        raise SystemExit("usage: python -m millm.rules.runner_worker "
                         "[serve|fetch DIR|precompile DIR|seccomp-check]")
    try:
        serve(WorkerConfig.from_env())
    except _Terminated:
        log.info("rule_runner_stopped signal=SIGTERM")
        raise


if __name__ == "__main__":
    main()
