"""A scripted stand-in for the `rule-runner` worker, speaking the real framed protocol (039 S3).

Used where a test must CONTROL the worker's timing or misbehaviour — hold a reply until released
(D41: the response must not wait for it), die mid-call, forget its cache, send a malformed reply.
Results that must be the guest's own are proven with the REAL worker (`tests/support/rule_runner.py`);
this stub never runs code.

It records what it received WITHOUT the text: op, hash, function and the text's length.
"""

from __future__ import annotations

import shutil
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from millm.rules import runner_protocol as proto


class StubRunner:
    def __init__(self, reply: Optional[Callable[[dict[str, Any]], Optional[dict[str, Any]]]] = None,
                 *, hold: bool = False) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="msr"))
        self.socket = self._dir / "r.sock"
        self.reply = reply or (lambda m: {"count": 2})
        self.received: list[dict[str, Any]] = []
        self.loaded: set[str] = set()
        self.call_seen = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self._server = socket.socket(socket.AF_UNIX)
        self._server.bind(str(self.socket))
        self._server.listen(8)
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "StubRunner":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop = True
        self.release.set()
        try:
            self._server.close()
        finally:
            shutil.rmtree(self._dir, ignore_errors=True)

    def forget(self) -> None:
        """A restarted worker: its module cache is empty."""
        self.loaded.clear()

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._conn, args=(conn,), daemon=True).start()

    def _conn(self, conn: socket.socket) -> None:
        with conn:
            while True:
                try:
                    message = proto.read_frame(conn)
                except (proto.ProtocolError, OSError):
                    return
                record = {k: message.get(k) for k in ("op", "code_sha256", "function")}
                if "text" in message:
                    record["text_len"] = len(message["text"])
                if "limits" in message:
                    record["limits"] = message["limits"]
                self.received.append(record)
                op = message.get("op")
                if op == proto.OP_LOAD:
                    self.loaded.add(message["code_sha256"])
                    answer: Optional[dict[str, Any]] = {"ok": True}
                elif op == proto.OP_CALL:
                    if message.get("code_sha256") not in self.loaded:
                        answer = {"error": proto.UNKNOWN_MODULE}
                    else:
                        self.call_seen.set()
                        self.release.wait(timeout=60)
                        answer = self.reply(message)
                elif op == proto.OP_HEALTH:
                    answer = {"status": "ok", "pid": -1}
                else:
                    answer = {"error": "request_invalid"}
                if answer is None:  # the scripted crash: drop the connection mid-call
                    return
                try:
                    proto.write_frame(conn, answer)
                except OSError:
                    return

    def calls(self) -> list[dict[str, Any]]:
        return [r for r in self.received if r["op"] == proto.OP_CALL]
