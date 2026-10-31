"""039 S3 Review R1: the producer's Review R1 fixes, PINNED in miLLM's port (and the image fetch).

The port (`runner_worker` / `runner_host` / `runner_contract` / `runner_protocol`) carries every fix
miDataworks 018 R1 made — but miLLM's suite stayed green with each of them reverted (R1 controls
A6, A7, A8, A10, A12, A17, A19, A37, A43): the inclusive input limit, a bool refused as a limit, the
slot-wait backstop, the `.cwasm` fallback, the document's own `runner_contract`, the module cache
bound, the image fetch's digest check, the frame bound and the ceiling. Each test below goes red
against its reverted line (review record, negative controls).
"""

from __future__ import annotations

import json
import shutil
import socket
import struct
import threading
from pathlib import Path

import pytest

from millm.rules import runner_contract as rc
from millm.rules import runner_protocol as proto
from tests.support import rule_runner as rr

CASES = json.loads(rc.CASES_PATH.read_text(encoding="utf-8"))


def _worker(bundle: Path, **env: str):
    rr.require_wasmtime()
    from millm.rules.runner_worker import Worker, WorkerConfig

    return Worker(WorkerConfig.from_env({"RULE_RUNNER_BUNDLE_DIR": str(bundle), **env}))


def _floor_limits(**over: int) -> dict[str, int]:
    floor = rc.floor()
    limits = {name: floor.get(name, 2000) for name in rc.contract()["request_limits"]}
    limits.update(over)
    return limits


def _load(worker, doc_id: str) -> str:
    code = CASES["documents"][doc_id]
    sha = rc.code_sha256(code)
    assert worker.handle({"op": "load", "code_sha256": sha,
                          "canonical_utf8": rc.canonical_json(code).decode("utf-8")}) == {"ok": True}
    return sha


def _call(worker, sha: str, text: str, **limits: int) -> dict:
    return worker.handle({"op": "call", "code_sha256": sha, "function": "f", "params": {},
                          "text": text, "limits": _floor_limits(**limits)})


# ── the worker's request checks (producer MS4, MS6; the ceiling) ─────────────────────────────


def test_the_input_limit_is_inclusive(tmp_path):
    worker = _worker(tmp_path)  # no snapshot: the input check comes before anything runs
    sha = _load(worker, "count_tokens")
    limit = rc.floor()["input_code_points"]
    assert _call(worker, sha, "a" * limit) == {"reason": "rule_worker_unavailable"}
    assert _call(worker, sha, "a" * (limit + 1)) == {"reason": "rule_input_too_large"}


@pytest.mark.parametrize("name", ["call_fuel", "memory_bytes", "wall_ms"])
def test_a_bool_is_never_a_limit(tmp_path, name):
    worker = _worker(tmp_path)
    sha = _load(worker, "count_tokens")
    assert _call(worker, sha, "abc", **{name: True}) == {"error": "request_invalid"}


def test_a_limit_above_the_workers_ceiling_is_refused(tmp_path):
    worker = _worker(tmp_path, RULE_RUNNER_MAX_WALL_MS="1000")
    sha = _load(worker, "count_tokens")
    assert _call(worker, sha, "abc", wall_ms=1001) == {"error": "limit_above_ceiling"}
    assert _call(worker, sha, "abc", wall_ms=1000) == {"reason": "rule_worker_unavailable"}


def test_the_module_cache_is_bounded(tmp_path):
    worker = _worker(tmp_path, RULE_RUNNER_MODULE_CACHE="2")
    first = _load(worker, "count_tokens")
    for doc_id in ("return_zero", "loop_forever"):
        _load(worker, doc_id)
    assert worker.health()["cached_modules"] == 2
    assert _call(worker, first, "abc") == {"error": proto.UNKNOWN_MODULE}


def test_a_full_worker_answers_unavailable_within_the_wall_time():
    """Producer MS3: the slot wait is bounded by the call's wall time; past it the call is
    `rule_worker_unavailable` (pending), never a hang."""
    worker = _worker(rr.bundle_dir(), RULE_RUNNER_SLOTS="1")
    assert worker.status == "ok"
    sha = _load(worker, "return_zero")
    assert worker._slots.acquire(timeout=1)
    out: list[dict] = []
    thread = threading.Thread(target=lambda: out.append(_call(worker, sha, "abc", wall_ms=100)),
                              daemon=True)
    try:
        thread.start()
        thread.join(timeout=10)
        alive = thread.is_alive()
    finally:
        worker._slots.release()
        thread.join(timeout=30)
        worker.close()
    assert not alive, "the call waited for a slot past its wall time"
    assert out == [{"reason": "rule_worker_unavailable"}]


# ── the document's own contract (producer MS2) ─────────────────────────────────────────────────


def test_a_document_for_another_runner_contract_is_refused():
    code = dict(CASES["documents"]["count_tokens"])
    code["runner_contract"] = "midataworks.runner/v2"
    text = rc.canonical_json(code).decode("utf-8")
    with pytest.raises(rc.CodeDocumentRefused) as caught:
        rc.parse_code_document(rc.code_sha256(code), text)
    assert caught.value.code == "runner_contract_unknown"


# ── the frame bound ────────────────────────────────────────────────────────────────────────────


def test_a_frame_over_the_bound_is_refused_before_its_body_is_read():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(struct.pack(">I", proto.MAX_FRAME + 1))
        b.settimeout(5)
        with pytest.raises(proto.ProtocolError):
            proto.read_frame(b)


# ── the precompiled form (producer MR21) ───────────────────────────────────────────────────────


def test_a_cwasm_for_another_engine_falls_back_to_the_verified_snapshot(tmp_path):
    """A `.cwasm` whose sidecar matches but which was built for another engine configuration
    (here: no fuel, no epochs) must not stop the worker: it compiles the verified snapshot."""
    rr.require_wasmtime()  # before the import: a venv without wasmtime skips loudly, not errors
    import wasmtime

    from millm.rules import runner_host as host

    source = rr.bundle_dir() / "snapshot.wasm"
    shutil.copyfile(source, tmp_path / "snapshot.wasm")
    contract = rc.contract()
    foreign = wasmtime.Module.from_file(wasmtime.Engine(), str(tmp_path / "snapshot.wasm"))
    (tmp_path / "snapshot.cwasm").write_bytes(foreign.serialize())
    stack = int(contract["limits_floor"]["wasm_stack_bytes"])
    (tmp_path / "snapshot.cwasm.json").write_text(json.dumps(
        host._cwasm_meta(contract["guest"]["bundle"]["snapshot"]["sha256"], stack)))
    loaded = host.load_bundle(host.Host.engine_for(contract), tmp_path, contract)
    assert loaded.status == "ok" and loaded.module is not None


# ── the image fetch fails closed (focus 7) ─────────────────────────────────────────────────────


def _published(monkeypatch, tmp_path: Path, payload: bytes, *, sha: str, size: int) -> None:
    served = tmp_path / "served.wasm"
    served.write_bytes(payload)
    monkeypatch.setattr(rc, "snapshot", lambda: {"url": served.as_uri(), "sha256": sha,
                                                 "size": size})


def test_the_image_fetch_refuses_bytes_that_are_not_the_published_guest(monkeypatch, tmp_path):
    rr.require_wasmtime()
    import hashlib

    from millm.rules import runner_worker

    good = b"\x00asm published"
    sha = hashlib.sha256(good).hexdigest()
    target = tmp_path / "bundle"
    _published(monkeypatch, tmp_path, b"\x00asm tampered!", sha=sha, size=len(good))
    with pytest.raises(SystemExit):
        runner_worker.fetch(target)
    assert sorted(p.name for p in target.iterdir()) == [], "nothing of the bad download is kept"
    _published(monkeypatch, tmp_path, good + b"x", sha=sha, size=len(good) + 1)
    with pytest.raises(SystemExit):
        runner_worker.fetch(target)
    _published(monkeypatch, tmp_path, good, sha=sha, size=len(good))
    assert runner_worker.fetch(target).read_bytes() == good
