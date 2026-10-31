"""The guest bundle: only the published snapshot runs; the precompiled form is a cache (039 S3).

- A worker whose snapshot is absent or not the contract's bytes stays UP and answers every call
  `rule_worker_unavailable` (it never runs an unverified guest), so the backend's results stay
  pending rather than wrong.
- `precompile` (the image build) writes a `.cwasm` the worker then deserialises instead of compiling;
  a sidecar naming another snapshot, Wasmtime or stack size is ignored (compiled from the verified
  snapshot instead).
"""

from __future__ import annotations

import shutil
import socket

import pytest

from millm.rules import runner_contract as rc
from millm.rules import runner_protocol as proto
from tests.support import rule_runner as rr

CODE = {"derived_from": {"sources": [], "statement": "constructed"},
        "functions": [{"name": "f", "output": {"type": "count"}}],
        "params_schema": {"type": "object"}, "runner_contract": "midataworks.runner/v1",
        "source": "def f(text, params):\n    return 1\n"}


def _ask(path, frames):
    out = []
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(str(path))
        for frame in frames:
            proto.write_frame(s, frame)
            out.append(proto.read_frame(s))
    return out


@pytest.mark.parametrize("snapshot", [None, b"\0asm not the published guest"])
def test_an_unverified_bundle_runs_nothing_and_says_unavailable(tmp_path, snapshot):
    rr.require_wasmtime()
    if snapshot is not None:
        (tmp_path / "snapshot.wasm").write_bytes(snapshot)
    sha = rc.code_sha256(CODE)
    with rr.RunnerProcess(tmp_path) as runner:
        health, loaded, called = _ask(runner.socket, [
            {"op": "health"},
            {"op": "load", "code_sha256": sha, "canonical_utf8": rc.canonical_json(CODE).decode()},
            {"op": "call", "code_sha256": sha, "function": "f", "params": {}, "text": "x",
             "limits": rr.floor_limits()},
        ])
    assert health["status"] == "bundle_mismatch"
    assert loaded == {"ok": True}
    assert called == {"reason": "rule_worker_unavailable"}


def test_the_precompiled_snapshot_is_used_and_a_foreign_one_ignored(tmp_path, monkeypatch):
    bundle = rr.bundle_dir()
    import json

    import wasmtime

    from millm.rules import runner_host as host

    shutil.copy(bundle / "snapshot.wasm", tmp_path / "snapshot.wasm")
    host.precompile(tmp_path, rc.contract())
    assert (tmp_path / "snapshot.cwasm").is_file()
    engine = host.Host.engine_for(rc.contract())
    compiled = []
    real = wasmtime.Module.from_file
    monkeypatch.setattr(wasmtime.Module, "from_file",
                        staticmethod(lambda *a, **k: compiled.append(1) or real(*a, **k)))
    assert host.load_bundle(engine, tmp_path, rc.contract()).status == "ok"
    assert compiled == [], "the precompiled form was not used"
    sidecar = tmp_path / "snapshot.cwasm.json"
    meta = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps({**meta, "wasmtime": "0.0.0"}))
    assert host.load_bundle(engine, tmp_path, rc.contract()).status == "ok"
    assert compiled == [1], "a sidecar for another Wasmtime must not be trusted"
