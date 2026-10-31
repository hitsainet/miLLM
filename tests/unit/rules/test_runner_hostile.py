"""The hostile rule-set suite on the live chat route and the REAL worker (039 FTASKS 18.2; FR-39.26;
FTDD §10.5; TD39-35).

One operator-code signal per case, all attached to one generated behaviour, one real chat whose user
message holds a sentinel, the published guest snapshot in a worker SUBPROCESS:

- (5) loop forever, (6) allocate past the memory bound, (7) return a megabyte, (8) return the
  matched text: each is terminated or refused with its reason, scored after the response (D41).
  The runner's own reason (`rule_resource_limit`, `rule_output_invalid`) is a published
  `mi.probe.result/v1` code since suite decision D56, so the result says it as it is and the log
  names it too (`code_followup.published`); (8) never carries the text — at the worker's boundary
  its reply is exactly `{"reason": "rule_output_invalid"}`.
- The sentinel appears on NO surface or log: body, header, both socket messages, stored rows, the
  events list, the captured log records, and the worker's own stdout/stderr.
- The probe results equal the same request with no rule set.
- The code ran OUTSIDE the API process: the worker's process id is not ours, and a fresh API-side
  interpreter that runs a whole after-response job never imports `wasmtime`.

⚠ NOT WRITTEN HERE (ENH-006 D48): cases (1) open a socket, (2) read a file, (3) read an environment
variable, (4) spawn a process — the guest-source ESCAPE forms. The operator writes them; until then
they are the named pending set (`test_runner_vectors.PENDING_HOSTILE_CLASSES`). The host defences
they will exercise are pinned at host level by `test_runner_wasi_table.py` (no preopens, empty
environment, trapping `sock_*`/`proc_raise`, markers once and in order) and the container's by
`test_rule_runner_manifest.py`.
"""

from __future__ import annotations

import json
import logging
import os
import random
import socket
import string
import subprocess
import sys
import textwrap

import pytest

from millm.core.config import settings
from millm.rules import runner_contract as rc
from millm.rules import runner_protocol as proto
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_runner as rr
from tests.support import rule_set_factory as rf

#: Mixed-case letters drawn per run: no 4-character piece of it occurs in ids, digests or times.
SENTINEL = "".join(random.SystemRandom().choice(string.ascii_letters) for _ in range(14))
TEXT = f"marker ZQ-1234 {SENTINEL} w1"
BODY = {**cap.BODY, "messages": [{"role": "user", "content": TEXT}]}
CASES = json.loads(rc.CASES_PATH.read_text(encoding="utf-8"))
#: FTDD §10.5 case -> (the vendored reference document, the runner's own reason).
HOSTILE = {
    "loops": ("loop_forever", "rule_resource_limit"),          # (5)
    "allocates": ("alloc_big", "rule_resource_limit"),         # (6)
    "megabyte": ("return_megabyte", "rule_output_invalid"),    # (7)
    "echoes": ("return_text", "rule_output_invalid"),          # (8)
}


def _rule_set(key: str) -> tuple[dict, dict]:
    doc = rf.document(key)
    modules = {}
    for name, (doc_id, _reason) in HOSTILE.items():
        code = CASES["documents"][doc_id]
        sha = rc.code_sha256(code)
        modules[sha] = rc.canonical_json(code).decode("utf-8")
        doc["signals"].append({
            "id": f"h_{name}", "kind": "code_rule", "code_sha256": sha,
            "runner_contract": rc.contract_format(), "function": "f", "params": {},
            "outputs": [{"name": "hits", "type": "int"}]})
        doc["expressions"][name] = {"signal": f"h_{name}"}
    return doc, modules


def _sentinel_free(blob: str) -> bool:
    """Neither the sentinel nor any 4+ character piece of it (FR-39.28)."""
    return not any(SENTINEL[i:i + 4] in blob for i in range(len(SENTINEL) - 3))


@pytest.mark.asyncio
async def test_the_hostile_cases_are_contained_and_nothing_leaks(monkeypatch, caplog):
    from unittest.mock import patch

    from tests.unit.api.test_rule_code_followup import (
        _live_loop,
        _stored,
        _wait_for,
    )

    bundle = rr.bundle_dir()
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 65536)
    # Fuel, not wall time, decides these cases (the contract); a wide backstop keeps a loaded
    # host from turning `rule_resource_limit` into `rule_timeout` (seen once under load).
    monkeypatch.setattr(settings, "RULE_CODE_CALL_WALL_MS", 30_000)
    async with cap.Harness() as h:
        key = bf.random_key()
        async with h.factory() as session:
            for i in range(2):
                await bf.import_probe(session, bf.definition(
                    f"{key}-gen-{i}", key=key, layer=i % 2, scale=0.13 + 0.2 * i))
        assert h.client.post(f"/api/probes/behaviours/{key}/arm", json={}).status_code == 200
        before = [e for e in h.post_chat(BODY, {}).json()["millm_probe_verdicts"]
                  if e.get("kind", "probe") == "probe"]
        doc, modules = _rule_set(key)
        r = h.client.post("/api/rule-sets/import", content=json.dumps(
            {"rule_set": doc, "code_modules": modules}).encode(),
            headers={"content-type": "application/json"})
        assert r.status_code == 200, r.text
        rule_set_id = r.json()["data"]["id"]
        assert h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                            json={"rule_set_ids": [rule_set_id]}).status_code == 200
        emitted: list[dict] = []
        from millm.sockets.progress import progress_emitter

        runner = rr.RunnerProcess(bundle)
        with runner, patch.object(progress_emitter, "emit_probe_rule_results", emitted.append), \
                _live_loop(h) as client:
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(runner.socket))
            h._fresh_socket_throttle()
            response = client.post("/v1/chat/completions", json=BODY)
            assert response.status_code == 200, response.text
            request_id = response.json()["id"]
            entries = response.json()["millm_probe_verdicts"]
            rows = _wait_for(client, h, request_id, lambda rows: all(
                r.outcome != "skipped" for r in rows if r.signal in HOSTILE), timeout=60)
            events = client.get("/api/probes/events", params={"limit": 200}).text
            # The worker's process is not ours: the code ran outside the API process.
            with socket.socket(socket.AF_UNIX) as s:
                s.connect(str(runner.socket))
                proto.write_frame(s, {"op": "health"})
                health = proto.read_frame(s)
            assert health["status"] == "ok" and health["pid"] not in (os.getpid(), None)
            # (8) at the worker's boundary: the reply is the reason and nothing else.
            echo = CASES["documents"]["return_text"]
            with socket.socket(socket.AF_UNIX) as s:
                s.connect(str(runner.socket))
                proto.write_frame(s, {"op": "load", "code_sha256": rc.code_sha256(echo),
                                      "canonical_utf8": rc.canonical_json(echo).decode()})
                assert proto.read_frame(s) == {"ok": True}
                proto.write_frame(s, {"op": "call", "code_sha256": rc.code_sha256(echo),
                                      "function": "f", "params": {}, "text": TEXT,
                                      "limits": rr.floor_limits()})
                assert proto.read_frame(s) == {"reason": "rule_output_invalid"}
        worker_out, worker_err = runner.output
        hostile = [r for r in rows if r.signal in HOSTILE]
        assert {r.signal for r in hostile} == set(HOSTILE)
        for row in hostile:
            assert (row.outcome, row.reason, row.fires, row.count, row.status) == (
                "failed", HOSTILE[row.signal][1], None, 0, "final"), (
                row.signal, row.outcome, row.reason)
        for name, (_doc, reason) in HOSTILE.items():
            assert f"signal=h_{name} " in caplog.text and f"reason={reason}" in caplog.text
        # The probes are untouched by the hostile set.
        same = ("probe_id", "window", "score", "threshold", "verdict", "outcome", "reason")
        assert [{k: e.get(k) for k in same} for e in entries
                if e.get("kind", "probe") == "probe"] == [
            {k: e.get(k) for k in same} for e in before]
        # FR-39.28: the sentinel on no surface or log.
        surfaces = {
            "body": json.dumps(response.json()["millm_probe_verdicts"]),
            "header": response.headers.get("x-millm-probe-verdicts", ""),
            "probe:results": json.dumps(h.socket_messages),
            "probe:rule_results": json.dumps(emitted),
            "stored": json.dumps([r.result for r in rows]),
            "events": events,
            "logs": caplog.text,
            "worker stdout": worker_out.decode("utf-8", "replace"),
            "worker stderr": worker_err.decode("utf-8", "replace"),
        }
        assert emitted, "the revision was never emitted"
        leaks = [name for name, blob in surfaces.items() if not _sentinel_free(blob)]
        assert leaks == [], f"the sentinel reached: {leaks}"


def test_the_api_process_never_loads_the_engine(tmp_path):
    """A fresh API-side interpreter imports the app and runs one WHOLE after-response job (against
    a scripted worker): `wasmtime` and the worker's modules are never imported. Evaluating operator
    code in-process would have to import them."""
    script = textwrap.dedent(f"""
        import asyncio, sys
        from types import SimpleNamespace
        import millm.main  # noqa: F401 - the whole app
        from millm.core.config import settings
        from millm.rules import code_followup, runner_contract as rc
        from millm.rules.registry import AttachedSet
        from millm.rules.evaluator import CompiledRuleSet
        sys.path.insert(0, {str(rr.REPO)!r})
        from tests.unit.api.test_rule_code_followup import rule_set_with_code, code_document
        from tests.support.stub_runner import StubRunner
        code = code_document()
        doc = rule_set_with_code("b-isolate1", code)
        compiled = CompiledRuleSet(doc)
        from millm.rules.registry import document_forms
        attached = AttachedSet("rs_000000000001", "a" * 64, 1, "b-isolate1", compiled, False,
                               ("operator",), document_forms(doc),
                               ((rc.code_sha256(code), rc.canonical_json(code).decode()),))
        job = code_followup.job_for([("b-isolate1", ("all",), attached)], {{"all": "a b c"}})
        pending = SimpleNamespace(kind="rule", reason="rule_worker_unavailable",
                                  status="unavailable", rule_set_id="rs_000000000001",
                                  signal="operator", window="all", result_id="r1")
        with StubRunner() as stub:
            settings.RULE_RUNNER_SOCKET = str(stub.socket)
            revised = asyncio.run(code_followup.score(job, [pending], settings=settings))
        assert [(rid, o.outcome, o.count) for rid, o in revised] == [("r1", "completed", 2)], revised
        loaded = sorted(m for m in sys.modules if m == "wasmtime" or m.startswith("wasmtime.")
                        or m in ("millm.rules.runner_host", "millm.rules.runner_wasi",
                                 "millm.rules.runner_worker"))
        print("LOADED", loaded)
        assert loaded == [], loaded
    """)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(rr.REPO)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])}
    result = subprocess.run([sys.executable, "-c", script], cwd=str(rr.REPO), env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "LOADED []" in result.stdout


def test_only_the_worker_imports_the_engine():
    """By AST over `millm/`: `wasmtime` is imported by the worker's three modules only, and the
    worker modules by each other only. A new import elsewhere puts the engine in the API process."""
    import ast
    from pathlib import Path

    root = rr.REPO / "millm"
    worker = {"millm/rules/runner_worker.py", "millm/rules/runner_host.py",
              "millm/rules/runner_wasi.py"}
    offenders = []
    for path in root.rglob("*.py"):
        rel = str(path.relative_to(rr.REPO))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                engine = name == "wasmtime" or name.startswith("wasmtime.")
                worker_mod = any(name.startswith(m) for m in (
                    "millm.rules.runner_host", "millm.rules.runner_wasi",
                    "millm.rules.runner_worker"))
                if (engine or worker_mod) and rel not in worker:
                    offenders.append((rel, name))
    assert offenders == []
    assert Path(root / "rules" / "runner_worker.py").exists()
