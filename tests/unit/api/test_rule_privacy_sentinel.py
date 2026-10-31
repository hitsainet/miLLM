"""A matched value never leaves the process (039 FR-39.28 S2 part; FTASKS 14.6).

Per declarative form, a rule set whose signal FIRES on a sentinel the user sent. Then the sentinel
— whole, or any substring of 4 or more characters — must appear on NO surface: the socket, the
header, the stream's probe chunk, the body field, the events list, the session timeline, a Batch
line, the captured logs and standard error. A rule result carries a count, never a value.

Each sentinel uses only letters OUTSIDE hexadecimal (g–z), so a 4-character substring cannot
collide with a UUID or a digest by chance.

Control (039 controls record): add the matched text to a rule result (`RuleItem` → `_rule_one`)
→ red for every form.

039 S3 (FTASKS 18.4): parametrised with the OPERATOR-CODE form too. Its rule runs in the real
`rule-runner` worker on the published guest, after the response (D41), so the run keeps one server
loop alive, waits for the revision to land, and adds the revision's `probe:rule_results` message
and the worker's own output to the surfaces. Skips loudly without the guest snapshot.
"""

from __future__ import annotations

import contextlib
import json
import logging
from unittest.mock import patch

import pytest

from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_runner as rr

pytestmark = pytest.mark.asyncio

SENTINEL = "qzvkwmrt"
#: 039 S3: an operator-code rule that fires on the sentinel's shape. Its source names only the
#: 3-character prefix (below the 4-character rule), never the sentinel.
CODE = {"derived_from": {"sources": [], "statement": "constructed for the privacy sentinel"},
        "functions": [{"name": "f", "output": {"type": "count"}}],
        "params_schema": {"type": "object"}, "runner_contract": "midataworks.runner/v1",
        "source": "def f(text, params):\n    return sum(1 for t in text.split() "
                  "if t.startswith('qzv'))\n"}


def _signal(form: str) -> tuple[list[dict], str]:
    """(signals, text) for one form; the expression `hit` names the last signal."""
    word = {"id": "w", "kind": "pattern", "pattern": r"\bqzv[g-z]{5}\b", "ignore_case": False,
            "fold_fullwidth_ascii": False, "reads": "text"}
    if form == "pattern":
        return [word], f"say {SENTINEL} now"
    if form == "proximity":
        return [word, {"id": "p", "kind": "proximity", "anchor": "w", "phrases": ["code"],
                       "word_boundary": True, "max_distance": 4, "unit": "code_points",
                       "side": "before", "ignore_case": False,
                       "fold_fullwidth_ascii": False}], f"code {SENTINEL} now"
    if form == "validator":
        return [{"id": "v", "kind": "validator", "validator": {"id": "length_charset",
                                                                "version": 1},
                 "params": {"lengths": [8], "charset": "upper_alnum"},
                 "candidates": {"scan": True}}], f"say {SENTINEL.upper()} now"
    if form == "code_rule":
        from millm.rules import runner_contract as rc

        return [{"id": "c", "kind": "code_rule", "code_sha256": rc.code_sha256(CODE),
                 "runner_contract": rc.contract_format(), "function": "f", "params": {},
                 "outputs": [{"name": "hits", "type": "int"}]}], f"say {SENTINEL} now"
    return [{"id": "l", "kind": "lexical_weights", "terms": [{"term": SENTINEL, "weight": 1}],
             "cut": 1, "counting": "presence", "ignore_case": False,
             "fold_fullwidth_ascii": False, "word_boundary": True}], f"say {SENTINEL} now"


def _document(key: str, form: str) -> tuple[dict, str]:
    signals, text = _signal(form)
    doc = {
        "format": "midataworks.rule-set/v1", "name": f"sentinel-{form.replace('_', '-')}", "behaviour_key": key,
        "description": "A privacy sentinel set.", "signals": signals,
        "expressions": {"hit": {"signal": signals[-1]["id"]}},
        "outputs": [{"column": "hit", "type": "bool", "value": {"fired": "hit"}}],
        "cases": [
            {"id": "quiet", "input": {"text": "nothing here"}, "note": "", "expect": {
                "outputs": {"hit": False}, "vetoed": False}, "source": {"kind": "constructed"}},
        ],
    }
    return doc, text


def _leaks(blob: str, sentinel: str) -> list[str]:
    lowered = blob.lower()
    s = sentinel.lower()
    return sorted({s[i:j] for i in range(len(s)) for j in range(i + 4, len(s) + 1)
                   if s[i:j] in lowered})


@pytest.mark.parametrize("form", ["pattern", "proximity", "validator", "lexical_weights",
                                  "code_rule"])
async def test_the_matched_value_reaches_no_surface(form, caplog, capfd, monkeypatch):
    from millm.core.config import settings

    bundle = rr.bundle_dir() if form == "code_rule" else None
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 16384)
    caplog.set_level(logging.DEBUG)
    with contextlib.ExitStack() as stack:
        await _run(form, bundle, stack, caplog, capfd, monkeypatch)


async def _run(form, bundle, stack, caplog, capfd, monkeypatch):
    from millm.core.config import settings
    from millm.sockets.progress import progress_emitter

    revisions: list[dict] = []
    async with cap.Harness() as h:
        key = bf.random_key()
        async with h.factory() as session:
            for i in range(2):
                await bf.import_probe(session, bf.definition(f"{key}-g{i}", key=key, layer=i,
                                                             scale=0.13 + 0.2 * i))
        assert h.client.post(f"/api/probes/behaviours/{key}/arm", json={}).status_code == 200
        doc, text = _document(key, form)
        payload = doc
        if form == "code_rule":
            from millm.rules import runner_contract as rc

            payload = {"rule_set": doc, "code_modules": {
                rc.code_sha256(CODE): rc.canonical_json(CODE).decode("utf-8")}}
        r = h.client.post("/api/rule-sets/import", content=json.dumps(payload),
                          headers={"content-type": "application/json"})
        assert r.status_code == 200, r.text
        rule_set_id = r.json()["data"]["id"]
        h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                     json={"rule_set_ids": [rule_set_id]})
        body = {**cap.BODY, "messages": [{"role": "user", "content": text}]}
        runner = None
        if form == "code_rule":
            from tests.unit.api.test_rule_code_followup import _live_loop, _wait_for

            runner = stack.enter_context(rr.RunnerProcess(bundle))
            stack.enter_context(patch.object(progress_emitter, "emit_probe_rule_results",
                                             revisions.append))
            client = stack.enter_context(_live_loop(h))
            monkeypatch.setattr(settings, "RULE_RUNNER_SOCKET", str(runner.socket))
        response = h.post_chat(body, {"X-miLLM-Session": f"sentinel-{form}"})
        entries = response.json()["millm_probe_verdicts"]
        if form == "code_rule":
            from tests.unit.api.test_rule_code_followup import _stored

            rows = _wait_for(client, h, response.json()["id"],
                             lambda rows: any(r.fires for r in rows))
            hit = [r for r in rows if r.fires]
        else:
            hit = [e for e in entries if e.get("kind") == "rule" and e["fires"]]
        assert hit, "the sentinel set did not fire — the test would prove nothing"
        stream = h.stream_chat(body, {})
        chunk = [line for line in stream.splitlines() if "millm_probe_verdicts" in line]
        events = h.client.get("/api/probes/events", params={"limit": 200}).text
        sessions = h.client.get("/api/probes/sessions").json()["data"]
        timeline = "".join(h.client.get(f"/api/probes/sessions/{s['id']}").text
                           for s in sessions)
        # Review R2 (follow-ups): with the live loop open, the Batch line runs ON that loop. The
        # harness has ONE aiosqlite connection whose SQLAlchemy `_execute_mutex` is an
        # asyncio.Lock; awaited here on the pytest loop while the stream's after-response job
        # held it on the live loop, its release woke this loop with a non-threadsafe
        # `set_result` — lost, so the loop slept in epoll for good (the 20-minute hang of the
        # 039 follow-up runs, more likely under host load because the job runs longer).
        batch = json.dumps(client.portal.call(h.batch_line, body) if runner is not None
                           else await h.batch_line(body))
        logs = "\n".join(r.getMessage() + " " + json.dumps(getattr(r, "__dict__", {}),
                                                           default=str)
                         for r in caplog.records if not r.name.startswith("aiosqlite"))
        err = capfd.readouterr().err
        worker = b""
        if runner is not None:
            from millm.rules import code_followup

            client.portal.call(code_followup.drain, 20.0)
            assert revisions, "the after-response revision was never emitted"
            worker = b"\n".join(runner.stop())
        surfaces = {
            "rule revisions": json.dumps(revisions),
            "worker output": worker.decode("utf-8", "replace"),
            "socket": json.dumps(h.socket_messages),
            "header": response.headers.get("x-millm-probe-verdicts", ""),
            "chunk": "\n".join(chunk),
            "body": json.dumps(entries),
            "events": events,
            "sessions": timeline,
            # The whole Batch line: its body is the model's reply and the probe entries.
            "batch": batch,
            "logs": logs,
            "stderr": err,
        }
        assert chunk, "the stream carried no probe chunk"
        for name, blob in surfaces.items():
            assert not _leaks(blob, SENTINEL), (name, _leaks(blob, SENTINEL))
