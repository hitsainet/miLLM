"""A matched value never leaves the process (039 FR-39.28 S2 part; FTASKS 14.6).

Per declarative form, a rule set whose signal FIRES on a sentinel the user sent. Then the sentinel
— whole, or any substring of 4 or more characters — must appear on NO surface: the socket, the
header, the stream's probe chunk, the body field, the events list, the session timeline, a Batch
line, the captured logs and standard error. A rule result carries a count, never a value.

Each sentinel uses only letters OUTSIDE hexadecimal (g–z), so a 4-character substring cannot
collide with a UUID or a digest by chance.

Control (039 controls record): add the matched text to a rule result (`RuleItem` → `_rule_one`)
→ red for every form.
"""

from __future__ import annotations

import json
import logging

import pytest

from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap

pytestmark = pytest.mark.asyncio

SENTINEL = "qzvkwmrt"


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


@pytest.mark.parametrize("form", ["pattern", "proximity", "validator", "lexical_weights"])
async def test_the_matched_value_reaches_no_surface(form, caplog, capfd, monkeypatch):
    from millm.core.config import settings

    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 16384)
    caplog.set_level(logging.DEBUG)
    async with cap.Harness() as h:
        key = bf.random_key()
        async with h.factory() as session:
            for i in range(2):
                await bf.import_probe(session, bf.definition(f"{key}-g{i}", key=key, layer=i,
                                                             scale=0.13 + 0.2 * i))
        assert h.client.post(f"/api/probes/behaviours/{key}/arm", json={}).status_code == 200
        doc, text = _document(key, form)
        r = h.client.post("/api/rule-sets/import", content=json.dumps(doc),
                          headers={"content-type": "application/json"})
        assert r.status_code == 200, r.text
        rule_set_id = r.json()["data"]["id"]
        h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                     json={"rule_set_ids": [rule_set_id]})
        body = {**cap.BODY, "messages": [{"role": "user", "content": text}]}
        response = h.post_chat(body, {"X-miLLM-Session": f"sentinel-{form}"})
        entries = response.json()["millm_probe_verdicts"]
        hit = [e for e in entries if e.get("kind") == "rule" and e["fires"]]
        assert hit, "the sentinel set did not fire — the test would prove nothing"
        stream = h.stream_chat(body, {})
        chunk = [line for line in stream.splitlines() if "millm_probe_verdicts" in line]
        events = h.client.get("/api/probes/events", params={"limit": 200}).text
        sessions = h.client.get("/api/probes/sessions").json()["data"]
        timeline = "".join(h.client.get(f"/api/probes/sessions/{s['id']}").text
                           for s in sessions)
        batch = json.dumps(await h.batch_line(body))
        logs = "\n".join(r.getMessage() + " " + json.dumps(getattr(r, "__dict__", {}),
                                                           default=str)
                         for r in caplog.records if not r.name.startswith("aiosqlite"))
        err = capfd.readouterr().err
        surfaces = {
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
