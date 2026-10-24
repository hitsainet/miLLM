"""Opt-in transcripts through the real `/v1` route, and where their text may and may not go
(FR-36.7, FR-36.8, FR-36.9, BR-002, D5).

⚠ PRIVACY TESTS ASSERT ABSENCE. A sentinel is sent with recording ON and must then appear in no
socket emission (captured at the REAL `progress_emitter`, not at the service), no log record, and
no list or timeline response — only in the one transcript route.

Controls (036 FTDD §10):
* the transcript writer call (`add_transcript`) removed           → `test_recording_on_writes_rows` red
* `strip_session_text` / the named-field build bypassed (text added) → `test_text_never_reaches_the_socket` red
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from sqlalchemy import select

from millm.core.config import settings
from millm.db.models.probe import ProbeEvent
from millm.db.models.probe_session import ProbeSessionTurn, ProbeTranscript
from tests.unit.api.test_probe_session_headers import _all, chat, post, served  # noqa: F401
from tests.unit.f28_fixtures import clean_state, db, no_cuda  # noqa: F401

SENTINEL = "w29 w28 w27"  # vocabulary words, so the tiny model's tokenizer keeps them


@pytest.fixture
def recording(monkeypatch):
    monkeypatch.setattr(settings, "PROBE_TRANSCRIPTS", "on")


async def test_recording_is_off_unless_turned_on(served):  # noqa: F811
    client, factory, _ = served
    r = post(client, chat([SENTINEL]))
    assert r.headers["X-miLLM-Transcript"] == "off"
    assert await _all(factory, ProbeTranscript) == []
    turns = await _all(factory, ProbeSessionTurn)
    assert [t.has_transcript for t in turns] == [False]


async def test_recording_on_writes_rows(served, recording):  # noqa: F811
    client, factory, _ = served
    r1 = post(client, chat([SENTINEL]))
    assert r1.headers["X-miLLM-Transcript"] == "recorded"
    reply = r1.json()["choices"][0]["message"]["content"]
    r2 = post(client, chat([SENTINEL, reply, "w5 w6"]))
    assert r2.headers["X-miLLM-Session-Id"] == r1.headers["X-miLLM-Session-Id"]
    texts = await _all(factory, ProbeTranscript)
    assert [(t.user_text, t.reply_text) for t in texts] == [
        (SENTINEL, reply), ("w5 w6", r2.json()["choices"][0]["message"]["content"]),
    ]  # the NEWEST user message only — never the history
    assert [t.has_transcript for t in await _all(factory, ProbeSessionTurn)] == [True, True]


async def test_streaming_and_non_streaming_record_the_same_row(served, recording):  # noqa: F811
    client, factory, _ = served
    plain = post(client, chat([SENTINEL]), {"X-miLLM-Session": "a"})
    with no_cuda(), client.stream(
        "POST", "/v1/chat/completions", json=chat([SENTINEL], stream=True),
        headers={"X-miLLM-Session": "b"},
    ) as r:
        assert r.headers["X-miLLM-Transcript"] == "recorded"
        streamed = ""
        for line in r.iter_lines():
            body = line.removeprefix("data: ").strip()
            if not body or body == "[DONE]":
                continue
            for choice in json.loads(body).get("choices") or []:
                streamed += (choice.get("delta") or {}).get("content") or ""
    texts = await _all(factory, ProbeTranscript)
    assert len(texts) == 2
    assert texts[0].reply_text == plain.json()["choices"][0]["message"]["content"]
    assert texts[1].reply_text == streamed  # exactly what the client received
    assert (texts[0].user_text, texts[0].reply_text) == (texts[1].user_text, texts[1].reply_text)


async def test_only_the_transcript_route_serves_the_text(served, recording):  # noqa: F811
    client, factory, _ = served
    sid = post(client, chat([SENTINEL])).headers["X-miLLM-Session-Id"]
    for path in ("/api/probes/sessions", f"/api/probes/sessions/{sid}", "/api/probes/events"):
        body = client.get(path).text
        assert SENTINEL not in body and "user_text" not in body and "reply_text" not in body, path
    rows = client.get(f"/api/probes/sessions/{sid}/transcript").json()["data"]
    assert [r["user_text"] for r in rows] == [SENTINEL]


async def test_delete_keep_and_delete_events(served, recording):  # noqa: F811
    client, factory, _ = served
    keep = post(client, chat([SENTINEL]), {"X-miLLM-Session": "k"}).headers["X-miLLM-Session-Id"]
    gone = post(client, chat([SENTINEL]), {"X-miLLM-Session": "d"}).headers["X-miLLM-Session-Id"]
    n_events = {s: sum(1 for e in await _all(factory, ProbeEvent) if e.session_id == s)
                for s in (keep, gone)}
    assert n_events[keep] >= 1 and n_events[gone] >= 1

    r = client.delete(f"/api/probes/sessions/{keep}", params={"events": "keep"}).json()["data"]
    assert r == {"turns": 1, "transcripts": 1, "events_deleted": 0, "events_kept": n_events[keep]}
    events = await _all(factory, ProbeEvent)
    detached = [e for e in events if e.session_id is None]
    assert len(detached) == n_events[keep] and {e.turn_index for e in detached} == {1}

    r = client.delete(f"/api/probes/sessions/{gone}", params={"events": "delete"}).json()["data"]
    assert r == {"turns": 1, "transcripts": 1, "events_deleted": n_events[gone], "events_kept": 0}
    assert len(await _all(factory, ProbeEvent)) == n_events[keep]
    assert await _all(factory, ProbeTranscript) == []
    assert await _all(factory, ProbeSessionTurn) == []
    assert client.get(f"/api/probes/sessions/{gone}/transcript").status_code == 404


async def test_text_never_reaches_the_socket(served, recording, monkeypatch):  # noqa: F811
    """The REAL emitter, with a recording socket server behind it."""
    from millm.sockets.progress import progress_emitter

    emitted: list[tuple[str, object]] = []

    class Sio:
        def emit(self, event, payload):
            emitted.append((event, payload))

            async def _done():
                return None

            return _done()

    monkeypatch.setattr(progress_emitter, "_sio", Sio())
    client, factory, _ = served
    post(client, chat([SENTINEL]), {"X-miLLM-Session": "sock"})
    await asyncio.sleep(0)
    events = {e for e, _ in emitted}
    assert {"probe:results", "probe:session"} <= events, emitted  # both were really emitted
    flat = json.dumps([p for _, p in emitted], default=str)
    assert SENTINEL not in flat
    for key in ("user_text", "reply_text", "context_text", "context_token_ids", "pairs"):
        assert key not in flat
    session_payloads = [p for e, p in emitted if e == "probe:session"]
    assert set(session_payloads[0]) == {
        "session_id", "turn_index", "request_id", "n_turns", "has_transcript", "at",
    }


async def test_text_never_reaches_a_log_record(served, recording, caplog, capsys):  # noqa: F811
    caplog.set_level(logging.DEBUG)
    client, factory, _ = served
    post(client, chat([SENTINEL]), {"X-miLLM-Session": "log"})
    assert await _all(factory, ProbeTranscript)  # it WAS recorded
    captured = capsys.readouterr()  # structlog's console output
    # The test database's DRIVER (aiosqlite) logs every bound parameter at DEBUG — a property of the
    # SQLite test driver at a level production never runs it at (production is asyncpg), and it
    # would show any column, `context_text` included. Every other logger is checked.
    logged = "\n".join(
        r.getMessage() + repr(r.__dict__) for r in caplog.records
        if not r.name.startswith(("aiosqlite", "sqlalchemy.engine"))
    )
    assert logged or captured.out, "nothing was logged at all — the check would be vacuous"
    assert SENTINEL not in logged
    assert SENTINEL not in captured.out and SENTINEL not in captured.err


async def test_one_turn_by_turn_index(served, recording):  # noqa: F811
    """036 phase 4: miStudio's spot-check reads ONE sampled turn (`?turn_index=`), or gets 404."""
    client, factory, _ = served
    r1 = post(client, chat([SENTINEL]), {"X-miLLM-Session": "t"})
    reply = r1.json()["choices"][0]["message"]["content"]
    post(client, chat([SENTINEL, reply, "w5 w6"]), {"X-miLLM-Session": "t"})
    sid = r1.headers["X-miLLM-Session-Id"]
    one = client.get(f"/api/probes/sessions/{sid}/transcript", params={"turn_index": 2}).json()
    # An OBJECT for one turn — what miStudio 038's spot-check reads off `data`.
    assert (one["data"]["turn_index"], one["data"]["user_text"]) == (2, "w5 w6")
    assert "reply_text" in one["data"]
    # Without `turn_index`: every recorded turn, as a list (the Sessions page's shape).
    every = client.get(f"/api/probes/sessions/{sid}/transcript").json()["data"]
    assert [r["turn_index"] for r in every] == [1, 2]
    missing = client.get(f"/api/probes/sessions/{sid}/transcript", params={"turn_index": 9})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "PROBE_SESSION_NOT_FOUND"
