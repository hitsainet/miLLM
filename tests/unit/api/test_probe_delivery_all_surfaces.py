"""Every probe × window, once, under its own behaviour, on every surface (039 FR-39.10, FR-39.11,
FR-39.28; FTASKS 4.5, 4.8, 4.10).

The REAL chat route (streaming and not) on the tiny real Llama, two generated behaviours ARMED
THROUGH THE BEHAVIOUR ROUTE (three probes and two, different layers, thresholds and scores):
the header (parsed by `http-sfv`), the terminal chunk, the body field, the stored event rows, the
captured `probe:results` emit and one Batch output line all carry the same ordered list.

Privacy (FR-39.28): a sentinel built from the tiny vocabulary (so its words really are decoded into
the stored context window — the positive control) appears on the event DETAIL route only, never on
the socket, header, chunk, body, events list, session cells, Batch line or log records — whole or
as any 4-character substring.

Controls (039 controls record):
* `strip_context` keeps `context_*`             → `test_the_sentinel_stays_on_the_detail_route` red
* `(probe_id, window)` context key reverted       → its 036 test red (previous-round re-mutation)
"""

from __future__ import annotations

import json
import logging

import http_sfv
import pytest

from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap

pytestmark = pytest.mark.asyncio

SENTINEL_WORDS = "w17 w23 w5 w11 w29 w13"


async def _arm_two(h):
    a, b = sorted([bf.random_key(), bf.random_key()])
    rows = {}
    async with h.factory() as session:
        for key, n, layers in ((a, 3, (0, 1, 1)), (b, 2, (1, 0))):
            rows[key] = [await bf.import_probe(session, bf.definition(
                f"{key}-p{i}", key=key, layer=layers[i], threshold=(-0.2, 0.0, 0.3)[i],
                scale=0.15 + 0.11 * i + (0.5 if key == b else 0.0))) for i in range(n)]
    for key in (a, b):
        response = h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
        assert response.status_code == 200, response.text
    return a, b, rows


def _expected(rows):
    return {(r.id, w) for key in rows for r in rows[key] for w in ("all", "prompt", "response")}


async def test_every_surface_carries_every_result_once_in_contract_order():
    async with cap.Harness() as h:
        a, b, rows = await _arm_two(h)
        key_of = {r.id: key for key in rows for r in rows[key]}
        body_req = {**cap.BODY, "messages": [{"role": "user", "content": "w1 w2 w3"}]}
        response = h.post_chat(body_req, {})
        assert response.status_code == 200, response.text
        rid = response.json()["id"]
        body = response.json()["millm_probe_verdicts"]
        header = http_sfv.List()
        header.parse(response.headers["X-miLLM-Probe-Verdicts"].encode("ascii"))
        rows_stored = await h.event_rows(rid)
        socket = [m for m in h.socket_messages if m["request_id"] == rid]
        text = h.stream_chat(body_req, {})
        chunk = cap._stream_chunk(text)["millm_probe_verdicts"]
        batch = await h.batch_line(body_req)
        batch_body = batch["response"]["body"]["millm_probe_verdicts"]

        expected = _expected(rows)
        surfaces = {
            "body": [(e["probe_id"], e["window"]) for e in body],
            "chunk": [(e["probe_id"], e["window"]) for e in chunk],
            "header": [(m.params["probe"], str(m.params["window"])) for m in header],
            "rows": [(r["probe_id"], r["window"]) for r in rows_stored],
            "socket": [(r["probe_id"], r["window"]) for r in socket[0]["results"]],
            "batch": [(e["probe_id"], e["window"]) for e in batch_body],
        }
        for name, seen in surfaces.items():
            assert sorted(seen) == sorted(expected), name
            assert len(seen) == len(set(seen)), f"{name}: a result appears twice"
            keys = [key_of[pid] for pid, _w in seen]
            assert keys == [a] * 9 + [b] * 6, f"{name}: a behaviour's results are not contiguous"
        # Each result's own behaviour and score, and the same ordered list on body and chunk.
        for e in body:
            assert e["behaviour"]["key"] == key_of[e["probe_id"]]
            assert e["set"]["complete"] is True
        assert [(e["probe_id"], e["window"]) for e in body] == \
            [(e["probe_id"], e["window"]) for e in chunk]
        assert len({e["score"] for e in body if e["window"] == "all"}) == 5, "distinct scores"
        for m in header:
            assert m.params["behaviour"] == key_of[m.params["probe"]]


async def test_the_sentinel_stays_on_the_detail_route(caplog):
    caplog.set_level(logging.DEBUG)
    async with cap.Harness() as h:
        await _arm_two(h)
        body_req = {**cap.BODY, "messages": [{"role": "user", "content": SENTINEL_WORDS}]}
        response = h.post_chat(body_req, {"X-miLLM-Session": "privacy"})
        assert response.status_code == 200, response.text
        rid = response.json()["id"]
        listed = h.client.get("/api/probes/events", params={"request_id": rid}).json()["data"]
        details = [h.client.get(f"/api/probes/events/{e['id']}").json()["data"] for e in listed]
        # POSITIVE CONTROL: the words really reach the stored context the detail route serves.
        assert any("w17" in (d.get("context_text") or "") for d in details), \
            "the sentinel never reached the context window; this test would assert nothing"
        session_id = response.headers.get("X-miLLM-Session-Id")
        cells = h.client.get(f"/api/probes/sessions/{session_id}").json()["data"]
        stream = h.stream_chat(body_req, {})
        batch = await h.batch_line(body_req)
        public = {
            "header": response.headers.get("X-miLLM-Probe-Verdicts"),
            "body": response.json(),
            "chunk": cap._stream_chunk(stream),
            "socket": h.socket_messages,
            "events": listed,
            "session": cells,
            "batch": batch,
            # miLLM's own log records. (The SQLite test driver, `aiosqlite`, logs every
            # statement's PARAMETERS at DEBUG — including the context window this test stores for
            # the detail route. Production runs asyncpg, which does not; recorded in the 039
            # controls record, not a miLLM log.)
            "logs": [r.getMessage() + json.dumps(getattr(r, "__dict__", {}), default=str)
                     for r in caplog.records if not r.name.startswith("aiosqlite")],
        }
        words = SENTINEL_WORDS.split()
        fragments = {" ".join(words[i:i + 2]) for i in range(len(words) - 1)}
        for name, value in public.items():
            text = json.dumps(value, default=str)
            leaked = [f for f in fragments | {SENTINEL_WORDS} if f in text]
            # `pytest.fail`, not a bare `assert ... not in text`: pytest's diff of a large text
            # takes minutes.
            if leaked:
                at = text.index(leaked[0])
                pytest.fail(f"{name} carries the prompt's words {leaked}: "
                            f"...{text[max(at - 200, 0):at + 80]}...")
