"""Sessions through the REAL `/v1` routes, the real service and the real record path (FR-36.1–36.4).

A tiny real Llama on the CPU, a real armed probe hooked on its layer 0, a real SQLite database
behind `async_session_factory`. Nothing between the HTTP request and the event row is stubbed, so
removing the route's `resolve_request_meta` call, the begin path's attach or the record's
`session_id` key each turns a test here red.

Wiring control (036 FTDD §10): the `resolve_request_meta(...)` call in `chat.py` →
`test_header_names_the_session` red.
"""

from __future__ import annotations

import pytest
import torch
from sqlalchemy import select

from millm.db.models.probe import ProbeEvent
from millm.db.models.probe_session import ProbeSession, ProbeSessionTurn
from millm.ml.probe_head import ProbeHead
from millm.services import probe_sessions
from millm.services.probe_runtime import ArmedProbe, ProbeRuntimeState
from tests.unit.f25_fixtures import make_client, make_service, model_row, word_model, word_tokenizer
from tests.unit.f28_fixtures import clean_state, db, no_cuda  # noqa: F401 - fixtures
from tests.unit.probe_fixtures import probe_definition


def chat(text_turns, **over):
    """A conversation of alternating user/assistant messages ending on a user one."""
    messages = []
    for i, text in enumerate(text_turns):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
    body = {"model": "tiny", "messages": messages, "max_tokens": 2, "temperature": 0.0}
    body.update(over)
    return body


@pytest.fixture
async def served(clean_state, db):  # noqa: F811
    from millm.api.dependencies import get_db
    from millm.db.repositories.probe_repository import ProbeRepository
    from millm.services.probe_service import ProbeService

    probe_sessions._forget_for_tests()
    ProbeRuntimeState.reset_for_tests()
    model, tokenizer = word_model(), word_tokenizer()
    svc = make_service(model, tokenizer)
    async with db() as session:
        row = await ProbeService(ProbeRepository(session)).import_definition(
            probe_definition(), raw_bytes=1, on_conflict="rename", origin="file"
        )
    ProbeRuntimeState().arm(ArmedProbe(
        probe_id=row.id, name=row.name,
        head=ProbeHead(weight=torch.ones(16), bias=0.0, layer=0),
        rule="mean", scope="all", layer=0, rung=2, rung_language="detects on unseen tasks",
        threshold=0.0, windows=("all",),
    ), model)
    client, _ = make_client(svc, model_row())

    async def _db():
        async with db() as session:
            yield session

    client.app.dependency_overrides[get_db] = _db
    yield client, db, row
    ProbeRuntimeState.reset_for_tests()
    probe_sessions._forget_for_tests()


def post(client, body, headers=None):
    with no_cuda():
        return client.post("/v1/chat/completions", json=body, headers=headers or {})


async def _all(factory, model):
    async with factory() as session:
        return (await session.execute(select(model))).scalars().all()


async def test_header_names_the_session(served):
    client, factory, _ = served
    r1 = post(client, chat(["w1"]), {"X-miLLM-Session": "s1"})
    r2 = post(client, chat(["w2"]), {"X-miLLM-Session": "s1"})
    r3 = post(client, chat(["w1"]), {"X-miLLM-Session": "s2"})
    for r in (r1, r2, r3):
        assert r.status_code == 200, r.text
    ids = [r.headers.get("X-miLLM-Session-Id") for r in (r1, r2, r3)]
    assert ids[0] and ids[0].startswith("ps_")
    assert ids[0] == ids[1] != ids[2]
    sessions = await _all(factory, ProbeSession)
    assert sorted(s.client_session_id for s in sessions) == ["s1", "s2"]
    events = await _all(factory, ProbeEvent)
    by_request = {e.request_id: e.session_id for e in events}
    assert by_request[r1.json()["id"]] == by_request[r2.json()["id"]] == ids[0]
    assert by_request[r3.json()["id"]] == ids[2]
    assert r1.headers["X-miLLM-Transcript"] == "off"


async def test_a_continuation_joins_with_no_header(served):
    client, factory, _ = served
    r1 = post(client, chat(["w1"]))
    reply = r1.json()["choices"][0]["message"]["content"]
    r2 = post(client, chat(["w1", reply, "w3"]))
    assert r1.headers["X-miLLM-Session-Id"] == r2.headers["X-miLLM-Session-Id"]
    turns = await _all(factory, ProbeSessionTurn)
    assert sorted(t.turn_index for t in turns) == [1, 2]
    # The control: an edited first turn is a different conversation.
    r3 = post(client, chat(["w2", reply, "w3"]))
    assert r3.headers["X-miLLM-Session-Id"] != r1.headers["X-miLLM-Session-Id"]


async def test_streaming_carries_the_session_header_before_the_first_byte(served):
    client, factory, _ = served
    with no_cuda(), client.stream(
        "POST", "/v1/chat/completions", json=chat(["w1"], stream=True),
        headers={"X-miLLM-Session": "st"},
    ) as r:
        assert r.status_code == 200
        session_id = r.headers.get("X-miLLM-Session-Id")
        body = "".join(r.iter_text())
    assert session_id and "[DONE]" in body
    events = await _all(factory, ProbeEvent)
    assert events and {e.session_id for e in events} == {session_id}


async def test_an_invalid_header_is_refused_without_echoing_it(served):
    client, factory, _ = served
    bad = "x" * 129
    r = post(client, chat(["w1"]), {"X-miLLM-Session": bad})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "session_id_invalid"
    assert bad not in r.text
    r = post(client, chat(["w1"]), {"X-miLLM-Session": "tab\there"})
    assert r.status_code == 400
    assert await _all(factory, ProbeEvent) == []  # refused before any generation


async def test_nothing_armed_means_no_session_and_no_headers(served):
    client, factory, _ = served
    ProbeRuntimeState.reset_for_tests()
    r = post(client, chat(["w1"]), {"X-miLLM-Session": "s1"})
    assert r.status_code == 200
    assert "X-miLLM-Session-Id" not in r.headers and "X-miLLM-Transcript" not in r.headers
    assert await _all(factory, ProbeSession) == []


async def test_a_text_completion_is_a_single_turn_session(served):
    client, factory, _ = served
    with no_cuda():
        r = client.post("/v1/completions",
                        json={"model": "tiny", "prompt": "w1 w2", "max_tokens": 2})
    assert r.status_code == 200, r.text
    sessions = await _all(factory, ProbeSession)
    assert [(s.id, s.source) for s in sessions] == [(r.headers["X-miLLM-Session-Id"], "single")]
    assert r.headers["X-miLLM-Transcript"] == "off"
