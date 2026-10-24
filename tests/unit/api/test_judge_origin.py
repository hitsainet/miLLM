"""`X-miLLM-Origin: judge` — a judge's own call is not user traffic (036 phase 4, FR-36.13, D14).

Through the real route, a real tiny model and a real armed probe: every probe records
`not_scored: judge_traffic` (the request never goes silently quiet — the verdict header lists each
probe), and there is no session, no turn and no text, even with recording on and a session header.

Wiring control (036 FTDD §10): removing the judge check in `_attach_probe_meta` turns
`test_judge_traffic_is_not_scored` red.
"""

from __future__ import annotations

import pytest

from millm.core.config import settings
from millm.db.models.probe import ProbeEvent
from millm.db.models.probe_session import ProbeSession, ProbeSessionTurn, ProbeTranscript
from millm.services.probe_runtime import ProbeRuntimeState
from tests.unit.api.test_probe_session_headers import _all, chat, post, served  # noqa: F401
from tests.unit.f28_fixtures import clean_state, db, no_cuda  # noqa: F401


@pytest.fixture
def recording(monkeypatch):
    monkeypatch.setattr(settings, "PROBE_TRANSCRIPTS", "on")


async def test_judge_traffic_is_not_scored(served, recording):  # noqa: F811
    client, factory, row = served
    r = post(client, chat(["w1 w2"]), {"X-miLLM-Origin": "judge", "X-miLLM-Session": "s1"})
    assert r.status_code == 200, r.text
    header = r.headers["X-miLLM-Probe-Verdicts"]
    assert 'not-scored;reason="judge_traffic"' in header and row.name in header
    assert "X-miLLM-Session-Id" not in r.headers and "X-miLLM-Transcript" not in r.headers
    events = await _all(factory, ProbeEvent)
    assert [(e.scored, e.not_scored_reason, e.session_id) for e in events] == [
        (False, "judge_traffic", None)]
    assert await _all(factory, ProbeSession) == []
    assert await _all(factory, ProbeSessionTurn) == []
    assert await _all(factory, ProbeTranscript) == []


async def test_a_streamed_judge_call_is_marked_too(served, recording):  # noqa: F811
    client, factory, _ = served
    with no_cuda(), client.stream("POST", "/v1/chat/completions", json=chat(["w1"], stream=True),
                                  headers={"X-miLLM-Origin": "Judge"}) as r:
        body = "".join(r.iter_text())
        assert "X-miLLM-Session-Id" not in r.headers
    assert "judge_traffic" in body  # the terminal verdict chunk says why
    assert [e.not_scored_reason for e in await _all(factory, ProbeEvent)] == ["judge_traffic"]
    assert await _all(factory, ProbeSession) == []


async def test_live_traffic_beside_it_is_still_scored(served):  # noqa: F811
    client, factory, _ = served
    post(client, chat(["w1"]), {"X-miLLM-Origin": "judge"})
    post(client, chat(["w1"]))
    events = await _all(factory, ProbeEvent)
    assert sorted((e.scored, e.not_scored_reason) for e in events) == [
        (False, "judge_traffic"), (True, None)]


async def test_an_unknown_origin_is_refused(served):  # noqa: F811
    client, factory, _ = served
    r = post(client, chat(["w1"]), {"X-miLLM-Origin": "batch"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "origin_invalid"
    assert await _all(factory, ProbeEvent) == []


async def test_a_judge_with_nothing_armed_is_served_unchanged(served):  # noqa: F811
    client, factory, _ = served
    ProbeRuntimeState.reset_for_tests()
    r = post(client, chat(["w1"]), {"X-miLLM-Origin": "judge"})
    assert r.status_code == 200
    assert "X-miLLM-Probe-Verdicts" not in r.headers and "X-miLLM-Session-Id" not in r.headers


async def test_text_completions_honour_the_marker(served):  # noqa: F811
    client, factory, _ = served
    with no_cuda():
        r = client.post("/v1/completions", json={"model": "tiny", "prompt": "w1", "max_tokens": 2},
                        headers={"X-miLLM-Origin": "judge"})
    assert r.status_code == 200, r.text
    assert [e.not_scored_reason for e in await _all(factory, ProbeEvent)] == ["judge_traffic"]
    assert await _all(factory, ProbeSession) == []
