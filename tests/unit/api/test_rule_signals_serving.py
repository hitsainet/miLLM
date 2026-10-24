"""Rule signals at serving, through the LIVE app and a real chat (039 FR-39.22 – FR-39.24; FTASKS
13.1 – 13.3, 13.7, 14.1 – 14.3).

A generated behaviour is armed through the route, a generated rule set is imported and attached
through the routes, and one real chat runs on the tiny real Llama. Then every surface is read: the
body field, the header, the stored `rule_signal_results` rows and the one `probe:results` socket
message.

Controls (039 controls record): rules run on a not-scored request → the judge test red; the switch
ignored → the switch test red; a set keyed under the wrong behaviour → the A/B test red; a veto
added (probe entries changed by a firing signal) → `test_no_veto` red; `_rules_finish` deleted at the
serving site → the surfaces test red; the generation counter not consulted → the budget tests red.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from millm.core.config import settings
from millm.db.models.rule_set import RuleSignalResult
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_set_factory as rf

pytestmark = pytest.mark.asyncio

TEXT = f"marker ZQ-1234 and {rf.LUHN_OK} w1 w2"
BODY = {**cap.BODY, "messages": [{"role": "user", "content": TEXT}]}


async def _arm(h, key, n=2, start=0):
    async with h.factory() as session:
        rows = [await bf.import_probe(session, bf.definition(
            f"{key}-gen-{i}", key=key, layer=i % 2, scale=0.13 + 0.2 * i))
            for i in range(start, start + n)]
    r = h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
    assert r.status_code == 200, r.text
    return rows


def _attach(h, key, doc=None, **body):
    doc = doc or rf.document(key)
    r = h.client.post("/api/rule-sets/import", content=rf.raw(doc),
                      headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    rule_set_id = r.json()["data"]["id"]
    r = h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                     json={"rule_set_ids": [rule_set_id], **body})
    assert r.status_code == 200, r.text
    return rule_set_id


def _rules(entries):
    return [e for e in entries if e.get("kind") == "rule"]


def _probes(entries):
    return [e for e in entries if e.get("kind", "probe") == "probe"]


async def _stored(h, request_id):
    async with h.factory() as session:
        return list((await session.execute(
            select(RuleSignalResult).where(RuleSignalResult.request_id == request_id)
            .order_by(RuleSignalResult.id))).scalars().all())


async def test_rule_entries_ride_every_surface_beside_the_probes(monkeypatch):
    # One behaviour with a five-signal set is one header group of 21 members: over the 2 KiB
    # default the WHOLE group is dropped (FR-39.12, `test_probe_header_bound.py`). Raised here so
    # the rule members can be read; the body, chunk and events always carry every result.
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 16384)
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        rule_set_id = _attach(h, key)
        response = h.post_chat(BODY, {})
        assert response.status_code == 200, response.text
        body = response.json()["millm_probe_verdicts"]
        rules = _rules(body)
        names = {"token", "cued", "number", "lexicon", "token_and_number"}
        assert {e["signal"] for e in rules} == names
        assert all(e["behaviour"] == {"key": key, "version": None} for e in rules)
        assert all(e["rule_set_id"] == rule_set_id and e["outcome"] == "completed" for e in rules)
        fired = {(e["signal"], e["window"]): e["fires"] for e in rules}
        assert fired[("token", "all")] is True and fired[("cued", "all")] is True
        token = next(e for e in rules if e["signal"] == "token")
        assert token["count"] == 1 and "score" not in token and "probe_id" not in token
        # Contiguous with the behaviour's probes, probes first (`kind` sorts before `rule`).
        kinds = [e.get("kind", "probe") for e in body]
        assert kinds == sorted(kinds)
        assert token["set"]["attached_rule_sets"][0]["rule_set_id"] == rule_set_id
        header = response.headers["x-millm-probe-verdicts"]
        assert f'"rule:{rule_set_id}.token";verdict=?1;count=1;behaviour="{key}";window=all' \
            in header
        stored = await _stored(h, response.json()["id"])
        assert len(stored) == len(rules)
        assert {r.result_id for r in stored} == {e["result_id"] for e in rules}
        assert all(r.behaviour_key == key and r.count is not None for r in stored)
        message = h.socket_messages[-1]
        assert {(e["signal"], e["window"]) for e in _rules(message["results"])} == {
            (e["signal"], e["window"]) for e in rules}
        events = h.client.get("/api/probes/events", params={"limit": 50}).json()["data"]
        assert {e["kind"] for e in events} == {"probe", "rule"}


async def test_no_veto_probe_entries_equal_the_request_without_the_set():
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        before = _probes(h.post_chat(BODY, {}).json()["millm_probe_verdicts"])
        _attach(h, key)
        after = h.post_chat(BODY, {}).json()["millm_probe_verdicts"]
        assert _rules(after)
        same = ("probe_id", "window", "score", "threshold", "verdict", "outcome", "reason")
        assert [{k: e.get(k) for k in same} for e in _probes(after)] == [
            {k: e.get(k) for k in same} for e in before]


async def test_judge_traffic_runs_no_rule_and_says_why():
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        _attach(h, key)
        response = h.post_chat(BODY, {"X-miLLM-Origin": "judge"})
        rules = _rules(response.json()["millm_probe_verdicts"])
        assert rules and all((e["outcome"], e["reason"], e["fires"], e["count"]) ==
                             ("skipped", "judge_traffic", None, 0) for e in rules)


async def test_the_switch_reports_disabled_and_keeps_the_attachment(monkeypatch):
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        rule_set_id = _attach(h, key)
        monkeypatch.setattr(settings, "RULE_SIGNALS_ENABLED", False)
        rules = _rules(h.post_chat(BODY, {}).json()["millm_probe_verdicts"])
        assert rules and all((e["outcome"], e["reason"]) == ("failed", "rule_signals_disabled")
                             for e in rules)
        view = h.client.get(f"/api/probes/behaviours/{key}/rule-signals").json()["data"]
        assert view["rule_set_ids"] == [rule_set_id]
        monkeypatch.setattr(settings, "RULE_SIGNALS_ENABLED", True)
        rules = _rules(h.post_chat(BODY, {}).json()["millm_probe_verdicts"])
        assert all(e["outcome"] == "completed" for e in rules)


async def test_a_set_runs_only_under_its_own_armed_behaviour():
    async with cap.Harness() as h:
        a, b, idle = bf.random_key(), bf.random_key(), bf.random_key()
        await _arm(h, a)
        await _arm(h, b, start=2)
        set_a = _attach(h, a)
        _attach(h, idle)  # attached, but no probe of `idle` is armed (D40): never runs
        rules = _rules(h.post_chat(BODY, {}).json()["millm_probe_verdicts"])
        assert rules and {e["behaviour"]["key"] for e in rules} == {a}
        assert {e["rule_set_id"] for e in rules} == {set_a}


async def test_a_narrowed_attachment_reports_only_its_signals():
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        doc = rf.document(key)
        r = h.client.post("/api/rule-sets/import", content=rf.raw(doc),
                          headers={"content-type": "application/json"})
        rule_set_id = r.json()["data"]["id"]
        h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                     json={"rule_set_ids": [rule_set_id], "signals": {rule_set_id: ["token"]}})
        rules = _rules(h.post_chat(BODY, {}).json()["millm_probe_verdicts"])
        assert {e["signal"] for e in rules} == {"token"}


async def test_operator_code_is_reported_pending_and_never_run():
    """D41 at serving: the code rule's expression reports `rule_worker_unavailable`; the inline
    signals of the same set run as usual."""
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        _attach(h, key, rf.document(key, code_rule=True))
        rules = _rules(h.post_chat(BODY, {}).json()["millm_probe_verdicts"])
        operator = [e for e in rules if e["signal"] == "operator"]
        assert operator and all(
            (e["outcome"], e["reason"], e["status"], e["fires"], e["form"]) ==
            ("skipped", "rule_worker_unavailable", "unavailable", None, "code_rule")
            for e in operator)
        assert any(e["signal"] == "token" and e["fires"] is True for e in rules)


async def test_a_set_in_use_cannot_be_deleted():
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        rule_set_id = _attach(h, key)
        r = h.client.delete(f"/api/rule-sets/{rule_set_id}")
        assert (r.status_code, r.json()["error"]["code"]) == (409, "RULE_SET_IN_USE")
        assert r.json()["error"]["details"]["behaviours"] == [key]
        assert r.json()["error"]["next"] == "millm_clear_behaviour_rule_signals"


async def test_each_trigger_rebuilds_the_registry():
    from millm.rules import registry

    async with cap.Harness() as h:
        key = bf.random_key()
        start = registry.generation
        rule_set_id = _attach(h, key)  # import + attach
        assert registry.generation == start + 2
        h.client.delete(f"/api/probes/behaviours/{key}/rule-signals")
        assert registry.generation == start + 3
        h.client.delete(f"/api/rule-sets/{rule_set_id}")
        assert registry.generation == start + 4
        assert not registry.current().for_behaviour(key)


async def test_the_session_timeline_shows_rule_entries_in_the_behaviours_column():
    async with cap.Harness() as h:
        key = bf.random_key()
        await _arm(h, key)
        rule_set_id = _attach(h, key)
        assert h.post_chat(BODY, {"X-miLLM-Session": "conv-rules-1"}).status_code == 200
        sessions = h.client.get("/api/probes/sessions").json()["data"]
        assert sessions, "no session was recorded"
        detail = h.client.get(f"/api/probes/sessions/{sessions[0]['id']}").json()["data"]
        column = next(c for c in detail["columns"] if c["column"] == key)
        assert column["rule_set_ids"] == [rule_set_id]
        entries = detail["turns"][0]["verdicts"]
        rules = [e for e in entries if e.get("kind") == "rule"]
        assert rules and all(e["column"] == key and "score" not in e for e in rules)
        kinds = [e.get("kind", "probe") for e in entries]
        assert kinds == sorted(kinds), "a turn's probe entries come first, then its rule entries"


async def test_stateless_scoring_by_behaviour_runs_its_attached_sets():
    """FR-39.13 rule part / FR-39.25: rule entries per input; spans only on request; token ids are
    decoded and said so; nothing is stored."""
    async with cap.Harness() as h:
        key = bf.random_key()
        async with h.factory() as session:
            for i in range(2):
                await bf.import_probe(session, bf.definition(f"{key}-s{i}", key=key, layer=i,
                                                             scale=0.13 + 0.2 * i))
        _attach(h, key)
        inputs = [{"messages": [{"role": "user", "content": "marker ZQ-1234"},
                                {"role": "assistant", "content": "w1 w2"}]},
                  {"text": "plain w1"}]
        r = h.client.post("/api/probes/score", json={"behaviours": [key], "inputs": inputs,
                                                     "return_rule_positions": True})
        assert r.status_code == 200, r.text
        first, second = r.json()["data"]["results"]
        token = [e for e in first["rules"] if e["signal"] == "token"]
        assert token and all(e["origin"] == "stateless" for e in first["rules"])
        prompt = next(e for e in token if e["window"] in ("prompt", "all"))
        assert prompt["fires"] is True and prompt["spans"] == [[7, 14]]
        assert not any(e["signal"] == "token" and e["fires"] for e in second["rules"])
        plain = h.client.post("/api/probes/score", json={"behaviours": [key], "inputs": inputs})
        assert all("spans" not in e for e in plain.json()["data"]["results"][0]["rules"])
        ids = h.client.post("/api/probes/score", json={
            "behaviours": [key], "inputs": [{"token_ids": [0, 6, 7]}]}).json()["data"]
        assert ids["results"][0]["rules_decoded_from_token_ids"] is True
        async with h.factory() as session:
            assert (await session.execute(select(RuleSignalResult))).scalars().all() == []
