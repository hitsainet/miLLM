"""The behaviour verdict through the LIVE app and a real chat on the tiny Llama (ENH-006 G19).

A random behaviour is armed through the route with two generated probes, a rule set with a MASKED form
(a veto) and an explicit positive (D53) is imported and attached, and a verdict definition is attached
through `PUT /api/probes/behaviours/{key}/verdict`. Member bars are forced (one member always fires or
never fires) so every verdict here is known in advance.

* AC 3 (binding): a mismatched member set, a combiner vector off by one verdict, an unattached rule set
  and a definition for another key are refused, nothing stored; a member re-cut after attach (another
  decision digest) makes every verdict provisional (`member_decision_changed`).
* AC 4 (no definition, no change): without a definition the header, the body and the stored events are
  those of the same request with none attached (the result ids and times aside).
* AC 5 (missing members): a member disarmed after attach → a request no armed member or rule proves
  reads `fires: null`, `decided_by: members_missing`, the member named.
* AC 6 (recording): the stored row equals the object the client received; no text, and not its base64
  form, in the verdict, the stored row, the socket message or the telemetry record.
* G19-3 / G19-4: a masked token vetoes the in-scope member and spares the out-of-scope one; an unmasked
  token is a rule match that beats the veto.
* AC 8 (registry): the three routes are in `app.openapi()["paths"]`.
"""

from __future__ import annotations

import base64
import copy
import json

import pytest
from sqlalchemy import select

from miserver.core.config import settings
from miserver.db.models.rule_set import BehaviourVerdictResult
from miserver.services import behaviour_verdict as bv
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_set_factory as rf
from tests.support import verdict_factory as vf

pytestmark = pytest.mark.asyncio

SECRET = "quoxel-secret-7731"
MATCH_TEXT = f"marker ZQ-1234 {SECRET} w1 w2"
MASKED_TEXT = f"see {vf.MASKED} here {SECRET} w1 w2"
PLAIN_TEXT = f"nothing here {SECRET} w1 w2"


def body(text: str) -> dict:
    return {**cap.BODY, "messages": [{"role": "user", "content": text}]}


async def arm(h, key, thresholds=(-1e9, 1e9)):
    """Two members: member 0 always fires, member 1 never fires (bars forced)."""
    async with h.factory() as session:
        for i, threshold in enumerate(thresholds):
            await bf.import_probe(session, bf.definition(f"{key}-m{i}", key=key, layer=i % 2,
                                                         scale=0.13 + 0.2 * i, threshold=threshold))
    r = h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
    assert r.status_code == 200, r.text
    members = vf.armed_members(key)
    by_name = {m.name: m for m in members}
    return [by_name[f"{key}-m{i}"] for i in range(len(thresholds))]


def attach_rules(h, key, doc):
    r = h.client.post("/api/rule-sets/import", content=rf.raw(doc), headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    r = h.client.put(f"/api/probes/behaviours/{key}/rule-signals", json={"rule_set_ids": [data["id"]]})
    assert r.status_code == 200, r.text
    return data["digest"]


def put(h, key, definition):
    return h.client.put(f"/api/probes/behaviours/{key}/verdict", json={"definition": definition, "reason": "test"})


async def setup(h, *, veto_scope=None, thresholds=(-1e9, 1e9), candidate="rules_or_any_of"):
    key = bf.random_key()
    members = await arm(h, key, thresholds)
    doc = vf.masked_document(key)
    digest = attach_rules(h, key, doc)
    definition = vf.definition(key, members, candidate=candidate, rule_doc=doc, veto_scope=veto_scope,
                               digest=digest)
    r = put(h, key, definition)
    assert r.status_code == 200, r.text
    return key, members, definition


def verdicts_of(entries):
    return [e for e in entries if e.get("kind") == "behaviour"]


async def test_a_rule_match_is_served_on_every_surface_and_stored_as_sent(monkeypatch):
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 16384)
    async with cap.Harness() as h:
        key, members, definition = await setup(h, thresholds=(1e9, 1e9))      # no member fires
        response = h.post_chat(body(MATCH_TEXT), {})
        assert response.status_code == 200, response.text
        entries = response.json()["millm_probe_verdicts"]
        (verdict,) = verdicts_of(entries)
        assert verdict["fires"] is True and verdict["decided_by"] == "rule_match"
        assert verdict["rules"]["match"] is True and verdict["rules"]["rule_set_digest"] == \
            definition["rules"]["rule_set_digest"]
        assert verdict["members_fired"] == [] and verdict["members_missing"] == []
        assert verdict["behaviour"] == {"key": key, "version": 1} and verdict["window"] == "all"
        assert verdict["definition_sha256"] == bv.definition_sha256(definition)
        # it leads its behaviour's group; the member results are all still there
        group = [e for e in entries if (e.get("behaviour") or {}).get("key") == key]
        assert group[0]["kind"] == "behaviour"
        assert {e["probe_id"] for e in group if e.get("kind", "probe") == "probe"} == {m.probe_id for m in members}
        header = response.headers["x-millm-probe-verdicts"]
        assert header.startswith(f'"behaviour:{key}";verdict=?1;decided-by="rule_match";behaviour="{key}";window=all')
        async with h.factory() as session:
            rows = (await session.execute(select(BehaviourVerdictResult).where(
                BehaviourVerdictResult.request_id == response.json()["id"]))).scalars().all()
        (row,) = rows
        stored = dict(row.result)
        assert stored.pop("persistence") == "persisted"
        assert stored == verdict                                          # AC 6: the row IS what was sent
        assert row.fires is True and row.decided_by == "rule_match" and row.behaviour_key == key
        message = h.socket_messages[-1]
        assert [e["result_id"] for e in verdicts_of(message["results"])] == [verdict["result_id"]]
        leak = json.dumps([verdict, row.result, verdicts_of(message["results"])])
        for form in (SECRET, base64.b64encode(SECRET.encode()).decode(), "ZQ-1234", "marker"):
            assert form not in leak, form


async def test_the_veto_suppresses_the_in_scope_member_and_spares_the_other():
    async with cap.Harness() as h:
        # member 0 (always fires) in the veto's scope, member 1 (never fires) outside it
        key, members, _definition = await setup(h, veto_scope=[0])
        (vetoed,) = verdicts_of(h.post_chat(body(MASKED_TEXT), {}).json()["millm_probe_verdicts"])
        assert vetoed["fires"] is False and vetoed["decided_by"] == "vetoed"
        assert [m["probe_id"] for m in vetoed["members_vetoed"]] == [members[0].probe_id]
        assert vetoed["rules"] == {"match": False, "veto": True, "unknown": False, "reason": None,
                                   "rule_set_digest": _definition["rules"]["rule_set_digest"]}
        (plain,) = verdicts_of(h.post_chat(body(PLAIN_TEXT), {}).json()["millm_probe_verdicts"])
        assert plain["fires"] is True and plain["decided_by"] == "members"
        assert [m["probe_id"] for m in plain["members_fired"]] == [members[0].probe_id]
    async with cap.Harness() as h:
        # the firing member OUTSIDE the scope fires through the veto
        key, members, _definition = await setup(h, veto_scope=[1])
        (spared,) = verdicts_of(h.post_chat(body(MASKED_TEXT), {}).json()["millm_probe_verdicts"])
        assert spared["fires"] is True and spared["decided_by"] == "members"
        assert spared["members_vetoed"] == [] and spared["rules"]["veto"] is True


async def test_no_definition_answers_as_before_on_every_surface(monkeypatch):
    """AC 4: the same request with and without an attached definition differ only by the verdict."""
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 16384)
    async with cap.Harness() as h:
        key = bf.random_key()
        await arm(h, key)
        doc = vf.masked_document(key)
        attach_rules(h, key, doc)
        before = h.post_chat(body(MATCH_TEXT), {})
        definition = vf.definition(key, vf.armed_members(key), rule_doc=doc,
                                   digest=h.client.get(f"/api/probes/behaviours/{key}/rule-signals")
                                   .json()["data"]["rule_sets"][0]["digest"])
        assert put(h, key, definition).status_code == 200
        with_def = h.post_chat(body(MATCH_TEXT), {})
        r = h.client.delete(f"/api/probes/behaviours/{key}/verdict", params={"reason": "done"})
        assert r.status_code == 200 and r.json()["data"]["attached"] is False
        after = h.post_chat(body(MATCH_TEXT), {})

        def strip(entries):
            drop = ("result_id", "event_time", "completed_at", "request_id", "trace_id", "span_id",
                    "evaluation_ms", "model_load_id", "id", "created_at", "source_event_id",
                    "session_id", "turn_index")  # each request opens its own session here
            return [{k: v for k, v in e.items() if k not in drop} for e in entries]

        b, a = before.json()["millm_probe_verdicts"], after.json()["millm_probe_verdicts"]
        assert not verdicts_of(b) and not verdicts_of(a)
        assert strip(b) == strip(a)
        assert before.headers["x-millm-probe-verdicts"] == after.headers["x-millm-probe-verdicts"]
        w = with_def.json()["millm_probe_verdicts"]
        assert strip([e for e in w if e.get("kind") != "behaviour"]) == strip(b)
        assert with_def.headers["x-millm-probe-verdicts"].split(", ", 1)[1] == \
            before.headers["x-millm-probe-verdicts"]
        async with h.factory() as session:
            assert len((await session.execute(select(BehaviourVerdictResult))).scalars().all()) == 1


async def test_a_disarmed_member_is_missing_never_a_false_clean():
    """AC 5. Member 0 (always fires) disarmed after attach: a plain text (no rule, member 1 silent)
    reads `fires: null`, member 0 missing. Control: return false for a missing member → red."""
    async with cap.Harness() as h:
        key, members, _definition = await setup(h)
        r = h.client.post(f"/api/probes/{members[0].probe_id}/disarm")
        assert r.status_code == 200, r.text
        (v,) = verdicts_of(h.post_chat(body(PLAIN_TEXT), {}).json()["millm_probe_verdicts"])
        assert v["fires"] is None and v["decided_by"] == "members_missing"
        assert v["members_missing"] == [{"detector_digest": members[0].descriptor.detector_digest,
                                         "probe_id": None}]
        # a rule still proves a positive with the member missing
        (m,) = verdicts_of(h.post_chat(body(MATCH_TEXT), {}).json()["millm_probe_verdicts"])
        assert m["fires"] is True and m["decided_by"] == "rule_match"
        header = h.post_chat(body(PLAIN_TEXT), {}).headers["x-millm-probe-verdicts"]
        assert f'"behaviour:{key}";verdict-unknown;decided-by="members_missing"' in header


async def test_attach_refusals_store_nothing():
    """AC 3: the members by digest, a combiner vector off by one verdict, the rule set, the key."""
    async with cap.Harness() as h:
        key = bf.random_key()
        members = await arm(h, key)
        doc = vf.masked_document(key)
        digest = attach_rules(h, key, doc)
        good = vf.definition(key, members, rule_doc=doc, digest=digest)

        one = vf.definition(key, members[:1], rule_doc=doc, digest=digest)
        r = put(h, key, one)
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "BEHAVIOUR_VERDICT_MEMBERS_MISMATCH"
        assert error["details"]["armed_not_in_definition"] == [members[1].descriptor.detector_digest]

        off = copy.deepcopy(good)
        off["combiner_vectors"][0]["expect"]["fires"] = not off["combiner_vectors"][0]["expect"]["fires"]
        r = put(h, key, off)
        assert r.status_code == 422 and r.json()["error"]["code"] == "VERDICT_PARITY_FAILED", r.text
        assert r.json()["error"]["details"]["vectors"] == [off["combiner_vectors"][0]["id"]]

        other = vf.definition(key, members, rule_doc=doc, digest="e" * 64)
        r = put(h, key, other)
        assert r.status_code == 409 and r.json()["error"]["code"] == "BEHAVIOUR_VERDICT_RULE_SET_NOT_ATTACHED"

        wrong_match = copy.deepcopy(good)
        wrong_match["rules"]["match"]["any_of"] = ["cued"]
        wrong_match["combiner_vectors"] = vf.vectors(wrong_match)
        r = put(h, key, wrong_match)
        assert r.status_code == 422 and r.json()["error"]["details"]["code"] == "match_mismatch", r.text

        tampered = copy.deepcopy(good)
        tampered["members_digest"] = "0" * 64
        r = put(h, key, tampered)
        assert r.status_code == 422 and r.json()["error"]["code"] == "BEHAVIOUR_VERDICT_INVALID"

        r = put(h, bf.random_key(), good)
        assert r.status_code == 422, r.text

        view = h.client.get(f"/api/probes/behaviours/{key}/verdict").json()["data"]
        assert view["attached"] is False and view["history"] == []
        assert verdicts_of(h.post_chat(body(MATCH_TEXT), {}).json()["millm_probe_verdicts"]) == []
        r = put(h, key, good)
        assert r.status_code == 200, r.text
        view = r.json()["data"]
        assert view["attached"] and view["serving"] and [x["action"] for x in view["history"]] == ["attach"]


async def test_a_member_recut_after_attach_makes_the_verdict_provisional():
    """AC 3: the armed decision digest moves away from the definition's → provisional, with why."""
    import dataclasses

    from miserver.services.probe_runtime import ProbeRuntimeState

    async with cap.Harness() as h:
        key, members, _definition = await setup(h)
        (before,) = verdicts_of(h.post_chat(body(PLAIN_TEXT), {}).json()["millm_probe_verdicts"])
        assert before["provisional"] is False and before["provisional_reasons"] == []
        state = ProbeRuntimeState()
        probe = state.get(members[1].probe_id)
        recut = dataclasses.replace(probe, descriptor=dataclasses.replace(
            probe.descriptor, decision=dataclasses.replace(probe.descriptor.decision, decision_digest="9" * 64)))
        assert state.refresh(recut)
        (after,) = verdicts_of(h.post_chat(body(PLAIN_TEXT), {}).json()["millm_probe_verdicts"])
        assert after["provisional"] is True and after["provisional_reasons"] == ["member_decision_changed"]


async def test_stateless_scoring_carries_each_inputs_verdict():
    async with cap.Harness() as h:
        key, members, _definition = await setup(h, thresholds=(1e9, 1e9))
        r = h.client.post("/api/probes/score", json={
            "probe_ids": [m.probe_id for m in members],
            "inputs": [{"messages": [{"role": "user", "content": MATCH_TEXT}]},
                       {"messages": [{"role": "user", "content": PLAIN_TEXT}]}]})
        assert r.status_code == 200, r.text
        results = r.json()["data"]["results"]
        first, second = (res["behaviours"] for res in results)
        assert [v["fires"] for v in first] == [True] and first[0]["decided_by"] == "rule_match"
        assert [v["fires"] for v in second] == [False] and second[0]["decided_by"] == "members"
        assert first[0]["origin"] == "stateless" and first[0]["input_index"] == 0
        assert first[0]["persistence"] == "not_persisted"
        assert SECRET not in json.dumps(results[0]["behaviours"])


def test_the_routes_are_in_the_live_openapi():
    """AC 8, from the live app's registry (never an import of the module)."""
    from miserver.main import create_app

    app = create_app()
    paths = app.openapi()["paths"]
    verdict = paths["/api/probes/behaviours/{key}/verdict"]
    assert {"get", "put", "delete"} <= set(verdict)
    schema = app.openapi()["components"]["schemas"]["BehaviourVerdictAttachRequest"]
    assert set(schema["properties"]) == {"definition", "reason"}


async def test_a_bounded_header_keeps_the_verdict_when_its_members_do_not_fit(monkeypatch):
    """The header bound drops whole member groups, but each behaviour's verdict is kept first
    (`build_probe_verdicts_header` → `bounded_members_keeping_leaders`). Control M15: bound the
    verdict with its group (the pre-G19 rule) → the verdict is lost with its members → red."""
    async with cap.Harness() as h:
        key, _members, _definition = await setup(h, thresholds=(1e9, 1e9))
        monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 160)
        response = h.post_chat(body(MATCH_TEXT), {})
        header = response.headers["x-millm-probe-verdicts"]
        assert len(header) <= 160
        assert header.startswith(f'"behaviour:{key}";verdict=?1;decided-by="rule_match"'), header
        assert ";truncated;total=" in header and '"rule:' not in header


# ── Suite-agent controls B6, B8, B9 (2026-11-09): each line below survived the whole suite when broken ──────────


async def _verdict_rows(h, key):
    async with h.factory() as session:
        return (await session.execute(select(BehaviourVerdictResult).where(
            BehaviourVerdictResult.behaviour_key == key))).scalars().all()


async def test_stored_verdicts_keep_the_live_cap(monkeypatch):
    """B6: deleting the `behaviour_verdict_store.prune(...)` call in the recorder left every verdict row forever
    (retention is a promise the probe events already keep; the verdicts share it)."""
    monkeypatch.setattr(settings, "PROBE_MAX_EVENTS_PER_PROBE", 2)
    async with cap.Harness() as h:
        key, _members, _definition = await setup(h, thresholds=(1e9, 1e9))
        for _ in range(4):
            r = h.post_chat(body(MATCH_TEXT), {})
            assert r.status_code == 200, r.text
        rows = await _verdict_rows(h, key)
        assert len(rows) == 2, [r.id for r in rows]


async def test_a_batch_rerun_stores_each_verdict_once():
    """B8: dropping `_without_recorded_batch_verdicts` stored a second copy of a batch line's verdict on a rerun."""
    async with cap.Harness() as h:
        key, _members, _definition = await setup(h, thresholds=(1e9, 1e9))
        await h.batch_line(body(MATCH_TEXT))
        first = await _verdict_rows(h, key)
        assert len(first) == 1 and first[0].batch_id == "batch_0123456789abcdef" and first[0].batch_line == 1
        # A rerun whose probe events are ALL recorded returns before any write (`without_recorded_batch_events`),
        # so the verdict skip only matters on a PARTIAL rerun: drop one recorded member event (e.g. pruned by
        # its own cap), rerun, and that event is written again while the verdict must not be.
        from sqlalchemy import delete

        from miserver.db.models.probe import ProbeEvent

        async with h.factory() as session:
            ids = (await session.execute(select(ProbeEvent.id).where(
                ProbeEvent.batch_id == "batch_0123456789abcdef").order_by(ProbeEvent.id))).scalars().all()
            assert ids, "the first run recorded no member events"
            await session.execute(delete(ProbeEvent).where(ProbeEvent.id == ids[0]))
            await session.commit()
        await h.batch_line(body(MATCH_TEXT))
        async with h.factory() as session:
            rewritten = (await session.execute(select(ProbeEvent.id).where(
                ProbeEvent.batch_id == "batch_0123456789abcdef"))).scalars().all()
        assert len(rewritten) == len(ids)                                  # the partial rerun did write
        again = await _verdict_rows(h, key)
        assert [r.id for r in again] == [first[0].id]


async def test_a_batch_lines_verdict_reaches_telemetry(monkeypatch):
    """B9: the batch path's `_publish(..., verdicts=verdict_rows)` with the verdicts dropped sent the probe and rule
    records to telemetry and never the behaviour verdict (a batch publishes to telemetry only, never the socket)."""
    from miserver import telemetry

    emitted: list = []
    monkeypatch.setattr(telemetry, "any_signal_on", lambda: True)
    monkeypatch.setattr(telemetry, "emit_results", lambda records: emitted.extend(records))
    async with cap.Harness() as h:
        key, _members, _definition = await setup(h, thresholds=(1e9, 1e9))
        socket_before = len(h.socket_messages)
        await h.batch_line(body(MATCH_TEXT))
        (row,) = await _verdict_rows(h, key)
        behaviour = [r for r in emitted if r.get("kind") == "behaviour"]
        assert [r["result_id"] for r in behaviour] == [row.result["result_id"]], emitted
        assert behaviour[0]["behaviour"]["key"] == key and behaviour[0]["fires"] is True
        assert len(h.socket_messages) == socket_before                   # a batch never reaches the socket
