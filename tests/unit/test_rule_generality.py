"""Rule sets, any behaviour, one flow (039 FR-39.16, FR-39.32 S2 part; FTASKS 15.1, 15.2).

The pilot's authored rule sets are DATA under `tests/fixtures/behaviour_packs/rule_sets/`; every key
is read from the packs or generated per run, none is typed here. The same route flow — arm, import,
attach, one real chat — runs for a generated behaviour and a pilot behaviour attached AT ONCE, and
each set's signals report only under their own behaviour.

* masked value while probes fire (FTDD §16): the probe entries are identical to the same request
  with no set attached (no veto, D9) and the set's `masked` signal fires.
* Control (039 controls record): swap the registry's keys between the two sets → red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_set_factory as rf

pytestmark = pytest.mark.asyncio
PACKS = Path(__file__).resolve().parents[1] / "fixtures" / "behaviour_packs"


def _pilot(index: int) -> tuple[str, dict]:
    key = json.loads((PACKS / "pilot.json").read_text())["behaviours"][index]["key"]
    return key, json.loads((PACKS / "rule_sets" / f"{key}.json").read_text())


def _masked_text(doc: dict) -> str:
    return next(c["input"]["text"] for c in doc["cases"] if c["id"] == "masked")


async def _arm(h, key, start):
    async with h.factory() as session:
        for i in range(start, start + 2):
            await bf.import_probe(session, bf.definition(f"{key}-g{i}", key=key, layer=i % 2,
                                                         scale=0.13 + 0.2 * i))
    assert h.client.post(f"/api/probes/behaviours/{key}/arm", json={}).status_code == 200


def _attach(h, key, doc) -> str:
    r = h.client.post("/api/rule-sets/import", content=json.dumps(doc),
                      headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    rule_set_id = r.json()["data"]["id"]
    assert h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                        json={"rule_set_ids": [rule_set_id]}).status_code == 200
    return rule_set_id


@pytest.mark.parametrize("index", [1, 2, 3], ids=lambda i: f"pilot-{i}")
async def test_a_masked_value_fires_masked_and_changes_no_probe(index):
    key, doc = _pilot(index)
    body = {**cap.BODY, "messages": [{"role": "user", "content": _masked_text(doc)}]}
    async with cap.Harness() as h:
        await _arm(h, key, 0)
        before = [e for e in h.post_chat(body, {}).json()["millm_probe_verdicts"]]
        _attach(h, key, doc)
        after = h.post_chat(body, {}).json()["millm_probe_verdicts"]
        same = ("probe_id", "window", "score", "threshold", "verdict", "outcome")
        probes = [e for e in after if e.get("kind", "probe") == "probe"]
        assert [{k: e.get(k) for k in same} for e in probes] == [
            {k: e.get(k) for k in same} for e in before]
        rules = {(e["signal"], e["window"]): e["fires"] for e in after if e.get("kind") == "rule"}
        windows = {w for (_s, w) in rules}
        assert any(rules[("masked", w)] for w in windows), rules
        assert not any(rules[("valid", w)] for w in windows), rules


async def test_two_behaviours_sets_attached_at_once_never_cross():
    pilot, pilot_doc = _pilot(1)
    generated = bf.random_key()
    text = (next(c["input"]["text"] for c in pilot_doc["cases"] if c["id"] == "valid-named")
            + " and marker ZQ-1234")
    body = {**cap.BODY, "messages": [{"role": "user", "content": text}]}
    async with cap.Harness() as h:
        await _arm(h, pilot, 0)
        await _arm(h, generated, 2)
        pilot_set = _attach(h, pilot, pilot_doc)
        generated_set = _attach(h, generated, rf.document(generated))
        rules = [e for e in h.post_chat(body, {}).json()["millm_probe_verdicts"]
                 if e.get("kind") == "rule"]
        owner = {pilot_set: pilot, generated_set: generated}
        assert {e["rule_set_id"] for e in rules} == set(owner)
        for e in rules:
            assert e["behaviour"]["key"] == owner[e["rule_set_id"]], e
        fired = {(e["rule_set_id"], e["signal"]) for e in rules if e["fires"]}
        assert (pilot_set, "valid") in fired and (generated_set, "token") in fired
