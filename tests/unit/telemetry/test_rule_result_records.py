"""Rule results in the exported result records (039 S2; FTDD 038 §6.5).

A recorded `kind: "rule"` result leaves as one `mi.probe.result` record with its ids, digest, form,
window and COUNT as attributes — nothing dropped, and no text. Control: remove a rule field from
the allowlist → `attributes_dropped` is no longer 0 → red.
"""

from __future__ import annotations

from millm.telemetry import allowlist
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_set_factory as rf
from tests.unit.telemetry.conftest import flush


async def test_a_rule_result_record_leaves_whole_and_without_text(stub, telemetry_on):
    telemetry_on()
    allowlist.reset_for_tests()
    async with cap.Harness() as h:
        key = bf.random_key()
        async with h.factory() as session:
            for i in range(2):
                await bf.import_probe(session, bf.definition(f"{key}-t{i}", key=key, layer=i,
                                                             scale=0.13 + 0.2 * i))
        assert h.client.post(f"/api/probes/behaviours/{key}/arm", json={}).status_code == 200
        r = h.client.post("/api/rule-sets/import", content=rf.raw(rf.document(key)),
                          headers={"content-type": "application/json"})
        rule_set_id = r.json()["data"]["id"]
        h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                     json={"rule_set_ids": [rule_set_id]})
        body = {**cap.BODY, "messages": [{"role": "user", "content": "marker ZQ-1234 w1"}]}
        assert h.post_chat(body, {}).status_code == 200
        flush()
    records = [rec["attributes"] for rec in stub.logs if rec["event_name"] == "mi.probe.result"
               and rec["attributes"].get("mi.probe.kind") == "rule"]
    assert records, "no rule result was exported"
    token = next(a for a in records if a["mi.probe.signal"] == "token")
    assert (token["mi.probe.rule_set_id"], token["mi.probe.form"], token["mi.probe.count"]) == (
        rule_set_id, "pattern", 1)
    assert token["mi.probe.fires"] is True
    assert not any("ZQ-1234" in str(v) for a in records for v in a.values())
    assert allowlist.attributes_dropped() == 0, "a rule result field was offered and dropped"
