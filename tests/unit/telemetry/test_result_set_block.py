"""A behaviour's `set` block in the exported result records (039 S1, FR-39.10; FTDD 038 §6.5).

The block's counts and digest are attributes (`mi.probe.set.*`); its two LISTS
(`missing_probe_ids`, `attached_rule_sets`) are not — no contract list may leave — and they are
removed before flattening, so a partial set is exported with NOTHING dropped and counted.

Control (S1-49): stop removing the lists in `telemetry.emit_results` → red (the lists reach the
allowlist filter and are dropped and counted).
"""

from __future__ import annotations

from millm.telemetry import allowlist
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.unit.telemetry.conftest import flush


async def test_a_partial_sets_counts_leave_and_its_lists_do_not(stub, telemetry_on):
    telemetry_on()
    allowlist.reset_for_tests()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])  # one probe of `b-alpha` armed …
        async with h.factory() as session:  # … and a second imported, not armed: a partial set
            await bf.import_probe(session, bf.definition("alpha-two", key=cap.ALPHA_KEY,
                                                             version=None))
        r = h.post_chat(cap.BODY, {})
        assert r.status_code == 200
        body = r.json()["millm_probe_verdicts"]
        assert body[0]["set"]["complete"] is False and body[0]["set"]["missing_probe_ids"], (
            "precondition: the set must be partial, or its lists are empty and this checks nothing")
        flush()
    records = [rec for rec in stub.logs if rec["event_name"] == "mi.probe.result"]
    assert records
    for rec in records:
        attrs = rec["attributes"]
        assert attrs["mi.probe.set.armed_members"] == 1
        assert attrs["mi.probe.set.total_members"] == 2
        assert attrs["mi.probe.set.complete"] is False
        assert "mi.probe.set.missing_probe_ids" not in attrs
    assert allowlist.attributes_dropped() == 0, "a set list was offered for export"
