"""Rev2 review of 040 P0 (2026-11-08): the three rule-set `next_tool` sites converted from literals
to `next_tools` constants (FTASKS 6.16) were pinned only by T15's "a constant is used" and "every
constant is used somewhere". Control C3 (the signal-limit hint swapped to DELETE_RULE_SET) left the
suite green; C2/C4 went red only because their constant happened to have no other user. The hint an
agent receives (`error.next`) is pinned here per refusal, through the real routes.
"""

from __future__ import annotations

from miserver.core.config import settings
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_set_factory as rf
from tests.unit.api.test_rule_set_routes import _post


async def test_each_rule_set_refusal_names_its_next_tool(monkeypatch):
    async with cap.Harness() as h:
        monkeypatch.setattr(settings, "RULE_SETS_MAX_IMPORTED", 1)
        first = _post(h, "/api/rule-sets/import", rf.document(bf.random_key()))
        assert first.status_code == 200, first.text
        r = _post(h, "/api/rule-sets/import", rf.document(bf.random_key()))
        assert (r.status_code, r.json()["error"]["next"]) == (409, "miserver_delete_rule_set"), r.text

        monkeypatch.setattr(settings, "RULE_SETS_MAX_IMPORTED", 100)
        monkeypatch.setattr(settings, "RULE_SET_MAX_SIGNALS", 8)  # the factory declares 4 + 5
        r = _post(h, "/api/rule-sets/import", rf.document(bf.random_key()))
        assert r.json()["error"]["details"]["limit"] == "RULE_SET_MAX_SIGNALS", r.text
        assert r.json()["error"]["next"] == "miserver_validate_rule_set", r.text
        monkeypatch.setattr(settings, "RULE_SET_MAX_SIGNALS", 64)

        rule_set_id = first.json()["data"]["id"]
        r = h.client.post(f"/api/rule-sets/{rule_set_id}/evaluate", json={})
        assert r.status_code in (400, 422), r.text
        assert r.json()["error"]["next"] == "miserver_evaluate_rule_set", r.text
