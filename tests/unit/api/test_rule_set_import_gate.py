"""ENH-008 L7: the rule-set import is gated ONLY when it names operator code, failing closed.

* an agent import carrying code: 202, nothing imported; approving imports it, code and all;
* an agent import of a declarative set: imported at once (200), nothing stored in the queue;
* a predicate that raises: the call is gated (a shape the check did not foresee is never run
  unapproved);
* `names_operator_code` finds a `code_rule` anywhere, and any `code_modules` key.

Control (recorded in `0xcc/reviews/enh008_controls_2026-10-28.md`): `decide` returns False when the
predicate raises -> `test_a_predicate_that_raises_gates_the_call` red.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from millm.api.approval_gate import decide
from millm.db.models.rule_set import RuleSet
from millm.rules import runner_contract as rc
from millm.rules.code_modules import names_operator_code
from tests.support import rule_set_factory as rf
from tests.unit.api.test_rule_code_followup import code_document, envelope, rule_set_with_code
from tests.unit.approval_fixtures import AGENT, harness, rows  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

JSON = {"content-type": "application/json"}


async def _imported(h) -> list[RuleSet]:
    async with h.db() as session:
        return list((await session.execute(select(RuleSet))).scalars().all())


async def test_an_import_with_code_waits_and_approving_imports_it(harness):  # noqa: F811
    h = harness
    code = code_document()
    body = envelope(rule_set_with_code("enh008-code", code), code)
    r = h.client.post("/api/rule-sets/import", content=body, params={"version": 2},
                      headers={**AGENT, **JSON})
    assert r.status_code == 202, r.text
    assert await _imported(h) == []
    approval_id = r.json()["data"]["approval_id"]
    done = h.client.post(f"/api/approvals/{approval_id}/approve").json()["data"]
    assert done["status"] == "executed", done
    (row,) = await _imported(h)
    assert row.code_modules == {rc.code_sha256(code): rc.canonical_json(code).decode("utf-8")}
    assert row.version == 2, "the stored query parameter is replayed"
    assert done["result"]["id"] == row.id and done["result"]["created"] is True


async def test_a_declarative_import_never_asks(harness):  # noqa: F811
    h = harness
    r = h.client.post("/api/rule-sets/import", content=json.dumps(rf.document("enh008-plain")),
                      headers={**AGENT, **JSON})
    assert r.status_code == 200, r.text
    assert len(await _imported(h)) == 1
    assert await rows(h.db) == []


async def test_a_predicate_that_raises_gates_the_call(harness, monkeypatch):  # noqa: F811
    import millm.rules.code_modules as code_modules

    def broken(_body):
        raise RuntimeError("a shape nobody foresaw")

    monkeypatch.setattr(code_modules, "names_operator_code", broken)
    h = harness
    r = h.client.post("/api/rule-sets/import", content=json.dumps(rf.document("enh008-raise")),
                      headers={**AGENT, **JSON})
    assert r.status_code == 202, r.text
    assert await _imported(h) == []


def test_decide_fails_closed():
    def raises(_values):
        raise ValueError("x")

    assert decide(raises, {}, "k") is True
    assert decide(lambda _v: False, {}, "k") is False
    assert decide(None, {}, "k") is True


@pytest.mark.parametrize("body,expected", [
    ({"rule_set": {}, "code_modules": {}}, True),
    ({"signals": [{"kind": "code_rule"}]}, True),
    ({"deep": [{"x": {"kind": "code_rule"}}]}, True),
    ({"signals": [{"kind": "token_set"}]}, False),
    ([], False),
])
def test_names_operator_code_looks_everywhere(body, expected):
    assert names_operator_code(body) is expected
