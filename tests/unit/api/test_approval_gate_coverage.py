"""ENH-008 L1, L2: every gated route, called by an agent, waits; called by the operator, runs.

The gated set is read from the LIVE app's OpenAPI (`x-approval-action`, published by
`mark_gated_routes` from the mounted route table) and must equal both the registered actions and
the CHECK constraint's list. Each row then calls its route through the live app:

* with `X-miLLM-Agent`: 202, `X-miLLM-Approval: pending`, exactly ONE stored row, and the work did
  NOT run (the service mock has 0 calls; nothing written to the database);
* without it: the route runs as before (the mock called once, with the request's arguments), and
  nothing is stored.

Controls (recorded in `0xcc/reviews/enh008_controls_2026-10-28.md`):
* delete one `@requires_approval_when_agent` line            -> `test_the_live_gated_set_*` red,
  and that route's `test_an_agent_call_waits_*` red (the work runs, 200, no row)
* `mark_gated_routes(app)` removed from `register_routes`     -> `test_the_live_gated_set_*` red
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import func, select

from millm.api.approval_gate import APPROVAL_ACTIONS, APPROVAL_HEADER
from millm.db.models.approval import APPROVAL_ACTION_NAMES
from millm.db.models.probe_capacity import ModelProbeLimits
from millm.db.models.rule_set import RuleSet
from millm.rules import runner_contract as rc
from tests.support import rule_set_factory as rf
from tests.unit.api.test_rule_code_followup import code_document, envelope, rule_set_with_code
from tests.unit.approval_fixtures import AGENT, harness, make_model, rows  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

EXPECTED = {
    ("post", "/api/models/{model_id}/load"): "model_load",
    ("post", "/api/models/{model_id}/unload"): "model_unload",
    ("delete", "/api/models/{model_id}"): "model_delete",
    ("put", "/api/models/{model_id}/probe-limits"): "probe_limits_write",
    ("post", "/api/models/{model_id}/probe-capacity/measure"): "capacity_measure",
    ("post", "/api/rule-sets/import"): "code_rule_import",
    # ENH-006 D76: an agent's write that names a telemetry setting (the predicate; other keys run).
    ("put", "/api/environment"): "telemetry_write",
    ("put", "/api/environment/{key}"): "telemetry_write",
    ("delete", "/api/environment/{key}"): "telemetry_write",
}
ACTIONS = sorted(set(EXPECTED.values()))


def _code_body() -> bytes:
    code = code_document()
    return envelope(rule_set_with_code("enh008-gate", code), code)


async def _limits(h) -> int:
    async with h.db() as session:
        return (await session.execute(
            select(func.count()).select_from(ModelProbeLimits))).scalar_one()


async def _env_overrides(h) -> int:
    from millm.db.models.env_override import EnvOverride

    async with h.db() as session:
        return (await session.execute(select(func.count()).select_from(EnvOverride))).scalar_one()


async def _rule_sets(h) -> int:
    async with h.db() as session:
        return (await session.execute(select(func.count()).select_from(RuleSet))).scalar_one()


#: action -> (call(h, headers) -> response, ran(h) -> number of times the work ran)
def _calls() -> dict[str, tuple[Callable[..., Any], Callable[..., Any]]]:
    def load(h, headers):
        return h.client.post(f"/api/models/{h.holder['row'].id}/load", json={"gpu": "auto"},
                             headers=headers)

    def unload(h, headers):
        return h.client.post(f"/api/models/{h.holder['row'].id}/unload", headers=headers)

    def delete(h, headers):
        return h.client.delete(f"/api/models/{h.holder['row'].id}", headers=headers)

    def limits(h, headers):
        return h.client.put(f"/api/models/{h.holder['row'].id}/probe-limits",
                            json={"max_behaviours": 3, "reason": "pilot"}, headers=headers)

    def measure(h, headers):
        return h.client.post(f"/api/models/{h.holder['row'].id}/probe-capacity/measure",
                             headers=headers)

    def code_import(h, headers):
        return h.client.post("/api/rule-sets/import", content=_code_body(),
                             headers={**headers, "content-type": "application/json"})

    def telemetry(h, headers):
        return h.client.put("/api/environment/TELEMETRY_TRACES_SAMPLER_RATIO",
                            json={"value": 0.5, "reason": "pilot"}, headers=headers)

    async def n(mock) -> int:
        return mock.await_count

    return {
        "model_load": (load, lambda h: n(h.service.load_model)),
        "model_unload": (unload, lambda h: n(h.service.unload_model)),
        "model_delete": (delete, lambda h: n(h.service.delete_model)),
        "probe_limits_write": (limits, _limits),
        "capacity_measure": (measure, lambda h: n(h.measure)),
        "code_rule_import": (code_import, _rule_sets),
        "telemetry_write": (telemetry, _env_overrides),
    }


CALLS = _calls()


async def test_the_live_gated_set_is_the_registered_set(harness):  # noqa: F811
    spec = harness.app.openapi()
    live = {(method, path): op["x-approval-action"]
            for path, ops in spec["paths"].items() for method, op in ops.items()
            if isinstance(op, dict) and "x-approval-action" in op}
    assert live == EXPECTED
    assert set(live.values()) == set(APPROVAL_ACTIONS) == set(APPROVAL_ACTION_NAMES)
    assert set(CALLS) == set(APPROVAL_ACTIONS), "every action has a call below"


@pytest.mark.parametrize("action", ACTIONS)
async def test_an_agent_call_waits_and_does_not_run(harness, action):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    call, ran = CALLS[action]
    response = call(h, AGENT)
    assert response.status_code == 202, response.text
    assert response.headers[APPROVAL_HEADER] == "pending"
    data = response.json()["data"]
    assert data["status"] == "pending" and data["action"] == action
    assert data["approval_id"].startswith("apr_")
    assert data["request_digest"].startswith("sha256:")
    assert data["hint"] == "Waiting for the operator to approve this in miLLM's Approvals page."
    (stored,) = await rows(h.db)
    assert (stored.id, stored.action, stored.status, stored.requested_by) == (
        data["approval_id"], action, "pending", "agent:test-bot")
    assert await ran(h) == 0, "an agent's gated call must not run before it is approved"


@pytest.mark.parametrize("action", ACTIONS)
async def test_the_operator_call_runs_unchanged(harness, action):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    call, ran = CALLS[action]
    response = call(h, {})
    assert response.status_code in (200, 202), response.text
    assert APPROVAL_HEADER not in response.headers
    assert await rows(h.db) == []
    assert await ran(h) == 1


async def test_the_operator_load_passes_its_arguments_through(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_model(h.db)
    h.client.post(f"/api/models/{row.id}/load", json={"gpu": "all"},
                  headers={"X-miLLM-Lease": "lease-op"})
    h.service.load_model.assert_awaited_once_with(row.id, gpu="all", lease_id="lease-op")


@pytest.mark.parametrize("value", ["", "mcp", "agent:", "  agent:x  ", "agent:" + "y" * 200])
async def test_any_agent_header_value_gates(harness, value):  # noqa: F811
    """Fail closed: whatever the header says, the call is an agent's; the name is a label."""
    h = harness
    h.holder["row"] = await make_model(h.db)
    response = h.client.post(f"/api/models/{h.holder['row'].id}/unload",
                             headers={"X-miLLM-Agent": value})
    assert response.status_code == 202, response.text
    (stored,) = await rows(h.db)
    assert stored.requested_by.startswith("agent") and len(stored.requested_by) <= 64
    assert h.service.unload_model.await_count == 0


async def test_an_unknown_model_is_refused_before_anything_is_stored(harness):  # noqa: F811
    h = harness
    response = h.client.post("/api/models/999/load", headers=AGENT)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"
    assert await rows(h.db) == []


async def test_the_actions_route_lists_what_the_live_app_gates(harness):  # noqa: F811
    data = harness.client.get("/api/approvals/actions").json()["data"]
    listed = {a["action"]: a for a in data["actions"]}
    assert set(listed) == set(APPROVAL_ACTIONS)
    assert listed["model_load"]["routes"] == ["POST /api/models/{model_id}/load"]
    assert listed["model_delete"]["label"] == "Delete a model from miLLM"


async def test_the_code_rule_card_carries_the_full_source(harness):  # noqa: F811
    h = harness
    code = code_document()
    response = h.client.post("/api/rule-sets/import", content=_code_body(),
                             headers={**AGENT, "content-type": "application/json"})
    summary = response.json()["data"]["summary"]
    (module,) = summary["code_modules"]
    assert module["code_sha256"] == rc.code_sha256(code)
    assert module["source"] == code["source"]
    assert module["functions"] == sorted(f["name"] for f in code["functions"])
    assert module["runner_contract"] == rc.contract_format()
    assert summary["Revocation"] == "miLLM asks for each import; there is no revoked state here."
    assert "passed" in summary["Cases"]
    # the declarative document of a plain set never asks (L7 has the rest)
    plain = h.client.post("/api/rule-sets/import", content=json.dumps(rf.document("enh008-p")),
                          headers={**AGENT, "content-type": "application/json"})
    assert plain.status_code == 200, plain.text
