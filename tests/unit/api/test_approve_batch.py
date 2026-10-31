"""ENH-006 D79 / D81: "Approve all settings changes" and "Approve all code rules" —
`POST /api/approvals/approve-batch` (`kind`: `settings`, the default, or `code_rules`).

Through the live app (`create_app`) over a real SQLite database (`approval_fixtures.harness`):

* the route is in the LIVE route table and is operator-only (an agent header: 403, nothing ran);
* a batch of three runs each exactly once THROUGH THE SINGLE APPROVE (`approval_service.approve`
  spied: call count, ids and the shared batch id), and the work itself ran once per request with
  that request's own arguments (`probe_limits.set_limits`, `runtime_settings.set_override`);
* an id that is not a waiting settings change (another action, already decided, unknown) refuses
  the WHOLE batch, 409 `approval_batch_refused` naming it, with nothing approved or run;
* one item failing while it runs does not stop the others; the response lists each outcome;
* each row is recorded on its own (operator, decided time) with the batch's `batch_id`, and the
  History listing shows each.

Settings changes are `probe_limits_write` and `telemetry_write`; code rules are
`code_rule_import` (`ApprovalAction.batch_kind`). A batch never mixes the two kinds: an id of the
other kind refuses the whole batch like any other bad id.

Controls (recorded in `0xcc/reviews/approve_all_settings_controls_2026-10-29.md`):
* the `@router.post("/approve-batch")` registration removed   -> `test_the_route_is_live_*` red
* the `if refused: raise` pre-check removed                     -> `test_a_non_settings_id_*` and
                                                                   `test_a_decided_or_unknown_id_*` red
* the loop approves only the first id                           -> `test_a_batch_of_three_*` red
* `refuse_agent(request)` removed from the batch route          -> `test_an_agent_cannot_*` red
* `batch_actions(kind)` replaced by every batchable action (kinds mixed)
                                                                -> `test_kinds_never_mix_*` red
* the code-rule batch approves only the first id                -> `test_a_batch_of_two_code_rules_*` red
"""

from __future__ import annotations

from typing import Any

import pytest

from millm.api.approval_gate import APPROVAL_ACTIONS, batch_actions, settings_actions
from millm.core.config import settings
from millm.services import approval_service, probe_limits, runtime_settings
from tests.unit.api.test_rule_code_followup import code_document, envelope, rule_set_with_code
from tests.unit.approval_fixtures import AGENT, harness, make_model, rows  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

BATCH = "/api/approvals/approve-batch"
RATIO = "TELEMETRY_TRACES_SAMPLER_RATIO"


@pytest.fixture
def setup(harness, monkeypatch):  # noqa: F811
    """The harness with two real models (the model service answers by id), spies on the single
    approve and on the two pieces of work, and the telemetry setting restored afterwards."""
    h = harness
    models: dict[int, Any] = {}

    async def _get_model(model_id: int) -> Any:
        return models[model_id]

    h.service.get_model.side_effect = _get_model
    h.models = models

    h.approves = []
    real_approve = approval_service.approve

    async def approve_spy(session, approval_id, **kw):
        h.approves.append((approval_id, kw.get("batch_id")))
        return await real_approve(session, approval_id, **kw)

    monkeypatch.setattr(approval_service, "approve", approve_spy)

    h.limits_calls = []
    real_limits = probe_limits.set_limits
    h.fail_model: set[int] = set()

    async def limits_spy(db_, model_id, body, **kw):
        h.limits_calls.append((model_id, body["max_behaviours"], body["reason"]))
        if model_id in h.fail_model:
            raise RuntimeError("boom")
        return await real_limits(db_, model_id, body, **kw)

    monkeypatch.setattr(probe_limits, "set_limits", limits_spy)

    h.override_calls = []
    real_override = runtime_settings.set_override

    async def override_spy(session, key, value, **kw):
        h.override_calls.append((key, value, kw.get("approval_id")))
        return await real_override(session, key, value, **kw)

    monkeypatch.setattr(runtime_settings, "set_override", override_spy)
    monkeypatch.setattr(settings, RATIO, getattr(settings, RATIO))
    return h


async def _model(h, name: str) -> Any:
    row = await make_model(h.db, name=name)
    h.models[row.id] = row
    return row


def _file_limits(h, row, behaviours: int) -> str:
    r = h.client.put(f"/api/models/{row.id}/probe-limits",
                     json={"max_behaviours": behaviours, "reason": f"pilot {behaviours}"},
                     headers=AGENT)
    assert r.status_code == 202, r.text
    assert r.json()["data"]["action"] == "probe_limits_write"
    return r.json()["data"]["approval_id"]


def _file_telemetry(h, ratio: float = 0.5) -> str:
    r = h.client.put(f"/api/environment/{RATIO}", json={"value": ratio, "reason": "sample"},
                     headers=AGENT)
    assert r.status_code == 202, r.text
    assert r.json()["data"]["action"] == "telemetry_write"
    return r.json()["data"]["approval_id"]


def _file_unload(h, row) -> str:
    r = h.client.post(f"/api/models/{row.id}/unload", headers=AGENT)
    assert r.status_code == 202, r.text
    return r.json()["data"]["approval_id"]


async def _three(h) -> tuple[Any, Any, list[str]]:
    a, b = await _model(h, "Model-A"), await _model(h, "Model-B")
    ids = [_file_limits(h, a, 2), _file_limits(h, b, 3), _file_telemetry(h)]
    return a, b, ids


async def _statuses(h) -> dict[str, str]:
    return {r.id: r.status for r in await rows(h.db)}


def test_the_settings_changes_are_probe_limits_and_telemetry():
    assert settings_actions() == batch_actions("settings") == {
        "probe_limits_write", "telemetry_write"}
    assert batch_actions("code_rules") == {"code_rule_import"}
    assert {n for n, a in APPROVAL_ACTIONS.items() if a.batch_kind is None} == {
        "model_load", "model_unload", "model_delete", "capacity_measure"}


async def test_the_route_is_live_and_the_payload_says_which_are_settings(setup):
    h = setup
    paths = h.app.openapi()["paths"]
    assert "post" in paths[BATCH], "approve-batch must be in the LIVE route table"
    assert "x-approval-action" not in paths[BATCH]["post"]
    row = await _model(h, "Model-A")
    limits_id, unload_id = _file_limits(h, row, 2), _file_unload(h, row)
    listed = {a["approval_id"]: a for a in
              h.client.get("/api/approvals?status=pending").json()["data"]["approvals"]}
    assert listed[limits_id]["settings_change"] is True
    assert listed[unload_id]["settings_change"] is False
    assert listed[limits_id]["batch_id"] is None
    actions = {a["action"]: a for a in h.client.get("/api/approvals/actions").json()["data"]["actions"]}
    assert {n for n, a in actions.items() if a["settings_change"]} == {
        "probe_limits_write", "telemetry_write"}


@pytest.mark.parametrize("value", ["agent:test-bot", ""])
async def test_an_agent_cannot_approve_a_batch(setup, value):
    h = setup
    _, _, ids = await _three(h)
    r = h.client.post(BATCH, json={"approval_ids": ids}, headers={"X-miLLM-Agent": value})
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "approval_operator_only"
    assert set((await _statuses(h)).values()) == {"pending"}
    assert h.approves == [] and h.limits_calls == [] and h.override_calls == []


async def test_a_batch_of_three_runs_each_once_through_the_single_approve(setup):
    h = setup
    a, b, ids = await _three(h)
    h.events.clear()
    r = h.client.post(BATCH, json={"approval_ids": ids})
    assert r.status_code == 202, r.text
    data = r.json()["data"]
    batch_id = data["batch_id"]
    assert batch_id.startswith("bat_") and len(batch_id) == 36
    assert data["results"] == [{"approval_id": i, "status": "executed", "error": None}
                               for i in ids]
    # Each through the single approve, once, in order, with the shared batch id.
    assert h.approves == [(i, batch_id) for i in ids]
    # The work ran once per request, with that request's own arguments.
    assert h.limits_calls == [(a.id, 2, "pilot 2"), (b.id, 3, "pilot 3")]
    assert h.override_calls == [(RATIO, 0.5, ids[2])]
    # Each recorded on its own row, by the operator, with the batch id.
    stored = {r.id: r for r in await rows(h.db)}
    for i in ids:
        assert (stored[i].status, stored[i].decided_by, stored[i].batch_id) == (
            "executed", "operator", batch_id)
        assert stored[i].decided_at is not None
    # One resolved event per request (the banner refreshes).
    assert sorted(p["id"] for e, p in h.events if e == "approval:resolved") == sorted(ids)
    # History shows each, with the batch id; nothing is left waiting.
    history = {x["approval_id"]: x for x in
               h.client.get("/api/approvals?status=decided").json()["data"]["approvals"]}
    assert set(history) == set(ids)
    assert {x["batch_id"] for x in history.values()} == {batch_id}
    assert h.client.get("/api/approvals?status=pending").json()["data"]["pending"] == 0
    # A second batch of the same ids is refused whole, and nothing runs again.
    again = h.client.post(BATCH, json={"approval_ids": ids})
    assert again.status_code == 409 and again.json()["error"]["code"] == "approval_batch_refused"
    assert len(h.limits_calls) == 2 and len(h.override_calls) == 1


async def test_a_non_settings_id_refuses_the_whole_batch(setup):
    h = setup
    a, _, ids = await _three(h)
    unload_id = _file_unload(h, a)
    r = h.client.post(BATCH, json={"approval_ids": [*ids, unload_id]})
    assert r.status_code == 409, r.text
    error = r.json()["error"]
    assert error["code"] == "approval_batch_refused"
    assert error["details"]["refused"] == [
        {"approval_id": unload_id, "reason": "not_a_settings_change (model_unload)"}]
    assert unload_id in error["message"] and error["message"].startswith("Nothing was approved")
    assert set((await _statuses(h)).values()) == {"pending"}
    assert h.approves == [] and h.limits_calls == [] and h.override_calls == []
    assert h.service.unload_model.await_count == 0
    assert all(r.batch_id is None and r.decided_by is None for r in await rows(h.db))


async def test_a_decided_or_unknown_id_refuses_the_whole_batch(setup):
    h = setup
    _, _, ids = await _three(h)
    assert h.client.post(f"/api/approvals/{ids[0]}/approve").status_code == 202
    h.approves.clear()
    unknown = "apr_" + "0" * 32
    r = h.client.post(BATCH, json={"approval_ids": [ids[0], ids[1], unknown]})
    assert r.status_code == 409, r.text
    assert r.json()["error"]["details"]["refused"] == [
        {"approval_id": ids[0], "reason": "not_pending (executed)"},
        {"approval_id": unknown, "reason": "not_found"}]
    statuses = await _statuses(h)
    assert statuses[ids[1]] == "pending" and statuses[ids[2]] == "pending"
    assert h.approves == [] and len(h.limits_calls) == 1


async def test_one_failing_item_does_not_stop_the_others(setup):
    h = setup
    a, b, ids = await _three(h)
    h.fail_model.add(a.id)
    r = h.client.post(BATCH, json={"approval_ids": ids})
    assert r.status_code == 202, r.text
    results = r.json()["data"]["results"]
    assert [x["status"] for x in results] == ["failed", "executed", "executed"]
    assert results[0]["error"]["code"] == "execution_failed"
    assert results[1]["error"] is None and results[2]["error"] is None
    assert [c[0] for c in h.limits_calls] == [a.id, b.id]
    assert len(h.override_calls) == 1
    batch_id = r.json()["data"]["batch_id"]
    assert {x.batch_id for x in await rows(h.db)} == {batch_id}


async def test_one_id_and_duplicates_are_accepted_an_empty_list_is_not(setup):
    h = setup
    row = await _model(h, "Model-A")
    one = _file_limits(h, row, 2)
    r = h.client.post(BATCH, json={"approval_ids": [one, one]})
    assert r.status_code == 202, r.text
    assert [x["approval_id"] for x in r.json()["data"]["results"]] == [one]
    assert len(h.limits_calls) == 1
    assert h.client.post(BATCH, json={"approval_ids": []}).status_code == 422
    assert h.client.post(BATCH, json={"approval_ids": ["not-an-id"]}).status_code == 422


# ── D81: code rules ───────────────────────────────────────────────────────────────


def _file_code_rule(h, key: str) -> str:
    code = code_document()
    r = h.client.post("/api/rule-sets/import", content=envelope(rule_set_with_code(key, code), code),
                      headers={**AGENT, "content-type": "application/json"})
    assert r.status_code == 202, r.text
    data = r.json()["data"]
    assert data["action"] == "code_rule_import"
    return data["approval_id"]


async def _rule_set_keys(h) -> list[str]:
    from sqlalchemy import select

    from millm.db.models.rule_set import RuleSet

    async with h.db() as session:
        return sorted(r.behaviour_key for r in (await session.execute(select(RuleSet))).scalars())


async def test_a_batch_of_two_code_rules_runs_each_once_through_the_single_approve(setup):
    h = setup
    ids = [_file_code_rule(h, "batch-one"), _file_code_rule(h, "batch-two")]
    listed = {a["approval_id"]: a for a in
              h.client.get("/api/approvals?status=pending").json()["data"]["approvals"]}
    assert {listed[i]["batch_kind"] for i in ids} == {"code_rules"}
    assert not any(listed[i]["settings_change"] for i in ids)
    r = h.client.post(BATCH, json={"approval_ids": ids, "kind": "code_rules"})
    assert r.status_code == 202, r.text
    data = r.json()["data"]
    assert data["kind"] == "code_rules"
    assert data["results"] == [{"approval_id": i, "status": "executed", "error": None}
                               for i in ids]
    assert h.approves == [(i, data["batch_id"]) for i in ids]
    assert await _rule_set_keys(h) == ["batch-one", "batch-two"]
    stored = {r.id: r for r in await rows(h.db)}
    assert {(stored[i].status, stored[i].decided_by, stored[i].batch_id) for i in ids} == {
        ("executed", "operator", data["batch_id"])}


async def test_kinds_never_mix_either_way(setup):
    h = setup
    row = await _model(h, "Model-A")
    limits_a, limits_b = _file_limits(h, row, 2), _file_limits(h, row, 3)
    code_a, code_b = _file_code_rule(h, "mix-one"), _file_code_rule(h, "mix-two")
    # settings batch naming a code rule
    r = h.client.post(BATCH, json={"approval_ids": [limits_a, limits_b, code_a]})
    assert r.status_code == 409, r.text
    assert r.json()["error"]["details"]["refused"] == [
        {"approval_id": code_a, "reason": "not_a_settings_change (code_rule_import)"}]
    # code-rule batch naming a settings change
    r = h.client.post(BATCH, json={"approval_ids": [code_a, code_b, limits_a],
                                   "kind": "code_rules"})
    assert r.status_code == 409, r.text
    error = r.json()["error"]
    assert error["details"]["kind"] == "code_rules"
    assert error["details"]["refused"] == [
        {"approval_id": limits_a, "reason": "not_a_code_rule (probe_limits_write)"}]
    assert '"Approve all code rules" approves only waiting imports of operator code' in (
        error["message"])
    assert h.approves == [] and h.limits_calls == [] and await _rule_set_keys(h) == []
    assert set((await _statuses(h)).values()) == {"pending"}
    assert h.client.post(BATCH, json={"approval_ids": [code_a], "kind": "other"}
                         ).status_code == 422
