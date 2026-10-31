"""ENH-008 L3, L5, L6, L8, L10 and the lifecycle around them, through the live app.

* L3  approve runs the stored request EXACTLY ONCE, with the stored arguments (payload and call
      count asserted); a second approve is 409 `approval_not_pending`; two concurrent approves run
      it once.
* L5  past `expires_at` (a moved clock): listed `expired`, approving is 409 `approval_expired`.
* L6  a payload changed after filing ends `failed` with `approval_tampered`, and nothing runs.
* L8  the lease id is in no API response, and `secret_payload` is cleared after every decision.
* L10 `approval:created` / `approval:resolved` are emitted once each, with their payloads.

Controls (recorded in `0xcc/reviews/enh008_controls_2026-10-28.md`):
* the claim loses `status == 'pending'`              -> `test_two_concurrent_approves_run_it_once` red
* `secret_payload=None` dropped from the finish        -> `test_the_lease_never_leaves_*` red
* the History filter `status != 'pending'` removed    -> `test_history_lists_decided_newest_*` red
* `reason=reason` dropped from `reject`                -> `test_reject_records_reason_*` red
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

from sqlalchemy import text, update

from millm.core.errors import ModelLeasedError
from millm.db.models.approval import ApprovalRequest
from millm.services import approval_service
from tests.unit.approval_fixtures import AGENT, harness, make_model, rows  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

LEASE = "lease-secret-qzx81"


def _file_load(h, lease: str | None = LEASE) -> str:
    headers = {**AGENT, **({"X-miLLM-Lease": lease} if lease else {})}
    r = h.client.post(f"/api/models/{h.holder['row'].id}/load", json={"gpu": "all"},
                      headers=headers)
    assert r.status_code == 202, r.text
    return r.json()["data"]["approval_id"]


async def test_approve_runs_the_stored_request_exactly_once_with_its_arguments(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_model(h.db)
    approval_id = _file_load(h)
    r = h.client.post(f"/api/approvals/{approval_id}/approve")
    assert r.status_code == 202, r.text
    data = r.json()["data"]
    assert data["status"] == "executed" and data["decided_by"] == "operator"
    assert data["result"]["id"] == row.id and data["result"]["name"] == row.name
    h.service.load_model.assert_awaited_once_with(row.id, gpu="all", lease_id=LEASE)
    again = h.client.post(f"/api/approvals/{approval_id}/approve")
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "approval_not_pending"
    assert h.service.load_model.await_count == 1


async def test_two_concurrent_approves_run_it_once(harness):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    approval_id = _file_load(h)

    async def one():
        async with h.db() as session:
            return await approval_service.approve(session, approval_id, app=h.app)

    outcomes = await asyncio.gather(one(), one(), return_exceptions=True)
    await approval_service.wait_all()
    refused = [o for o in outcomes if isinstance(o, Exception)]
    assert len(refused) == 1 and refused[0].code == "approval_not_pending"
    assert h.service.load_model.await_count == 1


async def test_a_failed_run_is_recorded_with_its_error(harness):  # noqa: F811
    """Risk 3: a lease that lapsed by approval time fails the run, with miLLM's own code."""
    h = harness
    h.holder["row"] = await make_model(h.db)
    h.service.load_model.side_effect = ModelLeasedError("held by miDataworks",
                                                        details={"holder": "miDataworks"})
    approval_id = _file_load(h)
    data = h.client.post(f"/api/approvals/{approval_id}/approve").json()["data"]
    assert data["status"] == "failed"
    assert data["error"]["code"] == "MODEL_LEASED"
    assert data["error"]["message"] == "held by miDataworks"


async def test_a_long_run_answers_executing_and_finishes_in_the_background(harness):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    gate = asyncio.Event()

    async def slow(model_id, **_kw):
        await gate.wait()
        return h.holder["row"]

    h.service.unload_model.side_effect = slow
    r = h.client.post(f"/api/models/{h.holder['row'].id}/unload", headers=AGENT)
    approval_id = r.json()["data"]["approval_id"]
    async with h.db() as session:
        row = await approval_service.approve(session, approval_id, app=h.app, wait_s=0.05)
    assert row.status == "executing"
    gate.set()
    await approval_service.wait_all()
    (stored,) = await rows(h.db)
    assert stored.status == "executed" and h.service.unload_model.await_count == 1


async def test_an_expired_request_is_listed_expired_and_never_runs(harness, monkeypatch):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    approval_id = _file_load(h)
    later = approval_service.now() + timedelta(hours=25)
    monkeypatch.setattr(approval_service, "now", lambda: later)
    listed = h.client.get("/api/approvals").json()["data"]
    assert listed["pending"] == 0
    (row,) = listed["approvals"]
    assert row["status"] == "expired" and row["decided_by"] == "system"
    r = h.client.post(f"/api/approvals/{approval_id}/approve")
    assert r.status_code == 409 and r.json()["error"]["code"] == "approval_expired"
    assert h.service.load_model.await_count == 0
    (stored,) = await rows(h.db)
    assert stored.secret_payload is None, "an expiry clears the lease id too"


async def test_approving_past_expiry_with_no_read_first_refuses_and_never_runs(
        harness, monkeypatch):  # noqa: F811
    """The lazy expiry inside `approve` itself: nothing listed the request first."""
    h = harness
    h.holder["row"] = await make_model(h.db)
    approval_id = _file_load(h)
    later = approval_service.now() + timedelta(hours=25)
    monkeypatch.setattr(approval_service, "now", lambda: later)
    r = h.client.post(f"/api/approvals/{approval_id}/approve")
    assert r.status_code == 409 and r.json()["error"]["code"] == "approval_expired"
    assert h.service.load_model.await_count == 0
    r = h.client.post(f"/api/approvals/{approval_id}/reject", json={"reason": "late"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "approval_expired"


async def test_the_ttl_setting_sets_the_expiry(harness, monkeypatch):  # noqa: F811
    from millm.core.config import settings

    h = harness
    h.holder["row"] = await make_model(h.db)
    monkeypatch.setattr(settings, "APPROVAL_TTL_HOURS", 2)
    _file_load(h)
    (stored,) = await rows(h.db)
    assert stored.expires_at - stored.created_at == timedelta(hours=2)


async def test_a_tampered_payload_fails_and_does_not_run(harness):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    approval_id = _file_load(h)
    async with h.db() as session:
        await session.execute(update(ApprovalRequest).where(ApprovalRequest.id == approval_id)
                              .values(payload={"model_id": 999, "request": {"gpu": "all"}}))
        await session.commit()
    r = h.client.post(f"/api/approvals/{approval_id}/approve")
    assert r.status_code == 409 and r.json()["error"]["code"] == "approval_tampered"
    (stored,) = await rows(h.db)
    assert stored.status == "failed" and stored.error["code"] == "approval_tampered"
    assert h.service.load_model.await_count == 0


async def test_the_lease_never_leaves_the_server_and_is_cleared_at_decision(harness):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    first = h.client.post(f"/api/models/{h.holder['row'].id}/load", json={},
                          headers={**AGENT, "X-miLLM-Lease": LEASE})
    second = _file_load(h)
    seen = [first.text,
            h.client.get("/api/approvals").text,
            h.client.get(f"/api/approvals/{second}").text,
            h.client.get("/api/approvals?status=pending").text]
    (pending_row, *_) = [r for r in await rows(h.db) if r.id == second]
    assert pending_row.secret_payload == {"x_millm_lease": LEASE}, "kept for the run"
    seen.append(h.client.post(f"/api/approvals/{second}/approve").text)
    seen.append(h.client.post(f"/api/approvals/{first.json()['data']['approval_id']}/reject",
                              json={"reason": "no"}).text)
    seen.append(h.client.get("/api/approvals?status=decided").text)
    for body in seen:
        assert LEASE not in body
        assert "secret_payload" not in body
    for stored in await rows(h.db):
        assert stored.status in ("executed", "rejected")
        assert stored.secret_payload is None, stored.status
    h.service.load_model.assert_awaited_once_with(h.holder["row"].id, gpu="all", lease_id=LEASE)


async def test_reject_records_reason_and_operator_and_never_runs(harness):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    approval_id = _file_load(h)
    r = h.client.post(f"/api/approvals/{approval_id}/reject", json={"reason": "live check"})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert (data["status"], data["decided_by"], data["reason"]) == (
        "rejected", "operator", "live check")
    assert data["decided_at"] is not None
    assert h.client.post(f"/api/approvals/{approval_id}/approve").status_code == 409
    assert h.service.load_model.await_count == 0


async def test_reject_without_a_body_is_accepted(harness):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    approval_id = _file_load(h)
    r = h.client.post(f"/api/approvals/{approval_id}/reject")
    assert r.status_code == 200 and r.json()["data"]["reason"] is None


async def test_history_lists_decided_newest_decision_first(harness):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    a, b, c = _file_load(h), _file_load(h), _file_load(h)
    h.client.post(f"/api/approvals/{a}/reject", json={"reason": "first"})
    h.client.post(f"/api/approvals/{c}/reject", json={"reason": "second"})
    history = h.client.get("/api/approvals?status=decided").json()["data"]
    assert [x["approval_id"] for x in history["approvals"]] == [c, a]
    assert history["pending"] == 1 and history["count"] == 2
    waiting = h.client.get("/api/approvals?status=pending").json()["data"]
    assert [x["approval_id"] for x in waiting["approvals"]] == [b]
    everything = h.client.get("/api/approvals").json()["data"]
    assert [x["approval_id"] for x in everything["approvals"]] == [c, b, a]
    older = h.client.get("/api/approvals?status=decided&limit=1&offset=1").json()["data"]
    assert [x["approval_id"] for x in older["approvals"]] == [a]
    assert h.client.get("/api/approvals?status=bogus").status_code == 400
    assert h.client.get("/api/approvals?limit=501").status_code == 422


async def test_unknown_ids_are_404(harness):  # noqa: F811
    h = harness
    missing = "apr_" + "0" * 32
    for r in (h.client.get(f"/api/approvals/{missing}"),
              h.client.post(f"/api/approvals/{missing}/approve"),
              h.client.post(f"/api/approvals/{missing}/reject")):
        assert r.status_code == 404 and r.json()["error"]["code"] == "approval_not_found"


async def test_events_are_emitted_once_each_with_their_payloads(harness):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    approve_id = _file_load(h)
    reject_id = _file_load(h)
    h.client.post(f"/api/approvals/{approve_id}/approve")
    h.client.post(f"/api/approvals/{reject_id}/reject", json={})
    assert h.events == [
        ("approval:created", {"id": approve_id, "action": "model_load"}),
        ("approval:created", {"id": reject_id, "action": "model_load"}),
        ("approval:resolved", {"id": approve_id, "status": "executed"}),
        ("approval:resolved", {"id": reject_id, "status": "rejected"}),
    ]


async def test_an_expiry_emits_resolved(harness, monkeypatch):  # noqa: F811
    h = harness
    h.holder["row"] = await make_model(h.db)
    approval_id = _file_load(h)
    later = approval_service.now() + timedelta(hours=25)
    monkeypatch.setattr(approval_service, "now", lambda: later)
    h.client.get("/api/approvals")
    h.client.get("/api/approvals")
    assert h.events[-1] == ("approval:resolved", {"id": approval_id, "status": "expired"})
    assert sum(1 for e in h.events if e[0] == "approval:resolved") == 1


async def test_a_restart_fails_an_interrupted_run(db):  # noqa: F811
    """Risk 4: the startup reset marks every `executing` row failed and clears its lease."""
    from millm.main import STALE_STATE_RESETS

    (sql,) = [s for name, s, _ in STALE_STATE_RESETS if name == "failed_interrupted_approvals"]
    now = approval_service.now()
    async with db() as session:
        for status in ("executing", "pending"):
            session.add(ApprovalRequest(
                id=f"apr_{status:0<32}"[:36], action="model_load", executor_key="k", summary={},
                payload={}, secret_payload={"x_millm_lease": LEASE}, request_digest="0" * 64,
                requested_by="agent:x", status=status, created_at=now,
                expires_at=now + timedelta(hours=1)))
        await session.commit()
        await session.execute(text(sql))
        await session.commit()
    by_status = {r.id: r for r in await rows(db)}
    interrupted = by_status["apr_executing" + "0" * 23]
    assert interrupted.status == "failed" and interrupted.secret_payload is None
    error = interrupted.error if isinstance(interrupted.error, dict) else json.loads(
        interrupted.error)
    assert error["code"] == "interrupted_by_restart"
    assert by_status["apr_pending" + "0" * 25].status == "pending"
