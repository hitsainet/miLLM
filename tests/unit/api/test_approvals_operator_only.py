"""ENH-008 L4 (OSD-56 in miLLM): an agent can never approve or reject.

Any `X-miLLM-Agent` header on approve or reject is refused 403 `approval_operator_only`, and the
row is unchanged: still pending, nothing ran, no decision recorded.

Control (recorded in `0xcc/reviews/enh008_controls_2026-10-28.md`): `refuse_agent(request)` removed
from the approve route -> `test_an_agent_cannot_approve` red (the request runs).
"""

from __future__ import annotations

import pytest

from tests.unit.approval_fixtures import AGENT, harness, make_model, rows  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture


async def _filed(h) -> str:
    h.holder["row"] = await make_model(h.db)
    r = h.client.post(f"/api/models/{h.holder['row'].id}/unload", headers=AGENT)
    assert r.status_code == 202, r.text
    return r.json()["data"]["approval_id"]


async def _unchanged(h) -> None:
    (stored,) = await rows(h.db)
    assert (stored.status, stored.decided_by, stored.decided_at, stored.reason) == (
        "pending", None, None, None)
    assert h.service.unload_model.await_count == 0


@pytest.mark.parametrize("value", ["agent:test-bot", "", "operator"])
async def test_an_agent_cannot_approve(harness, value):  # noqa: F811
    h = harness
    approval_id = await _filed(h)
    r = h.client.post(f"/api/approvals/{approval_id}/approve", headers={"X-miLLM-Agent": value})
    assert r.status_code == 403, r.text
    error = r.json()["error"]
    assert error["code"] == "approval_operator_only"
    assert error["message"] == ("An agent cannot approve or reject a request; the operator "
                                "decides in miLLM's Approvals page.")
    await _unchanged(h)


@pytest.mark.parametrize("value", ["agent:test-bot", ""])
async def test_an_agent_cannot_reject(harness, value):  # noqa: F811
    h = harness
    approval_id = await _filed(h)
    r = h.client.post(f"/api/approvals/{approval_id}/reject", json={"reason": "nope"},
                      headers={"X-miLLM-Agent": value})
    assert r.status_code == 403 and r.json()["error"]["code"] == "approval_operator_only"
    await _unchanged(h)


async def test_an_agent_can_read_its_request(harness):  # noqa: F811
    h = harness
    approval_id = await _filed(h)
    r = h.client.get(f"/api/approvals/{approval_id}", headers=AGENT)
    assert r.status_code == 200 and r.json()["data"]["status"] == "pending"
