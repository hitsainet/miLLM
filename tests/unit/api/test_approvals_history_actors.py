"""T28: approval history stays readable across the rename (040 FR-040.32).

A request stored BEFORE P4 holds only its action key and the actor id the agent sent then
(the old MCP actor id, DR-17; its one literal lives in the DR-17 exercise). Its label is read at answer time from `APPROVAL_ACTIONS`, so the
card says miServer without the row changing; and the new actor id (`agent:miworkbench-mcp`) needs
no server change, because the header's PRESENCE makes a request an agent's.

Control NC-87b (0xcc/reviews/040_P4_controls_2026-11-11.md): put "miLLM" back into one label ->
`test_old_actor_id_lists_with_the_new_label` and `test_no_action_label_names_the_old_brand` red.
"""

from __future__ import annotations

import pytest

from miserver.api.approval_gate import APPROVAL_ACTIONS
from tests.unit.approval_fixtures import harness, make_model, rows  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture
from tests.unit.legacy.exercises.test_frozen_wire_values import OLD_ACTOR

pytestmark = pytest.mark.asyncio

NEW_ACTOR = "agent:miworkbench-mcp"


async def _file(h, actor: str) -> dict:
    h.holder["row"] = row = await make_model(h.db)
    r = h.client.post(f"/api/models/{row.id}/unload", headers={"X-miLLM-Agent": actor})
    assert r.status_code == 202, r.text
    assert r.json()["data"]["status"] == "pending"
    (stored,) = await rows(h.db)
    assert stored.requested_by == actor, "the actor id is stored exactly as sent"
    listed = h.client.get("/api/approvals").json()["data"]["approvals"]
    assert len(listed) == 1, listed
    h.service.unload_model.assert_not_awaited()  # an agent's request waited: it did not run
    return listed[0]


async def test_old_actor_id_lists_with_the_new_label(harness):  # noqa: F811
    listed = await _file(harness, OLD_ACTOR)
    assert listed["requested_by"] == OLD_ACTOR
    assert listed["action"] == "model_unload"
    assert listed["action_label"] == "Unload a model in miServer"
    assert listed["action_label"] == APPROVAL_ACTIONS["model_unload"].label


async def test_new_actor_id_is_an_agent(harness):  # noqa: F811
    listed = await _file(harness, NEW_ACTOR)
    assert listed["requested_by"] == NEW_ACTOR
    assert listed["status"] == "pending"


def test_no_action_label_names_the_old_brand():
    assert len(APPROVAL_ACTIONS) >= 7
    for key, action in APPROVAL_ACTIONS.items():
        text = f"{action.label} {action.description}"
        assert "miLLM" not in text and "millm" not in text.lower(), (key, text)
        assert "miServer" in action.label, (key, action.label)
