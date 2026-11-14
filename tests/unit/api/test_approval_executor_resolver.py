"""P5 (041 FR-041.11, TID §3.5; DR-35 permanent): an approval filed under an earlier package name runs.

`approval_requests.executor_key` stores `<module>.<qualname>` of the gated route, so every row filed
before P8 names a `millm.` module. After the rename the exact lookup misses; `resolve_executor` maps a
key that starts with a prefix in `LEGACY_PREFIXES` onto today's package, and nothing else. The stored row
is never rewritten.

`[synthetic]` (C4) uses an invented earlier name, `oldpkg`, through a monkeypatched `LEGACY_PREFIXES`.
`[real]` (P8's rename commit) stores a real pre-P8 key under `millm.` and runs it against today's
`miserver.` executor with the shipped `LEGACY_PREFIXES`.
"""

from __future__ import annotations

import pytest
from sqlalchemy import update

from miserver.api import approval_gate
from miserver.db.models.approval import ApprovalRequest
from miserver.services import approval_service
from tests.unit.approval_fixtures import AGENT, harness, make_model, rows  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

CURRENT = approval_gate.__name__.split(".", 1)[0]
UNLOAD = f"{CURRENT}.api.routes.management.models.unload_model"
#: A key as a pre-P8 server stored it (DR-35). It must stay `millm.` for good.
REAL_LEGACY_KEY = "millm.api.routes.management.models.unload_model"  # p8: keep


async def _file_unload(h):
    h.holder["row"] = row = await make_model(h.db)
    r = h.client.post(f"/api/models/{row.id}/unload", headers=AGENT)
    assert r.status_code == 202, r.text
    (stored,) = await rows(h.db)
    assert stored.executor_key == UNLOAD, stored.executor_key
    return row, stored


async def _store_key(h, approval_id: str, key: str) -> None:
    async with h.db() as session:  # the row as an earlier server stored it
        await session.execute(update(ApprovalRequest).where(ApprovalRequest.id == approval_id)
                              .values(executor_key=key))
        await session.commit()


@pytest.mark.parametrize("legacy", ["synthetic", "real"])
async def test_legacy_key_resolves_and_runs_once(harness, monkeypatch, legacy):  # noqa: F811
    h = harness
    if legacy == "synthetic":
        monkeypatch.setattr(approval_gate, "LEGACY_PREFIXES", ("oldpkg.",))
        old_key = "oldpkg." + UNLOAD.split(".", 1)[1]
    else:
        assert CURRENT == "miserver" and REAL_LEGACY_KEY.split(".", 1)[1] == UNLOAD.split(".", 1)[1]
        assert REAL_LEGACY_KEY not in approval_gate.EXECUTORS, "the exact lookup must miss"
        old_key = REAL_LEGACY_KEY
    row, stored = await _file_unload(h)
    await _store_key(h, stored.id, old_key)
    r = h.client.post(f"/api/approvals/{stored.id}/approve")
    assert r.status_code == 202, r.text
    await approval_service.wait_all()
    (after,) = await rows(h.db)
    assert after.status == "executed", (after.status, after.error)
    assert after.executor_key == old_key, "the stored key is read, never rewritten"
    h.service.unload_model.assert_awaited_once_with(row.id, lease_id=None)


async def test_unknown_first_segment_still_raises(harness, monkeypatch):  # noqa: F811
    h = harness
    monkeypatch.setattr(approval_gate, "LEGACY_PREFIXES", ("oldpkg.",))
    row, stored = await _file_unload(h)
    await _store_key(h, stored.id, "otherpkg." + UNLOAD.split(".", 1)[1])
    r = h.client.post(f"/api/approvals/{stored.id}/approve")
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "approval_executor_missing", r.text
    (after,) = await rows(h.db)
    assert after.status == "pending"
    h.service.unload_model.assert_not_awaited()
    # And the resolver alone: an unlisted head, a listed head with no such route, no dot at all.
    assert approval_gate.resolve_executor("otherpkg." + UNLOAD.split(".", 1)[1]) is None
    assert approval_gate.resolve_executor("oldpkg.api.routes.management.models.nope") is None
    assert approval_gate.resolve_executor("oldpkg") is None


def test_current_key_unchanged(monkeypatch):
    assert UNLOAD in approval_gate.EXECUTORS, sorted(approval_gate.EXECUTORS)[:5]
    assert approval_gate.resolve_executor(UNLOAD) is approval_gate.EXECUTORS[UNLOAD]
    # Today's package in LEGACY_PREFIXES must not loop or re-map a current key.
    monkeypatch.setattr(approval_gate, "LEGACY_PREFIXES", (CURRENT + ".",))
    assert approval_gate.resolve_executor(UNLOAD) is approval_gate.EXECUTORS[UNLOAD]
    assert approval_gate.resolve_executor(f"{CURRENT}.api.routes.management.models.nope") is None
