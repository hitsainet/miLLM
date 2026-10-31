"""Every S1 refusal names the MCP tool that resolves it (039 FR-39.30, FTDD §5.4; FTASKS 6.1 – 6.3).

Each code is reached ONCE THROUGH A ROUTE of the live app and its `error.next` compared with the
FTDD §5.4 table (`NEXT`), which is also what miStudio's live-registry check reads: every tool named
here must exist in miStudio's registry (039 S1 co-release, FTASKS 9.2). `/v1` errors keep the
OpenAI envelope and carry no `next` (FTID Discrepancy 3).

Control: drop `next_tool=` serialisation in the management handler → every case red.
"""

from __future__ import annotations

import pytest

from millm.core import errors
from millm.core.config import settings
from millm.services import probe_cost, probe_limits
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap

pytestmark = pytest.mark.asyncio

#: FTDD §5.4: code → the tool that resolves it. Exported for miStudio's registry check.
NEXT = {
    "PROBE_LIMIT": "millm_disarm_behaviour",
    "PROBE_BUDGET": "millm_get_probe_limit_recommendation",
    "BEHAVIOUR_LIMIT": "millm_get_probe_limit_recommendation",
    "BEHAVIOUR_VERSION_MIXED": "millm_arm_behaviour",
    "INVALID_PROBE_SELECTION": "millm_list_probes",
    "MODEL_CHANGED": "millm_arm_behaviour",
    "CATALOGUE_LIMIT": "millm_delete_probe",
    "RECOMMENDATION_CHANGED": "millm_get_probe_limit_recommendation",
    "INVALID_CURSOR": "millm_list_probes",
    "INVALID_PROBE_SCORE_REQUEST": "millm_score_probes",
}


def test_the_table_is_the_classes_own():
    """The table above is the classes' `next_tool`, not a copy that can drift."""
    by_code = {cls.code: cls.next_tool for cls in _subclasses(errors.MiLLMError)
               if cls.code in NEXT}
    assert by_code == NEXT


def _subclasses(cls):
    out = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(_subclasses(sub))
    return out


async def _import(h, key, n, **over):
    async with h.factory() as session:
        return [await bf.import_probe(session, bf.definition(
            f"{key}-{i}", key=key, layer=i % 2, scale=0.1 + 0.2 * i, **over)) for i in range(n)]


def _refusal(response, code):
    assert response.status_code in (400, 409), response.text
    error = response.json()["error"]
    assert error["code"] == code, error
    assert error["next"] == NEXT[code]
    return error


async def test_behaviour_limit_is_no_longer_raised():
    """ENH-006 D83: the code stays in the table (an older client may still read it), but no
    behaviour count refuses: a second behaviour arms."""
    async with cap.Harness() as h:
        a, b = bf.random_key(), bf.random_key()
        await _import(h, a, 1)
        await _import(h, b, 1)
        assert h.client.post(f"/api/probes/behaviours/{a}/arm", json={}).status_code == 200
        assert h.client.post(f"/api/probes/behaviours/{b}/arm", json={}).status_code == 200


async def test_probe_limit(monkeypatch):
    monkeypatch.setattr(settings, "PROBE_MAX_ARMED", 2)
    async with cap.Harness() as h:
        key = bf.random_key()
        await _import(h, key, 3)
        _refusal(h.client.post(f"/api/probes/behaviours/{key}/arm", json={}), "PROBE_LIMIT")


async def test_probe_budget(monkeypatch):
    monkeypatch.setattr(settings, "PROBE_MAX_OVERHEAD_MS_PER_PASS", 0.01)
    monkeypatch.setattr(probe_cost, "_timer", lambda fn, k, device: (fn(), 1.0)[1])
    async with cap.Harness() as h:
        key = bf.random_key()
        await _import(h, key, 2)
        error = _refusal(h.client.post(f"/api/probes/behaviours/{key}/arm", json={}),
                         "PROBE_BUDGET")
        assert len(error["details"]["probes"]) == 2


async def test_version_mixed_and_invalid_selection():
    async with cap.Harness() as h:
        key, other = bf.random_key(), bf.random_key()
        await _import(h, key, 1, version=1)
        async with h.factory() as session:
            await bf.import_probe(session, bf.definition(f"{key}-v2", key=key, version=2))
        foreign = (await _import(h, other, 1))[0]
        _refusal(h.client.post(f"/api/probes/behaviours/{key}/arm", json={}),
                 "BEHAVIOUR_VERSION_MIXED")
        _refusal(h.client.post(f"/api/probes/behaviours/{key}/arm",
                               json={"probe_ids": [foreign.id]}), "INVALID_PROBE_SELECTION")


async def test_catalogue_limit_on_import(monkeypatch):
    monkeypatch.setattr(settings, "PROBE_MAX_IMPORTED", 1)
    async with cap.Harness() as h:
        key = bf.random_key()
        assert h.client.post("/api/probes/import",
                             json=bf.definition("one", key=key)).status_code == 200
        error = _refusal(h.client.post("/api/probes/import",
                                       json=bf.definition("two", key=key)), "CATALOGUE_LIMIT")
        assert error["details"] == {**error["details"], "limit": "PROBE_MAX_IMPORTED", "value": 1}


async def test_model_changed(monkeypatch):
    """The model moves between the gates and the commit (TD39-9)."""
    from datetime import datetime

    from millm.ml.model_loader import LoadedModelState
    from millm.services.probe_arming import ProbeArmingService

    async with cap.Harness() as h:
        key = bf.random_key()
        await _import(h, key, 1)
        real = ProbeArmingService.check

        async def moving(self, probe, **kw):
            plan = await real(self, probe, **kw)
            LoadedModelState().current.loaded_at = datetime(2026, 10, 24, 12, 0)
            return plan

        monkeypatch.setattr(ProbeArmingService, "check", moving)
        _refusal(h.client.post(f"/api/probes/behaviours/{key}/arm", json={}), "MODEL_CHANGED")


async def test_recommendation_changed_cursor_and_score_request(monkeypatch):
    async with cap.Harness() as h:
        _refusal(h.client.get("/api/probes", params={"cursor": "%%%"}), "INVALID_CURSOR")
        _refusal(h.client.post("/api/probes/score", json={
            "behaviours": ["x-y"], "probe_ids": ["pr_000000000000"],
            "inputs": [{"token_ids": [2, 6]}]}), "INVALID_PROBE_SCORE_REQUEST")
        from millm.db.models.probe_capacity import ProbeCapacityMeasurement

        async with h.factory() as session:
            session.add(ProbeCapacityMeasurement(
                repo_id="tiny/llama", quantization="FP16", device="cpu", method="seeded",
                source="test", decode_ms_per_token=5.0, prefill_ms_per_1k=None,
                dense={"decode_own_ms": 0.02, "decode_fixed_ms": 0.04,
                       "prefill_own_ms_per_1k": 0.05, "prefill_fixed_ms_per_1k": 0.04}))
            await session.commit()
        monkeypatch.setattr(probe_limits, "latest_measurement",
                            _latest_by_repo("tiny/llama", h))
        _refusal(h.client.put("/api/models/1/probe-limits", json={
            "source": "recommended-approved", "recommendation_digest": "0" * 64}),
            "RECOMMENDATION_CHANGED")


def _latest_by_repo(repo, h):
    async def latest(db, _model_row):
        from sqlalchemy import select

        from millm.db.models.probe_capacity import ProbeCapacityMeasurement as M

        return (await db.execute(select(M).where(M.repo_id == repo))).scalars().first()
    return latest


def test_v1_errors_never_carry_next():
    """The OpenAI envelope stays exact (FR-29.6.4, I-5)."""
    from millm.api.routes.openai.errors import create_openai_error

    body = create_openai_error(message="m", error_type="invalid_request_error",
                               code="behaviour_limit", status_code=409)
    assert "next" not in body.body.decode()
