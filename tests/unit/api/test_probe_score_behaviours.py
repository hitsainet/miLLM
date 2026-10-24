"""`POST /api/probes/score` by behaviour (039 FR-39.13, TD39-14; FTASKS 5.5) and the selector's four
callers (FR-39.1; FTASKS 5.7).

The live app, the tiny real Llama and per-run behaviour fixtures. Stateless: no event row, no probe
row and no armed state changes; every pre-039 key of the response is present (the pre-039 golden
`tests/fixtures/goldens/pre039/armed.json` pins the shape; `test_pre039_goldens.py` compares it).

Controls (039 controls record):
* the behaviour cap reverts to the global                     → `test_the_cap_is_the_resident_models_effective_limit` red
* the score cap reads a constant, not `PROBE_MAX_ARMED`        → 037's `test_probe_scoring` cap test red (re-mutation)
* one caller re-implements selection                          → `test_the_four_callers_use_the_one_selector` red
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

from millm.core.config import settings
from millm.db.models.probe import Probe, ProbeEvent
from millm.services import probe_limits, probe_selection
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap

pytestmark = pytest.mark.asyncio
INPUTS = [{"token_ids": [0, 2, 6, 7, 8], "prompt_tokens": 3},
          {"messages": [{"role": "user", "content": "w1 w2"}]}]


async def _import(h, key, n, **over):
    async with h.factory() as session:
        return [await bf.import_probe(session, bf.definition(
            f"{key}-{i}", key=key, layer=i % 2, scale=0.1 + 0.2 * i, **over)) for i in range(n)]


async def _counts(h):
    async with h.factory() as session:
        events = (await session.execute(select(func.count()).select_from(ProbeEvent))).scalar_one()
        armed = (await session.execute(
            select(func.count()).select_from(Probe).where(Probe.armed.is_(True)))).scalar_one()
    return events, armed


async def test_behaviours_score_every_probe_and_write_nothing():
    async with cap.Harness() as h:
        key = bf.random_key()
        rows = await _import(h, key, 3)
        before = await _counts(h)
        response = h.client.post("/api/probes/score", json={"behaviours": [key], "inputs": INPUTS})
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert {p["probe_id"] for p in data["probes"]} == {r.id for r in rows}
        for result in data["results"]:
            assert {v["probe_id"] for v in result["verdicts"]} == {r.id for r in rows}
            assert all(v["behaviour"]["key"] == key for v in result["verdicts"])
        assert await _counts(h) == before, "a scoring call wrote a row or armed a probe"


async def test_probe_ids_and_behaviours_together_are_refused():
    async with cap.Harness() as h:
        key = bf.random_key()
        rows = await _import(h, key, 1)
        response = h.client.post("/api/probes/score", json={
            "behaviours": [key], "probe_ids": [rows[0].id], "inputs": INPUTS})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_PROBE_SCORE_REQUEST"
        assert response.json()["error"]["next"] == "millm_score_probes"


async def test_the_cap_is_the_resident_models_effective_limit(monkeypatch):
    """TD39-14: per-model `max_behaviours` 2 with the global at 1 → a two-behaviour request
    scores (red on pre-039 code, which had no behaviours and capped at the global)."""
    monkeypatch.setattr(settings, "PROBE_MAX_ARMED_BEHAVIOURS", 1)
    async with cap.Harness() as h:
        a, b = bf.random_key(), bf.random_key()
        await _import(h, a, 1)
        await _import(h, b, 1)
        refused = h.client.post("/api/probes/score", json={"behaviours": [a, b],
                                                           "inputs": INPUTS})
        assert refused.status_code == 400
        assert refused.json()["error"]["details"]["max_behaviours"] == 1

        async def per_model(_session, _model_id):
            return dataclasses.replace(probe_limits.global_limits(), max_behaviours=2)

        monkeypatch.setattr(probe_limits, "effective_for_model", per_model)
        response = h.client.post("/api/probes/score", json={"behaviours": [a, b],
                                                            "inputs": INPUTS})
        assert response.status_code == 200, response.text


async def test_a_behaviour_with_nothing_armed_still_scores():
    async with cap.Harness() as h:
        key, armed = bf.random_key(), bf.random_key()
        await _import(h, armed, 1)
        h.client.post(f"/api/probes/behaviours/{armed}/arm", json={})
        idle = await _import(h, key, 2)
        response = h.client.post("/api/probes/score", json={"behaviours": [key], "inputs": INPUTS})
        assert response.status_code == 200, response.text
        assert {v["probe_id"] for v in response.json()["data"]["results"][0]["verdicts"]} == \
            {r.id for r in idle}


async def test_the_four_callers_use_the_one_selector():
    """FR-39.1: the arm route, its dry run, the capacity preview and stateless scoring by
    behaviour all call `probe_selection.select_behaviour` — by payload and call count."""
    async with cap.Harness() as h:
        key = bf.random_key()
        await _import(h, key, 2)
        real = probe_selection.select_behaviour
        calls: list[tuple[str, object]] = []

        async def spy(repository, k, **kw):
            calls.append((k, kw.get("version", probe_selection.UNSET)))
            return await real(repository, k, **kw)

        with patch.object(probe_selection, "select_behaviour", spy):
            assert h.client.post(f"/api/probes/behaviours/{key}/arm",
                                 json={"dry_run": True}).status_code == 200
            assert h.client.post(f"/api/probes/behaviours/{key}/arm",
                                 json={"version": 1}).status_code == 200
            assert h.client.get("/api/models/1/probe-capacity",
                                params={"behaviour": key}).status_code == 200
            assert h.client.post("/api/probes/score", json={
                "behaviours": [key], "inputs": INPUTS[:1]}).status_code == 200
        assert [c[0] for c in calls] == [key] * 4
        assert calls[0][1] is probe_selection.UNSET and calls[1][1] == 1
