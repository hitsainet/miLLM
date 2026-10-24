"""The behaviour routes through the LIVE app (039 FR-39.2 – FR-39.6, FR-39.9, FR-39.15; FTASKS 2.10 –
2.13, 3.7, 5.2, 5.4).

The real `create_app()`, a real SQLite database, the tiny real Llama and probe definitions whose
test vectors are computed on it per run (`tests/support/behaviour_factory.py`): parity runs for real.
Wiring is asserted on the live `app.openapi()`, and shadowing by a live request.

Controls (039 controls record):
* delete the arm route's decorator                 → `test_the_behaviour_routes_are_served` red
* move `GET /behaviours` below `GET /{probe_id}`    → `test_behaviours_is_not_shadowed_by_a_probe_id` red
* the dry run falls through to the arm              → `test_a_dry_run_arms_nothing` red
* `disarm` clears `last_cost`                        → `test_the_preview_reads_measured_costs_after_disarm` red
"""

from __future__ import annotations

import pytest

from millm.core.config import settings
from millm.services.probe_runtime import ProbeRuntimeState
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap

pytestmark = pytest.mark.asyncio

ROUTES = {
    "/api/probes/behaviours": "get",
    "/api/probes/behaviours/{key}/arm": "post",
    "/api/probes/behaviours/{key}/disarm": "post",
    "/api/models/{model_id}/probe-limits/recommendation": "get",
}


async def _import(h, key, n, *, layers=(0, 1, 1), thresholds=(0.0, 0.5, -0.5), version=1,
                  start=0, **over):
    rows = []
    async with h.factory() as session:
        for i in range(start, start + n):
            rows.append(await bf.import_probe(session, bf.definition(
                f"{key}-p{i}", key=key, version=version, layer=layers[i % len(layers)],
                threshold=thresholds[i % len(thresholds)], scale=0.2 + 0.07 * i, **over)))
    return rows


async def test_the_behaviour_routes_are_served():
    from millm.main import create_app

    paths = create_app().openapi()["paths"]
    for path, method in ROUTES.items():
        assert method in paths.get(path, {}), f"{method.upper()} {path} is not served"


async def test_behaviours_is_not_shadowed_by_a_probe_id():
    async with cap.Harness() as h:
        response = h.client.get("/api/probes/behaviours")
        assert response.status_code == 200, response.text
        assert response.json()["data"] == [] and response.json()["meta"]["next"] is None


async def test_a_three_probe_behaviour_arms_through_the_route_and_another_stays_off():
    async with cap.Harness() as h:
        key, other = bf.random_key(), bf.random_key()
        rows = await _import(h, key, 3)
        stranger = (await _import(h, other, 1))[0]
        response = h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert sorted(p["id"] for p in data["armed"]) == sorted(r.id for r in rows)
        assert data["set"]["complete"] is True and data["set"]["armed_members"] == 3
        state = ProbeRuntimeState()
        for row in rows:
            assert state.get(row.id).threshold == row.threshold
        assert state.get(stranger.id) is None
        assert {p.layer for p in state.armed()} == {0, 1}


async def test_a_refusal_names_every_probe_and_next():
    async with cap.Harness() as h:
        key = bf.random_key()
        await _import(h, key, 2)
        await _import(h, key, 1, start=2, break_parity=True)
        response = h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "PROBE_PARITY_FAILED"
        outcomes = [p["outcome"] for p in error["details"]["probes"]]
        assert sorted(outcomes) == ["passed", "passed", "refused"]
        assert ProbeRuntimeState().armed() == []


async def test_mixed_versions_are_refused_with_next_and_version_narrows():
    async with cap.Harness() as h:
        key = bf.random_key()
        await _import(h, key, 1, version=1)
        await _import(h, key, 2, version=2, start=1)
        refused = h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "BEHAVIOUR_VERSION_MIXED"
        assert refused.json()["error"]["next"] == "millm_arm_behaviour"
        armed = h.client.post(f"/api/probes/behaviours/{key}/arm", json={"version": 2})
        assert armed.status_code == 200, armed.text
        assert len(armed.json()["data"]["armed"]) == 2


async def test_a_dry_run_arms_nothing():
    async with cap.Harness() as h:
        key = bf.random_key()
        rows = await _import(h, key, 2)
        await _import(h, key, 1, start=2, rung=1)
        response = h.client.post(f"/api/probes/behaviours/{key}/arm", json={"dry_run": True})
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["dry_run"] is True and data["armed"] == []
        assert data["would_refuse"]["code"] == "UNVALIDATED_PROBE"
        assert ProbeRuntimeState().armed() == []
        listed = {p["id"]: p for p in h.client.get("/api/probes").json()["data"]}
        assert not any(listed[r.id]["armed"] for r in rows)


async def test_disarm_route_and_the_listing_partial_state():
    async with cap.Harness() as h:
        key = bf.random_key()
        rows = await _import(h, key, 3)
        h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
        h.client.post(f"/api/probes/{rows[0].id}/disarm")
        listed = h.client.get("/api/probes/behaviours", params={"key": key}).json()["data"]
        assert listed[0]["set"]["text"] == "partial: 2 of 3 armed"
        assert listed[0]["set"]["missing_probe_ids"] == [rows[0].id]
        response = h.client.post(f"/api/probes/behaviours/{key}/disarm", json={})
        assert response.status_code == 200, response.text
        assert sorted(response.json()["data"]["disarmed"]) == sorted(r.id for r in rows[1:])
        assert ProbeRuntimeState().armed() == []


async def test_status_lists_behaviours_and_the_catalogue_as_counts():
    async with cap.Harness() as h:
        key, idle = bf.random_key(), bf.random_key()
        await _import(h, key, 2)
        await _import(h, idle, 3)
        h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
        status = h.client.get("/api/probes/status").json()["data"]
        assert [b["key"] for b in status["behaviours"]] == [key]
        assert status["catalogue"] == {"imported": 5, "behaviours": 2}
        assert status["limits"]["max_behaviours"] == settings.PROBE_MAX_ARMED_BEHAVIOURS
        assert status["limits"]["max_behaviours_source"] == "global"
        assert len(status["armed"]) == 2


async def test_the_probe_list_is_paged_by_cursor(monkeypatch):
    async with cap.Harness() as h:
        keys = sorted(bf.random_key() for _ in range(3))
        for key in keys:
            await _import(h, key, 2)
        async with h.factory() as session:
            await bf.import_probe(session, bf.definition("keyless", key=None))
        seen, cursor, pages = [], None, 0
        while True:
            params = {"limit": 2, **({"cursor": cursor} if cursor else {})}
            body = h.client.get("/api/probes", params=params).json()
            seen.extend((p["behaviour"] or {}).get("key") for p in body["data"])
            pages += 1
            cursor = body["meta"]["next"]
            if cursor is None:
                break
        assert pages == 4 and seen == [k for k in keys for _ in range(2)] + [None]
        bad = h.client.get("/api/probes", params={"cursor": "not-a-cursor"})
        assert bad.status_code == 400 and bad.json()["error"]["code"] == "INVALID_CURSOR"


async def test_the_preview_reads_measured_costs_after_disarm():
    """FR-39.9 / TD39-11: a dry run measures every probe (`last_cost`); a disarm never clears
    it, so the preview reads `measured`, not `estimated`, for the resident fingerprint."""
    async with cap.Harness() as h:
        key = bf.random_key()
        rows = await _import(h, key, 2)
        h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
        h.client.post(f"/api/probes/behaviours/{key}/disarm", json={})
        data = h.client.get("/api/models/1/probe-capacity", params={"behaviour": key}).json()
        preview = data["data"]["preview"]
        assert {p["probe_id"]: p["source"] for p in preview["probes"]} == {
            r.id: "measured" for r in rows}
        # The SAME projection the arm gate computes: the registry (empty now) plus each member's
        # cost through `ProbeRuntimeState.projected_cost`, i.e. `probe_cost.project`.
        from types import SimpleNamespace

        from millm.services.probe_cost import cost_from_dict

        gate = ProbeRuntimeState().projected_cost(extra=[
            SimpleNamespace(probe_id=p["probe_id"], cost=cost_from_dict(p["cost"]))
            for p in preview["probes"]])
        assert preview["projected"] == pytest.approx(gate)
