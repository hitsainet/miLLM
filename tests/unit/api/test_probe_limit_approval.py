"""Approving, overriding and staleness of a model's probe limits (039 FR-39.8; TD39-15, TD39-31;
FTASKS 3.1, 3.3 – 3.5).

Through the live app (`create_app`) over a real SQLite database: the recommendation route, the
digest-checked approval and the operator override, the append-only history, and staleness by
FINGERPRINT (never by measurement id).

Controls (039 controls record):
* delete the recommendation route's decorator             → `test_the_recommendation_route_is_served` red
* approval writes other values than the recomputed ones    → `test_approval_writes_exactly_the_recommended_values` red
* approval skips the digest check                          → `test_a_stale_digest_is_refused_and_nothing_written` red
* staleness compares measurement ids, not fingerprints     → `test_a_newer_measurement_under_the_same_fingerprint_is_not_stale` red
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from sqlalchemy import select

from millm.db.models.probe_capacity import (
    ModelProbeLimits,
    ModelProbeLimitsHistory,
    ProbeCapacityMeasurement,
)
from millm.services import probe_limits
from tests.unit.api.test_model_probe_limits import LFM, _model, client  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

pytestmark = pytest.mark.asyncio
REPO = Path(__file__).resolve().parents[3]


async def _cpu_model(db):  # noqa: F811
    """A model whose recorded measurement was taken on the device it is resident on (the CPU
    here), so an approval rests on the resident fingerprint."""
    row = await _model(db, seed=False)
    async with db() as session:
        session.add(ProbeCapacityMeasurement(repo_id=row.repo_id, quantization="FP16",
                                             device="cpu", method="seeded", source="test", **LFM))
        await session.commit()
    return row


@pytest.fixture
def resident(monkeypatch):
    """Make the test's model row RESIDENT (its fingerprint is then the loaded one)."""
    from datetime import datetime

    from millm.ml.model_loader import LoadedModel, LoadedModelState
    from tests.unit.f25_fixtures import clear_loaded, word_model

    def make(row, dtype="float16"):
        LoadedModelState().set(LoadedModel(
            model_id=row.id, model_name=row.name, model=word_model(), tokenizer=None,
            loaded_at=datetime(2026, 10, 24), device="cpu", dtype=dtype))
        monkeypatch.setattr(probe_limits, "resident_model_id", lambda: row.id)

    yield make
    clear_loaded()


async def test_the_recommendation_route_is_served(client):  # noqa: F811
    paths = client.app.openapi()["paths"]
    assert "get" in paths["/api/models/{model_id}/probe-limits/recommendation"]


async def test_approval_writes_exactly_the_recommended_values(client, db, resident):  # noqa: F811
    row = client.holder["row"] = await _cpu_model(db)
    resident(row)
    rec = client.get(f"/api/models/{row.id}/probe-limits/recommendation").json()["data"]
    assert rec["values"]["max_behaviours"] == 1
    response = client.put(f"/api/models/{row.id}/probe-limits", json={
        "source": "recommended-approved", "recommendation_digest": rec["recommendation_digest"],
        "set_by": "admin-ui",
        # values sent beside an approval are IGNORED: the server writes the recomputed ones
        "max_behaviours": 9, "decode_budget": {"value": 5.0, "unit": "ms"},
    })
    assert response.status_code == 200, response.text
    async with db() as session:
        stored = await session.get(ModelProbeLimits, row.id)
        history = (await session.execute(select(ModelProbeLimitsHistory))).scalars().all()
    assert stored.max_behaviours == rec["values"]["max_behaviours"]
    assert (stored.decode_budget_value, stored.decode_budget_unit) == (
        rec["values"]["decode_budget_ms"], "ms")
    assert (stored.prefill_budget_value, stored.prefill_budget_unit) == (
        rec["values"]["prefill_budget_ms"], "ms")
    assert stored.source == "recommended-approved" and stored.set_by == "admin-ui"
    assert stored.recommendation["recommendation_digest"] == rec["recommendation_digest"]
    assert stored.fingerprint == {"repo_id": row.repo_id, "quantization": "FP16",
                                  "dtype": "float16", "devices": ["cpu"]}
    assert len(history) == 1 and history[0].source == "recommended-approved"
    effective = response.json()["data"]["effective"]
    assert effective["max_behaviours"] == 1
    assert effective["sources"]["max_behaviours"] == "recommended-approved"


async def test_a_stale_digest_is_refused_and_nothing_written(client, db):  # noqa: F811
    row = client.holder["row"] = await _cpu_model(db)
    response = client.put(f"/api/models/{row.id}/probe-limits", json={
        "source": "recommended-approved", "recommendation_digest": "0" * 64})
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "RECOMMENDATION_CHANGED"
    assert error["next"] == "millm_get_probe_limit_recommendation"
    assert error["details"]["values"]["max_behaviours"] == 1
    async with db() as session:
        assert await session.get(ModelProbeLimits, row.id) is None
        assert (await session.execute(select(ModelProbeLimitsHistory))).scalars().all() == []


async def test_an_override_records_operator(client, db):  # noqa: F811
    row = client.holder["row"] = await _model(db)
    response = client.put(f"/api/models/{row.id}/probe-limits",
                          json={"max_behaviours": 3, "reason": "two pilots and a canary"})
    assert response.status_code == 200, response.text
    effective = response.json()["data"]["effective"]
    assert effective["max_behaviours"] == 3
    assert effective["sources"]["max_behaviours"] == "operator"
    assert effective["stale"] is False


async def test_changing_the_dtype_makes_the_approval_stale_but_keeps_it_in_force(
        client, db, resident):  # noqa: F811
    row = client.holder["row"] = await _cpu_model(db)
    resident(row)
    rec = client.get(f"/api/models/{row.id}/probe-limits/recommendation").json()["data"]
    client.put(f"/api/models/{row.id}/probe-limits", json={
        "source": "recommended-approved", "recommendation_digest": rec["recommendation_digest"]})
    async with db() as session:
        assert (await probe_limits.effective_for_model(session, row.id)).stale is False
    resident(row, dtype="bfloat16")  # reloaded at another precision
    async with db() as session:
        limits = await probe_limits.effective_for_model(session, row.id)
    assert limits.stale is True
    assert limits.max_behaviours == 1, "a stale limit stays in force (F39-8)"


async def test_a_newer_measurement_under_the_same_fingerprint_is_not_stale(
        client, db, resident):  # noqa: F811
    row = client.holder["row"] = await _cpu_model(db)
    resident(row)
    rec = client.get(f"/api/models/{row.id}/probe-limits/recommendation").json()["data"]
    client.put(f"/api/models/{row.id}/probe-limits", json={
        "source": "recommended-approved", "recommendation_digest": rec["recommendation_digest"]})
    async with db() as session:
        session.add(ProbeCapacityMeasurement(
            repo_id=row.repo_id, quantization="FP16", device="cpu",
            method="re-measured", source="test", **LFM))
        await session.commit()
        limits = await probe_limits.effective_for_model(session, row.id)
    assert limits.stale is False
    assert limits.approval["newer_recommendation_available"] is True


def test_the_history_tables_are_never_updated_or_deleted():
    """Append-only by service rule (FTID Discrepancy 10): no module issues an UPDATE or DELETE
    against `ModelProbeLimitsHistory`; the one writer `session.add`s."""
    offenders = []
    for path in (REPO / "millm").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in ("update", "delete", "sa_update"):
                if any(isinstance(a, ast.Name) and a.id == "ModelProbeLimitsHistory"
                       for a in node.args):
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr in ("update", "delete") and \
                    isinstance(node.value, ast.Name) and node.value.id == "ModelProbeLimitsHistory":
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert offenders == []
    writers = sorted({
        p.name for p in (REPO / "millm").rglob("*.py")
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "ModelProbeLimitsHistory"
    })
    assert writers == ["probe_limits.py"], "one writer, and it only ever adds rows"
