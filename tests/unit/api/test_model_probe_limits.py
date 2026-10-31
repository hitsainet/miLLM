"""Per-model probe limits (operator 2026-10-21) through the live app and the real admission path.

"I want to know the actual limits, what drives them, and to be able to adjust on a per model basis."

Controls (recorded in `0xcc/reviews/036_037_controls_2026-10-21.md`):
* the arm route stops passing `limits=`                    → `test_the_arm_route_passes_the_resident_models_limits` red
* `effective()` ignores the configured `max_armed`         → `test_a_model_can_carry_its_own_count_backstop` red
* a `pct` budget not converted through the step time        → `test_a_percentage_budget_is_of_the_measured_step_time` red
* the capacity router include removed                       → `test_the_capacity_routes_are_served` red
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from millm.core.config import settings
from millm.core.errors import ProbeLimitError
from millm.db.models.probe_capacity import ProbeCapacityMeasurement
from millm.services import probe_limits
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

LFM = dict(decode_ms_per_token=6.69, prefill_ms_per_1k=None,
           dense={"decode_own_ms": 0.0253, "decode_fixed_ms": 0.0496,
                  "prefill_own_ms_per_1k": 0.0569, "prefill_fixed_ms_per_1k": 0.0411},
           sparse={"decode_own_ms": 0.1149, "decode_fixed_ms": 0.0752,
                   "prefill_own_ms_per_1k": 0.1465, "prefill_fixed_ms_per_1k": 0.0143, "k": 128})


async def _model(db, *, repo="LiquidAI/LFM2.5-1.2B-Instruct", seed=True):  # noqa: F811
    from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType

    async with db() as session:
        row = Model(name=repo.split("/")[1], source=ModelSource("huggingface"), repo_id=repo,
                    quantization=QuantizationType("FP16"), cache_path="/x", params="1B",
                    architecture="text-generation", status=ModelStatus.READY)
        session.add(row)
        await session.commit()
        if seed:
            session.add(ProbeCapacityMeasurement(
                repo_id=repo, quantization="FP16", device="cuda:GPU-247aa582", method="seeded",
                source="record", **LFM))
            await session.commit()
        return row


@pytest.fixture
def client(db, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    from millm.api.dependencies import get_db, get_model_service
    from millm.main import create_app

    app = create_app()

    async def _db():
        async with db() as session:
            yield session

    holder: dict = {}
    service = MagicMock()
    service.get_model = AsyncMock(side_effect=lambda model_id: holder["row"])
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_model_service] = lambda: service
    monkeypatch.setattr(probe_limits, "resident_model_id", lambda: None)
    c = TestClient(app)
    c.holder = holder
    return c


def test_the_capacity_routes_are_served(client):
    paths = client.app.openapi()["paths"]
    assert "get" in paths["/api/models/{model_id}/probe-capacity"]
    assert "put" in paths["/api/models/{model_id}/probe-limits"]
    assert "post" in paths["/api/models/{model_id}/probe-capacity/measure"]


async def test_capacity_shows_what_drives_the_limits(client, db):  # noqa: F811
    client.holder["row"] = await _model(db)
    data = client.get(f"/api/models/{client.holder['row'].id}/probe-capacity").json()["data"]
    assert data["measured"]["decode_ms_per_token"] == 6.69
    assert data["effective"]["max_armed"] == settings.PROBE_MAX_ARMED
    sources = data["effective"]["sources"]
    # D83: decode is 5% of the measured 6.69 ms; this seed row measured no prefill step, so that
    # half is the absolute fallback and says how to get 5% instead.
    assert sources["max_armed"] == "global" and sources["decode"] == "global (5% of 6.69 ms)"
    assert sources["prefill"].startswith("global fallback 0.5 ms") and \
        "/probe-capacity/measure" in sources["prefill"]
    assert "not enforced" in sources["max_behaviours"]
    dense = data["implied"]["dense"]
    # decode: (0.3345 − 0.0496)/0.0253 = 11.3 → 11; with 0.9 headroom (0.3011) → 9.
    assert dense["decode"] == {"fits_budget": 11, "fits_with_headroom": 9}
    # prefill: (0.5 − 0.0411)/0.0569 = 8.06 → 8; with headroom → 7.
    assert dense["prefill"] == {"fits_budget": 8, "fits_with_headroom": 7}
    assert dense["binding"] == "prefill" and dense["n"] == min(7, settings.PROBE_MAX_ARMED)
    assert data["implied"]["sparse"]["decode"]["fits_with_headroom"] == 1


async def test_a_percentage_budget_is_of_the_measured_step_time(client, db):  # noqa: F811
    client.holder["row"] = row = await _model(db)
    r = client.put(f"/api/models/{row.id}/probe-limits", json={
        "max_armed": 12, "decode_budget": {"value": 2, "unit": "pct"},
        "prefill_budget": {"value": 10, "unit": "pct"}, "reason": "lfm is small"})
    assert r.status_code == 200, r.text
    eff = r.json()["data"]["effective"]
    assert eff["max_armed"] == 12
    assert eff["decode_ms_per_pass"] == pytest.approx(0.02 * 6.69)
    assert eff["sources"]["decode"].startswith("model (2% of 6.69 ms)")
    # No prefill step time was measured: the percentage cannot be converted, so the global stands
    # — and it says why, never a guessed number.
    assert eff["prefill_ms_per_1k"] == settings.PROBE_MAX_PREFILL_MS_PER_1K
    assert "not measured" in eff["sources"]["prefill"]
    configured = r.json()["data"]["configured"]
    assert configured["decode_budget"] == {"value": 2.0, "unit": "pct"}


async def test_ms_budgets_and_clearing_back_to_global(client, db):  # noqa: F811
    client.holder["row"] = row = await _model(db)
    client.put(f"/api/models/{row.id}/probe-limits",
               json={"decode_budget": {"value": 0.5, "unit": "ms"}})
    eff = client.get(f"/api/models/{row.id}/probe-capacity").json()["data"]["effective"]
    assert eff["decode_ms_per_pass"] == 0.5 and eff["sources"]["decode"] == "model (ms)"
    client.put(f"/api/models/{row.id}/probe-limits", json={})
    eff = client.get(f"/api/models/{row.id}/probe-capacity").json()["data"]["effective"]
    # Cleared: the global, which for a measured model is 5% of its own time (D83).
    assert eff["decode_ms_per_pass"] == pytest.approx(
        settings.PROBE_SLOWDOWN_BUDGET_PCT / 100 * 6.69)


@pytest.mark.parametrize("body", [
    {"max_armed": 0}, {"max_armed": 65}, {"decode_budget": {"value": 1, "unit": "s"}},
    {"decode_budget": {"value": 200, "unit": "pct"}}, {"decode_budget": {"value": 20, "unit": "ms"}},
    {"maxarmed": 3},
])
async def test_invalid_limits_are_refused(client, db, body):  # noqa: F811
    client.holder["row"] = row = await _model(db)
    assert client.put(f"/api/models/{row.id}/probe-limits", json=body).status_code == 422


async def test_the_model_read_carries_its_capacity(client, db):  # noqa: F811
    """`GET /api/models/{id}` — what miStudio's `millm_get_model` MCP tool reads."""
    client.holder["row"] = row = await _model(db)
    from millm.api.dependencies import get_model_service

    service = client.app.dependency_overrides[get_model_service]()
    service.get_download_progress = MagicMock(return_value=None)
    service.get_loaded_model_info = MagicMock(return_value=None)
    data = client.get(f"/api/models/{row.id}").json()["data"]
    # D83: 5% of the measured 6.69 ms = 0.3345 ms → (0.3345 − 0.0496) / 0.0253 = 11.3 → 11.
    assert data["probe_capacity"]["implied"]["dense"]["decode"]["fits_budget"] == 11


async def test_a_model_can_carry_its_own_count_backstop(db):  # noqa: F811
    """Admission (the real `ProbeArmingService`) uses the RESIDENT model's effective limits."""
    from tests.unit.services.test_probe_arming import TinyModel, definition, loaded
    from millm.db.repositories.probe_repository import ProbeRepository
    from millm.services.probe_arming import ProbeArmingService
    from millm.services.probe_runtime import ProbeRuntimeState
    from millm.services.probe_service import ProbeService

    ProbeRuntimeState.reset_for_tests()
    row = await _model(db)
    async with db() as session:
        await probe_limits.set_limits(session, row.id, {"max_armed": 1})
        limits = await probe_limits.effective_for_model(session, row.id)
        assert limits.max_armed == 1 and limits.sources["max_armed"] == "model"
        repo = ProbeRepository(session)
        service, arming = ProbeService(repo), ProbeArmingService(repo, ProbeRuntimeState())

        async def run_inline(fn):
            return fn()

        def forward(input_ids, context):
            context.observe(11, torch.full((1, input_ids.shape[1], 8), 0.5))

        for i, name in enumerate(("a", "b")):
            probe = await service.import_definition(definition(name=name), raw_bytes=1,
                                                    on_conflict="fail", origin="file")
            if i == 0:
                await arming.arm(probe, executor=run_inline, model=TinyModel(), loaded=loaded(),
                                 forward=forward, limits=limits)
            else:
                with pytest.raises(ProbeLimitError) as refused:
                    await arming.arm(probe, executor=run_inline, model=TinyModel(),
                                     loaded=loaded(), forward=forward, limits=limits)
                # 039: plus how many are armed and how many this call adds.
                assert refused.value.details == {"max_armed": 1, "source": "model",
                                                 "armed": 1, "size": 1}
    ProbeRuntimeState.reset_for_tests()


def test_the_arm_route_passes_the_resident_models_limits(monkeypatch):
    """WIRING: the route computes the resident model's limits and hands them to `arm` — payload
    and call count, through the real route."""
    from tests.unit.api.test_probe_arm_route import _client, _probe_row
    import millm.api.routes.management.probes as routes
    from millm.services.probe_identity import LoadedIdentity

    identity = LoadedIdentity(hf_id="m", d_model=8, n_layers=16, chat_template="x")
    monkeypatch.setattr(routes, "loaded_identity",
                        AsyncMock(return_value=(identity, MagicMock(), MagicMock())))
    monkeypatch.setattr(routes, "build_parity_forward", lambda *_a, **_k: (lambda *a, **k: None))
    monkeypatch.setattr(routes, "build_probe_encoder", AsyncMock(return_value=None))
    sentinel = probe_limits.EffectiveLimits(max_armed=3, decode_ms_per_pass=0.4,
                                            prefill_ms_per_1k=0.6, headroom=0.9,
                                            sources={"max_armed": "model"})
    asked = []

    async def eff(session, model_id):
        asked.append(model_id)
        return sentinel

    monkeypatch.setattr(probe_limits, "effective_for_model", eff)
    monkeypatch.setattr(probe_limits, "resident_model_id", lambda: 42)
    client, _repo, arming, *_ = _client(probe=_probe_row())
    assert client.post("/api/probes/pr_1/arm", json={}).status_code == 200
    assert asked == [42]
    assert arming.arm.await_count == 1
    assert arming.arm.await_args.kwargs["limits"] is sentinel


async def test_the_resident_model_shows_the_armed_set_against_each_budget(client, db, monkeypatch):  # noqa: F811
    from millm.ml.probe_head import ProbeHead
    from millm.services.probe_cost import ProbeCost
    from millm.services.probe_runtime import ArmedProbe, ProbeRuntimeState

    client.holder["row"] = row = await _model(db)
    monkeypatch.setattr(probe_limits, "resident_model_id", lambda: row.id)
    ProbeRuntimeState.reset_for_tests()
    state = ProbeRuntimeState()
    cost = ProbeCost(decode_ms=0.05, prefill_ms_per_1k=0.1, layer_fixed_ms=0.04,
                     layer_fixed_prefill_ms_per_1k=0.08, layer=3, device="cpu", k=1, measured_at="t")
    state._armed["pr_x"] = ArmedProbe(
        probe_id="pr_x", name="x", head=ProbeHead(weight=torch.ones(4), bias=0.0, layer=3),
        rule="mean", scope="all", layer=3, rung=2, rung_language="x", cost=cost)
    try:
        armed = client.get(f"/api/models/{row.id}/probe-capacity").json()["data"]["armed"]
        assert armed["count"] == 1
        assert armed["projected"] == {"decode": pytest.approx(0.09), "prefill": pytest.approx(0.18)}
        # D83: the decode budget is 5% of this model's measured 6.69 ms.
        assert armed["share"]["decode"] == pytest.approx(0.09 / (0.05 * 6.69 * 0.9))
        assert armed["share"]["count"] == pytest.approx(1 / settings.PROBE_MAX_ARMED)
    finally:
        ProbeRuntimeState.reset_for_tests()
