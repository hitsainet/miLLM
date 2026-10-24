"""`GET /api/probes/status` reports the RESIDENT model's effective limits (operator 2026-10-21)."""

import pytest

from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services import probe_limits
from millm.services.probe_event_service import ProbeEventService
from tests.unit.api.test_model_probe_limits import _model
from tests.unit.f28_fixtures import db  # noqa: F401


@pytest.mark.asyncio
async def test_status_uses_the_resident_models_limits(db, monkeypatch):  # noqa: F811
    row = await _model(db)
    monkeypatch.setattr(probe_limits, "resident_model_id", lambda: row.id)
    async with db() as session:
        await probe_limits.set_limits(session, row.id, {
            "max_armed": 9, "decode_budget": {"value": 0.4, "unit": "ms"}})
        status = await ProbeEventService(ProbeRepository(session),
                                         ProbeEventRepository(session)).status()
    assert status["max_armed"] == 9
    assert status["budget"]["decode_ms_per_pass"] == 0.4
    assert status["budget_sources"]["decode"] == "model (ms)"
