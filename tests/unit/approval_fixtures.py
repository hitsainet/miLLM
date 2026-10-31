"""Shared fixtures for the ENH-008 approval queue tests: the LIVE app (`create_app`, so every
route, decorator and router include is the production one) over a real SQLite database, with the
model service and the capacity measurement replaced by mocks whose calls are counted.

The model row is real (the gate's describers read it through the gate's own session), and so are
probe limits and rule-set imports: what "did not run" means is checked on the database, not only
on a mock.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

AGENT = {"X-miLLM-Agent": "agent:test-bot"}


async def make_model(factory: Any, name: str = "LFM2.5-1.2B-Instruct") -> Any:
    from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType

    async with factory() as session:
        row = Model(name=name, source=ModelSource("huggingface"),
                    repo_id=f"LiquidAI/{name}", quantization=QuantizationType("FP16"),
                    cache_path="/x", params="1B", architecture="text-generation",
                    status=ModelStatus.READY, disk_size_mb=2400)
        session.add(row)
        await session.commit()
        return row


async def rows(factory: Any) -> list[Any]:
    from millm.db.models.approval import ApprovalRequest

    async with factory() as session:
        return list((await session.execute(select(ApprovalRequest))).scalars().all())


@pytest.fixture
def harness(db, monkeypatch):  # noqa: F811
    """`h.client` (TestClient over the live app), `h.db` (session factory), `h.service` (the model
    service mock), `h.measure` (the capacity-measurement mock), `h.events` (socket events)."""
    from fastapi.testclient import TestClient

    from millm.api.dependencies import get_db, get_inference_service, get_model_service
    from millm.main import create_app
    from millm.services import probe_capacity_measure, probe_limits
    from millm.sockets.progress import progress_emitter

    app = create_app()

    async def _db():
        async with db() as session:
            yield session

    holder: dict[str, Any] = {}

    async def _get_model(model_id: int) -> Any:
        return holder["row"]

    async def _returns_row(model_id: int, **_kw: Any) -> Any:
        return holder["row"]

    service = MagicMock()
    service.get_model = AsyncMock(side_effect=_get_model)
    service.load_model = AsyncMock(side_effect=_returns_row)
    service.unload_model = AsyncMock(side_effect=_returns_row)
    service.delete_model = AsyncMock(return_value=None)
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_model_service] = lambda: service
    app.dependency_overrides[get_inference_service] = lambda: MagicMock()
    measure = AsyncMock(return_value=None)
    monkeypatch.setattr(probe_capacity_measure, "measure_and_record", measure)
    monkeypatch.setattr(probe_limits, "resident_model_id", lambda: None)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(progress_emitter, "emit_approval_created",
                        lambda p: events.append(("approval:created", dict(p))))
    monkeypatch.setattr(progress_emitter, "emit_approval_resolved",
                        lambda p: events.append(("approval:resolved", dict(p))))
    return SimpleNamespace(client=TestClient(app), app=app, db=db, service=service,
                           measure=measure, events=events, holder=holder)
