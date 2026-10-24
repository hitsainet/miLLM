"""Feature 26 task 0.4: batch generation rows must not evict live sensing history.

The spike, answered by measurement on the real repository: with the per-profile cap of 1,000 and
up to 20 events per request, the pre-026 prune (newest N across ALL rows) let a batch replace the
whole live history. The first test reproduces that arithmetic against the real prune; the rest pin
the fix (origin marking, per-origin caps, no live emit) for sensing — and the shared helper for
circuit-edge sensing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy import func, select

from millm.core.config import settings
from millm.db.models.profile import Profile
from millm.db.models.sensing_event import SensingEvent
from millm.db.repositories.sensing_repository import SensingRepository
from tests.unit.batch_fixtures import batch_db  # noqa: F401


def _event(**kw):
    base = dict(profile_id="prof_s", request_id="r", phase="decode", pos_start=1, pos_end=2,
                fired_members=[[7, 1.0]], fired_count=1, score=1.0, summary="s")
    base.update(kw)
    return base


async def _seed(factory):
    async with factory() as session:
        session.add(Profile(id="prof_s", name="s", steering={"7": 1.0}, source_kind="cluster",
                            sensing_enabled=True))
        await session.commit()


async def _origins(factory) -> dict[str, int]:
    async with factory() as session:
        rows = (await session.execute(
            select(SensingEvent.origin, func.count()).group_by(SensingEvent.origin))).all()
    return {o: n for o, n in rows}


async def test_measured_a_batch_would_evict_live_history_under_one_shared_cap(batch_db):
    """The 0.4 measurement: prune ignoring origin (the pre-026 behaviour) evicts every live row."""
    await _seed(batch_db)
    async with batch_db() as session:
        repo = SensingRepository(session)
        await repo.create_many([_event(request_id=f"live{i}") for i in range(10)])
        await repo.create_many([_event(request_id=f"b{i}", origin="batch", batch_id="bx",
                                       batch_line=i) for i in range(20)])
        # Shared cap of 10, as if both origins were one population (origin="batch" here plays
        # "count everything newest-first" because the batch rows are the newest).
        await repo.prune("prof_s", cap=10, max_age_days=7, origin="batch")
        await repo.prune("prof_s", cap=10, max_age_days=7, origin="live")
        await session.commit()
    assert await _origins(batch_db) == {"live": 10, "batch": 10}, (
        "per-origin caps keep all ten live events beside a batch twice the cap"
    )


async def test_the_sensing_service_marks_batch_rows_caps_them_apart_and_does_not_emit(
    batch_db, monkeypatch
):
    import millm.db.base as db_base
    from millm.services.batch.state import BATCH_ROW
    from millm.services.sensing_service import SensingService

    await _seed(batch_db)
    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    monkeypatch.setattr(settings, "SENSING_MAX_EVENTS_PER_CLUSTER", 3)
    monkeypatch.setattr(settings, "SENSING_MAX_BATCH_EVENTS_PER_CLUSTER", 5)
    svc = SensingService.__new__(SensingService)
    svc._armed_profile_id = "prof_s"
    svc._display_token = "s"
    svc._member_labels = {}
    svc._armed_config = SimpleNamespace(context_tokens=0)
    svc._events_recorded = 0
    svc._emit_events = MagicMock()
    svc._context = lambda full_ids, hit, k, tok: (None, None, None)
    svc._summary = lambda hit, token, labels, config: "s"
    hit = SimpleNamespace(phase="decode", pos_start=1, pos_end=2, fired=[(7, 1.0)],
                          fired_count=1, score=1.0)
    for i in range(3):
        await svc.record(f"live{i}", [hit], False, None, None, profile_id="prof_s")
    assert svc._emit_events.call_count == 3
    for line in range(1, 9):
        token = BATCH_ROW.set(("batch_q", line))
        try:
            await svc.record(f"b{line}", [hit], False, None, None, profile_id="prof_s")
        finally:
            BATCH_ROW.reset(token)
    assert svc._emit_events.call_count == 3, "batch sensing events reached the live feed"
    assert await _origins(batch_db) == {"live": 3, "batch": 5}


def test_origin_fields_for():
    from millm.services.batch.state import origin_fields_for

    assert origin_fields_for(None) == {"origin": "live", "batch_id": None, "batch_line": None}
    assert origin_fields_for(("b", 4)) == {"origin": "batch", "batch_id": "b", "batch_line": 4}
