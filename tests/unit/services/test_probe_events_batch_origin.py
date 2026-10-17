"""Feature 26 task 7.3: batch-marked probe events (T-70, FR-26.4.8) and probe-score rows.

Real `ProbeEventService` over a real `ProbeEventRepository` on SQLite: the caps, the dedupe and the
origin are exercised in the database, not asserted on a mock. M10's target is
`test_a_batch_bigger_than_the_live_cap_leaves_every_live_event_in_place`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import func, select

from millm.core.config import settings
from millm.db.models.probe import ProbeEvent
from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services.probe_event_service import ProbeEventService
from tests.unit.batch_fixtures import (  # noqa: F401
    batch_db,
    batch_dir,
    client_for,
    harness,
    jsonl,
    upload,
)
from tests.unit.services.test_probe_survives_no_restart import _probe_row


def _verdict(window: str = "all", score: float = 1.0):
    return SimpleNamespace(
        probe_id="pr_a", window=window, scored=True, not_scored_reason=None, score=score,
        threshold=0.5, fires=True, rung=2, top_positions=[], n_scored_tokens=3,
        provisional=False, threshold_revision=1,
    )


async def _service(factory):
    session = factory()
    s = await session.__aenter__()
    svc = ProbeEventService(ProbeRepository(s), ProbeEventRepository(s))
    svc._emit_events = MagicMock()
    return svc, s


async def _counts(factory) -> dict[str, int]:
    async with factory() as session:
        rows = (await session.execute(
            select(ProbeEvent.origin, func.count()).group_by(ProbeEvent.origin)
        )).all()
    return {origin: n for origin, n in rows}


@pytest.fixture
async def probe_db(batch_db):
    async with batch_db() as session:
        session.add(_probe_row("pr_a", armed=True))
        await session.commit()
    return batch_db


async def test_a_batch_bigger_than_the_live_cap_leaves_every_live_event_in_place(
    probe_db, monkeypatch
):
    monkeypatch.setattr(settings, "PROBE_MAX_EVENTS_PER_PROBE", 3)
    monkeypatch.setattr(settings, "PROBE_MAX_BATCH_EVENTS_PER_PROBE", 4)
    svc, session = await _service(probe_db)
    try:
        for i in range(3):
            await svc.record(f"live-{i}", [_verdict()])
        for line in range(1, 8):
            await svc.record(f"b-{line}", [_verdict()], batch_row=("batch_x", line))
    finally:
        await session.close()
    counts = await _counts(probe_db)
    assert counts == {"live": 3, "batch": 4}
    async with probe_db() as s:
        lines = sorted((await s.execute(
            select(ProbeEvent.batch_line).where(ProbeEvent.origin == "batch")
        )).scalars().all())
    assert lines == [4, 5, 6, 7], "the batch cap keeps the newest batch events"


async def test_a_live_event_past_its_cap_does_not_evict_batch_events(probe_db, monkeypatch):
    monkeypatch.setattr(settings, "PROBE_MAX_EVENTS_PER_PROBE", 2)
    svc, session = await _service(probe_db)
    try:
        await svc.record("b-1", [_verdict()], batch_row=("batch_x", 1))
        for i in range(5):
            await svc.record(f"live-{i}", [_verdict()])
    finally:
        await session.close()
    assert await _counts(probe_db) == {"live": 2, "batch": 1}


async def test_a_re_run_row_records_no_second_event(probe_db):
    svc, session = await _service(probe_db)
    try:
        assert await svc.record("b", [_verdict("all"), _verdict("prompt")],
                                batch_row=("batch_x", 9)) == 2
        assert await svc.record("b", [_verdict("all"), _verdict("prompt")],
                                batch_row=("batch_x", 9)) == 0
    finally:
        await session.close()
    async with probe_db() as s:
        rows = (await s.execute(select(ProbeEvent))).scalars().all()
    assert len(rows) == 2
    assert {(r.origin, r.batch_id, r.batch_line) for r in rows} == {("batch", "batch_x", 9)}


async def test_batch_events_never_reach_the_socket_and_live_ones_do(probe_db):
    svc, session = await _service(probe_db)
    try:
        await svc.record("b", [_verdict()], batch_row=("batch_x", 1))
        assert svc._emit_events.call_count == 0
        await svc.record("live", [_verdict()])
        assert svc._emit_events.call_count == 1
        (payloads,), _ = svc._emit_events.call_args
        assert len(payloads) == 1 and payloads[0]["request_id"] == "live"
    finally:
        await session.close()


async def test_probe_record_passes_the_batch_row_it_runs_under(probe_db, monkeypatch):
    """The WIRING: `_probe_record` reads BATCH_ROW and hands it to `record` (call and payload)."""
    import millm.api.dependencies as deps
    import millm.db.base as db_base
    from millm.services.batch.state import BATCH_ROW
    from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

    monkeypatch.setattr(db_base, "async_session_factory", probe_db)
    record = AsyncMock(return_value=1)
    monkeypatch.setattr(ProbeEventService, "record", record)
    monkeypatch.setattr(deps, "_probe_event_service", None)
    svc = make_service(word_model(), word_tokenizer())
    context = SimpleNamespace(request_id="r1", overhead_ms=None, n_passes=0,
                              finish=lambda: [_verdict()], prompt_length=None, last_user_span=None)
    try:
        token = BATCH_ROW.set(("batch_q", 12))
        try:
            await svc._probe_record(context, detached=True)
        finally:
            BATCH_ROW.reset(token)
        await svc._probe_record(context, detached=True)
    finally:
        clear_loaded()
    assert record.await_count == 2
    assert record.await_args_list[0].kwargs["batch_row"] == ("batch_q", 12)
    assert record.await_args_list[1].kwargs["batch_row"] is None


async def test_probe_score_rows_run_one_input_per_call_and_write_no_events(harness, monkeypatch):
    """FR-26.5.3 / 027 FR-27.4f: one `score` call per row (one input each), whatever `pack` says;
    the body is the endpoint's `ApiResponse`; no probe_events row is written."""
    from millm.services.probe_scoring import ProbeScoringService

    calls: list[int] = []

    async def fake_score(self, request, session):
        calls.append(len(request.inputs))
        return {"results": [{"input": 0}]}

    monkeypatch.setattr(ProbeScoringService, "score", fake_score)
    lines = [{"custom_id": f"p{i}", "method": "POST", "url": "/api/probes/score",
              "body": {"inputs": [{"text": "w1"}]}} for i in range(3)]
    async with client_for(harness.app()) as client:
        created = await harness.create(client, lines, endpoint="/api/probes/score", pack=True)
    assert created.status_code == 200, created.text
    batch_id = created.json()["id"]
    await harness.drain()
    async with client_for(harness.app()) as client:
        batch = await harness.batch(batch_id)
        out = await harness.lines(client, batch.output_file_id)
        err = await harness.lines(client, batch.error_file_id)
    assert batch.status == "completed", err
    assert calls == [1, 1, 1]
    assert [l["response"]["millm"]["packed"] for l in out] == [False] * 3
    assert out[0]["response"]["body"] == {"success": True, "data": {"results": [{"input": 0}]},
                                          "error": None}
    assert await _counts(harness.factory) == {}


def test_a_probe_chunk_is_one_row():
    from millm.services.batch.runner import BatchRunner

    runner = BatchRunner(None, inference_provider=lambda: None, model_service_factory=lambda s: None)
    batch = SimpleNamespace(pack=True)
    assert runner.chunk_size(batch, "probe") == 1
    assert runner.chunk_size(batch, "generation") == 1
