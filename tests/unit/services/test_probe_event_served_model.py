"""Probe events persist and carry `served_model` (033 D-14, addendum D-36, migration 022).

Real `ProbeEventService` over a real `ProbeEventRepository` on SQLite: the column is written and
read back from the database, and the socket payload and the REST shape (`event_summary`, one
function for both) carry it. The two verdict payloads (the stream's `_verdict_payload` and the
scoring route's `verdict_payload`) carry it too. NULL / None = not recorded.

MUTATION CONTROLS (W3-B report, each verified red): M7 `record` drops `served_model` from the row
dict; M8 `event_summary` drops it; M9 `_verdict_payload` drops it; M10 `verdict_payload` drops it.
Migration 022 / ORM: M11 (width 64 in the migration), M13 (downgrade keeps the column) red in
`tests/schema/test_probe_served_model_migration.py`; M12 (ORM width 64) red in
`tests/schema/test_schema_guards.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from millm.db.models.probe import ProbeEvent
from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services.inference_service import _verdict_payload
from millm.services.probe_event_service import ProbeEventService, event_summary
from millm.services.probe_runtime import Verdict
from millm.services.probe_scoring import verdict_payload
from tests.unit.batch_fixtures import batch_db, batch_dir  # noqa: F401
from tests.unit.services.test_probe_survives_no_restart import _probe_row

ADAPTED = "org/base:r9:adapter-200"


def _verdict(served_model, *, scored: bool = True) -> Verdict:
    return Verdict(
        probe_id="pr_a", name="concept", rung=2, rung_language="detects on unseen tasks",
        scored=scored, score=1.0 if scored else None, threshold=0.5 if scored else None,
        fires=True if scored else None, not_scored_reason=None if scored else "no_scored_tokens",
        n_scored_tokens=3 if scored else 0, provisional=served_model == ADAPTED,
        served_model=served_model,
    )


@pytest.fixture
async def probe_db(batch_db):  # noqa: F811
    async with batch_db() as session:
        session.add(_probe_row("pr_a", armed=True))
        await session.commit()
    return batch_db


async def test_the_event_row_records_which_model_was_read(probe_db):
    async with probe_db() as session:
        service = ProbeEventService(ProbeRepository(session), ProbeEventRepository(session))
        service._emit_request = MagicMock()  # 039: one message per request
        await service.record("req-adapted", [_verdict(ADAPTED)])
        await service.record("req-base", [_verdict("tiny", scored=False)])
        # A verdict object from before D-14 has no attribute: NULL, never a guess.
        legacy = SimpleNamespace(**{k: v for k, v in vars(_verdict("x")).items()
                                    if k != "served_model"})
        await service.record("req-legacy", [legacy])

    async with probe_db() as session:
        rows = dict((await session.execute(
            select(ProbeEvent.request_id, ProbeEvent.served_model)
        )).all())
    assert rows == {"req-adapted": ADAPTED, "req-base": "tiny", "req-legacy": None}

    emitted = [call.args[1][0] for call in service._emit_request.call_args_list]
    # 038 FR-38.6: the socket payload's `served_model` is the contract's, which says "unknown"
    # (never a guess) where the row records NULL. The row above keeps its NULL.
    assert [e["served_model"] for e in emitted] == [ADAPTED, "tiny", "unknown"]
    assert emitted[0]["provisional"] is True


def test_event_summary_carries_it_and_none_when_not_recorded():
    row = SimpleNamespace(
        id=1, probe_id="pr_a", request_id="r", scored=True, not_scored_reason=None, score=1.0,
        threshold=0.5, verdict=True, rung=2, top_positions=[], n_scored_tokens=3, summary="s",
        window="all", provisional=True, threshold_revision=1, created_at=None,
        served_model=ADAPTED,
    )
    assert event_summary(row)["served_model"] == ADAPTED
    row.served_model = None
    assert event_summary(row)["served_model"] is None


@pytest.mark.parametrize("scored", [True, False])
def test_both_verdict_payloads_carry_it(scored):
    verdict = _verdict(ADAPTED, scored=scored)
    assert _verdict_payload(verdict)["served_model"] == ADAPTED
    assert verdict_payload(verdict)["served_model"] == ADAPTED
    assert _verdict_payload(_verdict(None, scored=scored))["served_model"] is None
