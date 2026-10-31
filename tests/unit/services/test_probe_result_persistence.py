"""Persistence, digests at import and re-cut, and the stored object (FTASKS 4.6 – 4.8).

* import stores both digests; a re-cut moves `decision_digest` with the revision in the row, the
  history entry and the LIVE descriptor, so the next result carries both (control: drop the
  `descriptor=` keyword from the re-cut's `replace`);
* `record()` writes `result` and its projected columns in one statement (T-S1-10), and
  `recorded_from_row(row)` minus `{persistence, source_event_id}` equals the finalized object the
  client received, byte for byte (T-S1-03);
* a forced write failure gives `not_persisted` objects and no `source_event_id`, never raises,
  and nothing reaches the socket (control M8: report it `persisted`).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import jsonschema
import pytest
import torch

from millm.api.schemas.probe_result import strict_schema, wire
from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services.probe_arming import armed_probe_from_row
from millm.services.probe_digest import decision_digest, detector_digest
from millm.services.probe_event_service import ProbeEventService
from millm.services.probe_recalibration import ProbeRecalibrationService
from millm.services.probe_result import build_probe_results, legacy_columns, recorded_from_row
from millm.services.probe_runtime import ExecutionFacts, ProbeRuntimeState, Verdict
from millm.services.probe_service import ProbeService
from tests.unit.f25_fixtures import word_model
from tests.unit.score_fixtures import tiny_definition

pytestmark = pytest.mark.asyncio
STRICT = jsonschema.Draft202012Validator(strict_schema())


@pytest.fixture(autouse=True)
def clean():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()


def _definition(threshold: float = 0.5) -> dict:
    doc = tiny_definition("probe-p", layer=0, threshold=threshold)
    doc["provenance"] = {"probe_id": "pm_src", "run_id": "run_1"}
    return doc


async def test_import_stores_both_digests(test_session):
    row = await ProbeService(ProbeRepository(test_session)).import_definition(_definition())
    assert row.detector_digest == detector_digest(_definition())
    assert row.decision_digest == decision_digest(_definition()["decision"])
    assert row.detector_digest_source == "computed"
    assert row.identity_algorithm == "mistudio.probe-identity/1"


async def test_a_recut_moves_the_decision_digest_and_the_revision_together(test_session):
    repo = ProbeRepository(test_session)
    row = await ProbeService(repo).import_definition(_definition())
    state = ProbeRuntimeState()
    state.arm(armed_probe_from_row(row, windows=["all"]), word_model())
    before = state.get(row.id).descriptor.decision.decision_digest

    new_decision = {**_definition()["decision"], "threshold": 0.9, "target_fpr": 0.02,
                    "threshold_source": "validation_negatives"}
    await ProbeRecalibrationService(repo, state).recalibrate(
        row, decision=new_decision, mistudio_probe_id="pm_src", calibration_id="cal_2",
    )
    expected = decision_digest(new_decision)
    assert row.decision_digest == expected and row.detector_digest == detector_digest(
        _definition()), "a re-cut cannot move the detector"
    assert row.threshold_history[-1]["decision_digest"] == expected

    live = state.get(row.id)
    assert live.descriptor.decision.decision_digest == expected != before
    verdict = Verdict(probe_id=row.id, name=row.name, rung=2,
                      rung_language="detects on unseen tasks", scored=True, score=1.0,
                      threshold=0.9, fires=True, n_scored_tokens=2,
                      threshold_revision=live.threshold_revision)
    context = SimpleNamespace(request_id="chatcmpl-r", probes=[live], session=None, trace=None,
                              execution=ExecutionFacts())
    data = wire(build_probe_results([verdict], context)[0])
    assert data["decision_digest"] == expected and data["threshold_revision"] == 2
    assert data["decision"]["calibration_id"] == "cal_2"


def _verdicts(probe_id: str) -> list[Verdict]:
    return [
        Verdict(probe_id=probe_id, name="probe-p", rung=2,
                rung_language="detects on unseen tasks", scored=True, score=1.25, threshold=0.5,
                fires=True, n_scored_tokens=4, window="all", served_model="tiny"),
        Verdict(probe_id=probe_id, name="probe-p", rung=2,
                rung_language="detects on unseen tasks", scored=False,
                not_scored_reason="no_scored_tokens", window="response", served_model="tiny"),
    ]


async def test_the_stored_object_is_the_finalized_one_byte_for_byte(test_session):
    row = await ProbeService(ProbeRepository(test_session)).import_definition(_definition())
    verdicts = _verdicts(row.id)
    context = SimpleNamespace(request_id="chatcmpl-s", probes=(), session=None, trace=None)
    finalized = build_probe_results(verdicts, context)
    service = ProbeEventService(ProbeRepository(test_session), ProbeEventRepository(test_session))
    service._emit_request = MagicMock()
    assert await service.record("chatcmpl-s", verdicts, results=finalized) == 2

    stored = await ProbeEventRepository(test_session).list_events(request_id="chatcmpl-s")
    by_id = {r.result_id: r for r in stored}
    for result in finalized:
        client = wire(result)
        event = by_id[client["result_id"]]
        recorded = recorded_from_row(event)
        STRICT.validate(recorded)
        assert recorded["persistence"] == "persisted" and recorded["source_event_id"] == event.id
        assert {k: v for k, v in recorded.items()
                if k not in ("persistence", "source_event_id")} == client, "T-S1-03"
        # T-S1-10: every projected column equals the projection of the stored object.
        for column, value in legacy_columns(result).items():
            if column in ("event_time",):
                assert getattr(event, column).replace(tzinfo=None) == value.replace(tzinfo=None)
                continue
            assert getattr(event, column) == value, column
    emitted = [p for call in service._emit_request.call_args_list for p in call.args[1]]
    assert {p["result_id"] for p in emitted} == {wire(r)["result_id"] for r in finalized}
    assert all(p["source_event_id"] == p["id"] for p in emitted)


async def test_a_failed_write_is_not_persisted_and_reaches_no_socket(test_session):
    row = await ProbeService(ProbeRepository(test_session)).import_definition(_definition())
    events = ProbeEventRepository(test_session)

    async def boom(_rows):
        raise RuntimeError("database is gone")

    events.create_many = boom
    service = ProbeEventService(ProbeRepository(test_session), events)
    service._emit_request = MagicMock()
    assert await service.record("chatcmpl-f", _verdicts(row.id)) == 0
    assert service._emit_request.call_count == 0, "a row with no id must never reach the socket"
    assert len(service.last_not_persisted) == 2
    for result in service.last_not_persisted:
        data = wire(result)
        assert data["persistence"] == "not_persisted"
        assert "source_event_id" not in data
        STRICT.validate(data)


async def test_a_failed_write_logs_the_exception_class_never_its_message(test_session, caplog):
    """Review R1: putting `str(exc)` back in `probe_event_persist_failed` survived the suite. A
    failed INSERT's message quotes the row's values — `context_text` is one of them."""
    import logging

    row = await ProbeService(ProbeRepository(test_session)).import_definition(_definition())
    events = ProbeEventRepository(test_session)

    async def boom(_rows):
        raise RuntimeError("INSERT failed: context_text='SENTINEL-the-users-words'")

    events.create_many = boom
    service = ProbeEventService(ProbeRepository(test_session), events)
    service._emit_request = MagicMock()
    with caplog.at_level(logging.DEBUG):
        assert await service.record("chatcmpl-f", _verdicts(row.id)) == 0
    logged = [r for r in caplog.records if r.getMessage() == "probe_event_persist_failed"]
    assert logged, "the failure must still be logged"
    assert all(getattr(r, "error", None) == "RuntimeError" for r in logged)
    everything = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert "SENTINEL-the-users-words" not in everything
