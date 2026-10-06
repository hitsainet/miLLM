"""Stateless scoring writes NOTHING and changes no armed state (FR-27.4f, FR-27.7; task 4.11).

Real SQLite, real repositories, a tiny real model, and a probe ARMED on the same layer the scoring
call reads. After a scoring call: the `probe_events` count is unchanged, the runtime's
`begin_request` was never called, `has_armed()` and the armed set are unchanged, the armed probe's
hook recorded nothing (no context was open), and the stored parity report is byte-for-byte the
same.

Mutations this catches (FTID §8): M13 (call `begin_request` inside `_score_one`) and M14 (write a
`probe_events` row from the score route).
"""

from __future__ import annotations

import copy
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services.probe_arming import armed_probe_from_row
from millm.services.probe_runtime import ProbeRuntimeState
from millm.services.probe_service import ProbeService
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.score_fixtures import tiny_definition, tiny_identity


@pytest.fixture(autouse=True)
def clean():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


async def test_a_scoring_call_writes_no_event_and_changes_no_state(test_session, monkeypatch):
    from millm.api.dependencies import get_db, get_inference_service, get_probe_repository
    from millm.main import create_app

    model, tokenizer = word_model(), word_tokenizer()
    inference = make_service(model, tokenizer)
    monkeypatch.setattr(
        "millm.services.probe_arm_bridge.loaded_identity",
        AsyncMock(return_value=(tiny_identity(), model, tokenizer)),
    )
    repo = ProbeRepository(test_session)
    events = ProbeEventRepository(test_session)
    row = await ProbeService(repo).import_definition(tiny_definition())
    stored_parity = {"passed": True, "max_abs_diff": 0.0, "model": {"hf_id": "tiny/llama"}}
    await repo.update(row, parity=copy.deepcopy(stored_parity))

    # ARMED on the same layer the scoring call reads.
    state = ProbeRuntimeState()
    state.arm(armed_probe_from_row(row, windows=["all"]), model)
    armed_before = [p.probe_id for p in state.armed()]
    observed = []
    real_on = state._on_activations

    def watching(layer, hidden):
        observed.append(state._request)
        real_on(layer, hidden)

    monkeypatch.setattr(state, "_on_activations", watching)
    begins = []
    real_begin = state.begin_request
    monkeypatch.setattr(state, "begin_request",
                        lambda rid: (begins.append(rid), real_begin(rid))[1])
    events_before = await events.count()

    async def db():
        yield test_session

    app = create_app()
    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_inference_service] = lambda: inference
    app.dependency_overrides[get_probe_repository] = lambda: repo
    # No `with`: entering the client runs the app's lifespan, which reconfigures logging
    # and blinded later tests' log capture.
    client = TestClient(app)
    response = client.post("/api/probes/score", json={
        "inputs": [{"token_ids": [2, 6, 7, 8]}, {"messages": [{"role": "user",
                                                                 "content": "w1 w2"}]}],
    })
    assert response.status_code == 200, response.text
    assert response.json()["data"]["results"][0]["verdicts"], "precondition: it scored"

    assert await events.count() == events_before, "a scoring call wrote a probe_events row"
    assert begins == [], "the runtime's begin_request must never be called by scoring"
    assert state.has_armed() is True and [p.probe_id for p in state.armed()] == armed_before
    assert observed and all(ctx is None for ctx in observed), (
        "the armed hook ran (same layer) and must have seen no open request context"
    )
    assert state.current_request() is None
    await test_session.refresh(row)
    assert row.parity == stored_parity, "a scoring call changed the stored parity report"
