"""A carried identity, through the real routes (038 R-039, FTASKS 11.3; D38.28, D38.29).

* import and hub import VERIFY a carried `identity`: miLLM recomputes both digests and refuses a
  mismatch `PROBE_IDENTITY_MISMATCH` (409, `details: field, carried, recomputed, algorithm,
  next_step`) and an algorithm it cannot compute `PROBE_IDENTITY_ALGORITHM_UNKNOWN` (409,
  `details: algorithm, supported, next_step`) — nothing is stored on a refusal;
* a verified identity is stored as the producer's: `detector_digest_source: producer`,
  `producer_detector_digest`, miStudio's `decision_revision` — and it reaches every result
  (`test_producer_digest_reaches_result`);
* recalibrate verifies a sent `decision_digest` against the decision it stores, stores
  `decision_revision`, and keeps the stored document's own `identity` describing it;
* the two new request fields are advertised in the live OpenAPI (miStudio feature-detects them).

The documents are miStudio's own case-file documents, so the identity they carry is the producer's.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

REPO = Path(__file__).resolve().parents[3]
CASES = {c["name"]: c for c in json.loads(
    (REPO / "docs" / "schemas" / "probe-identity-cases.json").read_text())["cases"]}
CARRYING = CASES["identity-revision-7"]
WINDOWS = CASES["windows-and-bands"]


def _doc(case=CARRYING) -> dict:
    return copy.deepcopy(case["document"])


def _import(client, doc):  # noqa: F811
    return client.post("/api/probes/import", json=doc)


def _probes(client):  # noqa: F811
    return client.get("/api/probes").json()["data"]


# ── import ───────────────────────────────────────────────────────────────────────


def test_a_verified_identity_is_stored_as_the_producers(client):  # noqa: F811
    response = _import(client, _doc())
    assert response.status_code == 200, response.text
    body = client.get(f"/api/probes/{response.json()['data']['id']}").json()["data"]
    assert body["detector_digest"] == CARRYING["detector_digest"]
    assert body["decision_digest"] == CARRYING["decision_digest"]
    assert body["detector_digest_source"] == "producer"
    assert body["identity_algorithm"] == "mistudio.probe-identity/1"
    assert body["decision_revision"] == 7 and body["threshold_revision"] == 1


def test_a_document_without_identity_is_still_computed(client):  # noqa: F811
    doc = _doc(CASES["residual"])
    response = _import(client, doc)
    body = client.get(f"/api/probes/{response.json()['data']['id']}").json()["data"]
    assert body["detector_digest_source"] == "computed" and body["decision_revision"] is None
    assert body["detector_digest"] == CASES["residual"]["detector_digest"]


@pytest.mark.parametrize("field", ["detector_digest", "decision_digest"])
def test_a_carried_digest_that_is_not_the_documents_own_is_refused(client, field):  # noqa: F811
    doc = _doc()
    doc["identity"][field] = "0" * 64
    response = _import(client, doc)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "PROBE_IDENTITY_MISMATCH"
    assert error["details"]["field"] == field and error["details"]["carried"] == "0" * 64
    assert error["details"]["recomputed"] == CARRYING[field]
    assert error["details"]["algorithm"] == "mistudio.probe-identity/1"
    assert "export_probe_definition" in error["details"]["next_step"]
    assert _probes(client) == [], "a refusal stores nothing"


def test_a_document_edited_after_its_identity_was_written_is_refused(client):  # noqa: F811
    """The threshold moved but the identity was not rebuilt: the decision half no longer agrees."""
    doc = _doc()
    doc["decision"]["threshold"] = 9.5
    response = _import(client, doc)
    assert response.status_code == 409
    assert response.json()["error"]["details"]["field"] == "decision_digest"
    doc = _doc()
    doc["head"]["weights"][0] = 0.5
    assert _import(client, doc).json()["error"]["details"]["field"] == "detector_digest"


def test_an_unknown_algorithm_is_refused(client):  # noqa: F811
    doc = _doc()
    doc["identity"]["algorithm"] = "mistudio.probe-identity/2"
    response = _import(client, doc)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "PROBE_IDENTITY_ALGORITHM_UNKNOWN"
    assert error["details"]["algorithm"] == "mistudio.probe-identity/2"
    assert error["details"]["supported"] == ["mistudio.probe-identity/1"]
    assert error["details"]["next_step"]
    assert _probes(client) == []


def test_an_identity_block_with_a_key_the_contract_does_not_list_is_refused(client):  # noqa: F811
    """Review R1 (6.5), R65-3: `identity` is closed (`additionalProperties: false`). Open, an extra
    key would be stored and re-exported beside a verified digest as if miStudio had written it."""
    doc = _doc()
    doc["identity"]["verified_by"] = "someone"
    response = _import(client, doc)
    assert response.status_code == 422, response.text
    assert "verified_by" in response.text
    assert _probes(client) == []


def test_import_then_read_returns_the_identity_block_unchanged(client):  # noqa: F811
    """Review R1 (6.5): the carried block survives import → read byte-for-byte (as JSON values),
    and the stored document still verifies — so a re-export is importable anywhere."""
    from millm.services.probe_digest import digests_of, identity_refusal

    doc = _doc()
    probe_id = _import(client, doc).json()["data"]["id"]
    stored = client.get(f"/api/probes/{probe_id}").json()["data"]["definition"]
    assert json.dumps(stored["identity"], sort_keys=True) == json.dumps(doc["identity"],
                                                                         sort_keys=True)
    assert identity_refusal(stored["identity"], *digests_of(stored)) is None


def test_hub_import_verifies_the_same_way(client):  # noqa: F811
    from millm.api.dependencies import get_probe_hub_service

    hub = MagicMock()
    client.app.dependency_overrides[get_probe_hub_service] = lambda: hub
    body = {"repo_id": "org/probes", "filename": "p.probe.json"}

    bad = _doc()
    bad["identity"]["detector_digest"] = "1" * 64
    hub.fetch_definition = AsyncMock(return_value=(None, bad, None))
    response = client.post("/api/probes/hub/import", json=body)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PROBE_IDENTITY_MISMATCH"
    assert _probes(client) == []

    hub.fetch_definition = AsyncMock(return_value=(None, _doc(), None))
    response = client.post("/api/probes/hub/import", json=body)
    assert response.status_code == 200, response.text
    stored = client.get(f"/api/probes/{response.json()['data']['id']}").json()["data"]
    assert stored["detector_digest_source"] == "producer" and stored["decision_revision"] == 7
    assert hub.fetch_definition.await_count == 1


async def test_producer_digest_reaches_result(client, db):  # noqa: F811
    """Control M7 (second half): the producer's identity must reach the result a client reads."""
    from millm.api.schemas.probe_result import wire
    from millm.db.models.probe import Probe
    from millm.services.probe_arming import armed_probe_from_row
    from millm.services.probe_result import build_probe_results
    from millm.services.probe_runtime import ExecutionFacts, Verdict

    probe_id = _import(client, _doc()).json()["data"]["id"]
    async with db() as session:
        row = await session.get(Probe, probe_id)
        assert row.producer_detector_digest == CARRYING["detector_digest"]
        live = armed_probe_from_row(row, windows=["all"])
    verdict = Verdict(probe_id=probe_id, name=row.name, rung=2,
                      rung_language="detects on unseen tasks", scored=True, score=3.0,
                      threshold=2.38, fires=True, n_scored_tokens=2, threshold_revision=1)
    context = SimpleNamespace(request_id="chatcmpl-p", probes=[live], session=None, trace=None,
                              execution=ExecutionFacts())
    data = wire(build_probe_results([verdict], context)[0])
    assert data["detector_digest"] == CARRYING["detector_digest"]
    assert data["detector_digest_source"] == "producer"
    assert data["identity_algorithm"] == "mistudio.probe-identity/1"
    assert data["decision_digest"] == CARRYING["decision_digest"]
    assert data["decision_revision"] == 7


# ── recalibrate (D38.28) ─────────────────────────────────────────────────────────


def _recut(decision: dict, **extra) -> dict:
    return {"decision": decision, "mistudio_probe_id": "pm_example",
            "mistudio_run_id": "pmr_example", "reason": "re-cut", **extra}


def _digest(decision: dict) -> str:
    from millm.services.probe_digest import decision_digest

    return decision_digest(decision)


def _new_decision() -> dict:
    decision = copy.deepcopy(CARRYING["document"]["decision"])
    decision.update(threshold=2.9, target_fpr=0.005, realised_fpr=0.0049)
    return decision


def test_openapi_advertises_the_two_recalibrate_fields(client):  # noqa: F811
    schema = client.app.openapi()["components"]["schemas"]["ProbeRecalibrationRequest"]
    props = schema["properties"]
    assert props["decision_digest"]["anyOf"][0]["pattern"] == "^[0-9a-f]{64}$"
    assert props["decision_revision"]["anyOf"][0]["minimum"] == 1


async def test_a_verified_recut_stores_the_revision_and_keeps_identity_true(client, db):  # noqa: F811
    from millm.db.models.probe import Probe
    from millm.services.probe_digest import digests_of, identity_refusal

    probe_id = _import(client, _doc()).json()["data"]["id"]
    decision = _new_decision()
    response = client.post(f"/api/probes/{probe_id}/recalibrate", json=_recut(
        decision, decision_digest=_digest(decision), decision_revision=8))
    assert response.status_code == 200, response.text
    body = client.get(f"/api/probes/{probe_id}").json()["data"]
    assert body["decision_revision"] == 8 and body["threshold_revision"] == 2
    assert body["decision_digest"] == _digest(decision)
    assert body["detector_digest_source"] == "producer", "a re-cut cannot move the detector"
    async with db() as session:
        row = await session.get(Probe, probe_id)
        identity = row.definition["identity"]
        assert identity["decision_revision"] == 8
        assert identity_refusal(identity, *digests_of(row.definition)) is None, (
            "the stored document's identity must still be its own, or a re-export is refused")
        assert row.threshold_history[-1]["decision_revision"] == 8


async def test_a_recut_without_a_digest_states_no_revision(client, db):  # noqa: F811
    from millm.db.models.probe import Probe

    probe_id = _import(client, _doc()).json()["data"]["id"]
    response = client.post(f"/api/probes/{probe_id}/recalibrate", json=_recut(_new_decision()))
    assert response.status_code == 200, response.text
    body = client.get(f"/api/probes/{probe_id}").json()["data"]
    assert body["decision_revision"] is None, "revision 7 described the PREVIOUS cut"
    async with db() as session:
        row = await session.get(Probe, probe_id)
        assert row.definition["identity"]["decision_digest"] == body["decision_digest"]


def test_a_recut_whose_digest_names_another_decision_is_refused(client):  # noqa: F811
    probe_id = _import(client, _doc()).json()["data"]["id"]
    decision = _new_decision()
    sent = _digest({**decision, "calibration": None})  # what a sender dropping a field computes
    response = client.post(f"/api/probes/{probe_id}/recalibrate", json=_recut(
        decision, decision_digest=sent, decision_revision=8))
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "PROBE_IDENTITY_MISMATCH"
    assert error["details"]["field"] == "decision_digest"
    assert error["details"]["carried"] == sent
    assert error["details"]["recomputed"] == _digest(decision)
    body = client.get(f"/api/probes/{probe_id}").json()["data"]
    assert body["threshold_revision"] == 1 and body["decision_revision"] == 7, "nothing written"
    assert body["decision_digest"] == CARRYING["decision_digest"]


async def test_a_recut_moves_the_live_descriptors_revision(db):  # noqa: F811
    """An ARMED probe's results must name THIS cut's revision, not the one it was armed with."""
    from millm.db.repositories.probe_repository import ProbeRepository
    from millm.services.probe_arming import armed_probe_from_row
    from millm.services.probe_recalibration import ProbeRecalibrationService
    from millm.services.probe_runtime import ProbeRuntimeState
    from millm.services.probe_service import ProbeService
    from tests.unit.f25_fixtures import word_model

    ProbeRuntimeState.reset_for_tests()
    try:
        async with db() as session:
            repo = ProbeRepository(session)
            from millm.services.probe_digest import digests_of

            doc = _doc()
            doc["read"]["layer"] = 0  # `word_model` is shallow; the identity is rebuilt to match
            doc["identity"]["detector_digest"], doc["identity"]["decision_digest"] = (
                digests_of(doc))
            row = await ProbeService(repo).import_definition(doc)
            state = ProbeRuntimeState()
            state.arm(armed_probe_from_row(row, windows=["all"]), word_model())
            assert state.get(row.id).descriptor.decision.decision_revision == 7
            decision = _new_decision()
            await ProbeRecalibrationService(repo, state).recalibrate(
                row, decision=decision, mistudio_probe_id="pm_example",
                mistudio_run_id="pmr_example", decision_digest=_digest(decision),
                decision_revision=8)
            live = state.get(row.id).descriptor.decision
            assert (live.decision_revision, live.decision_digest) == (8, _digest(decision))
    finally:
        ProbeRuntimeState.reset_for_tests()
