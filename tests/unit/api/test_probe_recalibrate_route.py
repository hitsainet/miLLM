"""The recalibrate route: the envelope that cannot carry a detector, and the gates before the write.

⚠ **WHY THIS ROUTE IS NOT `on_conflict=replace` WEARING A DIFFERENT HAT.**

`replace` is refused on import because overwriting a definition in place would change the DETECTOR
underneath a running monitor while every event kept the same `probe_id` — the history would
describe two different detectors as one. The refusal is normative in five places and
`test_probe_arm_route.py::test_replace_is_refused_by_the_schema` guards it; that test must stay
green and unmodified, and it is the proof this carve-out did not widen anything.

The carve-out holds on four properties, and this file asserts each of them at the HTTP boundary:

1. The request type is `extra="forbid"`, so it is **structurally incapable** of carrying a
   detector. Not "accepts it and ignores it", which is one review away from honouring it.
2. The cut must **prove it describes the same probe**. There is no digest of a probe head's
   weights anywhere in the contract — `weights_sha256` belongs to the `sae` block, the SAE FILE —
   so identity rests on `provenance.probe_id`, and a definition carrying none **refuses**.
3. The incoming object is **never stored as the definition**; the detector half travels through.
4. A `threshold` with no budget and no source is **refused**.

Every refusal asserts `repository.update` was never awaited. A gate that runs after the write is
not a gate.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from millm.main import create_app

MISTUDIO_ID = "pm_36d1a65f7953"
RUN_ID = "pmr_07e1c12e383a"
D = 4


def _decision(**over) -> dict:
    base = {
        "threshold": 14.5201,
        "target_fpr": 0.005,
        "realised_fpr": 0.005,
        "threshold_source": "calibration_set",
    }
    base.update(over)
    return base


def _definition(**over) -> dict:
    base = {
        "head": {
            "weights": [1.0] * D,
            "bias": 0.0,
            "norm_mean": [0.0] * D,
            "norm_std": [1.0] * D,
        },
        "read": {"layer": 11, "hook_point": "resid_post"},
        "aggregation": {"rule": "mean", "params": {}},
        "decision": _decision(threshold=11.9144, target_fpr=0.01, realised_fpr=0.01),
        "provenance": {"probe_id": MISTUDIO_ID, "run_id": RUN_ID},
    }
    base.update(over)
    return base


def _row(**over):
    row = MagicMock()
    row.id = "pr_1"
    row.name = "high-stakes"
    row.layer = 11
    row.rule = "mean"
    row.scope = "all"
    row.rung = 2
    row.armed = False
    row.threshold = 11.9144
    row.target_fpr = 0.01
    row.threshold_revision = 1
    row.threshold_history = None
    row.created_at = None
    row.definition = _definition()
    for key, value in over.items():
        setattr(row, key, value)
    return row


def _client(probe=None):
    from millm.api.dependencies import get_probe_repository

    repo = MagicMock()
    repo.get = AsyncMock(return_value=probe)
    repo.update = AsyncMock()
    app = create_app()
    app.dependency_overrides[get_probe_repository] = lambda: repo
    return TestClient(app, raise_server_exceptions=False), repo


def _body(**over) -> dict:
    base = {"decision": _decision(), "mistudio_probe_id": MISTUDIO_ID}
    base.update(over)
    return base


def _post(client, body):
    return client.post("/api/probes/pr_1/recalibrate", json=body)


class TestTheHappyPathWritesOnceAndKeepsTheDetector:
    def test_it_answers_200_with_both_ends_of_the_move(self):
        client, repo = _client(_row())
        response = _post(client, _body())
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["threshold"] == 14.5201
        assert data["previous_threshold"] == 11.9144
        assert data["threshold_revision"] == 2
        assert data["previous_revision"] == 1
        assert data["registry_updated"] is False

    def test_the_row_is_updated_EXACTLY_ONCE(self):
        """Two calls would open a window in which the column and the definition disagree."""
        client, repo = _client(_row())
        _post(client, _body())
        assert repo.update.await_count == 1

    def test_the_update_carries_the_definition_AND_the_projection(self):
        """⚠ THE PAYLOAD, NOT THAT IT WAS CALLED. The model's doctrine is that nothing writes a
        projected column without writing the definition in the same transaction."""
        client, repo = _client(_row())
        _post(client, _body())
        kwargs = repo.update.await_args.kwargs
        assert "definition" in kwargs
        assert kwargs["threshold"] == 14.5201
        assert kwargs["target_fpr"] == 0.005
        assert kwargs["threshold_revision"] == 2

    def test_the_stored_definition_keeps_the_DETECTOR_byte_for_byte(self):
        """Refusal 3 of the carve-out: the incoming object is never stored as the definition.
        This is what keeps "nothing replaces a definition in place" true of the weights."""
        row = _row()
        original = dict(row.definition)
        client, repo = _client(row)
        _post(client, _body())
        written = repo.update.await_args.kwargs["definition"]
        for key in ("head", "read", "aggregation", "provenance"):
            assert written[key] == original[key]
        assert written["decision"]["threshold"] == 14.5201

    def test_the_reason_is_recorded_against_the_revision(self):
        client, repo = _client(_row())
        _post(client, _body(reason="tighter budget for triage"))
        history = repo.update.await_args.kwargs["threshold_history"]
        assert history[-1]["reason"] == "tighter budget for triage"
        assert history[-1]["revision"] == 2

    def test_the_calibration_id_travels_verbatim(self):
        client, repo = _client(_row())
        _post(client, _body(calibration_id="pmd_3222ea3bc591"))
        assert repo.update.await_args.kwargs["threshold_calibration_id"] == "pmd_3222ea3bc591"


class TestTheEnvelopeCannotCarryADetector:
    """⚠ `extra="forbid"` IS THE DOCTRINAL BOUNDARY EXPRESSED AS A TYPE."""

    @pytest.mark.parametrize(
        "field",
        ["head", "read", "aggregation", "scope", "basis", "model", "evidence", "sae"],
    )
    def test_a_detector_field_is_REFUSED_not_ignored(self, field):
        client, repo = _client(_row())
        response = _post(client, {**_body(), field: {"anything": 1}})
        assert response.status_code == 422, (
            f"{field!r} was accepted. A route that parses a detector field and ignores it is one "
            f"review away from honouring it — the boundary must be structural"
        )
        assert repo.update.await_count == 0

    def test_a_bare_float_is_not_a_decision(self):
        client, repo = _client(_row())
        response = _post(client, {"decision": 14.5, "mistudio_probe_id": MISTUDIO_ID})
        assert response.status_code == 422
        assert repo.update.await_count == 0

    def test_the_decision_block_itself_stays_ADDITIVE(self):
        """`Decision` is `extra="allow"`, so a newer producer can add `decision.*` fields and they
        survive the round trip. Closed outside the bar, open inside it."""
        client, repo = _client(_row())
        response = _post(
            client, _body(decision=_decision(some_future_field={"k": "v"}))
        )
        assert response.status_code == 200, response.text
        written = repo.update.await_args.kwargs["definition"]["decision"]
        assert written["some_future_field"] == {"k": "v"}


class TestEveryRefusalHappensBEFORETheWrite:
    def test_an_unknown_probe_is_404(self):
        client, repo = _client(None)
        assert _post(client, _body()).status_code == 404
        assert repo.update.await_count == 0

    def test_a_cut_naming_a_different_probe_is_409_with_its_own_code(self):
        client, repo = _client(_row())
        response = _post(client, _body(mistudio_probe_id="pm_somebody_else"))
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "PROBE_RECALIBRATION_MISMATCH"
        assert repo.update.await_count == 0

    def test_a_cut_from_a_different_RUN_is_refused(self):
        client, repo = _client(_row())
        response = _post(client, _body(mistudio_run_id="pmr_elsewhere"))
        assert response.status_code == 409
        assert repo.update.await_count == 0

    def test_a_definition_with_no_PROVENANCE_refuses_rather_than_assuming(self):
        client, repo = _client(_row(definition=_definition(provenance={})))
        response = _post(client, _body())
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "PROBE_RECALIBRATION_MISMATCH"
        assert repo.update.await_count == 0

    def test_a_threshold_with_no_budget_is_409_UNCALIBRATED(self):
        """A distinct code: "this is a different probe" and "this is not a calibrated bar" send an
        operator to different places."""
        client, repo = _client(_row())
        response = _post(client, _body(decision=_decision(target_fpr=None)))
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "PROBE_THRESHOLD_UNCALIBRATED"
        assert repo.update.await_count == 0

    def test_a_window_entry_the_runtime_would_DROP_is_refused(self):
        client, repo = _client(_row())
        response = _post(
            client, _body(decision=_decision(windows={"prompt": {"threshold": None}}))
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "PROBE_THRESHOLD_UNCALIBRATED"
        assert repo.update.await_count == 0


class TestTheCarveOutDidNotWidenImport:
    def test_on_conflict_REPLACE_is_still_refused(self):
        """⚠ THE PROOF THIS FEATURE CHANGED NOTHING ABOUT IMPORT. If this ever passes `replace`,
        the carve-out became the thing it was carved out of."""
        client, _repo = _client(_row())
        response = client.post(
            "/api/probes/import",
            json={"definition": _definition(), "on_conflict": "replace"},
        )
        assert response.status_code == 422
