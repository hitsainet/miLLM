"""`X-miLLM-Steering` reports the λ a circuit is APPLYING, through the real app (2026-10-18).

Found on the 3090 serving miStudio's `crc_45f22d113063` (calibrated onset 0.4 / sweet spot 0.85 /
cliff 1.0): after `PUT /api/circuits/active/intensity` the model ran at the PUT value while every
response's `X-miLLM-Steering` still said `intensity="0.85"`. The report read the document's authored
budget; the management dial applies (and stores) its own value. The header is the disclosure of what
ran, so it now reads the row, which activation and the dial both write with the value they applied.
The authored value stays visible as `authored_intensity` on the circuit's GET.

A real `InferenceService` over a tiny real Llama with two real hooked SAEs; the circuit is imported,
activated, dialled and served through the HTTP routes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.unit.f25_fixtures import make_client, model_row
from tests.unit.f28_fixtures import (
    SAE_A,
    SAE_B,
    build,
    clean_state,  # noqa: F401 - fixture
    db,  # noqa: F401 - fixture
    no_cuda,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "real_circuit_definition.json"
CHAT = {"model": "tiny", "messages": [{"role": "user", "content": "w1 w2"}], "max_tokens": 3,
        "temperature": 0.0}


def calibrated_doc() -> dict:
    """The shipped fixture re-pointed at the two attached SAEs, with a calibrated budget."""
    doc = copy.deepcopy(json.loads(FIXTURE.read_text()))
    doc["name"] = "calibrated"
    doc["saes"] = [
        {**doc["saes"][0], "mistudio_sae_id": SAE_A, "layer": 0, "n_features": 8, "d_model": 16},
        {**doc["saes"][1], "mistudio_sae_id": SAE_B, "layer": 1, "n_features": 8, "d_model": 16},
    ]
    doc["members"][0]["layer"] = 0
    doc["members"][0]["feature"].update(feature_idx=3, strength=2.0)
    doc["members"][1]["layer"] = 1
    doc["members"][1]["feature"].update(feature_idx=5, strength=4.0)
    edge = doc["edges"][0]
    edge["up"].update(layer=0, feature_idx=3)
    edge["down"].update(layer=1, feature_idx=5)
    doc["budget"] = {"formula_id": doc["budget"]["formula_id"], "layers": {},
                     "intensity": 0.85, "intensity_range": [0.4, 1.0]}
    return doc


@pytest.fixture
def served(clean_state, db):  # noqa: F811
    from unittest.mock import AsyncMock, MagicMock

    from millm.api.dependencies import get_cluster_service, get_db, get_sae_service
    from millm.services.sae_service import AttachedSAEState, SAEService

    s = build([(SAE_A, 0, 11), (SAE_B, 1, 12)])
    client, _ = make_client(s.svc, model_row())
    # The REAL serving math (`set_circuit_steering`) over the attached registry; only the
    # constructor's disk cache is skipped. No cluster co-tenant exists in this test.
    sae_service = SAEService.__new__(SAEService)
    sae_service._sae_state = AttachedSAEState()
    cluster_service = MagicMock()
    cluster_service.get_active_cluster = AsyncMock(return_value=None)
    client.app.dependency_overrides[get_sae_service] = lambda: sae_service
    client.app.dependency_overrides[get_cluster_service] = lambda: cluster_service

    async def _db():
        async with db() as session:
            yield session

    client.app.dependency_overrides[get_db] = _db
    yield s, client
    for h in s.handles:
        h.remove()


def _chat(client):
    with no_cuda():
        r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 200, r.text
    return r.headers["X-miLLM-Steering"]


def _row(client, circuit_id):
    rows = client.get("/api/circuits").json()["data"]["circuits"]
    return next(r for r in rows if r["id"] == circuit_id)


async def test_the_header_reports_the_applied_lambda_after_a_management_dial(served):
    s, client = served
    imported = client.post("/api/circuits/import", json=calibrated_doc()).json()
    assert imported["success"], imported
    cid = imported["data"]["id"]
    act = client.post(f"/api/circuits/{cid}/activate").json()
    assert act["success"] and act["data"]["serving_mode"] == "full", act

    assert _chat(client) == f'circuit;id="{cid}";intensity="0.85"'

    # Out of band: clamped to the band's ceiling, applied, and disclosed as applied.
    put = client.put("/api/circuits/active/intensity", json={"intensity": 1.5}).json()
    assert put["success"] and put["data"]["intensity"] == pytest.approx(1.0), put
    assert s.sae(SAE_A, 0).get_steering_values() == {3: pytest.approx(2.0 * 1.0)}
    assert s.sae(SAE_B, 1).get_steering_values() == {5: pytest.approx(4.0 * 1.0)}
    assert _chat(client) == f'circuit;id="{cid}";intensity="1.0"'

    # In band: applied as sent.
    client.put("/api/circuits/active/intensity", json={"intensity": 0.6})
    assert s.sae(SAE_A, 0).get_steering_values() == {3: pytest.approx(2.0 * 0.6)}
    assert _chat(client) == f'circuit;id="{cid}";intensity="0.6"'
    row = _row(client, cid)
    assert (row["intensity"], row["authored_intensity"]) == (pytest.approx(0.6), 0.85)

    # Re-activation serves the authored λ again, and the row and the header say so.
    client.post(f"/api/circuits/{cid}/deactivate")
    client.post(f"/api/circuits/{cid}/activate")
    assert s.sae(SAE_A, 0).get_steering_values() == {3: pytest.approx(2.0 * 0.85)}
    assert _row(client, cid)["intensity"] == pytest.approx(0.85)
    assert _chat(client) == f'circuit;id="{cid}";intensity="0.85"'
