"""Exercises for DR-01 (stored `mistudio.*` document kinds) and DR-03 (the probe identity scheme
`mistudio.probe-identity/1`) — 040 FR-040.4, CF-1.

A stored or published document keeps its kind forever, and miLLM's importers reject an unknown
kind, so a renamed kind constant would refuse every document an operator has. These run committed
copies of real documents through the LIVE routes, then list them.
"""

from __future__ import annotations

import json
from pathlib import Path

from miserver.services import probe_digest
from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

REPO = Path(__file__).resolve().parents[4]
FIXTURES = REPO / "tests" / "fixtures"


def _doc(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_stored_documents_of_frozen_kinds_import(client, monkeypatch, tmp_path):  # noqa: F811
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))  # the real SAE service, no /app
    probe = _doc("lfm2_probe_definition.json")
    circuit = _doc("real_circuit_definition.json")
    cluster = _doc("mistudio_export.cluster.json")
    assert probe["kind"] == "mistudio.probe-definition/v1"
    assert circuit["kind"] == "mistudio.circuit-definition"
    assert cluster["kind"] == "mistudio.cluster-definition"

    r = client.post("/api/probes/import", json=probe)
    assert r.status_code == 200, r.text
    r = client.post("/api/circuits/import", json=circuit)
    assert r.status_code == 200 and r.json()["success"] is True, r.text
    r = client.post("/api/clusters/import", json=cluster)
    assert r.status_code == 200 and r.json()["success"] is True, r.text

    assert [p["name"] for p in client.get("/api/probes").json()["data"]] == [probe["name"]]
    circuits = client.get("/api/circuits").json()["data"]
    assert circuit["name"] in json.dumps(circuits)
    clusters = client.get("/api/clusters").json()["data"]
    assert cluster["name"] in json.dumps(clusters)

    # The WRITER side (review M05): a circuit served as a per-layer slice is handed to the cluster
    # path as a `mistudio.cluster-definition` document, which the cluster importer must accept.
    from unittest.mock import MagicMock

    from miserver.api.schemas.circuit import CircuitDefinitionV1
    from miserver.api.schemas.cluster import ClusterDefinitionV1
    from miserver.services.circuit_service import CircuitService

    definition = CircuitDefinitionV1.model_validate(circuit)
    layer = definition.saes[0].layer
    slice_doc = CircuitService(repository=MagicMock()).to_layer_slice(definition, layer,
                                                                      circuit_rung_value=2)
    assert slice_doc["kind"] == "mistudio.cluster-definition"
    assert ClusterDefinitionV1.model_validate(slice_doc).kind == "mistudio.cluster-definition"


def test_probe_identity_scheme_reproduces_shared_cases():
    cases = json.loads((REPO / "docs" / "schemas" / "probe-identity-cases.json").read_text())
    assert probe_digest.IDENTITY_ALGORITHM == "mistudio.probe-identity/1"
    assert cases["algorithm"] == probe_digest.IDENTITY_ALGORITHM
    checked = 0
    for case in cases["cases"]:
        if "detector_digest" not in case:
            continue
        detector, decision = probe_digest.digests_of(case["document"])
        assert (detector, decision) == (case["detector_digest"], case["decision_digest"]), case["name"]
        checked += 1
    assert checked >= 3
