"""Schema-sync test (Feature 13, task 2.3): the pydantic mirror in
``millm/api/schemas/circuit.py`` must stay conformant with the vendored frozen
contract at ``docs/schemas/circuit-definition-v1.json``.

Unlike the cluster mirror (which regenerates miStudio's generator output
byte-for-byte), miLLM's circuit mirror deliberately REUSES miLLM's own cluster
sub-models and adds a consumer-side per-layer member cap. So this test pins
STRUCTURAL CONFORMANCE instead of byte equality: every field the frozen
contract defines must exist on the mirror, required fields must stay required,
and the caps must match. If this fails the mirror drifted from v1 — fix the
mirror, never the vendored file (v1 is frozen; changes require a v2).
"""

import json
from pathlib import Path

import pytest

from millm.api.schemas.circuit import (
    CIRCUIT_DEFINITION_KIND,
    CIRCUIT_SCHEMA_VERSION,
    MAX_EDGES,
    MAX_SAES,
    CircuitDefinitionV1,
    CircuitEdge,
    CircuitMemberV1,
    CircuitNodeRef,
)

VENDORED = (
    Path(__file__).resolve().parents[3] / "docs" / "schemas" / "circuit-definition-v1.json"
)


@pytest.fixture(scope="module")
def frozen() -> dict:
    assert VENDORED.exists(), f"vendored circuit schema missing at {VENDORED}"
    return json.loads(VENDORED.read_text())


@pytest.fixture(scope="module")
def defs(frozen) -> dict:
    return frozen["$defs"]


def test_vendored_is_the_v1_contract(frozen):
    assert frozen["$id"].endswith("circuit-definition-v1.json")
    assert "CircuitDefinitionV1" in frozen["$defs"]


@pytest.mark.parametrize(
    "def_name,model",
    [
        ("CircuitDefinitionV1", CircuitDefinitionV1),
        ("CircuitEdge", CircuitEdge),
        ("CircuitNodeRef", CircuitNodeRef),
        ("CircuitMember", CircuitMemberV1),
    ],
)
def test_every_contract_field_exists_on_the_mirror(defs, def_name, model):
    """No field of the frozen contract may be missing from the mirror —
    a missing field would be silently dropped on import/re-export."""
    contract_fields = set((defs[def_name].get("properties") or {}).keys())
    mirror_fields = set(model.model_fields.keys())
    missing = contract_fields - mirror_fields
    assert not missing, f"{def_name}: mirror is missing contract fields {sorted(missing)}"


def test_required_fields_match_contract(defs):
    """Required stays required — relaxing one would accept an invalid doc."""
    assert set(defs["CircuitDefinitionV1"].get("required", [])) == {
        "name",
        "saes",
        "members",
    }
    assert set(defs["CircuitEdge"].get("required", [])) == {"up", "down"}
    assert set(defs["CircuitNodeRef"].get("required", [])) == {"layer"}
    assert set(defs["CircuitMember"].get("required", [])) == {"layer"}

    required_on_mirror = {
        n for n, f in CircuitDefinitionV1.model_fields.items() if f.is_required()
    }
    assert required_on_mirror == {"name", "saes", "members"}


def test_caps_match_contract(defs):
    """maxItems in the frozen contract must equal the mirror's caps."""
    assert defs["CircuitDefinitionV1"]["properties"]["saes"]["maxItems"] == MAX_SAES
    assert defs["CircuitDefinitionV1"]["properties"]["edges"]["maxItems"] == MAX_EDGES


def test_kind_and_version_literals_match(defs):
    """The kind discriminator and schema_version are the contract's identity."""
    props = defs["CircuitDefinitionV1"]["properties"]
    assert CIRCUIT_DEFINITION_KIND in json.dumps(props["kind"])
    assert CIRCUIT_SCHEMA_VERSION in json.dumps(props["schema_version"])


def test_rung_range_matches_evidence_ladder(defs):
    """rung is the 0..3 EvidenceRung ladder — the claims vocabulary."""
    rung_def = defs["EvidenceRung"]
    assert rung_def.get("enum") == [0, 1, 2, 3] or rung_def.get("type") == "integer"
    field = CircuitEdge.model_fields["rung"]
    # Mirror constrains to the same 0..3 window.
    meta = json.dumps(CircuitEdge.model_json_schema())
    assert '"maximum": 3' in meta and '"minimum": 0' in meta
    assert field.default == 0


def test_unknown_additive_fields_survive_round_trip():
    """extra='allow': a newer miStudio may emit additive fields (Tier-2.5
    position data etc.) and they MUST survive, or re-export is lossy."""
    doc = {
        "kind": CIRCUIT_DEFINITION_KIND,
        "schema_version": "1",
        "name": "c",
        "saes": [{"layer": 10, "n_features": 8192}],
        "members": [{"layer": 10, "feature": {"feature_idx": 1, "strength": 10.0}}],
        "future_top_level": {"hello": "world"},
        "edges": [
            {
                "up": {"layer": 10, "feature_idx": 1},
                "down": {"layer": 13, "feature_idx": 2},
                "future_edge_field": 42,
            }
        ],
    }
    parsed = CircuitDefinitionV1.model_validate(doc)
    dumped = parsed.model_dump(mode="json")
    assert dumped["future_top_level"] == {"hello": "world"}
    assert dumped["edges"][0]["future_edge_field"] == 42
