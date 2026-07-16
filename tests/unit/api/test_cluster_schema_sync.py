"""
Schema-sync test (Feature 8, Task 2.3): the pydantic mirror in
millm/api/schemas/cluster.py must regenerate EXACTLY the vendored frozen
contract at docs/schemas/cluster-definition-v1.json.

If this fails, the mirror drifted from the frozen v1 contract — fix the
mirror, never the vendored file (v1 is frozen; changes require a v2).
"""

import json
from pathlib import Path

from millm.api.schemas.cluster import ClusterBundleV1, ClusterDefinitionV1

VENDORED = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "cluster-definition-v1.json"


def _generate() -> dict:
    # Wrapper structure matches the miStudio generator byte-for-byte.
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://mistudio.hitsai.net/schemas/cluster-definition-v1.json",
        "title": "miStudio Cluster Definition v1",
        "description": (
            "Portable cluster definition (mistudio.cluster-definition/v1) — the consumer-neutral "
            "interchange artifact for cluster profiles (IDL-30). Also includes the bundle wrapper "
            "(mistudio.cluster-bundle/v1) under $defs. Generated from the pydantic contract; "
            "regenerate via backend/tests/unit/test_cluster_definition_schema_sync.py instructions."
        ),
        "oneOf": [
            {"$ref": "#/$defs/ClusterDefinitionV1"},
            {"$ref": "#/$defs/ClusterBundleV1"},
        ],
        "$defs": {},
    }
    def_schema = ClusterDefinitionV1.model_json_schema(ref_template="#/$defs/{model}")
    bundle_schema = ClusterBundleV1.model_json_schema(ref_template="#/$defs/{model}")
    for s in (def_schema, bundle_schema):
        defs = s.pop("$defs", {})
        schema["$defs"].update(defs)
    schema["$defs"]["ClusterDefinitionV1"] = def_schema
    schema["$defs"]["ClusterBundleV1"] = bundle_schema
    return schema


def test_mirror_matches_vendored_contract() -> None:
    assert VENDORED.exists(), f"Vendored schema missing: {VENDORED}"
    vendored = json.loads(VENDORED.read_text())
    assert _generate() == vendored, (
        "millm/api/schemas/cluster.py drifted from the frozen v1 contract "
        "(docs/schemas/cluster-definition-v1.json). Fix the mirror, not the vendored file."
    )
