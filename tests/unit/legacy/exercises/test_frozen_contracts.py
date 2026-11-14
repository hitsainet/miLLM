"""Exercises for DR-04 (schema `$id` URLs) and DR-05 (contract file names) — 040 FR-040.7.

A published schema's `$id` is how a consumer names it, and the two generated contracts are vendored
byte-identical by miForge and miStudio under these file names. The rebrand changes neither.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCHEMAS = REPO / "docs" / "schemas"

#: Every `$id` under `docs/schemas/` that carries a DR-04 literal, pinned by value.
FROZEN_IDS = {
    "circuit-definition-v1.json":
        "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/circuit-definition-v1.json",
    "cluster-definition-v1.json":
        "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/cluster-definition-v1.json",
    "jspace-contracts-v1.json":
        "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/jspace-contracts-v1.json",
    "millm-scoring-api-v1.json": "https://hitsai.net/schemas/millm-scoring-api-v1.json",
    "millm-weights-api-v1.json": "https://hitsai.net/schemas/millm-weights-api-v1.json",
    "probe-definition-v1.json":
        "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/probe-definition-v1.json",
    "probe-result-v1.json": "https://hitsai.net/schemas/millm/probe-result-v1.json",
}
DR04 = ("raw.githubusercontent.com/hitsainet/miStudio/", "hitsai.net/schemas/millm")


def test_schema_ids_are_unchanged():
    found = {}
    for path in sorted(SCHEMAS.glob("*.json")):
        doc = json.loads(path.read_text())
        if isinstance(doc, dict) and any(lit in str(doc.get("$id", "")) for lit in DR04):
            found[path.name] = doc["$id"]
    assert found == FROZEN_IDS


def test_contracts_regenerate_at_frozen_file_names():
    from miserver.api.schemas import scoring_contract, weights_contract

    assert scoring_contract.CONTRACT_FILENAME == "millm-scoring-api-v1.json"
    assert weights_contract.CONTRACT_FILENAME == "millm-weights-api-v1.json"
    for module, build in ((scoring_contract, scoring_contract.scoring_api_contract),
                          (weights_contract, weights_contract.weights_api_contract)):
        committed = SCHEMAS / module.CONTRACT_FILENAME
        assert committed.is_file(), committed
        assert committed.read_text(encoding="utf-8") == weights_contract.render(build()), committed.name
