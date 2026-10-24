"""`docs/schemas/millm-scoring-api-v1.json` is what the live /v1/completions route declares.

GENERATED (`millm/api/schemas/scoring_contract.py`) from `register_routes`' OpenAPI and the error
table, so it moves when the scoring request or response moves — and this test goes red until the
committed file is regenerated (`MILLM_WRITE_CONTRACTS=1 pytest <this file>`), which is the moment
miForge's vendored copy must move too (FR-25.15; D61; miForge FTDD §6.10).
"""

from __future__ import annotations

import os
from pathlib import Path

from millm.api.schemas.scoring_contract import CONTRACT_FILENAME, scoring_api_contract
from millm.api.schemas.weights_contract import render

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "docs" / "schemas" / CONTRACT_FILENAME


def test_the_committed_contract_is_the_generated_one():
    generated = render(scoring_api_contract())
    if os.environ.get("MILLM_WRITE_CONTRACTS") == "1":
        CONTRACT.write_text(generated, encoding="utf-8")
    assert CONTRACT.is_file(), f"{CONTRACT} is missing; run with MILLM_WRITE_CONTRACTS=1"
    assert CONTRACT.read_text(encoding="utf-8") == generated, (
        f"{CONTRACT.name} no longer matches POST /v1/completions. Regenerate it "
        "(MILLM_WRITE_CONTRACTS=1) and re-vendor it into miForge's docs/schemas/."
    )


def _deref(doc: dict, node: dict) -> dict:
    while "$ref" in node:
        node = doc["$defs"][node["$ref"].removeprefix("#/$defs/")]
    return node


def test_the_request_and_the_echo_carry_scoring_precision():
    """The two fields miForge relies on, present in what a consumer validates against."""
    doc = scoring_api_contract()
    request = _deref(doc, doc["request"])
    response = _deref(doc, doc["response"])
    for schema in (request, response):
        field = schema["properties"]["scoring_precision"]
        values = {v for option in field.get("anyOf", [field]) for v in option.get("enum", [])}
        assert values == {"native", "float32"}, field
    assert "scoring_precision" not in response.get("required", [])
    assert {"logprobs", "allowed_token_ids", "max_tokens"} <= set(request["properties"])


def test_the_refusal_is_read_from_the_error_table():
    refusal = scoring_api_contract()["scoring_precision"]["refusal"]
    assert refusal["code"] == "scoring_precision_unavailable"
    assert (refusal["status"], refusal["type"]) == (409, "invalid_request_error")
    assert refusal["reason_header"] == "X-miLLM-Scoring-Precision-Reason"
    assert set(refusal["reasons"]) == {"memory", "device", "model_not_float32_castable", "busy"}
    assert scoring_api_contract()["scoring_precision"]["echo_header"] == "X-miLLM-Scoring-Precision"
