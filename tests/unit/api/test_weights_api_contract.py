"""`docs/schemas/millm-weights-api-v1.json` is what the live routes declare, byte for byte.

The contract is GENERATED (`millm/api/schemas/weights_contract.py`) from `register_routes`'
OpenAPI, so it moves when a weights route or its response model moves - and this test goes red
until the committed file is regenerated (`MILLM_WRITE_CONTRACTS=1 pytest <this file>`), which
is the moment a consumer's vendored copy must move too (miForge vendors it and validates its fake
weights surface against it, suite decision D39 follow-up, 2026-10-17).
"""

from __future__ import annotations

import os
from pathlib import Path

from millm.api.schemas.weights_contract import CONTRACT_FILENAME, render, weights_api_contract

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "docs" / "schemas" / CONTRACT_FILENAME


def test_the_committed_contract_is_the_generated_one():
    generated = render(weights_api_contract())
    if os.environ.get("MILLM_WRITE_CONTRACTS") == "1":
        CONTRACT.write_text(generated, encoding="utf-8")
    assert CONTRACT.is_file(), f"{CONTRACT} is missing; run with MILLM_WRITE_CONTRACTS=1"
    assert CONTRACT.read_text(encoding="utf-8") == generated, (
        f"{CONTRACT.name} no longer matches the registered /api/weights/* routes. Regenerate "
        f"it (MILLM_WRITE_CONTRACTS=1) and re-vendor it into miForge's docs/schemas/."
    )


def test_it_covers_the_routes_a_trainer_calls_and_is_not_empty():
    """A generator that silently matched nothing would pin an empty file; name the routes."""
    responses = weights_api_contract()["responses"]
    for route in (
        "POST /api/weights/attach",
        "POST /api/weights/detach",
        "POST /api/weights/heartbeat",
        "POST /api/weights/quiesce",
        "POST /api/weights/resume",
        "POST /api/weights/commit-begin",
        "POST /api/weights/commit-end",
        "POST /api/weights/save",
        "POST /api/weights/load",
        "GET /api/weights/status",
    ):
        assert route in responses, route


def _payload_def(doc: dict, route: str) -> dict:
    """The `data` model of a route's envelope, by FOLLOWING the refs (def names are
    qualified by FastAPI when two modules share a class name, e.g. SAE vs weights attach)."""
    defs = doc["$defs"]

    def deref(node: dict) -> dict:
        while "$ref" in node:
            node = defs[node["$ref"].removeprefix("#/$defs/")]
        return node

    envelope = deref(doc["responses"][route])
    assert {"success", "data", "error"} <= set(envelope["properties"]), envelope
    data = envelope["properties"]["data"]
    options = [deref(o) for o in data.get("anyOf", [data]) if o.get("type") != "null"]
    assert len(options) == 1, options
    return options[0]


def test_the_attach_payload_states_storages_parameters_and_config():
    """The three facts miForge's fake got wrong (D39): handles on storages, parameters as
    views, and the served config - present in the contract a consumer validates against."""
    attach = _payload_def(weights_api_contract(), "POST /api/weights/attach")
    assert {"storages", "parameters", "config", "attachment_id"} <= set(attach["properties"])
    assert attach.get("additionalProperties") is False
