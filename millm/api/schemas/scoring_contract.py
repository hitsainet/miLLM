"""The published contract of `/v1/completions` SCORING, derived from the live app (FR-25.15).

Suite decision D61 (2026-10-18): miForge F11 R-02.2 asks this endpoint for float32 scoring and
refuses any answer that does not echo it (miForge FTDD §6.10). Its fake of this endpoint must be
checkable against what this process serves, the way its fake weights surface is checked against
`millm-weights-api-v1.json` (D39 follow-up). So this contract is GENERATED, never hand-written:

* `request` / `response` — the POST /v1/completions request body and 200 body schemas, from
  `register_routes` -> FastAPI's OpenAPI (refs rewritten to `$defs`);
* `scoring_precision` — the echo header, the refusal code with its HTTP status and OpenAI error
  type (read from `ERROR_STATUS_MAP`), the reason header and the reasons (read from
  `millm.core.errors`).

`docs/schemas/millm-scoring-api-v1.json` is its committed form, pinned by
`tests/unit/api/test_scoring_api_contract.py` (regenerate with `MILLM_WRITE_CONTRACTS=1`).
"""

from __future__ import annotations

from typing import Any

from millm.api.schemas.weights_contract import _rewrite

CONTRACT_FILENAME = "millm-scoring-api-v1.json"
PATH = "/v1/completions"


def scoring_api_contract() -> dict[str, Any]:
    from fastapi import FastAPI

    from millm.api.provenance import SCORING_PRECISION_HEADER
    from millm.api.routes import register_routes
    from millm.api.routes.openai.errors import ERROR_STATUS_MAP
    from millm.core.errors import SCORING_PRECISION_REASONS, ScoringPrecisionUnavailableError

    app = FastAPI()
    register_routes(app)
    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})
    operation = spec["paths"][PATH]["post"]

    reached: set[str] = set()
    request = _rewrite(
        operation["requestBody"]["content"]["application/json"]["schema"], reached
    )
    response = _rewrite(
        operation["responses"]["200"]["content"]["application/json"]["schema"], reached
    )
    defs: dict[str, Any] = {}
    pending = sorted(reached)
    while pending:
        name = pending.pop()
        if name in defs:
            continue
        more: set[str] = set()
        defs[name] = _rewrite(components[name], more)
        pending.extend(sorted(more - set(defs)))

    code = ScoringPrecisionUnavailableError.code
    status, error_type = ERROR_STATUS_MAP[code]
    reason_header = next(iter(ScoringPrecisionUnavailableError("busy", "x").headers or {}))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://hitsai.net/schemas/" + CONTRACT_FILENAME,
        "title": "miLLM /v1/completions scoring",
        "description": (
            "GENERATED from miLLM's registered POST /v1/completions route (its request and "
            "response models, via FastAPI's OpenAPI) and its error table; do not edit by hand. "
            "Scoring mode is `logprobs` and/or `allowed_token_ids` with max_tokens=1. "
            "`scoring_precision: \"float32\"` runs the forward over a float32 copy of the "
            "resident parameters; the response echoes the precision that RAN in the body "
            "field `scoring_precision` and the echo header, or the request is refused with "
            "the refusal code and one of the reasons, never answered in another precision."
        ),
        "path": PATH,
        "request": request,
        "response": response,
        "scoring_precision": {
            "values": ["native", "float32"],
            "echo_body_field": "scoring_precision",
            "echo_header": SCORING_PRECISION_HEADER,
            "refusal": {
                "code": code.lower(),
                "status": status,
                "type": error_type,
                "param": "scoring_precision",
                "reason_header": reason_header,
                "reasons": list(SCORING_PRECISION_REASONS),
            },
        },
        "$defs": dict(sorted(defs.items())),
    }
