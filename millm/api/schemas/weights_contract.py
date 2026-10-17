"""The published response contract of `/api/weights/*`, DERIVED from the live route registry.

Suite decision D39 follow-up (2026-10-17): miForge's fake weights surface had drifted from this
surface on four counts (no envelope, handles per parameter, an invented status shape, no
`config`), and every miForge test passed against the fiction. The cure is a contract a consumer
can check its double against that cannot drift from what this process serves: it is generated
from `register_routes` -> FastAPI's OpenAPI -> each `/api/weights/*` route's declared
`response_model`, never written by hand. `docs/schemas/millm-weights-api-v1.json` is its
committed form, pinned by `tests/unit/api/test_weights_api_contract.py` (regenerate with
`MILLM_WRITE_CONTRACTS=1`). miForge vendors it byte-identical and validates its fake against it.
"""

from __future__ import annotations

import json
from typing import Any

CONTRACT_FILENAME = "millm-weights-api-v1.json"
_PREFIX = "/api/weights/"
_OPENAPI_REF = "#/components/schemas/"
_JSON_SCHEMA_REF = "#/$defs/"


def _rewrite(node: Any, found: set[str]) -> Any:
    """OpenAPI component refs -> JSON Schema `$defs` refs, collecting every name reached."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str) and value.startswith(_OPENAPI_REF):
                name = value[len(_OPENAPI_REF):]
                found.add(name)
                out[key] = _JSON_SCHEMA_REF + name
            else:
                out[key] = _rewrite(value, found)
        return out
    if isinstance(node, list):
        return [_rewrite(item, found) for item in node]
    return node


def weights_api_contract() -> dict[str, Any]:
    """The success-response schema of every registered `/api/weights/*` route, by route."""
    from fastapi import FastAPI

    from millm.api.routes import register_routes

    app = FastAPI()
    register_routes(app)
    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})

    responses: dict[str, Any] = {}
    reached: set[str] = set()
    for path, methods in sorted(spec["paths"].items()):
        if not path.startswith(_PREFIX):
            continue
        for method, operation in sorted(methods.items()):
            ok = next(
                (code for code in sorted(operation.get("responses", {})) if code.startswith("2")),
                None,
            )
            if ok is None:
                continue
            content = operation["responses"][ok].get("content", {}).get("application/json")
            if not content or "schema" not in content:
                continue
            responses[f"{method.upper()} {path}"] = _rewrite(content["schema"], reached)

    defs: dict[str, Any] = {}
    pending = sorted(reached)
    while pending:
        name = pending.pop()
        if name in defs:
            continue
        more: set[str] = set()
        defs[name] = _rewrite(components[name], more)
        pending.extend(sorted(more - set(defs)))

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://hitsai.net/schemas/" + CONTRACT_FILENAME,
        "title": "miLLM weights API responses",
        "description": (
            "GENERATED from miLLM's registered /api/weights/* routes (their response_model "
            "declarations, via FastAPI's OpenAPI); do not edit by hand. Every route answers in "
            "the management envelope {success, data, error}. Keys of `responses` are "
            "'<METHOD> <path>'; each value is that route's success-response schema."
        ),
        "responses": responses,
        "$defs": dict(sorted(defs.items())),
    }


def render(contract: dict[str, Any]) -> str:
    """The committed byte form: sorted keys, two-space indent, trailing newline."""
    return json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
