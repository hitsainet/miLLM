"""The JSON Schema subset `response_format: json_schema` accepts (Feature 25, FR-25.10.9, FR-25.11.2).

THE GUARANTEE: a schema is either inside a subset whose every keyword the installed constrained
decoder ENFORCES, or it is refused before any model load, naming the keyword and its JSON pointer.

THE DEFECT IT PREVENTS: xgrammar 0.2.8 compiles `multipleOf`, `not` and `uniqueItems` and does
NOT enforce them (measured, 025_FTDD §3.1) — a violating string is accepted. A schema using one
would produce output that "conforms" to a constraint nobody applied. The subset is therefore
miLLM's own allowlist, not "whatever the library compiles", and
`tests/unit/api/test_json_schema_subset.py` probes the INSTALLED library for every allowlisted
keyword: a keyword joins `ALLOWED_KEYWORDS` only with a violating example the library rejects. A
second line of defence validates every complete output with `jsonschema`
(`millm/ml/constrained_decoding.py::validate_output`).

Pure, no imports from `services` or `ml`; runs in the route between the request policy and the
auto-load.
"""

from __future__ import annotations

import json
from typing import Any

from millm.core.errors import ResponseFormatUnsupportedError

#: Constraint keywords (025_FTDD §4.3). Each has an enforcement probe in the test suite.
ALLOWED_KEYWORDS: frozenset[str] = frozenset({
    "type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
    "enum", "const", "minimum", "maximum", "minLength", "maxLength", "pattern", "anyOf",
    "$ref", "$defs",
})
#: Annotations: no constraint, so nothing to enforce; their values are not walked.
ANNOTATIONS: frozenset[str] = frozenset({"title", "description", "default", "examples"})

JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})

MAX_SCHEMA_BYTES = 64 * 1024
MAX_DEPTH = 32
MAX_PROPERTIES = 256
MAX_REPORTED = 20


def _pointer(path: list[str]) -> str:
    return "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in path) if path else ""


def _walk(node: Any, path: list[str], depth: int, out: list[tuple[str, str]]) -> None:
    if len(out) >= MAX_REPORTED:
        return
    if depth > MAX_DEPTH:
        out.append(("(depth)", _pointer(path)))
        return
    if isinstance(node, bool):
        return  # `true` / `false` schemas
    if not isinstance(node, dict):
        out.append(("(not a schema)", _pointer(path)))
        return
    for key, value in node.items():
        here = path + [key]
        if key in ANNOTATIONS:
            continue
        if key not in ALLOWED_KEYWORDS:
            out.append((key, _pointer(here)))
            continue
        if key == "type":
            kinds = value if isinstance(value, list) else [value]
            if not kinds or any(not isinstance(k, str) or k not in JSON_TYPES for k in kinds):
                out.append(("type", _pointer(here)))
        elif key in ("properties", "$defs"):
            if not isinstance(value, dict):
                out.append((key, _pointer(here)))
                continue
            if key == "properties" and len(value) > MAX_PROPERTIES:
                out.append(("properties (more than 256)", _pointer(here)))
                continue
            for name, sub in value.items():
                _walk(sub, here + [name], depth + 1, out)
        elif key in ("items", "additionalProperties"):
            # `items` as a LIST is the draft-4 tuple form, which the subset does not include.
            if key == "items" and isinstance(value, list):
                out.append(("items (tuple form)", _pointer(here)))
            else:
                _walk(value, here, depth + 1, out)
        elif key == "anyOf":
            if not isinstance(value, list) or not value:
                out.append(("anyOf", _pointer(here)))
                continue
            for i, sub in enumerate(value):
                _walk(sub, here + [str(i)], depth + 1, out)
        elif key == "$ref":
            if not isinstance(value, str) or not value.startswith("#/$defs/"):
                out.append(("$ref (only local #/$defs/ references)", _pointer(here)))
        elif key == "required":
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                out.append(("required", _pointer(here)))
        # enum, const, minimum, maximum, minLength, maxLength, minItems, maxItems, pattern:
        # values, not schemas — nothing to walk.


def offending_keywords(schema: Any) -> list[tuple[str, str]]:
    """Every (keyword, JSON pointer) outside the subset, up to 20."""
    out: list[tuple[str, str]] = []
    _walk(schema, [], 0, out)
    return out


def check(schema: Any) -> None:
    """Refuse a schema outside the subset or over the size caps.

    Raises:
        ResponseFormatUnsupportedError: naming the first keyword and listing up to 20, each with
            its JSON pointer (`details["keywords"]`).
    """
    try:
        size = len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ResponseFormatUnsupportedError(
            f"response_format schema is not serialisable JSON: {exc}",
            details={"param": "response_format"},
        ) from exc
    if size > MAX_SCHEMA_BYTES:
        raise ResponseFormatUnsupportedError(
            f"response_format schema is {size} bytes; the limit is {MAX_SCHEMA_BYTES}",
            details={"param": "response_format", "schema_bytes": size},
        )
    offenders = offending_keywords(schema)
    if offenders:
        listed = ", ".join(f"{k} at {p or '/'}" for k, p in offenders)
        raise ResponseFormatUnsupportedError(
            f"response_format schema uses what the supported JSON Schema subset does not "
            f"include: {listed}. Supported keywords: {', '.join(sorted(ALLOWED_KEYWORDS))}; "
            f"annotations: {', '.join(sorted(ANNOTATIONS))}.",
            details={
                "param": "response_format",
                "keywords": [{"keyword": k, "pointer": p} for k, p in offenders],
            },
        )
