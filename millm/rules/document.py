"""A rule-set document: size, schema, semantic checks and identity (039 FR-39.17, FR-39.21; TD39-25).

The document is miDataworks' (`docs/schemas/midataworks-rule-set-v1.json`, vendored byte-identically
and pinned by `tests/unit/rules/test_vendored_rule_contracts.py`). miLLM validates it with
`jsonschema` against the vendored copy and NEVER mirrors it as a model (TD39-25): the producer owns
every field, and a hand-written mirror would silently drop the next one it adds (suite memory
`circuit-contract-mirror-drift`).

Checks, in order (each a refusal `RuleSetInvalid` with a stable `code` and a JSON pointer or a name,
never the document's text):

1. size (`RULE_SET_MAX_BYTES`) BEFORE parsing — the route passes the raw bytes;
2. JSON with no NaN/Infinity and no repeated key;
3. the vendored JSON Schema (Draft 2020-12);
4. every string NFC (`not_nfc`), every integer within ±(2^53 − 1) (`unsafe_integer`), and
   `provenance` absent or an object, never null — the producer's before-validators;
5. references: unique ids, `anchor`/`span_of` name an EARLIER signal, expressions name known
   signals within depth, outputs name known expressions with the matching type, vetoes and
   renderings name known names, every case states every output column (the producer's
   `_references`, ported over dicts).

Identity (the producer's D38-4): the lower-case hex SHA-256 of the RFC 8785 (JCS) bytes of the
document AS RECEIVED — parsed values, never re-serialised through a model; an omitted list and an
empty list differ. Pinned by the vendored `rule_set_identity` cases.
"""

from __future__ import annotations

import functools
import hashlib
import json
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import rfc8785
from jsonschema import Draft202012Validator

#: The vendored producer schema (the path pattern of `services/adapter_manifest.py`; the image
#: copies `docs/schemas/`).
VENDORED_SCHEMA = (
    Path(__file__).resolve().parents[2] / "docs" / "schemas" / "midataworks-rule-set-v1.json"
)
FORMAT = "midataworks.rule-set/v1"
SAFE_INTEGER = 2**53 - 1
#: The producer's `MAX_EXPRESSION_DEPTH`.
MAX_EXPRESSION_DEPTH = 8
COMPOSITION_FORMS = ("all", "any", "not", "within_n")


class RuleSetInvalid(ValueError):
    """The document is refused. `code` is stable; `details` names where, never the text."""

    def __init__(self, code: str, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@functools.lru_cache(maxsize=1)
def schema() -> dict[str, Any]:
    return json.loads(VENDORED_SCHEMA.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(schema())


def signal_kinds() -> tuple[str, ...]:
    """Every signal `kind` the vendored schema declares (the forms registry is derived from it)."""
    defs = schema()["$defs"]
    kinds = []
    for name, node in defs.items():
        if name.endswith("Signal") and "kind" in node.get("properties", {}):
            kind = node["properties"]["kind"].get("const")
            if kind:
                kinds.append(kind)
    return tuple(sorted(kinds))


# ── parsing ───────────────────────────────────────────────────────────────────────────────


def _no_constants(token: str) -> Any:
    raise RuleSetInvalid("rule_set_invalid", f"{token} is not JSON", {"token": token})


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RuleSetInvalid("rule_set_invalid", "a key is repeated in one object",
                                 {"key": key})
        out[key] = value
    return out


def parse_bytes(raw: bytes, max_bytes: int) -> Any:
    """Size first, then strict JSON. Raises `RuleSetInvalid`."""
    if len(raw) > max_bytes:
        raise RuleSetInvalid("rule_set_too_large",
                             f"the document is {len(raw)} bytes; the limit is {max_bytes}",
                             {"bytes": len(raw), "limit": max_bytes,
                              "setting": "RULE_SET_MAX_BYTES"})
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_no_constants,
                          object_pairs_hook=_unique_pairs)
    except RuleSetInvalid:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuleSetInvalid("rule_set_invalid", "the document is not UTF-8 JSON",
                             {"position": getattr(exc, "pos", None)}) from None


# ── the producer's before-validators ──────────────────────────────────────────────────────


def _escape(key: Any) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def _walk_strings(value: Any, pointer: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield pointer, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield pointer + "/" + _escape(key), str(key)
            yield from _walk_strings(item, pointer + "/" + _escape(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _walk_strings(item, f"{pointer}/{i}")


def _walk_integers(value: Any, pointer: str = "") -> Iterator[tuple[str, int]]:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield pointer, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_integers(item, pointer + "/" + _escape(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _walk_integers(item, f"{pointer}/{i}")


def first_non_nfc(document: Any) -> Optional[str]:
    for pointer, text in _walk_strings(document):
        if not unicodedata.is_normalized("NFC", text):
            return pointer
    return None


def first_unsafe_integer(document: Any) -> Optional[str]:
    for pointer, number in _walk_integers(document):
        if abs(number) > SAFE_INTEGER:
            return pointer
    return None


# ── identity ──────────────────────────────────────────────────────────────────────────────


def canonical_bytes(document: Any) -> bytes:
    """RFC 8785 (JCS) bytes of the document as received."""
    return rfc8785.dumps(document)


def identity_digest(document: Any) -> str:
    """The rule-set identity: lower-case hex SHA-256 of `canonical_bytes` (the producer's D38-4)."""
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


# ── expressions ───────────────────────────────────────────────────────────────────────────


def expression_form(expr: dict[str, Any]) -> str:
    """`signal`, `all`, `any`, `not` or `within_n` (the one key an expression object has)."""
    return next(iter(expr))


def expression_depth(expr: dict[str, Any]) -> int:
    form = expression_form(expr)
    if form == "signal":
        return 1
    if form in ("all", "any"):
        return 1 + max(expression_depth(e) for e in expr[form])
    if form == "not":
        return 1 + expression_depth(expr["not"])
    body = expr["within_n"]
    return 1 + max(expression_depth(body["a"]), expression_depth(body["b"]))


def signals_named(expr: dict[str, Any]) -> list[str]:
    form = expression_form(expr)
    if form == "signal":
        return [expr["signal"]]
    if form in ("all", "any"):
        return [s for e in expr[form] for s in signals_named(e)]
    if form == "not":
        return signals_named(expr["not"])
    body = expr["within_n"]
    return signals_named(body["a"]) + signals_named(body["b"])


def output_value_key(value: dict[str, Any]) -> str:
    """`fired`, `count`, `first`, `spans` or `field`."""
    return next(iter(value))


_VALUE_TYPE = {"fired": "bool", "count": "int", "first": "string", "spans": "string"}


def signal_reference(signal: dict[str, Any]) -> Optional[str]:
    """The earlier signal a signal reads (`anchor` or `span_of`), or None."""
    if signal["kind"] == "proximity":
        return signal["anchor"]
    if signal["kind"] == "validator":
        return (signal.get("candidates") or {}).get("span_of")
    return None


# ── the check ─────────────────────────────────────────────────────────────────────────────


def _unique(items: list[str], what: str) -> None:
    seen: set[str] = set()
    for item in items:
        if item in seen:
            raise RuleSetInvalid("rule_set_invalid", f"{what} {item!r} is repeated",
                                 {"what": what, "name": item})
        seen.add(item)


def _references(doc: dict[str, Any]) -> None:
    signals = doc["signals"]
    _unique([s["id"] for s in signals], "signal id")
    earlier: set[str] = set()
    for signal in signals:
        ref = signal_reference(signal)
        if ref is not None and ref not in earlier:
            raise RuleSetInvalid("rule_set_invalid",
                                 f"signal {signal['id']!r} refers to {ref!r}, which is not an "
                                 "earlier signal", {"signal": signal["id"], "refers_to": ref})
        if signal["kind"] == "lexical_weights":
            _unique([t["term"] for t in signal["terms"]], f"term of {signal['id']!r}")
        earlier.add(signal["id"])
    expressions = doc["expressions"]
    for name, expr in expressions.items():
        if expression_depth(expr) > MAX_EXPRESSION_DEPTH:
            raise RuleSetInvalid("rule_set_invalid",
                                 f"expression {name!r} nests deeper than {MAX_EXPRESSION_DEPTH}",
                                 {"expression": name})
        for s in signals_named(expr):
            if s not in earlier:
                raise RuleSetInvalid("rule_set_invalid",
                                     f"expression {name!r} names an unknown signal {s!r}",
                                     {"expression": name, "signal": s})
    columns = [o["column"] for o in doc["outputs"]]
    _unique(columns, "output column")
    for out in doc["outputs"]:
        key = output_value_key(out["value"])
        if key == "field":
            sig = out["value"]["field"][0]
            if sig not in earlier:
                raise RuleSetInvalid("rule_set_invalid",
                                     f"output {out['column']!r} reads an unknown signal {sig!r}",
                                     {"column": out["column"], "signal": sig})
        else:
            if out["value"][key] not in expressions:
                raise RuleSetInvalid("rule_set_invalid",
                                     f"output {out['column']!r} names an unknown expression",
                                     {"column": out["column"]})
            if _VALUE_TYPE[key] != out["type"]:
                raise RuleSetInvalid("rule_set_invalid",
                                     f"output {out['column']!r}: {key} is a {_VALUE_TYPE[key]}, "
                                     f"not {out['type']}", {"column": out["column"]})
    vetoes = doc.get("vetoes") or []
    _unique([v["id"] for v in vetoes], "veto id")
    for veto in vetoes:
        if veto["when"] not in expressions:
            raise RuleSetInvalid("rule_set_invalid",
                                 f"veto {veto['id']!r} names an unknown expression",
                                 {"veto": veto["id"]})
        for s in veto["removes"]:
            if s not in earlier:
                raise RuleSetInvalid("rule_set_invalid",
                                     f"veto {veto['id']!r} removes an unknown signal {s!r}",
                                     {"veto": veto["id"], "signal": s})
    renderings = doc.get("renderings") or []
    _unique([r["rendering"] for r in renderings], "rendering")
    for rendering in renderings:
        for s in rendering.get("invisible_to") or []:
            if s not in earlier:
                raise RuleSetInvalid("rule_set_invalid",
                                     f"rendering {rendering['rendering']!r}: unknown signal {s!r}",
                                     {"rendering": rendering["rendering"], "signal": s})
    decoys = doc.get("decoys") or []
    _unique([d["id"] for d in decoys], "decoy id")
    for decoy in decoys:
        for c in decoy.get("must_not_fire") or []:
            if c not in columns:
                raise RuleSetInvalid("rule_set_invalid",
                                     f"decoy {decoy['id']!r}: unknown output column {c!r}",
                                     {"decoy": decoy["id"], "column": c})
    _unique([s["id"] for s in doc.get("substitutions") or []], "substitution id")
    cases = doc["cases"]
    _unique([c["id"] for c in cases], "case id")
    for index, case in enumerate(cases):
        if set(case["expect"]["outputs"]) != set(columns):
            raise RuleSetInvalid("rule_set_invalid",
                                 f"case {case['id']!r} must state every output column",
                                 {"case": case["id"], "index": index, "columns": columns})
        for name in case["expect"].get("spans") or {}:
            if name not in expressions:
                raise RuleSetInvalid("rule_set_invalid",
                                     f"case {case['id']!r} states spans of an unknown name",
                                     {"case": case["id"], "index": index, "name": name})


def check(document: Any) -> dict[str, Any]:
    """Run checks 3–5 on a parsed document; return it unchanged. Raises `RuleSetInvalid`."""
    if isinstance(document, dict):
        pointer = first_non_nfc(document)
        if pointer is not None:
            raise RuleSetInvalid("not_nfc", "a string is not Unicode NFC; normalise it first",
                                 {"pointer": pointer})
        pointer = first_unsafe_integer(document)
        if pointer is not None:
            raise RuleSetInvalid("unsafe_integer",
                                 "an integer is outside ±(2^53 − 1); RFC 8785 cannot write it",
                                 {"pointer": pointer})
        if "provenance" in document and document["provenance"] is None:
            raise RuleSetInvalid("rule_set_invalid",
                                 "provenance is absent or an object, never null",
                                 {"pointer": "/provenance"})
    errors = sorted(_validator().iter_errors(document), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        pointer = "".join(f"/{_escape(p)}" for p in first.absolute_path)
        raise RuleSetInvalid("rule_set_invalid", "the document does not match the rule-set "
                             "schema", {"pointer": pointer or "/", "rule": first.validator})
    _references(document)
    return document


def origin_of(document: dict[str, Any]) -> str:
    """`derived` when the document carries provenance, else `authored` (FR-39.21). The evaluator
    never reads this."""
    return "derived" if document.get("provenance") is not None else "authored"


#: Fields that serve dataset building, stored and digested but never used at serving (OQ-39.2 (a)).
NOT_USED_AT_SERVING = ("vetoes", "renderings", "decoys", "substitutions", "outputs")
