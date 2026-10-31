"""The isolated runner contract, read from its vendored copy (039 S3; FR-39.26; TD39-33, TD39-34).

`docs/schemas/midataworks-runner-v1.json` is miDataworks 018's file, vendored byte-identically
(`tests/unit/rules/test_vendored_runner_contracts.py`). Every value a host needs — the floor, the
reasons, the count cap, the hash rule, the guest bundle's digests — is READ from it here; nothing
is restated, so a re-vendored contract moves miLLM with it.

⚠ STANDARD LIBRARY ONLY. Both sides import this module: the API process (import checks, the
client's limits) and the `rule-runner` worker. It must never import `wasmtime` (the API process
never loads the engine, FR-39.26) or anything that reads settings, a database or a secret.

Hash rule (the contract's `hash_rule`): `code_sha256` = SHA-256 of the canonical JSON of the
code-rule document — keys sorted by code point, `,`/`:` separators, UTF-8 with non-ASCII as itself,
NaN and infinity refused. The bytes a caller sends are exactly the bytes hashed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The vendored contract (the image copies `docs/schemas/`, `Dockerfile`).
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "midataworks-runner-v1.json"
#: The vendored shared vectors (tests read them; the worker never does).
CASES_PATH = CONTRACT_PATH.with_name("runner-cases-v1.json")

#: The keys a code-rule document has, exactly (the contract's top-level `required`).
DOCUMENT_KEYS = frozenset({"runner_contract", "source", "functions", "params_schema", "derived_from"})

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


@functools.lru_cache(maxsize=1)
def contract() -> dict[str, Any]:
    """The vendored contract, parsed once per process."""
    data: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return data


def contract_format() -> str:
    return str(contract()["format"])


def floor() -> dict[str, int]:
    """`limits_floor`: what every host grants at least (a deployment may raise, never lower)."""
    return {k: int(v) for k, v in contract()["limits_floor"].items()}


def reasons() -> tuple[str, ...]:
    return tuple(contract()["reasons"])


def count_cap() -> int:
    return int(contract()["count_cap"])


def snapshot() -> dict[str, Any]:
    """`guest.bundle.snapshot`: the published guest bytes' url, sha256 and size."""
    return dict(contract()["guest"]["bundle"]["snapshot"])


# ── the hash rule ────────────────────────────────────────────────────────────────────────


def canonical_json(value: Any) -> bytes:
    """The contract's canonical JSON (one serialiser for hashing and sending)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_sha256(document: Mapping[str, Any]) -> str:
    return bytes_sha256(canonical_json(dict(document)))


def bundle_digest(bundle: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of `guest.bundle` without its own `bundle_digest`."""
    return bytes_sha256(canonical_json({k: v for k, v in bundle.items() if k != "bundle_digest"}))


def replace_lone_surrogates(text: str) -> str:
    """The caller's half of `text_rules`: every lone surrogate becomes U+FFFD before encoding."""
    return _LONE_SURROGATE.sub("�", text)


# ── a code-rule document as bytes ────────────────────────────────────────────────────────


class CodeDocumentRefused(ValueError):
    """A code-rule document that is not what it claims. `code` is stable; the message names no
    source text."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CodeDocument:
    code_sha256: str
    source: str
    functions: frozenset[str]
    params_schema: dict[str, Any]


def parse_code_document(code_sha256_hex: str, canonical_utf8: str, *,
                        max_source_bytes: int = 1 << 20) -> CodeDocument:
    """Check one code-rule document's bytes against the hash it is filed under.

    The bytes must hash to `code_sha256_hex`, be a document of exactly the contract's keys, be
    the canonical bytes themselves (a re-serialised copy would hash differently elsewhere), and
    name this contract's format. Raises `CodeDocumentRefused`.
    """
    if not isinstance(code_sha256_hex, str) or not _HEX64.match(code_sha256_hex):
        raise CodeDocumentRefused("code_hash_invalid", "the hash is not 64 lower-case hex digits")
    if not isinstance(canonical_utf8, str):
        raise CodeDocumentRefused("code_document_invalid", "the document is not a string")
    try:
        raw = canonical_utf8.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise CodeDocumentRefused("code_document_invalid", "the document is not UTF-8") from None
    if bytes_sha256(raw) != code_sha256_hex:
        raise CodeDocumentRefused("code_hash_mismatch", "the bytes do not hash to the declared hash")
    try:
        doc: Any = json.loads(raw)
    except ValueError:
        raise CodeDocumentRefused("code_document_invalid", "the document is not JSON") from None
    if not isinstance(doc, dict) or set(doc) != DOCUMENT_KEYS:
        raise CodeDocumentRefused("code_document_invalid", "not a code-rule document")
    try:
        if canonical_json(doc) != raw:
            raise CodeDocumentRefused("code_document_invalid", "not the canonical bytes")
    except ValueError:
        raise CodeDocumentRefused("code_document_invalid", "not the canonical bytes") from None
    if doc["runner_contract"] != contract_format():
        raise CodeDocumentRefused("runner_contract_unknown",
                                  "this miLLM runs " + contract_format())
    source, functions = doc["source"], doc["functions"]
    if not isinstance(source, str) or not source or len(source.encode("utf-8")) > max_source_bytes:
        raise CodeDocumentRefused("code_document_invalid", "source missing or over the limit")
    if not isinstance(functions, list) or not functions or not all(
        isinstance(f, dict) and isinstance(f.get("name"), str) for f in functions
    ):
        raise CodeDocumentRefused("code_document_invalid", "functions malformed")
    params_schema = doc["params_schema"] if isinstance(doc["params_schema"], dict) else {}
    return CodeDocument(code_sha256_hex, source, frozenset(f["name"] for f in functions),
                        params_schema)


__all__ = [
    "CASES_PATH", "CONTRACT_PATH", "CodeDocument", "CodeDocumentRefused", "bundle_digest",
    "bytes_sha256", "canonical_json", "code_sha256", "contract", "contract_format", "count_cap",
    "floor", "parse_code_document", "reasons", "replace_lone_surrogates", "snapshot",
]
