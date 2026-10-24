"""Detector and decision digests, computed by miLLM (FR-38.7, D38.26 – D38.29).

The algorithm is miStudio 039's `mistudio.probe-identity/1`, which the PRODUCER owns: SHA-256 over
the RFC 8785 (JCS) canonical bytes of a wrapper `{"algorithm", "digest": "detector"|"decision",
"payload"}`. The detector payload is an INCLUSIVE projection of the definition (named keys only);
the decision payload projects the `decision` block the same way.

⚠ THE AUTHORITY IS miSTUDIO's CASE FILE, NOT THIS MODULE. `docs/schemas/probe-identity-cases.json`
is vendored byte-identically (R-039, FTASKS 11.1) and every case — documents, canonical texts,
digests, number vectors and the published `projection` tables — is recomputed against this module
by `tests/unit/services/test_probe_identity_cases.py`. The rules below are miStudio's
`backend/src/schemas/probe_identity.py::_project`, line for line:

* the projection is a table keyed by PATH (`""` = the root; `x[]` each list element, `x{}` each map
  value); only listed keys are hashed, and every listed key is present;
* an ABSENT key is written as its `DEFAULTS` value (`null` when not listed) — an absent block whose
  default is not listed (`model`, `read`, `head`, `aggregation`) is therefore `null`, never a dict
  of defaults;
* a PRESENT key is written as given — an explicit `null` hashes as `null`, even where the default
  is not (`head.bias: null` ≠ `head.bias` absent);
* a nested object with its own path entry (`render.bos_handling`, `decision.calibration`, …) is
  projected key by key; any other value is copied whole (`aggregation.params`).

R-039 changed three of those rules from S1's reading (038 Review R1 debt 1): `bos_handling` was
hashed whole, an explicit `null` was hashed as the default, an absent required block was a filled
dict. Migration `033` recomputes the stored digests once (FTASKS 11.4, I-38.8).

⚠ PARSED VALUES, NEVER A DUMP. Hashing `json.dumps(sort_keys=True)` is not JCS: Python writes
`1.0`, `-0.0` and `1e-07`, JCS writes `1`, `0` and `1e-7`. Because JCS formats a number by its
IEEE-754 value, a PostgreSQL JSONB round trip of the stored definition (which may turn `1e2` into
`100`) cannot change a digest (T-S1-14) — with ONE exception: JSONB writes a double of magnitude
≥ 1e16 that Python wrote in exponent form (`1e+21`) as an integer literal, which reads back as an
`int` beyond ±(2^53 − 1) and is then refused. A digest recomputed from such a JSONB row (migration
`033`, recalibrate, arm-time fill) is `None` where the import-time one existed (Review R1 6.5,
R65-2; no real probe carries such a value, and miStudio stores JSONB the same way).

⚠ REFUSED, NOT GUESSED — BOTH OR NEITHER. A NaN or infinity, or an integer outside ±(2^53 − 1),
anywhere in the projected top-level values (`decision` included), or any other failure of either
half, means NEITHER digest exists — miStudio's `document_identity` refuses the whole document.

Standard library only, and no service import: migrations `032` and `033` carry FROZEN copies of
this module's projection and serializer, because a migration must not import code that changes
after it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping, Optional

IDENTITY_ALGORITHM = "mistudio.probe-identity/1"

KNOWN_ALGORITHMS = frozenset({IDENTITY_ALGORITHM})

#: The detector projection, /1 (miStudio `probe_identity.DETECTOR_V1`; pinned to the case file's
#: `projection.detector.include`).
DETECTOR_V1: dict[str, tuple[str, ...]] = {
    "": ("model", "read", "scope", "basis", "head", "sae", "aggregation", "render"),
    "model": ("hf_id", "revision", "d_model", "n_layers", "architecture",
              "chat_template_sha256", "load_dtype", "quantization"),
    "read": ("layer", "hook_point"),
    "head": ("weights", "bias", "norm_mean", "norm_std", "attention_query"),
    "sae": ("hf_repo", "path", "revision", "weights_sha256", "architecture", "d_model",
            "n_features", "normalization", "feature_indices"),
    "aggregation": ("rule", "params", "streamable"),
    "render": ("generation_prompt", "add_special_tokens", "bos_handling"),
    "render.bos_handling": ("template_wrote_bos", "tokenizer_added_bos", "bos_count"),
}
#: The decision projection, /1 (miStudio `probe_identity.DECISION_V1`; pinned to the case file).
_BAND = ("min_tokens", "max_tokens", "threshold", "threshold_source", "target_fpr",
         "realised_fpr", "n_negatives")
DECISION_V1: dict[str, tuple[str, ...]] = {
    "": ("threshold", "target_fpr", "realised_fpr", "threshold_source", "calibration",
         "windows", "length_bands"),
    "calibration": ("hf_id", "config", "split", "revision"),
    "windows{}": ("threshold", "target_fpr", "realised_fpr", "n_negatives", "scope",
                  "length_bands"),
    "windows{}.length_bands[]": _BAND,
    "length_bands[]": _BAND,
}
#: What an ABSENT key is written as, by `<detector|decision>.<path>.<key>` — the case file's
#: `projection.defaults`, pinned equal to it. An absent key not listed here is `null`.
DEFAULTS: dict[str, Any] = {
    "detector.read.hook_point": "resid_post",
    "detector.head.bias": 0.0,
    "detector.sae.normalization": {},
    "detector.aggregation.params": {},
    "detector.render.add_special_tokens": False,
}
#: The top-level values checked for non-finite numbers and unsafe integers before either digest.
PROJECTED = frozenset(DETECTOR_V1[""]) | {"decision"}
_SAFE_INTEGER = 2 ** 53 - 1


class NonFiniteNumber(ValueError):
    """JCS has no form for NaN or infinity, nor an exact one for an integer outside ±(2^53 − 1);
    a definition carrying one has no digest (miStudio refuses it the same way)."""


# ── RFC 8785 ─────────────────────────────────────────────────────────────────────


def _number(value: float) -> str:
    """ECMAScript `Number.prototype.toString` of an IEEE-754 double (RFC 8785 §3.2.2.3)."""
    if math.isnan(value) or math.isinf(value):
        raise NonFiniteNumber(repr(value))
    if value == 0:
        return "0"  # -0.0 included
    sign = "-" if value < 0 else ""
    text = repr(abs(float(value)))  # the shortest round-trip digits
    mantissa, _, exp = text.partition("e")
    whole, _, frac = mantissa.partition(".")
    digits = (whole + frac).lstrip("0")
    # n: the position of the decimal point relative to the first significant digit.
    point = len(whole) + (int(exp) if exp else 0)
    lead = len(whole + frac) - len((whole + frac).lstrip("0"))
    n = point - lead
    digits = digits.rstrip("0") or "0"
    k = len(digits)
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        out = (digits if k == 1 else digits[0] + "." + digits[1:]) + "e" + ("+" if e >= 0 else "-") + str(abs(e))
    return sign + out


def jcs(value: Any) -> str:
    """The RFC 8785 canonical JSON text of parsed JSON values."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > _SAFE_INTEGER:
            raise NonFiniteNumber(f"integer {value} is outside ±(2^53 − 1)")
        return str(int(value))
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(jcs(item) for item in value) + "]"
    if isinstance(value, Mapping):
        keys = sorted(value, key=lambda k: str(k).encode("utf-16-be"))
        return "{" + ",".join(json.dumps(str(k), ensure_ascii=False) + ":" + jcs(value[k])
                              for k in keys) + "}"
    raise TypeError(f"not a JSON value: {type(value).__name__}")


# ── projections ──────────────────────────────────────────────────────────────────


def _join(path: str, key: str) -> str:
    return key if path == "" else f"{path}.{key}"


def _project(value: Any, path: str, table: Mapping[str, tuple[str, ...]], prefix: str) -> dict:
    """miStudio's `_project`: listed keys only, absent → default (else null), present → as given."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{prefix}.{path or '<root>'} is not an object")
    out: dict[str, Any] = {}
    for key in table.get(path, ()):
        child = _join(path, key)
        item = value[key] if key in value else copy.deepcopy(DEFAULTS.get(f"{prefix}.{child}"))
        if item is None:
            out[key] = None
        elif child in table:
            out[key] = _project(item, child, table, prefix)
        elif f"{child}[]" in table:
            out[key] = [_project(entry, f"{child}[]", table, prefix) for entry in item]
        elif f"{child}{{}}" in table:
            if not isinstance(item, Mapping):
                raise TypeError(f"{prefix}.{child} is not an object")
            out[key] = {str(name): _project(entry, f"{child}{{}}", table, prefix)
                        for name, entry in item.items()}
        else:
            out[key] = copy.deepcopy(item)
    return out


def refuse_unhashable(value: Any) -> None:
    """Raise `NonFiniteNumber` on a NaN/±inf or an integer outside ±(2^53 − 1), anywhere."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteNumber(repr(value))
        return
    if isinstance(value, int):
        if abs(value) > _SAFE_INTEGER:
            raise NonFiniteNumber(f"integer {value} is outside ±(2^53 − 1)")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            refuse_unhashable(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            refuse_unhashable(item)


def detector_payload(definition: Mapping[str, Any]) -> dict[str, Any]:
    """The inclusive detector projection: every /1 key present.

    ⚠ NEVER "the whole definition minus the threshold keys". A subtractive digest moves on every
    rebuild (`evidence`, `provenance.exported_at`, `test_vectors`) and would never equal the
    producer's; this one changes only when the detector does.
    """
    return _project(definition, "", DETECTOR_V1, "detector")


def decision_payload(decision: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The decision projection, every key present. A missing or `null` block is `{}` (miStudio's
    `document.get("decision") or {}`)."""
    return _project(decision or {}, "", DECISION_V1, "decision")


def canonical_bytes(kind: str, payload: Any) -> bytes:
    return jcs({"algorithm": IDENTITY_ALGORITHM, "digest": kind, "payload": payload}).encode()


def detector_digest(definition: Mapping[str, Any]) -> str:
    """SHA-256 of the detector wrapper, lowercase hex. Independent of every decision key."""
    return hashlib.sha256(canonical_bytes("detector", detector_payload(definition))).hexdigest()


def decision_digest(decision: Optional[Mapping[str, Any]]) -> str:
    """SHA-256 of the decision wrapper, lowercase hex."""
    return hashlib.sha256(canonical_bytes("decision", decision_payload(decision))).hexdigest()


def digests_of(definition: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """(detector, decision) for a stored definition, or `(None, None)`. Never raises.

    ⚠ BOTH OR NEITHER. miStudio's `document_identity` computes the two digests in one call, so
    ANY failure — a non-finite number or unsafe integer in the projected values, a projected
    block of the wrong shape (a `null` window entry, a `bos_handling` that is not an object), a
    string with a lone surrogate (no UTF-8 form, so no RFC 8785 bytes) — means the authority
    reports no identity for the document at all. Keeping the half miLLM could compute would
    store, and report on every result, a digest the producer says does not exist (038 R-039
    Review R1, finding R65-1).
    """
    try:
        if not isinstance(definition, Mapping):
            return None, None
        refuse_unhashable({key: definition.get(key) for key in PROJECTED if key in definition})
        return detector_digest(definition), decision_digest(definition.get("decision"))
    except (NonFiniteNumber, TypeError, ValueError, AttributeError):
        # `UnicodeEncodeError` (a lone surrogate) is a `ValueError`.
        return None, None


# ── a carried identity (miStudio 039 `identity`; R-039, D38.28 / D38.29) ─────────

#: Where a refused document goes next: miStudio rebuilds the block from the document it holds.
NEXT_STEP_REEXPORT = (
    "re-export the definition from miStudio (export_probe_definition) and import that file "
    "unchanged; a carried identity must be the document's own"
)


def identity_refusal(
    carried: Mapping[str, Any], detector: Optional[str], decision: Optional[str]
) -> Optional[dict[str, Any]]:
    """Why a carried identity cannot be accepted, or `None` when it is the document's own.

    Mirrors miStudio's `check_identity`: the algorithm first, then the detector digest, then the
    decision digest. `decision_revision` is not checked — it is miStudio's history position, not
    derivable from the document. A digest miLLM cannot compute (`None`, a non-finite number) is a
    mismatch with `recomputed: null`: it cannot be verified, so it is not stored as verified.
    """
    algorithm = carried.get("algorithm")
    if algorithm not in KNOWN_ALGORITHMS:
        return {
            "refused": "identity_algorithm_unknown",
            "algorithm": algorithm,
            "supported": sorted(KNOWN_ALGORITHMS),
            "next_step": (
                f"this server verifies {', '.join(sorted(KNOWN_ALGORITHMS))} only; re-export from "
                "a miStudio that writes it (export_probe_definition), or upgrade miLLM"
            ),
        }
    for field, recomputed in (("detector_digest", detector), ("decision_digest", decision)):
        if carried.get(field) != recomputed:
            return {
                "refused": "identity_mismatch",
                "field": field,
                "carried": carried.get(field),
                "recomputed": recomputed,
                "algorithm": algorithm,
                "next_step": NEXT_STEP_REEXPORT,
            }
    return None
