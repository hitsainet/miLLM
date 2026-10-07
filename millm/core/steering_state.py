"""The steering-set hash and the `X-miLLM-Steering` header (Feature 28, FTDD §5.2 – §5.3).

Pure: the standard library and `millm.core.steering_range` only, so a consumer can copy it and a
test runs in microseconds. Nothing else in miLLM formats a steering hash or this header — the
response header, the streaming terminal chunk and a batch output line all call
`serialize_steering_header` (FR-28.3.10), so the three cannot disagree.

⚠ THE HASH IS CANONICAL ACROSS THE SUITE (X-07). miDataworks 007 pins test vectors TV-1 – TV-4
(FTDD §5.3, the API reference, and `tests/unit/core/test_steering_state.py`). Changing ANY byte of
the canonical form — the version line, the LF, the ordering, the bit-pattern spelling, the zero
rule — breaks every steered pair a consumer has already recorded. It is versioned
(`millm.steering-set/v1`) so a change is a new version, never an edit.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from millm.core.steering_range import clamp_steering

CANONICAL_VERSION = "millm.steering-set/v1"

#: The header kinds, FTDD §5.2. `none` and `unknown` are only ever the sole member.
KINDS = ("none", "unknown", "profile", "inline", "manual", "circuit")
SOLE_KINDS = frozenset({"none", "unknown"})

#: `reason` values on an `unknown` item (FTDD §5.2).
REASONS = frozenset({"claims_unreadable", "profile_unreadable", "llamacpp_entries", "read_failed"})

#: `source` values on a `profile` item.
SOURCES = frozenset({"request", "active"})

# RFC 8941 §3.3.4 token: first char ALPHA or "*", then tchar / ":" / "/".
_TOKEN_RE = re.compile(r"^[A-Za-z*][!#$%&'*+\-.^_`|~0-9A-Za-z:/]*$")


def applied_set(pairs: Iterable[tuple[int, float]]) -> dict[int, float]:
    """`index → clamp_steering(strength)`, every zero removed — what the hook reads.

    Clamp FIRST, then drop zeros: the hash describes the applied value, never the requested one
    (TV-4). `-0.0 == 0.0` in IEEE-754, so a negative zero is dropped too — it would otherwise
    hash as `8000000000000000`, a different spelling of "no steering" (TV-3).
    """
    out: dict[int, float] = {}
    for index, strength in pairs:
        value = clamp_steering(float(strength))
        if value != 0.0:
            out[int(index)] = value
    return out


def count_clamped(pairs: Iterable[tuple[int, float]]) -> int:
    """How many strengths `clamp_steering` changed (FR-28.1.7) — the header's `clamped`."""
    return sum(1 for _i, s in pairs if clamp_steering(float(s)) != float(s))


def canonical_set_form(sae_id: str, applied: Mapping[int, float]) -> bytes:
    """The canonical bytes of an applied set (FTDD §5.3, normative; X-07).

    Line 1 `millm.steering-set/v1`, line 2 `sae=<sae_id>`, then one `<index>:<bits>` line per
    non-zero feature in ascending index order; every line ends in exactly one LF, the last
    included; UTF-8, no BOM, no spaces. `<bits>` is the binary64 value big-endian as 16
    lowercase hex digits — a bit pattern, because shortest-decimal text differs between languages
    (`1e-05` against `1e-5`) and a bit pattern has one spelling.
    """
    if "\n" in sae_id or "\r" in sae_id:
        # A line break would let an SAE id forge a feature line. Real ids never contain one:
        # they are repo paths with `/` replaced by `--` (sae_downloader).
        raise ValueError(f"an SAE id must not contain a line break: {sae_id!r}")
    lines = [CANONICAL_VERSION, f"sae={sae_id}"]
    for index in sorted(applied):
        value = float(applied[index])
        if not math.isfinite(value):
            raise ValueError(f"a non-finite strength cannot be hashed: feature {index}={value!r}")
        if value == 0.0:
            # The applied set never carries a zero (FTDD §5.3). Dropped, not refused, so a
            # caller passing a raw map still gets the definition's answer.
            continue
        if int(index) < 0:
            raise ValueError(f"a feature index must be non-negative: {index}")
        lines.append(f"{int(index)}:{struct.pack('>d', value).hex()}")
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def steering_set_hash(sae_id: str, applied: Mapping[int, float]) -> str:
    """`sha256:` + 64 lowercase hex digits over `canonical_set_form` (FR-28.3.4)."""
    return "sha256:" + hashlib.sha256(canonical_set_form(sae_id, applied)).hexdigest()


def format_intensity(value: float) -> str:
    """λ as the shortest decimal that round-trips to its binary64 value — Python's `repr`.

    A String, not an RFC 8941 Decimal: a Decimal carries at most three fractional digits
    (RFC 8941 §3.3.2), so λ = 0.4375 would arrive as 0.438 and fail an exact comparison.
    """
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"intensity must be finite: {value!r}")
    return repr(value)


def encode_name(name: str) -> str:
    """Percent-encode a profile name for an RFC 8941 String (FTDD §5.2).

    RFC 8941 Strings are printable ASCII. Each UTF-8 byte outside `%x20-7E`, and each of `%`,
    `"` and `\\`, becomes `%XX` in uppercase hex, so an operator-chosen name can never inject
    header syntax. The consumer decodes.
    """
    out: list[str] = []
    for byte in name.encode("utf-8"):
        if 0x20 <= byte <= 0x7E and byte not in (0x25, 0x22, 0x5C):
            out.append(chr(byte))
        else:
            out.append(f"%{byte:02X}")
    return "".join(out)


def _sf_string(value: str) -> str:
    """An RFC 8941 sf-string (§4.1.6). Refuses a character outside printable ASCII rather than
    emit a header the consumer's parser rejects."""
    for ch in value:
        if not 0x20 <= ord(ch) <= 0x7E:
            raise ValueError(f"not representable as an RFC 8941 String: {value!r}")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _sf_token(value: str) -> str:
    if not _TOKEN_RE.match(value):
        raise ValueError(f"not an RFC 8941 Token: {value!r}")
    return value


def _sf_integer(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"not an integer: {value!r}")
    if abs(value) > 999_999_999_999_999:
        raise ValueError(f"integer out of RFC 8941 range: {value}")
    return str(value)


@dataclass(frozen=True)
class SteeringItem:
    """One member of `X-miLLM-Steering`: a kind and the parameters FTDD §5.2 gives it."""

    kind: str
    sae: Optional[str] = None
    layer: Optional[int] = None
    features: Optional[int] = None
    hash: Optional[str] = None
    clamped: int = 0
    name: Optional[str] = None
    source: Optional[str] = None
    intensity: Optional[float] = None
    circuit_id: Optional[str] = None
    composed: bool = False
    changed: bool = False
    reason: Optional[str] = None

    def sort_key(self) -> tuple:
        """Member order (FTDD §5.2): circuits first by id, then SAE items by layer, then sae."""
        if self.kind == "circuit":
            return (0, str(self.circuit_id or ""), 0, "")
        return (1, "", int(self.layer if self.layer is not None else -1), str(self.sae or ""))


def _require(item: SteeringItem, *fields: str) -> None:
    missing = [f for f in fields if getattr(item, f) is None]
    if missing:
        raise ValueError(f"a `{item.kind}` item needs {missing}")


def _sae_params(item: SteeringItem) -> list[str]:
    _require(item, "sae", "layer", "features", "hash")
    return [
        f"sae={_sf_string(item.sae)}",  # type: ignore[arg-type]
        f"layer={_sf_integer(item.layer)}",  # type: ignore[arg-type]
        f"features={_sf_integer(item.features)}",  # type: ignore[arg-type]
        f"hash={_sf_string(item.hash)}",  # type: ignore[arg-type]
    ]


def _serialize_item(item: SteeringItem) -> str:
    """One list member, its parameters in the FTDD §5.2 order. A true boolean is bare
    (`changed`, never `changed=?1`, RFC 8941 §4.1.1.2); a false one is omitted."""
    params: list[str] = []
    kind = item.kind
    if kind == "none":
        pass
    elif kind == "unknown":
        if item.reason is not None:
            if item.reason not in REASONS:
                raise ValueError(f"unknown reason {item.reason!r}")
            params.append(f"reason={_sf_token(item.reason)}")
    elif kind == "profile":
        _require(item, "name", "source", "intensity")
        if item.source not in SOURCES:
            raise ValueError(f"profile source must be one of {sorted(SOURCES)}: {item.source!r}")
        params.append(f"name={_sf_string(encode_name(item.name))}")  # type: ignore[arg-type]
        params.append(f"source={_sf_token(item.source)}")  # type: ignore[arg-type]
        params.append(f"intensity={_sf_string(format_intensity(item.intensity))}")  # type: ignore[arg-type]
        params.extend(_sae_params(item))
        if item.clamped:
            params.append(f"clamped={_sf_integer(item.clamped)}")
    elif kind == "inline":
        params.extend(_sae_params(item))
        if item.clamped:
            params.append(f"clamped={_sf_integer(item.clamped)}")
    elif kind == "manual":
        params.extend(_sae_params(item))
    elif kind == "circuit":
        _require(item, "circuit_id", "intensity")
        params.append(f"id={_sf_string(item.circuit_id)}")  # type: ignore[arg-type]
        params.append(f"intensity={_sf_string(format_intensity(item.intensity))}")  # type: ignore[arg-type]
        if item.composed:
            params.append("composed")
    else:
        raise ValueError(f"unknown steering kind {kind!r}")
    if item.changed:
        params.append("changed")
    return ";".join([kind, *params])


def serialize_steering_header(items: Sequence[SteeringItem]) -> str:
    """The `X-miLLM-Steering` field value: an RFC 8941 List (§3.1, serialised per §4.1).

    An empty sequence is `none`. `none` or `unknown` beside another member raises: that is a bug
    in the caller, not data to be reported.
    """
    if not items:
        return "none"
    kinds = [item.kind for item in items]
    if len(items) > 1 and any(k in SOLE_KINDS for k in kinds):
        raise ValueError(f"`none`/`unknown` must be the only member, got {kinds}")
    ordered = sorted(items, key=SteeringItem.sort_key)
    return ", ".join(_serialize_item(item) for item in ordered)
