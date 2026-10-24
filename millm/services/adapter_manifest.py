"""The adapter manifest validator seam and field extraction (Feature 33, FR-33.1.5 / FR-33.6).

miForge owns `miforge.adapter-manifest` v1 and miLLM vendors it byte-identical at
`docs/schemas/adapter-manifest-v1.json` (BRD-01 R-01.28; the sync test is
`tests/unit/services/test_adapter_manifest_fields_match_schema.py`). Three rules are load-bearing:

**The vendored file is read at CALL time, never at import.** An absent file refuses every load with
`ADAPTER_SCHEMA_NOT_VENDORED` (409) — it never degrades to "valid", because a manifest nobody
checked is how a corrupted or foreign adapter would load silently.

**`date-time` is actually checked (addendum D-29).** `jsonschema` skips `format: date-time` unless
`rfc3339-validator` is installed, and it is not; `created_at: "not-a-date"` validated. The checker is
registered by hand with the RFC 3339 rule in `millm.services.rfc3339`, which the manifest tool and
032's checkpoint manifest also use, so every writer and reader agree on one definition.

**Fields read with a default are read with THAT default.** The validator does not inject defaults,
so `adapter["scaling"]` raises `KeyError` on a manifest that legitimately omits it.
`FIELDS_REQUIRED` and `FIELDS_OPTIONAL_WITH_DEFAULT` record which is which; the sync test pins both
against the vendored schema (033 review finding 16). A scaling rule other than `alpha_over_r` is
refused rather than guessed, as the schema requires of a reader.

The schema is closed (`additionalProperties: false` on every object), so an unknown key is refused
with `ADAPTER_MANIFEST_INVALID`, the offending key named by its JSON path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from millm.core.errors import (
    AdapterInvalidError,
    AdapterManifestInvalidError,
    AdapterSchemaNotVendoredError,
)
from millm.ml.adapter_hooks import AdapterPlanError, lora_scaling
from millm.services.rfc3339 import date_time_format_checker

#: The vendored contract. Read at call time (`vendored_validator`), never at import.
VENDORED = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "adapter-manifest-v1.json"

#: Read unconditionally: each is a required property of the vendored schema (sync test).
FIELDS_REQUIRED: tuple[str, ...] = (
    "name",
    "base.name",
    "adapter.rank",
    "adapter.alpha",
    "adapter.target_modules",
    "run_id",
    "step",
    "weights_sha256",
)
#: Read with a default: each exists in the schema, is optional, and carries this default.
FIELDS_OPTIONAL_WITH_DEFAULT: dict[str, Any] = {
    "base.manifest_sha256": None,
    "adapter.scaling": "alpha_over_r",
}


class AdapterManifestValidator(Protocol):
    """Validates one parsed manifest. Returns every problem (`path` as a JSON path, `message`),
    sorted by path; an empty list means valid."""

    def errors(self, manifest: Any) -> list[dict[str, str]]: ...


def _json_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


class _SchemaValidator:
    """Draft 2020-12 over a schema dict, with `date-time` checked (D-29)."""

    def __init__(self, schema: dict) -> None:
        import jsonschema

        self._validator = jsonschema.Draft202012Validator(
            schema, format_checker=date_time_format_checker()
        )

    def errors(self, manifest: Any) -> list[dict[str, str]]:
        problems: list[dict[str, str]] = []
        for error in self._validator.iter_errors(manifest):
            parts = list(error.absolute_path)
            if error.validator == "additionalProperties" and isinstance(error.instance, dict):
                # Name each unknown key by its own path, not its parent's.
                allowed = set((error.schema or {}).get("properties", {}))
                extras = sorted(k for k in error.instance if k not in allowed)
                for key in extras:
                    problems.append(
                        {"path": _json_path(parts + [key]), "message": "unknown field"}
                    )
                if extras:
                    continue
            problems.append({"path": _json_path(parts), "message": error.message})
        problems.sort(key=lambda p: (p["path"], p["message"]))
        return problems


def schema_validator(schema: dict) -> AdapterManifestValidator:
    """A validator for an explicit schema dict (tests, or a caller holding the schema)."""
    return _SchemaValidator(schema)


def vendored_validator(path: Optional[Path] = None) -> AdapterManifestValidator:
    """The production validator: the vendored schema, read now.

    Raises `AdapterSchemaNotVendoredError` when the file is absent or unreadable. Never returns
    a permissive validator in its place.
    """
    schema_path = VENDORED if path is None else path
    try:
        schema = json.loads(schema_path.read_bytes())
    except FileNotFoundError:
        raise AdapterSchemaNotVendoredError(
            "the adapter manifest schema is not vendored, so no adapter can be verified "
            "(docs/schemas/adapter-manifest-v1.json is missing)",
            details={"schema": "docs/schemas/adapter-manifest-v1.json"},
        ) from None
    except (OSError, ValueError) as e:
        raise AdapterSchemaNotVendoredError(
            "the vendored adapter manifest schema cannot be read",
            details={"schema": "docs/schemas/adapter-manifest-v1.json", "error": type(e).__name__},
        ) from None
    if not isinstance(schema, dict):
        raise AdapterSchemaNotVendoredError(
            "the vendored adapter manifest schema is not a JSON object",
            details={"schema": "docs/schemas/adapter-manifest-v1.json"},
        )
    return _SchemaValidator(schema)


ValidatorFactory = Callable[[], AdapterManifestValidator]


def validate_manifest(manifest: Any, validator: AdapterManifestValidator) -> None:
    """Raise `AdapterManifestInvalidError` naming every failing field by JSON path."""
    problems = validator.errors(manifest)
    if problems:
        first = problems[0]
        more = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
        raise AdapterManifestInvalidError(
            f"adapter_manifest.json is invalid: {first['path']}: {first['message']}{more}",
            details={"errors": problems},
        )


@dataclass(frozen=True)
class AdapterManifestFields:
    """Every manifest field miLLM reads, with the defaults applied."""

    name: str
    run_id: str
    step: int
    weights_sha256: str
    base_name: str
    base_manifest_sha256: Optional[str]
    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    scaling_rule: str
    #: `alpha / rank` for `alpha_over_r`, the only rule v1 defines.
    scaling: float


def extract_fields(manifest: dict) -> AdapterManifestFields:
    """The fields read, from a manifest that has already validated.

    `base.manifest_sha256` and `adapter.scaling` are optional with a default and read with
    `.get(...)`. A scaling rule miLLM does not implement is refused (`ADAPTER_INVALID`, reason
    `scaling`) — reachable only through a validator that admits it, since v1's enum does not.
    """
    base = manifest["base"]
    adapter = manifest["adapter"]
    rule = adapter.get("scaling", FIELDS_OPTIONAL_WITH_DEFAULT["adapter.scaling"])
    try:
        scaling = lora_scaling(adapter["alpha"], adapter["rank"], rule)
    except AdapterPlanError as e:
        raise AdapterInvalidError(
            f"adapter scaling cannot be applied: {e.message}",
            details={"reason": e.reason, "scaling": str(rule)},
        ) from None
    return AdapterManifestFields(
        name=manifest["name"],
        run_id=manifest["run_id"],
        step=manifest["step"],
        weights_sha256=manifest["weights_sha256"],
        base_name=base["name"],
        base_manifest_sha256=base.get(
            "manifest_sha256", FIELDS_OPTIONAL_WITH_DEFAULT["base.manifest_sha256"]
        ),
        rank=adapter["rank"],
        alpha=float(adapter["alpha"]),
        target_modules=tuple(adapter["target_modules"]),
        scaling_rule=rule,
        scaling=scaling,
    )
