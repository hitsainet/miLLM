"""The checkpoint manifest contract and canonical naming (Feature 32, FTASKS 1.0).

miForge owns `miforge.checkpoint-manifest` v1; miLLM vendors it byte-identical at
`docs/schemas/checkpoint-manifest-v1.json` (sync test: `tests/unit/api/
test_checkpoint_manifest_schema_sync.py`). This module reads that file and nothing else: there is
no hand-kept field list here, so a re-vendored schema changes behaviour only through the schema.

WHO WRITES WHICH FIELD IS DERIVED, NEVER LISTED (FR-32.3.1). A top-level property whose description
begins "Written by the trainer." is the caller's. Every other property except the two constants
(`kind`, `schema_version`) is the checkpoint WRITER's, and each writer description must contain the
schema's exact WRITER sentence. A re-vendored schema that adds a property marked neither way would
otherwise silently become miLLM's to fill in; `assert_schema_ownership()` refuses it. It runs from
the application `lifespan` and from a unit test, NOT at import (032 review 5.13): a module that
fails to import hides its names from every test that imports it defensively.

Only TOP-LEVEL properties are classified. `ScoreRecord.provider` (nested) also begins "Written by
the trainer."; it belongs to the object that contains it, not to the manifest's ownership split.

THREE REFUSALS, IN THIS ORDER, ON A SAVE (`validate_caller_fields`)
1. A writer field from the caller → `MANIFEST_FIELD_NOT_CALLERS`, naming every such field.
2. `kind`/`schema_version` unequal to their `const`, or a key the schema does not define →
   `INVALID_MANIFEST`. A value EQUAL to the const is accepted and popped BEFORE the projection is
   validated (addendum D-30): the projection's properties are the caller fields only and the
   schema closes every object, so an un-popped `kind` would be refused as unknown.
3. The projection (the full schema with the manifest's `properties` and `required` reduced to the
   caller's, `$defs` kept) → `INVALID_MANIFEST` with every error, sorted by JSON path. Then the
   strict save rule for `run_id` → `INVALID_RUN_ID`.

ON A LOAD (`validate_full`) the manifest was written by someone else, usually miForge, which may
omit `kind` and `schema_version` (neither is `required`; D-30). It is validated against the
unmodified schema; unknown keys are refused and named one by one.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator

from millm.core.errors import (
    InvalidManifestError,
    InvalidRunIdError,
    ManifestFieldNotCallersError,
)
from millm.services.rfc3339 import date_time_format_checker, utc_now_rfc3339

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "schemas" / "checkpoint-manifest-v1.json"
)
#: The definition the schema's root `$ref` names.
MANIFEST_DEF = "CheckpointManifestV1"

TRAINER = "Written by the trainer."
WRITER = (
    "Written by whichever process writes the checkpoint — miForge by default, "
    "miLLM in the section 5.10 mode."
)
#: The two constants: no ownership mark; miLLM writes their `const` on save (FR-32.3.2).
UNMARKED = frozenset({"kind", "schema_version"})

#: The save rule (FR-32.4.3): one safe path segment, no `:`. Matched with `fullmatch`, because
#: `$` also matches before a trailing newline.
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
#: The load rule (addendum D-22): miForge's run_id format is unpinned, so a load accepts any safe
#: path segment — no `/`, NUL or `:`, and not `.` or `..`.
_LOAD_FORBIDDEN = ("/", "\x00", ":")
#: A path segment longer than this cannot exist on the filesystems in use (NAME_MAX).
MAX_SEGMENT_BYTES = 255

_STEP_SEGMENT = re.compile(r"step-([0-9]+)")

_CHECKER = date_time_format_checker()


class SchemaOwnershipError(RuntimeError):
    """The vendored schema has a property whose owner cannot be derived.

    Not fatal at startup (addendum D-35): `weights_checkpoint_service.check_schema_ownership_on_startup`
    logs it and every 032 save and load refuses with `INVALID_MANIFEST`, reason `schema_ownership`.
    """


@dataclass(frozen=True)
class FieldOwnership:
    caller: frozenset[str]           # description begins TRAINER
    writer: frozenset[str]           # every other property minus UNMARKED
    caller_required: frozenset[str]  # caller ∩ the schema's `required`


# --------------------------------------------------------------------------- schema


@lru_cache(maxsize=1)
def _load_schema_cached() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_schema() -> dict[str, Any]:
    """The vendored schema, read from disk once. A deep copy: callers cannot corrupt the cache."""
    return copy.deepcopy(_load_schema_cached())


def manifest_definition(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The object schema the root `$ref` names (`$defs.CheckpointManifestV1`)."""
    ref = schema.get("$ref", f"#/$defs/{MANIFEST_DEF}")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        raise SchemaOwnershipError(f"checkpoint manifest schema root $ref is not local: {ref!r}")
    return schema["$defs"][ref.removeprefix("#/$defs/")]


def derive_ownership(schema: Mapping[str, Any]) -> FieldOwnership:
    """Classify the manifest's TOP-LEVEL properties; refuse a writer field without the WRITER
    sentence (`SchemaOwnershipError`, naming every offending field)."""
    definition = manifest_definition(schema)
    props: Mapping[str, Any] = definition.get("properties", {})
    caller = {
        name
        for name, spec in props.items()
        if str(spec.get("description", "")).startswith(TRAINER)
    }
    writer = set(props) - caller - UNMARKED
    unmarked = sorted(
        name for name in writer if WRITER not in str(props[name].get("description", ""))
    )
    if unmarked:
        raise SchemaOwnershipError(
            "checkpoint manifest properties with no derivable owner (neither "
            f"{TRAINER!r} nor the WRITER sentence): {unmarked}. Re-vendor from miForge, or "
            "teach miLLM who writes them; never let them default to miLLM."
        )
    required = set(definition.get("required", []))
    return FieldOwnership(
        caller=frozenset(caller),
        writer=frozenset(writer),
        caller_required=frozenset(caller & required),
    )


@lru_cache(maxsize=1)
def ownership() -> FieldOwnership:
    """The vendored schema's ownership split, derived once (and verified while deriving)."""
    return derive_ownership(_load_schema_cached())


def assert_schema_ownership() -> FieldOwnership:
    """Raise `SchemaOwnershipError` on an unmarked field. Called per save/load and, logged rather
    than raised, from the `lifespan` (D-35)."""
    return derive_ownership(_load_schema_cached())


@lru_cache(maxsize=1)
def _projection() -> dict[str, Any]:
    """The full schema with the manifest reduced to the caller's fields; `$defs` kept."""
    schema = load_schema()
    own = ownership()
    definition = manifest_definition(schema)
    definition["properties"] = {
        k: v for k, v in definition["properties"].items() if k in own.caller
    }
    definition["required"] = [r for r in definition.get("required", []) if r in own.caller]
    return schema


@lru_cache(maxsize=1)
def _full_validator() -> Draft202012Validator:
    return Draft202012Validator(_load_schema_cached(), format_checker=_CHECKER)


@lru_cache(maxsize=1)
def _projection_validator() -> Draft202012Validator:
    return Draft202012Validator(_projection(), format_checker=_CHECKER)


def _consts() -> dict[str, Any]:
    props = manifest_definition(_load_schema_cached())["properties"]
    return {name: props[name]["const"] for name in UNMARKED}


# --------------------------------------------------------------------------- naming


def _check_step(step: object) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    return step


def step_path_segment(step: int) -> str:
    """The directory segment: `step-` plus the step zero-padded to at least six digits."""
    return f"step-{_check_step(step):06d}"


def step_name_segment(step: int) -> str:
    """The name segment: `step-` plus the step unpadded (FR-32.4.1)."""
    return f"step-{_check_step(step)}"


def render_name(base_model: str, run_id: str, step: int) -> str:
    """`<base_model>:<run_id>:step-<step>` (FR-32.4.2)."""
    return f"{base_model}:{run_id}:{step_name_segment(step)}"


def parse_name(name: str) -> tuple[str, str, int]:
    """(base_model, run_id, step) from a name, split from the RIGHT twice, so a base containing
    `:` parses. Raises `ValueError` on anything `render_name` could not have produced."""
    if not isinstance(name, str):
        raise ValueError(f"not a checkpoint name: {name!r}")
    parts = name.rsplit(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise ValueError(f"not a checkpoint name: {name!r}")
    base, run_id, segment = parts
    match = _STEP_SEGMENT.fullmatch(segment)
    if match is None:
        raise ValueError(f"not a checkpoint name: {name!r}")
    step = int(match.group(1))
    if segment != step_name_segment(step):  # `step-0500` is not the unpadded form
        raise ValueError(f"not a checkpoint name (step must be unpadded): {name!r}")
    return base, run_id, step


@lru_cache(maxsize=1)
def model_name_limit() -> int:
    """`Model.name`'s column length: a carrier row is named by the manifest `name`."""
    from millm.db.models.model import Model

    return int(Model.__table__.c.name.type.length)


def validate_run_id(run_id: object, *, mode: Literal["save", "load"] = "save") -> str:
    """The `run_id` rule: strict on save (FR-32.4.3), a safe path segment on load (D-22)."""
    if mode == "save":
        if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
            raise InvalidRunIdError(
                f"run_id {run_id!r} must match {RUN_ID.pattern} (one safe path segment, no ':')",
                details={"run_id": run_id if isinstance(run_id, str) else repr(run_id),
                         "pattern": RUN_ID.pattern},
            )
        return run_id
    if (
        not isinstance(run_id, str)
        or run_id in ("", ".", "..")
        or any(ch in run_id for ch in _LOAD_FORBIDDEN)
        or len(run_id.encode("utf-8")) > MAX_SEGMENT_BYTES
    ):
        raise InvalidRunIdError(
            f"run_id {run_id!r} is not a safe path segment (no '/', NUL or ':', not '.' or '..')",
            details={"run_id": run_id if isinstance(run_id, str) else repr(run_id),
                     "pattern": "safe-segment"},
        )
    return run_id


def check_name_length(base_model: str, run_id: str, step: int) -> str:
    """The rendered name, refused with `INVALID_RUN_ID` when it cannot fit `Model.name` (D-22).
    Checked before any row insert, so the database never raises a 500 on it."""
    name = render_name(base_model, run_id, step)
    limit = model_name_limit()
    if len(name) > limit:
        raise InvalidRunIdError(
            f"checkpoint name is {len(name)} characters; the model row holds at most {limit}. "
            "Use a shorter run_id.",
            details={"run_id": run_id, "name_length": len(name), "limit": limit},
        )
    return name


# --------------------------------------------------------------------------- validation


def _error(json_path: str, message: str) -> dict[str, str]:
    return {"json_path": json_path, "message": message}


def _schema_errors(validator: Draft202012Validator, doc: Any) -> list[dict[str, str]]:
    return [_error(e.json_path, e.message) for e in validator.iter_errors(doc)]


def _sorted(errors: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(errors, key=lambda e: (e["json_path"], e["message"]))


def _raise_invalid(errors: list[dict[str, str]], what: str) -> None:
    errors = _sorted(errors)
    raise InvalidManifestError(
        f"{what} does not match the checkpoint manifest schema: "
        + "; ".join(f"{e['json_path']}: {e['message']}" for e in errors),
        details={"errors": errors},
    )


def validate_caller_fields(fields: object) -> dict[str, Any]:
    """The caller's fields for a save, validated; returns them without `kind`/`schema_version`."""
    if not isinstance(fields, Mapping):
        _raise_invalid([_error("$", "manifest fields must be an object")], "manifest fields")
    own = ownership()
    doc = dict(fields)  # type: ignore[arg-type]
    writer = sorted(k for k in doc if k in own.writer)
    if writer:
        raise ManifestFieldNotCallersError(
            f"these manifest fields are written by miLLM, not the caller: {writer}",
            details={"fields": writer},
        )
    errors: list[dict[str, str]] = []
    for name, const in _consts().items():
        if name in doc:
            if doc[name] != const:
                errors.append(_error(f"$.{name}", f"{doc[name]!r} is not {const!r}"))
            doc.pop(name)  # equal or not, never reaches the projection (D-30)
    for key in sorted(k for k in doc if k not in own.caller):
        errors.append(_error(f"$.{key}", f"{key!r} is not a checkpoint manifest field"))
        doc.pop(key)
    errors.extend(_schema_errors(_projection_validator(), doc))
    if errors:
        _raise_invalid(errors, "manifest fields")
    validate_run_id(doc["run_id"], mode="save")
    return doc


def _ordered(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Keys in schema property order, so a manifest diffs cleanly."""
    order = list(manifest_definition(_load_schema_cached())["properties"])
    return {k: doc[k] for k in order if k in doc}


def build_manifest(
    fields: Mapping[str, Any],
    *,
    base_model: str,
    weights_sha256: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """The complete manifest miLLM writes: the caller's fields plus every writer field and both
    constants, validated against the full schema (FR-32.3.4, FR-32.3.5)."""
    doc = validate_caller_fields(fields)
    name = check_name_length(base_model, doc["run_id"], doc["step"])
    doc.update(_consts())
    doc.update(
        base_model=base_model,
        name=name,
        created_at=created_at if created_at is not None else utc_now_rfc3339(),
        weights_sha256=weights_sha256,
    )
    return validate_full(_ordered(doc))


def validate_full(doc: object) -> dict[str, Any]:
    """A complete manifest against the unmodified schema (format checker on). Unknown top-level
    keys are named one by one; `kind`/`schema_version` may be absent (D-30)."""
    if not isinstance(doc, Mapping):
        _raise_invalid([_error("$", "a manifest must be an object")], "manifest")
    props = manifest_definition(_load_schema_cached())["properties"]
    known = {k: v for k, v in doc.items() if k in props}  # type: ignore[union-attr]
    errors = [
        _error(f"$.{key}", f"{key!r} is not a checkpoint manifest field")
        for key in doc  # type: ignore[union-attr]
        if key not in props
    ]
    errors.extend(_schema_errors(_full_validator(), known))
    if errors:
        _raise_invalid(errors, "manifest")
    return dict(doc)  # type: ignore[arg-type]


def serialise_manifest(doc: Mapping[str, Any]) -> bytes:
    """The one serialisation: the file's bytes and `manifest_sha256` are both over exactly this."""
    return (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
