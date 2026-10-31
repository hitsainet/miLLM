"""THE validator registry (039 FR-39.19, TD39-27; mirrors the producer's FR-016.1/016.2).

``VALIDATORS`` is the one live registry, keyed by the exact ``(id, version)``, mirroring
miDataworks' registered ids and versions one for one. A rule names an exact version and
:func:`resolve` returns exactly that version, never "the latest". Adding a validator is a reviewed
code change that arrives with shared vectors (``tests/unit/rules/test_validators.py`` fails on a
registered validator without one).

The ids are DATA here: 038's generality guard exempts exactly the ids it reads from this dict,
inside this package only (``registered_ids``).
"""

from __future__ import annotations

from typing import Any

from .algorithms import GENERIC_VALIDATORS
from .base import Candidate, Validator, ValidatorDecl, ValidatorUnknown, Verdict
from .formats import FORMAT_VALIDATORS
from .masked import MASKED

__all__ = [
    "VALIDATORS",
    "Candidate",
    "Validator",
    "ValidatorDecl",
    "ValidatorUnknown",
    "Verdict",
    "declaration",
    "registered_ids",
    "resolve",
    "superseded_by",
]

_ALL: tuple[Validator, ...] = (*FORMAT_VALIDATORS, *GENERIC_VALIDATORS, MASKED)
VALIDATORS: dict[tuple[str, int], Validator] = {(v.decl.id, v.decl.version): v for v in _ALL}


def registered_ids() -> frozenset[str]:
    """Every registered validator id (the generality guard's exemption, TD39-27)."""
    return frozenset(i for i, _v in VALIDATORS)


def resolve(validator_id: str, version: int) -> Validator:
    """The validator ``validator_id@version`` exactly; refuses an unregistered version."""
    found = VALIDATORS.get((validator_id, int(version)))
    if found is None:
        raise ValidatorUnknown(validator_id, int(version))
    return found


def superseded_by(validator_id: str, version: int) -> int | None:
    """The next registered version of the same id, or None when this is the newest."""
    later = [v for (i, v) in VALIDATORS if i == validator_id and v > version]
    return min(later) if later else None


def declaration(validator: Validator) -> dict[str, Any]:
    """The JSON form of a declaration, as `GET /api/rule-sets/validators` returns it."""
    decl = validator.decl
    shape = decl.mask_shape
    return {
        "id": decl.id,
        "version": decl.version,
        "ref": decl.ref,
        "description": decl.description,
        "params_schema": decl.params_schema,
        "reasons": list(decl.reasons),
        "outputs": [{"name": n, "type": t} for n, t in decl.outputs],
        "algorithms": list(decl.algorithms),
        "value_bearing": decl.value_bearing,
        "mask_shape": (
            None
            if shape is None
            else {
                "lengths": list(shape.lengths),
                "layouts": [list(g) for g in shape.layouts],
                "delimiters": shape.delimiters,
                "keep_leading": shape.keep_leading,
                "keep_trailing": shape.keep_trailing,
            }
        ),
        "superseded_by": superseded_by(decl.id, decl.version),
    }
