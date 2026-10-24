"""Recompute the stored probe identity digests under miStudio's projection (038 R-039, I-38.8).

S1 (migration `032`, `probe_digest.py` at 032) read three rules of `mistudio.probe-identity/1`
differently from miStudio 039, the algorithm's owner (038 Review R1 debt 1):

(a) `render.bos_handling` was hashed whole; miStudio projects its three keys;
(b) an explicit `null` on a key with a non-null default was hashed as the default; miStudio hashes
    `null`;
(c) an absent required block (`model`, `read`, `head`, `aggregation`) was a dict of defaults;
    miStudio hashes `null`.

and S1 kept the detector half of a document whose `decision` carried a non-finite number, and wrote
an integer beyond ±(2^53 − 1) as a double, where miStudio refuses both digests.

All 70 of miStudio's vendored cases agreed under both readings, so a stored digest moves only for a
definition that hits one of those edges. Which stored probes do could not be read from production
here (the build agent has no production read), so this migration settles it by recomputation:
every `probes` row whose digests are miLLM's own (`detector_digest_source` other than `producer`)
is recomputed from its stored raw definition; rows that already agree are left untouched, and the
ids of the rows that moved are logged — that log line IS the record of whether production had any.

**Why a new revision and not an edit to `032`.** `032` shipped with S1 and has run where S1 was
deployed; editing it would change nothing on those databases and make one revision mean two
things. A database upgrading from `031` runs `032` and then this.

**Not changed:** `probe_events` rows keep the digest their result recorded (a result is a record of
what was reported, and `recorded_from_row` must stay byte-stable); `threshold_history` entries keep
theirs. Neither moves for a definition that passed import: the decision projections agree for
every document the mirror accepts.

**Downgrade** is a no-op: 032's values were wrong where they differ, and nothing reads them.

⚠ FROZEN. The serializer and the projection below are COPIES of `millm/services/probe_digest.py`
as of this revision. A migration must not import service code that changes after it.
`tests/schema/test_probe_identity_recompute_migration.py` recomputes miStudio's case file through
this copy.

Revision ID: 033
Revises: 032
Create Date: 2026-10-23
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
from typing import Any, Optional

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

JSONVariant = sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True),
                                                      "postgresql")

IDENTITY_ALGORITHM = "mistudio.probe-identity/1"
_SAFE_INTEGER = 2 ** 53 - 1

# ── frozen projection and RFC 8785 serializer (probe_digest.py at 033) ───────────
DETECTOR_V1 = {
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
_BAND = ("min_tokens", "max_tokens", "threshold", "threshold_source", "target_fpr",
         "realised_fpr", "n_negatives")
DECISION_V1 = {
    "": ("threshold", "target_fpr", "realised_fpr", "threshold_source", "calibration",
         "windows", "length_bands"),
    "calibration": ("hf_id", "config", "split", "revision"),
    "windows{}": ("threshold", "target_fpr", "realised_fpr", "n_negatives", "scope",
                  "length_bands"),
    "windows{}.length_bands[]": _BAND,
    "length_bands[]": _BAND,
}
DEFAULTS = {
    "detector.read.hook_point": "resid_post",
    "detector.head.bias": 0.0,
    "detector.sae.normalization": {},
    "detector.aggregation.params": {},
    "detector.render.add_special_tokens": False,
}
PROJECTED = frozenset(DETECTOR_V1[""]) | {"decision"}


class _Unhashable(ValueError):
    pass


def _number(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise _Unhashable("non-finite")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    text = repr(abs(float(value)))
    mantissa, _, exp = text.partition("e")
    whole, _, frac = mantissa.partition(".")
    digits = (whole + frac).lstrip("0")
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


def _jcs(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > _SAFE_INTEGER:
            raise _Unhashable("unsafe integer")
        return str(int(value))
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_jcs(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda k: str(k).encode("utf-16-be"))
        return "{" + ",".join(json.dumps(str(k), ensure_ascii=False) + ":" + _jcs(value[k])
                              for k in keys) + "}"
    raise TypeError(type(value).__name__)


def _refuse(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _Unhashable("non-finite")
        return
    if isinstance(value, int):
        if abs(value) > _SAFE_INTEGER:
            raise _Unhashable("unsafe integer")
        return
    if isinstance(value, dict):
        for item in value.values():
            _refuse(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _refuse(item)


def _join(path: str, key: str) -> str:
    return key if path == "" else f"{path}.{key}"


def _project(value: Any, path: str, table: dict, prefix: str) -> dict:
    if not isinstance(value, dict):
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
            if not isinstance(item, dict):
                raise TypeError(f"{prefix}.{child} is not an object")
            out[key] = {str(name): _project(entry, f"{child}{{}}", table, prefix)
                        for name, entry in item.items()}
        else:
            out[key] = copy.deepcopy(item)
    return out


def _digest(kind: str, payload: Any) -> str:
    wrapper = {"algorithm": IDENTITY_ALGORITHM, "digest": kind, "payload": payload}
    return hashlib.sha256(_jcs(wrapper).encode()).hexdigest()


def digests(definition: Any) -> tuple[Optional[str], Optional[str]]:
    """(detector, decision) under miStudio's /1 projection, or `(None, None)`: both or neither,
    as miStudio's `document_identity` (any failure refuses the whole document)."""
    if not isinstance(definition, dict):
        return None, None
    try:
        _refuse({key: definition.get(key) for key in PROJECTED if key in definition})
        return (
            _digest("detector", _project(definition, "", DETECTOR_V1, "detector")),
            _digest("decision", _project(definition.get("decision") or {}, "", DECISION_V1,
                                         "decision")),
        )
    except (ValueError, TypeError, AttributeError):
        # `_Unhashable` and `UnicodeEncodeError` (a lone surrogate) are both `ValueError`s.
        return None, None


def upgrade() -> None:
    bind = op.get_bind()
    probes = sa.table(
        "probes", sa.column("id", sa.String), sa.column("definition", JSONVariant),
        sa.column("detector_digest", sa.String), sa.column("detector_digest_source", sa.String),
        sa.column("identity_algorithm", sa.String), sa.column("decision_digest", sa.String),
    )
    moved: list[str] = []
    rows = bind.execute(
        sa.select(probes).where(probes.c.detector_digest_source != "producer")
    ).mappings().all()
    for row in rows:
        definition = row["definition"]
        if isinstance(definition, str):
            definition = json.loads(definition)
        det, dec = digests(definition)
        values = {
            "detector_digest": det,
            "decision_digest": dec,
            "detector_digest_source": "computed" if det else "unknown",
            "identity_algorithm": IDENTITY_ALGORITHM if det else None,
        }
        if all(row[key] == value for key, value in values.items()):
            continue
        moved.append(row["id"])
        bind.execute(probes.update().where(probes.c.id == row["id"]).values(**values))
    logger.info("probe_identity_recompute checked=%d moved=%d ids=%s", len(rows), len(moved),
                ",".join(moved) or "-")


def downgrade() -> None:
    """Nothing to undo: the digests 032 wrote were wrong wherever this changed them."""
