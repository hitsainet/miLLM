"""Probe result contract: stored `mi.probe.result/v1` objects and identity digests (Feature 038).

`probe_events` gains the stored contract object (`result`) and its filterable projections
(`result_id`, `trace_id`, `span_id`, `event_time`, `event_time_source`, `outcome`, `reason`,
`status`, `persistence`, the digests and `decision_revision`, `migrated`). `probes` gains the
identity digests (`detector_digest`, `detector_digest_source`, `identity_algorithm`,
`producer_detector_digest`, `decision_digest`, `decision_revision`).

**Backfill, once, for every existing row** (FR-38.10, FTID §4.1):

* `probes`: both digests computed from the stored raw definition, source `computed`.
* `probe_events`, in batches of 2,000 by `id`:
  - `result_id = uuid5(NAMESPACE, "{id}|{probe_id}|{request_id or ''}|{created_at UTC isoformat}")`
    — deterministic, so a downgrade and upgrade mint the same ids (OI-1: the row's own immutable
    fields, because the installation id D38.31 named would have to live in an encrypted table);
    `created_at` is normalised to UTC first, or SQLite and PostgreSQL would mint different ids;
  - `not_scored_reason` REWRITTEN to its code (`encoder_failed: <exception text>` →
    `encoder_failed`), and `summary` of every not-completed row rewritten from that code; the
    free text is not kept (D38.9) and the downgrade does not restore it;
  - `event_time = created_at`, `event_time_source = 'inserted'`, `migrated = true`;
  - the detector digest is the probe's; `decision_digest` only where the event's
    `threshold_revision` equals the probe's current one — otherwise `unknown`, never a guess;
  - `engine` / `execution_mode` are read from the recorded reason where it states them
    (`continuous_batching` → cbm, `engine_unsupported` → llama.cpp, `batched_request` → batched),
    otherwise the serial transformers path every other event was recorded on; provenance that was
    never recorded (`steering_ref`, `parity_status`, `model_load_id`, timing) reads `unknown`/null;
  - `result` = the recorded object WITHOUT `source_event_id`, which is the row's own id
    (FTID X-05, I-38.1).

⚠ FROZEN. The serializer, the projections and the reason table below are COPIES of
`millm/services/probe_digest.py` and `probe_reasons.py` as of this revision. A migration must not
import service code that changes after it.

Revision ID: 032
Revises: 031
Create Date: 2026-10-23
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None

JSONVariant = sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True),
                                                      "postgresql")

#: Fixed namespace for migrated result ids. Never change it: a downgrade/upgrade must re-mint the
#: same ids (FR-38.10).
RESULT_NAMESPACE = uuid.UUID("0c8c3c4e-5b8f-5f0b-8a32-038032032032")
BATCH = 2000
PRODUCER_VERSION = "0.5.0"  # the version every pre-032 event was recorded under
IDENTITY_ALGORITHM = "mistudio.probe-identity/1"

# ── frozen reason table (probe_reasons.py at 032) ────────────────────────────────
OUTCOME_OF = {
    "judge_traffic": "skipped", "batched_request": "skipped", "continuous_batching": "skipped",
    "engine_unsupported": "skipped", "speculative_decoding": "skipped",
    "prompt_boundary_unknown": "skipped", "last_user_span_unknown": "skipped",
    "last_user_span_unresolved": "skipped", "no_user_turn": "skipped",
    "no_user_header": "skipped", "no_chat_template": "skipped",
    "text_completion_has_no_user_turn": "skipped", "no_scored_tokens": "skipped",
    "prompt_boundary_invalid": "failed", "prompt_boundary_conflict": "failed",
    "encoder_failed": "failed", "head_mismatch": "failed", "non_finite_score": "failed",
    "client_disconnected": "interrupted", "cancelled": "interrupted",
    "generation_error": "interrupted", "generation_thread_hung": "interrupted",
}
_REASON = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_NAME = re.compile(r"^[A-Za-z0-9._:+\-() ]{1,200}$")
_MODEL = re.compile(r"^([A-Za-z0-9._:/+@\- ]{1,255})$")
_REVISION = re.compile(r"^([0-9a-f]{40}|[A-Za-z0-9._/\-]{1,100})$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PRECISION = re.compile(r"^[a-z0-9_]{1,32}$")
_BEHAVIOUR = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$")
RUNG_LANGUAGE = {
    0: "trained", 1: "detects on held-out data", 2: "detects on unseen tasks",
    3: "detects on unseen tasks, compared with a judge",
}
RULES = ("mean", "max", "last", "softmax", "attention", "rolling_mean_max")

# ── frozen RFC 8785 serializer and projections (probe_digest.py at 032) ──────────
DETECTOR_V1 = {
    "model": ("hf_id", "revision", "d_model", "n_layers", "architecture",
              "chat_template_sha256", "load_dtype", "quantization"),
    "read": ("layer", "hook_point"),
    "scope": None,
    "basis": None,
    "head": ("weights", "bias", "norm_mean", "norm_std", "attention_query"),
    "sae": ("hf_repo", "path", "revision", "weights_sha256", "architecture", "d_model",
            "n_features", "normalization", "feature_indices"),
    "aggregation": ("rule", "params", "streamable"),
    "render": ("generation_prompt", "add_special_tokens", "bos_handling"),
}
NULLABLE_BLOCKS = frozenset({"sae", "render"})
DEFAULTS = {"head.bias": 0.0, "read.hook_point": "resid_post", "render.add_special_tokens": False,
            "aggregation.params": {}, "sae.normalization": {}}
DECISION_KEYS = ("threshold", "target_fpr", "realised_fpr", "threshold_source", "calibration",
                 "windows", "length_bands")
CALIBRATION_KEYS = ("hf_id", "config", "split", "revision")
WINDOW_KEYS = ("threshold", "target_fpr", "realised_fpr", "n_negatives", "scope", "length_bands")
BAND_KEYS = ("min_tokens", "max_tokens", "threshold", "threshold_source", "target_fpr",
             "realised_fpr", "n_negatives")


def _number(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("non-finite")
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
        return _number(float(value)) if abs(value) >= 2 ** 53 else str(int(value))
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


def _fill(block: Any, keys: tuple, prefix: str) -> dict:
    block = block if isinstance(block, dict) else {}
    return {key: block[key] if key in block and block[key] is not None
            else DEFAULTS.get(f"{prefix}.{key}") for key in keys}


def _detector_payload(definition: dict) -> dict:
    out: dict[str, Any] = {}
    for key, fields in DETECTOR_V1.items():
        value = definition.get(key)
        if fields is None:
            out[key] = value
        elif key in NULLABLE_BLOCKS and value is None:
            out[key] = None
        else:
            out[key] = _fill(value, fields, key)
    return out


def _bands(value: Any) -> Any:
    return None if value is None else [_fill(band, BAND_KEYS, "band") for band in value]


def _decision_payload(decision: Any) -> dict:
    decision = decision if isinstance(decision, dict) else {}
    out = {key: decision.get(key) for key in DECISION_KEYS}
    if isinstance(out["calibration"], dict):
        out["calibration"] = {k: out["calibration"].get(k) for k in CALIBRATION_KEYS}
    if isinstance(out["windows"], dict):
        out["windows"] = {name: {**_fill(entry, WINDOW_KEYS, "window"),
                                 "length_bands": _bands((entry or {}).get("length_bands"))}
                          for name, entry in out["windows"].items()}
    out["length_bands"] = _bands(out["length_bands"])
    return out


def _digest(kind: str, payload: Any) -> str:
    wrapper = {"algorithm": IDENTITY_ALGORITHM, "digest": kind, "payload": payload}
    return hashlib.sha256(_jcs(wrapper).encode()).hexdigest()


def digests(definition: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(definition, dict):
        return None, None
    out: list = []
    for kind, build in (("detector", lambda: _detector_payload(definition)),
                        ("decision", lambda: _decision_payload(definition.get("decision")))):
        try:
            out.append(_digest(kind, build()))
        except (ValueError, TypeError):
            out.append(None)
    return out[0], out[1]


# ── the row mapping ──────────────────────────────────────────────────────────────


def utc(moment: Any) -> datetime:
    """`created_at` as an aware UTC datetime. SQLite returns it naive (it was stored as UTC);
    PostgreSQL returns it aware. Without this the two would mint different ids for one row."""
    if isinstance(moment, str):
        moment = datetime.fromisoformat(moment)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def result_id_for(row_id: int, probe_id: str, request_id: Optional[str], created_at: Any) -> str:
    return str(uuid.uuid5(
        RESULT_NAMESPACE, f"{row_id}|{probe_id}|{request_id or ''}|{utc(created_at).isoformat()}"
    ))


def reason_code(text: Optional[str]) -> Optional[str]:
    """`encoder_failed: <exception text>` → `encoder_failed`; a code stays itself."""
    if text is None:
        return None
    head = str(text).split(":", 1)[0].strip()
    return head if _REASON.match(head) else "unlisted_reason"


def _rfc3339(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _or_unknown(value: Any, pattern: "re.Pattern[str]") -> str:
    return value if isinstance(value, str) and pattern.match(value) else "unknown"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def migrated_row(event: dict, probe: Optional[dict]) -> dict:
    """The column values for one existing event: the recorded result and its projections."""
    created = utc(event["created_at"])
    reason = None
    score = event["score"] if _finite(event["score"]) else None
    if event["scored"] and score is not None:
        outcome, scored = "completed", True
    elif event["scored"]:
        outcome, scored, reason = "failed", False, "non_finite_score"
    else:
        scored = False
        reason = reason_code(event["not_scored_reason"]) or "unlisted_reason"
        # A code outside the list is a defect path, reported as a failure, never dressed up as a
        # skip — the builder's rule (`probe_result._one`), review R1.
        outcome = OUTCOME_OF.get(reason, "failed")
    threshold = event["threshold"] if _finite(event["threshold"]) else None
    verdict = event["verdict"] if outcome == "completed" and threshold is not None else None
    definition = (probe or {}).get("definition") or {}
    det, dec = (probe or {}).get("detector_digest"), (probe or {}).get("decision_digest")
    same_bar = probe is not None and int(event["threshold_revision"] or 1) == int(
        probe.get("threshold_revision") or 1)
    behaviour = definition.get("behaviour") if isinstance(definition, dict) else None
    key = behaviour.get("key") if isinstance(behaviour, dict) else None
    version = behaviour.get("version") if isinstance(behaviour, dict) else None
    engine, mode = "transformers", "serial"
    if reason == "continuous_batching":
        engine, mode = "cbm", "continuous_batching"
    elif reason == "engine_unsupported":
        engine = "llamacpp"
    elif reason == "batched_request":
        mode = "batched"
    rung = int(event["rung"]) if event["rung"] is not None else 0
    name = (probe or {}).get("name")
    target_fpr = (probe or {}).get("target_fpr")
    stamp = _rfc3339(created)
    result = {
        "schema": "mi.probe.result/v1", "kind": "probe",
        "result_id": result_id_for(event["id"], event["probe_id"], event["request_id"], created),
        "event_time": stamp, "event_time_source": "inserted",
        "origin": event["origin"] if event["origin"] in ("live", "batch") else "live",
        "producer": {"name": "millm", "version": PRODUCER_VERSION},
        "migrated": True,
        "request_id": event["request_id"] if isinstance(event["request_id"], str)
        and _ID.match(event["request_id"]) else None,
        "trace_id": None, "span_id": None,
        "batch_id": event["batch_id"] if isinstance(event["batch_id"], str)
        and _ID.match(event["batch_id"]) else None,
        "batch_line": event["batch_line"],
        "session_id": event["session_id"] if isinstance(event["session_id"], str)
        and _ID.match(event["session_id"]) else None,
        "turn_index": event["turn_index"],
        "behaviour": {"key": key, "version": version if isinstance(version, int)
                      and not isinstance(version, bool) and version >= 1 else None}
        if isinstance(key, str) and _BEHAVIOUR.match(key) else None,
        "set": None,
        "outcome": outcome, "reason": reason, "persistence": "persisted",
        "window": event["window"] if event["window"] in ("all", "prompt", "response", "last_user")
        else "all",
        "status": "final",
        "choice_index": None, "input_index": None,
        "probe_id": event["probe_id"],
        "probe_name": name if isinstance(name, str) and _NAME.match(name) else None,
        "detector_digest": det if det else "unknown",
        "detector_digest_source": "computed" if det else "unknown",
        "identity_algorithm": IDENTITY_ALGORITHM if det else None,
        "definition_schema": "mistudio.probe-definition/v1",
        "target": "positive_class" if probe is not None else "unknown",
        "score_semantics": "raw_margin",
        "layer": (probe or {}).get("layer"),
        "hook": ((definition.get("read") or {}).get("hook_point") or "resid_post")
        if probe is not None else "unknown",
        "basis": (probe or {}).get("basis") or "unknown",
        "rule": (probe or {}).get("rule") if (probe or {}).get("rule") in RULES else "unknown",
        "served_model": _or_unknown(event["served_model"], _MODEL),
        "model_revision": _or_unknown((probe or {}).get("revision"), _REVISION),
        "template_sha256": _or_unknown((probe or {}).get("template_sha256"), _HEX64),
        "precision": _or_unknown((definition.get("model") or {}).get("load_dtype"), _PRECISION),
        "engine": engine, "execution_mode": mode,
        "steering_ref": {"kind": "unknown", "id": None},
        "definition_scope": (probe or {}).get("scope") or "unknown",
        "n_scored_tokens": int(event["n_scored_tokens"] or 0) if score is not None else 0,
        "scored": scored,
        "score": score if outcome == "completed" else None,
        "probability": None, "calibration_ref": None,
        "threshold": threshold, "verdict": verdict, "fires": verdict, "comparator": ">=",
        "decision_digest": dec if (dec and same_bar) else "unknown",
        "decision_revision": None,
        "threshold_revision": max(1, int(event["threshold_revision"] or 1)),
        "decision": {
            "calibration_id": (probe or {}).get("threshold_calibration_id") if same_bar
            and isinstance((probe or {}).get("threshold_calibration_id"), str)
            and _ID.match(probe["threshold_calibration_id"]) else None,
            "target_fpr": target_fpr if same_bar and _finite(target_fpr)
            and 0 <= target_fpr <= 1 else None,
            "provisional": bool(event["provisional"]),
        },
        "rung": rung, "rung_language": RUNG_LANGUAGE.get(rung, "trained"),
        "parity_status": "unknown", "model_load_id": "unknown",
        "evaluation_ms": None, "completed_at": stamp,
    }
    return {
        "result": result, "result_id": result["result_id"], "event_time": created,
        "event_time_source": "inserted", "outcome": outcome, "reason": reason,
        "not_scored_reason": None if outcome == "completed" else reason,
        # ⚠ REVIEW R1: `summary` was `not scored: <not_scored_reason>` — for an encoder failure,
        # the exception text this migration removes from the reason. Rewritten from the code, as
        # `probe_result.summary_of` writes it, or the free text survived on the events list.
        **({} if outcome == "completed" else {"summary": f"not scored: {reason}"}),
        "score": result["score"], "verdict": verdict,
        "status": "final", "persistence": "persisted",
        "detector_digest": det, "detector_digest_source": "computed" if det else "unknown",
        "identity_algorithm": IDENTITY_ALGORITHM if det else None,
        "decision_digest": dec if (dec and same_bar) else None,
        "migrated": True,
    }


# ── schema ───────────────────────────────────────────────────────────────────────

EVENT_COLUMNS = [
    ("result", JSONVariant, None),
    ("result_id", sa.String(length=36), None),
    ("trace_id", sa.String(length=32), None),
    ("span_id", sa.String(length=16), None),
    ("event_time", sa.DateTime(timezone=True), None),
    ("event_time_source", sa.String(length=9), "finalized"),
    ("outcome", sa.String(length=12), None),
    ("reason", sa.String(length=64), None),
    ("status", sa.String(length=12), "final"),
    ("persistence", sa.String(length=14), "persisted"),
    ("detector_digest", sa.String(length=64), None),
    ("detector_digest_source", sa.String(length=8), "unknown"),
    ("identity_algorithm", sa.String(length=40), None),
    ("decision_digest", sa.String(length=64), None),
    ("decision_revision", sa.Integer(), None),
    ("migrated", sa.Boolean(), sa.false()),
]
PROBE_COLUMNS = [
    ("detector_digest", sa.String(length=64), None),
    ("detector_digest_source", sa.String(length=8), "unknown"),
    ("identity_algorithm", sa.String(length=40), None),
    ("producer_detector_digest", sa.String(length=64), None),
    ("decision_digest", sa.String(length=64), None),
    ("decision_revision", sa.Integer(), None),
]
NOT_NULL_AFTER_BACKFILL = [
    ("result", JSONVariant), ("result_id", sa.String(length=36)),
    ("event_time", sa.DateTime(timezone=True)), ("outcome", sa.String(length=12)),
]
INDEXES = [
    ("uq_probe_events_result_id", ["result_id"], True),
    ("ix_probe_events_trace_id", ["trace_id"], False),
    ("ix_probe_events_event_time", ["event_time"], False),
    ("ix_probe_events_outcome", ["outcome"], False),
    ("ix_probe_events_detector_digest", ["detector_digest"], False),
]


def _add(table: str, columns: list) -> None:
    """Nullable for now, or NOT NULL with its server default (the ORM declares the same)."""
    for name, type_, default in columns:
        if default is None:
            op.add_column(table, sa.Column(name, type_, nullable=True))
        else:
            op.add_column(table, sa.Column(name, type_, nullable=False, server_default=default))


def upgrade() -> None:
    _add("probes", PROBE_COLUMNS)
    _add("probe_events", EVENT_COLUMNS)
    bind = op.get_bind()

    probes = sa.table(
        "probes", sa.column("id", sa.String), sa.column("name", sa.String),
        sa.column("definition", JSONVariant), sa.column("threshold_revision", sa.Integer),
        sa.column("threshold_calibration_id", sa.String), sa.column("target_fpr", sa.Float),
        sa.column("layer", sa.Integer), sa.column("basis", sa.String),
        sa.column("rule", sa.String), sa.column("scope", sa.String),
        sa.column("revision", sa.String), sa.column("template_sha256", sa.String),
        sa.column("detector_digest", sa.String), sa.column("detector_digest_source", sa.String),
        sa.column("identity_algorithm", sa.String), sa.column("decision_digest", sa.String),
    )
    by_id: dict[str, dict] = {}
    for row in bind.execute(sa.select(probes)).mappings():
        probe = dict(row)
        definition = probe["definition"]
        if isinstance(definition, str):
            definition = json.loads(definition)
            probe["definition"] = definition
        det, dec = digests(definition)
        probe["detector_digest"], probe["decision_digest"] = det, dec
        by_id[probe["id"]] = probe
        bind.execute(probes.update().where(probes.c.id == probe["id"]).values(
            detector_digest=det, decision_digest=dec,
            detector_digest_source="computed" if det else "unknown",
            identity_algorithm=IDENTITY_ALGORITHM if det else None,
        ))

    events = sa.table(
        "probe_events", sa.column("id", sa.Integer), sa.column("probe_id", sa.String),
        sa.column("request_id", sa.String), sa.column("scored", sa.Boolean),
        sa.column("not_scored_reason", sa.String), sa.column("score", sa.Float),
        sa.column("threshold", sa.Float), sa.column("verdict", sa.Boolean),
        sa.column("rung", sa.Integer), sa.column("n_scored_tokens", sa.Integer),
        sa.column("window", sa.String), sa.column("provisional", sa.Boolean),
        sa.column("threshold_revision", sa.Integer), sa.column("served_model", sa.String),
        sa.column("origin", sa.String), sa.column("batch_id", sa.String),
        sa.column("batch_line", sa.Integer), sa.column("session_id", sa.String),
        sa.column("turn_index", sa.Integer), sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("summary", sa.String), sa.column("result", JSONVariant), sa.column("result_id", sa.String),
        sa.column("event_time", sa.DateTime(timezone=True)),
        sa.column("event_time_source", sa.String), sa.column("outcome", sa.String),
        sa.column("reason", sa.String), sa.column("status", sa.String),
        sa.column("persistence", sa.String), sa.column("detector_digest", sa.String),
        sa.column("detector_digest_source", sa.String), sa.column("identity_algorithm", sa.String),
        sa.column("decision_digest", sa.String), sa.column("migrated", sa.Boolean),
    )
    read = [events.c[name] for name in (
        "id", "probe_id", "request_id", "scored", "not_scored_reason", "score", "threshold",
        "verdict", "rung", "n_scored_tokens", "window", "provisional", "threshold_revision",
        "served_model", "origin", "batch_id", "batch_line", "session_id", "turn_index",
        "created_at")]
    last = 0
    while True:
        batch = [dict(r) for r in bind.execute(
            sa.select(*read).where(events.c.id > last).order_by(events.c.id).limit(BATCH)
        ).mappings()]
        if not batch:
            break
        for event in batch:
            values = migrated_row(event, by_id.get(event["probe_id"]))
            bind.execute(events.update().where(events.c.id == event["id"]).values(**values))
        last = batch[-1]["id"]

    with op.batch_alter_table("probe_events") as batch_op:
        for name, type_ in NOT_NULL_AFTER_BACKFILL:
            batch_op.alter_column(name, existing_type=type_, nullable=False)
    for name, columns, unique in INDEXES:
        op.create_index(name, "probe_events", columns, unique=unique)


def downgrade() -> None:
    for name, _columns, _unique in INDEXES:
        op.drop_index(name, table_name="probe_events")
    with op.batch_alter_table("probe_events") as batch_op:
        for name, _type, _default in reversed(EVENT_COLUMNS):
            batch_op.drop_column(name)
    with op.batch_alter_table("probes") as batch_op:
        for name, _type, _default in reversed(PROBE_COLUMNS):
            batch_op.drop_column(name)
