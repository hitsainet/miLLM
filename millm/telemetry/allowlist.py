"""Privacy by filter (FR-38.21, FTDD §8): the ONLY attributes any exported signal may carry.

Every span, log record, metric dimension and resource attribute passes `filter(signal, attrs)`
BEFORE it is queued. A key not listed for its signal, or a value that does not match its key's
pattern or type, is dropped and counted (`attributes_dropped` in health) — never queued, never
sent. The list is written out by hand on purpose: a field added to the result contract later is
dropped (and counted) until someone decides here that it may leave the deployment.

⚠ WHAT CAN NEVER PASS. No key below can hold free text: every string value is an enum or a
pattern of identifier characters, so a prompt, a reply, decoded context, an exception message, a
header value or an acknowledgement `reason` is refused even if a caller put it under a listed key
(the privacy sentinels test plants each of them, AT10). Numbers and booleans are type-checked; a
bool is not accepted where a number is expected and vice versa.

No OpenTelemetry import (D38.17).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from typing import Any

from millm.api.schemas.probe_result import (
    BASES,
    BEHAVIOUR_KEY_PATTERN,
    DIGEST_OR_UNKNOWN,
    DIGEST_SOURCES,
    ENGINES,
    EXECUTION_MODES,
    HOOKS,
    ID_PATTERN,
    MODEL_NAME_PATTERN,
    NAME_PATTERN,
    ORIGINS,
    PARITY,
    PERSISTENCE,
    PRECISION_PATTERN,
    REVISION_PATTERN,
    RFC3339_UTC,
    RULE_NAMES,
    SEMVER_PATTERN,
    STATUSES,
    STEERING_KINDS,
    TARGETS,
    UUID_PATTERN,
    WINDOW_NAMES,
)
from millm.services.probe_reasons import REASON_PATTERN

Check = Callable[[Any], bool]


def _pattern(pattern: str) -> Check:
    rx = re.compile(pattern)
    return lambda v: isinstance(v, str) and len(v) <= 255 and rx.fullmatch(v) is not None


def _enum(*values: str) -> Check:
    allowed = frozenset(values)
    return lambda v: isinstance(v, str) and v in allowed


def _int(lo: int = 0, hi: int = 2**53) -> Check:
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _number(v: Any) -> bool:
    import math

    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _bool(v: Any) -> bool:
    return isinstance(v, bool)


_ID = _pattern(ID_PATTERN)
_HEX32 = _pattern(r"^[0-9a-f]{32}$")
_HEX16 = _pattern(r"^[0-9a-f]{16}$")
_DIGEST = _pattern(DIGEST_OR_UNKNOWN)
_MODEL = _pattern(MODEL_NAME_PATTERN)
#: An error CLASS name (`TimeoutError`, `requests.exceptions.ConnectionError`): never its message.
ERROR_CLASS = _pattern(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
#: A lifecycle `code`: a refusal code (`PROBE_PARITY_FAILED`) or a lower-case cause
#: (`model_unloaded`).
_CODE = _pattern(r"^([A-Z][A-Z0-9_]{0,63}|[a-z][a-z0-9_]{0,63}|[A-Z][A-Za-z0-9]{0,63})$")
_RUNG_LANGUAGE = _pattern(r"^[a-z0-9 ,()'.\-]{1,80}$")

# ── resource (built from settings only) ──────────────────────────────────────────

RESOURCE: dict[str, Check] = {
    "service.name": _pattern(r"^[A-Za-z0-9._\-]{1,64}$"),
    "service.version": _pattern(SEMVER_PATTERN),
    "service.instance.id": _pattern(r"^[A-Za-z0-9._:\-]{1,64}$"),
    "deployment.environment.name": _pattern(r"^[A-Za-z0-9._\-]{1,64}$"),
}

# ── spans (FTDD §6.3) ────────────────────────────────────────────────────────────

OPERATIONS = ("chat", "text_completion", "millm.probe_score")
#: The inference span is named by its GenAI operation (`chat`, ...): the convention's
#: `{operation} {model}` form would put a value in the name, where no allowlist can see it.
SPAN_NAMES = frozenset({*OPERATIONS, "millm.queue.wait", "millm.probe.evaluate"})

SPAN: dict[str, Check] = {
    "gen_ai.operation.name": _enum(*OPERATIONS),
    "gen_ai.request.model": _MODEL,
    "millm.engine": _enum(*ENGINES),
    "millm.execution_mode": _enum(*EXECUTION_MODES),
    "millm.stream": _bool,
    "millm.queue.position_at_entry": _int(),
    "millm.probe.results.completed": _int(),
    "millm.probe.results.skipped": _int(),
    "millm.probe.results.failed": _int(),
    "millm.probe.results.interrupted": _int(),
    "millm.batch.id": _ID,
    "error.type": ERROR_CLASS,
}

# ── result log records: the contract flattened under `mi.probe.*` (FTDD §6.5) ─────────

_RESULT_FIELDS: dict[str, Check] = {
    "schema": _enum("mi.probe.result/v1"),
    "kind": _enum("probe", "rule"),
    "result_id": _pattern(UUID_PATTERN),
    "event_time": _pattern(RFC3339_UTC),
    "event_time_source": _enum("finalized", "inserted"),
    "completed_at": _pattern(RFC3339_UTC),
    "origin": _enum(*ORIGINS),
    "producer.name": _enum("millm"),
    "producer.version": _pattern(SEMVER_PATTERN),
    "migrated": _bool,
    "request_id": _ID,
    "trace_id": _HEX32,
    "span_id": _HEX16,
    "source_event_id": _int(1),
    "batch_id": _ID,
    "batch_line": _int(1),
    "session_id": _ID,
    "turn_index": _int(),
    "behaviour.key": _pattern(BEHAVIOUR_KEY_PATTERN),
    "behaviour.version": _int(1),
    # 039 S1 (FR-39.10): the behaviour's `set` block — its counts and digest. Its two LISTS
    # (`missing_probe_ids`, `attached_rule_sets`) never leave: `emit_results` removes them before
    # flattening, as no contract list may (`flatten`).
    "set.armed_members": _int(),
    "set.total_members": _int(),
    "set.complete": _bool,
    "set.members_digest": _DIGEST,
    "outcome": _enum("completed", "skipped", "failed", "interrupted"),
    "reason": _pattern(REASON_PATTERN),
    "persistence": _enum(*PERSISTENCE),
    "window": _enum(*WINDOW_NAMES),
    "status": _enum(*STATUSES),
    "choice_index": _int(),
    "input_index": _int(),
    "probe_id": _ID,
    "probe_name": _pattern(NAME_PATTERN),
    "detector_digest": _DIGEST,
    "detector_digest_source": _enum(*DIGEST_SOURCES),
    "identity_algorithm": _enum("mistudio.probe-identity/1"),
    "definition_schema": _enum("mistudio.probe-definition/v1"),
    "target": _enum(*TARGETS),
    "score_semantics": _enum("raw_margin"),
    "layer": _int(),
    "hook": _enum(*HOOKS),
    "basis": _enum(*BASES),
    "rule": _enum(*RULE_NAMES),
    "served_model": _MODEL,
    "model_revision": _pattern(REVISION_PATTERN),
    "template_sha256": _DIGEST,
    "precision": _pattern(PRECISION_PATTERN),
    "engine": _enum(*ENGINES),
    "execution_mode": _enum(*EXECUTION_MODES),
    "steering_ref.kind": _enum(*STEERING_KINDS),
    "steering_ref.id": _ID,
    "definition_scope": _enum(*WINDOW_NAMES, "unknown"),
    "n_scored_tokens": _int(),
    "scored": _bool,
    "score": _number,
    "threshold": _number,
    "verdict": _bool,
    "fires": _bool,
    "comparator": _enum(">="),
    "decision_digest": _DIGEST,
    "decision_revision": _int(1),
    "threshold_revision": _int(1),
    "decision.calibration_id": _ID,
    "decision.target_fpr": _number,
    "decision.provisional": _bool,
    "rung": _int(0, 3),
    "rung_language": _RUNG_LANGUAGE,
    "parity_status": _enum(*PARITY),
    "model_load_id": _pattern(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|unknown)$"),
    "evaluation_ms": _number,
    # 039 S2: a `kind: "rule"` result — ids, a digest, a form and a COUNT; never a value.
    "rule_set_id": _ID,
    "rule_set_digest": _DIGEST,
    "rule_set_version": _int(1),
    "signal": _ID,
    "form": _pattern(r"^[a-z][a-z0-9_.\-]{0,63}$"),
    "count": _int(),
}
RESULT_PREFIX = "mi.probe."
LOG_RESULT: dict[str, Check] = {RESULT_PREFIX + k: v for k, v in _RESULT_FIELDS.items()}

# ── lifecycle log records (FTDD §6.5) ────────────────────────────────────────────

LIFECYCLE_ACTIONS = ("import", "arm", "arm_refused", "disarm", "parity", "recalibrate", "delete")
LOG_LIFECYCLE: dict[str, Check] = {
    "mi.lifecycle.subject_kind": _enum("probe", "probe_set", "rule_set"),
    "mi.lifecycle.action": _enum(*LIFECYCLE_ACTIONS),
    "mi.lifecycle.outcome": _enum("ok", "refused", "failed"),
    "mi.lifecycle.code": _CODE,
    "mi.lifecycle.probe_id": _ID,
    "mi.lifecycle.behaviour": _pattern(BEHAVIOUR_KEY_PATTERN),
    "mi.lifecycle.detector_digest": _DIGEST,
    "mi.lifecycle.threshold_revision": _int(1),
    "mi.lifecycle.decision_revision": _int(1),
    "mi.lifecycle.decision_digest": _DIGEST,
    "mi.lifecycle.previous_threshold_revision": _int(1),
    "mi.lifecycle.previous_decision_digest": _DIGEST,
}

EVENT_NAMES = frozenset({"mi.probe.result", "mi.probe.lifecycle"})

# ── metric dimensions (FTDD §6.4) ────────────────────────────────────────────────

METRIC: dict[str, Check] = {
    "probe_id": _ID,
    "behaviour": _pattern(r"^(none|[a-z0-9][a-z0-9-]{0,62}[a-z0-9])$"),
    "kind": _enum("probe", "rule"),
    "window": _enum(*WINDOW_NAMES),
    "engine": _enum(*ENGINES),
    "execution_mode": _enum(*EXECUTION_MODES),
    "origin": _enum(*ORIGINS),
    "outcome": _enum("completed", "skipped", "failed", "interrupted"),
    "reason": _pattern(r"^(none|" + REASON_PATTERN.strip("^$") + r")$"),
    "otel.metric.overflow": _bool,
}

SIGNALS: dict[str, dict[str, Check]] = {
    "resource": RESOURCE,
    "span": SPAN,
    "log_result": LOG_RESULT,
    "log_lifecycle": LOG_LIFECYCLE,
    "metric": METRIC,
}

# ── the counter of what was refused (FR-38.21) ───────────────────────────────────

_dropped_lock = threading.Lock()
_dropped = 0


def attributes_dropped() -> int:
    return _dropped


def _count(n: int) -> None:
    global _dropped
    if n:
        with _dropped_lock:
            _dropped += n


def reset_for_tests() -> None:
    global _dropped
    with _dropped_lock:
        _dropped = 0


def filter(signal: str, attrs: Mapping[str, Any]) -> tuple[dict[str, Any], int]:  # noqa: A001
    """`(kept, dropped)`. Pure apart from the global drop counter, which it increments.

    ⚠ A value is kept only when its KEY is listed for this signal AND the value passes that key's
    check. An unknown signal keeps nothing.
    """
    table = SIGNALS.get(signal, {})
    kept: dict[str, Any] = {}
    dropped = 0
    for key, value in attrs.items():
        check = table.get(key)
        if check is not None and value is not None and check(value):
            kept[key] = value
        elif value is not None:
            dropped += 1
    _count(dropped)
    return kept, dropped


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """`{"a": {"b": 1}}` → `{"a.b": 1}`; lists and `None` are not carried (OTel attributes hold
    no null, and no contract list may leave)."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, name + "."))
        elif isinstance(value, (list, tuple)):
            if value:
                out[name] = value  # unlisted shape → dropped and counted by `filter`
        elif value is not None:
            out[name] = value
    return out
