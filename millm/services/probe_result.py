"""THE builder of `mi.probe.result/v1` objects (FR-38.3, D38.18, D38.23, D38.25).

Every surface a probe result reaches takes it from here:

* `build_probe_results` makes the FINALIZED objects, once per (probe × window), in `_probe_finish`
  (or in the interrupted path's record). It mints `result_id` exactly once.
* `recorded` makes the RECORDED copy — the same object plus `persistence` and, once the row exists,
  `source_event_id` — for the stored row, the socket, the events API, session cells and telemetry.
* `legacy_projection` and `legacy_columns` are the ONLY two mappings from a result to the names
  surfaces used before 038 (`name`, `fires`, `not_scored_reason`, ...). A surface payload is
  `{**legacy_projection(r), **wire(r)}`, so where a key exists in both the two cannot disagree.

⚠ NO DATABASE, NO PROBE ROW. The builder reads the `Verdict`, the armed probe's `ProbeDescriptor`
(resolved at arm time), the context's trace, execution facts, session and interruption, the Batch
row, the steering report and the loaded model's identity — nothing else. A result therefore
describes the detector and the bar that were ARMED, which is what judged it.

⚠ IT NEVER RAISES TO A SERVING PATH IN PRODUCTION. `_enforce` checks the conditional rules of FTDD
§5.1; under `MILLM_STRICT_RESULTS=1` (set by `tests/conftest.py`) a violation raises, so a defect
is loud in tests; otherwise it is logged.
"""

from __future__ import annotations

import contextvars
import logging
import math
import os
import re
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

from millm.api.schemas.probe_result import (
    BASES,
    BEHAVIOUR_KEY_PATTERN,
    DIGEST_SOURCES,
    HOOKS,
    IDENTITY_ALGORITHM,
    PARITY,
    RULE_NAMES,
    TARGETS,
    ID_PATTERN,
    MODEL_NAME_PATTERN,
    NAME_PATTERN,
    PRECISION_PATTERN,
    REVISION_PATTERN,
    BehaviourOut,
    DecisionOut,
    ProbeResult,
    RuleResult,
    Producer,
    SteeringRef,
    wire,
)
from millm.services.probe_reasons import OUTCOME_OF, REASON_PATTERN

logger = logging.getLogger(__name__)

#: UUIDv5 namespace for `model_load_id` (one per (model id, loaded_at)).
MODEL_LOAD_NAMESPACE = uuid.UUID("6f1c2a52-3b6e-5d0a-9c1e-038a0de1d0ad")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(ID_PATTERN)
_NAME = re.compile(NAME_PATTERN)
_MODEL = re.compile(MODEL_NAME_PATTERN)
_REVISION = re.compile(REVISION_PATTERN)
_PRECISION = re.compile(PRECISION_PATTERN)
_BEHAVIOUR = re.compile(BEHAVIOUR_KEY_PATTERN)
_REASON = re.compile(REASON_PATTERN)
_RUNG_LANGUAGE = re.compile(r"^[a-z0-9 ,()'.\-]{1,80}$")

#: How many results reported `probe_name: null` because the stored name falls outside the
#: identifier pattern (OI-3, D38.59). Counted, never silent.
names_outside_pattern = 0


# ── the context variable (beside `_PROBE_VERDICTS`) ──────────────────────────────

#: This request's FINALIZED results, published by `_probe_finish` and read by the header, the body
#: field and the Batch line. ⚠ Reset in `reset_steering_memo` beside `_PROBE_VERDICTS` (FTID X-11):
#: packed Batch rows run in one task, and without the reset row N would carry row N-1's results.
_PROBE_RESULTS: "contextvars.ContextVar[list]" = contextvars.ContextVar(
    "millm_probe_results", default=[]
)


def set_probe_results(results: Sequence[Any]) -> None:
    _PROBE_RESULTS.set(list(results))


def get_probe_results() -> list:
    """This request's finalized results. Empty when nothing was armed."""
    return list(_PROBE_RESULTS.get())


def reset_probe_results() -> None:
    _PROBE_RESULTS.set([])


# ── ids ──────────────────────────────────────────────────────────────────────────

_uuid7_lock = threading.Lock()
_uuid7_last_ms = 0
_uuid7_seq = 0


def uuid7() -> uuid.UUID:
    """A UUIDv7 (RFC 9562 §5.7): 48-bit Unix ms, version 7, a 12-bit counter, variant, 62 random
    bits. Python 3.13's `uuid` has no `uuid7` (I-38.7). Monotonic within this process: the counter
    orders ids minted in one millisecond, and a clock step back never reorders them."""
    global _uuid7_last_ms, _uuid7_seq
    with _uuid7_lock:
        ms = time.time_ns() // 1_000_000
        if ms <= _uuid7_last_ms:
            ms = _uuid7_last_ms
            _uuid7_seq += 1
            if _uuid7_seq > 0xFFF:
                ms += 1
                _uuid7_seq = 0
        else:
            _uuid7_seq = 0
        _uuid7_last_ms = ms
        seq = _uuid7_seq
    rand_b = secrets.randbits(62)
    value = (ms & ((1 << 48) - 1)) << 80 | 0x7 << 76 | seq << 64 | 0b10 << 62 | rand_b
    return uuid.UUID(int=value)


def rfc3339(moment: datetime) -> str:
    """UTC, microseconds, `Z` — the one timestamp form every copy of a result carries."""
    moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _strict() -> bool:
    return os.environ.get("MILLM_STRICT_RESULTS") == "1"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _or_unknown(value: Any, pattern: "re.Pattern[str]") -> str:
    return value if isinstance(value, str) and pattern.match(value) else "unknown"


#: `definition_scope`'s listed values (the model's Literal).
_SCOPES = ("all", "prompt", "response", "last_user", "unknown")


def _listed(value: Any, allowed: Sequence[str]) -> str:
    """A listed value, or `unknown`.

    ⚠ REVIEW R1: these fields are Literals on the model, read from the ARMED probe. A value the
    importer accepts but the contract does not list (a rule, basis or scope miStudio adds before
    this list learns it) used to raise `ValidationError` inside the builder: the response fell
    back to the old shape, and `record()` — which builds outside its own try — lost EVERY event
    of every request for that probe, with one warning line. `unknown` is the contract's honest
    word for a value it cannot name.
    """
    return value if isinstance(value, str) and value in allowed else "unknown"


def _producer() -> Producer:
    from millm import __version__

    return Producer(version=__version__)


def _model_load_id() -> str:
    try:
        from millm.ml.model_loader import LoadedModelState

        current = LoadedModelState().current
        if current is None:
            return "unknown"
        loaded_at = getattr(current, "loaded_at", None)
        stamp = loaded_at.isoformat() if hasattr(loaded_at, "isoformat") else str(loaded_at)
        return str(uuid.uuid5(MODEL_LOAD_NAMESPACE, f"{current.model_id}|{stamp}"))
    except Exception:  # noqa: BLE001 - an observer never fails the request
        return "unknown"


def _steering_ref() -> SteeringRef:
    """From the request's published steering report; `unknown` when there is none (FR-38.6).

    ⚠ NEVER "none" WITHOUT A REPORT. A result that said "unsteered" because nothing was published
    would be an evidence claim about a trajectory nobody observed.
    """
    try:
        from millm.services.inference_service import get_steering_report

        report = get_steering_report()
        if report is None:
            return SteeringRef(kind="unknown")
        items = [i for i in report.items if getattr(i, "kind", None) != "none"]
        if not items:
            return SteeringRef(kind="none")
        # The first steering member (circuits sort first, `SteeringItem.sort_key`). `manual`
        # (live values with no profile) is reported as `inline`: values the request did not name.
        item = items[0]
        kind = getattr(item, "kind", "unknown")
        mapped = {"profile": "profile", "circuit": "circuit", "inline": "inline",
                  "manual": "inline"}.get(kind, "unknown")
        ident = (getattr(item, "name", None) if kind == "profile"
                 else getattr(item, "circuit_id", None) if kind == "circuit" else None)
        return SteeringRef(kind=mapped, id=ident if isinstance(ident, str) and _ID.match(ident)
                           else None)
    except Exception:  # noqa: BLE001
        return SteeringRef(kind="unknown")


def _batch_row() -> Optional[tuple[str, int]]:
    try:
        from millm.services.batch.state import BATCH_ROW

        return BATCH_ROW.get()
    except Exception:  # noqa: BLE001
        return None


# ── the builder ──────────────────────────────────────────────────────────────────


def build_probe_results(
    verdicts: Sequence[Any],
    context: Any,
    *,
    now: Optional[datetime] = None,
    interrupted: Optional[str] = None,
    evaluation_ms: Optional[float] = None,
    rule_results: Optional[Iterable[Any]] = None,
    origin: Optional[str] = None,
) -> list[ProbeResult]:
    """One finalized result per verdict, in the verdicts' order. Reads no database.

    `rule_results` is miLLM 039's extension point (its `_rules_finish`); 038 never passes it.
    `origin` overrides `live`/`batch` (the score route passes `stateless`).
    """
    moment = rfc3339(now or datetime.now(timezone.utc))
    interrupted = interrupted or getattr(context, "interrupted", None)
    probes = {p.probe_id: p for p in (getattr(context, "probes", None) or ())}
    execution = getattr(context, "execution", None)
    trace = getattr(context, "trace", None)
    session = getattr(context, "session", None)
    batch = _batch_row()
    if batch is not None:
        session = None  # a batch row never belongs to a session (BR-001)
    shared = dict(
        event_time=moment,
        completed_at=moment,
        origin=origin or ("batch" if batch is not None else "live"),
        producer=_producer(),
        request_id=_id_or_none(getattr(context, "request_id", None)),
        trace_id=getattr(trace, "trace_id", None),
        span_id=getattr(trace, "span_id", None),
        batch_id=_id_or_none(batch[0]) if batch is not None else None,
        batch_line=int(batch[1]) if batch is not None else None,
        session_id=_id_or_none(getattr(session, "id", None)) if session is not None else None,
        turn_index=getattr(session, "turn_index", None) if session is not None else None,
        engine=getattr(execution, "engine", "transformers"),
        execution_mode=getattr(execution, "mode", "serial"),
        steering_ref=_steering_ref(),
        model_load_id=_model_load_id(),
        evaluation_ms=max(0.0, float(evaluation_ms)) if _finite(evaluation_ms) else None,
    )
    results = [_one(v, probes.get(v.probe_id), shared, interrupted) for v in verdicts]
    for result in results:
        _enforce(result)
    # 039 S2 (FR-39.23): the request's rule signals, BESIDE the probes and never blended — one
    # `kind: "rule"` result per item `_rules_finish` left on the context (or the caller passed).
    if rule_results is None:
        rule_results = getattr(context, "rule_items", None)
        if not isinstance(rule_results, (list, tuple)):
            rule_results = ()
    for item in rule_results or ():
        try:
            results.append(_rule_one(item, shared))
        except Exception as exc:  # noqa: BLE001 - one malformed item never loses the probes
            if os.environ.get("MILLM_STRICT_RESULTS") == "1":
                raise
            logger.error("rule_result_build_failed error=%s", type(exc).__name__)
    # ⚠ 038 FR-38.18: THE COUNTING POINT. Every minted result is counted here, once, before any
    # sampling, persistence or export can lose it. Identifier dimensions are admitted only for
    # the probes this build evaluated (armed, or the score route's set). Never raises.
    from millm import telemetry

    telemetry.observe(results, admitted_ids=set(probes))
    return results


#: The shared fields a rule result carries (the rest of `shared` is probe-only).
_RULE_SHARED = ("event_time", "origin", "producer", "request_id", "trace_id", "span_id",
                "batch_id", "batch_line", "session_id", "turn_index")


def _rule_one(item: Any, shared: dict[str, Any]) -> RuleResult:
    """One `kind: "rule"` result (039 FR-39.23). No score, threshold, rung or detector block;
    `count` is how many matches, never a value; `behaviour.version` is null ("not recorded":
    the producer's rule set records none, OQ-39.5 (a))."""
    key = getattr(item, "behaviour_key", None)
    behaviour = BehaviourOut(key=key, version=None) if isinstance(key, str) and \
        _BEHAVIOUR.match(key) else None
    outcome = item.outcome
    completed = outcome == "completed"
    reason = None if completed else item.reason
    if not completed and (reason is None or reason not in OUTCOME_OF):
        logger.error("rule_result_reason_unlisted reason=%s", reason)
        outcome, reason = "failed", "unlisted_reason"
    return RuleResult(
        **{k: shared[k] for k in _RULE_SHARED},
        result_id=str(uuid7()),
        rule_set_id=item.rule_set_id,
        rule_set_digest=item.rule_set_digest,
        rule_set_version=max(1, int(item.rule_set_version or 1)),
        signal=item.signal,
        form=item.form,
        behaviour=behaviour,
        window=item.window,
        fires=bool(item.fires) if completed and item.fires is not None else None,
        count=max(0, int(item.count or 0)) if completed else 0,
        outcome=outcome,
        reason=reason,
        status=item.status if item.status in ("final", "partial", "unavailable") else "final",
    )


def _id_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and _ID.match(value) else None


def _one(verdict: Any, probe: Any, shared: dict[str, Any],
         interrupted: Optional[str]) -> ProbeResult:
    global names_outside_pattern
    desc = getattr(probe, "descriptor", None)
    decision = getattr(desc, "decision", None)
    raw_score = verdict.score if verdict.scored else None
    threshold = getattr(verdict, "threshold", None)
    threshold = threshold if _finite(threshold) else None
    n_scored = int(getattr(verdict, "n_scored_tokens", 0) or 0)
    verdict_name = getattr(verdict, "name", None)
    score: Optional[float] = None
    verdict_value: Optional[bool] = None
    status = "final"
    scored = False
    # ── the outcome mapping, in order (FTID §3.3) ──────────────────────────────────
    if interrupted:
        outcome, reason = "interrupted", interrupted
        if verdict.scored and _finite(raw_score) and n_scored >= 1:
            status, score = "partial", float(raw_score)
        else:
            status = "unavailable"
    elif verdict.scored and not _finite(raw_score):
        # ⚠ A NaN OR AN INFINITY IS NOT A SCORE (FR-38.5). `_verdict_for` has no finiteness
        # check, so a NaN used to report `score: NaN, verdict: false` — a "no" from a probe that
        # measured nothing. Pre-existing defect, closed here.
        outcome, reason = "failed", "non_finite_score"
    elif verdict.scored:
        outcome, reason, scored = "completed", None, True
        score = float(raw_score)
        # ⚠ `None`, not `False`, when no threshold was placed: the probe ranks, it does not decide.
        verdict_value = None if threshold is None else bool(getattr(verdict, "fires", None))
    else:
        reason = getattr(verdict, "not_scored_reason", None) or "unlisted_reason"
        if not _REASON.match(reason):
            logger.error("probe_result_reason_outside_pattern probe_id=%s", verdict.probe_id)
            reason = "unlisted_reason"
        outcome = OUTCOME_OF.get(reason, "failed")
        if reason not in OUTCOME_OF:
            # A defect path: every code a writer may use is in the published list, and the AST
            # test holds that. Reported as a failure rather than dressed up as a skip.
            logger.error("probe_result_reason_unlisted probe_id=%s reason=%s",
                         verdict.probe_id, reason)

    name = verdict_name if isinstance(verdict_name, str) and _NAME.match(verdict_name) else None
    if name is None and verdict_name:
        names_outside_pattern += 1
        logger.warning("probe_result_name_outside_pattern count=%s", names_outside_pattern)
    behaviour = None
    key = getattr(desc, "behaviour_key", None)
    if isinstance(key, str) and _BEHAVIOUR.match(key):
        version = getattr(desc, "behaviour_version", None)
        behaviour = BehaviourOut(key=key, version=version if isinstance(version, int)
                                 and not isinstance(version, bool) and version >= 1 else None)
    phrase = getattr(verdict, "rung_language", None)
    rung_language = phrase if isinstance(phrase, str) and _RUNG_LANGUAGE.match(phrase) \
        else _vocabulary_phrase(verdict.rung)
    target_fpr = getattr(decision, "target_fpr", None)
    result = ProbeResult(
        **shared,
        # ⚠ MINTED HERE, ONCE PER (probe × window), AT FINALIZATION (D38.25). Every surface copies
        # it; nothing re-mints it later (a read-time id is control M10's defect).
        result_id=str(uuid7()),
        probe_id=verdict.probe_id,
        probe_name=name,
        behaviour=behaviour,
        detector_digest=_digest(getattr(desc, "detector_digest", None)),
        detector_digest_source=(
            _listed(getattr(desc, "detector_digest_source", "unknown"), DIGEST_SOURCES)
            if _digest(getattr(desc, "detector_digest", None)) != "unknown" else "unknown"
        ),
        identity_algorithm=(None if getattr(desc, "identity_algorithm", None) != IDENTITY_ALGORITHM
                            else IDENTITY_ALGORITHM)
        if _digest(getattr(desc, "detector_digest", None)) != "unknown" else None,
        target=_listed(getattr(desc, "target", "unknown"), TARGETS),
        layer=_layer(getattr(probe, "layer", None)),
        hook=_listed(getattr(desc, "hook", "unknown"), HOOKS),
        basis=_listed(getattr(desc, "basis", "unknown"), BASES),
        rule=_listed(getattr(probe, "rule", None), RULE_NAMES),
        served_model=_or_unknown(getattr(verdict, "served_model", None), _MODEL),
        model_revision=_or_unknown(getattr(desc, "model_revision", None), _REVISION),
        template_sha256=_or_unknown(getattr(desc, "template_sha256", None), _HEX64),
        precision=_or_unknown(getattr(desc, "precision", None), _PRECISION),
        definition_scope=_listed(getattr(probe, "scope", None), _SCOPES),
        window=getattr(verdict, "window", "all"),
        n_scored_tokens=n_scored if score is not None else 0,
        status=status,
        scored=scored,
        outcome=outcome,
        reason=reason,
        score=score,
        threshold=threshold,
        verdict=verdict_value,
        fires=verdict_value,
        decision_digest=_digest(getattr(decision, "decision_digest", None)),
        decision_revision=getattr(decision, "decision_revision", None),
        threshold_revision=max(1, int(getattr(verdict, "threshold_revision", 1) or 1)),
        decision=DecisionOut(
            calibration_id=_id_or_none(getattr(decision, "calibration_id", None)),
            target_fpr=target_fpr if _finite(target_fpr) and 0.0 <= target_fpr <= 1.0 else None,
            provisional=bool(getattr(verdict, "provisional", False)),
        ),
        rung=int(verdict.rung),
        rung_language=rung_language,
        parity_status=_listed(getattr(desc, "parity_status", "unknown"), PARITY),
    )
    result._legacy_name = verdict_name
    return result


def _layer(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _vocabulary_phrase(rung: Any) -> str:
    from millm.core.probe_evidence import probe_rung_language

    try:
        return probe_rung_language(int(rung))
    except Exception:  # noqa: BLE001
        return "trained"


def _digest(value: Any) -> str:
    return value if isinstance(value, str) and _HEX64.match(value) else "unknown"


class ProbeResultRuleViolation(AssertionError):
    """A result breaks a conditional rule of FTDD §5.1 (raised only under MILLM_STRICT_RESULTS)."""


def _violations(r: Any) -> list[str]:
    bad = []
    if r.outcome == "completed":
        if not (r.scored and r.score is not None and r.status == "final" and r.reason is None):
            bad.append("completed needs scored, a finite score, status final and no reason")
    if r.outcome in ("skipped", "failed"):
        if r.scored or r.score is not None or r.verdict is not None or not r.reason:
            bad.append(f"{r.outcome} needs scored false, no score, no verdict and a reason")
    if r.outcome == "interrupted":
        if r.verdict is not None or r.status not in ("partial", "unavailable") or not r.reason:
            bad.append("interrupted needs no verdict, status partial/unavailable and a reason")
    if r.status == "partial" and (r.score is None or r.n_scored_tokens < 1):
        bad.append("partial needs a finite score over at least one token")
    if r.status == "unavailable" and r.score is not None:
        bad.append("unavailable carries no score")
    if r.threshold is None and r.verdict is not None:
        bad.append("no threshold means no verdict")
    if r.fires != r.verdict:
        bad.append("fires duplicates verdict")
    return bad


def _enforce(result: Any) -> None:
    bad = _violations(result)
    if not bad:
        return
    if _strict():
        raise ProbeResultRuleViolation(f"{result.probe_id}/{result.window}: {'; '.join(bad)}")
    logger.error("probe_result_rule_violated probe_id=%s rules=%s", result.probe_id, bad)


# ── the second moment and the projections ─────────────────────────────────────────


def recorded(result: ProbeResult, *, event_id: Optional[int], persistence: str) -> ProbeResult:
    """The RECORDED copy: the finalized object plus `persistence` and, when the row exists,
    `source_event_id`. Nothing else changes (T-S1-03)."""
    copy = result.model_copy(update={"persistence": persistence, "source_event_id": event_id})
    copy._legacy_name = result._legacy_name
    return copy


def stored_result(result: ProbeResult) -> dict[str, Any]:
    """What `probe_events.result` holds: the recorded object WITHOUT `source_event_id`, because the
    autoincrement id does not exist until the INSERT returns (FTID X-05, I-38.1 (a))."""
    return wire(recorded(result, event_id=None, persistence="persisted"))


def recorded_from_row(row: Any) -> Optional[dict[str, Any]]:
    """⚠ THE SINGLE INJECTION POINT for `source_event_id`: the stored object plus the row's own id."""
    stored = getattr(row, "result", None)
    if not isinstance(stored, dict):
        return None
    out = dict(stored)
    if getattr(row, "id", None) is not None:
        out["source_event_id"] = int(row.id)
    return out


def legacy_projection(result: Any) -> dict[str, Any]:
    """The pre-038 keys every probe surface carried, derived from the result itself.

    ⚠ THE ONLY result→legacy mapping for payloads (the other is `legacy_columns`, for the row).
    `served_model` keeps its old meaning — `null` when not recorded — while the contract's own
    key says `"unknown"`; a surface merges contract keys last, so the contract wins there.
    """
    data = result if isinstance(result, dict) else wire(result)
    name = getattr(result, "_legacy_name", None) if not isinstance(result, dict) else None
    completed = data.get("outcome") == "completed"
    served = data.get("served_model")
    return {
        "name": name if name is not None else data.get("probe_name"),
        "probe_id": data.get("probe_id"),
        "window": data.get("window"),
        "provisional": bool((data.get("decision") or {}).get("provisional")),
        "scored": bool(data.get("scored")),
        "score": data.get("score"),
        "threshold": data.get("threshold"),
        "verdict": data.get("verdict"),
        "rung": data.get("rung"),
        "rung_language": data.get("rung_language"),
        "served_model": None if served in (None, "unknown") else served,
        "threshold_revision": data.get("threshold_revision"),
        "n_scored_tokens": data.get("n_scored_tokens"),
        "not_scored_reason": None if completed else data.get("reason"),
        "reason": data.get("reason"),
    }


def surface_entry(result: Any) -> dict[str, Any]:
    """One entry of the terminal chunk and the `millm_probe_verdicts` body field: the legacy keys,
    then the contract object (FR-38.3, FR-38.4). A rule result (039 S2) has no legacy keys: it
    never existed before 038, so it is the contract object alone (no `score`, no `scored`)."""
    if getattr(result, "kind", "probe") == "rule":
        return wire(result)
    return {**legacy_projection(result), **wire(result)}


def legacy_columns(result: ProbeResult) -> dict[str, Any]:
    """The `probe_events` columns that are projections of the result, written in the same
    statement as `result` itself (T-S1-10: one source, two representations)."""
    data = wire(result)
    legacy = legacy_projection(result)
    unknown = lambda v: None if v in (None, "unknown") else v  # noqa: E731
    return {
        "probe_id": data["probe_id"],
        "request_id": data.get("request_id"),
        "scored": legacy["scored"],
        "not_scored_reason": legacy["not_scored_reason"],
        "score": data.get("score"),
        "threshold": data.get("threshold"),
        "verdict": data.get("verdict"),
        "rung": data.get("rung"),
        "n_scored_tokens": data.get("n_scored_tokens"),
        "window": data.get("window"),
        "provisional": legacy["provisional"],
        "threshold_revision": data.get("threshold_revision") or 1,
        "served_model": legacy["served_model"],
        "origin": data.get("origin"),
        "batch_id": data.get("batch_id"),
        "batch_line": data.get("batch_line"),
        "session_id": data.get("session_id"),
        "turn_index": data.get("turn_index"),
        # the contract's own columns (migration 032)
        "result_id": data["result_id"],
        "trace_id": data.get("trace_id"),
        "span_id": data.get("span_id"),
        "event_time": datetime.strptime(data["event_time"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc),
        "event_time_source": data.get("event_time_source", "finalized"),
        "outcome": data["outcome"],
        "reason": data.get("reason"),
        "status": data.get("status", "final"),
        "persistence": "persisted",
        "detector_digest": unknown(data.get("detector_digest")),
        "detector_digest_source": data.get("detector_digest_source", "unknown"),
        "identity_algorithm": data.get("identity_algorithm"),
        "decision_digest": unknown(data.get("decision_digest")),
        "decision_revision": data.get("decision_revision"),
        "migrated": bool(data.get("migrated")),
    }


def summary_of(result: Any) -> str:
    """The event row's one-line `summary`, from the result (a code, never text)."""
    data = result if isinstance(result, dict) else wire(result)
    if data.get("outcome") != "completed":
        return f"not scored: {data.get('reason') or 'unknown'}"
    if data.get("verdict") is None:
        return f"score {data['score']:.4f} (no threshold placed — ranks, does not decide)"
    return f"score {data['score']:.4f} {'above' if data['verdict'] else 'below'} threshold"


def stateless_results(verdicts: Sequence[Any], probes: Sequence[Any], *, input_index: int,
                      evaluation_ms: Optional[float] = None) -> list[ProbeResult]:
    """Results of `POST /api/probes/score` (and a Batch `probe` row, I-38.5): `origin: stateless`,
    `persistence: not_persisted`, `input_index`, never stored (FR-38.17)."""
    from types import SimpleNamespace

    from millm.services.probe_runtime import STATELESS_EXECUTION
    from millm.telemetry import capture_trace

    # 038 S2: the score route's own trace (its `millm.probe_score` span when export is on, else
    # the client's `traceparent`, else null), so an exported stateless record joins its request.
    context = SimpleNamespace(probes=list(probes), request_id=None, trace=capture_trace(),
                              session=None, interrupted=None,
                              execution=STATELESS_EXECUTION)
    built = build_probe_results(verdicts, context, evaluation_ms=evaluation_ms, origin="stateless")
    out = []
    for result in built:
        stateless = result.model_copy(update={"input_index": input_index})
        stateless._legacy_name = result._legacy_name
        out.append(recorded(stateless, event_id=None, persistence="not_persisted"))
    return out
