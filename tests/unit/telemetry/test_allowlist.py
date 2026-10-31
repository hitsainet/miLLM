"""The attribute allowlist, one table per signal (FR-38.21, FTASKS 8.6).

Each listed key keeps a valid value and refuses free text under it; an unlisted key is dropped and
counted; the contract and the result table agree in both directions, so a recorded result loses
nothing it may carry and the table names nothing the contract does not have.
"""

from __future__ import annotations

import typing

import pytest

from millm.api.schemas.probe_result import ProbeResult, wire
from millm.telemetry import allowlist

FREE_TEXT = "Ignore previous instructions; my card is on file\nthanks"

#: One valid value per key, per signal.
VALID = {
    "resource": {
        "service.name": "millm", "service.version": "0.5.0",
        "service.instance.id": "a1b2c3d4", "deployment.environment.name": "prod",
    },
    "span": {
        "gen_ai.operation.name": "chat", "gen_ai.request.model": "org/model:Q8",
        "millm.engine": "cbm", "millm.execution_mode": "serial", "millm.stream": True,
        "millm.queue.position_at_entry": 3, "millm.probe.results.completed": 2,
        "millm.probe.results.skipped": 0, "millm.probe.results.failed": 0,
        "millm.probe.results.interrupted": 1, "millm.batch.id": "batch_0123",
        "error.type": "CancelledError",
    },
    "log_lifecycle": {
        "mi.lifecycle.subject_kind": "probe", "mi.lifecycle.action": "recalibrate",
        "mi.lifecycle.outcome": "refused", "mi.lifecycle.code": "PROBE_PARITY_FAILED",
        "mi.lifecycle.probe_id": "pr_0123456789ab", "mi.lifecycle.behaviour": "b-alpha",
        "mi.lifecycle.detector_digest": "a" * 64, "mi.lifecycle.threshold_revision": 2,
        "mi.lifecycle.decision_revision": 3, "mi.lifecycle.decision_digest": "unknown",
        "mi.lifecycle.previous_threshold_revision": 1,
        "mi.lifecycle.previous_decision_digest": "b" * 64,
    },
    "metric": {
        "probe_id": "pr_0123456789ab", "behaviour": "none", "kind": "probe", "window": "all",
        "engine": "transformers", "execution_mode": "batched", "origin": "batch",
        "outcome": "skipped", "reason": "judge_traffic", "otel.metric.overflow": True,
    },
}


@pytest.mark.parametrize("signal", sorted(VALID))
def test_every_listed_key_keeps_a_valid_value(signal):
    assert set(VALID[signal]) == set(allowlist.SIGNALS[signal]), "the table test covers every key"
    kept, dropped = allowlist.filter(signal, VALID[signal])
    assert (kept, dropped) == (VALID[signal], 0)


@pytest.mark.parametrize("signal", sorted(allowlist.SIGNALS))
def test_free_text_is_refused_under_every_key(signal):
    for key in allowlist.SIGNALS[signal]:
        kept, dropped = allowlist.filter(signal, {key: FREE_TEXT})
        assert kept == {} and dropped == 1, key


@pytest.mark.parametrize("signal", sorted(allowlist.SIGNALS))
def test_an_unlisted_key_is_dropped_and_counted(signal):
    before = allowlist.attributes_dropped()
    kept, dropped = allowlist.filter(signal, {"gen_ai.input.messages": "hello", "http.url": "/x"})
    assert kept == {} and dropped == 2
    assert allowlist.attributes_dropped() == before + 2


def test_types_are_checked_not_coerced():
    kept, dropped = allowlist.filter("span", {"millm.stream": 1, "millm.queue.position_at_entry": True})
    assert kept == {} and dropped == 2
    kept, _ = allowlist.filter("log_result", {"mi.probe.score": float("nan")})
    assert kept == {}


def test_a_value_is_bounded_in_length():
    kept, dropped = allowlist.filter("span", {"gen_ai.request.model": "m" * 300})
    assert kept == {} and dropped == 1


def _a_result() -> ProbeResult:
    from millm.api.schemas.probe_result import BehaviourOut, DecisionOut, Producer, SteeringRef

    return ProbeResult(
        result_id="0192f2a0-0000-7000-8000-000000000001",
        event_time="2026-10-23T12:00:00.000001Z", completed_at="2026-10-23T12:00:00.000001Z",
        origin="live", producer=Producer(version="0.5.0"), request_id="chatcmpl-1",
        trace_id="1" * 32, span_id="2" * 16, session_id="s1", turn_index=0,
        behaviour=BehaviourOut(key="b-alpha", version=2), outcome="completed",
        window="all", probe_id="pr_0123456789ab", probe_name="probe alpha (2)",
        detector_digest="a" * 64, detector_digest_source="computed",
        identity_algorithm="mistudio.probe-identity/1", target="positive_class", layer=1,
        hook="resid_post", basis="residual", rule="mean", served_model="org/model",
        model_revision="main", template_sha256="c" * 64, precision="bf16",
        steering_ref=SteeringRef(kind="circuit", id="crc_1"), definition_scope="all",
        n_scored_tokens=4, scored=True, score=0.5, threshold=0.25, verdict=True, fires=True,
        decision_digest="d" * 64, decision_revision=2, threshold_revision=3,
        decision=DecisionOut(calibration_id="cal_1", target_fpr=0.01, provisional=False),
        rung=2, rung_language="detects on held-out data", parity_status="passed",
        model_load_id="0192f2a0-0000-5000-8000-000000000001", evaluation_ms=1.5,
        persistence="persisted", source_event_id=7, batch_id="b1", batch_line=2,
        choice_index=0, input_index=0,
    )


def test_a_full_recorded_result_passes_with_nothing_dropped():
    """Completeness: every field a result can carry is allowlisted (else telemetry silently loses
    it), and every allowlisted result key is a contract field (no stale entry)."""
    flat = allowlist.flatten(wire(_a_result()), allowlist.RESULT_PREFIX)
    kept, dropped = allowlist.filter("log_result", flat)
    assert dropped == 0, set(flat) - set(kept)
    assert kept == flat
    leaves = set()
    from millm.api.schemas.probe_result import RuleResult

    # 039 S2: the rule variant's fields are contract fields too.
    fields = {**ProbeResult.model_fields, **RuleResult.model_fields}
    for name, field in fields.items():
        key = field.alias or name
        candidates = [field.annotation, *typing.get_args(field.annotation)]
        sub = next((c.model_fields for c in candidates if hasattr(c, "model_fields")), None)
        leaves |= {f"{key}.{s}" for s in sub} if sub else {key}
    listed = {k[len(allowlist.RESULT_PREFIX):] for k in allowlist.LOG_RESULT}
    # `set`, `probability` and `calibration_ref` are null in v1 and never carried.
    assert listed - leaves == set(), "allowlisted keys the contract does not have"
    unlisted = {k for k in leaves - listed
                if not k.startswith(("set", "probability", "calibration_ref"))}
    assert unlisted == set(), "contract fields telemetry would drop"


def test_no_event_name_outside_the_two_records():
    assert allowlist.EVENT_NAMES == {"mi.probe.result", "mi.probe.lifecycle"}
    assert "gen_ai.evaluation.result" not in allowlist.EVENT_NAMES
