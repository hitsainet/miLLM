"""Probe coverage counters (FR-38.18, FR-38.19; FTASKS 7.4, 7.5, 8.14).

* AT05 through the real route: the counters equal the stored rows (per probe, window, outcome),
  and crossings equal the completed thresholded rows with verdict true.
* A skip is never a crossing and never a "no" (control M18).
* Churn: 500 behaviours against a cap of 50 stay bounded; nothing is lost from the totals (M19).
* Prometheus: the existing lines are unchanged (golden, numbers normalised), the probe lines are
  present with telemetry off, and labels are escaped. Reachability: delete the rendering call.
* With metrics export on, the OTel counters read the same store and the histograms arrive without
  exemplars.
"""

from __future__ import annotations

import re
from collections import Counter
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from millm.telemetry import counters
from millm.telemetry.counters import CROSSINGS, EVALUATIONS, EXPECTED, STATELESS, CounterStore
from tests.support import capture_pre038_goldens as cap
from tests.unit.telemetry.conftest import flush

#: The Prometheus route's lines BEFORE 038, numbers normalised to <n>. Pinned byte for byte.
PRE_038_LINES = """\
# HELP millm_requests_total Total number of requests processed
# TYPE millm_requests_total counter
millm_requests_total <n>

# HELP millm_requests_active Currently active requests
# TYPE millm_requests_active gauge
millm_requests_active <n>

# HELP millm_circuits_serving Circuits currently steering
# TYPE millm_circuits_serving gauge
millm_circuits_serving <n>

# HELP millm_circuit_layers_served Distinct layers steered by circuits
# TYPE millm_circuit_layers_served gauge
millm_circuit_layers_served <n>

# HELP millm_circuit_layers_composed Layers carrying more than one circuit (rung header suppressed)
# TYPE millm_circuit_layers_composed gauge
millm_circuit_layers_composed <n>

# HELP millm_circuit_claim_faults_total Claim/steering divergences that could not be repaired
# TYPE millm_circuit_claim_faults_total counter
millm_circuit_claim_faults_total <n>

# HELP millm_request_errors_total Total number of request errors
# TYPE millm_request_errors_total counter
millm_request_errors_total <n>

# HELP millm_model_loaded Whether a model is currently loaded
# TYPE millm_model_loaded gauge
millm_model_loaded <n>

# HELP millm_model_loads_total Total number of model loads
# TYPE millm_model_loads_total counter
millm_model_loads_total <n>

# HELP millm_model_unloads_total Total number of model unloads
# TYPE millm_model_unloads_total counter
millm_model_unloads_total <n>

# HELP millm_circuit_breaker_open Whether circuit breaker is open
# TYPE millm_circuit_breaker_open gauge
millm_circuit_breaker_open{name="huggingface"} <n>

# HELP millm_circuit_breaker_trips_total Total circuit breaker trips
# TYPE millm_circuit_breaker_trips_total counter
millm_circuit_breaker_trips_total <n>

# HELP millm_uptime_seconds Server uptime in seconds
# TYPE millm_uptime_seconds gauge
millm_uptime_seconds <n>
"""


def _prometheus(client=None) -> str:
    from millm.api.dependencies import get_model_loader
    from millm.main import create_app

    app = create_app()
    app.dependency_overrides[get_model_loader] = lambda: MagicMock(is_loaded=False)
    response = TestClient(app).get("/api/health/metrics/prometheus")
    assert response.status_code == 200
    return response.text


def _result(**over):
    base = dict(probe_id="pr_0123456789ab", behaviour={"key": "b-alpha"}, kind="probe",
                window="all", engine="transformers", execution_mode="serial", origin="live",
                outcome="completed", reason=None, threshold=0.5, verdict=True, score=0.9)
    base.update(over)
    return base


# ── the store ────────────────────────────────────────────────────────────────────


def test_a_skip_is_never_a_crossing_and_keeps_its_reason():
    """M18: count a skip as a completed "no", or a "no" as a crossing → red."""
    store = CounterStore()
    store.observe([
        _result(),
        _result(outcome="skipped", reason="judge_traffic", threshold=0.5, verdict=None, score=None),
        _result(outcome="completed", verdict=False, score=0.1),
        _result(outcome="completed", threshold=None, verdict=None),
        _result(outcome="interrupted", reason="client_disconnected", verdict=None),
    ])
    assert store.total(CROSSINGS) == 1
    assert store.total(EXPECTED) == 5
    assert store.total(EVALUATIONS, outcome="completed") == 3
    assert store.total(EVALUATIONS, outcome="skipped", reason="judge_traffic") == 1
    assert store.total(EVALUATIONS, outcome="interrupted", reason="client_disconnected") == 1
    assert store.total(EVALUATIONS, outcome="completed", reason="none") == 3


def test_stateless_results_count_apart():
    store = CounterStore()
    store.observe([_result(origin="stateless"), _result(origin="stateless", outcome="failed",
                                                       reason="encoder_failed", verdict=None)])
    assert store.total(EXPECTED) == 0 and store.total(CROSSINGS) == 0
    assert store.total(STATELESS) == 2
    assert store.total(STATELESS, outcome="failed", reason="encoder_failed") == 1


def test_churn_is_bounded_and_loses_no_count():
    """M19: remove the cap → 500 series → red."""
    store = CounterStore(series_cap=50)
    results = [_result(behaviour={"key": f"b{i:04d}"}) for i in range(500)]
    store.observe(results)
    for instrument in (EXPECTED, EVALUATIONS, CROSSINGS):
        assert store.series_count(instrument) == 51, instrument  # 50 + the overflow series
        assert store.total(instrument) == 500, instrument
        assert store.total(instrument, **{"otel.metric.overflow": True}) == 450
    assert store.series_overflow == 3 * 450


def test_an_identifier_not_evaluated_folds_into_overflow():
    store = CounterStore()
    store.observe([_result(probe_id="pr_aaaaaaaaaaaa"), _result(probe_id="pr_bbbbbbbbbbbb")],
                  admitted_ids={"pr_aaaaaaaaaaaa"})
    assert store.total(EXPECTED, probe_id="pr_aaaaaaaaaaaa") == 1
    assert store.total(EXPECTED, probe_id="pr_bbbbbbbbbbbb") == 0
    assert store.total(EXPECTED, **{"otel.metric.overflow": True}) == 1


def test_free_text_never_becomes_a_dimension():
    store = CounterStore()
    store.observe([_result(window="hello world, my number is 42")])
    snap = store.snapshot()
    assert all("hello" not in repr(k) for series in snap.values() for k in series)
    assert store.total(EXPECTED, **{"otel.metric.overflow": True}) == 1


# ── Prometheus ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("export", [False, True], ids=["off", "stub"])
def test_the_existing_lines_are_unchanged_and_the_probe_lines_follow(export, stub, telemetry_on):
    if export:  # FR-38.23: the route answers the same with every signal on
        telemetry_on()
    text = _prometheus()
    head = text.split("# HELP millm_probe_evaluations_expected_total", 1)
    assert len(head) == 2, "the probe counters are not on the route (telemetry is off)"
    normalised = re.sub(r"(?m) [0-9.]+$", " <n>", head[0])
    assert normalised == PRE_038_LINES + "\n"
    for name in ("millm_probe_evaluations_expected_total", "millm_probe_evaluations_total",
                 "millm_probe_threshold_crossings_total",
                 "millm_probe_stateless_evaluations_total", "millm_probe_series_overflow_total"):
        assert f"# TYPE {name} counter" in text, name


def test_label_values_are_escaped():
    store = CounterStore()
    with store._lock:
        store._values[EXPECTED][(("probe_id", 'a"b\\c\nd'),)] = 2
    lines = store.render_prometheus()
    assert 'millm_probe_evaluations_expected_total{probe_id="a\\"b\\\\c\\nd"} 2' in lines


async def test_at05_the_counters_equal_the_stored_rows():
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA, cap.BETA])
        ids = []
        for headers in ({}, {"X-miLLM-Origin": "judge"}, {}):
            r = h.post_chat(cap.BODY, headers)
            assert r.status_code == 200
            ids.append(r.json()["id"])
        rows = [row for rid in ids for row in await h.event_rows(rid)]
        text = _prometheus()
    stored = Counter((r["probe_id"], r["window"], r["outcome"], r["reason"] or "none")
                     for r in rows)
    counted = Counter()
    for (attrs, value) in counters.STORE.snapshot()[EVALUATIONS].items():
        a = dict(attrs)
        counted[(a["probe_id"], a["window"], a["outcome"], a["reason"])] += value
    assert counted == stored
    assert counters.STORE.total(EXPECTED) == len(rows) == 9
    assert counters.STORE.total(EVALUATIONS, outcome="skipped", reason="judge_traffic") == 3
    crossings = sum(1 for r in rows if r["outcome"] == "completed"
                    and r["threshold"] is not None and r["verdict"] is True)
    assert counters.STORE.total(CROSSINGS) == crossings
    assert counters.STORE.series_overflow == 0
    # And the same numbers on the live Prometheus route.
    assert f"millm_probe_evaluations_total{{" in text
    total = sum(int(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                if line.startswith("millm_probe_evaluations_expected_total{"))
    assert total == 9


# ── OTel metrics (8.14) ──────────────────────────────────────────────────────────


async def test_the_otel_counters_read_the_store_and_histograms_carry_no_exemplars(
        stub, telemetry_on):
    telemetry_on(TELEMETRY_TRACES_ENABLED=False, TELEMETRY_LOGS_ENABLED=False)
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA, cap.BETA])
        assert h.post_chat(cap.BODY, {}).status_code == 200
        flush()
    points = stub.metric_points
    names = {p["name"] for p in points}
    assert {"millm.probe.evaluations", "millm.probe.evaluations.expected",
            "millm.probe.series_overflow", "millm.probe.evaluation.duration"} <= names, names
    exported = sum(p["value"] for p in points if p["name"] == "millm.probe.evaluations")
    assert exported == counters.STORE.total(EVALUATIONS) == 3
    for p in points:
        assert p["exemplars"] == 0, p
        assert set(p["attributes"]) <= {"probe_id", "behaviour", "kind", "window", "engine",
                                        "execution_mode", "origin", "outcome", "reason",
                                        "otel.metric.overflow"}
    durations = [p for p in points if p["name"] == "millm.probe.evaluation.duration"]
    assert durations and all(set(p["attributes"]) == {"probe_id", "window", "engine"}
                             for p in durations)


async def test_a_negative_margin_is_exported_not_dropped(stub, telemetry_on):
    """Review R1 of S2: `millm.probe.score_margin` is score − threshold (D38.41), NEGATIVE for
    every result below its threshold. The SDK's `Histogram.record` drops a negative amount with a
    warning, so before the fix the histogram held only the crossings. The fixture mixes signs so
    the two behaviours differ: both measurements must be counted, one in a negative bucket.
    Control: record through `hist.record` again → count 1."""
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
    )

    pipeline = telemetry_on(TELEMETRY_TRACES_ENABLED=False, TELEMETRY_LOGS_ENABLED=False)
    dims = {"probe_id": "pr_0123456789ab", "window": "all"}
    pipeline.enqueue_measurements([("millm.probe.score_margin", -0.75, dims),
                                   ("millm.probe.score_margin", 0.25, dims)])
    flush()
    margins = [p for p in stub.metric_points if p["name"] == "millm.probe.score_margin"]
    assert [p["value"] for p in margins] == [2], margins
    negative = []
    for path, body, _headers in stub.raw:
        if path != "/v1/metrics":
            continue
        for rm in ExportMetricsServiceRequest.FromString(body).resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == "millm.probe.score_margin":
                        for point in metric.exponential_histogram.data_points:
                            negative.append(sum(point.negative.bucket_counts))
                            assert point.min == -0.75 and point.max == 0.25
    assert negative == [1], negative
