"""Exercises for DR-22 (Prometheus metric names `millm_*`), DR-23 (`service.name` default `millm`
and span names `millm.*`) and DR-33 (instrumentation scope `millm`) — 040 FR-040.6.

Dashboards, alerts and saved queries in the operator's own observability stack select on these
names; a rename empties every panel with no error anywhere. The telemetry runs for real against the
OTLP stub (`tests/support/otlp_stub.py`).
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from tests.support import capture_pre038_goldens as cap
from tests.unit.telemetry.conftest import _telemetry_off, flush, stub, telemetry_on  # noqa: F401

#: The served families on 2026-11-08, pinned by value.
FAMILIES = frozenset({
    "millm_circuit_breaker_open", "millm_circuit_breaker_trips_total",
    "millm_circuit_claim_faults_total", "millm_circuit_layers_composed",
    "millm_circuit_layers_served", "millm_circuits_serving", "millm_model_loaded",
    "millm_model_loads_total", "millm_model_unloads_total",
    "millm_probe_evaluations_expected_total", "millm_probe_evaluations_total",
    "millm_probe_series_overflow_total", "millm_probe_stateless_evaluations_total",
    "millm_probe_threshold_crossings_total", "millm_request_errors_total", "millm_requests_active",
    "millm_requests_total", "millm_uptime_seconds",
})


def test_prometheus_metric_families():
    from miserver.main import create_app

    r = TestClient(create_app()).get("/api/health/metrics/prometheus")
    assert r.status_code == 200
    served = {line.split()[2] for line in r.text.splitlines() if line.startswith("# TYPE ")}
    assert served == FAMILIES
    # The SAMPLE lines are what Prometheus stores (review M06): each belongs to a frozen family.
    samples = {re.split(r"[{\s]", line, maxsplit=1)[0]
               for line in r.text.splitlines() if line and not line.startswith("#")}
    assert {s for s in samples if not any(s == f or s.startswith(f + "_") for f in FAMILIES)} == set()
    assert {"millm_requests_total", "millm_requests_active", "millm_uptime_seconds"} <= samples


async def _one_scored_chat(stub, telemetry_on):  # noqa: F811
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        assert h.post_chat(cap.BODY, {}).status_code == 200
        flush()
    return stub


async def test_default_service_name_and_span_names(stub, telemetry_on):  # noqa: F811
    from miserver.core.config import Settings

    assert Settings.model_fields["TELEMETRY_SERVICE_NAME"].default == "millm"
    got = await _one_scored_chat(stub, telemetry_on)
    assert got.resources and {r.get("service.name") for r in got.resources} == {"millm"}
    names = {s["name"] for s in got.spans}
    assert {"millm.queue.wait", "millm.probe.evaluate"} <= names, names


async def test_instrumentation_scope(stub, telemetry_on):  # noqa: F811
    from miserver.telemetry import pipeline

    assert pipeline.SCOPE == "millm"
    got = await _one_scored_chat(stub, telemetry_on)
    assert got.scopes and set(got.scopes) == {"millm"}, set(got.scopes)
    assert got.logs and {rec["scope"] for rec in got.logs} == {"millm"}
