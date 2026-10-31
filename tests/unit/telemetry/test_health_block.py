"""The `telemetry` block of the LIVE `GET /api/health/detailed` (FR-38.13; FTDD §5.7, §7.5).

Every state of the per-signal machine — disabled, healthy, degraded, unavailable (no endpoint,
plaintext refused, a 401, records waiting past the window) — read through the real route; the
deletion note and the result-log filter are shown; no secret, no exception message. Control M13:
delete the block's assembly in `detailed_health_check`; hide the filter from the view.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from millm import telemetry
from millm.telemetry import health
from tests.unit.telemetry.conftest import flush, telemetry_settings


def _block() -> dict:
    from millm.api.dependencies import get_inference_service, get_model_loader
    from millm.main import create_app

    app = create_app()
    app.dependency_overrides[get_model_loader] = lambda: MagicMock(is_loaded=False)
    app.dependency_overrides[get_inference_service] = lambda: MagicMock()
    response = TestClient(app).get("/api/health/detailed")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("telemetry") is not None, "the detailed route carries no telemetry block"
    return body["telemetry"]


def _states(block: dict) -> dict:
    return {name: s["state"] for name, s in block["signals"].items()}


def test_off_by_default_every_signal_reads_disabled():
    block = _block()
    assert block["enabled"] is False
    assert _states(block) == {"traces": "disabled", "metrics": "disabled", "logs": "disabled"}
    assert block["deletion_note"] == health.DELETION_NOTE
    assert block["config"]["result_log_filter"] == {"origins": ["live", "batch"]}
    assert block["config"]["stateless_export_permitted"] is False
    assert block["config"]["headers_set"] is False and block["config"]["endpoint"] is None
    for s in block["signals"].values():
        assert (s["exported"], s["failed"], s["dropped"], s["last_error"]) == (0, 0, 0, None)


def test_healthy_after_a_success_and_the_filter_shown(stub, telemetry_on):
    telemetry_on(TELEMETRY_RESULT_LOG_ORIGINS="live")
    telemetry.emit_lifecycle(MagicMock(id="pr_0123456789ab", behaviour_key=None,
                                       detector_digest=None, threshold_revision=1,
                                       decision_revision=None, decision_digest=None), "import")
    assert stub.wait_for(lambda s: s.logs, timeout=10)
    block = _block()
    assert block["enabled"] is True
    assert _states(block) == {"traces": "healthy", "metrics": "healthy", "logs": "healthy"}
    assert block["signals"]["logs"]["exported"] == 1
    assert block["signals"]["logs"]["last_success_at"].endswith("Z")
    assert block["signals"]["logs"]["queue"]["capacity"] == 8192
    assert block["config"]["result_log_filter"] == {"origins": ["live"]}
    assert block["config"]["transport"] == "plaintext_trusted_local"
    assert block["config"]["sdk_version"] == "1.45.1"
    assert block["config"]["semconv_schema_url"].endswith("/1.44.0")


def test_degraded_after_a_drop(stub, telemetry_on):
    pipeline = telemetry_on(TELEMETRY_QUEUE_SIZE=1)
    pipeline._accepting = False  # every enqueue is now a counted drop
    assert pipeline.enqueue("logs", object()) is False
    block = _block()
    assert block["signals"]["logs"]["state"] == "degraded"
    assert block["signals"]["logs"]["dropped"] == 1
    assert block["signals"]["traces"]["state"] == "healthy"


@pytest.mark.parametrize("overrides, error", [
    ({"endpoint": None}, "NoEndpoint"),
    ({"TELEMETRY_ALLOW_PLAINTEXT": False}, "PlaintextNotAllowed"),
    ({"TELEMETRY_PROTOCOL": "grpc"}, "UnsupportedProtocol"),
    ({"TELEMETRY_HEADERS": "no-equals-sign"}, "InvalidHeaders"),
])
def test_a_refused_configuration_is_unavailable_and_sends_nothing(stub, telemetry_on, overrides,
                                                                  error):
    assert telemetry_on(**overrides) is None, "a refused configuration starts no pipeline"
    block = _block()
    assert _states(block) == {"traces": "unavailable", "metrics": "unavailable",
                              "logs": "unavailable"}
    assert block["signals"]["logs"]["last_error"]["type"] == error
    assert stub.raw == []


def test_a_401_is_unavailable_with_the_class_only(stub, telemetry_on):
    stub.mode = "401"
    telemetry_on(TELEMETRY_HEADERS="authorization=Bearer SENTINEL-TOKEN")
    telemetry.emit_lifecycle(MagicMock(id="pr_0123456789ab", behaviour_key=None,
                                       detector_digest=None, threshold_revision=1,
                                       decision_revision=None, decision_digest=None), "delete")
    flush()
    block = _block()
    assert block["signals"]["logs"]["state"] == "unavailable"
    assert block["signals"]["logs"]["last_error"]["type"] == "HTTP401"
    assert block["signals"]["logs"]["failed"] == 1
    assert block["config"]["headers_set"] is True
    assert "SENTINEL-TOKEN" not in str(block)
    # The credential was SENT to the receiver (that is its job), as a header, not a payload.
    assert any(h.get("authorization") == "Bearer SENTINEL-TOKEN" for _p, _b, h in stub.raw)


def test_unavailable_when_records_wait_past_the_window():
    state = health.SignalHealth("logs", clock=lambda: clock[0])
    clock = [1000.0]
    state.reset(enabled=True, capacity=10)
    state.failure("ConnectionError", 1, final=False)
    assert state.state(60.0, 300.0) == "degraded"
    clock[0] += 299.0
    assert state.state(60.0, 300.0) == "degraded"
    clock[0] += 2.0
    assert state.state(60.0, 300.0) == "unavailable"
    state.success(1)
    assert state.state(60.0, 300.0) == "healthy", "a success ends the wait"
    state.failure("ConnectionError", 1, final=True)
    assert state.state(60.0, 300.0) == "degraded"
    clock[0] += 61.0
    assert state.state(60.0, 300.0) == "degraded", "still no success since the failure"
    state.success(1)
    assert state.state(60.0, 300.0) == "healthy"
    state.drop(1)
    assert state.state(60.0, 300.0) == "degraded"
    clock[0] += 61.0
    assert state.state(60.0, 300.0) == "healthy", "a drop recovers after the window"


def test_a_half_full_queue_is_degraded():
    state = health.SignalHealth("traces")
    state.reset(enabled=True, capacity=10)
    state.queue_used = lambda: 5
    assert state.state(60.0, 300.0) == "degraded"
    state.queue_used = lambda: 4
    assert state.state(60.0, 300.0) == "healthy"


def test_the_endpoint_is_shown_without_userinfo_query_or_fragment(stub, telemetry_on):
    endpoint = stub.endpoint.replace("http://", "http://user:SENTINEL-PW@") + "/base?key=SENTINEL-Q#f"
    telemetry.configure(telemetry_settings(endpoint))
    block = _block()
    assert block["config"]["endpoint"] == stub.endpoint + "/base"
    assert "SENTINEL" not in str(block)


def test_a_snapshot_failure_reports_the_class_and_the_route_still_answers(monkeypatch):
    def boom():
        raise RuntimeError("SENTINEL-MESSAGE")

    monkeypatch.setattr(telemetry, "current_settings", boom)
    block = _block()
    assert block["error"] == "RuntimeError"
    assert "SENTINEL" not in str(block)


def test_an_unknown_key_in_the_snapshot_never_reaches_the_response(monkeypatch):
    real = health.snapshot

    def leaky():
        data = real()
        data["config"]["headers"] = "SENTINEL-SECRET"
        data["leak"] = "SENTINEL-SECRET"
        return data

    monkeypatch.setattr(health, "snapshot", leaky)
    assert "SENTINEL" not in str(_block())


def test_an_error_that_is_not_a_class_name_is_shown_as_error():
    """Review R1 of S2: `last_error.type` is a CLASS name. The pipeline passes `type(exc).__name__`
    or `HTTPnnn`; this is the backstop for a future caller handing over a message. Control: drop
    the `ERROR_CLASS` check in `SignalHealth.failure` → the text reaches the view."""
    state = health.SignalHealth("logs")
    state.reset(enabled=True, capacity=1)
    state.failure("connect to https://user:SENTINEL@host failed", 1)
    view = state.view(60, 300)
    assert view["last_error"]["type"] == "Error", view
    assert "SENTINEL" not in str(view)
