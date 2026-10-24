"""`telemetry_mode` (038 FR-38.23, FTASKS 8.13): run an existing suite with export OFF and again
with every signal ON against the OTLP stub.

The client-visible behaviour must be identical in both modes (the suite's own assertions), and in
`stub` mode the teardown also requires that spans were exported, that every export succeeded and
nothing was dropped, and that NOTHING unlisted was
offered to the allowlist (`attributes_dropped == 0`) — so every route, stream and socket test
doubles as a privacy and wiring check of the export path.

Use: `from tests.support.telemetry_mode import _stub_exports, telemetry_mode  # noqa: F401` plus
`pytestmark = pytest.mark.usefixtures("telemetry_mode")`.
"""

from __future__ import annotations

import pytest

from millm import telemetry
from millm.core.config import settings
from millm.telemetry import allowlist, counters, health
from tests.support.otlp_stub import OtlpStub


def _reset() -> None:
    telemetry.shutdown(0.5)
    telemetry.configure(settings)
    counters.STORE.reset(series_cap=settings.TELEMETRY_SERIES_CAP)
    allowlist.reset_for_tests()


@pytest.fixture(scope="module")
def _stub_exports():
    """Spans exported across the module's `stub` runs. ⚠ At least one module test must send
    traffic through the export path, or removing that wiring would leave stub mode green."""
    seen = {"spans": 0, "runs": 0}
    yield seen
    if seen["runs"]:
        assert seen["spans"] > 0, "no span reached the stub in any stub-mode test of this module"


@pytest.fixture(params=["off", "stub"])
def telemetry_mode(request, _stub_exports):
    _reset()
    if request.param == "off":
        yield "off"
        _reset()
        return
    stub = OtlpStub().start()
    telemetry.configure(settings.model_copy(update={
        "TELEMETRY_TRACES_ENABLED": True, "TELEMETRY_METRICS_ENABLED": True,
        "TELEMETRY_LOGS_ENABLED": True, "TELEMETRY_ENDPOINT": stub.endpoint,
        "TELEMETRY_ALLOW_PLAINTEXT": True, "TELEMETRY_METRIC_INTERVAL_S": 3600.0,
        "TELEMETRY_RETRY_HORIZON_S": 2.0, "TELEMETRY_EXPORT_TIMEOUT_S": 5.0,
    }))
    assert telemetry.any_signal_on()
    try:
        yield "stub"
        telemetry.shutdown(5.0)
        failed = {s: h.failed for s, h in health.SIGNALS.items() if h.failed}
        assert not failed, f"exports failed in stub mode: {failed}"
        dropped = {s: h.dropped for s, h in health.SIGNALS.items() if h.dropped}
        assert not dropped, f"records dropped in stub mode: {dropped}"
        _stub_exports["runs"] += 1
        _stub_exports["spans"] += len(stub.spans)
        assert allowlist.attributes_dropped() == 0, "an unlisted attribute was offered for export"
    finally:
        stub.stop()
        _reset()
