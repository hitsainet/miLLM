"""Fixtures for the 038 S2 telemetry tests: the OTLP stub and a switch to turn export on against it.

Every test leaves telemetry OFF, exactly as the process default (the autouse `_telemetry_off`):
a pipeline left running would make every later test in the worker export.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from millm import telemetry
from millm.core.config import settings
from millm.telemetry import allowlist, counters
from tests.support.otlp_stub import OtlpStub


def telemetry_settings(endpoint: str | None, **overrides: Any) -> Any:
    """A copy of the process settings with every signal on against `endpoint` (plaintext)."""
    values = {
        "TELEMETRY_TRACES_ENABLED": True,
        "TELEMETRY_METRICS_ENABLED": True,
        "TELEMETRY_LOGS_ENABLED": True,
        "TELEMETRY_ENDPOINT": endpoint,
        "TELEMETRY_ALLOW_PLAINTEXT": True,
        "TELEMETRY_METRIC_INTERVAL_S": 3600.0,
        "TELEMETRY_RETRY_HORIZON_S": 2.0,
        "TELEMETRY_EXPORT_TIMEOUT_S": 2.0,
    }
    values.update(overrides)
    return settings.model_copy(update=values)


def reset_all() -> None:
    telemetry.shutdown(0.5)
    telemetry.configure(settings)
    counters.STORE.reset(series_cap=settings.TELEMETRY_SERIES_CAP)
    allowlist.reset_for_tests()


@pytest.fixture(autouse=True)
def _telemetry_off():
    reset_all()
    yield
    reset_all()


@pytest.fixture
def stub():
    s = OtlpStub().start()
    yield s
    s.stop()


@pytest.fixture
def telemetry_on(stub: OtlpStub) -> Callable[..., Any]:
    """`telemetry_on(**overrides)` → the running pipeline, exporting to `stub`."""

    def _on(**overrides: Any) -> Any:
        endpoint = overrides.pop("endpoint", stub.endpoint)
        telemetry.configure(telemetry_settings(endpoint, **overrides))
        return telemetry._PIPELINE

    return _on


def flush(deadline: float = 5.0) -> None:
    """Drain every queue to the receiver and export metrics once: shut the pipeline down."""
    telemetry.shutdown(deadline)
