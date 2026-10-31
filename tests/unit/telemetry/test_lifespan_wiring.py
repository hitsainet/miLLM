"""`lifespan` configures telemetry at startup and drains it at shutdown, AFTER the probe records
(FTASKS 7.7, FTDD §7.4) — by RUNNING the real lifespan, not by reading its source.

Off by default: the same lifespan with the process settings starts nothing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from millm import telemetry
from tests.unit.batch_fixtures import batch_db  # noqa: F401 - fixture


def _quiet_startup(monkeypatch, batch_db):
    import millm.db.base as db_base
    import millm.main as main

    monkeypatch.setattr(db_base, "async_session_factory", batch_db)
    monkeypatch.setattr(main, "start_batch_api", AsyncMock())
    monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
    monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
    monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)
    return main


async def test_the_lifespan_starts_export_and_drains_it_after_the_records(monkeypatch, batch_db,
                                                                         stub):
    main = _quiet_startup(monkeypatch, batch_db)
    for key, value in {"TELEMETRY_LOGS_ENABLED": True, "TELEMETRY_ENDPOINT": stub.endpoint,
                       "TELEMETRY_ALLOW_PLAINTEXT": True}.items():
        monkeypatch.setattr(main.settings, key, value)
    order: list[str] = []
    from millm.services.inference_service import InferenceService

    real_drain = InferenceService.drain_probe_records.__func__

    async def drain(cls, timeout=10.0):
        order.append("records")
        return await real_drain(cls, timeout)

    monkeypatch.setattr(InferenceService, "drain_probe_records", classmethod(drain))
    real_shutdown = telemetry.shutdown

    def shutdown(deadline_s=None):
        order.append("telemetry")
        return real_shutdown(deadline_s)

    monkeypatch.setattr(telemetry, "shutdown", shutdown)
    async with main.lifespan(MagicMock()):
        assert telemetry.any_signal_on(), "lifespan did not configure telemetry"
        telemetry.emit_lifecycle(MagicMock(id="pr_0123456789ab", behaviour_key=None,
                                           detector_digest=None, threshold_revision=1,
                                           decision_revision=None, decision_digest=None),
                                 "import")
    assert order[-2:] == ["records", "telemetry"], order
    assert not telemetry.any_signal_on(), "the pipeline outlived the lifespan"
    assert [r["event_name"] for r in stub.logs] == ["mi.probe.lifecycle"], "not drained"


async def test_the_lifespan_with_default_settings_starts_nothing(monkeypatch, batch_db):

    main = _quiet_startup(monkeypatch, batch_db)
    async with main.lifespan(MagicMock()):
        assert not telemetry.any_signal_on()
        assert not telemetry.current_settings().any_on
    assert telemetry._PIPELINE is None
