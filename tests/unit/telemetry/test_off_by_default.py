"""Export is OFF by default and off means free (FR-38.12; D31, D-08, D38.17; FTASKS 7.2, 7.7).

* Every telemetry setting is environment-only: classified `environment`, never writable, the
  switches default off, and no route body or query accepts a `TELEMETRY_*` name (control M12).
* T-S2-01: after a chat, a stream and a score with every switch off, no `opentelemetry.sdk*` or
  `opentelemetry.exporter*` module is loaded — in a SUBPROCESS, because other tests in this worker
  import the SDK.
* T-S2-02 / M23: with every switch on, `FASTAPI_OTEL_AUTO_CONFIGURE=true` and the standard
  `OTEL_EXPORTER_OTLP_ENDPOINT` pointing at the receiver, only miLLM's own scope reaches it — no
  `fastapi` (or `mcp`) scope — and no global provider is installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from millm.core.config import Settings, settings
from millm.services import runtime_settings as rs
from tests.unit.telemetry.conftest import flush

REPO = Path(__file__).resolve().parents[3]
TELEMETRY_NAMES = sorted(k for k in Settings.model_fields
                         if k.startswith("TELEMETRY_") or k == "OTEL_SDK_DISABLED")


def test_environment_classification():
    """M12: a TELEMETRY_* key in RUNTIME_KEYS turns this red."""
    assert len(TELEMETRY_NAMES) == 26, TELEMETRY_NAMES
    for key in TELEMETRY_NAMES:
        assert rs.group_of(key) == "environment", key
        assert key not in rs.RUNTIME_KEYS, key
        assert key in rs.NEVER_WRITABLE, key
    assert {"TELEMETRY_HEADERS", "TELEMETRY_TLS_CLIENT_KEY_FILE"} <= rs.SECRET_KEYS


def test_the_defaults_export_nothing():
    fresh = Settings(_env_file=None)
    assert not (fresh.TELEMETRY_TRACES_ENABLED or fresh.TELEMETRY_METRICS_ENABLED
                or fresh.TELEMETRY_LOGS_ENABLED)
    assert fresh.TELEMETRY_ENDPOINT is None
    assert fresh.TELEMETRY_STATELESS_EXPORT_PERMITTED is False
    assert fresh.TELEMETRY_ALLOW_PLAINTEXT is False
    from millm.telemetry.settings import load

    cfg = load(fresh)
    assert not cfg.any_on and cfg.refusal is None


def test_no_route_accepts_a_telemetry_name():
    """No body property and no query parameter of the live OpenAPI is a telemetry setting."""
    from millm.main import create_app

    spec = create_app().openapi()
    names: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    names.update(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(spec.get("components", {}))
    for path in spec["paths"].values():
        for op in path.values():
            if isinstance(op, dict):
                names.update(p["name"] for p in op.get("parameters", []))
                walk(op.get("requestBody", {}))
    assert names, "the walk found nothing"
    assert not {n for n in names if n.upper().startswith(("TELEMETRY_", "OTEL_"))}


# The settings page shows TELEMETRY_HEADERS and TELEMETRY_TLS_CLIENT_KEY_FILE only as set / not
# set: `tests/unit/api/test_environment_routes.py::TestListing::test_no_secret_appears_anywhere`
# carries a sentinel in each (and its PUT of each is refused), with every other secret.


# ── T-S2-01: off means free ──────────────────────────────────────────────────────

_SCRIPT = r"""
import asyncio, json, sys
sys.path.insert(0, {repo!r})
from tests.support import capture_pre038_goldens as cap

async def main():
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA, cap.BETA])
        r = h.post_chat(cap.BODY, {{"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}})
        assert r.status_code == 200, r.text
        assert r.json()["millm_probe_verdicts"]
        text = h.stream_chat(cap.BODY, {{}})
        assert "millm_probe_verdicts" in text
        s = h.client.post("/api/probes/score", json={{
            "probe_ids": [row.id for row in rows],
            "inputs": [{{"token_ids": [0, 2, 6, 7, 8], "prompt_tokens": 3}}],
            "export_results": True}})
        assert s.status_code == 200, s.text
        prom = h.client.get("/api/health/metrics/prometheus").text
        assert "millm_probe_evaluations_total" in prom
    loaded = sorted(m for m in sys.modules
                    if m.startswith(("opentelemetry.sdk", "opentelemetry.exporter")))
    print(json.dumps({{"loaded": loaded,
                      "api": "opentelemetry.trace" in sys.modules}}))

asyncio.run(main())
"""


def test_off_means_no_sdk_module_is_loaded():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TELEMETRY_", "OTEL_"))}
    env["MILLM_STRICT_RESULTS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _SCRIPT.format(repo=str(REPO))],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["loaded"] == [], report
    # The middleware parsed a traceparent with the API alone: the check is not vacuous.
    assert report["api"] is True


# ── T-S2-02 / M23: no foreign instrumentation ────────────────────────────────────


async def test_no_foreign_scopes(stub, telemetry_on, monkeypatch):
    """Everything on, FastAPI told to auto-configure and the standard OTLP endpoint set: only
    the `millm` scope reaches the receiver, and no global provider exists. Controls: remove the
    `telemetry=` argument in `create_app`; separately, install a global provider.

    ⚠ The app's LIFESPAN runs (FastAPI auto-configures inside it), with miLLM's own startup
    replaced by a no-op so no database or model load is involved.
    """
    import contextlib

    from fastapi.testclient import TestClient
    from opentelemetry import metrics, trace

    from tests.support import capture_pre038_goldens as cap

    monkeypatch.setenv("FASTAPI_OTEL_AUTO_CONFIGURE", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", stub.endpoint)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    pipeline = telemetry_on()
    assert pipeline is not None
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        app = h.client.app

        @contextlib.asynccontextmanager
        async def _no_startup(_app):
            yield

        app.router.lifespan_context = _no_startup
        with TestClient(app) as client:
            h._fresh_socket_throttle()
            response = client.post("/v1/chat/completions", json=cap.BODY)
            assert response.status_code == 200, response.text
            assert client.get("/api/missing?q=SENTINEL-QUERY").status_code == 404
        flush()
    assert stub.spans and stub.logs, "the receiver saw miLLM's own records"
    assert set(stub.scopes) == {"millm"}, sorted(set(stub.scopes))
    assert b"SENTINEL-QUERY" not in stub.all_bytes()
    # ⚠ No global provider: the API still hands out its deferred proxies.
    assert type(trace.get_tracer_provider()).__name__ == "ProxyTracerProvider"
    assert type(metrics.get_meter_provider()).__name__ == "_ProxyMeterProvider"


def test_otel_sdk_disabled_forces_every_signal_off(stub):
    """D38.38: the one standard variable honoured. Every switch on plus OTEL_SDK_DISABLED → no
    pipeline, every signal `disabled`, nothing sent."""
    from millm import telemetry
    from millm.telemetry import health
    from tests.unit.telemetry.conftest import telemetry_settings

    telemetry.configure(telemetry_settings(stub.endpoint, OTEL_SDK_DISABLED=True))
    assert telemetry._PIPELINE is None
    assert not telemetry.current_settings().any_on
    assert {s: h.view(60, 300)["state"] for s, h in health.SIGNALS.items()} == {
        "traces": "disabled", "metrics": "disabled", "logs": "disabled"}
    telemetry.emit_lifecycle(None, "import")
    assert stub.raw == []
