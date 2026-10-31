"""The benchmark launcher (FTASKS 13.1, `tests/hardware/otel_bench_server.py`) on the CPU.

It switches the telemetry arm in the process by calling the product's own `telemetry.configure()`
with a copy of the process settings, and times the product's enqueue. What must hold:

* a preset changes ONLY its switches (and the sampler / stateless flag): never the endpoint,
  headers or TLS — no request can name a destination;
* the arm route really starts and stops the product's pipeline, and the product's health block
  says so (the harness checks that block);
* the enqueue timer is wired into the product's classes — a real `Pipeline` instance, built before
  or after the patch, is timed — and is removed cleanly;
* the routes answer loopback clients only, and the PRODUCT app has no bench route and imports
  nothing from this file. Since ENH-006 D75 the product DOES switch export by a request — its
  own Settings route, which writes a recorded override and, for an agent, waits for the
  operator (D76) — so the product-side test now pins that route as the gated one instead of
  asserting that no route exists (the OT-D5 claim it used to make).
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests" / "hardware"))
import otel_bench_server as bench  # noqa: E402

from millm import telemetry  # noqa: E402
from millm.core.config import settings  # noqa: E402
from millm.telemetry import health  # noqa: E402
from tests.support.otlp_stub import OtlpStub  # noqa: E402


@pytest.fixture
def stub():
    s = OtlpStub().start()
    yield s
    s.stop()


@pytest.fixture
def timer():
    t = bench.EnqueueTimer()
    yield t
    t.uninstall()
    telemetry.shutdown(0.5)
    telemetry.configure(settings)


def _base(endpoint: str):
    return settings.model_copy(update={
        "TELEMETRY_ENDPOINT": endpoint, "TELEMETRY_ALLOW_PLAINTEXT": True,
        "TELEMETRY_METRIC_INTERVAL_S": 3600.0, "TELEMETRY_EXPORT_TIMEOUT_S": 2.0,
        "TELEMETRY_RETRY_HORIZON_S": 2.0})


def _client(timer, base, hosts=frozenset({"testclient"})) -> TestClient:
    app = FastAPI()
    app.include_router(bench.build_router(timer, allowed_hosts=hosts, settings_source=lambda: base))
    return TestClient(app)


def test_a_preset_changes_only_its_switches():
    base = _base("http://127.0.0.1:1")
    allowed = set(bench.SWITCHES) | {"TELEMETRY_TRACES_SAMPLER_RATIO",
                                     "TELEMETRY_STATELESS_EXPORT_PERMITTED"}
    for name in bench.PRESETS:
        assert set(bench.PRESETS[name]) <= allowed
        out = bench.preset_settings(base, name)
        changed = {k for k in type(base).model_fields if getattr(out, k) != getattr(base, k)}
        assert changed <= set(bench.PRESETS[name]), (name, changed)
        assert out.TELEMETRY_ENDPOINT == base.TELEMETRY_ENDPOINT
    assert not any(bench.PRESETS["off"].values())
    with pytest.raises(ValueError):
        bench.preset_settings(base, "elsewhere")


def test_the_arm_route_starts_and_stops_the_products_pipeline(stub, timer):
    client = _client(timer, _base(stub.endpoint))
    assert not timer.installed
    state = client.post("/__otel_bench/arm", json={"preset": "on"}).json()
    assert telemetry.any_signal_on() and state["pipeline_running"]
    # The route itself installed the enqueue timer (nothing else did): a real enqueue is timed.
    assert timer.installed
    telemetry._PIPELINE.enqueue_log("mi.probe.lifecycle", "log_lifecycle",
                                    {"mi.lifecycle.action": "import"}, None, time.time_ns())
    assert timer.calls["enqueue_log"] == 1
    assert health.snapshot()["enabled"] is True
    assert {s["state"] for s in health.snapshot()["signals"].values()} <= {"healthy", "degraded"}
    assert telemetry._PIPELINE.cfg.sampler_ratio == 1.0
    client.post("/__otel_bench/arm", json={"preset": "on_sampler0"})
    assert telemetry._PIPELINE.cfg.sampler_ratio == 0.0
    client.post("/__otel_bench/arm", json={"preset": "on_stateless"})
    assert telemetry._PIPELINE.cfg.stateless_export_permitted is True
    state = client.post("/__otel_bench/arm", json={"preset": "off"}).json()
    assert not telemetry.any_signal_on() and telemetry._PIPELINE is None
    assert state["telemetry"]["enabled"] is False
    assert client.post("/__otel_bench/arm", json={"preset": "nowhere"}).status_code == 400


def test_the_enqueue_timer_times_a_real_pipeline_built_before_or_after_the_patch(stub, timer):
    telemetry.configure(bench.preset_settings(_base(stub.endpoint), "on"))
    before = telemetry._PIPELINE  # built BEFORE the patch
    timer.install()
    assert before.enqueue_log("mi.probe.lifecycle", "log_lifecycle",
                              {"mi.lifecycle.action": "import"}, None, time.time_ns())
    assert timer.calls["enqueue_log"] == 1
    client = _client(timer, _base(stub.endpoint))
    client.post("/__otel_bench/arm", json={"preset": "on"})  # a NEW pipeline, after the patch
    assert telemetry._PIPELINE is not before and timer.calls["enqueue_log"] == 0  # arm resets
    telemetry._PIPELINE.enqueue_log("mi.probe.lifecycle", "log_lifecycle",
                                    {"mi.lifecycle.action": "arm"}, None, time.time_ns())
    state = client.get("/__otel_bench/state").json()
    assert state["enqueue"]["calls"]["enqueue_log"] >= 1
    assert all(v > 0 for v in state["enqueue"]["samples_us"]["enqueue_log"])
    assert client.get("/__otel_bench/state?samples=0").json()["enqueue"].get("samples_us") is None
    assert stub.wait_for(lambda s: len(s.logs) >= 1, timeout=10)
    timer.uninstall()
    from millm.telemetry import pipeline

    assert not hasattr(pipeline.Pipeline.enqueue_log, "__otel_bench_kind__")
    assert not hasattr(telemetry.emit_results, "__otel_bench_kind__")


def test_the_routes_answer_loopback_clients_only(timer):
    client = _client(timer, _base("http://127.0.0.1:1"), hosts=bench.LOOPBACK)
    assert client.get("/__otel_bench/state").status_code == 403
    assert client.post("/__otel_bench/arm", json={"preset": "on"}).status_code == 403
    assert not telemetry.any_signal_on()


def test_the_product_switches_export_only_through_its_gated_settings_route():
    """The product app has no bench route and never imports the launcher; its one way to switch
    export at run time is the Settings route (D75), every write of which an agent must have
    approved (D76: `telemetry_write` on the live route table)."""
    from millm.api.approval_gate import gated_routes
    from millm.main import create_app

    app = create_app()
    paths = list(app.openapi()["paths"])
    assert "/v1/chat/completions" in paths  # the reader sees included routers (positive control)
    assert not any(p.startswith(bench.ROUTE_PREFIX) for p in paths)
    assert TestClient(app).get(f"{bench.ROUTE_PREFIX}/state").status_code == 404
    telemetry_writes = {(m, p) for m, p, action in gated_routes(app)
                        if action == "telemetry_write"}
    assert telemetry_writes == {("PUT", "/api/environment"), ("PUT", "/api/environment/{key}"),
                                ("DELETE", "/api/environment/{key}")}
    offenders = []
    for path in (REPO / "millm").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any("otel_bench_server" in n for n in names):
                offenders.append(str(path))
    assert offenders == []


def test_the_launcher_mounts_on_the_fastapi_app_inside_the_socketio_wrapper():
    from millm import main as millm_main

    # `main()` attaches to `millm.main.app.other_asgi_app`: it must be the FastAPI app.
    assert isinstance(millm_main.app.other_asgi_app, FastAPI)
    app = FastAPI()
    t = bench.attach(app)
    assert {"/__otel_bench/arm", "/__otel_bench/state", "/__otel_bench/reset"} <= set(
        app.openapi()["paths"])
    assert isinstance(t, bench.EnqueueTimer)
    # Attached to the PRODUCT's app (a fresh one, as `main()` attaches to `millm.main.app`'s), the
    # route is reached — and refuses this non-loopback test client (403, not 404).
    from millm.main import create_app

    product = create_app()
    bench.attach(product)
    assert TestClient(product).get(f"{bench.ROUTE_PREFIX}/state").status_code == 403
