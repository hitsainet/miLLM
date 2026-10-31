"""ENH-006 D75–D78 (2026-10-28): telemetry settings change while miLLM runs.

Through the LIVE app (`create_app`, the approval fixtures' SQLite database), against the in-process
OTLP stub:

* the operator's save of a switch, an endpoint and plaintext starts export (the receiver gets
  records, health counts them); switching off drains what was queued and leaves `_PIPELINE` None;
* a save of several telemetry keys rebuilds the pipeline ONCE, with every value already in place;
* the environment-only five (headers, client key, CA and client-cert paths, `OTEL_SDK_DISABLED`)
  are refused 422 to the operator and to an agent, and an endpoint carrying a user name, a query
  or a fragment is refused without echoing it;
* an agent's save waits (202, `telemetry_write`), changes nothing, and runs exactly once when the
  operator approves it — the history names the agent and the approval;
* a saved override survives a restart (load, then `configure()`), the startup load does not
  rebuild the pipeline itself, and clearing an override returns to the deployment's value;
* off means free: switching on with no endpoint, changing the sampler, and switching off again
  never imports an SDK module (subprocess).

Controls (recorded in `0xcc/reviews/038_runtime_telemetry_controls_2026-10-29.md`):
* the telemetry entry removed from `APPLY_HOOKS`           -> the toggle tests red
* the hook calling `configure()` without `shutdown()` first -> `..._drains_for_the_old_drain_deadline` red
* `_run_hooks` without its de-duplication                   -> `..._rebuilds_once` red
* `skip_hooks=STARTUP_DEFERRED_HOOKS` removed from the load -> the restart test red
* the gate on `PUT /api/environment` removed                -> the agent test red (200, applied)
* `check_written_endpoint`'s userinfo / query clause removed -> the refusal test red
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from millm import telemetry
from millm.core.config import settings
from millm.db.models.env_override import EnvOverride, EnvOverrideChange
from millm.services import runtime_settings as rs
from millm.telemetry import health
from tests.unit.approval_fixtures import AGENT, harness, rows  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

REPO = Path(__file__).resolve().parents[3]
BASE = "/api/environment"
WATCHED = ("TELEMETRY_LOGS_ENABLED", "TELEMETRY_TRACES_ENABLED", "TELEMETRY_ENDPOINT",
           "TELEMETRY_ALLOW_PLAINTEXT", "TELEMETRY_EXPORT_TIMEOUT_S")
ENV_ONLY = ("TELEMETRY_HEADERS", "TELEMETRY_TLS_CLIENT_KEY_FILE", "TELEMETRY_TLS_CA_FILE",
            "TELEMETRY_TLS_CLIENT_CERT_FILE", "OTEL_SDK_DISABLED")


@pytest.fixture
def configure_calls(monkeypatch) -> list[dict[str, Any]]:
    """Every `telemetry.configure()` call, with the watched settings as they stood AT the call
    (the hook reads `millm.telemetry.configure` at call time, so this sees the real wiring)."""
    calls: list[dict[str, Any]] = []
    real = telemetry.configure

    def spy(source: Any = None) -> None:
        calls.append({k: getattr(settings, k) for k in WATCHED} | {"source": source})
        real(source)

    monkeypatch.setattr(telemetry, "configure", spy)
    return calls


def _on(stub) -> dict[str, Any]:
    return {"TELEMETRY_ENDPOINT": stub.endpoint, "TELEMETRY_ALLOW_PLAINTEXT": True,
            "TELEMETRY_LOGS_ENABLED": True, "TELEMETRY_EXPORT_TIMEOUT_S": 2.0,
            "TELEMETRY_RETRY_HORIZON_S": 2.0}


async def _changes(h) -> list[Any]:
    async with h.db() as session:
        return list((await session.execute(
            select(EnvOverrideChange).order_by(EnvOverrideChange.id))).scalars().all())


async def _overrides(h) -> dict[str, Any]:
    async with h.db() as session:
        return {r.key: r.value for r in (await session.execute(select(EnvOverride))).scalars()}


# ── the operator's save ──────────────────────────────────────────────────────────


async def test_switching_on_exports_and_switching_off_drains(harness, stub,  # noqa: F811
                                                             configure_calls):
    h = harness
    assert telemetry._PIPELINE is None
    r = h.client.put(BASE, json={"values": _on(stub), "reason": "pilot"})
    assert r.status_code == 200, r.text
    assert telemetry.any_signal_on(), "the save did not start export"
    assert telemetry._PIPELINE.signals == ("logs",)
    assert telemetry._PIPELINE.cfg.endpoint == stub.endpoint
    # Records leave, and health counts them.
    telemetry.emit_lifecycle(None, "import")
    assert stub.wait_for(lambda s: len(s.logs) >= 1, timeout=10), "nothing reached the receiver"
    assert health.SIGNALS["logs"].view(60, 300)["exported"] >= 1
    # Queued at the moment of the switch-off: drained, not lost.
    pipeline = telemetry._PIPELINE
    sent_before = len(stub.logs)
    telemetry.emit_lifecycle(None, "arm")
    r = h.client.put(f"{BASE}/TELEMETRY_LOGS_ENABLED", json={"value": False})
    assert r.status_code == 200, r.text
    assert telemetry._PIPELINE is None and not telemetry.any_signal_on()
    assert pipeline._accepting is False and not pipeline._thread.is_alive()
    assert len(stub.logs) >= sent_before + 1, "the switch-off lost a queued record"
    assert {s: v.view(60, 300)["state"] for s, v in health.SIGNALS.items()} == {
        "traces": "disabled", "metrics": "disabled", "logs": "disabled"}
    # One rebuild per save: the five-key save, then the switch-off.
    assert len(configure_calls) == 2
    assert configure_calls[0]["TELEMETRY_LOGS_ENABLED"] is True
    assert configure_calls[1]["TELEMETRY_LOGS_ENABLED"] is False


async def test_a_switch_off_drains_for_the_old_drain_deadline(harness):  # noqa: F811
    """Against a SLOW receiver (0.3 s per send, one record per send), records queued when the
    switch goes off are still delivered: the save drains for `TELEMETRY_DRAIN_DEADLINE_S` before
    it rebuilds. Control: the hook calling `configure()` alone (which stops with no drain) loses
    them."""
    from tests.support.otlp_stub import OtlpStub

    slow = OtlpStub(mode="slow", slow_s=0.3).start()
    try:
        values = {**_on(slow), "TELEMETRY_BATCH_SIZE": 1, "TELEMETRY_DRAIN_DEADLINE_S": 10.0}
        assert harness.client.put(BASE, json={"values": values}).status_code == 200
        for action in ("import", "arm", "disarm", "delete"):
            telemetry.emit_lifecycle(None, action)
        r = harness.client.put(f"{BASE}/TELEMETRY_LOGS_ENABLED", json={"value": False})
        assert r.status_code == 200, r.text
        assert telemetry._PIPELINE is None
        assert len(slow.logs) == 4, f"the switch-off lost records ({len(slow.logs)} of 4 sent)"
    finally:
        slow.stop()


async def test_a_save_of_several_telemetry_keys_rebuilds_once(harness, stub,  # noqa: F811
                                                             configure_calls):
    """Five keys, one save: ONE `configure()`, and it saw every new value (never a half-applied
    configuration, never five drains)."""
    r = harness.client.put(BASE, json={"values": _on(stub)})
    assert r.status_code == 200, r.text
    assert [k["key"] for k in r.json()["data"]["settings"]] == list(_on(stub))
    assert len(configure_calls) == 1, configure_calls
    (call,) = configure_calls
    assert (call["TELEMETRY_ENDPOINT"], call["TELEMETRY_ALLOW_PLAINTEXT"],
            call["TELEMETRY_LOGS_ENABLED"], call["TELEMETRY_EXPORT_TIMEOUT_S"]) == (
        stub.endpoint, True, True, 2.0)
    assert call["source"] is None, "configure() must read the singleton, overrides applied"
    # A non-telemetry key in a save rebuilds nothing.
    harness.client.put(f"{BASE}/PROBE_MAX_ARMED", json={"value": 3})
    assert len(configure_calls) == 1


async def test_a_refused_configuration_shows_in_health_and_exports_nothing(  # noqa: F811
        harness, stub, configure_calls):
    """An http:// endpoint without the plaintext choice is SAVED (order of saves never matters)
    and refused when the pipeline is built: health says why, nothing leaves."""
    values = {**_on(stub), "TELEMETRY_ALLOW_PLAINTEXT": False}
    assert harness.client.put(BASE, json={"values": values}).status_code == 200
    assert telemetry._PIPELINE is None
    view = health.SIGNALS["logs"].view(60, 300)
    assert view["state"] == "unavailable" and view["last_error"]["type"] == "PlaintextNotAllowed"
    assert stub.raw == []


# ── what stays environment-only, and what is refused ─────────────────────────────


def test_the_settings_cards_are_exactly_the_writable_telemetry_keys():
    """The writable set (`TELEMETRY_KEYS - TELEMETRY_ENV_ONLY`) and the registered cards
    (`RUNTIME_SPECS`) are two lists of one thing. If they drift, a key is either writable with no
    card or carded with no write; the suite otherwise noticed only as a KeyError in conftest."""
    from millm.services import runtime_settings as rs

    carded = rs.TELEMETRY_KEYS & rs.RUNTIME_KEYS
    assert carded == rs.TELEMETRY_RUNTIME_KEYS
    assert not (rs.TELEMETRY_ENV_ONLY & rs.RUNTIME_KEYS)
    assert set(ENV_ONLY) == set(rs.TELEMETRY_ENV_ONLY)


@pytest.mark.parametrize("headers", [{}, AGENT], ids=["operator", "agent"])
async def test_the_environment_only_five_are_refused(harness, headers):  # noqa: F811
    h = harness
    for key in ENV_ONLY:
        before = getattr(settings, key)
        for r in (h.client.put(f"{BASE}/{key}", json={"value": "x"}, headers=headers),
                  h.client.delete(f"{BASE}/{key}", headers=headers),
                  h.client.put(BASE, json={"values": {key: "x"}}, headers=headers)):
            assert r.status_code == 422, (key, r.status_code, r.text)
            assert r.json()["error"]["code"] == "SETTING_NOT_RUNTIME", key
        assert getattr(settings, key) == before, key
    assert await rows(h.db) == [], "a refused request was filed for approval"
    assert await _overrides(h) == {}


@pytest.mark.parametrize("url", [
    "https://user:SENTINEL-PW@otlp.example:4318/base",
    "https://SENTINEL-TOKEN@otlp.example:4318/base",
    "https://otlp.example:4318/base?api_key=SENTINEL-Q",
    "https://otlp.example:4318/base#SENTINEL-F",
    "ftp://otlp.example/SENTINEL",
    "not a url SENTINEL",
])
@pytest.mark.parametrize("headers", [{}, AGENT], ids=["operator", "agent"])
async def test_an_endpoint_with_credentials_or_junk_is_refused(harness, url, headers):  # noqa: F811
    h = harness
    before = settings.TELEMETRY_ENDPOINT
    for r in (h.client.put(f"{BASE}/TELEMETRY_ENDPOINT", json={"value": url}, headers=headers),
              h.client.put(BASE, json={"values": {"TELEMETRY_ENDPOINT": url}}, headers=headers)):
        assert r.status_code == 422, r.text
        assert r.json()["error"]["code"] == "SETTING_INVALID"
        assert "SENTINEL" not in r.text, "the refusal echoed the value"
    assert settings.TELEMETRY_ENDPOINT == before
    assert await rows(h.db) == [] and await _overrides(h) == {}


async def test_the_history_never_stores_the_environments_credential(harness, monkeypatch,  # noqa: F811
                                                                    stub):
    url = "https://user:SENTINEL-URL-PW@otlp.example:4318/base?api_key=SENTINEL-Q"
    monkeypatch.setattr(settings, "TELEMETRY_ENDPOINT", url)
    h = harness
    assert h.client.put(f"{BASE}/TELEMETRY_ENDPOINT",
                        json={"value": stub.endpoint}).status_code == 200
    assert h.client.delete(f"{BASE}/TELEMETRY_ENDPOINT").status_code == 200
    assert settings.TELEMETRY_ENDPOINT == url, "the reset did not restore the deployment value"
    history = h.client.get(f"{BASE}/history").text
    stored = json.dumps([(c.previous_value, c.new_value) for c in await _changes(h)])
    # An agent's request shows the deployment's endpoint on its card: redacted there too.
    pending = h.client.put(f"{BASE}/TELEMETRY_ENDPOINT", json={"value": stub.endpoint},
                           headers=AGENT)
    assert pending.status_code == 202, pending.text
    card = pending.json()["data"]["summary"]["Receiver endpoint (TELEMETRY_ENDPOINT)"]
    assert card == f"now https://otlp.example:4318/base; after approval {stub.endpoint}"
    for text in (history, stored, pending.text, h.client.get("/api/approvals").text):
        assert "SENTINEL" not in text
    assert "https://otlp.example:4318/base" in stored


def test_the_instance_id_stays_the_same_across_rebuilds(monkeypatch):
    """With no TELEMETRY_SERVICE_INSTANCE_ID, the random id is drawn ONCE per process: a save
    rebuilds the pipeline, and must not make this process look like a new instance."""
    from millm.telemetry.settings import load

    monkeypatch.setattr(settings, "TELEMETRY_SERVICE_INSTANCE_ID", None)
    first, second = load(settings), load(settings)
    assert first.service_instance_id == second.service_instance_id
    assert len(first.service_instance_id) == 16


# ── an agent's save (D76) ────────────────────────────────────────────────────────


async def test_an_agent_save_waits_then_runs_exactly_once(harness, stub,  # noqa: F811
                                                        configure_calls):
    h = harness
    r = h.client.put(BASE, json={"values": _on(stub), "reason": "turn on the pilot"},
                     headers=AGENT)
    assert r.status_code == 202, r.text
    assert r.headers["X-miLLM-Approval"] == "pending"
    data = r.json()["data"]
    assert data["action"] == "telemetry_write" and data["status"] == "pending"
    summary = data["summary"]
    assert summary["Export result logs (TELEMETRY_LOGS_ENABLED)"] == "now off; after approval on"
    assert summary["Receiver endpoint (TELEMETRY_ENDPOINT)"] == (
        f"now not set; after approval {stub.endpoint}")
    assert summary["Reason"] == "turn on the pilot"
    # Nothing applied: no singleton change, no override row, no rebuild, nothing exported.
    assert settings.TELEMETRY_LOGS_ENABLED is False and settings.TELEMETRY_ENDPOINT is None
    assert await _overrides(h) == {} and configure_calls == [] and telemetry._PIPELINE is None

    approved = h.client.post(f"/api/approvals/{data['approval_id']}/approve")
    assert approved.status_code == 202, approved.text
    assert approved.json()["data"]["status"] == "executed", approved.text
    assert len(configure_calls) == 1, "approval must apply exactly once"
    assert configure_calls[0]["TELEMETRY_LOGS_ENABLED"] is True
    assert configure_calls[0]["TELEMETRY_ENDPOINT"] == stub.endpoint
    assert telemetry.any_signal_on()
    assert await _overrides(h) == {k: v for k, v in _on(stub).items()}
    changes = await _changes(h)
    assert len(changes) == len(_on(stub))
    assert {(c.actor, c.approval_id, c.reason) for c in changes} == {
        ("agent:test-bot", data["approval_id"], "turn on the pilot")}

    again = h.client.post(f"/api/approvals/{data['approval_id']}/approve")
    assert again.status_code == 409 and len(configure_calls) == 1
    assert len(await _changes(h)) == len(_on(stub))


async def test_an_agent_reset_waits_and_shows_the_deployment_value(harness, stub):  # noqa: F811
    h = harness
    assert h.client.put(BASE, json={"values": _on(stub)}).status_code == 200
    r = h.client.delete(f"{BASE}/TELEMETRY_LOGS_ENABLED", params={"reason": "stop"},
                        headers=AGENT)
    assert r.status_code == 202, r.text
    assert r.json()["data"]["summary"]["Export result logs (TELEMETRY_LOGS_ENABLED)"] == (
        "now on; after approval off (the environment's value)")
    assert settings.TELEMETRY_LOGS_ENABLED is True and telemetry.any_signal_on()


async def test_an_agent_save_of_other_keys_runs_and_names_the_agent(harness):  # noqa: F811
    """The gate's predicate: only telemetry keys wait. Another key runs (Feature 35's posture),
    and the history now names the agent instead of calling it the operator."""
    h = harness
    r = h.client.put(f"{BASE}/PROBE_MAX_ARMED", json={"value": 3}, headers=AGENT)
    assert r.status_code == 200, r.text
    assert await rows(h.db) == []
    (change,) = await _changes(h)
    assert (change.actor, change.approval_id) == ("agent:test-bot", None)


@pytest.mark.parametrize("key", ["telemetry_logs_enabled", " TELEMETRY_LOGS_ENABLED",
                                 "OTEL_SDK_DISABLED"])
async def test_a_near_miss_telemetry_name_gates_and_is_then_refused(harness, key):  # noqa: F811
    """Fail closed: the predicate reads the name generously; the route's own check refuses."""
    r = harness.client.put(f"{BASE}/{key}", json={"value": True}, headers=AGENT)
    assert r.status_code in (404, 422), r.text
    assert await rows(harness.db) == []


# ── restart (D77) ────────────────────────────────────────────────────────────────


async def test_an_override_survives_a_restart_and_clearing_returns_to_the_deployment(  # noqa: F811
        harness, stub, configure_calls):
    h = harness
    deployment = {k: getattr(settings, k) for k in rs.TELEMETRY_RUNTIME_KEYS}
    assert h.client.put(BASE, json={"values": _on(stub)}).status_code == 200
    # The process ends: the pipeline stops, memory is forgotten, the singleton is the deployment.
    telemetry.shutdown(0.5)
    rs._forget_for_tests()
    for key, value in deployment.items():
        setattr(settings, key, value)
    telemetry.configure(settings)
    assert telemetry._PIPELINE is None
    configure_calls.clear()

    # `main.lifespan`: load the saved overrides FIRST, then configure telemetry once.
    report = await rs.load_runtime_overrides_on_startup(h.db)
    assert set(_on(stub)) <= set(report["applied"]) and report["ignored"] == []
    assert configure_calls == [], "the startup load rebuilt the pipeline before lifespan does"
    assert settings.TELEMETRY_LOGS_ENABLED is True
    telemetry.configure()
    assert telemetry.any_signal_on() and telemetry._PIPELINE.cfg.endpoint == stub.endpoint

    # Clearing the switch's override returns it to the deployment's value (off).
    r = h.client.delete(f"{BASE}/TELEMETRY_LOGS_ENABLED")
    assert r.status_code == 200 and r.json()["data"]["source"] == "env"
    assert settings.TELEMETRY_LOGS_ENABLED is deployment["TELEMETRY_LOGS_ENABLED"] is False
    assert telemetry._PIPELINE is None


def test_main_loads_the_overrides_before_it_configures_telemetry():
    """D77 rests on this order in `lifespan` (an override applied after `configure()` would sit
    on the singleton while the pipeline ran on the deployment's values until the next save)."""
    import inspect

    from millm import main

    source = inspect.getsource(main.lifespan)
    load = source.index("await load_runtime_overrides_on_startup(")
    configure = source.index("telemetry.configure()")
    assert load < configure


# ── off means free (subprocess: other tests in this worker import the SDK) ───────

_SCRIPT = r"""
import json, sys
sys.path.insert(0, {repo!r})
from millm.services import runtime_settings as rs
from millm.telemetry import health
from millm import telemetry
rs._apply({{"TELEMETRY_TRACES_SAMPLER_RATIO": 0.5}})
rs._apply({{"TELEMETRY_TRACES_ENABLED": True, "TELEMETRY_METRICS_ENABLED": True}})
state = health.SIGNALS["traces"].view(60, 300)
rs._apply({{"TELEMETRY_TRACES_ENABLED": False, "TELEMETRY_METRICS_ENABLED": False}})
loaded = sorted(m for m in sys.modules
                if m.startswith(("opentelemetry.sdk", "opentelemetry.exporter")))
print(json.dumps({{"loaded": loaded, "state": state["state"],
                  "error": (state["last_error"] or {{}}).get("type"),
                  "pipeline": telemetry._PIPELINE is not None,
                  "after": health.SIGNALS["traces"].view(60, 300)["state"]}}))
"""


def test_saves_that_never_reach_a_receiver_import_no_sdk_module():
    """Switches on with no endpoint (refused: NoEndpoint), the sampler changed, switches off: the
    hook ran each time (health shows the refusal, then disabled) and no SDK module was imported."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TELEMETRY_", "OTEL_"))}
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _SCRIPT.format(repo=str(REPO))],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report == {"loaded": [], "state": "unavailable", "error": "NoEndpoint",
                      "pipeline": False, "after": "disabled"}, report
