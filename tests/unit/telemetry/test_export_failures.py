"""Export failures never touch inference (FR-38.22, AT09; FTDD §7.6; FTASKS 8.12).

A receiver that is absent, answers 401, or is slower than the export timeout, and a queue held
full: every request still answers 200 with its probe results, the counters move (`failed`,
`dropped`) and the signal's state changes. With the queue held full the request completes —
the enqueue never waits.

Control M22: make the enqueue blocking (`put` instead of `put_nowait`).
"""

from __future__ import annotations

import threading

import pytest

from millm.telemetry import health
from tests.support import capture_pre038_goldens as cap
from tests.unit.telemetry.conftest import flush


def _chat_ok(h) -> dict:
    r = h.post_chat(cap.BODY, {})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["millm_probe_verdicts"], "the request kept its probe results"
    return body


@pytest.mark.parametrize("mode, error, state", [
    ("absent", "ConnectionError", "degraded"),
    ("401", "HTTP401", "unavailable"),
    ("slow", "ReadTimeout", "degraded"),
])
async def test_a_failing_receiver_leaves_inference_alone(stub, telemetry_on, mode, error, state):
    telemetry_on(TELEMETRY_EXPORT_TIMEOUT_S=0.3, TELEMETRY_RETRY_HORIZON_S=0.0)
    if mode == "absent":
        stub.go_absent()
    else:
        stub.mode = mode
        # ⚠ Longer than the test runs (review R1 of S2): at 1.5 s, a cold first run took longer
        # than that between the first POST and the final assertion, so the stub finished sleeping,
        # DECODED the timed-out batch and `stub.spans == []` went red. The handler is a daemon.
        stub.slow_s = 30.0
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        _chat_ok(h)
        _chat_ok(h)
        flush()
    for signal in ("traces", "logs"):
        view = health.SIGNALS[signal].view(60, 300)
        assert view["exported"] == 0 and view["failed"] >= 1, (signal, view)
        assert view["last_error"]["type"] == error, (signal, view)
        assert view["state"] == state, (signal, view)
    assert stub.spans == [] and stub.logs == []


async def test_the_request_completes_with_the_queue_held_full(stub, telemetry_on, monkeypatch):
    pipeline = telemetry_on(TELEMETRY_QUEUE_SIZE=1)
    gate = threading.Event()
    real_take = pipeline._take
    outcome: dict = {}
    try:
        async with cap.Harness() as h:
            await h.arm([cap.ALPHA, cap.BETA])
            # The import records drain first; from here on the export thread is held, so the
            # queues stay full for the whole request.
            assert stub.wait_for(
                lambda s: s.logs and all(q.empty() for q in pipeline.queues.values()), timeout=10)

            def held(q):
                gate.wait()
                return real_take(q)

            monkeypatch.setattr(pipeline, "_take", held)
            pipeline._wake.set()

            def call():
                try:
                    outcome["body"] = _chat_ok(h)
                except BaseException as exc:  # noqa: BLE001
                    outcome["error"] = exc

            worker = threading.Thread(target=call, daemon=True)
            worker.start()
            worker.join(timeout=60)
            finished = not worker.is_alive()
            gate.set()
            worker.join(timeout=60)
    finally:
        gate.set()
    assert finished, "the request waited on a full telemetry queue"
    assert "error" not in outcome, outcome.get("error")
    assert len(outcome["body"]["millm_probe_verdicts"]) == 3
    dropped = {s: health.SIGNALS[s].view(60, 300) for s in ("traces", "logs")}
    assert dropped["traces"]["dropped"] >= 1 and dropped["logs"]["dropped"] >= 1, dropped
    assert dropped["traces"]["state"] == "degraded"
    flush()


async def test_a_failing_signal_never_holds_the_others(stub, telemetry_on):
    """Review (head-of-line): logs answered 503 for the whole retry horizon while traces are
    healthy. The traces must still be exported while logs wait in their own retry slot."""
    stub.mode = "fail_logs"
    telemetry_on(TELEMETRY_RETRY_HORIZON_S=30.0, TELEMETRY_METRICS_ENABLED=False)
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        assert stub.wait_for(lambda s: s.requests_by_path.get("/v1/logs", 0) >= 1, timeout=10)
        _chat_ok(h)
        assert stub.wait_for(lambda s: any(sp["name"] == "chat" for sp in s.spans), timeout=10), \
            "traces waited behind the failing logs signal"
        assert health.SIGNALS["logs"].view(60, 300)["state"] == "degraded"
        flush()
    assert health.SIGNALS["traces"].view(60, 300)["dropped"] == 0


def test_every_measurement_that_does_not_fit_is_counted(stub, telemetry_on, monkeypatch):
    """Review: a full metrics queue drops EACH measurement that does not fit, and counts each."""
    pipeline = telemetry_on(TELEMETRY_QUEUE_SIZE=2, TELEMETRY_TRACES_ENABLED=False,
                            TELEMETRY_LOGS_ENABLED=False)
    monkeypatch.setattr(pipeline, "_record_measurements", lambda: None)  # nothing drains
    items = [("millm.probe.score_margin", 0.1, {"probe_id": "pr_0123456789ab", "window": "all"})]
    pipeline.enqueue_measurements(items * 6)
    assert health.SIGNALS["metrics"].view(60, 300)["dropped"] == 4


def test_a_thread_outliving_shutdown_cannot_touch_the_next_configurations_health(
        stub, telemetry_on):
    """Review: if the export thread is still in a POST when shutdown's join gives up, the pipeline
    is RETIRED — its late success/failure goes to a detached counter, never to the live health of
    the configuration that replaced it."""
    from types import SimpleNamespace

    from millm import telemetry

    old = telemetry_on()
    real_thread = old._thread
    old._thread = SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: True)
    telemetry.shutdown(0.0)
    assert old._retired is True
    telemetry_on()  # the next configuration, fresh counters
    old._health("logs").success(5)
    old._health("logs").failure("ConnectionError", 3)
    view = health.SIGNALS["logs"].view(60, 300)
    assert (view["exported"], view["failed"], view["last_error"]) == (0, 0, None)
    old._stop.set()
    old._wake.set()
    real_thread.join(timeout=5)


# ── hostile receivers (review R1 of S2) ──────────────────────────────────────────


async def test_a_redirect_is_a_failure_and_is_never_followed(stub, telemetry_on):
    """A 307 to another host must not re-POST the records (and the credential headers) there:
    `allow_redirects=False`. Control: `allow_redirects=True` → the second receiver gets them."""
    from millm import telemetry
    from tests.support.otlp_stub import OtlpStub

    elsewhere = OtlpStub().start()
    try:
        stub.mode = "redirect"
        stub.redirect_to = elsewhere.endpoint
        telemetry_on(TELEMETRY_RETRY_HORIZON_S=0.0, TELEMETRY_METRICS_ENABLED=False,
                     TELEMETRY_HEADERS="authorization=Bearer SENTINEL-CRED")
        telemetry.emit_lifecycle(None, "import")
        flush()
        assert stub.requests_by_path.get("/v1/logs") == 1, stub.requests_by_path
        assert elsewhere.raw == [], "the redirect was followed to another receiver"
    finally:
        elsewhere.stop()
    view = health.SIGNALS["logs"].view(60, 300)
    assert (view["exported"], view["failed"]) == (0, 1), view
    assert view["last_error"]["type"] == "HTTP307", view


async def test_a_huge_response_body_is_never_read(stub, telemetry_on):
    """The receiver answers 200 and declares a 1 GB body it never sends. Only the status is
    used: the export succeeds at once, the body is not read (nor decoded). Control: drop
    `stream=True` → `requests` reads the body, times out and counts the records failed."""
    import time

    from millm import telemetry

    stub.mode = "huge"
    stub.slow_s = 30.0
    telemetry_on(TELEMETRY_RETRY_HORIZON_S=0.0, TELEMETRY_METRICS_ENABLED=False,
                 TELEMETRY_EXPORT_TIMEOUT_S=1.0)
    started = time.monotonic()
    telemetry.emit_lifecycle(None, "import")
    assert stub.wait_for(lambda s: s.logs, timeout=10)
    flush()
    view = health.SIGNALS["logs"].view(60, 300)
    assert (view["exported"], view["failed"], view["last_error"]) == (1, 0, None), view
    assert time.monotonic() - started < 5.0


def test_the_signal_path_is_appended_to_the_path_not_the_query(stub, telemetry_on):
    """Review R1 of S2: a base URL with a query (`/base?tenant=t1`) must POST to
    `/base/v1/logs?tenant=t1`. Control: the old string append → `/base?tenant=t1/v1/logs`."""
    from millm import telemetry
    from millm.telemetry.pipeline import signal_url

    telemetry_on(endpoint=stub.endpoint + "/base/?tenant=t1#frag",
                 TELEMETRY_METRICS_ENABLED=False)
    telemetry.emit_lifecycle(None, "import")
    flush()
    assert [path for path, _body, _headers in stub.raw] == ["/base/v1/logs?tenant=t1"]
    assert signal_url("https://h:4318", "traces") == "https://h:4318/v1/traces"
    assert signal_url("https://u:p@h/x/", "metrics") == "https://u:p@h/x/v1/metrics"


# ── 12.5a F6: what shutdown loses is LOGGED, not only counted in memory ─────────────────
#: The 12.5 run lost records at a restart and the only witness was a counter that died with the
#: process. Control F6: delete the `telemetry_shutdown_lost` warning in `Pipeline.shutdown`.

_LOST = "millm.telemetry.pipeline"


def _lost_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == _LOST and r.getMessage().startswith("telemetry_shutdown_lost")]


def test_f6_records_lost_at_shutdown_are_logged_with_their_counts(stub, telemetry_on,
                                                                   monkeypatch, caplog):
    import logging
    import time

    from millm import telemetry

    pipeline = telemetry_on(TELEMETRY_TRACES_ENABLED=False, TELEMETRY_METRICS_ENABLED=False)
    monkeypatch.setattr(pipeline, "_service", lambda signal: False)  # nothing is ever sent
    sentinel = "SENTINEL-LOST-RECORD"
    for _ in range(3):
        assert pipeline.enqueue_log("mi.probe.lifecycle", "log_lifecycle",
                                    {"mi.lifecycle.action": "arm", "mi.lifecycle.code": sentinel},
                                    None, time.time_ns())
    # A batch of two waiting for its retry when shutdown begins: never retried, counted failed.
    pipeline._retry["logs"] = (b"", 2, time.monotonic() + 60, 1, time.monotonic() + 60)
    caplog.set_level(logging.WARNING, logger=_LOST)
    telemetry.shutdown(0.2)
    view = health.SIGNALS["logs"].view(60, 300)
    assert (view["dropped"], view["failed"]) == (3, 2), view
    assert _lost_lines(caplog) == ["telemetry_shutdown_lost signal=logs dropped=3 failed=2"]
    assert sentinel not in caplog.text


def test_f6_a_clean_shutdown_logs_no_loss(stub, telemetry_on, caplog):
    import logging
    import time

    from millm import telemetry

    pipeline = telemetry_on()
    pipeline.enqueue_log("mi.probe.lifecycle", "log_lifecycle", {"mi.lifecycle.action": "arm"},
                         None, time.time_ns())
    caplog.set_level(logging.WARNING, logger=_LOST)
    telemetry.shutdown(5.0)
    assert health.SIGNALS["logs"].view(60, 300)["exported"] == 1
    assert _lost_lines(caplog) == []


def test_r1_a_shutdown_that_loses_only_retries_is_logged(stub, telemetry_on, caplog):
    """12.5a R1: a loss that is ALL `failed` (abandoned retries, nothing left queued) is a loss
    too. Control R1-F: log only when `dropped`."""
    import logging
    import time

    from millm import telemetry

    pipeline = telemetry_on(TELEMETRY_TRACES_ENABLED=False, TELEMETRY_METRICS_ENABLED=False)
    pipeline._retry["logs"] = (b"", 2, time.monotonic() + 60, 1, time.monotonic() + 60)
    caplog.set_level(logging.WARNING, logger=_LOST)
    telemetry.shutdown(0.2)
    assert _lost_lines(caplog) == ["telemetry_shutdown_lost signal=logs dropped=0 failed=2"]


def test_r1_losses_before_shutdown_are_not_reported_as_shutdown_losses(stub, telemetry_on,
                                                                        monkeypatch, caplog):
    """12.5a R1: the line counts what SHUTDOWN lost (the difference over it), not the process's
    running totals, which a full queue earlier in the day already counted. Control R1-D: take
    the `before` snapshot as zeros."""
    import logging
    import time

    from millm import telemetry

    pipeline = telemetry_on(TELEMETRY_TRACES_ENABLED=False, TELEMETRY_METRICS_ENABLED=False)
    health.SIGNALS["logs"].drop(5)  # lost long before shutdown (a full queue)
    monkeypatch.setattr(pipeline, "_service", lambda signal: False)
    for _ in range(3):
        assert pipeline.enqueue_log("mi.probe.lifecycle", "log_lifecycle",
                                    {"mi.lifecycle.action": "arm"}, None, time.time_ns())
    caplog.set_level(logging.WARNING, logger=_LOST)
    telemetry.shutdown(0.2)
    assert health.SIGNALS["logs"].view(60, 300)["dropped"] == 8
    assert _lost_lines(caplog) == ["telemetry_shutdown_lost signal=logs dropped=3 failed=0"]
