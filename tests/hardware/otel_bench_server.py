#!/usr/bin/env python3
"""Feature 038 benchmark launcher (FTASKS 13.1) — runs miLLM with a BENCH-ONLY arm switch.

⚠ TEST CODE FOR A THROWAWAY POD. NEVER PART OF THE IMAGE, NEVER THE LIVE miLLM.
The image copies `millm/` only (`Dockerfile`); this file lives under `tests/hardware/` and runs
from the `git archive` of the branch under test on the pod's PYTHONPATH, exactly as the Feature
31–33 and 37 harnesses do (`millm_hwtest_pod.yaml`). This file adds no route to the product:
`tests/unit/hardware/test_otel_bench_server.py` asserts that `millm.main`'s app has no
`/__otel_bench` path and that nothing under `millm/` imports this file.

ENH-006 D75 (2026-10-28) since made the product's telemetry settings changeable while it runs:
`PUT /api/environment[/{key}]` saves an override and rebuilds the pipeline the same way this file
does (drain, then `telemetry.configure()` on the settings in force; an agent's save waits for the
operator, D76). This launcher stays for the benchmark because it times the enqueue, switches
WITHOUT writing overrides to the pod's database (a benchmark must leave no saved setting behind),
and offers only the A/B presets below. 14.1 (switch-on) no longer needs this file or a restart:
the operator sets the endpoint and the switches on the Settings page.

WHY IT EXISTS (the design of the telemetry arms, recorded in the 038 controls record §13.1).
FR-38.29 / D38.52 interleave telemetry OFF and ON in blocks of 10 (A-B-B-A) inside one cell.
When this was written every `TELEMETRY_*` setting was environment-only (OT-D5, since reversed by
ENH-006 D75), so the product could switch export only by a restart, and a restart drops the resident model (a reload per block, 13 per cell, each one a
gated load on the live server and minutes of GPU time on a test pod). Two pods side by side do not
fit either: two copies of an 8B model in 16-bit are 32 GB on a 24 GB card. So the arm is switched
IN THE PROCESS, by calling the product's own `telemetry.configure()` — the function `lifespan`
calls at start — with a copy of the process settings whose switches differ. The model stays
resident; a switch costs one pipeline drain (`TELEMETRY_DRAIN_DEADLINE_S`).

What a preset may change, and nothing else: the three switches, the trace sampler ratio (AT08)
and the stateless permission (AT07). The endpoint, headers, TLS files and every other value are
the POD'S ENVIRONMENT, as the operator set it; no request can name a destination. The route
answers only loopback clients (the harness runs inside the pod over `kubectl exec`; the pod has no
Service).

What "off" means here: `configure()` with every switch off shuts the pipeline down and leaves
`_PIPELINE = None`, so every entry point returns at its first line and no SDK provider exists —
the product's off path. The SDK MODULES stay imported after the first "on" block; they cost
nothing per request. "Off means free" across processes (no SDK import at all) is the separate
pre-038 pair (13.4), run against two servers.

The export enqueue (target: p95 ≤ 2 ms, FR-38.29) is timed here, around the product's own
methods, because the product does not time itself: `Pipeline.enqueue_log` (allowlist filter +
`put_nowait`, per result or lifecycle record), `_QueueingProcessor.on_end` (per span),
`Pipeline.enqueue_measurements` (per batch of histogram points) and `telemetry.emit_results`
(per call, informational). The harness refuses a measured "on" block whose timer saw nothing, so a
patch that silently stopped applying cannot report a fast enqueue.

Run (inside the throwaway pod, instead of `uvicorn millm.main:app`):
    python tests/hardware/otel_bench_server.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import collections
import threading
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request

#: The arm presets: the only values a request may choose between (module doc).
SWITCHES = ("TELEMETRY_TRACES_ENABLED", "TELEMETRY_METRICS_ENABLED", "TELEMETRY_LOGS_ENABLED")
PRESETS: dict[str, dict[str, Any]] = {
    "off": dict.fromkeys(SWITCHES, False),
    "on": dict.fromkeys(SWITCHES, True),
    "on_sampler0": {**dict.fromkeys(SWITCHES, True), "TELEMETRY_TRACES_SAMPLER_RATIO": 0.0},
    "on_stateless": {**dict.fromkeys(SWITCHES, True), "TELEMETRY_STATELESS_EXPORT_PERMITTED": True},
}
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
ROUTE_PREFIX = "/__otel_bench"
TIMED_KINDS = ("enqueue_log", "span_on_end", "enqueue_measurements", "emit_results")


def preset_settings(base: Any, preset: str) -> Any:
    """A copy of `base` (the process settings) with ONLY the preset's keys changed."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {', '.join(PRESETS)}")
    return base.model_copy(update=dict(PRESETS[preset]))


class EnqueueTimer:
    """Wall time of the product's enqueue calls, per kind, in nanoseconds (module doc)."""

    def __init__(self, maxlen: int = 500_000) -> None:
        self._lock = threading.Lock()
        self.samples: dict[str, collections.deque[int]] = {
            k: collections.deque(maxlen=maxlen) for k in TIMED_KINDS}
        self.calls: dict[str, int] = dict.fromkeys(TIMED_KINDS, 0)
        self.installed = False

    def record(self, kind: str, ns: int) -> None:
        with self._lock:
            self.samples[kind].append(ns)
            self.calls[kind] += 1

    def wrap(self, kind: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        if getattr(fn, "__otel_bench_kind__", None) == kind:
            return fn  # already wrapped (idempotent install)

        def timed(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                self.record(kind, time.perf_counter_ns() - t0)

        timed.__otel_bench_kind__ = kind  # type: ignore[attr-defined]
        timed.__wrapped__ = fn  # type: ignore[attr-defined]
        return timed

    def install(self) -> None:
        """Patch the CLASSES (an instance built before or after still resolves through them) and
        the `millm.telemetry` module attribute its callers read (`telemetry.emit_results(...)`)."""
        from millm import telemetry
        from millm.telemetry import pipeline

        pipeline.Pipeline.enqueue_log = self.wrap(  # type: ignore[method-assign]
            "enqueue_log", pipeline.Pipeline.enqueue_log)
        pipeline.Pipeline.enqueue_measurements = self.wrap(  # type: ignore[method-assign]
            "enqueue_measurements", pipeline.Pipeline.enqueue_measurements)
        pipeline._QueueingProcessor.on_end = self.wrap(  # type: ignore[method-assign]
            "span_on_end", pipeline._QueueingProcessor.on_end)
        telemetry.emit_results = self.wrap("emit_results", telemetry.emit_results)
        self.installed = True

    def uninstall(self) -> None:
        """Put the product's own functions back (tests share one process)."""
        from millm import telemetry
        from millm.telemetry import pipeline

        for owner, name in ((pipeline.Pipeline, "enqueue_log"),
                            (pipeline.Pipeline, "enqueue_measurements"),
                            (pipeline._QueueingProcessor, "on_end"),
                            (telemetry, "emit_results")):
            fn = getattr(owner, name)
            if getattr(fn, "__otel_bench_kind__", None):
                setattr(owner, name, fn.__wrapped__)
        self.installed = False

    def reset(self) -> None:
        with self._lock:
            for k in TIMED_KINDS:
                self.samples[k].clear()
                self.calls[k] = 0

    def snapshot(self, with_samples: bool = True) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {"installed": self.installed, "calls": dict(self.calls)}
            if with_samples:
                out["samples_us"] = {k: [ns / 1000.0 for ns in v] for k, v in self.samples.items()}
            return out


def rss_kib() -> int | None:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def bench_state(timer: EnqueueTimer, preset: str | None, with_samples: bool) -> dict[str, Any]:
    from millm import telemetry
    from millm.telemetry import counters, health

    return {
        "preset": preset,
        "pipeline_running": telemetry.any_signal_on(),
        "telemetry": health.snapshot(),
        "counter_store": {
            "series": {i: counters.STORE.series_count(i) for i in counters.INSTRUMENTS},
            "series_cap": counters.STORE.series_cap,
            "series_overflow": counters.STORE.series_overflow,
        },
        "rss_kib": rss_kib(),
        "enqueue": timer.snapshot(with_samples),
    }


def build_router(timer: EnqueueTimer, allowed_hosts: frozenset[str] = LOOPBACK,
                 settings_source: Callable[[], Any] | None = None) -> Any:
    """The bench routes. `settings_source` returns the process settings (tests pass their own).

    ⚠ `APIRouter`, `HTTPException` and `Request` are imported at MODULE level: this file uses
    postponed annotations, and FastAPI resolves an endpoint's annotations in the module's globals —
    a `Request` imported inside this function would be read as a query parameter (422).
    """
    from millm import telemetry

    router = APIRouter(prefix=ROUTE_PREFIX)
    current: dict[str, str | None] = {"preset": None}

    def _base() -> Any:
        if settings_source is not None:
            return settings_source()
        from millm.core.config import settings

        return settings

    def _guard(request: Request) -> None:
        host = request.client.host if request.client else None
        if host not in allowed_hosts:
            raise HTTPException(status_code=403, detail="bench routes answer loopback clients only")

    @router.post("/arm")
    def arm(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        _guard(request)
        preset = str(body.get("preset", ""))
        if preset not in PRESETS:
            raise HTTPException(status_code=400, detail=f"unknown preset; known: {sorted(PRESETS)}")
        if preset != "off" and not timer.installed:
            timer.install()
        telemetry.configure(preset_settings(_base(), preset))
        current["preset"] = preset
        timer.reset()
        return bench_state(timer, preset, with_samples=False)

    @router.get("/state")
    def state(request: Request, samples: int = 1) -> dict[str, Any]:
        _guard(request)
        return bench_state(timer, current["preset"], with_samples=bool(samples))

    @router.post("/reset")
    def reset(request: Request) -> dict[str, Any]:
        _guard(request)
        timer.reset()
        return {"reset": True}

    return router


def attach(fastapi_app: Any, timer: EnqueueTimer | None = None) -> EnqueueTimer:
    timer = timer or EnqueueTimer()
    fastapi_app.include_router(build_router(timer))
    return timer


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - runs the server
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    import uvicorn

    from millm import main as millm_main

    # `millm.main.app` is the Socket.IO wrapper; the FastAPI app is its `other_asgi_app`.
    attach(millm_main.app.other_asgi_app)
    uvicorn.run(millm_main.app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
