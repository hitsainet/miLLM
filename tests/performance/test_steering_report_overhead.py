"""Feature 28 FTASKS 8.4: what the steering report adds to the path it changed.

200 serial chat completions on a tiny real model with a real hooked SAE and an inline set, timed
with the report and with it removed (finish = bare restore, no describe). The p95 delta must stay
under 5 ms (FTDD §9). It benchmarks the changed path: capture + describe + publish around a real
generation, not the serialiser in isolation.

Not in `tests/unit` (CI does not time things: a wall-clock bound on a shared runner is a flake).
Run: `pytest tests/performance/test_steering_report_overhead.py -s`.
"""

from __future__ import annotations

import statistics
import time

import pytest

from tests.unit.f28_fixtures import SAE_A, build, chat, clean_state, db, no_cuda  # noqa: F401

N = 200


async def _timed(svc, request) -> list[float]:
    out = []
    with no_cuda():
        for _ in range(N):
            start = time.perf_counter()
            await svc.create_chat_completion(request)
            out.append(time.perf_counter() - start)
    return out


def _p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=100)[94]


@pytest.mark.slow
@pytest.mark.parametrize("case", ["inline-record-match", "live-manual-two-db-reads"])
async def test_report_overhead_p95_under_5ms(case, clean_state, db, monkeypatch):  # noqa: F811
    """`inline-record-match` labels from the request record (no read); `live-manual-two-db-reads`
    is the worst case: nothing claims the entry, so the reader reads the circuits and the active
    profile (SQLite here, so a lower bound on a networked Postgres)."""
    served = build([(SAE_A, 0, 11)])
    svc = served.svc
    if case == "inline-record-match":
        request = chat(max_tokens=2, steering={"features": [{"index": 1, "strength": 4.0}]})
    else:
        served.sae(SAE_A, 0).set_steering_batch({1: 4.0})
        served.sae(SAE_A, 0).enable_steering(True)
        request = chat(max_tokens=2)
    try:
        await _timed(svc, request)  # warm-up
        with_report = await _timed(svc, request)

        async def no_describe(self, snapshot, **kw):
            return None

        def bare_finish(saved):
            svc._restore_request_profile(saved)

        monkeypatch.setattr(type(svc), "_describe_steering", no_describe)
        monkeypatch.setattr(svc, "_finish_request_steering", bare_finish)
        without = await _timed(svc, request)
    finally:
        for h in served.handles:
            h.remove()
    delta = _p95(with_report) - _p95(without)
    print(f"\n{case}: steering report overhead: p95 with={_p95(with_report) * 1e3:.3f} ms "
          f"without={_p95(without) * 1e3:.3f} ms delta={delta * 1e3:.3f} ms "
          f"(median delta {(statistics.median(with_report) - statistics.median(without)) * 1e3:.3f} ms)")
    assert delta < 0.005
