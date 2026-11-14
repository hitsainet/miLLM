"""040 TPD-17 benchmark: the cost of `LegacyHostMiddleware` per request (recorded, never asserted).

Times 100,000 ASGI calls of a no-op app, bare and wrapped, with a new and an old `Host`, three
repetitions each; prints the medians in microseconds per call. Run:
    venv/bin/python tests/hardware/legacy_host_overhead.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

from miserver.api.legacy_host import LegacyHostMiddleware
from miserver.core import legacy_names

N = 100_000


async def _noop(scope, receive, send):
    return None


async def _time(app, host: bytes) -> float:
    scope = {"type": "http", "path": "/v1/chat/completions",
             "headers": [(b"host", host), (b"content-type", b"application/json"),
                         (b"accept", b"*/*"), (b"user-agent", b"bench")]}
    start = time.perf_counter()
    for _ in range(N):
        await app(scope, None, None)
    return (time.perf_counter() - start) / N * 1e6


async def main() -> None:
    wrapped = LegacyHostMiddleware(_noop)
    rows = {}
    for label, app, host in (("bare, new host", _noop, b"k8s-miserver.hitsai.local"),
                             ("wrapped, new host", wrapped, b"k8s-miserver.hitsai.local"),
                             ("wrapped, old host", wrapped, b"k8s-millm.hitsai.local")):
        rows[label] = statistics.median([await _time(app, host) for _ in range(3)])
        legacy_names.drain_hits()
    for label, us in rows.items():
        print(f"{label:20s} {us:7.3f} us/call")


if __name__ == "__main__":
    asyncio.run(main())
