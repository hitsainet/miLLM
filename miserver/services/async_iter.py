"""Iterate a BLOCKING iterator from async code without pinning the event loop.

`TextIteratorStreamer.__next__` blocks on a queue until the generation thread
produces the next token. Iterating it directly inside an `async def` generator
holds the event loop for the entire generation, which has two consequences:

  1. Nothing already yielded reaches the socket. Measured on granite-4.2-8b
     (2026-09-02, before this existed): every SSE chunk arrived at 6.12s with a
     spread of 0.00s -- including the `role` chunk that is yielded BEFORE
     generation starts. The client waited out the whole completion and then got
     it in one burst, which is indistinguishable from a hung server.
  2. Every other task on the loop starves for the duration -- other requests,
     health checks, the monitoring hooks.

Awaiting each step in a worker thread hands control back between items.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterable

_END = object()


async def aiter_blocking(iterable: Iterable[Any]) -> AsyncIterator[Any]:
    """Async-iterate `iterable`, awaiting each blocking `next()` off-thread."""
    loop = asyncio.get_running_loop()
    it = iter(iterable)
    while True:
        # `next(it, _END)` rather than catching StopIteration: a StopIteration
        # raised inside an executor does not propagate cleanly across the
        # future boundary.
        item = await loop.run_in_executor(None, lambda: next(it, _END))
        if item is _END:
            return
        yield item
