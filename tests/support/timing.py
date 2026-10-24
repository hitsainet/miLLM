"""Same-run timing comparisons for cost tests.

A bare wall-clock ceiling asserts something about the machine, not about the code:
`test_a_realistic_long_prefill_stays_within_the_per_layer_budget` measured 160 ms against its 60 ms
budget on a correct build at load average 19 (suite memory `check-ci-not-just-local-green`). So a
cost test here times the code under test AND a reference workload in the same run, interleaved, and
asserts the RATIO. Contention slows both and cancels; a regression slows only the subject.

Three choices make the ratio hold under load (measured 2026-10-20, see
tests/unit/ml/test_edge_sensing_baseline.py):

- **The calling thread's CPU time (`time.thread_time`), not wall time.** Under oversubscription a
  long subject is descheduled mid-run far more often than a short reference, so wall-time ratios
  inflated with load (4096-token pass/encode 17.6x-42.1x with a busy loop on every core). Thread CPU
  time does not count time spent descheduled: 16.7x-19.3x under the same burner at load average 31.
  The price, stated: a regression that WAITS (a sleep, a lock) rather than computes is invisible
  here. These tests guard computational cost.
- **One torch intra-op thread** (`single_torch_thread`), so the work happens on the thread being
  timed, and because the thread pool is pure overhead and noise on shapes this small.
- **Minimum of interleaved rounds.** Load only ever adds time, so the minimum is the best estimate of
  each side's cost, and interleaving puts both sides under the same conditions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def single_torch_thread() -> Iterator[None]:
    """Run the block on one torch intra-op thread, restoring the previous count."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _ms(fn: Callable[[], object]) -> float:
    started = time.thread_time()
    fn()
    return (time.thread_time() - started) * 1000.0


def paired_min_ms(
    subject: Callable[[], object],
    reference: Callable[[], object],
    rounds: int = 7,
    setup: Callable[[], tuple[Callable[[], object], Callable[[], object]]] | None = None,
) -> tuple[float, float]:
    """Best-of-``rounds`` milliseconds for ``subject`` and ``reference``, measured alternately.

    With ``setup``, each round calls it for a fresh ``(subject, reference)`` pair (for code under
    test that mutates its own state) and the positional callables are used only to warm up.
    """
    subject()
    reference()
    best_s = best_r = float("inf")
    for _ in range(rounds):
        s, r = setup() if setup else (subject, reference)
        best_s = min(best_s, _ms(s))
        best_r = min(best_r, _ms(r))
    return best_s, best_r
