"""The REAL rule-runner worker stops on SIGTERM (039 19.7 finding, gpu_window3_2026-10-25).

In the sidecar the worker is PID 1, and the kernel delivers to PID 1 only the signals it HANDLES:
with the default disposition, kubelet's SIGTERM was dropped and every rollout waited the full grace
period for the force-kill (`kill -INT 1` was the only signal that stopped it). A PID namespace
cannot be made here (unprivileged user namespaces are restricted on the hosts and runners), so the
PID-1 fact is asserted directly: the live process CATCHES SIGTERM (`SigCgt` in /proc), and on
SIGTERM it exits 0 within a few seconds — not killed by the signal (`-15`), which is how a
non-PID-1 process with no handler would "pass".

Negative control (controls record § SIGTERM): remove `install_sigterm_handler()` from `serve` → red.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path

from tests.support import rule_runner as rr

EXIT_WITHIN_S = 5.0


def _caught_signals(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("SigCgt:"):
            return int(line.split()[1], 16)
    raise AssertionError("no SigCgt line")


def test_sigterm_stops_the_real_worker_cleanly_and_quickly():
    bundle = rr.bundle_dir()
    with rr.RunnerProcess(bundle) as runner:
        proc = runner.proc
        assert proc is not None
        # What PID 1 needs: a handler, so the kernel does not drop the signal.
        assert _caught_signals(proc.pid) & (1 << (signal.SIGTERM - 1)), "SIGTERM is not handled"
        sent = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        code = proc.wait(timeout=EXIT_WITHIN_S + 5)
        took = time.monotonic() - sent
    assert code == 0, f"exit {code}: killed by the signal instead of handling it"
    assert took < EXIT_WITHIN_S, took
    _out, err = runner.output
    log = err.decode(errors="replace")
    # The seccomp filter was still installed first (D55), and the stop was the handled one.
    assert "rule_runner_seccomp_installed" in log, log[-2000:]
    assert "rule_runner_stopped signal=SIGTERM" in log, log[-2000:]
