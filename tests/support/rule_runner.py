"""The REAL `rule-runner` worker as a subprocess, for the S3 tests (039 FR-39.26; TD39-35).

The guest snapshot is the PUBLISHED one (`guest.bundle.snapshot` in the vendored runner contract),
fetched by digest into ``$RULE_RUNNER_BUNDLE_DIR`` (``python -m millm.rules.runner_worker fetch
<dir>``; CI's `rule-runner` job does this). It is never committed. Without it — or without
`wasmtime` — these tests SKIP LOUDLY, and FAIL instead under ``MILLM_REQUIRE_RUNNER=1``.

The worker is started with an environment holding ``PATH``, ``PYTHONPATH`` and ``RULE_RUNNER_*``
only (plus whatever a test adds on purpose): no database URL and no secret, so a worker that
silently needed one would fail to answer. Its socket lives under a SHORT directory (an `AF_UNIX`
path is limited to ~107 bytes).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import pytest

REPO = Path(__file__).resolve().parents[2]
REQUIRED = os.environ.get("MILLM_REQUIRE_RUNNER") == "1"


def _skip(message: str) -> None:
    if REQUIRED:
        pytest.fail("RULE RUNNER: " + message + " and MILLM_REQUIRE_RUNNER=1")
    # `allow_module_level`: `test_runner_wasi_table.py` calls `require_wasmtime()` at import. Without
    # it, a venv lacking wasmtime got a COLLECTION ERROR that aborted the whole unit run instead of
    # this loud skip (found by 038 13.1's full run, 2026-10-24).
    pytest.skip("RULE RUNNER: " + message + " — the runner tests did NOT run",
                allow_module_level=True)


def require_wasmtime() -> None:
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        _skip("wasmtime is not installed")


def bundle_dir() -> Path:
    """The verified bundle directory, or a loud skip."""
    require_wasmtime()
    from millm.rules import runner_contract as rc

    env = os.environ.get("RULE_RUNNER_BUNDLE_DIR")
    if not env:
        _skip("RULE_RUNNER_BUNDLE_DIR is not set; fetch the published snapshot with "
              "`python -m millm.rules.runner_worker fetch <dir>`")
    directory = Path(env)
    snapshot = directory / "snapshot.wasm"
    if not snapshot.is_file():
        _skip(f"no snapshot.wasm in {directory}")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if digest != rc.snapshot()["sha256"]:
        _skip(f"snapshot.wasm digest {digest} is not the contract's {rc.snapshot()['sha256']}")
    return directory


def _connectable(path: Path) -> bool:
    with socket.socket(socket.AF_UNIX) as s:
        try:
            s.connect(str(path))
            return True
        except OSError:
            return False


class RunnerProcess:
    """`with RunnerProcess(bundle) as runner:` — the worker running, `runner.socket` its path."""

    def __init__(self, bundle: Optional[Path], *, extra_env: Optional[dict[str, str]] = None,
                 slots: int = 2) -> None:
        self.bundle = bundle
        self.extra_env = dict(extra_env or {})
        self.slots = slots
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self._dir = Path(tempfile.mkdtemp(prefix="mrr"))
        self.socket = self._dir / "r.sock"

    def env(self) -> dict[str, str]:
        paths = [str(REPO)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": os.pathsep.join(paths),
            "RULE_RUNNER_SOCKET": str(self.socket),
            "RULE_RUNNER_BUNDLE_DIR": str(self.bundle or self._dir / "absent"),
            "RULE_RUNNER_SLOTS": str(self.slots),
            **self.extra_env,
        }

    def start(self) -> "RunnerProcess":
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "millm.rules.runner_worker"], env=self.env(), cwd=str(REPO),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                _out, err = self.proc.communicate()
                raise RuntimeError("rule runner exited early: " + err.decode()[-2000:])
            if self.socket.exists() and _connectable(self.socket):
                return self
            time.sleep(0.05)
        self.stop()
        raise RuntimeError("rule runner did not start listening")

    def stop(self) -> tuple[bytes, bytes]:
        out: tuple[bytes, bytes] = (b"", b"")
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                out = self.proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                out = self.proc.communicate()
        elif self.proc is not None:
            out = self.proc.communicate()
        self.proc = None
        return out

    def __enter__(self) -> "RunnerProcess":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.output = self.stop()
        shutil.rmtree(self._dir, ignore_errors=True)


def floor_limits(**over: int) -> dict[str, int]:
    from millm.rules import runner_contract as rc

    f = rc.floor()
    limits = {"call_fuel": f["call_fuel"], "load_fuel": f["load_fuel"],
              "prelude_fuel": f["prelude_fuel"], "memory_bytes": f["memory_bytes"],
              "input_code_points": f["input_code_points"], "output_bytes": f["output_bytes"],
              "wall_ms": 10_000}
    limits.update(over)
    return limits
