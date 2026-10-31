"""`tests/support/rule_runner.require_wasmtime` — the loud skip, its module-level form, and that it
does NOT skip when wasmtime is importable (review R1 of 038 13.1, 2026-10-25).

`test_runner_wasi_table.py` calls `require_wasmtime()` at import. Commit 10468f2b added
`allow_module_level=True` because without it a venv lacking wasmtime got a collection error that
aborted the whole unit run. What must hold, each pinned here by behaviour, not by source:

* without wasmtime, a module calling it at import is SKIPPED (not a collection error);
* with wasmtime importable, it returns — so the runner tests RUN (a helper that always skipped
  would make every runner test a silent pass outside the `rule-runner` CI job);
* under `MILLM_REQUIRE_RUNNER=1` a missing wasmtime FAILS instead of skipping.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tests.support import rule_runner as rr

REPO = Path(__file__).resolve().parents[3]


def test_with_wasmtime_importable_the_helper_does_not_skip(monkeypatch):
    monkeypatch.setitem(sys.modules, "wasmtime", types.ModuleType("wasmtime"))
    monkeypatch.setattr(rr, "REQUIRED", False)
    # A Skipped escaping here would turn THIS test into a skip (a silent pass): catch it.
    try:
        rr.require_wasmtime()
    except pytest.skip.Exception as exc:
        pytest.fail(f"require_wasmtime() skipped with wasmtime importable: {exc}")


def test_without_wasmtime_the_helper_skips_and_fails_when_required(monkeypatch):
    monkeypatch.setitem(sys.modules, "wasmtime", None)  # `import wasmtime` raises ImportError
    monkeypatch.setattr(rr, "REQUIRED", False)
    with pytest.raises(pytest.skip.Exception, match="wasmtime is not installed") as skipped:
        rr.require_wasmtime()
    assert skipped.value.allow_module_level is True
    monkeypatch.setattr(rr, "REQUIRED", True)
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as exc:
        rr.require_wasmtime()
    assert exc.type is pytest.fail.Exception, "skipped under MILLM_REQUIRE_RUNNER=1"
    assert "MILLM_REQUIRE_RUNNER=1" in str(exc.value)


def test_a_module_calling_it_at_import_is_skipped_not_a_collection_error(tmp_path):
    (tmp_path / "conftest.py").write_text(
        "import sys\nsys.modules['wasmtime'] = None  # no wasmtime in this run\n")
    (tmp_path / "test_needs_wasmtime.py").write_text(
        "from tests.support import rule_runner as rr\n"
        "rr.require_wasmtime()\n\n"
        "def test_runs():\n    pass\n")
    env = {k: v for k, v in os.environ.items() if k != "MILLM_REQUIRE_RUNNER"}
    env["PYTHONPATH"] = str(REPO)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-o", "addopts=",
                        "-p", "no:cacheprovider", "--rootdir", str(tmp_path), str(tmp_path)],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    # 5 = "no tests collected" (the module skipped whole); a collection error would be 2.
    assert r.returncode == 5, r.stdout + r.stderr
    assert "1 skipped" in r.stdout and "error" not in r.stdout.lower(), r.stdout
    assert "RULE RUNNER: wasmtime is not installed" in r.stdout
