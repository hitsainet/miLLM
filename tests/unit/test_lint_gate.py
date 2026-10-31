"""The lint gate (026 F-6, suite tracker 4.16) is wired into CI and green.

The gate runs in the unit job of backend-tests.yml: ruff's bug-class rules (syntax, undefined
names, unused variables, redefinitions) over millm/, and mypy --strict on the packages written
green. This file pins the CI step (its removal goes red) and runs the same commands here, so an
undefined name fails locally before it fails in CI. Skips loudly where a tool is not installed.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "backend-tests.yml"


def _gate_step() -> dict:
    yaml = pytest.importorskip("yaml")
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["test"]["steps"]
    gate = [s for s in steps if "ruff check" in s.get("run", "")]
    assert len(gate) == 1, "the unit job has no lint gate step"
    names = [s.get("name", "") for s in steps]
    assert names.index(gate[0]["name"]) < next(
        i for i, s in enumerate(steps) if "pytest tests/unit/" in s.get("run", "")
    ), "the gate must run before the unit tests"
    return gate[0]


def _has(module: str) -> bool:
    return subprocess.run([sys.executable, "-m", module, "--version"],
                          capture_output=True).returncode == 0


def test_ci_runs_both_tools_over_their_scopes():
    step = _gate_step()
    assert "python -m ruff check millm --select E9,F --ignore F401" in step["run"]
    assert "python -m mypy --follow-imports=silent $MYPY_GATED" in step["run"]
    assert "millm/services/batch" in step["env"]["MYPY_GATED"].split()


def test_ruff_bug_rules_are_clean():
    if not _has("ruff"):
        pytest.skip("ruff not installed: the lint gate did NOT run here")
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "millm", "--select", "E9,F", "--ignore", "F401"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout[-3000:]


def test_mypy_is_clean_on_the_gated_scope():
    if not _has("mypy"):
        pytest.skip("mypy not installed: the type gate did NOT run here")
    scope = shlex.split(_gate_step()["env"]["MYPY_GATED"])
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--follow-imports=silent", *scope],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:]
