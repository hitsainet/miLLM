"""The `adapter-manifest-sync` CI job's test files run with pytest ALONE (suite tracker 0.9).

That job installs only pytest (the job checks two JSON files against miForge's). The adapter
sync test used to import `millm.services.adapter_manifest`, which drags in the package's whole
runtime (structlog first, then torch and transformers) through the package `__init__`s, so the
job died at collection on every push — red for a reason that said nothing about EITHER contract,
and easy to misread as the deploy-key problem.

This runs the job's exact file list in a subprocess where any import outside the standard library
and pytest's own dependencies raises, i.e. the conditions of a runner with only `pip install
pytest`. The file list is read from the workflow, so a file added to the job is held too.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "backend-tests.yml"

_RUNNER = r"""
import importlib.abc, sys
ALLOWED = set(sys.stdlib_module_names) | {
    "pytest", "_pytest", "pluggy", "iniconfig", "packaging", "py", "exceptiongroup", "tomli",
    "pygments", "colorama", "tests",
}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in ALLOWED:
            raise ImportError(f"BLOCKED: {name} is not available on a pytest-only runner")
        return None
sys.meta_path.insert(0, Block())
import pytest
sys.exit(pytest.main(["--noconftest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", *sys.argv[1:]]))
"""


def _job_files() -> list[str]:
    text = WORKFLOW.read_text()
    job = text[text.index("  adapter-manifest-sync:"):]
    nxt = re.search(r"\n  [a-z][a-z0-9-]*:\n", job[1:])
    job = job[: nxt.start() + 1] if nxt else job
    code = "\n".join(line for line in job.splitlines() if not line.lstrip().startswith("#"))
    files = re.findall(r"tests/unit/[\w/]+\.py", code)
    assert files, "found no test files in the adapter-manifest-sync job"
    return files


def test_the_job_runs_the_two_sync_files():
    assert set(_job_files()) == {
        "tests/unit/api/test_adapter_manifest_schema_sync.py",
        "tests/unit/api/test_checkpoint_manifest_schema_sync.py",
    }


def test_they_pass_with_pytest_alone():
    env = {k: v for k, v in os.environ.items() if k != "MILLM_REQUIRE_CROSS_REPO_CHECKS"}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, *_job_files()],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        "the adapter-manifest-sync job's files do not run on a pytest-only runner:\n"
        + proc.stdout[-3000:] + proc.stderr[-2000:]
    )
    assert "BLOCKED" not in proc.stdout + proc.stderr


def test_the_job_requires_the_check_and_fails_on_a_failed_checkout():
    """The secret exists (2026-10-16), so the job must not skip: the switch is on, the checkout
    uses the deploy key, and a failed checkout exits non-zero rather than warning."""
    yaml = __import__("pytest").importorskip("yaml")
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["adapter-manifest-sync"]
    assert job["env"]["MILLM_REQUIRE_CROSS_REPO_CHECKS"] == "1"
    checkout = next(s for s in job["steps"] if s.get("id") == "miforge")
    assert checkout["with"]["repository"] == "Onegaishimas/miForge"
    assert checkout["with"]["ssh-key"] == "${{ secrets.MIFORGE_DEPLOY_KEY }}"
    run = next(s["run"] for s in job["steps"] if "pytest" in s.get("run", "")
               and "--noconftest" in s.get("run", ""))
    guard = run[: run.index("cd miLLM")]
    assert 'steps.miforge.outcome }}" != "success"' in guard and "exit 1" in guard
    assert "::warning::" not in run
