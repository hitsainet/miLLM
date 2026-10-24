"""Backend Tests still runs on the mirror, after the push, and nothing waits on it.

hitsainet/millm receives an orphan snapshot (one commit, force-pushed). A `paths:` filter needs a diff
base, so on the mirror backend-tests.yml's push trigger matches nothing; the sync dispatches it instead.
Until 2026-10-20 the mirror's image build waited on that run (it refused every backend change while the
dispatch was missing, 2026-10-16, e142853). Since then the image build trusts the PRIVATE repo's CI
(test_private_ci_trust.py), so the mirror run is informational - the one check that the stripped
snapshot is self-consistent - and a failed dispatch warns instead of failing the sync.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _load(name: str) -> dict:
    doc = yaml.safe_load((WORKFLOWS / name).read_text())
    doc["on"] = doc.pop(True, doc.get("on"))  # PyYAML reads the bare key `on` as boolean True
    return doc


def test_backend_tests_can_be_dispatched():
    assert "workflow_dispatch" in _load("backend-tests.yml")["on"]


def test_the_sync_dispatches_backend_tests_on_the_mirror_after_pushing():
    steps = _load("sync-to-clean.yml")["jobs"]
    job = steps["sync"]
    names = [s.get("name", "") for s in job["steps"]]
    push = names.index("Push to clean repository")
    dispatch = next(i for i, s in enumerate(job["steps"]) if "backend-tests.yml/dispatches" in s.get("run", ""))
    assert dispatch > push, "dispatching before the push would test the previous snapshot"
    step = job["steps"][dispatch]
    assert "${MIRROR}" in step["run"]
    assert "github.ref_type == 'branch'" in step.get("if", "")


def test_a_failed_dispatch_warns_and_does_not_fail_the_sync(tmp_path):
    """Nothing waits on the mirror's run any more, so a refused dispatch must not turn a published,
    verified snapshot into a red sync. RUN the step against a stub curl that answers 404."""
    import os
    import subprocess

    job = _load("sync-to-clean.yml")["jobs"]["sync"]
    step = next(s for s in job["steps"] if "backend-tests.yml/dispatches" in s.get("run", ""))
    stub = tmp_path / "curl"
    stub.write_text("#!/bin/sh\nprintf 404\n")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "MIRROR": "x/y", "SYNC_TOKEN": "t"}
    res = subprocess.run(["bash", "-e", "-c", step["run"]], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "::warning::" in res.stdout
