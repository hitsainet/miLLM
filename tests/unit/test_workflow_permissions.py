"""Every workflow states least-privilege GITHUB_TOKEN permissions (CodeQL alert 7, 2026-10-17).

Mirror alert 7 (actions/missing-workflow-permissions) flagged `docker-images.yml`'s `detect` job:
no `permissions:` at workflow or job level, so it ran with the repository's default token, which
may be read-write. These workflows also run on the public mirror (it is synced from here), where
the default is not ours to choose. Each workflow now declares a top-level block; jobs that need
more grant it themselves. Never `write-all`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = sorted((Path(__file__).resolve().parents[2] / ".github" / "workflows").glob("*.yml"))


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_job_runs_with_declared_permissions(workflow):
    doc = yaml.safe_load(workflow.read_text())
    top = doc.get("permissions")
    for name, job in (doc.get("jobs") or {}).items():
        declared = job.get("permissions", top)
        assert declared is not None, f"{workflow.name}:{name} runs with the default token"
        assert declared not in ("write-all", "read-all"), f"{workflow.name}:{name}: {declared}"


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_the_workflow_default_is_read_only(workflow):
    top = yaml.safe_load(workflow.read_text()).get("permissions")
    assert isinstance(top, dict), f"{workflow.name} has no top-level permissions block"
    assert all(v in ("read", "none") for v in top.values()), f"{workflow.name}: {top}"
