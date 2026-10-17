"""The mirror's image build can only publish a backend image if Backend Tests runs THERE.

hitsainet/millm receives an orphan snapshot (one commit, force-pushed). A `paths:` filter needs a diff
base, so on the mirror backend-tests.yml's push trigger matches nothing, and the image build's gate then
refuses every backend change: it bit on 2026-10-16 (e142853). The fix has two halves and both are
pinned: Backend Tests accepts a dispatch, and the sync dispatches it on the mirror after a branch push.
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
    job = next(iter(steps.values()))
    names = [s.get("name", "") for s in job["steps"]]
    push = names.index("Push to clean repository")
    dispatch = next(i for i, s in enumerate(job["steps"]) if "backend-tests.yml/dispatches" in s.get("run", ""))
    assert dispatch > push, "dispatching before the push would test the previous snapshot"
    step = job["steps"][dispatch]
    assert "${MIRROR}" in step["run"]
    assert step.get("if") == "github.ref_type == 'branch'"
    assert '"204"' in step["run"], "a failed dispatch must fail the sync, not be discovered at the gate"
