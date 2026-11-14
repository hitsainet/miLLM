"""Review of 040 P0 (2026-11-08): every file a guard reads triggers the CI that runs the guard.

sync-to-clean's `gate` (`.github/scripts/private_ci_gate.py`) publishes a commit that has no run of
a required workflow when nothing under that workflow's push `paths:` filter changed since a passing
ancestor. Backend Tests filtered on `miserver/`, `tests/` and a few files, while P0's guards read the
export, manifests, workflows, Dockerfile, compose files and integrations: a commit renaming
`PGUSER` in `db-backup.yaml` and widening the mirror keep rule (publishing the private export) passed
the gate with two tests red. The covered set is computed with the gate's OWN pathspec function over
the tracked files, so this test and the gate cannot disagree.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / ".github" / "scripts" / "private_ci_gate.py"

#: How each required workflow is shown to cover what its tests read. Backend Tests reads files
#: across the whole tree, so it may carry NO push filter at all (asserted below); Frontend CI's
#: inputs are DERIVED from the admin UI test sources (test_frontend_ci_covers_repo_reads.py).
COVERED_BY = {
    "backend-tests.yml": "test_backend_tests_has_no_push_filter",
    "frontend-ci.yml": "tests/unit/deploy/test_frontend_ci_covers_repo_reads.py",
}


def _gate():
    spec = importlib.util.spec_from_file_location("private_ci_gate_for_cover", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_workflows() -> list[str]:
    job = yaml.safe_load((ROOT / ".github/workflows/sync-to-clean.yml").read_text())["jobs"]["gate"]
    return [w for w in job["env"]["REQUIRED_WORKFLOWS"].split() if w]


def _ls(*specs: str) -> set[str]:
    out = subprocess.run(["git", "ls-files", "--", *specs], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    return set(out)


def _covered(workflow: str) -> set[str] | None:
    gate = _gate()
    specs = gate.push_pathspecs(gate.load_workflow(ROOT, workflow))
    return None if specs is None else _ls(*specs)


def test_every_required_workflow_has_a_coverage_proof():
    assert set(_required_workflows()) == set(COVERED_BY), _required_workflows()
    assert (ROOT / COVERED_BY["frontend-ci.yml"]).is_file()


def test_backend_tests_has_no_push_filter():
    gate = _gate()
    doc = gate.load_workflow(ROOT, "backend-tests.yml")
    push = gate._on(doc).get("push")
    assert push is not None and push.get("branches") == ["main"], push
    assert gate.push_pathspecs(doc) is None, (
        "Backend Tests reads files across the whole tree; a push filter (paths or paths-ignore) lets "
        "the gate publish an untested commit through its ancestor fallback (review of 040 P0)")


def test_the_mirror_leak_check_names_the_full_export():
    steps = yaml.safe_load((ROOT / ".github/workflows/sync-to-clean.yml").read_text())["jobs"]["sync"]["steps"]
    (verify,) = [s for s in steps if s.get("name") == "Verify the published mirror"]
    loop = next(line for line in verify["run"].splitlines() if line.strip().startswith("for p in "))
    assert "docs/legacy-identifiers.md" in loop.removesuffix("; do").split()
