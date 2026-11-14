"""Rev2 review of 040 P0 (2026-11-08): the admin UI suite reads repository files OUTSIDE `admin-ui/`,
and every one of them must trigger Frontend CI, or sync-to-clean's gate publishes a commit that
changed them through its ancestor fallback with those tests never run.

`test_ci_covers_guard_inputs.py` lists Frontend CI's inputs by hand (`admin-ui/**` and the legacy
export) and so misses `manual/docs/**` (help-link and feature-name tests), `docs/schemas/probe-result-v1.json`
and `tests/fixtures/contracts/**` (the probe-result fixture test). Here the inputs are DERIVED from
the test sources: every quoted string in an admin-UI test file that names a tracked path outside
`admin-ui/` (a file, or a directory with tracked files) must be covered by Frontend CI's own push
filter, computed with the gate's `push_pathspecs`.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / ".github" / "scripts" / "private_ci_gate.py"
QUOTED = re.compile(r"""(['"`])([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+)\1""")


def _ls(*specs: str) -> set[str]:
    return set(subprocess.run(["git", "ls-files", "--", *specs], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.split())


def _gate():
    spec = importlib.util.spec_from_file_location("private_ci_gate_for_frontend_reads", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo_reads() -> dict[str, set[str]]:
    """{quoted path: tracked files it names} for every admin-UI test file."""
    tracked = _ls(".")
    tests = sorted(f for f in tracked if f.startswith("admin-ui/")
                   and re.search(r"(\.test\.[cm]?[jt]sx?|/tests/setup\.ts)$", f))
    assert tests, "no admin UI test file found: the listing is failing open"
    out: dict[str, set[str]] = {}
    for rel in tests:
        for _q, lit in QUOTED.findall((ROOT / rel).read_text(encoding="utf-8")):
            lit = lit.strip("/")
            if lit.startswith("admin-ui/"):
                continue
            named = {f for f in tracked if f == lit or f.startswith(lit + "/")}
            if named and not all(f.startswith("admin-ui/") for f in named):
                out.setdefault(lit, set()).update(f for f in named if not f.startswith("admin-ui/"))
    return out


def test_every_repo_file_an_admin_ui_test_reads_triggers_frontend_ci():
    reads = repo_reads()
    # Anchors: two reads known today, so a scan that finds nothing cannot pass.
    assert "manual/docs" in reads and "docs/schemas/probe-result-v1.json" in reads, sorted(reads)
    gate = _gate()
    specs = gate.push_pathspecs(gate.load_workflow(ROOT, "frontend-ci.yml"))
    if specs is None:
        return  # no filter: every push runs it
    covered = _ls(*specs)
    missing = {lit: sorted(files - covered)[:5] for lit, files in reads.items() if files - covered}
    assert not missing, f"Frontend CI's push filter misses files its tests read: {missing}"
