"""The vitest exercises of the legacy-identifier export must PASS in Frontend CI's own run (rev2 review
of 040 P0). `.github/scripts/check_vitest_exercises.py` reads Vitest's JSON report; this pins its
rules and that Frontend CI writes the report and runs the check."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github" / "scripts" / "check_vitest_exercises.py"


def _script():
    spec = importlib.util.spec_from_file_location("check_vitest_exercises", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(status: str | None, file: str, title: str) -> dict:
    results = [] if status is None else [{"title": title, "status": status}]
    return {"testResults": [{"name": f"/w/{file}", "assertionResults": results}]}


def test_the_exports_vitest_exercises_are_found():
    wanted = _script().exercises(ROOT / "docs")
    assert ("DR-25", "admin-ui/src/__tests__/legacy/frozenStorageKeys.test.ts",
            "keeps the frozen browser storage keys") in wanted


def test_only_a_passed_exercise_passes():
    s = _script()
    wanted = s.exercises(ROOT / "docs")
    _entry, file, title = wanted[0]
    assert s.problems(_report("passed", file, title), wanted[:1]) == []
    for status in ("skipped", "pending", "todo", "failed", None):
        assert s.problems(_report(status, file, title), wanted[:1]), status
    assert s.problems(_report("passed", "admin-ui/src/other.test.ts", title), wanted[:1])


def test_frontend_ci_writes_the_report_and_runs_the_check():
    steps = yaml.safe_load((ROOT / ".github/workflows/frontend-ci.yml").read_text())["jobs"]["ci"]["steps"]
    runs = [s.get("run", "") for s in steps]
    test_at = next(i for i, r in enumerate(runs) if "vitest run" in r)
    assert "--reporter=json" in runs[test_at] and "--outputFile.json=vitest-report.json" in runs[test_at]
    check_at = next(i for i, r in enumerate(runs) if "check_vitest_exercises.py" in r)
    assert check_at > test_at
    assert runs[check_at].split()[-2:] == ["admin-ui/vitest-report.json", "docs"]


def test_the_check_runs_on_a_real_report(tmp_path):
    s = _script()
    wanted = s.exercises(ROOT / "docs")
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"testResults": [{"name": "/x/" + f, "assertionResults": [
        {"title": t, "status": "passed"}]} for _e, f, t in wanted]}))
    assert s.main(["x", str(report), str(ROOT / "docs")]) == 0
