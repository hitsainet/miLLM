"""T03: every register entry this repo carries is present and its exercise RUNS green (040 FR-040.1 to
FR-040.7; FTDD §6.6; TID §3.6; CF-1).

The suite register (`deferred-renames.md`) lists the names the rebrand must NOT change; the census
export (`docs/legacy-identifiers.md`) carries the entries this repo holds, each with the test that
proves the name still works (`exercise`). For every entry:

* `test_no_entry_is_pending` — a bootstrap record (`pending`, written by `--allow-pending`) is red
  until its exercise exists and its id is copied into census (CF-1);
* `test_entry_is_present` — a `frozen`/`scheduled` entry's pattern still matches the tree and each
  literal is still there verbatim; a `retired` one matches nothing; `history_only` skips by name;
* `test_exercise_is_real` — the named test is collected and PASSES. Every pytest exercise runs once,
  in one subprocess (ID-2); a missing, failing, erroring or SKIPPED exercise names the entry (a
  skipped exercise must not read green). A `vitest:` exercise is checked textually (the file exists
  and holds the named `it(...)`), and runs in the admin UI suite.

On the public mirror only `.public.md` exists; a withheld id is skipped by name (CF-2).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from tests.support import legacy_identifiers as li

ROOT = Path(__file__).resolve().parents[3]
EXCLUDE = (":!0xcc", ":(exclude,glob)**/docs/legacy-identifiers*.md")

EXPORT = li.load()
ENTRIES = {r["id"]: r for r in EXPORT.entries()}
IDS = sorted(set(ENTRIES) | {w for w in EXPORT.withheld() if w.startswith("DR-")})


def _entry(entry_id: str) -> dict[str, Any]:
    li.skip_if_withheld(EXPORT, entry_id)
    return ENTRIES[entry_id]


# ── the three checks, as functions the self-tests can call ───────────────────────────


def pending_problem(record: dict[str, Any]) -> str | None:
    if "pending" in record["flags"]:
        return (f"{record['id']} is pending: write its exercise and copy the id into census.py "
                f"EXERCISE, then regenerate the export without --allow-pending (CF-1)")
    return None


def _git(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None


def _walk_texts():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if (not path.is_file() or rel.startswith(("0xcc/", ".git/")) or "node_modules" in rel
                or re.search(r"(^|/)docs/legacy-identifiers[^/]*\.md$", rel)):
            continue
        try:
            yield path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def tree_matches(pattern: str, fixed: bool = False) -> bool:
    """Does any tracked file (outside history and the export itself) match?"""
    run = _git("grep", "-q", "-I", "-F" if fixed else "-P", "-e", pattern, "--", ".", *EXCLUDE)
    if run is not None and run.returncode in (0, 1):
        return run.returncode == 0
    rx = re.compile(re.escape(pattern) if fixed else pattern)
    return any(rx.search(text) for text in _walk_texts())


def presence_problem(record: dict[str, Any]) -> str | None:
    if record["status"] == "retired":
        return (f"{record['id']} is retired but its pattern still matches the tree"
                if tree_matches(record["pattern"]) else None)
    if not tree_matches(record["pattern"]):
        return f"{record['id']}: pattern {record['pattern']!r} matches nothing in the tree"
    missing = [lit for lit in record["literals"] if not tree_matches(lit, fixed=True)]
    if missing:
        return f"{record['id']}: frozen literals no longer in the tree: {missing}"
    return None


def _braces(glob: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", glob)
    if not m:
        return [glob]
    return [x for alt in m.group(1).split(",") for x in _braces(glob[:m.start()] + alt + glob[m.end():])]


def _config_globs(text: str, key: str) -> list[str]:
    m = re.search(rf"\b{key}\s*:\s*\[([^\]]*)\]", text)
    return re.findall(r"""['"]([^'"]+)['"]""", m.group(1)) if m else []


def vitest_collects(admin_ui: Path, file: Path) -> bool:
    """Does `admin-ui/vitest.config.ts`'s `test.include` match the file, and no `test.exclude`?"""
    import fnmatch

    text = (admin_ui / "vitest.config.ts").read_text()
    test_block = text[text.index("test:"):]
    include, exclude = _config_globs(test_block, "include"), _config_globs(test_block, "exclude")
    assert include, "admin-ui/vitest.config.ts names no test.include: the check would fail open"
    rel = file.resolve().relative_to(admin_ui.resolve()).as_posix()

    def match(glob: str) -> bool:
        return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(rel, g.removeprefix("**/"))
                   for g in _braces(glob))

    return any(match(g) for g in include) and not any(match(g) for g in exclude)


NAME = "__EXERCISE_NAME__"


def _sanitize(text: str, name: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
            continue
        if c in "'\"`":
            j, buf = i + 1, []
            while j < n and text[j] != c:
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j:j + 2])
                    j += 2
                    continue
                buf.append(text[j])
                j += 1
            value = "".join(buf)
            out.append(NAME if value == name else "S")
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


CHAIN = r"\b(?:describe|suite|it|test)(?:\.\w+)*\."


def vitest_problem(file: Path, name: str, admin_ui: Path | None = None) -> str | None:
    """The named Vitest test exists, is collected by the admin UI's Vitest config, is called with a
    test function, and nothing in its file skips, conditions or `.only`s it (rev2 review of 040 P0;
    the reviewer's hardened check). Comments and strings are removed before matching. The run itself
    is checked by Frontend CI from Vitest's JSON report (`.github/scripts/check_vitest_exercises.py`)."""
    if not file.is_file():
        return f"{file} does not exist"
    if admin_ui is not None and not vitest_collects(admin_ui, file):
        return f"{file} is not under a path admin-ui/vitest.config.ts collects"
    code = _sanitize(file.read_text(), name)
    call = rf"(?<![\w.$])(?:it|test)\(\s*{NAME}\s*,"
    if not re.search(call, code):
        return f"{file} has no it/test named {name!r} (comments and strings do not count)"
    if not re.search(call + r"\s*(?:async\s*)?(?:\(|function\b|[A-Za-z_$][\w$]*\s*=>)", code):
        return f"{file}: {name!r} is called with options, not a test function"
    if re.search(CHAIN + r"(?:skip|todo|skipIf|runIf)\b", code) or re.search(r"\bx(?:it|test|describe)\(", code):
        return f"{file}: a describe/it/test in the file is skipped or conditional"
    if re.search(CHAIN + r"only\b", code) or re.search(r"\b(?:skip|todo|only)\s*:\s*true\b", code):
        return f"{file}: an only/skip/todo flag in the file"
    if re.search(r"\.skip\s*\(\s*\)", code):
        return f"{file}: a context skip in the file"
    return None


def exercise_problem(record: dict[str, Any], outcomes: dict[str, str]) -> str | None:
    exercise = record["exercise"]
    if exercise is None:
        if set(record["flags"]) & {"history_only", "text_only"}:
            return None
        return f"{record['id']}: no exercise"
    kind, _, node = exercise.partition(":")
    if kind == "vitest":
        path, _, name = node.partition("::")
        problem = vitest_problem(ROOT / path, name, admin_ui=ROOT / "admin-ui")
        return f"{record['id']}: {problem}" if problem else None
    outcome = outcomes.get(node, "not collected")
    if outcome != "passed":
        return f"{record['id']}: exercise {node} {outcome}"
    return None


# ── the exercise run (one subprocess for every pytest exercise, ID-2) ─────────────────


def _pytest_nodes() -> list[str]:
    return sorted({r["exercise"][len("pytest:"):] for r in ENTRIES.values()
                   if r["exercise"] and r["exercise"].startswith("pytest:")})


def _junit_key(case: ET.Element) -> str:
    module = case.get("classname", "")
    parts = module.split(".")
    for i in range(len(parts), 0, -1):  # a test class adds a trailing component
        path = Path(*parts[:i]).with_suffix(".py")
        if (ROOT / path).is_file():
            return "::".join([path.as_posix(), *parts[i:], case.get("name", "")])
    return f"{module}::{case.get('name')}"


def run_exercises(nodes: list[str]) -> tuple[dict[str, str], float]:
    files = sorted({n.split("::", 1)[0] for n in nodes if (ROOT / n.split("::", 1)[0]).is_file()})
    present = [n for n in nodes if n.split("::", 1)[0] in files]
    if not present:
        return {}, 0.0
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "-o", "addopts=", *files], cwd=ROOT, capture_output=True, text=True, timeout=600)
    known = set(collected.stdout.split())
    runnable = [n for n in present if n in known]
    if not runnable:
        return {}, 0.0
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "exercises.xml"
        started = time.monotonic()
        subprocess.run(
            [sys.executable, "-m", "pytest", *runnable, "-q", "-p", "no:cacheprovider",
             "-o", "addopts=", "-rA", f"--junitxml={junit}"],
            cwd=ROOT, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
        elapsed = time.monotonic() - started
        outcomes: dict[str, str] = {}
        if junit.is_file():
            for case in ET.parse(junit).iter("testcase"):
                if case.find("failure") is not None:
                    state = "failed"
                elif case.find("error") is not None:
                    state = "errored"
                elif case.find("skipped") is not None:
                    state = "skipped"
                else:
                    state = "passed"
                outcomes[_junit_key(case)] = state
    return outcomes, elapsed


@pytest.fixture(scope="module")
def exercise_outcomes() -> dict[str, str]:
    outcomes, elapsed = run_exercises(_pytest_nodes())
    # Recorded in the P0 review record (TID §3.6): the exercises run a second time here.
    print(f"\n[frozen identifiers] {len(outcomes)} exercise(s) ran in {elapsed:.1f} s")
    return outcomes


# ── the parametrised checks ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("entry_id", IDS)
def test_no_entry_is_pending(entry_id):
    problem = pending_problem(_entry(entry_id))
    assert problem is None, problem


@pytest.mark.parametrize("entry_id", IDS)
def test_entry_is_present(entry_id):
    record = _entry(entry_id)
    if "history_only" in record["flags"]:
        pytest.skip(f"history_only: {entry_id} is carried only by dated records")
    problem = presence_problem(record)
    # No mirror leniency: since census 09b4737 a public variant lists only the literals the mirror
    # keeps (read from sync-to-clean.yml's strip step), so a missing literal is a real failure.
    assert problem is None, problem


@pytest.mark.parametrize("entry_id", IDS)
def test_exercise_is_real(entry_id, exercise_outcomes):
    problem = exercise_problem(_entry(entry_id), exercise_outcomes)
    assert problem is None, problem


def test_the_export_carries_entries():
    assert len(ENTRIES) >= 20, f"only {len(ENTRIES)} entries: the export lost records"


# ── self-tests on temp exports ────────────────────────────────────────────────────────


def _temp_export(tmp_path: Path, change) -> li.Export:
    lines = li.default_path().read_text(encoding="utf-8").splitlines()
    import json

    for i, raw in enumerate(lines):
        if raw.startswith('{"section":"entry"'):
            rec = json.loads(raw)
            if rec["exercise"] and rec["exercise"].startswith("pytest:"):
                change(rec)
                lines[i] = json.dumps(rec)
                break
    path = tmp_path / li.default_path().name
    path.write_text("\n".join(lines) + "\n")
    return li.load(path)


def test_a_pending_record_fails(tmp_path):
    export = _temp_export(tmp_path, lambda r: r.update(exercise=None, flags=["pending"]))
    pending = [r for r in export.entries() if "pending" in r["flags"]]
    assert pending and pending_problem(pending[0]) is not None
    assert all(pending_problem(r) is None for r in export.entries() if "pending" not in r["flags"])


def test_a_skipped_exercise_fails():
    record = next(r for r in ENTRIES.values() if r["exercise"] and r["exercise"].startswith("pytest:"))
    node = record["exercise"][len("pytest:"):]
    assert exercise_problem(record, {node: "passed"}) is None
    for state in ("skipped", "failed", "errored"):
        assert state in (exercise_problem(record, {node: state}) or ""), state
    assert "not collected" in (exercise_problem(record, {}) or "")


@pytest.mark.parametrize("body,bad", [
    ("it('the name', () => {});", False),
    ("it.skip('the name', () => {});", True),
    ("xit('the name', () => {});", True),
    ("it.todo('the name');", True),
    ("describe.skip('x', () => { it('the name', () => {}); });", True),
    ("it('the name', () => {}); it.only('other', () => {});", True),
    ("it('another name', () => {});", True),
])
def test_a_skipped_vitest_exercise_fails(tmp_path, body, bad):
    file = tmp_path / "x.test.ts"
    file.write_text(body)
    assert (vitest_problem(file, "the name") is not None) is bad


def test_the_exercise_file_must_be_collected(tmp_path):
    admin = tmp_path / "admin-ui"
    (admin / "src" / "e2e").mkdir(parents=True)
    (admin / "vitest.config.ts").write_text(
        "export default { test: { include: ['**/*.{test,spec}.{ts,tsx}'], exclude: ['e2e/**', '**/e2e/**'] } };\n")
    good, helper, excluded = admin / "src" / "a.test.ts", admin / "src" / "a.helper.ts", admin / "src" / "e2e" / "b.test.ts"
    for f in (good, helper, excluded):
        f.write_text("it('the name', () => {});\n")
    assert vitest_problem(good, "the name", admin_ui=admin) is None
    assert "collects" in (vitest_problem(helper, "the name", admin_ui=admin) or "")
    assert "collects" in (vitest_problem(excluded, "the name", admin_ui=admin) or "")
    real = ROOT / "admin-ui" / "src" / "__tests__" / "legacy" / "frozenStorageKeys.test.ts"
    assert vitest_collects(ROOT / "admin-ui", real)


def test_the_exercise_wiring_checks_collection(monkeypatch):
    """T03's own call passes the admin UI root, so a real exercise is checked for collection too."""
    import sys

    calls = []
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "vitest_problem",
                        lambda file, name, admin_ui=None: calls.append((file, name, admin_ui)))
    record = next(r for r in ENTRIES.values() if (r["exercise"] or "").startswith("vitest:"))
    exercise_problem(record, {})
    assert len(calls) == 1 and calls[0][2] == ROOT / "admin-ui", calls
