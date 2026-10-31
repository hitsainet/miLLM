"""miLLM runs on ONE Python version, and every place that names it agrees.

Operator decision 2026-10-21 (suite memory ``python-313-upgrade-decision``): the suite moves
from 3.11 to 3.13 one app at a time; miForge went first (its guard is the model for this one),
miLLM second. The version is written in places that nothing tied together:

* ``Dockerfile``'s ``FROM python:X.Y-slim`` - what production runs;
* every ``actions/setup-python`` step in ``.github/workflows/`` - what CI tests and lints on
  (the three unit shards, the PostgreSQL schema guards, both cross-repo jobs, the mirror gate);
* ``requires-python``'s floor - what an install promises;
* the ``Programming Language :: Python :: X.Y`` classifiers - what the package claims;
* Ruff's ``target-version`` and MyPy's ``python_version`` - what the source is checked as.

Drift between them is the failure suite memory ``dev-venv-vs-image-version-drift`` records: a
suite green on one version, an image shipping another. miLLM has a concrete stake in it:
``asyncio.Semaphore`` hands a released permit to the woken waiter on 3.12+ and does not on 3.11
(``millm/services/request_queue.py``, ``acquire_background``), and ``asyncio.wait_for`` stopped
wrapping its coroutine in a new task in 3.12 - so the interpreter CI tests on decides which
admission semantics the suite actually exercised.

Every value is read from the file it governs; nothing here restates "3.13" except the parser
self-checks, so the next upgrade edits those files and this test stays as is.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
DOCKERFILE = REPO / "Dockerfile"
WORKFLOWS = REPO / ".github" / "workflows"

_FROM_PYTHON = re.compile(r"^FROM\s+(?:--platform=\S+\s+)?python:(\d+)\.(\d+)[^\s]*", re.M | re.I)
_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (\d+\.\d+)$")


def _dockerfile_versions(text: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in _FROM_PYTHON.findall(text)]


def _setup_python_steps(workflow: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every ``actions/setup-python`` step, as (job id, its ``python-version`` or None)."""
    found: list[tuple[str, Any]] = []
    for job_id, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            if str(step.get("uses", "")).startswith("actions/setup-python"):
                found.append((job_id, (step.get("with") or {}).get("python-version")))
    return found


def _mm(text: Any) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)\.(\d+)", str(text).strip())
    assert m, f"not a MAJOR.MINOR version: {text!r}"
    return int(m.group(1)), int(m.group(2))


def _the_shipped_version() -> tuple[int, int]:
    versions = _dockerfile_versions(DOCKERFILE.read_text())
    assert versions, "Dockerfile has no `FROM python:X.Y` line - the parser is blind"
    assert len(set(versions)) == 1, f"Dockerfile names several Pythons: {versions}"
    return versions[0]


def _all_setup_python_steps() -> list[tuple[str, str, Any]]:
    steps: list[tuple[str, str, Any]] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job, version in _setup_python_steps(yaml.safe_load(path.read_text())):
            steps.append((path.name, job, version))
    return steps


# ── the parsers, against inputs whose answers are known ───────────────────────────────


class TestTheParsersSeeWhatTheyClaimTo:
    def test_a_from_line_is_read_and_a_comment_is_not(self) -> None:
        text = "# FROM python:3.9-slim\nFROM python:3.13-slim AS runtime\nFROM runtime AS dev\n"
        assert _dockerfile_versions(text) == [(3, 13)]

    def test_a_lowercase_from_is_read_too(self) -> None:
        # Docker keywords are case-insensitive; a lowercase line must not make the parser blind.
        assert _dockerfile_versions("from python:3.12-slim as runtime\n") == [(3, 12)]

    def test_a_setup_python_step_without_a_version_is_reported_not_skipped(self) -> None:
        wf = {"jobs": {"a": {"steps": [{"uses": "actions/setup-python@v5"}]}}}
        assert _setup_python_steps(wf) == [("a", None)]

    def test_a_quoted_and_an_unquoted_version_read_the_same(self) -> None:
        # Unquoted `3.10` is the YAML float 3.1 - the reason workflows quote it.
        wf = yaml.safe_load(
            "jobs:\n a:\n  steps:\n  - uses: actions/setup-python@v5\n"
            "    with:\n     python-version: '3.13'\n"
        )
        assert [_mm(v) for _, v in _setup_python_steps(wf)] == [(3, 13)]


# ── every place agrees ─────────────────────────────────────────────────────────────────


class TestEveryPlaceNamesTheShippedVersion:
    def test_every_ci_python_is_the_image_python(self) -> None:
        shipped = _the_shipped_version()
        steps = _all_setup_python_steps()
        # backend-tests.yml has four (unit shards, schema guards, two cross-repo jobs) and
        # sync-to-clean.yml one (the mirror gate). Fewer means the walk stopped seeing them,
        # and an empty comparison below would pass.
        assert len([s for s in steps if s[0] == "backend-tests.yml"]) >= 4, steps
        assert len([s for s in steps if s[0] == "sync-to-clean.yml"]) >= 1, steps
        wrong = [s for s in steps if s[2] is None or _mm(s[2]) != shipped]
        assert not wrong, f"image runs {shipped}; these CI steps do not: {wrong}"

    def test_requires_python_floor_is_the_image_python(self) -> None:
        spec = tomllib.loads(PYPROJECT.read_text())["project"]["requires-python"]
        m = re.fullmatch(r">=\s*(\d+\.\d+)", spec.strip())
        assert m, f"requires-python should be a bare floor, got {spec!r}"
        assert _mm(m.group(1)) == _the_shipped_version()

    def test_the_version_classifiers_name_only_the_image_python(self) -> None:
        classifiers = tomllib.loads(PYPROJECT.read_text())["project"]["classifiers"]
        named = [_mm(m.group(1)) for c in classifiers if (m := _CLASSIFIER.match(c))]
        assert named, "no `Programming Language :: Python :: X.Y` classifier - parser blind?"
        assert set(named) == {_the_shipped_version()}, named

    def test_ruff_and_mypy_target_the_image_python(self) -> None:
        tools = tomllib.loads(PYPROJECT.read_text())["tool"]
        major, minor = _the_shipped_version()
        assert tools["ruff"]["target-version"] == f"py{major}{minor}"
        assert _mm(tools["mypy"]["python_version"]) == (major, minor)

    def test_the_interpreter_running_this_suite_meets_the_floor(self) -> None:
        """CI installs the shipped version, so this is equality there; a newer local venv
        is allowed, an older one is not a miLLM environment."""
        assert sys.version_info[:2] >= _the_shipped_version()
