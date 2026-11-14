"""P3 (041 FR-041.7, TID §3.3): CI tests and builds the package under its current directory name.

The package's directory is read from `pyproject.toml` (`package_name()`), so this file is the same before
and after P8 renamed the package directory (`millm` to `miserver`). Between the two, C2 lists both directories; R drops the old.

* A `pull_request` path filter of Backend Tests that names Python paths lists `<package>/**`, so a change
  to the package runs the suite.
* The backend image's path expression (docker-images.yml `paths_for backend`, run in bash and applied
  with `grep -E` exactly as the detect step does) matches `<package>/main.py`; once the package is no
  longer `millm`, it matches nothing that exists only under the old directory.
* Every `MYPY_GATED` path exists, and every `python -m <module>` the workflows run resolves.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.support.package_root import LEGACY_PACKAGES, REPO_ROOT, package_name

WORKFLOWS = REPO_ROOT / ".github" / "workflows"
BACKEND_TESTS = WORKFLOWS / "backend-tests.yml"
DOCKER_IMAGES = WORKFLOWS / "docker-images.yml"
#: `python -m` modules that are tools, not the package.
TOOLS = frozenset({"pytest", "ruff", "mypy", "alembic", "uvicorn", "pip", "build", "coverage"})


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # YAML 1.1 reads the key `on` as True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def _runs(path: Path) -> list[str]:
    return [step["run"] for job in _load(path)["jobs"].values() for step in job.get("steps", [])
            if isinstance(step.get("run"), str)]


def _backend_regex() -> str:
    detect = next(step["run"] for job in _load(DOCKER_IMAGES)["jobs"].values()
                  for step in job.get("steps", []) if step.get("id") == "detect")
    fn = re.search(r"^ *paths_for\(\) \{.*?^ *\}$", detect, flags=re.M | re.S)
    assert fn, "paths_for() not found in docker-images.yml's detect step"
    regex = subprocess.run(["bash", "-c", fn.group(0) + '\npaths_for backend'], capture_output=True,
                           text=True, check=True).stdout.strip()
    assert regex and regex != "^$", f"paths_for backend gave {regex!r}"
    return regex


def _matches(regex: str, path: str) -> bool:
    """`echo "$CHANGED" | grep -qE "$(paths_for backend)"`, as the detect step runs it."""
    proc = subprocess.run(["grep", "-qE", regex], input=path + "\n", text=True)
    assert proc.returncode in (0, 1), proc
    return proc.returncode == 0


def test_paths_filters_list_the_package():
    on = _load(BACKEND_TESTS)["on"]
    filters = {event: spec["paths"] for event, spec in on.items()
               if isinstance(spec, dict) and "paths" in spec}
    python_filters = {e: p for e, p in filters.items() if "tests/**" in p}
    assert python_filters, f"no path filter of Backend Tests lists tests/**: {filters}"
    for event, paths in python_filters.items():
        assert f"{package_name()}/**" in paths, (
            f"backend-tests.yml `on.{event}.paths` does not list {package_name()}/**: {paths}")


def test_backend_regex_matches_the_package_main():
    regex = _backend_regex()
    assert _matches(regex, f"{package_name()}/main.py"), regex
    assert _matches(regex, f"{package_name()}/rules/runner_worker.py"), regex
    assert not _matches(regex, "admin-ui/src/main.tsx"), "the backend expression is not that loose"


def test_backend_regex_matches_nothing_only_under_an_old_package():
    if package_name() in LEGACY_PACKAGES:
        pytest.skip(f"the package is still {package_name()!r}: active after the P8 rename (041 R)")
    regex = _backend_regex()
    for old in LEGACY_PACKAGES:
        assert not _matches(regex, f"{old}/main.py"), f"{regex} still builds on a change under {old}/"


def test_mypy_gated_paths_exist():
    text =BACKEND_TESTS.read_text(encoding="utf-8")
    found = re.findall(r"^\s*MYPY_GATED:\s*(.+)$", text, flags=re.M)
    assert len(found) == 1, f"expected one MYPY_GATED line, found {len(found)}"
    gated = found[0].split()
    assert len(gated) >= 10, gated
    missing = [p for p in gated if not (REPO_ROOT / p).exists()]
    assert not missing, f"MYPY_GATED names paths that do not exist: {missing}"
    assert all(p.split("/", 1)[0] == package_name() for p in gated), gated


def _python_m(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        if "python" not in line or " -m " not in line:
            continue
        try:
            words = shlex.split(line.replace("\\", " "))
        except ValueError:
            words = line.split()
        out += [words[i + 1] for i, w in enumerate(words[:-1]) if w == "-m" and "python" in words[i - 1]]
    return out


def test_python_m_modules_in_workflows_resolve():
    modules = [m for path in sorted(WORKFLOWS.glob("*.yml")) for run in _runs(path) for m in _python_m(run)]
    package = [m for m in modules if m.split(".")[0] not in TOOLS]
    assert package, f"no package module run by `python -m` in any workflow (found {modules})"
    for module in package:
        assert module.split(".")[0] == package_name(), f"`python -m {module}` does not name {package_name()}"
        assert importlib.util.find_spec(module) is not None, f"`python -m {module}` does not resolve"
