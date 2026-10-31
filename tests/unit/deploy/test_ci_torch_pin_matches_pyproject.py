"""CI's pre-installed CPU torch must satisfy the floor in pyproject.toml.

`backend-tests.yml` installs a CPU-only torch before `pip install -e ".[dev]"`,
to keep the multi-gigabyte CUDA build and the nvidia-* stack off the runner.
That only works if the pre-installed version SATISFIES the project's own
constraint. It was pinned to 2.9.1 while pyproject required >=2.10.0, so the
next step found the constraint unsatisfied, uninstalled the CPU wheel, and
pulled the CUDA build — failing with "[Errno 28] No space left on device"
before a single test ran. The step meant to save space was causing the
overrun.

Nothing else notices this: the job simply fails at install time with an error
that reads like infrastructure, and it stayed green for months only because the
runner happened to have enough disk.

This is a source-scrape guard, so it FAILS CLOSED — every anchor must be found,
because a regex that quietly matches nothing asserts nothing.

MUTATION CONTROLS (each must turn this file red):
  * pin CI back to torch==2.9.1        -> "satisfies the floor" fails
  * drop the CPU index-url from CI     -> "installs from the CPU index" fails
  * raise the pyproject floor to 2.99  -> "satisfies the floor" fails
"""

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "backend-tests.yml"
PYPROJECT = REPO / "pyproject.toml"


@pytest.fixture(scope="module")
def workflow_text() -> str:
    assert WORKFLOW.is_file(), f"workflow not found at {WORKFLOW}"
    return WORKFLOW.read_text()


@pytest.fixture(scope="module")
def project_torch_requirement() -> Requirement:
    """The torch constraint the project actually declares."""
    text = PYPROJECT.read_text()
    match = re.search(r'"(torch(?:\[[^\]]*\])?[^"]*)"', text)
    assert match, "no torch requirement found in pyproject.toml — anchor moved"
    return Requirement(match.group(1))


@pytest.fixture(scope="module")
def ci_torch_spec(workflow_text: str) -> str:
    """The torch spec CI pre-installs."""
    match = re.search(
        r"pip install\s+\"?(torch[^\"\s]*)\"?\s+--index-url", workflow_text
    )
    assert match, (
        "no `pip install <torch spec> --index-url` line in backend-tests.yml — "
        "the CPU pre-install step moved or was removed"
    )
    return match.group(1)


def test_ci_installs_torch_from_the_cpu_index(workflow_text: str) -> None:
    """Without the CPU index the default (CUDA) wheels are pulled."""
    assert "https://download.pytorch.org/whl/cpu" in workflow_text, (
        "CI must install torch from the CPU index; the CUDA build plus the "
        "nvidia-* stack does not fit on the runner"
    )


def test_the_ci_pin_satisfies_the_project_floor(
    ci_torch_spec: str, project_torch_requirement: Requirement
) -> None:
    """The whole point: pip must not replace what CI pre-installed."""
    ci_req = Requirement(ci_torch_spec)
    assert ci_req.name == "torch"

    exact = [s for s in ci_req.specifier if s.operator == "=="]
    if exact:
        version = Version(exact[0].version)
        assert project_torch_requirement.specifier.contains(version), (
            f"CI pre-installs torch=={version} but pyproject requires "
            f"{project_torch_requirement.specifier}. pip will uninstall the CPU "
            f"wheel and pull the CUDA build, exhausting the runner's disk."
        )
        return

    # A range: it must not permit anything the project forbids, or the resolver
    # is free to pick a version that gets replaced.
    for spec in ci_req.specifier:
        if spec.operator in (">=", ">"):
            assert project_torch_requirement.specifier.contains(spec.version), (
                f"CI allows torch {spec} but pyproject requires "
                f"{project_torch_requirement.specifier} — the lower bounds disagree"
            )
