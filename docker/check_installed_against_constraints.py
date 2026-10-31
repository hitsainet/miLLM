"""Fail when the installed Python environment differs from constraints.txt.

Run by the image build (Dockerfile, after every `pip install`) and by
tests/unit/test_dependency_lock.py in CI and the dev venv, so the lock is checked where it is
installed, not only where it is written. Two failures:

* a distribution is installed that constraints.txt does not name — it floated in unpinned;
* a distribution is installed at a version other than its constraint.

A local version label is ignored (`torch 2.14.1+cpu` in CI and `+cu130` in the image both satisfy
`torch==2.14.1`, which is how pip itself compares them). `millm` (the project) and `pip` (the
installer, which arrives with the interpreter) are the only exemptions.

Usage: python docker/check_installed_against_constraints.py [path/to/constraints.txt]
Exit 0 when the environment matches, 1 with one line per problem otherwise.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

#: Installed but never pinned: the project itself, and the installer.
EXEMPT = frozenset({"millm", "pip"})


def normalise(name: str) -> str:
    """PEP 503 name normalisation (`Jinja2`, `typing_extensions` -> `jinja2`, `typing-extensions`)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_constraints(path: Path) -> dict[str, str]:
    """`name -> version` for every `name==version` line; anything else is an error."""
    pins: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;=<>!~,]+)", line)
        if not match:
            raise ValueError(f"{path}:{number}: not an exact `name==version` pin: {raw!r}")
        name = normalise(match.group(1))
        if name in pins:
            raise ValueError(f"{path}:{number}: {name} is pinned twice")
        pins[name] = match.group(2)
    return pins


def installed() -> dict[str, str]:
    """`name -> version` for every distribution visible to this interpreter."""
    found: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            found[normalise(name)] = dist.version
    return found


def problems(pins: dict[str, str], present: dict[str, str]) -> list[str]:
    out: list[str] = []
    for name, version in sorted(present.items()):
        if name in EXEMPT:
            continue
        if name not in pins:
            out.append(f"{name}=={version} is installed but not pinned in constraints.txt")
        elif version.split("+", 1)[0] != pins[name]:
            out.append(f"{name} is installed at {version}; constraints.txt pins {pins[name]}")
    return out


def main(argv: list[str]) -> int:
    path = (
        Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "constraints.txt"
    )
    found = problems(read_constraints(path), installed())
    for line in found:
        print(f"constraints drift: {line}", file=sys.stderr)
    if not found:
        print(f"environment matches {path}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
