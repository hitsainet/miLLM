"""Where the server's Python package lives (041 FR-041.4, TID §3.1).

The name is read from `pyproject.toml` (`[tool.setuptools.packages.find] include`), so every test that
walks or opens the package is the same file before and after P8 renames the package. A missing root, or
a root that holds no `.py` file, RAISES `PackageRootMissing`: a walk over nothing must never pass.

`MISERVER_TEST_PACKAGE_ROOT` (read through `tests/support/env_flags.py`, new-only) points the helpers
at another directory; `tests/unit/test_package_walks_fail_closed.py` points it at a directory that does
not exist and requires every walk to fail.
"""

from __future__ import annotations

import functools
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Names the package had before. A test that spells one of these as a path segment is a walk that will
#: go quiet when the package moves (`test_package_walks_fail_closed.py` fails on it).
LEGACY_PACKAGES: tuple[str, ...] = ("millm",)


class PackageRootMissing(RuntimeError):
    """The package directory is absent or holds no Python file."""


@functools.cache
def package_name() -> str:
    """The single `packages.find.include` entry of `pyproject.toml`, without its trailing `*`."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]
    if len(include) != 1:
        raise PackageRootMissing(f"pyproject.toml packages.find.include must name one package, got {include!r}")
    name = include[0].rstrip("*")
    if not name.isidentifier():
        raise PackageRootMissing(f"pyproject.toml packages.find.include {include[0]!r} is not a package name")
    return name


def package_root() -> Path:
    """The package directory; raises `PackageRootMissing` unless it is a directory holding a `.py` file."""
    from tests.support import env_flags  # here, not at import: env_flags reads package_name()

    override = env_flags.flag("TEST_PACKAGE_ROOT")
    root = Path(override) if override else REPO_ROOT / package_name()
    if not root.is_dir() or next(root.rglob("*.py"), None) is None:
        raise PackageRootMissing(f"package root {root} is missing or holds no .py file")
    return root


def package_path(*parts: str) -> Path:
    """`package_root()` joined with `parts`; the root is checked, the leaf is not."""
    return package_root().joinpath(*parts)


def package_rel(*parts: str) -> str:
    """A repository-relative POSIX path inside the package, for `git ls-files` and report keys. The root
    is checked like `package_root()`, and must lie inside the repository."""
    root = package_root()
    try:
        rel = root.resolve().relative_to(REPO_ROOT)
    except ValueError:
        raise PackageRootMissing(f"package root {root} is outside the repository {REPO_ROOT}") from None
    return "/".join((rel.as_posix(), *parts))
