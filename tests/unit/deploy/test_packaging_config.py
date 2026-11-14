"""P8 (041 FR-041.5, TID §3.7): packaging and migrations follow the package.

`pyproject.toml` names the package in six places and `alembic.ini` points at its migrations. A rename
that misses one installs a console script that imports nothing, measures coverage of nothing, or
finds no migration head - each only visible when the thing runs. Every value here is compared with
`package_name()`, so this file is the same before and after P8.
"""

from __future__ import annotations

import ast
import tomllib

from alembic.config import Config
from alembic.script import ScriptDirectory

from tests.support.package_root import LEGACY_PACKAGES, REPO_ROOT, package_name, package_path


def test_pyproject_names_the_package():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pkg = package_name()
    project = data["project"]
    tool = data["tool"]
    assert project["name"] == pkg, project["name"]
    assert project["scripts"] == {pkg: f"{pkg}.main:run"}, project["scripts"]
    assert tool["setuptools"]["packages"]["find"]["include"] == [f"{pkg}*"]
    assert tool["coverage"]["run"]["source"] == [pkg], tool["coverage"]["run"]["source"]
    omit = tool["coverage"]["run"]["omit"]
    assert f"{pkg}/db/migrations/*" in omit, omit
    assert tool["ruff"]["lint"]["isort"]["known-first-party"] == [pkg]
    if pkg not in LEGACY_PACKAGES:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for old in LEGACY_PACKAGES:
            stale = [v for v in (project["name"], *project["scripts"], *project["scripts"].values(), *omit)
                     if old in v]
            assert not stale, f"pyproject.toml still names {old}: {stale}"
            assert f'"{old}*"' not in text and f'["{old}"]' not in text


def _revision(path) -> str:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "revision"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path} has no `revision = ...`")


def test_alembic_has_one_head_equal_to_the_newest_file():
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    location = cfg.get_main_option("script_location")
    assert location == f"{package_name()}/db/migrations", location
    cfg.set_main_option("script_location", str(REPO_ROOT / location))
    heads = ScriptDirectory.from_config(cfg).get_heads()
    files = sorted(p for p in package_path("db", "migrations", "versions").glob("[0-9]*.py"))
    assert len(files) >= 40, f"only {len(files)} migration files found"
    assert heads == [_revision(files[-1])], (heads, files[-1].name)
