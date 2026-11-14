"""P2 (041 FR-041.4, TID §3.1): the package-root helper reads `pyproject.toml` and fails closed.

Every test that walks the package goes through `tests/support/package_root.py`; if that helper could
return a directory that is absent or empty, every walk built on it would pass over nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support import env_flags, package_root as pr


@pytest.fixture
def fresh_name_cache():
    pr.package_name.cache_clear()
    yield
    pr.package_name.cache_clear()


def _repo(tmp_path: Path, include: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\n\n[tool.setuptools.packages.find]\ninclude = ["{include}"]\n',
        encoding="utf-8")
    return tmp_path


def test_package_name_comes_from_pyproject(tmp_path, monkeypatch, fresh_name_cache):
    # The real tree: the name is the directory that holds the server.
    assert (pr.REPO_ROOT / pr.package_name() / "main.py").is_file(), pr.package_name()
    assert (pr.REPO_ROOT / pr.package_name() / "__init__.py").is_file(), pr.package_name()
    # A different pyproject gives a different name: it is read, not spelled.
    pr.package_name.cache_clear()
    monkeypatch.setattr(pr, "REPO_ROOT", _repo(tmp_path, "otherpkg*"))
    assert pr.package_name() == "otherpkg"
    pkg = tmp_path / "otherpkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.delenv(env_flags.new_name("TEST_PACKAGE_ROOT"), raising=False)
    assert pr.package_root() == pkg
    assert pr.package_path("mod.py") == pkg / "mod.py"
    assert pr.package_rel("db", "x.py") == "otherpkg/db/x.py"


def test_missing_root_raises(tmp_path, monkeypatch, fresh_name_cache):
    monkeypatch.setenv(env_flags.new_name("TEST_PACKAGE_ROOT"), str(tmp_path / "missing"))
    with pytest.raises(pr.PackageRootMissing, match="missing"):
        pr.package_root()
    with pytest.raises(pr.PackageRootMissing):
        pr.package_path("main.py")
    with pytest.raises(pr.PackageRootMissing):
        pr.package_rel("main.py")
    # Without the override: a pyproject naming a package that is not there.
    monkeypatch.delenv(env_flags.new_name("TEST_PACKAGE_ROOT"))
    monkeypatch.setattr(pr, "REPO_ROOT", _repo(tmp_path, "gone*"))
    with pytest.raises(pr.PackageRootMissing):
        pr.package_root()


def test_root_without_python_files_raises(tmp_path, monkeypatch, fresh_name_cache):
    empty = tmp_path / "pkg"
    (empty / "sub").mkdir(parents=True)
    (empty / "sub" / "notes.txt").write_text("not python\n", encoding="utf-8")
    monkeypatch.setenv(env_flags.new_name("TEST_PACKAGE_ROOT"), str(empty))
    with pytest.raises(pr.PackageRootMissing, match="no .py file"):
        pr.package_root()
    (empty / "sub" / "m.py").write_text("X = 1\n", encoding="utf-8")
    assert pr.package_root() == empty


def test_override_is_read_through_env_flags(tmp_path, monkeypatch, fresh_name_cache):
    pkg = tmp_path / "elsewhere"
    pkg.mkdir()
    (pkg / "m.py").write_text("X = 1\n", encoding="utf-8")
    asked: list[str] = []

    def fake_flag(name, default=None):
        asked.append(name)
        return str(pkg) if name == "TEST_PACKAGE_ROOT" else default

    monkeypatch.setattr(env_flags, "flag", fake_flag)
    assert pr.package_root() == pkg
    assert asked == ["TEST_PACKAGE_ROOT"], asked
    # And the real reader: the environment variable, under its new name only.
    monkeypatch.undo()
    monkeypatch.setenv(env_flags.new_name("TEST_PACKAGE_ROOT"), str(pkg))
    assert pr.package_root() == pkg
