"""Test and CI switches (040 FR-040.9, TID §3.5.1).

`flag("X")` reads `MISERVER_X` through the server's one reader, `miserver.core.legacy_names.env`. Until
P7 that reader fell back to the old spelling; P7 (FR-040.40) retired every old name, so only the new
one is read, and this module no longer spells the old ones. Tests never read a `MISERVER_*` name
directly (`tests/unit/test_env_literal_guard.py`).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from tests.support.package_root import REPO_ROOT, package_name


def _legacy_names():
    """`miserver.core.legacy_names`, or the same file loaded alone when the `millm` package cannot be
    imported (the pytest-only CI job: `tests/unit/test_cross_repo_sync_files_need_only_pytest.py`).
    The module is standard-library only by design (T11 pins it), so loading it alone is exact."""
    try:
        from miserver.core import legacy_names as module

        return module
    except ImportError:
        name = "_millm_legacy_names_standalone"
        if name in sys.modules:
            return sys.modules[name]
        path = REPO_ROOT / package_name() / "core" / "legacy_names.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None, path
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


legacy_names = _legacy_names()

NEW = "MISERVER_"


def flag(name: str, default: str | None = None) -> str | None:
    return legacy_names.env(f"{NEW}{name}", default)


def setdefault_flag(name: str, value: str) -> None:
    """Set `MISERVER_<name>` unless it is already set (tests/conftest.py, TPD-19)."""
    os.environ.setdefault(f"{NEW}{name}", value)


def new_name(name: str) -> str:
    return f"{NEW}{name}"
