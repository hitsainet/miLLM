"""P7 (041 FR-041.2, TID §3.9): the server's package is `miserver`, and nothing reaches the old `millm`.

* The old package cannot be imported at all: a leftover `millm/` (a stale `__pycache__`, an editable
  install of the old name) would let a missed `import millm` run old code and pass.
* Every module of the package imports, at least as many as at P8's start minus 5 % (263 → floor 249), and
  afterwards no `millm` module is loaded.
* No `import`/`from … import` anywhere in the Python trees names `millm`, at any depth (lazy imports
  included), found by AST.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
import subprocess
import sys

from tests.support.package_root import REPO_ROOT, package_name, package_root

OLD = "millm"
#: Modules counted at P8 start (041 record §1.1: 263), less 5 %.
MODULE_FLOOR = 249
#: Third-party modules a dev or CI environment may lack; a module of the package that needs one is
#: skipped from the import, never the package itself (the sidecar's `wasmtime`, GGUF's `llama_cpp`).
OPTIONAL = frozenset({"wasmtime", "llama_cpp"})
TREES = (package_name(), "tests", "docker", "integrations", "scripts", ".github/scripts")


def test_old_package_is_not_importable():
    assert package_name() != OLD
    assert importlib.util.find_spec(OLD) is None, (
        f"`import {OLD}` resolves ({importlib.util.find_spec(OLD).origin}): a leftover directory or an old "
        f"editable install (`pip uninstall -y {OLD} && pip install -e .`, FTASKS 10.21)")


def test_every_module_imports():
    names = [m.name for m in pkgutil.walk_packages([str(package_root())], package_name() + ".")]
    assert len(names) >= MODULE_FLOOR, f"only {len(names)} modules found (floor {MODULE_FLOOR})"
    failed, optional = [], []
    for name in names:
        if ".db.migrations" in name:  # alembic's env.py runs a migration when imported
            continue
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] in OPTIONAL:
                optional.append(name)
            else:
                failed.append(f"{name}: {exc!r}")
        except Exception as exc:  # noqa: BLE001 - every failure is listed
            failed.append(f"{name}: {exc!r}")
    assert not failed, "\n".join(failed)
    assert len(optional) <= 4, optional
    loaded = sorted(k for k in sys.modules if k == OLD or k.startswith(OLD + "."))
    assert not loaded, loaded


def _python_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", *TREES], cwd=REPO_ROOT, capture_output=True, text=True,
                         check=True).stdout.split("\n")
    return [f for f in out if f.endswith(".py")]


def test_no_import_names_millm():
    files = _python_files()
    assert len(files) > 900, f"only {len(files)} Python files listed: the scan would read nothing"
    found = []
    for rel in files:
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            found += [f"{rel}:{node.lineno}: {n}" for n in names if n == OLD or n.startswith(OLD + ".")]
    assert not found, "imports of the old package:\n" + "\n".join(found)
