"""T06: no direct read of a `MILLM_*` or `MISERVER_*` environment name outside the helpers
(040 FR-040.8, FR-040.9).

Names are read through ONE reader (`miserver.core.legacy_names.env`, and `tests/support/env_flags.py`
over it). Until P7 that reader honoured both spellings; since P7 it reads the new one and reports a
retired old one that is still set. A direct `os.environ.get("MILLM_X")` would bypass both, so an old
name could go dark, or stay in use unseen. Writes in tests
(`monkeypatch.setenv`) are not reads and are allowed.
"""

from __future__ import annotations

import ast
import re
import subprocess
import warnings
from pathlib import Path
from tests.support.package_root import package_rel

ROOT = Path(__file__).resolve().parents[2]
PREFIX = re.compile(r"^(MILLM|MISERVER)_[A-Z]")

#: The two files that spell the names (TID §1.1 principle 1).
HELPERS = frozenset({package_rel("core", "legacy_names.py"), "tests/support/env_flags.py"})

#: Direct reads left in place on purpose, each with its reason and the line text that must still
#: exist (so the entry is removed the day the site is converted).
DEFERRED: dict[str, tuple[str, str]] = {
    # Empty since main (G38, D53) merged into rebrand/p0 (2026-11-08): both deferred reads converted.
    # A future entry exempts only its read: its line and as many reads as that line text holds.
}

_READ_CALLS = {"get", "getenv", "setdefault", "pop", "getenvb"}
#: Test writes may take an aliased name; the reader is matched by IMPORT (`_reader_names`), never by
#: a bare name, so a local `env()` wrapper is still a read (rev2 review of 040 P0).
_WRITE_CALLS = {"setenv", "delenv"}
READER_MODULE = "miserver.core.legacy_names"


def _reader_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(names bound to the reader module, names bound to its `env` function) in one file."""
    modules, functions = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == READER_MODULE:
                    modules.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if node.module == "miserver.core" and a.name == "legacy_names":
                    modules.add(a.asname or a.name)
                elif node.module == READER_MODULE and a.name == "env":
                    functions.add(a.asname or a.name)
    return modules, functions


def _is_allowed_call(node: ast.Call, modules: set[str], functions: set[str]) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr in _WRITE_CALLS:
            return True
        target = ast.unparse(func.value)
        return func.attr == "env" and target in modules
    return isinstance(func, ast.Name) and func.id in functions


def _alias_names() -> frozenset[str]:
    from miserver.core import legacy_names

    return frozenset(n for a in legacy_names.ENV_ALIASES for n in (a.old, a.new))


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _is_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ")


def reads(source: str) -> list[tuple[int, str]]:
    """Every direct environment read of a `MILLM_*`/`MISERVER_*` name: `.get`, `getenv`,
    `setdefault`, `pop` on the environment, `environ[...]`, and `... in environ`."""
    found: list[tuple[int, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    modules, functions = _reader_names(tree)
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            func = node.func
            if func.attr in _READ_CALLS and (_is_environ(func.value) or func.attr == "getenv"):
                name = _name_of(node.args[0])
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            name = _name_of(node.slice)
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and isinstance(node.ops[0], (ast.In, ast.NotIn)) and _is_environ(node.comparators[0]):
            name = _name_of(node.left)
        if name and PREFIX.match(name):
            found.append((node.lineno, name))
        elif isinstance(node, ast.Call) and node.args and not _is_allowed_call(node, modules, functions):
            # Indirection (review of the G38 merge): `_env("MILLM_X")` reads the old name through a
            # local wrapper. Passing an aliased name to anything but the reader is a read.
            arg = _name_of(node.args[0])
            if arg in _alias_names():
                found.append((node.lineno, arg))
    return found


def _python_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", package_rel(), "tests"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return [rel for rel in out if rel.endswith(".py") and (ROOT / rel).is_file()]


def test_no_millm_or_miserver_literal_outside_the_helpers():
    files = _python_files()
    assert len(files) > 500, f"listed only {len(files)} Python files: the guard would fail open"
    hits = []
    for rel in files:
        if rel in HELPERS:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8")
        found = reads(text)
        if rel in DEFERRED:  # only the deferred READ is exempt: its line, its count (review D02)
            line_text = DEFERRED[rel][0]
            allowed = len(reads(line_text))
            deferred_lines = {n for n, src in enumerate(text.splitlines(), start=1) if line_text in src}
            kept, used = [], dict.fromkeys(deferred_lines, 0)
            for line, name in found:
                if line in used and used[line] < allowed:
                    used[line] += 1
                    continue
                kept.append((line, name))
            found = kept
        hits += [f"{rel}:{line}: {name}" for line, name in found]
    assert not hits, (
        "Read these through `miserver.core.legacy_names.env` (server) or "
        "`tests.support.env_flags.flag` (tests), which honour both spellings and record the old "
        "one (040 FR-040.8/9):\n  " + "\n  ".join(hits))


def test_every_deferred_site_is_still_there():
    """A deferral that no longer applies is removed, not kept as a silent exemption."""
    assert isinstance(DEFERRED, dict)
    for rel, (line_text, _why) in DEFERRED.items():
        assert line_text in (ROOT / rel).read_text(encoding="utf-8"), (
            f"{rel} no longer reads {line_text!r}: remove its DEFERRED entry")


def test_the_guard_finds_one_allowed_literal_per_context():
    """The floor: each read form is found, so a guard that matches nothing cannot pass."""
    planted = (
        'import os\nfrom os import environ\n'
        'a = os.environ.get("MILLM_A")\n'
        'b = os.getenv("MISERVER_B", "x")\n'
        'c = os.environ["MILLM_C"]\n'
        'd = "MISERVER_D" in os.environ\n'
        'e = os.environ.setdefault("MILLM_E", "1")\n'
        'f = os.environ.pop(f"MILLM_F{x}", None)\n'
        'g = environ.get("MILLM_G")\n'
        'ok1 = os.environ.get("MISTUDIO_REPO")\n'
        'ok2 = monkeypatch.setenv("MISERVER_H", "1")\n'
        'ok3 = "MILLM_I"\n'
        'h = _env("MILLM_REAL_TOKENIZERS")\n'
        'ok4 = monkeypatch.delenv("MILLM_STRICT_RESULTS")\n'
        'from miserver.core import legacy_names as ln\n'
        'ok5 = ln.env("MISERVER_STRICT_RESULTS")\n'
        'def env(name):\n    return name\n'
        'i = env("MILLM_REQUIRE_PG")\n'
        'j = flag("MISERVER_BATCH_HW_URL")\n'
        'k = other_module.env("MILLM_REQUIRE_RUNNER")\n'
    )
    assert [name for _l, name in reads(planted)] == [
        "MILLM_A", "MISERVER_B", "MILLM_C", "MISERVER_D", "MILLM_E", "MILLM_F", "MILLM_G",
        "MILLM_REAL_TOKENIZERS", "MILLM_REQUIRE_PG", "MISERVER_BATCH_HW_URL", "MILLM_REQUIRE_RUNNER"]
    helpers = {rel: reads((ROOT / rel).read_text()) for rel in HELPERS}
    assert all(isinstance(v, list) for v in helpers.values())
