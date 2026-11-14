"""P1 (041 FR-041.4, TID §3.2): every test that walks or opens the package finds it through
`tests/support/package_root.py`, and every one of them FAILS when the package is not there.

Why: a guard that walks `REPO / "millm"` passes over nothing once the package moves, and a walk over
nothing finds no violation. P8 renames the package; the walks must break loudly, never go quiet.

* Static: every `tests/**/*.py` is parsed. A package name (today's, read from `pyproject.toml`, or an
  earlier one in `LEGACY_PACKAGES`) spelled as a path - a `/` operand, a `Path(...)`/`joinpath(...)`
  argument, a root in a tuple or list handed to a call or iterated, a glob under the package, or
  `<package>.__path__` - fails, naming file and line. The call sites of the helpers are counted
  (floor), so a discovery that finds nothing fails too.
* Dynamic: every test that reaches a helper call is run once in a subprocess with
  `MISERVER_TEST_PACKAGE_ROOT` pointing at a directory that does not exist; each must fail or error.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from tests.support import env_flags
from tests.support.package_root import LEGACY_PACKAGES, REPO_ROOT, package_name

HELPERS = frozenset({"package_root", "package_path", "package_rel"})
#: Files that spell package names on purpose: the helper itself, this test, the helper's own test.
EXEMPT = frozenset({
    "tests/support/package_root.py",
    "tests/unit/test_package_walks_fail_closed.py",
    "tests/unit/support/test_package_root.py",
})
#: Fixture trees are data, not tests (P12's tree under tests/fixtures/p8_rename/ names `millm` on purpose).
EXEMPT_DIRS = ("tests/fixtures/",)
PATH_CALLS = frozenset({"Path", "PurePath", "PurePosixPath", "joinpath"})
GLOB_CHARS = frozenset("*?[")
FLOOR = 10


def _names() -> frozenset[str]:
    return frozenset({package_name(), *LEGACY_PACKAGES})


def _test_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", "tests"], cwd=REPO_ROOT, capture_output=True, text=True,
                         check=True).stdout.split()
    files = [f for f in out if f.endswith(".py") and f not in EXEMPT and not f.startswith(EXEMPT_DIRS)]
    assert len(files) > 500, f"only {len(files)} test files listed - the discovery would look at nothing"
    return files


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _segment(value: object, names: frozenset[str]) -> bool:
    """A string that names the package as a path: `<pkg>`, `<pkg>/`, `<pkg>/core/x.py`."""
    return isinstance(value, str) and value.split("/", 1)[0] in names and value.strip() == value


def _parse(source: str) -> ast.Module:
    with warnings.catch_warnings():  # an invalid escape in a scanned file is that file's business
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def literal_sites(source: str, names: frozenset[str]) -> list[tuple[int, str]]:
    """(line, form) of every package name spelled as a path in `source`."""
    tree = _parse(source)
    hits: list[tuple[int, str]] = []

    def const(node: ast.AST) -> object:
        return node.value if isinstance(node, ast.Constant) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            for side in (node.left, node.right):
                if _segment(const(side), names):
                    hits.append((side.lineno, "a `/` operand"))
        elif isinstance(node, ast.Call):
            if _call_name(node) in PATH_CALLS:
                for arg in node.args:
                    if _segment(const(arg), names):
                        hits.append((arg.lineno, f"a {_call_name(node)}() argument"))
            for arg in [*node.args, *(k.value for k in node.keywords)]:
                if isinstance(arg, (ast.Tuple, ast.List)):
                    for elt in arg.elts:
                        if const(elt) in names:
                            hits.append((elt.lineno, "a root handed to a call"))
        elif isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.iter, (ast.Tuple, ast.List)):
            for elt in node.iter.elts:
                if const(elt) in names:
                    hits.append((elt.lineno, "an iterated root"))
        elif isinstance(node, ast.Attribute) and node.attr in ("__path__", "__file__") \
                and isinstance(node.value, ast.Name) and node.value.id in names:
            hits.append((node.lineno, f"`{node.value.id}.{node.attr}`"))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "/" in node.value \
                and node.value.split("/", 1)[0] in names and GLOB_CHARS & set(node.value):
            hits.append((node.lineno, "a glob under the package"))
    return sorted(set(hits))


@dataclass(frozen=True)
class Site:
    path: str
    line: int
    nodes: tuple[str, ...]  # pytest node ids that reach it; () when nothing does


def _test_nodes(path: str, tree: ast.Module) -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    """(test function name -> node id, function name -> names it uses, names used at module level).

    Only top-level functions and the methods of top-level `Test*` classes are tests; a helper or a
    fixture is reached through the names a test uses (a fixture by its argument name)."""
    tests: dict[str, str] = {}
    uses: dict[str, set[str]] = {}

    def used(fn: ast.AST) -> set[str]:
        out = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        out |= {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out |= {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        return out

    module_level: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            uses[stmt.name] = used(stmt)
            module_level |= {n.id for d in stmt.decorator_list for n in ast.walk(d) if isinstance(n, ast.Name)}
            if stmt.name.startswith("test"):
                tests[stmt.name] = f"{path}::{stmt.name}"
        elif isinstance(stmt, ast.ClassDef):
            for sub in stmt.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = f"{stmt.name}.{sub.name}"
                    uses[key] = used(sub)
                    uses.setdefault(sub.name, set()).update(uses[key])
                    if stmt.name.startswith("Test") and sub.name.startswith("test"):
                        tests[key] = f"{path}::{stmt.name}::{sub.name}"
        else:
            module_level |= {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    return tests, uses, module_level


def derived_sites() -> list[Site]:
    sites: list[Site] = []
    for path in _test_files():
        tree = _parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _call_name(n) in HELPERS]
        if not calls:
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        tests, uses, module_level = _test_nodes(path, tree)
        for call in calls:
            chain = []
            node: ast.AST | None = call
            while node is not None:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chain.append(node)
                node = parents.get(node)
            funcs = [n for n in chain if not isinstance(n, ast.ClassDef)]
            if not funcs:
                sites.append(Site(path, call.lineno, (path,)))  # module level: the import must fail
                continue
            outer = funcs[-1]
            cls = next((n for n in chain if isinstance(n, ast.ClassDef)), None)
            key = f"{cls.name}.{outer.name}" if cls is not None else outer.name
            if key in tests:
                sites.append(Site(path, call.lineno, (tests[key],)))
                continue
            # A helper or a fixture: every test that reaches it through the names it uses.
            reach = {outer.name}
            while True:
                more = {k.split(".")[-1] for k, u in uses.items() if u & reach} - reach
                if not more:
                    break
                reach |= more
            if reach & module_level:
                sites.append(Site(path, call.lineno, (path,)))
                continue
            nodes = tuple(sorted(n for k, n in tests.items() if uses.get(k, set()) & reach))
            sites.append(Site(path, call.lineno, nodes))
    return sites


# ── the tests ──────────────────────────────────────────────────────────────────────


def test_no_walk_uses_a_literal_package_segment():
    names = _names()
    found = []
    for path in _test_files():
        for line, form in literal_sites((REPO_ROOT / path).read_text(encoding="utf-8"), names):
            found.append(f"{path}:{line}: package name spelled as a path ({form})")
    assert not found, (
        "these tests name the package directory literally; use tests/support/package_root.py "
        "(package_root/package_path/package_rel/package_name) so the walk fails when the package is "
        "not there:\n" + "\n".join(found))


def test_the_literal_check_recognises_each_form():
    """The static check is a scan; prove it matches each form it claims to (a scan that matches
    nothing asserts nothing)."""
    names = frozenset({"pkgx"})
    forms = {
        'R / "pkgx"': "a `/` operand",
        'R / "pkgx/core/a.py"': "a `/` operand",
        'Path("pkgx")': "a Path() argument",
        'R.joinpath("pkgx", "a")': "a joinpath() argument",
        'run(["git", "ls-files", "--", "pkgx"])': "a root handed to a call",
        'Tree("python", ("pkgx",), (".py",))': "a root handed to a call",
        'for r in ("pkgx", "admin-ui/src"): pass': "an iterated root",
        'x = [p for p in ("pkgx",)]': "an iterated root",
        'walk_packages(pkgx.__path__, "pkgx.")': "`pkgx.__path__`",
        'S = ("pkgx/**/*.py",)': "a glob under the package",
    }
    for snippet, form in forms.items():
        assert [f for _, f in literal_sites(snippet, names)] == [form], snippet
    for clean in ('d["pkgx"]', 'assert x == {"pkgx"}', 'NAME = "pkgx"', 'R / package_name()',
                  '"pkgx.core.mod"', 'f(name="pkgx")', '{"pkgx/a.py": 1}'):
        assert literal_sites(clean, names) == [], clean


def test_discovery_finds_at_least_ten_package_walks():
    sites = derived_sites()
    assert len(sites) >= FLOOR, f"only {len(sites)} helper call sites found; floor {FLOOR}"
    unreached = [f"{s.path}:{s.line}" for s in sites if not s.nodes]
    assert not unreached, (
        "these helper calls are reached by no test this check can name (call them from a test, a "
        "fixture or module level):\n" + "\n".join(unreached))


def test_every_package_walk_errors_on_a_missing_root(tmp_path):
    sites = derived_sites()
    assert len(sites) >= FLOOR, len(sites)
    nodes = sorted({n for s in sites for n in s.nodes})
    junit = tmp_path / "walks.xml"
    env = {k: v for k, v in os.environ.items() if k not in ("CI_SHARD", "PYTEST_ADDOPTS", "PYTEST_XDIST_WORKER")}
    env[env_flags.new_name("TEST_PACKAGE_ROOT")] = str(tmp_path / "missing")
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *nodes, "-q", "-p", "no:cacheprovider", "-o", "addopts=",
         "--continue-on-collection-errors", f"--junitxml={junit}"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - started
    assert junit.is_file(), proc.stdout[-3000:] + proc.stderr[-3000:]
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    passed, reported = [], set()
    for case in cases:
        # a test: classname "tests.unit.x.test_y[.TestC]"; a module that failed to import: classname ""
        # and name "tests.unit.x.test_y"
        dotted = case.get("classname") or case.get("name", "")
        reported.add(dotted)
        if not {child.tag for child in case} & {"failure", "error"}:
            skipped = any(child.tag == "skipped" for child in case)
            passed.append(f"{dotted}::{case.get('name')} ({'skipped' if skipped else 'passed'})")
    assert not passed, (
        f"with the package root missing these still pass or skip (ran {len(nodes)} node ids in "
        f"{elapsed:.0f} s):\n" + "\n".join(passed))
    files = {n.split("::")[0] for n in nodes}
    missing = []
    for f in sorted(files):
        mod = f[:-3].replace("/", ".")
        if not any(d == mod or d.startswith(mod + ".") for d in reported):
            missing.append(f)
    assert not missing, f"no result reported for: {missing}\n{proc.stdout[-3000:]}"
    print(f"P1 dynamic: {len(sites)} sites, {len(nodes)} node ids, {len(files)} files, "
          f"{len(cases)} results, {elapsed:.1f} s")
