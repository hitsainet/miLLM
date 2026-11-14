"""T15: every agent hint comes from `miserver/core/next_tools.py`, and the published copy equals it
(040 FR-040.17, FR-040.35; ID-3). miWorkbench 042 reads `docs/schemas/next-tool-hints.json` and
checks each name against its live MCP registry (BR-071)."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from miserver.core import next_tools
from tests.support.env_flags import flag
from tests.support.package_root import package_path, package_rel

ROOT = Path(__file__).resolve().parents[3]
PUBLISHED = ROOT / "docs" / "schemas" / "next-tool-hints.json"

#: Literal hints left in place on purpose (relpath, value), each removed the day it is converted.
DEFERRED: set[tuple[str, str]] = set()  # empty since D53 merged (2026-11-08)


def render() -> str:
    return json.dumps({"version": 1, "next_tools": sorted(next_tools.ALL)}, indent=2) + "\n"


def sites() -> list[tuple[str, int, ast.AST]]:
    """Every `next_tool = <value>` assignment and `next_tool=<value>` keyword in `miserver/`."""
    files = subprocess.run(["git", "ls-files", "--", package_rel()], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    out = []
    for rel in files:
        if not rel.endswith(".py") or rel == package_rel("core", "next_tools.py"):
            continue
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(node, ast.ClassDef):  # a class attribute: the error's hint
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == "next_tool" for t in stmt.targets):
                        out.append((rel, stmt.lineno, stmt.value))
            elif isinstance(node, ast.keyword) and node.arg == "next_tool":
                out.append((rel, node.value.lineno, node.value))
    return out


def _is_constant_ref(value: ast.AST) -> bool:
    return (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
            and value.value.id == "next_tools" and value.attr in next_tools.__all__)


def test_every_site_uses_a_next_tools_constant():
    found = sites()
    assert len(found) >= 26, f"found only {len(found)} next_tool sites: the walk is failing open"
    bad, deferred_seen = [], set()
    for rel, line, value in found:
        if isinstance(value, ast.Constant) and value.value is None:
            continue  # the base class default (no hint)
        if _is_constant_ref(value):
            continue
        if isinstance(value, ast.Constant) and (rel, value.value) in DEFERRED:
            assert value.value in next_tools.ALL, (rel, value.value)
            deferred_seen.add((rel, value.value))
            continue
        bad.append(f"{rel}:{line}: {ast.unparse(value)}")
    assert not bad, "use a miserver.core.next_tools constant (040 FR-040.17):\n  " + "\n  ".join(bad)
    assert deferred_seen == DEFERRED, f"a deferral no longer applies: {DEFERRED - deferred_seen}"


def test_published_file_equals_the_module():
    if flag("WRITE_CONTRACTS") == "1":
        PUBLISHED.write_text(render(), encoding="utf-8")
    assert PUBLISHED.is_file(), f"{PUBLISHED} is missing; run with MISERVER_WRITE_CONTRACTS=1"
    assert PUBLISHED.read_text(encoding="utf-8") == render(), (
        f"{PUBLISHED.name} differs from miserver/core/next_tools.py; regenerate it "
        "(MISERVER_WRITE_CONTRACTS=1) — miWorkbench reads the file")


def test_every_constant_is_used():
    used = {value.attr for _r, _l, value in sites() if _is_constant_ref(value)}
    used |= {name for name in next_tools.__all__ if (None, getattr(next_tools, name)) in
             {(None, v) for _r, v in DEFERRED}}
    assert used == set(next_tools.__all__), set(next_tools.__all__) - used


def test_fourteen_names_from_p4():
    """P4-H (FR-040.35): miWorkbench's MCP rename is live, so every hint names a `miserver_*` tool."""
    assert len(next_tools.ALL) == len(next_tools.__all__) == 14
    assert all(name.startswith("miserver_") for name in next_tools.ALL), "miserver_* from P4-H"
    tree = ast.parse(package_path("core", "next_tools.py").read_text())
    annotated = {n.target.id for n in tree.body
                 if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    assert not annotated & set(next_tools.__all__), "plain assignments (F-6)"
