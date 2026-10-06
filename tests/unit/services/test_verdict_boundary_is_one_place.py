"""P-03 / FR-27.10: `score >= threshold` lives in ONE place in `millm/` — `_verdict_for`.

A second comparison of a probe score with a threshold anywhere in the server is a second definition
of what a verdict means, and two definitions drift: miLLM fired on `>` until 2026-10-03 while
miStudio fired on `>=`, and an input exactly on the bar got opposite verdicts in the two repos.

The guard walks the AST for ORDERING COMPARISONS (`>`, `>=`, `<`, `<=`) where one side names a
threshold and the other a score or a value, and asserts the set of functions holding one is EXACTLY
`{ProbeRequestContext._verdict_for}` — exactly, so a walker that finds nothing fails rather than
passing. Comments and strings cannot satisfy it.

Mutation this catches (FTID §8, M12b): a second such comparison added anywhere in `millm/`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import millm

ROOT = Path(millm.__file__).resolve().parent
ORDERING = (ast.Gt, ast.GtE, ast.Lt, ast.LtE)


def _names(node: ast.AST) -> set[str]:
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id.lower())
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr.lower())
    return out


def _is_verdict_compare(node: ast.Compare) -> bool:
    if not any(isinstance(op, ORDERING) for op in node.ops):
        return False
    sides = [node.left, *node.comparators]
    names = [_names(side) for side in sides]
    has_threshold = any(any("threshold" in n for n in side) for side in names)
    has_score = any(any(("score" in n or n == "value") and "threshold" not in n for n in side)
                    for side in names)
    return has_threshold and has_score


def flagged_functions(root: Path = ROOT) -> set[str]:
    found: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack: list[tuple[ast.AST, str]] = [(tree, "")]
        while stack:
            node, owner = stack.pop()
            for child in ast.iter_child_nodes(node):
                name = owner
                if isinstance(child, ast.ClassDef):
                    name = child.name
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f"{owner}.{child.name}" if owner else child.name
                if isinstance(child, ast.Compare) and _is_verdict_compare(child):
                    rel = path.relative_to(root).as_posix()
                    found.add(f"{rel}::{owner}")
                stack.append((child, name))
    return found


def test_the_only_score_threshold_comparison_is_verdict_for():
    assert flagged_functions() == {
        "services/probe_runtime.py::ProbeRequestContext._verdict_for"
    }


def test_the_walker_finds_a_comparison_and_ignores_text(tmp_path):
    (tmp_path / "a.py").write_text(
        "def f(score, threshold):\n"
        "    '''score >= threshold in a docstring does not count'''\n"
        "    # value > threshold in a comment does not count\n"
        "    return score > threshold\n"
    )
    assert flagged_functions(tmp_path) == {"a.py::f"}
