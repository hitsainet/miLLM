"""Review of 040 P0 (2026-11-08): two T03 exercises that read a proxy instead of the served name.

* DR-08: `test_frozen_headers` derives the header set with a regex over SOURCE TEXT, comments and
  docstrings included. Renaming the code site of `X-miLLM-Probe-Verdicts` (control M03) or of the
  streaming `X-miLLM-Seed` (M02) left T03 green: a comment still spelled the name. Here the names
  come from string CONSTANTS in code (docstrings excluded), and no header constant may carry another
  brand.
* DR-22: `test_prometheus_metric_families` reads only the `# TYPE` lines; the SAMPLE lines are what
  Prometheus stores. Renaming the sample of `millm_requests_active` (M06) left T03 green.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from tests.unit.legacy.exercises.test_frozen_headers import FROZEN
from tests.unit.legacy.exercises.test_frozen_telemetry_identity import FAMILIES
from tests.support.package_root import package_rel

ROOT = Path(__file__).resolve().parents[3]
HEADER = re.compile(r"^X-mi[A-Za-z]+-[A-Za-z][A-Za-z-]*$", re.IGNORECASE)
#: Empty: the one header that existed only in a comment (the dropped `…-Probe-Stream` opt-in) is gone.
COMMENT_ONLY: set[str] = set()


def _docstring_ids(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def _header_constants() -> set[str]:
    files = subprocess.run(["git", "ls-files", "--", package_rel()], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    found: set[str] = set()
    for rel in files:
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        skip = _docstring_ids(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in skip and HEADER.match(node.value)):
                found.add(node.value)
    return found


def test_header_constants_are_the_frozen_names():
    found = _header_constants()
    assert "X-miLLM-Model" in found, "the scan sees nothing: it is looking in the wrong place"
    other_brand = sorted(h for h in found if not h.startswith("X-miLLM-"))
    assert not other_brand, f"a frozen X-miLLM-* header was renamed in code: {other_brand}"
    assert found == FROZEN - COMMENT_ONLY, (sorted(FROZEN - COMMENT_ONLY - found),
                                            sorted(found - FROZEN))


def test_prometheus_sample_lines_use_the_frozen_families():
    from miserver.main import create_app

    text = TestClient(create_app()).get("/api/health/metrics/prometheus").text
    samples = {re.split(r"[{\s]", line, maxsplit=1)[0]
               for line in text.splitlines() if line and not line.startswith("#")}
    assert samples, "no sample line read"
    unknown = sorted(s for s in samples
                     if not any(s == f or s.startswith(f + "_") for f in FAMILIES))
    assert not unknown, f"sample lines outside the frozen families: {unknown}"
    assert {"millm_requests_active", "millm_requests_total", "millm_uptime_seconds"} <= samples
