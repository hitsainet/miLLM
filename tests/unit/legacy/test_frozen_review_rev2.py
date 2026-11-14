"""Rev2 review of 040 P0 (2026-11-08): two frozen exercises that a rename still slips past.

* DR-08 (control F1): `X-miLLM-Seed` is sent from two sites (`provenance.py` and the streaming branch
  of `chat.py`). Renaming ONE site out of the `X-mi*` shape (`"X-Seed"`) leaves the set of
  `X-mi*` constants unchanged, so the set-equality check stays green while streaming clients lose
  the header. Every `X-` header constant in `miserver/` must be a frozen name or a listed non-brand one.
* DR-22 (control F4): a sample line is accepted when it STARTS WITH a family name plus `_`, so
  `millm_model_loaded_flag 1` passes while the stored series `millm_model_loaded` disappears. Every
  family is a gauge or a counter (no histogram or summary suffixes exist), so a sample's name must
  be a family name exactly.
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
#: `X-` constants that are not miLLM headers: the request-id header and an OpenAPI extension key.
NOT_BRAND = {"X-Request-ID", "x-approval-action"}


def _x_constants() -> set[str]:
    files = subprocess.run(["git", "ls-files", "--", package_rel()], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    found: set[str] = set()
    for rel in files:
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        docs = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docs.add(id(body[0].value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
                    and re.match(r"^[Xx]-[A-Za-z0-9-]+$", node.value)):
                found.add(node.value)
    return found


def test_every_x_header_constant_is_frozen_or_listed():
    found = _x_constants()
    assert "X-miLLM-Model" in found, "the scan sees nothing: it is looking in the wrong place"
    stray = sorted(found - FROZEN - NOT_BRAND)
    assert not stray, f"an X- header constant that is neither frozen nor listed: {stray}"


def test_every_sample_is_exactly_a_frozen_family():
    from miserver.main import create_app

    text = TestClient(create_app()).get("/api/health/metrics/prometheus").text
    types = {line.split()[2]: line.split()[3] for line in text.splitlines() if line.startswith("# TYPE ")}
    assert types and set(types.values()) <= {"gauge", "counter"}, types
    samples = {re.split(r"[{\s]", line, maxsplit=1)[0]
               for line in text.splitlines() if line and not line.startswith("#")}
    assert samples, "no sample line read"
    assert samples <= FAMILIES, sorted(samples - FAMILIES)
