"""Feature 26 task 8.2: every route §4f of `docs/mcp-contract.md` documents is SERVED.

Parsed from the contract's own table and checked against the live app's OpenAPI document, in both
directions: a documented route nobody serves, and a served batch route nobody documented, both
fail. miStudio builds its batch tools against this section only (034 phase 6).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CONTRACT = Path(__file__).resolve().parents[3] / "docs" / "mcp-contract.md"

#: SOURCE-ONLY, and skipped LOUDLY where absent — the same split as
#: `test_recalibration_doctrine_is_consistent.py`. `sync-to-clean` strips everything under
#: `docs/` except `schemas/`, and the MIRROR is where the backend image is built: this file
#: first shipped without the skip, failed the mirror's Backend Tests, and the image gate
#: refused to build Feature 26 at all (2026-10-07). The contract cannot be checked where it is
#: not present; "unverified here" is the honest result, not a pass and not a red build.
pytestmark = pytest.mark.skipif(
    not CONTRACT.exists(),
    reason="docs/mcp-contract.md is not in this checkout — the public mirror strips docs/ "
    "except schemas/, and images are built there. Checked in the source repo.",
)


def _section_rows() -> set[tuple[str, str]]:
    text = CONTRACT.read_text()
    start = text.index("### 4f. Batch API")
    end = text.index("## 5. Error codes", start)
    rows = re.findall(r"^\| `(GET|POST|PUT|DELETE) (/[^`]+)`", text[start:end], re.MULTILINE)
    assert rows, "the §4f table parsed to nothing; the guard would assert nothing"
    return {(m, re.sub(r"\{id\}", "{X}", p)) for m, p in rows}


def _served() -> set[tuple[str, str]]:
    from millm.main import create_app

    paths = create_app().openapi()["paths"]
    out = set()
    for path, ops in paths.items():
        if path.startswith(("/v1/files", "/v1/batches")):
            for method in ops:
                out.add((method.upper(), re.sub(r"\{(file_id|batch_id)\}", "{X}", path)))
    return out


def test_every_documented_batch_route_is_served_and_every_served_one_documented():
    documented, served = _section_rows(), _served()
    assert documented == served, (
        f"documented but not served: {sorted(documented - served)}; "
        f"served but not documented: {sorted(served - documented)}"
    )
    assert len(served) == 10


def test_the_contract_version_names_the_batch_api():
    """The Batch API arrived in contract v1.11; any later version must still name Feature 26.

    Compares the CURRENT version (the last `**Version:**` on the status line) numerically, so
    an additive bump does not break this test.
    """
    head = CONTRACT.read_text().splitlines()[2]
    current = head.rsplit("**Version:**", 1)[1].split()[0]
    assert tuple(int(p) for p in current.split(".")) >= (1, 11), current
    assert "Feature 26" in head
