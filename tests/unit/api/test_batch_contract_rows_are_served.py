"""Feature 26 task 8.2: every route §4f of `docs/mcp-contract.md` documents is SERVED.

Parsed from the contract's own table and checked against the live app's OpenAPI document, in both
directions: a documented route nobody serves, and a served batch route nobody documented, both
fail. miStudio builds its batch tools against this section only (034 phase 6).
"""

from __future__ import annotations

import re
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[3] / "docs" / "mcp-contract.md"


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
    head = CONTRACT.read_text().splitlines()[2]
    assert "Version:** 1.11" in head and "Feature 26" in head
