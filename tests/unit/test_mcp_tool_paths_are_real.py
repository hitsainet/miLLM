"""Feature 20 R1-01 — every MCP tool path is a route this server actually serves.

The reachability harness in miStudio asserts each tool issues its DOCUMENTED
method and path. That catches a tool that calls nothing, or that drifts from its
own documentation. It cannot catch a tool whose documented path was WRONG from
the start: the caller test and the tool would agree, and both would be wrong.

This is the miLLM half. It reads the tool module from the sibling repo, extracts
the paths, and checks them against THIS server's routers — the only authority on
what is actually served.

I verified this by hand while building F20 and the verification lived in a shell
session. A check that exists only in someone's terminal is precisely the gap
this feature exists to close, so it is a test.

SKIPS LOUDLY when miStudio is absent: "unverified" is not "verified clean".
"""

import os
import re
from pathlib import Path

import pytest

MISTUDIO = Path(
    os.environ.get(
        "MISTUDIO_REPO", str(Path(__file__).resolve().parents[3] / "miStudio")
    )
)
TOOL_MODULE = (
    MISTUDIO / "backend" / "src" / "mcp_server" / "tools" / "millm_circuits.py"
)

ROUTERS = {
    "millm/api/routes/management/circuits.py": "/api/circuits",
    "millm/api/routes/management/circuit_sensing.py": "/api/circuit-sensing",
}

#: `millm.get("/api/x")` / `millm.raw_get(...)` / `millm.post(f"/api/x/{id}")`
CALL = re.compile(
    r"millm\.(get|post|put|delete|raw_get)\(\s*f?\"([^\"]+)\"", re.MULTILINE
)

VERB = {"get": "GET", "raw_get": "GET", "post": "POST", "put": "PUT",
        "delete": "DELETE"}


def _served_routes() -> set[tuple[str, str]]:
    """(METHOD, path) pairs this server actually serves."""
    repo = Path(__file__).resolve().parents[2]
    found: set[tuple[str, str]] = set()
    for rel, prefix in ROUTERS.items():
        text = (repo / rel).read_text()
        for m in re.finditer(
            r'@router\.(get|post|put|delete)\(\s*\n?\s*"([^"]*)"', text
        ):
            found.add((m.group(1).upper(), prefix + m.group(2)))
    return found


def _tool_calls() -> set[tuple[str, str]]:
    """(METHOD, path) pairs the MCP tools issue, f-string params normalised."""
    if not TOOL_MODULE.exists():
        pytest.skip(
            f"miStudio checkout not found at {MISTUDIO} — the MCP tool module "
            "lives there and this guard cannot run without it. Set "
            "MISTUDIO_REPO. Skipping loudly: a cross-repo guard that passes "
            "vacuously reports green for the condition it exists to detect."
        )
    calls: set[tuple[str, str]] = set()
    for method, path in CALL.findall(TOOL_MODULE.read_text()):
        # `f"/api/circuits/{circuit_id}/activate"` → the router's literal form.
        normalised = re.sub(r"\{[a-z_]+\}", lambda m: m.group(0), path)
        calls.add((VERB[method], normalised))
    return calls


class TestEveryToolPathIsServed:
    def test_the_extraction_found_the_tools(self):
        """An empty set passes every assertion below it."""
        calls = _tool_calls()
        assert len(calls) >= 14, (
            f"only extracted {len(calls)} tool calls — the regex has drifted "
            "from the module's call style, so this guard is checking nothing"
        )

    def test_every_tool_path_exists_on_this_server(self):
        served = _served_routes()
        assert served, "no routes extracted — has the router format changed?"

        missing = sorted(c for c in _tool_calls() if c not in served)
        assert not missing, (
            "MCP tools call routes this server does NOT serve. The caller test "
            "in miStudio asserts each tool matches its DOCUMENTED path, so a "
            "path that was wrong from the start passes there and 404s in "
            f"production: {missing}"
        )

    def test_the_routers_actually_parsed(self):
        """Specificity: if the router regex matched nothing, the test above
        would pass vacuously with an empty `served` set — except it asserts
        non-empty. This pins the shape it expects."""
        served = _served_routes()
        assert ("GET", "/api/circuits/claims") in served
        assert ("POST", "/api/circuits/claims/release") in served
        assert ("GET", "/api/circuit-sensing/status") in served
