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
#: Accepts either quote style; `\s*` spans newlines, so a call whose path sits
#: on the following line is matched too.
CALL = re.compile(
    r"millm\.(get|post|put|delete|raw_get)\(\s*f?[\"']([^\"']+)[\"']",
    re.MULTILINE,
)

#: Every call SITE, regardless of how its path is written. R2-15: `CALL` can
#: only parse literal paths, so a call built from a constant, an aliased
#: receiver, or a verb not in the list above is invisible to it — and an
#: extraction that silently drops a call reports green for the exact call it
#: failed to check. Counting sites independently turns that silent miss into a
#: failure that names the unparseable line.
CALL_SITE = re.compile(r"millm\.([a-z_]+)\(", re.MULTILINE)

#: Verbs `CALL_SITE` may see that are not HTTP calls and need no path check.
NON_HTTP_ATTRS = {"close", "aclose"}

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
    source = TOOL_MODULE.read_text()
    calls: set[tuple[str, str]] = set()
    for method, path in CALL.findall(source):
        # `f"/api/circuits/{circuit_id}/activate"` → the router's literal form.
        normalised = re.sub(r"\{[a-z_]+\}", lambda m: m.group(0), path)
        calls.add((VERB[method], normalised))

    # R2-15: every call site must have been PARSED, not merely most of them.
    sites = [v for v in CALL_SITE.findall(source) if v not in NON_HTTP_ATTRS]
    unknown = sorted({v for v in sites if v not in VERB})
    assert not unknown, (
        f"MCP tool calls use verb(s) {unknown}, which this guard cannot map to "
        "an HTTP method — so their paths are NEVER checked against the served "
        "routes. Add them to VERB (or to NON_HTTP_ATTRS if they are not HTTP "
        "calls). Failing loudly beats silently skipping the call."
    )
    parsed = len(CALL.findall(source))
    assert parsed == len(sites), (
        f"{len(sites)} millm.* call sites exist but only {parsed} had a "
        "parseable literal path. The unparsed ones are invisible to this "
        "guard — a typo in them would ship. Rewrite the path as a literal, or "
        "teach CALL to read the form used."
    )
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
