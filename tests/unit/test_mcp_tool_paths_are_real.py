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

SKIPS LOUDLY when miStudio is absent: "unverified" is not "verified clean" — and
**FAILS instead of skipping under `MILLM_REQUIRE_CROSS_REPO_CHECKS=1`**, which is how
a co-release is proved. That flag existed for the sibling consistency guard and this
file did not read it, so the one check meant to make the cross-repo gate mandatory
silently stayed optional here: setting it and pointing MISTUDIO_REPO at nothing
reported `6 passed, 13 skipped`. A gate that can be satisfied by removing the other
repo is not a gate.
"""

import os
import re
from pathlib import Path

import pytest


def _absent(reason: str) -> None:
    """Skip, or FAIL when the cross-repo checks are required.

    ⚠ The distinction is the whole point of `MILLM_REQUIRE_CROSS_REPO_CHECKS`: in a
    developer's tree the sibling repo may genuinely be absent and skipping is right; in
    CI, or in a release check, "the other repo was missing" must be a failure rather
    than a green run. Both halves of a co-release have to be present for either to be
    verified.
    """
    if os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1":
        pytest.fail(
            "MILLM_REQUIRE_CROSS_REPO_CHECKS=1 but the cross-repo guard cannot run: "
            + reason
        )
    pytest.skip(reason)

MISTUDIO = Path(
    os.environ.get(
        "MISTUDIO_REPO", str(Path(__file__).resolve().parents[3] / "miStudio")
    )
)
TOOLS_DIR = MISTUDIO / "backend" / "src" / "mcp_server" / "tools"

#: One entry per cross-repo tool surface: the miStudio module, the miLLM routers
#: it may call, a floor on the number of calls the extraction must find, and a
#: few served routes pinned by hand.
#:
#: ⚠ **PARAMETRISED RATHER THAN COPIED.** A sibling file for the probe surface
#: would be a second extraction implementation, and the first time either was
#: fixed the two would drift — which is the failure mode this repo keeps
#: finding. One extraction, two surfaces.
SURFACES = {
    "millm_circuits": {
        "module": "millm_circuits.py",
        "routers": {
            "millm/api/routes/management/circuits.py": "/api/circuits",
            "millm/api/routes/management/circuit_sensing.py": "/api/circuit-sensing",
        },
        "min_calls": 14,
        "pins": [
            ("GET", "/api/circuits/claims"),
            ("POST", "/api/circuits/claims/release"),
            ("GET", "/api/circuit-sensing/status"),
        ],
    },
    # v1.6 — Feature 24. Added with the routes, not after them: five of those
    # routes did not exist when phase 7 was marked done, and the arming
    # service, the identity gate, the parity engine and the k-sparse slice
    # therefore had no caller at all.
    "millm_probes": {
        "module": "millm_probes.py",
        "routers": {
            "millm/api/routes/management/probes.py": "/api/probes",
        },
        "min_calls": 6,
        "pins": [
            ("POST", "/api/probes/{probe_id}/arm"),
            ("POST", "/api/probes/hub/import"),
            ("GET", "/api/probes/status"),
        ],
    },
    # ⚠ THESE FOUR WERE ADDED 2026-09-27 AND WERE NEVER PATH-CHECKED BEFORE.
    # The completeness test below went red the moment it was written, naming all
    # four — a guard covers what it is pointed at, and for two years this one was
    # pointed at `millm_circuits` alone. Every path in these modules could have
    # been wrong from the start and 404'd in production, with miStudio's caller
    # test agreeing with the tool because both read the same wrong literal.
    "millm_clusters": {
        "module": "millm_clusters.py",
        "routers": {"millm/api/routes/management/clusters.py": "/api/clusters"},
        "min_calls": 7,
        "pins": [
            ("GET", "/api/clusters/hub/search"),
            ("POST", "/api/clusters/import"),
        ],
    },
    "millm_models": {
        "module": "millm_models.py",
        "routers": {
            "millm/api/routes/management/models.py": "/api/models",
            "millm/api/routes/system/health.py": "/api/health",
        },
        "min_calls": 8,
        "pins": [
            ("POST", "/api/models/{model_id}/load"),
            ("GET", "/api/health/inference"),
        ],
    },
    "millm_runtime": {
        "module": "millm_runtime.py",
        "routers": {
            "millm/api/routes/management/profiles.py": "/api/profiles",
            "millm/api/routes/management/clusters.py": "/api/clusters",
            "millm/api/routes/system/health.py": "/api/health",
        },
        "min_calls": 5,
        "pins": [
            ("GET", "/api/health/detailed"),
            ("PUT", "/api/clusters/active/intensity"),
        ],
    },
    "millm_sensing": {
        "module": "millm_sensing.py",
        "routers": {"millm/api/routes/management/sensing.py": "/api/sensing"},
        "min_calls": 5,
        "pins": [
            ("GET", "/api/sensing/status"),
            ("PUT", "/api/sensing/{profile_id}/config"),
        ],
    },
    # v1.11 — Feature 26. The batch routes JOIN the served set now; miStudio's batch tools
    # (034 phase 6) are built against §4f afterwards, so the module may not exist yet.
    "millm_batches": {
        "module": "millm_batches.py",
        "routers": {
            "millm/api/routes/openai/files.py": "/v1",
            "millm/api/routes/openai/batches.py": "/v1",
        },
        "min_calls": 5,
        "pins": [
            ("POST", "/v1/files"),
            ("GET", "/v1/files/{file_id}/content"),
            ("POST", "/v1/batches"),
            ("POST", "/v1/batches/{batch_id}/cancel"),
            ("POST", "/v1/batches/{batch_id}/lease"),
        ],
        "pending": "miStudio 034 phase 6 builds the batch tools after this contract section",
    },
}

SURFACE_IDS = sorted(SURFACES)

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


def _served_routes(surface: str) -> set[tuple[str, str]]:
    """(METHOD, path) pairs this server actually serves for one surface."""
    repo = Path(__file__).resolve().parents[2]
    found: set[tuple[str, str]] = set()
    for rel, prefix in SURFACES[surface]["routers"].items():
        text = (repo / rel).read_text()
        for m in re.finditer(
            r'@router\.(get|post|put|delete)\(\s*\n?\s*"([^"]*)"', text
        ):
            path = prefix + m.group(2)
            found.add((m.group(1).upper(), path))
            # A `:path` converter is an implementation detail of the converter,
            # not of the address. Record both spellings so a tool may write
            # either.
            stripped = re.sub(r"\{(\w+):[^}]+\}", r"{\1}", path)
            found.add((m.group(1).upper(), stripped))
    return found


def _tool_calls(surface: str) -> set[tuple[str, str]]:
    """(METHOD, path) pairs one surface's MCP tools issue, params normalised."""
    module = TOOLS_DIR / SURFACES[surface]["module"]
    if not module.exists() and SURFACES[surface].get("pending") and TOOLS_DIR.exists():
        # A tool surface not built YET is not a missing checkout: skip with the reason even
        # under MILLM_REQUIRE_CROSS_REPO_CHECKS, and let the served-routes pins below still run.
        pytest.skip(f"{surface}: {SURFACES[surface]['pending']}")
    if not module.exists():
        _absent(
            f"miStudio checkout not found at {MISTUDIO} — the {surface} MCP "
            "tool module lives there and this guard cannot run without it. Set "
            "MISTUDIO_REPO. A cross-repo guard that passes vacuously reports "
            "green for the condition it exists to detect."
        )
    source = module.read_text()
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


@pytest.mark.parametrize("surface", SURFACE_IDS)
class TestEveryToolPathIsServed:
    def test_the_extraction_found_the_tools(self, surface):
        """An empty set passes every assertion below it."""
        calls = _tool_calls(surface)
        floor = SURFACES[surface]["min_calls"]
        assert len(calls) >= floor, (
            f"only extracted {len(calls)} {surface} tool calls (floor {floor}) "
            "— the regex has drifted from the module's call style, so this "
            "guard is checking nothing"
        )

    def test_every_tool_path_exists_on_this_server(self, surface):
        served = _served_routes(surface)
        assert served, "no routes extracted — has the router format changed?"

        missing = sorted(c for c in _tool_calls(surface) if c not in served)
        assert not missing, (
            "MCP tools call routes this server does NOT serve. The caller test "
            "in miStudio asserts each tool matches its DOCUMENTED path, so a "
            "path that was wrong from the start passes there and 404s in "
            f"production: {missing}"
        )

    def test_the_routers_actually_parsed(self, surface):
        """Specificity: if the router regex matched nothing, the test above
        would pass vacuously with an empty `served` set — except it asserts
        non-empty. This pins the shape it expects."""
        served = _served_routes(surface)
        for pin in SURFACES[surface]["pins"]:
            assert pin in served, f"{pin} is not served — the extraction drifted"


def test_every_millm_tool_module_in_mistudio_has_a_surface_here():
    """⚠ A `millm_*` tool module with no entry in `SURFACES` is NEVER path-checked.

    This is the gap that let the 024 arc ship five documented-but-unserved routes:
    a guard covers what it is pointed at, and a hand-maintained pointer list is only
    as good as the list. Driven off the directory so a module added later is caught
    the moment it exists.
    """
    if not TOOLS_DIR.exists():
        _absent(f"miStudio checkout not found at {MISTUDIO} — set MISTUDIO_REPO")
    on_disk = {
        p.name
        for p in TOOLS_DIR.glob("millm_*.py")
        if "millm." in p.read_text()  # it issues HTTP calls at all
    }
    covered = {spec["module"] for spec in SURFACES.values()}
    uncovered = sorted(on_disk - covered)
    assert not uncovered, (
        f"these miStudio millm_* tool modules issue HTTP calls that NOTHING here "
        f"checks against the served routes: {uncovered}. Add a SURFACES entry, or "
        "their paths can be wrong from the start and 404 in production."
    )
