"""The `/api/probes` surface, asserted as a SET and against the FPRD.

⚠ **THIS FILE EXISTS BECAUSE A SUBSET ASSERTION PASSED OVER FIVE MISSING ROUTES.**

`test_probe_events.py` asserted `for expected in (…): assert expected in paths` — seven literals,
every one served. And task 7.5's text said "7 paths", and the module had exactly seven. The count
agreed while the SET was wrong: `POST /{id}/arm`, `POST /{id}/parity`, `GET /hub/search`,
`GET /hub/{repo_id}/definitions` and `POST /hub/import` were all absent, so

* `ProbeArmingService.arm` — four gates, the acknowledgement gate, `PROBE_MAX_ARMED`,
* `check_identity` and `resolve_revision`,
* `ProbeParityEngine.run`,
* `SaeFeatureSlice` — the whole k-sparse probe path,
* `ProbeHubService`

had, between them, no production caller. Every one had passing unit tests, because every test
imported the service directly. **A probe could be imported and never armed.**

Two rules follow, and both are enforced below:

1. **Assert the SET, not a subset.** A membership loop cannot fail for a route that is missing from
   both the code and the list.
2. **Derive the expected set from the SPEC.** A list maintained beside the code drifts with the
   code; the FPRD's §7 is what the feature promised.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FPRD = (
    Path(__file__).resolve().parents[3]
    / "0xcc"
    / "prds"
    / "024_FPRD|Probe_Monitor_Runtime.md"
)

#: The live surface. Kept here as well as parsed from the FPRD so that a change to either one alone
#: is a failure: the code may not grow a route the spec does not promise, and may not lose one.
EXPECTED_PATHS = {
    "/api/probes",
    "/api/probes/import",
    "/api/probes/hub/search",
    "/api/probes/hub/{repo_id}/definitions",
    "/api/probes/hub/import",
    "/api/probes/status",
    "/api/probes/events",
    "/api/probes/events/{event_id}",
    "/api/probes/{probe_id}",
    "/api/probes/{probe_id}/arm",
    "/api/probes/{probe_id}/parity",
    "/api/probes/{probe_id}/disarm",
}


def _served_probe_paths() -> set[str]:
    """Every `/api/probes` path the BUILT app serves.

    ⚠ `app.openapi()["paths"]`, never `app.routes` — this FastAPI version wraps included routers in
    objects with no `.path`, so introspecting `app.routes` reports an empty app that serves fine.
    """
    from fastapi import FastAPI

    from millm.api.routes import register_routes

    app = FastAPI()
    register_routes(app)
    # OpenAPI renders `{repo_id:path}` as `{repo_id}`; normalise so both sides speak one dialect.
    return {
        re.sub(r"\{(\w+):[^}]+\}", r"{\1}", path)
        for path in app.openapi()["paths"]
        if path == "/api/probes" or path.startswith("/api/probes/")
    }


def _fprd_paths() -> set[str]:
    """§7's endpoint list, normalised to what FastAPI serves.

    The FPRD writes `{id}` where the route names its parameter `probe_id`, appends query strings
    and, for `arm`, a request-body sketch. All three are stripped; nothing else is.
    """
    text = FPRD.read_text(encoding="utf-8")
    section = text.split("## 7. API/Integration Specifications", 1)[1]
    section = section.split("## 8.", 1)[0]
    paths: set[str] = set()
    for item in re.findall(r"`([^`]+)`", section):
        match = re.match(r"^(GET|POST|PUT|DELETE|PATCH)\s+(/\S*)", item.strip())
        if not match:
            continue
        path = match.group(2)
        path = path.split("?", 1)[0].split(" ", 1)[0]
        path = re.sub(r"\{(\w+):[^}]+\}", r"{\1}", path)
        path = path.replace("{id}", "{probe_id}")
        full = "/api/probes" + ("" if path == "/" else path)
        paths.add(full.rstrip("/") if full != "/api/probes" else full)
    return paths


class TestTheWholeSurfaceIsServed:
    def test_the_served_set_equals_the_expected_set(self):
        served = _served_probe_paths()
        assert served == EXPECTED_PATHS, (
            f"missing from the app: {sorted(EXPECTED_PATHS - served)}; "
            f"served but not declared: {sorted(served - EXPECTED_PATHS)}"
        )

    def test_the_fprd_and_the_app_agree(self):
        """The spec is the authority. A route the FPRD promises and the app does not serve is a
        capability nobody can reach; one the app serves and the FPRD does not promise is undocumented.
        """
        promised = _fprd_paths()
        assert promised, "parsed no endpoints out of the FPRD — the parser, not the app, is broken"
        served = _served_probe_paths()
        assert promised == served, (
            f"promised and not served: {sorted(promised - served)}; "
            f"served and not promised: {sorted(served - promised)}"
        )


class TestTheGatesHaveACaller:
    """Reachability of the gates THEMSELVES, not merely of a path string.

    A route that exists and never reaches the arming service would satisfy the tests above.

    ⚠ **SCOPED TO ONE HANDLER, NOT THE MODULE.** The first version of this guard walked the whole
    module's AST, and two mutations survived it: bypassing `loaded_identity` in `arm_probe` left the
    call in `check_parity`, and dropping `build_probe_encoder` from `arm_probe` left it there too.
    A module-wide scan was satisfied by the wrong occurrence — the same trap this estate has now hit
    in five separate arcs. The calls are asserted inside the function that must make them.
    """

    ROUTE_MODULE = "millm/api/routes/management/probes.py"

    @staticmethod
    def _calls_in_function(module_path: str, function: str) -> set[str]:
        """Every name called inside ONE function body, including its nested awaits."""
        import ast

        tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))
        target = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
                target = node
                break
        assert target is not None, f"{function} is not defined in {module_path}"
        names: set[str] = set()
        for node in ast.walk(target):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    names.add(func.id)
                elif isinstance(func, ast.Attribute):
                    names.add(func.attr)
        return names

    @pytest.mark.parametrize(
        ("handler", "called"),
        [
            # The arm route is the ONLY caller of the four gates.
            ("arm_probe", "arm"),
            ("arm_probe", "loaded_identity"),
            ("arm_probe", "build_parity_forward"),
            ("arm_probe", "build_probe_encoder"),
            # The on-demand parity route re-runs the engine without arming.
            ("check_parity", "run"),
            ("check_parity", "loaded_identity"),
            ("check_parity", "armed_probe_from_row"),
            # The hub routes reach the hub service, not a second HTTP client.
            ("hub_search", "search"),
            ("hub_definitions", "list_definitions"),
            ("hub_import", "fetch_definition"),
            ("hub_import", "import_definition"),
        ],
    )
    def test_the_handler_calls_it(self, handler: str, called: str):
        calls = self._calls_in_function(self.ROUTE_MODULE, handler)
        assert called in calls, (
            f"{handler}() never calls {called}() — that capability is unreachable through it"
        )

    def test_the_arm_route_stores_the_operator_acknowledgement(self):
        """The ack must travel to the service, or the gate is decorative.

        Passing `acknowledge_below_rung2=False` unconditionally would leave a route that refuses
        every weak probe and an acknowledgement an operator can never give.
        """
        source = Path(self.ROUTE_MODULE).read_text(encoding="utf-8")
        assert "acknowledge_below_rung2=request.acknowledge_below_rung2" in source

    def test_the_arming_service_is_reached_through_the_bridge_not_reimplemented(self):
        """The bridge supplies the model; it must not re-run the gates itself.

        Two implementations of the gates is how one gets fixed and the other does not.
        """
        import ast

        tree = ast.parse(Path("millm/services/probe_arm_bridge.py").read_text(encoding="utf-8"))
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "check_identity" not in calls
