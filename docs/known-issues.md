# Known issues

Findings that are real, reproducible somewhere, and not yet fixed. Recorded so
the next person meets a note instead of a mystery.

---

*(No open entries.)*

## RESOLVED 2026-09-08 — "create_app returns an unrouted app late in a CI run"

Kept because the wrong diagnosis is the useful part.

**What it looked like:** in CI only, `create_app()` appeared to return an app
with just the four framework defaults, and a route guard reported "route
registration itself is broken". A whole entry was written here about
module-state pollution, with a suspect (a `sys.modules` swap in
`test_gguf_engine.py`) and a plan to diff module identities across the run.

**What it actually was:** nothing was broken. Since **FastAPI 0.141 /
Starlette 1.6**, `include_router` no longer flattens routes into `app.routes`.
It appends one lazy `fastapi.routing._IncludedRouter` with no `.path`, and the
real routes live behind it:

    starlette.routing.Route          path='/openapi.json'
    starlette.routing.Route          path='/docs'
    ...
    fastapi.routing._IncludedRouter  path=None      <- all real routes in here

So `{r.path for r in app.routes if hasattr(r, "path")}` returns the defaults and
nothing else, on an app that routes every request correctly. Reproduced exactly
in a throwaway venv at CI's versions.

**Why it looked CI-only and position-dependent:** CI installs into a clean
environment and resolved fastapi 0.141.1 / starlette 1.6.0; a long-lived local
venv sat on 0.128.0 / 0.50.0, where the old flattening behaviour still applies.
And the tests that "passed at 5% and 16%" use `TestClient` — they make real
requests, which always worked. Only the introspecting guard at 70% looked at
`app.routes`. There was never a transition partway through the run; there were
two different techniques, one valid and one not.

**The fix:** read the SERVED OpenAPI schema — `app.openapi()["paths"]` — never
`app.routes`.

**The part worth remembering:** miStudio had already hit this and written it
down. `test_body_limit.py`, `test_mcp_tools_call_real_routes.py` and
`test_cancel_registry_completeness.py` all carry a comment saying to read the
served schema because included routers are wrapped with no `.path`. Checking
the sibling repo would have cost minutes instead of hours, and no fictional
defect would have been documented here in the meantime.

**Second lesson:** a local environment that has drifted from CI does not just
cause false greens — it invents phantom bugs. The same drift produced three
"1955 passed" reports against a red CI earlier the same day.
