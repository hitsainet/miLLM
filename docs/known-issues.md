# Known issues

Findings that are real, reproducible somewhere, and not yet fixed. Recorded so
the next person meets a note instead of a mystery.

---

## `create_app()` returns an unrouted app late in a CI run

**Status:** open, CI-only, not reproduced locally.
**Found:** 2026-09-07, by a route-path guard added the same day.

In GitHub Actions, `millm.main.create_app()` returns a FastAPI app carrying only
the four framework defaults — `/openapi.json`, `/docs`, `/docs/oauth2-redirect`,
`/redoc` — with **no `/api` or `/v1` routes at all**. `register_routes(app)` is
called unconditionally at the end of `create_app`, so something has left the
process in a state where `include_router` adds nothing.

**It is position-dependent, which is the whole finding.** In the same CI run:

| position | test | result |
|---|---|---|
| 5%  | `test_gguf_refused_before_load.py` — `create_app()` + `TestClient`, POSTs `/v1/completions` | PASSED |
| 16% | `test_model_name_disambiguation.py` — `create_app()`, POSTs `/v1/chat/completions` | PASSED |
| 70% | a guard reading `create_app().routes` directly | saw 4 routes |

So routes register correctly early and are gone later. Nothing else in the suite
notices, because every other consumer reaches routes through `TestClient` and
runs before the break.

**What was ruled out** (each measured, not assumed):

* not the environment — CI's `SECRET_KEY`/`DATA_DIR` reproduced locally: 89 routes
* not the pytest version — local upgraded to CI's 9.1.1: still green
* not `pytest-mock` — installed locally to match: still green
* not the CI command — its exact `--ignore` set run locally: 1731 passed
* not a mocked `register_routes` — nothing in `tests/` patches it
* not `llama_cpp` presence alone — CI has it, local does not, and a stub module
  on `PYTHONPATH` did not reproduce it

The remaining suspect is module-state pollution from the `sys.modules` surgery in
`tests/unit/ml/test_gguf_engine.py`, which swaps `llama_cpp` for a broken mock
and re-imports `millm.ml.model_loader`. That test restores what it replaces and
asserts it did, and it runs at 34% — before the break. It is a suspect, not a
conclusion.

**Why it is not simply fixed:** it does not reproduce outside CI, and shipping a
test that fails only there would mean a permanently red suite for a defect
nobody can iterate on.

**What was done instead:** the guard that found it no longer builds a whole app.
It resolves routes through `register_routes(FastAPI())`, which is the thing its
claim is actually about, and carries a negative control that FAILS LOUDLY on an
empty route set — so if this recurs, the message says "route registration itself
is broken" rather than misreporting a correct path as wrong. That misreport is
what cost the investigation: it named the wrong defect convincingly.

**If you pick this up:** the cheap next step is a CI-only diagnostic that dumps
`sys.modules` keys for `millm.*` and the identity of `millm.api.routes` at both
5% and 70% of the run. The divergence between those two points is the answer.
