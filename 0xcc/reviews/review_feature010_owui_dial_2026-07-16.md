# Multi-Agent Review — Feature 10: OWUI Cluster Dial

**Date:** 2026-07-16 · **Scope:** feature (commits `566c774..HEAD`, miLLM)
**Goal gate:** ≥10 findings per round, 3 rounds.

---

## Round 1 (multi-angle /code-review, 2 parallel finder agents) — 24 raw / 19 unique: 15 fixed, 4 documented

**Critical, fixed:**
1. **Streaming steering leak** — apply ran at the top of the semaphore block but tokenization,
   `_check_context_length` (raises) and `thread.start()` sat BEFORE the try/finally that restores;
   a context-length error with `dial=off` left steering globally disabled (or 2× overdriven with
   `dial=2.0`) for all users. → setup window wrapped in restore-and-reraise; regression test
   simulates a tokenizer explosion and asserts restore.
2. **Named empty-steering profile + dial fell through to the live-values base** — the request got
   steered by a profile the caller never named (and the cluster gate was skipped for
   empty-membership clusters, since it lived inside the steering-truthy branch). → gate hoisted
   before ALL decisions (pre-010 ordering restored); named-empty is a logged no-op with or
   without a dial.
3. **Dial could re-enable operator-disabled steering** — dial-only requests over the active profile
   called `enable_steering(True)` unconditionally. → dial-only requests never enable disabled
   steering (explicitly named profiles keep pre-010 enable semantics).
4. **Numeric λ bypassed the authored `intensity_range` on unauthenticated /v1** — management
   `set_intensity` enforces the declared safe envelope; the request dial accepted any λ in [0,2].
   → numeric λ clamps into the cluster's declared range at apply (dial-to-0 exempt, matching
   set_intensity), logged when it engages.
5. **/v1 validation errors were raw pydantic 422 dumps** (errors.pydantic.dev URLs to OpenAI-SDK
   clients; EC-10.3 wanted OpenAI-style 400). → `RequestValidationError` handler: /v1 → 400
   `{"error":{message,type,param,code}}` (benefits every /v1 param, not just the dial); non-/v1
   keeps FastAPI's 422. 14 pre-existing test assertions updated.

**Also fixed:** echo header honesty (suppressed when no SAE attached / named profile missing;
DB failure degrades to no-header instead of 500ing symbolic-dial requests); swapped (`[hi,lo]`)
or garbage authored ranges normalized/fallback instead of inverting min/max or 500ing;
OWUI filter dropped dict-shaped `__user__["valves"]` (getattr on dict) — reads both shapes now;
free-text dial valve replaced with a Literal dropdown + typed `custom_lambda` (typos
unrepresentable, spec'd DIAL-F2 shape); per-model enablement guidance for mixed-provider OWUI
(strict upstreams 400 on unknown fields); routing condition de-triplicated into
`_has_steering_override` with an honest `per_request_steering_override` log reason and a
call-site spy test; API reference updated (it described Feature 10 as unshipped in the same
commit range that shipped it — field row, header, error row, CBM note added); type annotations
on the new service functions; hardcoded logger name → `__name__`; non-finite λ pinning tests
(nan currently rejected only via the chained-comparison form — now regression-pinned).

**Documented (not fixed this round):**
- Global `set_intensity`/`activate` mutating SAE state is not serialized with the inference
  queue — a concurrent global dial change during an in-flight request can be clobbered by the
  request's restore. Pre-existing class of race (any activation during generation), needs
  queue-serialized steering mutations as its own work item.
- Symbolic echo TOCTOU (profile switch while queued) — inherent to headers-before-body;
  documented in the API reference.
- Symbolic dials cost two profile reads (route echo + apply); the second is inside the semaphore
  and load-bearing, the first degrades gracefully — acceptable.
- `steering_intensity` remains chat-completions-only (text completions lack `profile` too);
  `_has_steering_override` is deliberately uniform so adding the fields later inherits routing.

## Round 2 (post-fix verification + fresh angles, 1 finder agent + inline) — 11 findings: 8 fixed, 3 documented

**All 8 round-1 fix verifications passed** (streaming restore has no double-restore/GeneratorExit
hole; gate hoisting is manual-profile-safe; the 400 handler is registration-order-proof and
admin-ui-neutral; filter valve matrix verified; manual anchors resolve).

**Fixed:**
1. **The R1 clamp was wrong in the down direction** — it clamped sub-floor λ UP to the authored
   floor (a 0.05 request against range [0.5, 1.5] applied at 0.5 — 10× stronger than asked, on a
   dial the user was turning toward off). set_intensity's authoritative bounds are [0, hi] →
   ceiling-only cap now; sub-floor passthrough pinned.
2. **The numeric echo lied whenever the cap engaged** (header echoed the raw λ, apply capped it).
   → echo and apply now share ONE pure decision core, `_plan_effective_intensity` (deep fix: they
   cannot drift), and the echo mirrors the ceiling cap.
3. **Echo still lied on no-op paths** (dial-only with steering disabled / empty base / named-empty
   profile) → the shared planner returns None for every no-op → header suppressed; docs updated.
4. **Three divergent interpreters of `budget.intensity_range`** (v1 contract enforces no ordering):
   /v1 swapped descending pairs, management `_intensity_bounds` didn't (same document produced
   envelope [0.5,1.5] on /v1 vs [0,0.5] on the management API), `_range_warnings` read `rng[1]`
   blind. → single shared parser `steering_range.declared_intensity_range` (normalized, guarded);
   all three consumers rewired.
5. **Management API still 500'd on garbage ranges** (the R1 hardening was /v1-only;
   `_intensity_bounds` did bare `float()`) → fixed via the shared parser.
6. **Streaming + bad named profile aborted the stream after a committed 200** instead of the
   documented 404 → pre-stream `ensure_profile_exists` check in the route (mirrors the existing
   pre-stream QueueFullError check); integration test.
7. **`PROFILE_NOT_FOUND`/`INVALID_FEATURE_INDEX` missing from ERROR_STATUS_MAP** — /v1 branded
   caller errors as `server_error` (pre-existing, exposed by the new test) → rows added.
8. **Stale `apply_request_profile` identifier** in the restore docstring → fixed.
   **Manual wording shipped the wrong (R1) clamp semantics** → corrected to ceiling-cap +
   sub-floor honored + λ=0-skips-validation note (inline finding).

**Documented:**
- λ=0 with a named empty profile disables live steering while λ=0.5 no-ops — FPRD DIAL-A3's
  unconditional "λ=0 disables" wins; discontinuity documented in the API reference, disable log
  now names the base.
- λ=0 deliberately skips per-feature index validation (nothing is applied) — pinned by test,
  documented.
- 0xcc design docs retain the historical `_apply_request_profile` name in their "generalize
  this" instructions — accurate as specs, left unchanged.

## Round 3 (/review, 4 perspectives) — 16 findings: 12 fixed, 4 documented

**Critical, fixed (all three empirically confirmed by the reviewer):**
1. **Hostile authored ranges bypassed the [0,2] envelope** — `intensity_range [0.5, 9]` let symbolic
   `max` apply λ=9; a negative floor let `min` apply SIGN-INVERTED steering; an all-negative range
   made the cap FORCE a positive dial negative — all on unauthenticated /v1.
   → `declared_intensity_range` intersects with [0,2] (outside → config fallback); consumer
   contracts documented in its docstring.
2. **Apply MERGED onto live steering** (`set_steering_batch` merges; restore clears, apply didn't) —
   a named profile was superimposed on operator-set values: response steered by BOTH.
   → `clear_steering()` before `set_steering_batch`; pinned by a stateful-fake test + a
   dial×named-profile concurrency interleave.
3. **Stored intensity 0 applied an all-zero-ENABLED batch while the header echoed "0"** — zero
   tensors still fire apply_steering per token and report steering on; DIAL-A3 violated; a live
   echo/apply drift. → planner returns 0.0 uniformly for effective-zero; apply branches on
   `effective`, not raw λ.

**Also fixed:** cap fell back to nothing for clusters WITHOUT an authored range (management rejects
λ>config-max; /v1 sailed through) → config-envelope fallback for cluster rows (manual profiles keep
the schema's [0,2] as documented); symbolic resolution + cap were still duplicated at both
consumers → absorbed INTO `_plan_effective_intensity` (keyword-only params — three bool-ish
positionals invited transposition); streaming gate/index errors aborted a committed 200 → OpenAI
error SSE event + `[DONE]` instead; echo failure logged a full traceback per request
(unauthenticated log-flood lever) → single-line warning; three sequential profile reads on
streaming dial+profile requests → echo resolution doubles as the pre-commit 404 check
(`ensure_named_profile` flag); the /v1 400 handler asserts nothing under `python -O` → isinstance
re-raise, and multi-error requests now say "(and N more validation errors)"; the filter gained a
`server` valve position (US-10.1: once the operator sets a default, users had no way to say
"server state governs") + documented replace-semantics for pre-existing body fields; filter test
file in the suite (17 tests — dict-vs-model valves, precedence, degradation were pinned only by an
ad-hoc shell run); Troubleshooting row for the six silent dial no-op causes; echo/apply parity
matrix (12 states), direct planner table tests, at-ceiling/cap-log/noop-log mutation pins,
string-typed n_features gate pin.

**Documented:**
- `ensure_profile_exists`/apply-time TOCTOU (profile deleted between pre-check and apply) →
  mid-stream SSE error event is now the defined behavior for that window.
- Management vs /v1 post-processing of the shared range parse deliberately differ ([0,hi] dial vs
  min/max positions) — contract documented in `declared_intensity_range`.
- InferenceService accumulating repository plumbing (echo/apply each open sessions) — a
  ProfileResolver service is the right home; deferred as refactor debt (three call sites named).
- First-error bias in /v1 validation responses matches OpenAI's own API; the count suffix
  mitigates round-trips.

## Perspective summaries
- **Product:** FR-10.1..10.4, DIAL-A1..A7 (A3 fixed this round), F1..F4, US/EC verified one-by-one;
  US-10.1's "default sends no field" restored via the `server` valve; §9 criterion 1 (observable
  output difference) rides the post-deploy E2E (task 4.3), the rest verified pre-deploy.
- **QA:** unauthenticated surface now bounded end-to-end: [0,2] envelope ∩ authored range ∩ config
  fallback; no PII in logs; log-flood lever closed; streaming never aborts silently for steering
  errors.
- **Architect:** ONE decision core owns resolution/cap/no-op for both echo and apply (drift now
  requires editing the planner itself); replace-not-merge asymmetry closed; range parsing has a
  single home with named consumer contracts.
- **Test:** 984 backend tests green; guarantees now asserted behaviorally (parity matrix,
  stateful-fake interleaves, log-event pins, filter suite) — the three R3 criticals each have a
  regression test that fails on the pre-fix code.

## Gate
**SHIP** — 46 findings across 3 rounds (35 fixed, 11 documented), acceptance evidence complete.
Post-deploy E2E (off/min/max on a validated cluster + OWUI walkthrough) rides the increment's
GitOps rollout (task 4.3).
