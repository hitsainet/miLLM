# Feature 14 — Circuit-Aware OWUI Dial: Review Round 3

**Date:** 2026-07-20
**Scope:** commits `4c6e293` (feature) + `cf38fcb` (R1) + `707854b` (R2)
**Method:** `/review all` — four independent perspectives run concurrently (QA/Test, Architect, Product, fresh-eyes correctness), each instructed to attack the previous rounds' fixes
**Findings:** 60 across four agents (14 QA · 14 Architect · 18 Product · 12 correctness) · **19 fixed** · remainder deferred or duplicated

**All four perspectives independently found the same #1 finding.** That convergence is itself the round's strongest signal.

---

## The headline: R2's memo was built on a false premise

R2-07 memoised `_steering_circuit()` on the InferenceService, justified in the docstring I wrote as
*"Memoised per InferenceService instance (which is request-scoped)"*. **It is not.**
`millm/api/dependencies.py:149` decorates `get_inference_service()` with `@lru_cache()` and its own
docstring reads *"Singleton inference service."* I asserted request-scope without checking, and the
memo was therefore written **once per process and never invalidated.**

Two failure directions, both proven by execution:

- **Stale-positive:** an operator deactivates the circuit → every subsequent response still advertises
  `X-miLLM-Circuit-Rung: 2; language="causally validated (edge)"` while apply (which did *not* use the
  memo) correctly no-ops. **Headers claim causal steering; nothing steers. Until restart.**
- **Stale-negative:** the first request arrives before the SAEs attach → `None` is cached permanently →
  steering later applies at λ=2 with **no rung header at all**, silently suppressing the disclosure.

This is R2-02 — the finding the entire round was built around — resurrected by a different route, and
in the stale-positive case worse than the original. The cached object is also a detached SQLAlchemy row
read from an already-closed session.

**Fix:** a `contextvars.ContextVar` plus an explicit `reset_steering_memo()` at the top of the chat
route. A contextvar cannot outlive the request that set it, and the reset is explicit because assuming
a fresh context per request would repeat the exact mistake. The perf win R2 wanted survives.
**Tests:** `TestSteeringMemoIsRequestScoped` — three tests, including one that reuses **one** service
across two simulated requests, which no previous test could express.

**Why the suite couldn't see it:** every dial test builds a fresh `InferenceService.__new__`, and
`tests/integration/conftest.py:39` calls `get_inference_service.cache_clear()` between tests. The memo
was always cold. Third consecutive round where fixtures that agree by construction hid the defect.

---

## Also fixed

### Correctness / safety

| # | Finding | Fix |
|---|---------|-----|
| R3-02 | `NaN`/`+inf` survive `max(lo, min(hi, x))` and resolve to the **ceiling** — a garbage dial silently producing the most aggressive intervention available | reject non-finite via `math.isfinite`; fail closed, not open |
| R3-03 | `int(circuit.rung)` unguarded; a NULL/garbage rung column raised into the route's bare `except`, **silently disabling the safety disclosure with nothing in the logs** | degrade downward to MINED (matching `_coerce`) and log `circuit_rung_uncoercible_degraded_to_mined` |
| R3-04 | The dial **discarded every hazard and clamp warning** — `set_circuit_steering`'s result was assigned nowhere. The management API reports them; the `/v1` dial reaching the same λ was silent. A hole in the PADR v1.2 over-steering mitigation | capture the outcome; log `circuit_dial_hazards` with counts on the applied line |
| R3-05 | `CIRCUIT_INTENSITY_MIN/MAX` unvalidated — inverted bounds invert the dial (`"max"` → the floor). `sae_service` already normalises its own envelope | normalise `lo > hi` in `_resolve_circuit_intensity` |
| R3-06 | Definition parsed and members flattened **twice** per dialled request; R2's fix left R1's block beneath it, making two failure branches unreachable. `circuit_dial_definition_unparseable` could never fire, so an operator grepping for it would wrongly conclude the document parsed | removed the dead second parse; the log events are now reachable |

### Evidence honesty (the product's core promise)

| # | Finding | Fix |
|---|---------|-----|
| R3-07 | **The OWUI filter was a FOURTH independent derivation of "is this circuit steering"** — it probes `/api/circuits/active`, which answers *what is active*, and rendered a rung suffix from it. The server deliberately suppresses its own header for slice-fallback / unparseable / unattached circuits. So the chat status line — the surface users actually read — overclaimed exactly where the header did not | the server now answers it: `steering: bool` on `CircuitSummary`, computed from `_steering_circuit()`. The filter returns `""` when it is `False`. `None` (older build) still renders, so no regression |
| R3-08 | **`min` ≡ `off`.** Circuits floor at `0.0` (clusters at `0.5`), so a circuit with no authored `intensity_range` makes `"min"` byte-identical to `"off"` — silently failing FPRD US-14.1 ("off/min/max produce observably different outputs") while the copy said "min (declared bound)" | `_min_is_off()` + explicit copy: *"min — this circuit declares no floor, so min is OFF"*. Documented in the manual, the API reference, and the valve help |
| R3-09 | The untrusted circuit **name** was rendered verbatim into the chat: no length cap, no newline stripping, no escaping. R1 hardened `rung_language` against a spoofed endpoint and left the adjacent field from the same untrusted response open. Names arrive via import from shared/marketplace definitions — attacker-influenced by design — and OWUI renders status text as markdown | `_safe_name()`: collapse all whitespace, strip markdown punctuation, cap at 60 chars. Verified against the reviewer's demonstrated payload |
| R3-10 | `off` rendered *with* a circuit attribution — "off for this reply · circuit X — associated" reads as a contradiction | suffix suppressed at λ=0; nothing is steering, so name nothing |
| R3-11 | With `millm_base_url` empty (the default since R1-05) the status line looks healthy while the evidence disclosure is silently off, with no operator signal | appends *"circuit evidence unavailable (set millm_base_url)"* |

### Documentation

| # | Finding | Fix |
|---|---------|-----|
| R3-12 | FPRD §14 required an `X-miLLM-Circuit-Rung` note in the OpenAI-API reference; it existed only inside the OWUI tutorial | new **`X-miLLM-Circuit-Rung`** section in `manual/docs/api/openai-compatible.md` with the full rung table and the omission semantics ("absence never means rung 0") |
| R3-13 | The API reference's dial semantics were cluster-only and **actively wrong for circuits** — it promised "dialing below the declared floor is always honored", which R2-05 deliberately reversed | new circuit subsection stating both ends clamp, the 0.0-vs-0.5 floor difference, the authored-basis re-derivation, and the clamping caveat |
| R3-14 | The tutorial's slice note described a state this dial cannot produce (a slice is steered by its backing cluster profile, on a different envelope) | corrected, with the 0.5 floor and the absent rung header called out |
| R3-15 | The "Dial has no effect" troubleshooting row had no circuit-specific causes | added: `min` resolving to 0, slice-fallback, partial SAE attachment |
| R3-16 | The filter's module docstring and `dial` valve help were still cluster-only after the v1.4.1 rename — the description OWUI shows operators contradicted the title | both updated to describe circuits, rungs, and the `millm_base_url` requirement |

### Test integrity

| # | Finding | Fix |
|---|---------|-----|
| R3-17 | R1's entire SSRF/redirect hardening had **zero direct tests** — a refactor dropping `build_opener(_NoRedirect)` would restore the vector with a green suite | `TestProbeSecurityControls`: scheme allow-list, redirect refusal, read cap, and R2-06's do-not-cache-failures rule |
| R3-18 | The untrusted-name sanitiser needed pinning against the actual payload | `TestUntrustedCircuitNameIsSanitised` — 5 tests including the reviewer's injection string |
| R3-19 | Nothing covered the server-verdict deferral or the min-is-off disclosure | `TestFilterDefersToTheServerSteeringVerdict`, `TestMinIsOffDisclosure` |

---

## Deferred — recorded with rationale

| # | Finding | Disposition |
|---|---------|-------------|
| A | **Attachment/steering epoch** (R2 deferred #11, designed in detail this round). The dial's save→generate→restore critical section is entirely unlocked, so it races admin `activate`/`deactivate`/`set_intensity`, and an operator's mid-request λ change is silently reverted (`set_intensity` even returns `"reapplied": true`). The architect evaluated three mechanisms and recommends a monotonic `steering_epoch` on `AttachedSAEState`, bumped by every authoritative writer, captured in the saved dict, and compared under the lock at restore — *last authoritative writer wins*. Rejected: extending the request semaphore to admin mutations (turns management calls into 503s behind long generations, inverts layering, deadlock risk) | **Own change, both paths at once.** The same window exists on the Feature 10 profile path, so this is one epoch field and one guard covering both — it should not be smuggled into F14's diff. Raised as task-list follow-on |
| B | Extract `CircuitSteeringEngine` so the dial stops constructing `SAEService.__new__` with only `_sae_state` set, and so `_serve_full`/`set_intensity`/dial share one serving derivation | Real layering debt, correctly diagnosed. Refactor of F12/F13 surface area — belongs with (A), not in a review round |
| C | Rung header emitted on undialled requests | Deliberate: it answers "what is steering", which is meaningful without a dial. Now honest post-R3-01/R3-07. Documented in the API reference |
| D | FTDD §5 says one base-selection branch; two paths shipped | The implementation choice is sound (the single-SAE base can only reach `layer[0]`). **FTDD to be updated at task 5.0 acceptance** to record what shipped and why |
| E | EC-14.3 specifies slice-fallback dial behavior; code makes it a no-op | Code is right (dialling there would double-apply through the cluster profile). **FPRD EC-14.3 to be corrected at acceptance** — flagged so the gate is not waved through |
| F | Integration tests + `docs/mcp-contract.md` circuit-dial section | **Task 4.0**, still owned and now explicitly sequenced next |
| G | CBM batches share global `AttachedSAEState` with serial dialled requests | Pre-existing (equally true of Feature 10); F14 widens the blast radius. Strongest argument for (A) being state-based |
| H | `make_sae` is an unspecced `MagicMock`; `service_with_circuit` reimplements the serving-mode gate inside the double | Real test-quality debt. Verified the four mocked methods match `sae_wrapper` today. Worth `spec=`-ing in the F15 cycle |
| I | Per-request O(layers) unbatched writes inside the semaphore | Fine at current circuit sizes; the (A) epoch enables the correct fix (skip apply when `(circuit_id, λ)` is unchanged) |

---

## Verification

- `pytest tests/unit tests/integration` → **1426 passed, 1 skipped**
- `test_circuit_dial.py` 29 → **32** · `test_dial_filter.py` 37 → **53**
- Injection payload re-run manually: renders as inert single-line quoted text; spoofed `rung_language` still yields `associated [UNVALIDATED]`
- Copy-audit (no "causal" below rung 2) passing, negative control included

**Round 3 outcome:** 1 critical regression in R2's own headline fix — found independently by all four
perspectives — plus 4 fail-open input/error paths, 5 evidence-honesty defects on the surface users
actually read, 5 documentation corrections, and 3 test-integrity gaps closed. The two largest
structural items (steering epoch, `CircuitSteeringEngine`) are deferred deliberately: both span
Feature 10's path as well, and both deserve their own change rather than being folded into a review fix.
