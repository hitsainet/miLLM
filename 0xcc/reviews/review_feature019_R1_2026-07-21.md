# F19 Concurrent Circuit Serving — Review Round 1

**20 findings, 20 fixed. Suite 2044 → 2070 backend, 288 → 294 frontend.**

Two independent review agents (correctness; resilience/reliability/UX) plus a
mutation sweep. The agents found defects more severe than anything the
implementation testing caught, and four of them were invisible to a fully green
suite.

## The feature's success state broke the feature

**R1-04.** `CircuitRepository.get_active()` used `scalar_one_or_none()`, which
RAISES `MultipleResultsFound` the moment two circuits are active — the exact
state F19 exists to create. **Proven by execution, not by reading.**

It would not have failed loudly. `_active_full_circuit` catches broadly and
returns None, so every chat request would have served UNSTEERED while both
circuit rows read active, and `GET /circuits/active` would 500. Eleven call
sites reached it.

Fixed with `list_active()` for callers that must act on all of them;
`get_active()` returns the most recently updated, matching what the operator
last asked for and what the migration downgrade keeps.

## The claim lifecycle had three separate holes

* **R1-03** — `deactivate()` released the steering owner and left the DB claim
  row LIVE FOREVER. Activating anything else on those layers was then refused
  NAMING the deactivated circuit, and the obvious remedy — deactivate it again
  — is a no-op. The layer became permanently unclaimable. **Routine
  deactivation was a leak, not an edge case.**
* **R1-01** — `reconcile()`, the only orphan collector, had ZERO production
  callers. Written, unit-tested, never invoked, under a docstring reading "runs
  UNCONDITIONALLY". Third instance of this anti-pattern in the increment.
* **R1-02** — startup deactivated every circuit without releasing its claims,
  so one restart orphaned all of them.
* **R1-18** — the self-release before a re-claim flushed OUTSIDE the savepoint,
  so a re-activating circuit that lost a race was left holding nothing while
  still active and steering.

No single reading found them all. The overlap between the two agents is the
argument for running more than one.

## Dead code that passed its own tests

**R1-06.** The ENTIRE contention UI — `ContentionDialog` and `ClaimsStrip` —
was written, exported and unit-tested with **no consumer anywhere**. A refusal
fell through to a generic toast that discards the incumbent, the measured
hazard, the colliding keys and both resolution actions.

BR-011 §6.2's binding retention condition (every override is surfaced in the
UI) was UNMET IN PRODUCTION while Vitest stayed green — because the component
tests render the component directly. A component test can never answer "does
anything call this".

## Honesty defects that under-claim

These are the ones no honesty test catches, because they remove a disclosure
rather than overstating one:

* **R1-08** — `composed` was PERMANENT. When the composing circuit left, the
  incumbent stayed flagged forever: a validated rung-2 circuit serving ALONE,
  permanently stripped of its rung header because something once composed onto
  it.
* **R1-07** — `_any_layer_composed` fails OPEN while its docstring and a second
  comment both said "fails CLOSED". Two of three statements wrong about a
  safety property. The trade-off is deliberate and defensible; the prose
  contradicting it is not.
* **R1-15** — the co-tenant warning promised "a circuit takes EXCLUSIVE
  ownership of the layers it steers". F19 made both halves false.

## Silent failure

* **R1-09** — the claim gate no-op'd silently when the repository had no
  session: a serve-without-claiming path that restored pre-F19 clobbering
  behind a healthy-looking response.
* **R1-10** — `/claims` returned `[]` on the same condition, which the UI
  renders as the affirmative "No layers are currently claimed" — a confident,
  wrong all-clear at the moment the subsystem is broken.
* **R1-16** — `deactivate` reported `cleared_steering: true` when the clear
  FAILED, because the flag was not derived from the outcome.
* **R1-17** — a failed sensing arm was indistinguishable from "armed and
  nothing co-fired": clean success, zero events forever.
* **R1-11** — no metric for circuits, layers or COMPOSITION, and the only
  metric named "circuit" is the unrelated HuggingFace HTTP breaker — an active
  trap for anyone alerting on the word.

## Defects reintroduced from earlier features

* **R1-12** — F18 R3-09/10's chimera, on the owner path: a collision raise
  partway through a multi-layer apply left earlier layers written, later ones
  stale, and the owner entry already replaced.
* **R1-13** — a detached layer's stale contribution could RESURRECT steering
  the operator had stopped, when a co-tenant's rebuild pulled it back in.

## Test quality

Three mutations SURVIVED and were fixed by writing the test, not by weakening
the mutation:

* `_release_active_circuit` had **zero coverage** — reverting it to the
  singular `get_active()` passed the whole suite.
* The R1-13 test drove `release_owner`, which pops the owner regardless, so it
  proved nothing about the drop path. Rewritten to reach it the way production
  does.
* No test asserted the wording of an operator-facing warning, so restoring the
  false "exclusive ownership" copy passed.

One control was **inconclusive rather than a survivor**: a 2-minute timeout
killed the loop mid-mutation, leaving `circuit_service.py` dirty. Caught by
checking the tree before reading results, restored, and re-run detached. A
control whose mutation may not have applied is not evidence either way.

## Round verdict

Every fix is pinned by a test proven to fail when reverted. The pattern that
held through F16–F18 held again — but with a new wrinkle: **the most dangerous
findings here were not in the new code's logic, they were in what the new code
ASSUMED about the old code.** `scalar_one_or_none`, the missing claim release,
the singular co-tenant release: each was correct before F19 and wrong the
moment concurrency became possible.
