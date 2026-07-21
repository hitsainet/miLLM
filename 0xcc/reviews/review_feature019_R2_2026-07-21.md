# F19 Concurrent Circuit Serving — Review Round 2

**20 findings, 20 fixed. Suite 2070 → 2095 backend, 294 → 300 frontend.**

## The feature did not work at the DB layer

**R2-01.** `CircuitRepository.set_active()` called `deactivate_all()`. One line.

Proven by execution: activating two DISJOINT circuits leaves exactly ONE active
row. The claim gate passes, both circuits steer through the owner map, and the
first circuit's row reads `is_active=False`. The model steers with nothing
recording it, `GET /circuits/active` reports only the second, and no operator
can stop the first through any surface that reads the row.

R1-04 fixed the READER — `get_active()` no longer raises on two rows, and
`list_active()` was added specifically to report several — and never touched the
WRITER. `list_active()` returned a list that could never hold more than one
element. **Twenty R1 findings were fixed above a feature that did not work.**

Two things about why it survived matter more than the fix:

* **No test touched `CircuitRepository`.** Every concurrent-serving test drove
  the in-memory owner map or the claim registry. The tests and the code agreed
  with each other while both disagreed with reality — "fixtures whose fields
  agree by construction", exactly.
* **A test PINNED the defect.** `test_set_active_deactivates_previous` asserted
  the first circuit becomes inactive, so FIXING the bug broke the test.
  Inverted rather than deleted, so the supersession is recorded where the old
  rule lived.

## A fix that recreated the bug another fix removed

**R2-06.** R1-18's restore collapsed per-layer `composed` into `any(...)`, so an
exclusive layer came back marked composed — outside the exclusive index, where
a third circuit could take it unopposed, with its rung header suppressed
forever on a layer nothing ever composed onto. That is **R1-19's exact defect
class, one round later, introduced by the fix for a different one.**

**R2-07**, in the same code: `claim()` is all-or-nothing inside its savepoint,
so one layer taken by the race winner lost every other layer too — the state
R1-18 exists to prevent.

## Reachability of the override

**R2-02.** "Compose anyway" dropped `acknowledgeUnvalidated`, so an operator who
ticked the acknowledgement and then hit contention was refused AGAIN telling
them to tick it. The override was **unreachable for exactly the circuits where
composition is riskiest** — the ones whose evidence is not causally validated.

The first test for this used a VALIDATED circuit, where `ack` is false either
way, so the mutation survived and the test proved nothing. Only the unvalidated
case exercises it, which is also the only case where it bites.

## Honesty defects

* **R2-14** — the circuit card rendered the rung badge while the runtime was
  SUPPRESSING that header for composed layers. The card and the response
  disagreed, and the card is what an operator looks at.
* **R2-17** — the composition warning stated the close-out measurement bare,
  without the "one model, one fixture" caveat that travels with it everywhere
  else. Same over-generalisation R1-15 fixed in the co-tenant warning.
* **R2-09** — startup reported its OWN NORMAL BEHAVIOUR as an anomaly. Reconcile
  ran after the bulk deactivation, so its orphan branch fired for every claim on
  every restart. A permanently false-positive warning trains operators to ignore
  the one signal that matters when a genuine orphan appears.

## Observability and recovery

* **R2-11** — `circuit_layers_composed` counted only `circuit:` owners, so it was
  blind to slice-fallback co-tenants: the alertable condition could never fire
  while the rung header was already suppressed. Two authorities disagreeing.
* **R2-16** — every F19 failure handler logged and continued, and nothing counted
  any of them. A green dashboard while layers leak.
* **R2-19** — a failed reconcile left the app serving with stale claims and a
  fully healthy `/health`. Now DEGRADED, because a readiness probe should see it.
* **R2-10** — every claim-leak path had ONE remedy: a full restart, dropping
  every loaded model and attached SAE to clear one stale row.
* **R2-20** — the two metric surfaces re-implemented the same scan and already
  differed in how they derived `layers_served`.

## Test quality — three of my own

* **R2-04** — my R1 wiring tests were `readFileSync` + `toContain`. Renaming the
  handler or commenting out the JSX would pass. **The third occurrence of this
  increment's named anti-pattern, in a test written to guard against it.**
  Replaced with a real render, which immediately found **R2-05**: `void
  activateCircuit(...)` left an unhandled rejection on every handled refusal.
* A test monkeypatched a module attribute to raise and **leaked into its
  sibling**, making it read 0. A test that corrupts shared module state is a
  test defect, not a code finding.
* A test drove a handler it never reached. Instrumenting the path — rather than
  guessing a third time — showed a collision raises during the MERGE, before any
  write, so no layer enters the rollback's `done` list.

## Process failures, recorded

* I **edited files while a mutation sweep was running** and got a 4-failure
  result that was pure noise. A suite result taken during a mutation sweep is
  not evidence.
* A control run timed out mid-mutation and left a file dirty. Caught by checking
  the tree BEFORE reading results — the same check that caught it in R1.

## Round verdict

The pattern from F16–F18 held, with a sharper edge: **R1's fixes were the most
productive target, and R1's own fixes reintroduced defects R1 had removed.** The
deepest finding was not in F19's logic at all — it was one line of pre-existing
code that was correct until concurrency became possible.
