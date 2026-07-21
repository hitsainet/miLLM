# F19 Concurrent Circuit Serving — Review Round 3

**20 findings, 20 fixed. Suite 2095 → 2114 backend, 300 → 306 frontend.**

## R2's two new mechanisms contradicted each other

**R3-01.** R2-19 added a degraded health flag. R2-10 added a claim-release
endpoint as the runtime remedy. The flag was a LATCH — nothing cleared it — so
an operator could run the documented remedy successfully and `/health` would
still report DEGRADED for the process lifetime, telling a readiness probe that
activations were refused when they were not.

They restart anyway: the multi-minute GPU outage the endpoint exists to avoid.
Both review agents put this first, independently.

## "Declaring is not wiring" — fourth and fifth occurrences

* **R3-03** — R2-12 added `all_incumbents` to the payload precisely so an
  operator is not sent to deactivate one incumbent, retry, and be refused by a
  second. It was never RENDERED. `grep all_incumbents admin-ui/src` returned
  nothing, so the scenario its commit message describes still happened verbatim
  in the browser — inside the fix for that problem.
* **R3-15** — `reconcile()` was DEAD IN BOTH DIRECTIONS. R1-01 wired it because
  nothing called it; R2-09 then made startup release every claim, emptying both
  of its input sets by construction. The test guarding it MOCKED the registry,
  so it asserted reconcile was *awaited* and would have passed after the method
  body was deleted.

## Honesty defects

* **R3-04** — the compose success toast asserted `causally validated (edge)` at
  the exact moment the runtime STOPS emitting that header. R2-14 fixed this
  contradiction on the card and missed the toast.
* **R3-17** — R2-11 widened the composed metric to "agree with header
  suppression". Suppression reads the CLAIMS TABLE; the metric reads the OWNER
  MAP. It agreed with a third authority, re-creating the disagreement it set out
  to remove. Both are now reported, because the DIVERGENCE is the signal.
* **R3-19** — `steering` is a per-row question, answered with R3-06's singular
  predicate, so every row reported `steering: false` in exactly the state the
  feature exists to support.

## The copy audit caught my own test — correctly

**R3-05.** Writing R3-04's test put "causally validated" into an assertion
checking the phrase is ABSENT, and the build-failing gate fired. It could not
tell an assertion-about-copy from copy. Excluded test files rather than
allow-listing the string, then verified the narrowing did not weaken it: a
genuine hand-written overclaim in production UI code still fails.

## A finding that was wrong

**R3-12.** R3 reported the total-restore-loss branch as an uncounted worst case.
I implemented the fix, and the mutation control SURVIVED — which is what
prompted tracing the branches instead of trusting the reasoning. `if lost:`
already covered it; the addition would have DOUBLE-COUNTED every total loss and
inflated the exact rate the metric exists for.

**The surviving control was the signal.** A fix whose control does not bite is
either untested or unnecessary. I should have traced before implementing.

## Test quality — three of my own, again

* **R3-10/R3-14** — two more `inspect.getsource` substring tests replaced with
  behavioural ones. One passed when the loop body was wrapped in `if False:`.
* Writing R3-10's replacement, `pytest.raises(Exception)` swallowed a FIXTURE
  SHAPE ERROR as though it were the race under test, so the assertion passed
  for the wrong reason. Narrowed the exception type and asserted the race
  actually happened.
* **R3-16** — my first bypass test called `structlog.configure()`, and
  `reset_defaults()` does not restore the app's configuration: it passed alone
  and failed in the FULL suite by ordering. Second time this session a test of
  mine corrupted shared global state; both were caught only by running the full
  suite rather than the file.
* The R3-18 specificity test was `expect(true).toBe(true)` — a placeholder that
  cannot fail, in the round removing exactly that anti-pattern.

## Round verdict

The theme across all three rounds: **the deepest defects were not in F19's new
logic but in what the new code assumed about the old.** `scalar_one_or_none`,
`set_active`'s `deactivate_all`, the missing claim release, the single-active
profile table — each correct before F19 and wrong the moment concurrency
became possible.

And three times in three consecutive rounds, a fix reintroduced a defect a
previous fix had removed. That is the argument for the third round existing at
all.
