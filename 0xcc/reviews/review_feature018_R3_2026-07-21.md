# F18 Single Serving Derivation — Review Round 3

**20 findings, 20 fixed. Suite 1956 → 1983. CI green.**

Round 3 ran two independent review agents (correctness; resilience/reliability/UX)
against R2's fixes, plus a mutation sweep on the load-bearing lines.

## The headline finding

**R3-01 — a rung header for an intervention that never ran.**

`X-miLLM-Circuit-Rung` is computed at request ENTRY. The dial applies LATER,
inside generation, and its `except Exception` deliberately never fails a chat
request. So an apply failure left the response advertising

    X-miLLM-Circuit-Rung: 2; language="causally validated (edge)"

for an intervention that provably did not happen — an evidence claim about
nothing, on the one surface a dial client actually reads.

Three separate defences did not catch it:

* `_steering_circuit`'s own docstring NAMES this hazard and says R1 fixed it.
  R1 fixed the LOOKUP path (nothing attached) and left the APPLY-FAILURE path
  open.
* Two full review rounds read past it.
* The copy audit could not see it. It greps for forbidden WORDS; here a
  CORRECT phrase was attached to a FALSE event. That is a structural honesty
  defect, not a lexical one, and no word-level gate can reach it.

Known limit, recorded rather than papered over: the streaming branch must
commit headers before the first byte, so its rung header remains best-effort.
Only the non-streaming path can retract.

## The structural finding: guard the SINK, not a subset of sources

Both agents converged independently on this. `max(lo, min(hi, nan))` returns
`hi` — a non-finite intensity resolves to the CEILING, maximum-aggression
steering, silently. R2-04 guarded ONE of the four paths into that clamp, and
its commit message was precisely accurate about it: "refuses any non-finite
OVERRIDE". The other three were open, and an authored `intensity_range` of
`"NaN"` reaches the clamp from any imported document, because `float("NaN")`
does not raise.

Fixed at the single point of convergence (R3-02), plus the derived branch one
line below the original guard (R3-04). Fail CLOSED: refusing to steer is
correct where the alternative is steering at the maximum the envelope allows.

## Findings that were found BY a fix

Three findings this round exist only because an earlier fix exposed them —
the argument for fixing latent defects in touched code rather than only
regressions.

* **R3-08** — `applied_epoch` unbound on the failed-apply path
  (`UnboundLocalError` out of a method whose contract there is "report the
  divergence"). Latent only because the unconditional epoch bump happened to
  bind it as a side effect. R3-07 removed that accident.
* **R3-10** — the first version of the R3-09 rollback had the same class of
  hole it was written to close: the layer that raises during the apply is the
  most likely to raise again during its restore, and that second failure was
  swallowed, leaving that layer silently zeroed.
* **R3-11** — F14-R1-01 reappearing INSIDE the module whose docstring says it
  exists to make F14-R1-01 structurally impossible. A dict-shaped budget found
  no `.intensity`, fell through, and served the stale DB column. Verified by
  execution: authored 1.7 served as 0.3.

## Silent failure

R3-13/14 closed two paths that returned None with NO operator signal at all. A
corrupt `circuit_meta` made the circuit read ACTIVE in the management API while
steering nothing, forever — discoverable only by noticing the model stopped
behaving differently. Going quietly dark is the failure mode this codebase
treats as worse than raising.

## Test quality — seven unfalsifiable tests, three of them mine this round

R3-17/18/19 rewrote three of my own earlier-round tests that could not fail:
a `co_names` membership check (asserts a mechanism is NAMED, not INVOKED — the
anti-pattern this file's module docstring explicitly excludes), a source
substring check (`"plan.intensity" not in src`, defeated by any indirection —
the same class R2-05 fixed ONE COMMIT EARLIER), and a DOCSTRING grep in a
disjunction whose second clause subsumed the first.

R3-06 and R3-08 were each a test written THIS ROUND that could not fail:

* R3-06 grepped the dial's source for `note_circuit_apply_failed()`. Emptying
  that function's BODY left the call text intact — grep green, flag never set,
  header never retracted, defect fully reintroduced under a green suite.
* R3-08's first version used `circuit_meta={}`, which exits at `_parse_stored`
  long before the branch under test, with its own `except Exception: pass`
  hiding that. It asserted an epoch that never moved because the code never
  got there.

**Both were caught by the negative control and by nothing else.** Reading them
would not have found either. That is now seven in this increment, and the ratio
is the point: careful reading has a floor it cannot get under, and mutation is
what gets under it.

## Investigated, not a bug, pinned anyway

**R3-16.** A concurrency scenario (two concurrent λ=0 dials wiping operator
steering) is NOT reachable at the default config: the dial runs inside the
request-queue semaphore and `MAX_CONCURRENT_REQUESTS` is 1. But note what
provides that safety — not the epoch guard, which cannot distinguish "someone
wrote after me" from "someone saved the state I was midway through clearing",
but a config value in a different file with nothing connecting them.

**F19 is Concurrent Circuit Serving — the increment whose entire purpose is to
raise that number.** The dependency is pinned in both directions, with a
failure message naming what must be built first. An unstated load-bearing
assumption is a defect in the same way an unpinned fix is.

## Round verdict

Every fix is pinned by a test proven to fail when the fix is reverted. Every
mutation was confirmed applied before being judged, and the tree confirmed
clean after each restore. One control fired on a narrower input than expected
(`-inf` only, because the new derived guard catches `nan`/`+inf` first) and is
recorded as defence in depth rather than tidied away.

The pattern that has held every round of this increment held again: **the most
productive target in round N is round N-1's fix.**
