# Feature 14 — Circuit-Aware OWUI Dial: Review Round 2

**Date:** 2026-07-20
**Scope:** commit `cf38fcb` (R1's fixes) + `4c6e293` (the feature)
**Method:** adversarial pass with the explicit mission *find bugs in Round 1's own fixes* — 16 findings, 15 CONFIRMED by execution, 1 PLAUSIBLE
**Findings:** 16 · **10 fixed** · 6 deferred

The premise held for the third consecutive feature: **two of R1's fixes reintroduced the exact defects they claimed to close.**

---

## Fixed

### F14-R2-01 — CRITICAL: R1 hardened the restore loop and left its input incomplete
R1-03 gave each per-layer restore its own try/except so one failure couldn't strand the rest. But the
**snapshot** filtered on `circuit.layers` (the DB column) while the **apply** drove off the
definition's member layers:

```python
entries = [e for e in state.entries() if e.layer in (circuit.layers or [])]  # save
members = self._circuit_serving_members(definition)                          # apply
```

Any layer present in one and not the other is dialled and never restored. Reviewer verified by
execution: row `layers=[10]`, definition members on L10+L13 → L13 pre-dial `{2: 30.0}`, post-restore
`{2: 60.0}`, still enabled. **A per-request override leaking permanently into global state** — the
precise class of bug R1-03 existed to prevent, one level up. The same mismatch hit the apply-failure
path, where there is no second chance.

**Fix:** the snapshot now derives its layers from the definition's members, the same source the apply
uses. The two can no longer drift.
**Test:** `test_a_layer_absent_from_the_db_column_is_still_restored`.

### F14-R2-02 — CRITICAL: R1's echo-parity fix landed on the wrong header
R1-02 was titled "the rung echo and the dial apply disagreed about what counts as active" — and fixed
`_resolve_active_circuit_intensity` (the **λ** header) while `active_circuit_rung()` (the **rung**
header, the one the finding named) kept calling `_active_full_circuit()` directly. Verified: with an
unparseable `circuit_meta` and no attached SAEs, apply → `None`, λ echo → `None`, but rung →
`(2, 'causally validated (edge)')`. The response still advertised
`X-miLLM-Circuit-Rung: 2; language="causally validated (edge)"` **while nothing was steering** — an
evidence-grade overclaim, the one thing the rung surface exists to prevent.

**Root cause was structural:** three surfaces (apply, λ echo, rung echo) each re-derived "is this
circuit steering?" independently, so fixing one never fixed the others.
**Fix:** one `_steering_circuit()` predicate; all three now ask it. Fixing the class, not the instance.
**Tests:** `test_rung_header_suppressed_when_nothing_is_actually_steering`,
`test_rung_header_suppressed_for_an_unparseable_definition`.

### F14-R2-03 — λ=0 left values resident behind a disabled flag
The dial's λ=0 fast path called only `enable_steering(False)`, while the λ>0 path
(`set_circuit_steering`) clears each SAE first. Verified: values stay at `{1: 40.0}` — visible to
`get_steering_values` and re-armed by any later enable. Now clears and disables.
**Test:** `test_lambda_zero_clears_rather_than_only_disabling`.

### F14-R2-04 — `{"steering_intensity": true}` silently dialled λ=1.0
`isinstance(raw, (int, float))` accepts `bool` (an `int` subclass). A JSON `true` was accepted as a
dial value rather than rejected. Now excludes `bool` — in both the circuit and profile paths.
**Test:** `test_a_bool_is_not_a_dial_value`.

### F14-R2-05 — the numeric dial ignored the authored floor R1 computed
`_resolve_circuit_intensity` intersects the authored range with the config envelope into `(lo, hi)`,
then the numeric path clamped only to `hi`. A numeric `0.1` against an authored `[0.5, 1.0]` returned
`0.1` — below a floor that symbolic `"min"` refuses to go below. Half the intersection was wired.
Now clamps both ends; dialling to exactly 0 (off) remains always allowed.
**Test:** `test_numeric_respects_the_authored_floor`.

### F14-R2-06 — the probe cached FAILURES, suppressing the safety disclosure
R1-04's 10s TTL cache stored `None` on failure for the full TTL. One timeout therefore blanked the
`[UNVALIDATED]` rung disclosure for 10 seconds of messages **after miLLM recovered** — the cache
silencing the exact safety surface the filter exists for. Failures are no longer cached; only
successes are. A failed probe simply retries next message.

### F14-R2-07 — the rung echo added a DB round-trip to every chat completion
`active_circuit_rung()` sits above the `steering_intensity` guard, so every completion paid a session
checkout + `SELECT` even on deployments using no circuits — and after F14-R2-02 unified the
predicate, the λ echo and apply each paid it again. `_steering_circuit()` is now memoised on the
request-scoped service: one lookup per request instead of three.

### F14-R2-08 — `_serving_members` was called unbound on an unwritten purity promise
The dial called `CircuitService._serving_members(None, definition)`, passing `None` for `self`. It is
pure today, but nothing enforced that — one future `self.repository` reference would turn every
dialled request into an `AttributeError` on an unguarded line. Promoted to `@staticmethod`, making it
a compile-time guarantee. Existing `self._serving_members(...)` call sites are unaffected.

### F14-R2-09 — filter renamed to "miLLM Steering Dial" (+ upgrade note)
It dials circuits as well as clusters; the old title misdescribed it. Bumped to v1.4.1. R1 deferred
this over migration risk, so the manual now has an **Upgrading** section: Open WebUI keys filters by
internal id, not title, so pasting over the existing filter upgrades in place and preserves valves
and per-model assignments — while creating a *new* filter would leave both enabled and both applying.

### F14-R2-10 — trailing whitespace on the header line R1-09 rewrote
W291 on `chat.py`. Removed.

---

## Why the suite missed all of this

Worth recording, because it is the round's most transferable finding: every dial test used
`make_circuit()` whose `layers=[10, 13]` **exactly matches** `make_meta()`'s member layers. R1-11
replaced the empty `{}` fixtures with a real document and, in the same stroke, locked in the one
alignment that makes F14-R2-01 unreachable. The restore test asserted only the surviving layer and
never checked the stranded one. Fixtures whose fields agree by construction cannot detect two code
paths disagreeing about those fields — the new tests deliberately set `layers=[10]` against L10+L13
members.

---

## Deferred

| # | Finding | Why deferred |
|---|---|---|
| 11 | `activate`/`deactivate`/`set_intensity` don't take the request semaphore, so an operator λ change mid-request is undone by the restore | Real but pre-existing and broader than F14 — the same window exists on the Feature 10 profile path. Needs an attachment-epoch design; raising for R3 as an architecture item |
| 12 | `_probe_sync` reads `self.valves.millm_base_url` mid-flight, so a valve edit can cache under a stale base | Benign: the cache key IS the base, so a stale entry is keyed to the old URL and simply misses |
| 13 | `_probe_cache` racy read of an immutable tuple | Benign by construction (single assignment of a whole tuple) |
| 14–16 | Naming nits, telemetry suggestions | Below the bar |

---

## Verification

- `pytest tests/unit tests/integration` → **1416 passed, 1 skipped**
- `test_circuit_dial.py` 23 → **29** (6 new regression tests, all failing before their fix)
- Copy-audit still passing, negative control included

**Round 2 outcome:** 2 critical regressions in R1's own fixes (permanent global-state leak; rung
overclaim), both traced to the same structural cause — duplicated derivations of the same fact —
and both fixed by unifying the derivation rather than patching the copies.
