# Feature 14 — Circuit-Aware OWUI Dial: Review Round 1

**Date:** 2026-07-20
**Scope:** commit `4c6e293` (F14 tasks 1.0–3.0) — per-request circuit dial, rung echo, `millm_dial_filter.py` v1.4.0, manual Step 6
**Method:** `/code-review` at high effort — 8 finder angles × 6 candidates, 1-vote recall-biased verification, run as two independent finder passes
**Findings:** 28 surfaced · **11 fixed** · 17 deferred/rejected (recorded below)

---

## Fixed

### F14-R1-01 — CRITICAL: the dial's de-scaling math was wrong in two independent ways
`inference_service.py` — the dial read each SAE's LIVE steering values and rescaled them
(`live / circuit.intensity × λ`) to reach the requested λ. That cannot recover the authored basis:

1. **Wrong divisor.** `circuit_service.py:415` serves with
   `definition.budget.intensity if definition.budget else circuit.intensity` — the *document's*
   intensity. The dial divided by the *DB column*. When a circuit is imported with a document
   intensity that differs from the stored column (routine — `set_intensity` writes the column only),
   every member is scaled by the wrong factor, silently, with no error.
2. **Clamping is lossy.** Apply clamps each member to ±200. A member authored at 150 served at λ=2
   stores `clamp(300) = 200`. Dialling back to 1.0 yields `200/2 = 100`, not 150. Division cannot
   invert a clamp.

**Fix:** re-derive from the authored basis. `_apply_request_circuit_steering` now parses
`CircuitDefinitionV1` from `circuit_meta` and calls `set_circuit_steering(members, λ)` with the
authored strengths — the same path activation uses. Added `_circuit_serving_members`, delegating to
`CircuitService._serving_members` so the dial and activation flatten members identically (one
definition of "which members serve", not two that can drift).
**Tests:** `test_dial_is_absolute_not_a_multiplier_of_the_stored_dial`,
`test_clamped_members_recover_their_authored_basis` (authored 150 @ λ=2 → dial 1.0 → asserts 150,
which the old code returned as 100).

### F14-R1-02 — the rung echo and the dial apply disagreed about what counts as active
`active_circuit_rung()` echoed a header for circuits the dial would then no-op on (no serveable
members, unparseable definition), so a response could advertise `X-miLLM-Circuit-Rung: 2` while
nothing was steering. `_resolve_active_circuit_intensity` now parses the definition and checks for
serveable members, mirroring apply's no-op rules exactly.

### F14-R1-03 — one failing per-layer restore stranded every later layer
`_restore_request_profile` restored layers in a loop with a single outer try. A layer detached
mid-request raised and skipped the rest, leaving them dialled for **all subsequent requests** — a
per-request override leaking into global state. Each layer's restore now owns its try/except.
**Test:** `test_restore_tolerates_a_layer_detached_mid_request`.

### F14-R1-04 — CRITICAL: the OWUI probe blocked the entire worker event loop
`_circuit_status` called `urllib.request.urlopen` directly inside `async def inlet`. `await`ing it
never yields — one slow or hung miLLM stalls **every concurrent chat** on that OWUI worker, not just
the one that triggered the probe. Now `asyncio.to_thread`, timeout 1.5s → 0.8s, plus a 10s TTL cache
(the active circuit changes far less often than once per message, so steady-state costs nothing).

### F14-R1-05 — the probe default was wrong for every documented deployment
`millm_base_url` defaulted to `http://localhost:8000`. Inside the OWUI container `localhost` is OWUI
itself — the repo's own k8s manifest puts OWUI in `open-webui-ns` reaching miLLM at
`millm-backend.millm.svc.cluster.local:8000`. The default therefore probed the wrong service on every
install, adding latency for a guaranteed failure. Now **empty by default** (off until configured),
with the Docker and k8s forms named in the valve description and the manual. Off-until-configured
beats silently-broken.

### F14-R1-06 — a spoofed endpoint could inject "causal" into rung-0 copy
The filter rendered the server's `rung_language` verbatim into the chat. The evidence-ladder
guarantee ("causal" never below rung 2) is the whole point of the rung surface, and it was delegated
to an unauthenticated HTTP response. The phrase is now validated against the local `RUNG_LANGUAGE`
mirror and falls back to the mirrored phrase on any mismatch.
**Test:** `test_a_spoofed_server_phrase_cannot_inject_causal_language`.

### F14-R1-07 — probe was redirect-following and scheme-unchecked
A server-side fetch from an operator-set valve that follows redirects can be pointed at arbitrary
internal addresses. Added a `HTTPRedirectHandler` that refuses redirects, an http/https scheme check,
and a 64 KB read cap.

### F14-R1-08 — status copy asserted "cluster's" when the probe had merely failed
`min`/`max` rendered `"cluster's declared bound"` whenever `circuit` was falsy — including when the
probe failed against an active circuit, actively misinforming the user. Now names only what was
observed: `"circuit's declared bound"` or the neutral `"declared bound"`.

### F14-R1-09 — `X-miLLM-Circuit-Rung` was unparseable
`2 causally validated (edge)` — space-separated with spaces and parens in the value. Now RFC 8941
structured: `2; language="causally validated (edge)"`. The rung stays a bare int for trivial parsing
and the phrase is a quoted-string, so ladder punctuation can never break a naive parser. Manual
updated.

### F14-R1-10 — the manual overstated proportion preservation
Claimed budgets "keep their relative proportions as you move the dial". Clamping breaks exactly that
at high λ. Added a note stating members clamp at ±200 and that relative proportions compress when
they do, pointing at the Circuits page clamp report.

### F14-R1-11 — dial tests ran against an empty `circuit_meta` stub
The fixtures passed `{}`, so the tests never exercised definition parsing — which is precisely where
F14-R1-01 lived. Added `make_meta()`, producing a real `circuit-definition/v1` document, and pointed
every dial test at it.

---

## Deferred (recorded, not fixed this round)

| # | Finding | Why deferred |
|---|---|---|
| 12 | No integration test covering `/v1/chat/completions` with `steering_intensity` end-to-end | Task 4.0 (integration) owns this; scheduled, not dropped |
| 13 | `docs/mcp-contract.md` doesn't document the `/v1` circuit dial or its min floor (circuit 0.0 vs cluster 0.5) | Contract edit batched into task 4.0 with the other route additions |
| 14 | Filter still titled "miLLM Cluster Dial" though it now dials circuits too | Renaming the filter changes its OWUI identity/id; needs a migration note for existing installs — R2 |
| 15–28 | Style/naming nits, speculative concurrency on `_probe_cache` (benign racy read of an immutable tuple), suggestions to add metrics/telemetry | Below the bar or out of F14 scope |

---

## Verification

- `pytest tests/unit tests/integration` → **1410 passed, 1 skipped** (skip: torch.compile backend unavailable in this env)
- `test_circuit_dial.py` 23 · `test_dial_filter.py` 37
- Copy-audit (`causal` forbidden below rung 2) still passing, including the negative control

**Round 1 outcome:** 2 critical (wrong-basis steering; blocked event loop), 1 security (rung-language
injection), 8 correctness/UX. All fixed and pinned by tests.
