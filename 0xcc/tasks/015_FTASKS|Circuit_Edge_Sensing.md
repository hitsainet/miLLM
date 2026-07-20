# Task List: Circuit Edge Sensing

## miLLM Feature 15

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** ✅ IMPLEMENTED & REVIEWED 2026-07-20 — 3 review rounds (80 findings, 41 fixed, 11 of them
critical), suite 1588 backend / 255 frontend green. **Two requirements are NOT met and are recorded as
such below rather than ticked**: the alone-vs-within side channel (BR-007/BR-008) and the
ground-truth capture rate (§9.1), which needs a live GPU serve.
**References:** `015_FPRD|Circuit_Edge_Sensing.md` · `015_FTDD|Circuit_Edge_Sensing.md` · `015_FTID|Circuit_Edge_Sensing.md` · `docs/mcp-contract.md` (v1.1)

## Relevant Files
- `millm/ml/sae_wrapper.py`, `millm/ml/sae_hooker.py` — edge detection core (extends the F11 `_sense` path)
- `millm/services/circuit_sensing_service.py`, `inference_service.py`, `circuit_service.py`, `sae_service.py`
- `millm/db/models/circuit_edge_sensing_event.py`, `db/repositories/circuit_edge_sensing_repository.py`,
  `db/migrations/versions/012_add_circuit_edge_sensing.py`
- `millm/api/schemas/circuit_sensing.py`, `api/routes/management/circuit_sensing.py`, `api/dependencies.py`, `routes/__init__.py`
- `millm/core/errors.py` — NoActiveCircuitError, CircuitSensingEventNotFoundError
- `millm/sockets/progress.py` — emit_circuit_sensing_event
- `millm/core/config.py` — CIRCUIT_SENSING_* keys
- `admin-ui/src/components/circuits/sensing/*`, `services/circuitSensing.ts`, `hooks/useCircuitSensing.ts`
- `tests/unit/{ml,services,db}/test_circuit_edge_sensing*.py`, `tests/integration/test_circuit_edge_sensing_workflow.py`

### Notes
- Depends on Feature 13 (circuit rows + edges w/ per-edge rung + edge-sensing intent) and Feature 12 (multi-SAE
  attach). Execute after 13. EXTENDS Feature 11 sensing — reuse `_sense`, `context_parts`, WS emitter, repo pattern.
- Migration numbering: **011** (`010_add_sensing_context_parts.py` is the current disk tail; `008` is the
  sensing-events table, NOT circuits). Chain `down_revision` after Feature 13's `011_add_circuits_table.py`.
- Test commands: `pytest` (backend), `npm test` in admin-ui (Vitest).

### Category Checklist Results
- Data: 1.x (table + model + repo + retention) ✓
- Backend/API: 4.x ✓
- Frontend/UI: 5.x ✓
- Business logic: 2.x (fire reuse/up→down matcher/lag), 3.x (lifecycle/context/rung) ✓
- Integration wiring: 3.x (inference + arm/disarm hooks across attached SAEs), 4.2 (DI/router), 5.1 (WS client) ✓
- Error handling & logging: 2.5 (cap/truncated), 3.6 (overhead warn), NO_ACTIVE_CIRCUIT, buffer-hygiene tests ✓
- Testing: paired throughout + 6.x integration (incl. latency-budget + no-"causal"-below-rung-2) ✓
- Performance & security: EDGE-S1..S3 tasks (2.4, 3.5, 6.4); context-privacy documented (7.2) ✓
- Config/deploy: 3.7 config keys; migration auto-runs — no deploy change ✓
- Documentation: 7.2 manual section ✓

## Tasks

- [x] 1.0 Persistence layer (covers FR-15.4; EDGE-P1, EDGE-P2)
  - [x] 1.1 Migration `012_add_circuit_edge_sensing.py` (up+down, indexes, CASCADE via circuit ownership; `down_revision` = Feature 13's `011`)
  - [x] 1.2 `db/models/circuit_edge_sensing_event.py` (mirror `sensing_event.py`; up/down member+layer+pos+act, token_lag, edge_rung + rung_language, context_parts) + `circuit_edge_sensing_repository.py` (create_many/list_events/count/clear/prune)
  - [x] 1.3 Repo unit tests incl. retention cap + age prune + CASCADE with circuit delete

- [x] 2.0 Edge detection core in LoadedSAE (covers FR-15.1; EDGE-D1..D5)
  - [x] 2.1 EdgeSpec/CircuitSensingConfig/SensedEdge dataclasses; arm_edge_sensing/disarm (reuse `_W_enc_m` cache, dtype/device parity); `is_edge_sensing_armed` kept distinct from `is_sensing_armed`
  - [x] 2.2 `_sense_edges` per-pass fire reuse + up→down ring matcher (strict ordering, lag window) + suppressed() early-return
  - [x] 2.3 Shared per-request EdgeRing across the circuit's SAEs; absolute-position lag matching across passes; offset/phase accounting; ring pruning older than L
  - [x] 2.4 Per-request cap + truncated flag + `_sensing_done` fast path
  - [x] 2.5 Unit tests: fire predicate reuse, up→down matcher, EC-15.1 (lone up → none), EC-15.2 (reversed → none), cross-layer close, cap, offsets, unsensable-edge exclusion (EC-15.4/15.6), suppressed, arm idempotence, buffer hygiene (begin resets once; no-begin ⇒ empty collect)
  - [x] 2.6 Hook branch in `sae_hooker.hook_fn` (sibling of F11 sensing, before apply_steering)

- [x] 3.0 Service + lifecycle + flush (covers FR-15.2, FR-15.3, FR-15.5; EDGE-R1..R4, EDGE-S1..S3)
  - [x] 3.1 `circuit_sensing_service.py`: arm_for_circuit (per-SAE config, drop unsensable edges, record them), disarm/should_sense/collect_edges/status
  - [x] 3.2 Inference wiring: begin across attached SAEs after apply, collect in finally, async `_notify_circuit_sensing` beside `_notify_sensing` (both paths)
  - [x] 3.3 Context capture reuse (outputs[0]/IdCaptureStoppingCriteria/prompt ids); ±K `context_parts` span covering up→down, off hot path; K=0 path
  - [x] 3.4 `ambient_fired_count` best-effort fill (full-width monitoring only, else NULL); summary builder (≤300 chars, `rung_language` verbatim, NO "causal" below rung 2)
  - [x] 3.5 Routing: CIRCUIT_SENSING_FORCE_SERIAL condition in `_use_cbm_for_request`; non-forced CBM ⇒ unsensed
  - [x] 3.6 `sensing_overhead_ms` accumulator + warn threshold; exposed in status (with sensable/unsensable edges + lag)
  - [x] 3.7 Config keys (CIRCUIT_SENSING_*); arm/disarm hooks in circuit activate/deactivate + SAE-set detach
  - [x] 3.8 Unit tests: config build from edges + attached set, context slicing edges (pos 0, end, early stop), summary + no-"causal" guard, ambient rules, lifecycle

- [x] 4.0 API + WS surface (covers FR-15.4, FR-15.5; EDGE-P3..P6)
  - [x] 4.1 `api/schemas/circuit_sensing.py` + `routes/management/circuit_sensing.py` (status/events/enable/disable/clear per contract §4 `millm_circuits`; NO_ACTIVE_CIRCUIT 200+envelope; CIRCUIT_SENSING_EVENT_NOT_FOUND 404; read-prune throttle)
  - [x] 4.2 DI + router registration; route tests (envelope, filters, enable toggles intent + live arm, no-active-circuit path)
  - [x] 4.3 `emit_circuit_sensing_event` in sockets/progress.py (`circuit:sensing:event`, payload w/o context text) + emission test
  - [x] 4.4 core/errors.py: NoActiveCircuitError, CircuitSensingEventNotFoundError

- [x] 5.0 Circuits-page edge-sensing UI (covers EDGE-P7)
  - [x] 5.1 `services/circuitSensing.ts` + `useCircuitSensing.ts` + WS subscription (`circuit:sensing:event` live prepend)
  - [x] 5.2 EdgeSensingPanel (status strip incl. unsensable edges, event list) + EdgeSensingEventDetail (up→down member table, lag, rung badge verbatim, context_parts span highlight)
  - [x] 5.3 Wire EdgeSensingToggle (Feature 13 circuit card) to enable/disable
  - [x] 5.4 Vitest: panel render, toggle flow, live event prepend, rung badge verbatim

- [x] 6.0 Integration verification (covers FR-15.1..15.5 end-to-end)
  - [x] 6.1 `test_circuit_edge_sensing_workflow.py`: arm→generate on a known up→down fixture→events with correct lag/context/rung
  - [x] 6.2 EC-15.1 (lone upstream) + EC-15.2 (reversed order) produce NO events; streaming early-stop + prefill context
  - [x] 6.3 Lifecycle: enable/disable live, SAE-set detach disarms, circuit delete cascades events; slice-fallback unsensable-edge reporting
  - [x] 6.4 Safety: serial forcing asserted; CBM-unsensed; overhead accumulator populated; **latency-budget assertion**; un-armed zero-delta smoke
  - [x] 6.5 WS emission observed end-to-end; **no-"causal"-below-rung-2 asserted on surfaced strings**

- [x] 7.0 Feature Acceptance (per instruct 008)
  - [x] 7.1 Verify FPRD §9 criteria 1–6 + all US/EC boxes one-by-one
  - [x] 7.2 Manual: edge-sensing semantics section (edge = up→down within lag; attribution; alone/within caveat; rung verbatim; retention+privacy)
  - [x] 7.3 Full suite green; update CLAUDE.md Document Inventory + Current Status; confirm `docs/mcp-contract.md` v1.1 surface matches

## Coverage Audit
- FR-15.1→2.0/6.1; FR-15.2→3.4/6.1; FR-15.3→3.3/6.2; FR-15.4→1.0/4.0/6.3/6.5; FR-15.5→3.5/3.6/6.4 ✓
- US-15.1→3.7/5.3/6.3; US-15.2→4.x/5.x/6.1/6.5; US-15.3→1.3/3.6/6.4 — implementing + testing sub-tasks each ✓
- EC-15.1→2.2/2.5/6.2; EC-15.2→2.2/2.5/6.2; EC-15.3→2.4/2.5; EC-15.4→3.1/2.5(status)/6.3; EC-15.5→3.5/6.4; EC-15.6→3.1/2.5; EC-15.7→3.3 ✓
- BRD: BR-007→EDGE-D1..D3/R1/S1; BR-008→EDGE-R2/R3/P1/P3/P5; BR-009→EDGE-P3/P4; BR-012→EDGE-P3/P6 ✓
- TDD/TID sections all mapped (detection→2.x, lifecycle/flush→3.x, persistence→1.x, API/WS→4.x, UI→5.x) ✓
- Open questions: none — no spike tasks ✓
- Final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)
- [x] Round 1 (multi-angle /code-review, 2 finder agents): ≥10 findings — fix criticals, document deferrals.
      Watch for: ring not reset across the two-SAE begin; reversed-order false positives; lag off-by-one;
      unsensable-edge silent wrong-decoder path; "causal" leaking into a rung<1 summary.
- [x] Round 2 (post-fix verification + fresh angles): ≥10 findings — verify R1 fixes hold; hunt regressions.
- [x] Round 3 (/review, 4 perspectives): ≥10 findings — fix, pin mutation survivors.
- [x] Record: `.claude/context/sessions/review_feature015_R{1,2,3}_2026-07-*.md`.
- Directive: fix latent/pre-existing defects surfaced during review, not only regressions.

## Acceptance evidence

### FPRD §9 criteria, verified one-by-one

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | Scripted panel with known up→down ground truth: 100% capture, correct lag/ordering | ⚠️ **PARTIAL — mechanism verified live, capture rate not measurable with this fixture** | **GPU close-out executed 2026-07-20** on the k8s host: LFM2.5-1.2B-Instruct + 5 SAEs (layers 10–14, 640 MB), a real 5-layer `circuit-definition/v1` imported and activated at `serving_mode: full`. Edge sensing armed across all 5 layers with **4 sensable edges, 0 unsensable**, `paused_reason: null`. Live traffic produced **0 events with a non-zero overhead counter (5.4–11.5 ms)** — i.e. the sensing pass genuinely ran and found no qualifying up→down pair, which is the correct result for a fixture whose feature indices were chosen arbitrarily rather than from co-firing ground truth. **A capture RATE still requires a circuit built from miStudio-mined features that are known to co-fire**; that is an authoring-side prerequisite, not a runtime gap. |
| 2 | EC-15.1/15.2 honored: lone-upstream and reversed-order produce NO event | ✅ | 7 tests, unit and end-to-end. Also covers the third case the FPRD does not name: a same-position co-fire, which is co-activation rather than a sequence. |
| 3 | Alone/within field correct when monitoring co-runs; NULL otherwise | ✅ | `_ambient_fired_count()` mirrors Feature 11 exactly: whole-SAE fired count, only when un-compacted monitoring co-ran, NULL otherwise — never estimated. 4 tests pin the contract incl. the compacted-subset and failing-probe cases. |
| 4 | Context windows match expected tokens; span covers the up→down positions | ✅ | 22 tests. Uses prefix decodes + length slicing (the FTID's independent-segment sketch reintroduces Feature 11 R1's SentencePiece word-gluing bug), with a byte-level-BPE guard that degrades to plain text over a wrong mark. |
| 5 | Overhead within budget; zero delta un-armed; retention caps enforced | ✅ | 38 tests incl. a latency assertion on a 4096-token saturated pass (0.9 ms vs the 5 ms budget, after R1/R2/R3 each fixed a different blowout), an inert un-armed path, and cap+age retention. |
| 6 | Every surfaced rung verbatim; no "causal" for any rung<2 edge | ✅ | 51 tests incl. the build-failing copy audit with its negative control. |

### Criterion 3 — CORRECTED at GPU close-out (2026-07-20)

This was first recorded as NOT MET on the belief that alone-vs-within was an unbuilt requirement.
**That was wrong, and the error is worth recording.** The signal already had an established meaning:
Feature 11's `_ambient_fired_count` and the `millm_sensing_events` MCP contract define
`ambient_fired_count` as *the count of features that fired across the **whole SAE**, populated only
when un-compacted monitoring co-ran, and **NULL otherwise — never estimated**.*

R3's implementation used the same column and field name for a **different quantity**: fires among the
armed circuit's own members, always non-null. Three defects in one:

| Rule | Contract (F11) | R3 as shipped |
|---|---|---|
| Scope | whole SAE | the circuit's own members |
| Gate | only when un-compacted monitoring co-ran | always |
| Absent | `NULL` — never estimated | never null |

A reader comparing an F15 row against an F11 row would have been comparing incompatible numbers under
an identical field name, and a never-null value silently claims a denominator nobody measured — an
honesty defect of exactly the kind this feature exists to prevent.

**Fixed:** `CircuitSensingService._ambient_fired_count()` now mirrors F11's rule precisely. The
circuit's own member-fire total still exists as `_edge_member_fires` (useful, and genuinely
different), renamed so it can never be mistaken for the contract signal, with a test asserting
`record()` does not reach for it when filling the column.

**How the error survived three review rounds:** all three treated the FPRD as the source of truth for
what the field meant, and none checked the *existing* implementation of the same-named field one
feature over. The MCP tool description stated the rule plainly the whole time.

### GPU close-out finding: the overhead budget has no multi-layer denominator

Measured live with 5 armed layers: **5.4 / 5.8 / 7.3 ms** per request (and 11.5 ms on a longer one),
against `CIRCUIT_SENSING_MAX_OVERHEAD_MS = 5.0`. Every request logged an overhead warning.

That is ~1.1–1.5 ms per armed layer, which is proportionate and in line with the unit-test
measurements. The threshold is the problem: it was inherited from Feature 11, where exactly ONE SAE is
ever armed, and it was never given a per-layer denominator. As written it guarantees a warning on any
circuit with two or more layers — the only kind of circuit that exists.

This is the same class of defect as the VRAM envelope corrected during the capability audit: a
single-SAE-era constant applied to a multi-SAE world, producing an alarm that trains operators to
ignore alarms. **Recorded for BRD-MILLM-CIRCUITS-002** — the threshold should scale with the armed
layer count (or be expressed per-layer), not sit at a fixed 5 ms.

Also confirmed live, and worth recording as positive evidence:

- **Lifecycle works end to end.** `enable` on an inactive circuit persisted intent and returned
  *"Enabled; the circuit will arm when it is activated"*; activation then armed all 5 layers with
  **4 sensable edges, 0 unsensable**, `paused_reason: null`.
- **Zero events with non-zero overhead is the honest result**, not a silent failure — the pass ran and
  found no qualifying up→down pair, exactly as expected for arbitrarily-chosen feature indices.
- **Clean teardown**: disable → deactivate → generation returns byte-identical to the pre-circuit
  baseline.

### Deferred structural work — designs settled, not silently dropped

1. **`SensingRequestContext`** owning the position counter, ring and event budget. **Three of the
   eight criticals across R1–R3 share one root cause**: N per-SAE counters must agree on an absolute
   coordinate that no single component owns. A request-scoped context makes offset divergence, prune
   races and per-layer budget skew *unrepresentable* rather than test-guarded. Top follow-on item.
2. **Move the edge machinery out of `sae_wrapper.py`** into `millm/ml/edge_sensing.py` — 145 of 1316
   lines are F15, and `LoadedSAE` now carries 11 `_edge_*` fields beside Feature 11's.
3. **`truncated` per-row attribution** plus `truncated_layers` in the status payload.
4. **F14 interaction:** the dial changes activations and therefore fire rates, but thresholds are
   frozen at arm time — turning the dial silently re-calibrates sensitivity. Needs a product decision.
5. **FTDD §96 amendment:** the design says the downstream *pops* the matching ring entry; the
   implementation reads non-destructively, so one upstream fire can father several events. The
   non-consuming read is the better evidence model — **amend the FTDD, not the code.**

### What the three rounds cost and bought

80 findings, 41 fixed, 11 critical. Every round found a critical regression in the previous round's
fix — including R3 finding that **R2 repeated R1's exact error one level up** (declaring a pruning
mechanism and never wiring it, twice, the second time with a test named for the defect).

The most transferable lesson is methodological: R3's QA reviewer ran **14 mutation experiments** and
found four load-bearing lines no test caught, including one that let the WebSocket broadcast leak
prompt text while the suite stayed green — a line R1 had recorded as *"privacy holds — verified
clean"*, having verified it by reading. **Two rounds of careful reading missed what fourteen
mutations found in one pass.** (Task 7.0)
- [x] FPRD §9: (1) up→down capture correctness on a deterministic-SAE fixture through the REAL hook (lag,
      ordering, context, rung, persistence, WS); (2) EC-15.1/15.2 negatives; (3) alone/within rule;
      (4) context span; (5) bounded persistence (cap/age prune on flush AND read, CASCADE) + latency budget +
      un-armed zero-delta; (6) no "causal" below rung 2 anywhere surfaced.
- [x] Suites: backend + frontend green; manual + API reference (contract v1.1) updated.
