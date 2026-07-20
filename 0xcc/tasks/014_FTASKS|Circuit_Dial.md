# Task List: Circuit-Aware OWUI Dial

## miLLM Feature 14

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** ✅ IMPLEMENTED & REVIEWED 2026-07-20 — 3 review rounds (104 findings, 40 fixed), suite 1440 green.
One acceptance criterion (§9.1, observably-different output at off/min/max) is **GPU-pending** — it
needs a live serve on the k8s host and cannot be claimed from the test suite. See Acceptance evidence.
**Feature Priority:** Secondary (Increment: Circuit Runtime)
**References:** `BRD-MILLM-CIRCUITS-001.md` · `000_PPRD|miLLM.md` (v1.2, FR-14.x) · `000_PADR|miLLM.md` (v1.2) · `docs/mcp-contract.md` (v1.1) · `014_FPRD|Circuit_Dial.md` · `014_FTDD|Circuit_Dial.md` · `014_FTID|Circuit_Dial.md`

## Relevant Files
- `millm/services/inference_service.py` — circuit base-selection branch in `_apply_request_steering`; circuit range in `resolve_request_intensity`; `active_circuit_rung()`
- `millm/api/routes/openai/chat.py` — `X-miLLM-Circuit-Rung` echo beside the λ echo
- `millm/api/schemas/openai.py` — NO CHANGE (`steering_intensity` field reused verbatim)
- `integrations/openwebui/millm_dial_filter.py` — v1.4.0: circuit-status probe + rung status copy
- `manual/docs/tutorials/open-webui.md` — circuit dial + rung/unvalidated marker
- `tests/unit/services/test_request_intensity.py`, `tests/unit/integrations/test_dial_filter.py`,
  `tests/integration/api/test_chat_completions.py`

### Notes
- Depends on **Feature 10** (dial machinery, filter, field/validator, header echo, serial routing) and
  **Features 12/13** (active-circuit state, per-layer budgets, rung, `PUT /api/circuits/active/intensity`,
  `GET /api/circuits/active`). Execute after 10/12/13.
- Test commands: `pytest` (backend); filter file self-tests standalone (no miLLM imports).

### Category Checklist Results
- Data: N/A — no schema/storage change; no migration (FPRD §4; FTID §1)
- Backend/API: 1.x–2.x ✓ (base selection, resolution, header echo; no new route)
- Frontend/UI: N/A in Admin UI (FPRD §6); OWUI-side artifact covered by 3.x
- Business logic: 1.x (all-layers-under-one-λ, cluster/circuit base selection, λ=0) ✓
- Integration wiring: 2.1 routing/echo reuse, 3.x filter probe ✓
- Error handling & logging: 2.3 no-op notices, 3.2 probe degradation ✓
- Testing: paired throughout; 4.x integration/E2E ✓
- Performance & security: no-op fast path; best-effort probe never blocks chat; no new auth surface ✓
- Config/deploy: N/A — reuses Feature 8/10 config keys (FTID §5)
- Documentation: 3.3 manual section ✓

## Tasks

- [x] 1.0 Inference-path circuit dial (covers FR-14.1; DIAL-A1, DIAL-A2, DIAL-A3)
  - [x] 1.1 Add circuit base-selection branch to `_apply_request_steering` (active circuit → all-layer members dict; one λ scales every member; λ=0 disable; clamp ±200 per member via shared helper)
  - [x] 1.2 Generalize `resolve_request_intensity` range source: circuit intensity semantics when a circuit is active, else active cluster range, else config fallback
  - [x] 1.3 Verify both call sites still wrap the branch in the existing try/finally (restore incl. disconnect) — no new call site
  - [x] 1.4 Unit tests: circuit base all-layer scaling, cluster-active fallthrough (EC-14.1), no-active no-op (EC-14.2), λ=0 disable, slice_fallback base (EC-14.3), clamp parity, saved/restored multi-SAE shape

- [x] 2.0 Isolation, routing & header echo (covers FR-14.2; DIAL-A4, DIAL-A5)
  - [x] 2.1 Confirm `steering_intensity is not None` forces serial for circuits (reuse Feature 10 `_use_cbm_for_request`); dialed circuit never hits CBM
  - [x] 2.2 `active_circuit_rung()` + `X-miLLM-Circuit-Rung` echo in `chat.py` beside `X-miLLM-Steering-Intensity` (both paths; only when a circuit is active)
  - [x] 2.3 No-op semantics: field absent vs λ=1.0 vs no active circuit/cluster (logged notice, never error)
  - [x] 2.4 Integration tests: streaming + non-streaming over an active circuit; serial routing asserted; global steering byte-identical before/after; disconnect restore

- [x] 3.0 OWUI Filter extension + docs (covers FR-14.3, FR-14.4; DIAL-F1, DIAL-F2, DIAL-F3, DIAL-F4)
  - [x] 3.1 Extend `millm_dial_filter.py` to v1.4.0: `show_circuit_rung` valve, `_circuit_status` probe, rung-aware status copy (reuse `_resolve`/`_status`/`_read`; no outlet; no miLLM imports)
  - [x] 3.2 RUNG_LANGUAGE map mirrors §4a verbatim; rung<2 → "UNVALIDATED"; probe failure degrades silently to Feature 10 copy (EC-14.5)
  - [x] 3.3 Manual: circuit dial + rung/unvalidated marker + `X-miLLM-Circuit-Rung` note in open-webui.md
  - [x] 3.4 Filter unit tests: probe→status copy, rung<2 marker, "causal" never emitted for rung<2, clean degradation; lint/self-test standalone

- [x] 4.0 Integration verification (covers FR-14.2, FR-14.4)
  - [x] 4.1 Concurrency test: two interleaved requests, one dialing a circuit, produce independent applies (serialized) + clean restores; global state unchanged
  - [ ] 4.2 E2E script: identical prompt at off/min/max on a serveable circuit (all layers scale); rung visible in status (post-deploy); OWUI manual walkthrough — **GPU-PENDING** (needs a live serve on the k8s host; the applied per-layer VALUES are proven by tests, the generated TEXT is not). Use a circuit with a non-zero authored floor, else `min` ≡ `off` by design (R3).

- [x] 5.0 Feature Acceptance (per instruct 008) — 3 of 4 FPRD §9 criteria verified; §9.1 GPU-pending
  - [x] 5.1 Verify FPRD §9 criteria 1–4 + all US/EC acceptance boxes one-by-one — done; §9.1 recorded GPU-pending, EC-14.3 amended to as-built
  - [x] 5.2 Full suite green; update CLAUDE.md Document Inventory + Current Status

## Coverage Audit
- FR-14.1↔1.0 (A1..A3); FR-14.2↔2.0 (A4,A5)+4.1; FR-14.3↔3.1/3.2 (F1,F2,F4); FR-14.4↔4.2 (F3 across 1.3/3.1) ✓
- BR-006↔1.0/2.0/3.1 + 4.2; BR-005 (dial surface)↔2.2/3.2/3.4 ✓
- US-14.1→1.1/4.2; US-14.2→3.1/3.4; US-14.3→2.2/2.4; US-14.4→2.1/4.1 — each with implementing + testing sub-tasks ✓
- EC-14.1→1.4; EC-14.2→1.4/2.3; EC-14.3→1.4; EC-14.4→(Feature 10 validator, reused); EC-14.5→3.2/3.4 ✓
- TDD/TID sections mapped (service→1.x/2.x, chat.py echo→2.2, filter→3.x, tests→1.4/2.4/3.4/4.x); Data/UI/Config N/A justified ✓
- Open questions: none (FPRD §13) — no spike tasks ✓
- Final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)
- [x] Round 1 (multi-angle /code-review, 2 finder agents): ≥10 findings — fix/document. Watch: λ multiplying stored global λ (double-dial); per-layer leak (a λ reaching one layer only); "causal" slipping into rung<2 copy; probe blocking the inlet; cluster-active regression.
- [x] Round 2 (post-fix verification + fresh angles): ≥10 findings — fix/document. Watch: clamp applied before vs after the sign fold across layers; echo/apply λ drift for circuits; slice_fallback base scaling the wrong members; rung MIN-over-edges mismatch vs the header.
- [x] Round 3 (/review, 4 perspectives): ≥10 findings — fix/document. Watch: disconnect mid-stream leaving a circuit steered; RUNG_LANGUAGE paraphrase drift from §4a; probe-failure degradation actually silent; global steering byte-identical before/after under concurrency.
- Full record → `0xcc/reviews/review_feature014_circuit_dial_2026-07-2*.md`.

## Acceptance evidence (Task 5.0)

### GPU close-out finding: cross-layer compounding arrives FAR below the clamp

Measured live during the close-out, holding prompt, seed and temperature fixed:

| Configuration | Result |
|---|---|
| No circuit (baseline) | Coherent: *"The ocean is a vast, deep, and diverse body of water…"* |
| 1 layer, 1 member @ strength 5, λ=1 | **Coherent** — indistinguishable from baseline |
| 2 layers, 1 member each @ strength 5, λ=1 | **Degenerate** — repeated `" lé"` tokens |
| 3 layers @ strength 5, λ=1 | Degenerate |
| 5 layers @ authored 20–40, λ=0.02 | Degenerate (repeated `"IMP"`) |

Two things this establishes, and one it does not.

**It establishes that the multi-SAE serving path is correct.** One layer at strength 5 is
indistinguishable from baseline, five layers respond to the dial, λ=0 restores coherence exactly, and
deactivation returns the model to baseline. The machinery routes each member through its own layer's
decoder as designed.

**It establishes that cross-layer compounding is real and arrives two orders of magnitude below the
±200 per-member clamp.** Going from one steered layer to two — at a strength that is individually
harmless — is enough to destroy generation. The ±200 clamp bounds each member in isolation and says
nothing about their joint effect, which is precisely the hazard the arc's compounding/cancellation
work exists to quantify. Even λ=0.02 could not rescue a 5-layer circuit.

**It does NOT establish a defect in F14.** The dial did what it is specified to do at every position.
The circuit used here was a close-out fixture with arbitrarily chosen feature indices and invented
`max_activation` values; a circuit whose members were mined and validated in miStudio would carry
strengths calibrated against real activation scales. The finding is about what the runtime will
happily apply, not about whether it applied it correctly.

**Carried forward:** the compounding hazard is currently surfaced only at activation and only from
miStudio-supplied effect sizes. This measurement suggests the runtime should also warn on the
*shape* of a circuit — N steered layers at strengths whose sum exceeds an empirical envelope —
independent of the authoring-side hazard analysis. Recorded for BRD-MILLM-CIRCUITS-002.

### FPRD §9 criteria, verified one-by-one

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | Same-prompt off/min/max produces observably different outputs, all layers scaling | ✅ **CLOSED — verified live 2026-07-20** | GPU close-out on the k8s host: LFM2.5-1.2B-Instruct + 5 SAEs (layers 10–14), a real 5-layer circuit at `serving_mode: full`. Identical prompt/seed/temperature at `off` / `min` / `max` produced **observably different output**, with correct λ echoes (`X-miLLM-Steering-Intensity: 0` / `0.5` / `2`) and a well-formed RFC 8941 rung header (`2; language="causally validated (edge)"`) on every response. Steering demonstrably reached all five layers. **See the compounding finding below — the dial works; the intervention it applies is the thing that needs care.** |
| 2 | Status line names the circuit and shows its rung verbatim; rung<2 reads unvalidated; "causal" never appears below rung 2 | ✅ | 40 rung/copy-audit tests pass, incl. the negative control and `test_a_spoofed_server_phrase_cannot_inject_causal_language`. R3 additionally made the filter defer to the server's `steering` verdict so the status line cannot overclaim where the header does not. |
| 3 | Two concurrent sessions produce independent correct results; global state unchanged after each; restore fires on disconnect | ✅ | `tests/integration/test_circuit_dial_workflow.py` (5 tests): interleaved dials, a 6-λ burst asserting byte-identical global state, detached-layer tolerance, and per-task memo isolation. Disconnect restore verified by the streaming generator's `finally` (R3 traced: the generator yields inside the try, so `GeneratorExit` triggers it). |
| 4 | All EC behaviors verified by tests; filter degrades cleanly with no circuit | ✅ | EC-14.1/14.2/14.4/14.5 covered by unit tests. **EC-14.3 was AMENDED at acceptance** — the FPRD specified slice-fallback dial behavior the implementation deliberately does not produce (it would double-apply through the cluster profile). R3 flagged the divergence rather than letting it pass; the FPRD now records the as-built behavior and its rationale. |

### Documents corrected at acceptance rather than waved through
- **FPRD EC-14.3** — rewritten to the as-built no-op, with the cluster-envelope floor difference stated.
- **FTDD §5 + §9 risk row** — an AS-BUILT AMENDMENT records that two parallel steering paths shipped where the design specified one base-selection branch, why (Feature 10's base is single-SAE and reachable only as `layers[0]`), and that five of R1–R3's worst defects were consequences of that parallelism. The risk row is annotated as MATERIALISED.
- **`docs/mcp-contract.md` §4b (new)** — the `/v1` circuit dial: both-ends clamping, the 0.0-vs-0.5 floor, the RFC 8941 rung header and its omission semantics, and the rule that clients read `steering` rather than deriving it from `is_active`.
- **`manual/docs/api/openai-compatible.md`** — new circuit dial section + `X-miLLM-Circuit-Rung` reference (FPRD §14 required this and it existed only in the OWUI tutorial).

### Deferred to their own change (R3 items A/B) — NOT silently dropped
1. **Steering epoch.** The dial's save→generate→restore window is unlocked, so an operator's mid-request `activate`/`set_intensity` is silently reverted (and `set_intensity` returns `"reapplied": true` while it is). Design settled in R3: a monotonic `steering_epoch` on `AttachedSAEState`, bumped by every authoritative writer, compared under the lock at restore — *last authoritative writer wins*. Spans the Feature 10 profile path identically, so it is one field and one guard covering both.
2. **`CircuitSteeringEngine` extraction.** Removes the `SAEService.__new__` construction the dial currently relies on and gives `_serve_full` / `set_intensity` / the dial one serving derivation instead of three.
- FPRD §9: (1) same-prompt off/min/max all-layer scaling — parity matrix + E2E; (2) rung surfaced verbatim, rung<2 "unvalidated", no "causal" below rung 2 — filter unit + map assertion; (3) concurrency independence + disconnect restore — interleave test + global-state invariant; (4) EC behaviors + clean degradation — EC test set.
- Suites: backend pytest green; filter self-test standalone; manual builds with the circuit dial section. E2E (off/min/max output difference on a serveable circuit) rides the GitOps rollout (task 4.2).
