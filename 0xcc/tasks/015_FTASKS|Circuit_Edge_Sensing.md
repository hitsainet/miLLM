# Task List: Circuit Edge Sensing

## miLLM Feature 15

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** 📋 DOCS COMPLETE 2026-07-20 — implementation not started
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

- [ ] 1.0 Persistence layer (covers FR-15.4; EDGE-P1, EDGE-P2)
  - [ ] 1.1 Migration `012_add_circuit_edge_sensing.py` (up+down, indexes, CASCADE via circuit ownership; `down_revision` = Feature 13's `011`)
  - [ ] 1.2 `db/models/circuit_edge_sensing_event.py` (mirror `sensing_event.py`; up/down member+layer+pos+act, token_lag, edge_rung + rung_language, context_parts) + `circuit_edge_sensing_repository.py` (create_many/list_events/count/clear/prune)
  - [ ] 1.3 Repo unit tests incl. retention cap + age prune + CASCADE with circuit delete

- [ ] 2.0 Edge detection core in LoadedSAE (covers FR-15.1; EDGE-D1..D5)
  - [ ] 2.1 EdgeSpec/CircuitSensingConfig/SensedEdge dataclasses; arm_edge_sensing/disarm (reuse `_W_enc_m` cache, dtype/device parity); `is_edge_sensing_armed` kept distinct from `is_sensing_armed`
  - [ ] 2.2 `_sense_edges` per-pass fire reuse + up→down ring matcher (strict ordering, lag window) + suppressed() early-return
  - [ ] 2.3 Shared per-request EdgeRing across the circuit's SAEs; absolute-position lag matching across passes; offset/phase accounting; ring pruning older than L
  - [ ] 2.4 Per-request cap + truncated flag + `_sensing_done` fast path
  - [ ] 2.5 Unit tests: fire predicate reuse, up→down matcher, EC-15.1 (lone up → none), EC-15.2 (reversed → none), cross-layer close, cap, offsets, unsensable-edge exclusion (EC-15.4/15.6), suppressed, arm idempotence, buffer hygiene (begin resets once; no-begin ⇒ empty collect)
  - [ ] 2.6 Hook branch in `sae_hooker.hook_fn` (sibling of F11 sensing, before apply_steering)

- [ ] 3.0 Service + lifecycle + flush (covers FR-15.2, FR-15.3, FR-15.5; EDGE-R1..R4, EDGE-S1..S3)
  - [ ] 3.1 `circuit_sensing_service.py`: arm_for_circuit (per-SAE config, drop unsensable edges, record them), disarm/should_sense/collect_edges/status
  - [ ] 3.2 Inference wiring: begin across attached SAEs after apply, collect in finally, async `_notify_circuit_sensing` beside `_notify_sensing` (both paths)
  - [ ] 3.3 Context capture reuse (outputs[0]/IdCaptureStoppingCriteria/prompt ids); ±K `context_parts` span covering up→down, off hot path; K=0 path
  - [ ] 3.4 `ambient_fired_count` best-effort fill (full-width monitoring only, else NULL); summary builder (≤300 chars, `rung_language` verbatim, NO "causal" below rung 2)
  - [ ] 3.5 Routing: CIRCUIT_SENSING_FORCE_SERIAL condition in `_use_cbm_for_request`; non-forced CBM ⇒ unsensed
  - [ ] 3.6 `sensing_overhead_ms` accumulator + warn threshold; exposed in status (with sensable/unsensable edges + lag)
  - [ ] 3.7 Config keys (CIRCUIT_SENSING_*); arm/disarm hooks in circuit activate/deactivate + SAE-set detach
  - [ ] 3.8 Unit tests: config build from edges + attached set, context slicing edges (pos 0, end, early stop), summary + no-"causal" guard, ambient rules, lifecycle

- [ ] 4.0 API + WS surface (covers FR-15.4, FR-15.5; EDGE-P3..P6)
  - [ ] 4.1 `api/schemas/circuit_sensing.py` + `routes/management/circuit_sensing.py` (status/events/enable/disable/clear per contract §4 `millm_circuits`; NO_ACTIVE_CIRCUIT 200+envelope; CIRCUIT_SENSING_EVENT_NOT_FOUND 404; read-prune throttle)
  - [ ] 4.2 DI + router registration; route tests (envelope, filters, enable toggles intent + live arm, no-active-circuit path)
  - [ ] 4.3 `emit_circuit_sensing_event` in sockets/progress.py (`circuit:sensing:event`, payload w/o context text) + emission test
  - [ ] 4.4 core/errors.py: NoActiveCircuitError, CircuitSensingEventNotFoundError

- [ ] 5.0 Circuits-page edge-sensing UI (covers EDGE-P7)
  - [ ] 5.1 `services/circuitSensing.ts` + `useCircuitSensing.ts` + WS subscription (`circuit:sensing:event` live prepend)
  - [ ] 5.2 EdgeSensingPanel (status strip incl. unsensable edges, event list) + EdgeSensingEventDetail (up→down member table, lag, rung badge verbatim, context_parts span highlight)
  - [ ] 5.3 Wire EdgeSensingToggle (Feature 13 circuit card) to enable/disable
  - [ ] 5.4 Vitest: panel render, toggle flow, live event prepend, rung badge verbatim

- [ ] 6.0 Integration verification (covers FR-15.1..15.5 end-to-end)
  - [ ] 6.1 `test_circuit_edge_sensing_workflow.py`: arm→generate on a known up→down fixture→events with correct lag/context/rung
  - [ ] 6.2 EC-15.1 (lone upstream) + EC-15.2 (reversed order) produce NO events; streaming early-stop + prefill context
  - [ ] 6.3 Lifecycle: enable/disable live, SAE-set detach disarms, circuit delete cascades events; slice-fallback unsensable-edge reporting
  - [ ] 6.4 Safety: serial forcing asserted; CBM-unsensed; overhead accumulator populated; **latency-budget assertion**; un-armed zero-delta smoke
  - [ ] 6.5 WS emission observed end-to-end; **no-"causal"-below-rung-2 asserted on surfaced strings**

- [ ] 7.0 Feature Acceptance (per instruct 008)
  - [ ] 7.1 Verify FPRD §9 criteria 1–6 + all US/EC boxes one-by-one
  - [ ] 7.2 Manual: edge-sensing semantics section (edge = up→down within lag; attribution; alone/within caveat; rung verbatim; retention+privacy)
  - [ ] 7.3 Full suite green; update CLAUDE.md Document Inventory + Current Status; confirm `docs/mcp-contract.md` v1.1 surface matches

## Coverage Audit
- FR-15.1→2.0/6.1; FR-15.2→3.4/6.1; FR-15.3→3.3/6.2; FR-15.4→1.0/4.0/6.3/6.5; FR-15.5→3.5/3.6/6.4 ✓
- US-15.1→3.7/5.3/6.3; US-15.2→4.x/5.x/6.1/6.5; US-15.3→1.3/3.6/6.4 — implementing + testing sub-tasks each ✓
- EC-15.1→2.2/2.5/6.2; EC-15.2→2.2/2.5/6.2; EC-15.3→2.4/2.5; EC-15.4→3.1/2.5(status)/6.3; EC-15.5→3.5/6.4; EC-15.6→3.1/2.5; EC-15.7→3.3 ✓
- BRD: BR-007→EDGE-D1..D3/R1/S1; BR-008→EDGE-R2/R3/P1/P3/P5; BR-009→EDGE-P3/P4; BR-012→EDGE-P3/P6 ✓
- TDD/TID sections all mapped (detection→2.x, lifecycle/flush→3.x, persistence→1.x, API/WS→4.x, UI→5.x) ✓
- Open questions: none — no spike tasks ✓
- Final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)
- [ ] Round 1 (multi-angle /code-review, 2 finder agents): ≥10 findings — fix criticals, document deferrals.
      Watch for: ring not reset across the two-SAE begin; reversed-order false positives; lag off-by-one;
      unsensable-edge silent wrong-decoder path; "causal" leaking into a rung<1 summary.
- [ ] Round 2 (post-fix verification + fresh angles): ≥10 findings — verify R1 fixes hold; hunt regressions.
- [ ] Round 3 (/review, 4 perspectives): ≥10 findings — fix, pin mutation survivors.
- [ ] Record: `.claude/context/sessions/review_feature015_R{1,2,3}_2026-07-*.md`.
- Directive: fix latent/pre-existing defects surfaced during review, not only regressions.

## Acceptance evidence (Task 7.0)
- [ ] FPRD §9: (1) up→down capture correctness on a deterministic-SAE fixture through the REAL hook (lag,
      ordering, context, rung, persistence, WS); (2) EC-15.1/15.2 negatives; (3) alone/within rule;
      (4) context span; (5) bounded persistence (cap/age prune on flush AND read, CASCADE) + latency budget +
      un-armed zero-delta; (6) no "causal" below rung 2 anywhere surfaced.
- [ ] Suites: backend + frontend green; manual + API reference (contract v1.1) updated.
