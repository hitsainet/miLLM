# Task List: Circuit-Aware OWUI Dial

## miLLM Feature 14

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** 📋 DOCS COMPLETE 2026-07-20 — implementation not started
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

- [ ] 1.0 Inference-path circuit dial (covers FR-14.1; DIAL-A1, DIAL-A2, DIAL-A3)
  - [ ] 1.1 Add circuit base-selection branch to `_apply_request_steering` (active circuit → all-layer members dict; one λ scales every member; λ=0 disable; clamp ±200 per member via shared helper)
  - [ ] 1.2 Generalize `resolve_request_intensity` range source: circuit intensity semantics when a circuit is active, else active cluster range, else config fallback
  - [ ] 1.3 Verify both call sites still wrap the branch in the existing try/finally (restore incl. disconnect) — no new call site
  - [ ] 1.4 Unit tests: circuit base all-layer scaling, cluster-active fallthrough (EC-14.1), no-active no-op (EC-14.2), λ=0 disable, slice_fallback base (EC-14.3), clamp parity, saved/restored multi-SAE shape

- [ ] 2.0 Isolation, routing & header echo (covers FR-14.2; DIAL-A4, DIAL-A5)
  - [ ] 2.1 Confirm `steering_intensity is not None` forces serial for circuits (reuse Feature 10 `_use_cbm_for_request`); dialed circuit never hits CBM
  - [ ] 2.2 `active_circuit_rung()` + `X-miLLM-Circuit-Rung` echo in `chat.py` beside `X-miLLM-Steering-Intensity` (both paths; only when a circuit is active)
  - [ ] 2.3 No-op semantics: field absent vs λ=1.0 vs no active circuit/cluster (logged notice, never error)
  - [ ] 2.4 Integration tests: streaming + non-streaming over an active circuit; serial routing asserted; global steering byte-identical before/after; disconnect restore

- [ ] 3.0 OWUI Filter extension + docs (covers FR-14.3, FR-14.4; DIAL-F1, DIAL-F2, DIAL-F3, DIAL-F4)
  - [ ] 3.1 Extend `millm_dial_filter.py` to v1.4.0: `show_circuit_rung` valve, `_circuit_status` probe, rung-aware status copy (reuse `_resolve`/`_status`/`_read`; no outlet; no miLLM imports)
  - [ ] 3.2 RUNG_LANGUAGE map mirrors §4a verbatim; rung<2 → "UNVALIDATED"; probe failure degrades silently to Feature 10 copy (EC-14.5)
  - [ ] 3.3 Manual: circuit dial + rung/unvalidated marker + `X-miLLM-Circuit-Rung` note in open-webui.md
  - [ ] 3.4 Filter unit tests: probe→status copy, rung<2 marker, "causal" never emitted for rung<2, clean degradation; lint/self-test standalone

- [ ] 4.0 Integration verification (covers FR-14.2, FR-14.4)
  - [ ] 4.1 Concurrency test: two interleaved requests, one dialing a circuit, produce independent applies (serialized) + clean restores; global state unchanged
  - [ ] 4.2 E2E script: identical prompt at off/min/max on a serveable circuit (all layers scale); rung visible in status (post-deploy); OWUI manual walkthrough

- [ ] 5.0 Feature Acceptance (per instruct 008)
  - [ ] 5.1 Verify FPRD §9 criteria 1–4 + all US/EC acceptance boxes one-by-one
  - [ ] 5.2 Full suite green; update CLAUDE.md Document Inventory + Current Status

## Coverage Audit
- FR-14.1↔1.0 (A1..A3); FR-14.2↔2.0 (A4,A5)+4.1; FR-14.3↔3.1/3.2 (F1,F2,F4); FR-14.4↔4.2 (F3 across 1.3/3.1) ✓
- BR-006↔1.0/2.0/3.1 + 4.2; BR-005 (dial surface)↔2.2/3.2/3.4 ✓
- US-14.1→1.1/4.2; US-14.2→3.1/3.4; US-14.3→2.2/2.4; US-14.4→2.1/4.1 — each with implementing + testing sub-tasks ✓
- EC-14.1→1.4; EC-14.2→1.4/2.3; EC-14.3→1.4; EC-14.4→(Feature 10 validator, reused); EC-14.5→3.2/3.4 ✓
- TDD/TID sections mapped (service→1.x/2.x, chat.py echo→2.2, filter→3.x, tests→1.4/2.4/3.4/4.x); Data/UI/Config N/A justified ✓
- Open questions: none (FPRD §13) — no spike tasks ✓
- Final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)
- [ ] Round 1 (multi-angle /code-review, 2 finder agents): ≥10 findings — fix/document. Watch: λ multiplying stored global λ (double-dial); per-layer leak (a λ reaching one layer only); "causal" slipping into rung<2 copy; probe blocking the inlet; cluster-active regression.
- [ ] Round 2 (post-fix verification + fresh angles): ≥10 findings — fix/document. Watch: clamp applied before vs after the sign fold across layers; echo/apply λ drift for circuits; slice_fallback base scaling the wrong members; rung MIN-over-edges mismatch vs the header.
- [ ] Round 3 (/review, 4 perspectives): ≥10 findings — fix/document. Watch: disconnect mid-stream leaving a circuit steered; RUNG_LANGUAGE paraphrase drift from §4a; probe-failure degradation actually silent; global steering byte-identical before/after under concurrency.
- Full record → `0xcc/reviews/review_feature014_circuit_dial_2026-07-2*.md`.

## Acceptance evidence (Task 5.0)
- FPRD §9: (1) same-prompt off/min/max all-layer scaling — parity matrix + E2E; (2) rung surfaced verbatim, rung<2 "unvalidated", no "causal" below rung 2 — filter unit + map assertion; (3) concurrency independence + disconnect restore — interleave test + global-state invariant; (4) EC behaviors + clean degradation — EC test set.
- Suites: backend pytest green; filter self-test standalone; manual builds with the circuit dial section. E2E (off/min/max output difference on a serveable circuit) rides the GitOps rollout (task 4.2).
