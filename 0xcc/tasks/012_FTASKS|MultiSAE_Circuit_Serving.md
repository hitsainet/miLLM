# Task List: Multi-SAE Attach & Circuit Serving

## miLLM Feature 12

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** 📋 DOCS COMPLETE 2026-07-20 — implementation not started; VRAM spike done (128 MB fp16 / 256 MB fp32, two SAEs)
**References:** `012_FPRD|MultiSAE_Circuit_Serving.md` · `012_FTDD|MultiSAE_Circuit_Serving.md` · `012_FTID|MultiSAE_Circuit_Serving.md`

## Relevant Files

### Backend
- `millm/services/sae_service.py` — `AttachedSAEState` → registry keyed by `(sae_id, layer)`; `attach_set`, `set_circuit_steering`, `_cross_layer_hazards`; plural `AttachmentStatusSet`
- `millm/ml/sae_hooker.py` — UNCHANGED; `install()` invoked once per referenced `(sae_id, layer)`
- `millm/ml/sae_wrapper.py` — UNCHANGED; `apply_steering` already matches miStudio; per-SAE `_steering_values`
- `millm/ml/sae_loader.py` — UNCHANGED; `load(..., dtype=torch.float16)` for fp16 attach
- `millm/core/steering_range.py` — REUSE shared `clamp_steering` / `STEERING_RANGE` (Feature 8)
- `millm/core/errors.py` — `SAESetIncompleteError`
- `millm/api/schemas/sae.py` — `AttachmentStatusSet`, `AttachedEntry` DTOs
- `millm/api/schemas/circuit.py` — `CircuitMember` DTO (shared with Feature 013)
- `millm/api/routes/management/sae.py` — `/attachments`, `/attach-set`, `/detach`
- `millm/core/config.py` — `MULTISAE_VRAM_ENVELOPE_MB`, `MULTISAE_ATTACH_DTYPE`, `CIRCUIT_INTENSITY_MIN/MAX`

### Frontend
- `admin-ui/src/components/circuits/AttachmentPanel.tsx` — plural `(sae_id, layer)` chips + total + VRAM badge
- `admin-ui/src/hooks/useAttachments.ts`, `admin-ui/src/services/sae.ts` — plural attachment shape

### Tests
- `tests/unit/services/test_attached_state_registry.py`, `test_circuit_steering.py`, `test_attach_set.py`
- `tests/integration/test_multisae_serving.py`

### Notes
- Follow `007_process-task-list.md`: one sub-task at a time; full suite + commit per parent task.
- Test commands: `pytest` (backend), `npm test` in admin-ui (Vitest).
- No migration — attachment state is in-memory (the singleton generalizes; nothing persisted).

### Category Checklist Results
- Data: no DB table (in-memory registry) — task 2.x ✓
- Backend/API: tasks 2.x–4.x ✓
- Frontend/UI: task 5.x ✓
- Business logic: tasks 3.x (group-by-layer, budgets/λ, clamp, hazards) ✓
- Integration wiring: task 4.x (routes/DTOs on existing SAE router + DI) ✓
- Error handling & logging: 3.4 (`SAE_SET_INCOMPLETE`, bounds gate), orphaned-hook guard ✓
- Testing: every parent has paired test sub-tasks; 6.x integration ✓
- Performance & security: 1.0 (VRAM spike), 6.3 (latency/VRAM harness), data-only posture inherited ✓
- Config/deploy: 4.4 (config keys); no migration ✓
- Documentation: 7.2 (manual + mcp-contract cross-ref) ✓

## Tasks

- [x] 1.0 Multi-SAE VRAM spike (run-before-build) (covers BR-001, FR-12.5; MSA-V1) — **DONE 2026-07-20**
  - [x] 1.1 On the RTX-3090 k8s host, load two distinct-layer real Gemma-2-2B SAEs (d_in=2048, d_sae=8192) and measure steering-tensor VRAM
  - [x] 1.2 **Result:** two SAEs = **256 MB fp32 (EXCEEDS the <200 MB envelope) / 128 MB fp16 (WITHIN it)** — ~128 MB/SAE fp32, ~64 MB/SAE fp16, linear in SAE count
  - [x] 1.3 **Decision:** attach steering weights in fp16 (`SAELoader` already casts to `target_dtype`); VRAM is dtype-conditional, not a blocker for 2–3 layer circuits → folded into FTDD §1 decision table + FTID config

- [ ] 2.0 Generalize attachment state to a registry (covers FR-12.1, FR-12.7; MSA-A1, MSA-A3, MSA-A4)
  - [ ] 2.1 `AttachedSAEState`: replace the 4 scalar fields with `_entries: dict[(sae_id, layer) → AttachedEntry]`; `set()` (per-key, keep orphaned-hook guard), `clear(sae_id?, layer?)`, `get`, `by_layer` (unique-or-None), `entries()`
  - [ ] 2.2 `AttachmentStatusSet` + `AttachedEntry` DTOs; derive legacy singular fields from `entries[0]` (back-compat)
  - [ ] 2.3 Unit tests: set/clear/idempotent-reattach per key; `by_layer` uniqueness (ambiguous → None); orphaned-hook removal

- [ ] 3.0 Circuit serving service (covers FR-12.2, FR-12.3, FR-12.4, FR-12.6; MSA-S1..S5, MSA-H1, MSA-H2)
  - [ ] 3.1 `set_circuit_steering(members, intensity)`: resolve via `by_layer`, group by `(sae_id, layer)`, per-layer `set_steering_batch` with only that layer's members
  - [ ] 3.2 Per-layer budgets under one global λ: `clamp_steering(budget·sign·λ)`, γ=0 ⇒ B=B_dir; bounds pre-check against THAT layer's `d_sae` (never a 500)
  - [ ] 3.3 `_cross_layer_hazards`: compounding/cancellation, labeled `validated:ES=…` / `heuristic:weight_prior=…`; returned only, config never mutated
  - [ ] 3.4 `SAESetIncompleteError` (422 `SAE_SET_INCOMPLETE`) with `{feature_idx, layer, sae_id}` offenders; `deactivate` clears steering on every participating layer
  - [ ] 3.5 Unit tests: group-by-layer mapping, clamp math, γ=0, hazard labeling, config-not-mutated, incomplete offender list, bounds gate

- [ ] 4.0 Attach + serve API surface (covers FR-12.1, FR-12.7 API)
  - [ ] 4.1 `attach_set(sae_layers)`: referenced-only load in fp16, one hook per `(sae_id, layer)`, total memory + VRAM warning (EC-12.3)
  - [ ] 4.2 Routes on existing SAE router: `GET /api/sae/attachments`, `POST /api/sae/attach-set`, `POST /api/sae/detach`
  - [ ] 4.3 Route tests (service mocked): plural status shape, attach-set idempotency, detach one-vs-all, `SAE_SET_INCOMPLETE` → 422 envelope
  - [ ] 4.4 Config keys (`MULTISAE_VRAM_ENVELOPE_MB`, `MULTISAE_ATTACH_DTYPE`, `CIRCUIT_INTENSITY_MIN/MAX`)

- [ ] 5.0 Attachment Admin-UI panel (covers FR-12.7; MSA-A4)
  - [ ] 5.1 `useAttachments` hook + `services/sae.ts` plural shape
  - [ ] 5.2 `AttachmentPanel` on the Circuits tab: chip per `(sae_id, layer)` with memory_mb, total readout, VRAM-warning badge
  - [ ] 5.3 Vitest: plural render, VRAM-warning badge, singular-shape back-compat in the existing SAE panel

- [ ] 6.0 Integration verification (covers FR-12.2 end-to-end, NFR-12.1)
  - [ ] 6.1 `test_multisae_serving.py`: attach two SAEs → serve circuit → per-layer `get_steering_values()` equals λ-clamped expectation on EACH layer
  - [ ] 6.2 Incomplete-set 422; detach clears per-layer; single-layer degenerate case; two-member-same-layer conflict (EC-12.2)
  - [ ] 6.3 Perf/VRAM harness (env-gated, GPU host): latency single-SAE vs two-SAE; peak VRAM two Gemma-2-2B SAEs fp16 vs envelope

- [ ] 7.0 Feature Acceptance (per instruct 007)
  - [ ] 7.1 Verify every FPRD §9 success criterion + §2 acceptance checkbox one-by-one
  - [ ] 7.2 Manual: multi-SAE attach + circuit serving; `docs/mcp-contract.md` cross-ref (circuit category); attachment-status shape-change note
  - [ ] 7.3 Full test suite green; update CLAUDE.md Document Inventory + Current Status

## Coverage Audit
- FR-12.1..12.7 each covered by ≥1 parent task (1.0–6.0 mapped above) ✓
- Every US acceptance criterion has an implementing sub-task and a test sub-task (2.3/3.5/4.3/5.3/6.x) ✓
- Every EC (12.1–12.5) has a test (3.5 bounds/incomplete, 4.1/4.3 VRAM warning, 6.2 conflict/degenerate/detach) ✓
- Every TDD/TID section maps to tasks (state→2.x, serving→3.x, API→4.x, UI→5.x, tests→throughout) ✓
- Open questions: none blocking (FPRD §13); the highest-uncertainty item (VRAM) is the run-before-build spike (task 1.0, DONE) ✓
- First task is the run-before-build spike; final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)
- Round 1 (multi-angle /code-review, 2 finder agents): pending.
- Round 2 (post-fix verification + fresh angles): pending.
- Round 3 (/review, 4 perspectives): pending.
Record: `0xcc/reviews/review_feature012_multisae_circuit_serving_2026-07-*.md` (to be created).

## Acceptance evidence (Task 7.0)
- Pending implementation. VRAM spike (task 1.0) DONE: two Gemma-2-2B SAEs = 128 MB fp16 (within <200 MB) /
  256 MB fp32 (exceeds) — attach-in-fp16 decision recorded (FTDD §1, FTID §5). Remaining FPRD §9 criteria
  (per-layer round-trip, `SAE_SET_INCOMPLETE`, hazard labeling, latency parity) verified at implementation.
