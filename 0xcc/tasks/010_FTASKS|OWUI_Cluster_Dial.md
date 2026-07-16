# Task List: OWUI Cluster Dial

## miLLM Feature 10

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Not started
**References:** `010_FPRD|OWUI_Cluster_Dial.md` · `010_FTDD|OWUI_Cluster_Dial.md` · `010_FTID|OWUI_Cluster_Dial.md`

## Relevant Files
- `millm/api/schemas/openai.py` — steering_intensity field + validator
- `millm/services/inference_service.py` — _resolve_intensity, _apply_request_steering, routing, call sites
- `millm/api/routes/openai/chat.py` — X-miLLM-Steering-Intensity echo
- `integrations/openwebui/millm_dial_filter.py` — OWUI Function
- `manual/docs/tutorials/open-webui.md` — install + usage
- `tests/unit/services/test_request_intensity.py`, `tests/unit/api/test_openai_schemas.py`,
  `tests/integration/api/test_chat_completions.py`

### Notes
- Depends on Feature 8 (clamp helper, intensity_range in cluster_meta, active cluster rows). Execute after 008.

### Category Checklist Results
- Data: N/A — no schema/storage change (request-scoped feature; FPRD §4)
- Backend/API: 1.x–2.x ✓
- Frontend/UI: N/A in Admin UI (FPRD §6); OWUI-side artifact covered by 3.x
- Business logic: 2.x (resolution, composition, override semantics) ✓
- Integration wiring: 2.4 routing, 2.5 header, 3.x plugin ✓
- Error handling & logging: 1.2 validation 400s, 2.6 no-op notices ✓
- Testing: paired throughout; 4.x integration/E2E ✓
- Performance & security: no-op fast path (2.6); no new auth surface — noted ✓
- Config/deploy: N/A — reuses Feature 8 config keys (TID §5)
- Documentation: 3.2 manual section ✓

## Tasks

- [ ] 1.0 Request schema extension (covers FR-10.1; DIAL-A1)
  - [ ] 1.1 Add `steering_intensity` field + validator to ChatCompletionRequest
  - [ ] 1.2 Schema unit tests: numeric bounds, symbolic set, 400 messages, extra="ignore" retained

- [ ] 2.0 Inference-path dial (covers FR-10.1, FR-10.2; DIAL-A2..A7)
  - [ ] 2.1 `_resolve_intensity` (symbolic→λ from intensity_range; config fallback; numeric passthrough)
  - [ ] 2.2 Generalize `_apply_request_profile` → `_apply_request_steering` (base selection: named > active > live; request-λ overrides stored λ; λ=0 disable path; clamp via shared helper)
  - [ ] 2.3 Swap both call sites (non-streaming + streaming) keeping restore-in-finally placement
  - [ ] 2.4 Extend serial-routing condition (`has_profile or steering_intensity is not None`)
  - [ ] 2.5 `X-miLLM-Steering-Intensity` echo on both paths (stash-at-apply for streaming)
  - [ ] 2.6 No-op semantics: field absent vs λ=1.0 vs no SAE/no profile (logged notice, never error)
  - [ ] 2.7 Unit tests: resolution matrix, override semantics, λ=0, composition with `profile`, clamp parity, saved/restored shape

- [ ] 3.0 OWUI Function + docs (covers FR-10.3, FR-10.4; DIAL-F1..F4)
  - [ ] 3.1 `integrations/openwebui/millm_dial_filter.py` (Valves/UserValves, inlet-only, version-note header)
  - [ ] 3.2 Manual: Function install + dial usage + global-vs-per-request distinction in open-webui.md
  - [ ] 3.3 Lint/self-test the plugin file standalone (no miLLM imports)

- [ ] 4.0 Integration verification (covers FR-10.2, FR-10.4)
  - [ ] 4.1 Integration tests: streaming + non-streaming with field; serial routing asserted; global state identical before/after; no-active-cluster no-op
  - [ ] 4.2 Concurrency test: two interleaved requests with different λ produce independent applies (serialized) and clean restores
  - [ ] 4.3 E2E script: identical prompt at off/min/max on a validated cluster (post-deploy); OWUI manual walkthrough

- [ ] 5.0 Feature Acceptance (per instruct 008)
  - [ ] 5.1 Verify FPRD §9 criteria 1–4 + all US/EC acceptance boxes one-by-one
  - [ ] 5.2 Full suite green; update CLAUDE.md Document Inventory + Current Status

## Coverage Audit
- FR-10.1..10.4 ↔ tasks 1.0/2.0 (A1..A7), 3.0 (F1..F4), 4.0 (verification) ✓
- US-10.1→3.1/4.3; US-10.2→1.1/2.5/4.1; US-10.3→2.4/4.2 — each with implementing + testing sub-tasks ✓
- EC-10.1→2.6/4.1; EC-10.2→2.1/2.7; EC-10.3→1.2; EC-10.4→1.2 (extra=ignore) + plugin header note ✓
- TDD/TID sections mapped (schema→1.x, service→2.x, plugin→3.x, tests→2.7/4.x); Data/UI/Config N/A justified ✓
- Open questions: none — no spike tasks ✓
- Final task is Feature Acceptance ✓
