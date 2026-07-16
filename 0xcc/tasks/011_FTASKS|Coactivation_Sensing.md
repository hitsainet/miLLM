# Task List: Co-Activation Sensing

## miLLM Feature 11

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Not started
**References:** `011_FPRD|Coactivation_Sensing.md` · `011_FTDD|Coactivation_Sensing.md` · `011_FTID|Coactivation_Sensing.md`

## Relevant Files
- `millm/ml/sae_wrapper.py`, `millm/ml/sae_hooker.py` — detection core
- `millm/services/sensing_service.py`, `inference_service.py`, `profile_service.py`, `sae_service.py`
- `millm/db/models/sensing_event.py`, `db/repositories/sensing_repository.py`,
  `db/migrations/versions/008_create_sensing_events_table.py`
- `millm/api/schemas/sensing.py`, `api/routes/management/sensing.py`, `api/dependencies.py`, `routes/__init__.py`
- `millm/sockets/progress.py` — emit_sensing_event
- `millm/core/config.py` — SENSING_* keys
- `admin-ui/src/components/clusters/sensing/*`, `services/sensing.ts`, `hooks/useSensing.ts`
- `tests/unit/{ml,services,db}/test_sensing*.py`, `tests/integration/test_sensing_workflow.py`

### Notes
- Depends on Feature 8 (cluster rows, sensing_enabled column, members w/ max_activation). Execute after 008.
- Migration numbering: 008 (Feature 8 owns 007).

### Category Checklist Results
- Data: 1.x (table + model + repo + retention) ✓
- Backend/API: 4.x ✓
- Frontend/UI: 5.x ✓
- Business logic: 2.x (predicate/debounce/thresholds), 3.x (lifecycle/context) ✓
- Integration wiring: 3.x (inference + arm/disarm hooks), 4.2 (DI/router), 5.1 (WS client) ✓
- Error handling & logging: 2.5 (cap/truncated), 3.6 (overhead warn), buffer-hygiene tests ✓
- Testing: paired throughout + 6.x integration ✓
- Performance & security: SEN-S1..S3 tasks (2.4, 3.5, 6.4); context-privacy documented (7.2) ✓
- Config/deploy: 3.7 config keys; migration auto-runs — no deploy change ✓
- Documentation: 7.2 manual section ✓

## Tasks

- [x] 1.0 Persistence layer (covers FR-11.4; SEN-P1, SEN-P2)
  - [x] 1.1 Migration `008_create_sensing_events_table.py` (up+down, indexes, FK CASCADE)
  - [x] 1.2 `db/models/sensing_event.py` + `sensing_repository.py` (create_many/list/clear/prune)
  - [x] 1.3 Repo unit tests incl. retention cap + age prune + CASCADE with profile delete

- [x] 2.0 Detection core in LoadedSAE (covers FR-11.1; SEN-D1..D5)
  - [x] 2.1 SensingConfig/SensedHit dataclasses; arm/disarm (W_enc_m cache, dtype/device parity with encode)
  - [x] 2.2 `_sense` per-pass evaluation (thresholds, min_k, hot positions) + suppressed() early-return
  - [x] 2.3 Debounce to spans incl. cross-pass tail-merge; offset/phase accounting (prefill/decode/speculative shapes)
  - [x] 2.4 Per-request cap + truncated flag + `_sensing_done` fast path
  - [x] 2.5 Unit tests: predicate matrix, ε-fallback→floor_only mode, spans, cap, offsets, suppressed, arm idempotence, buffer hygiene (begin resets; no-begin ⇒ empty collect)
  - [x] 2.6 Hook branch in `sae_hooker.hook_fn` (sibling of monitoring, before apply_steering)

- [x] 3.0 Service + lifecycle + flush (covers FR-11.2, FR-11.3, FR-11.5; SEN-R1..R4, SEN-S1..S3)
  - [x] 3.1 `sensing_service.py`: _build_config from cluster_meta (overrides), arm_for_profile/disarm/should_sense/status
  - [x] 3.2 Inference wiring: begin after apply, collect in finally, async `_notify_sensing` beside `_notify_monitoring` (both paths)
  - [x] 3.3 Context capture: outputs[0] (non-stream), IdCaptureStoppingCriteria (stream), prompt ids (prefill); ±K decode off hot path; K=0 path
  - [x] 3.4 `ambient_fired_count` best-effort fill (full-width monitoring only, else NULL); summary builder (≤300 chars)
  - [x] 3.5 Routing: SENSING_FORCE_SERIAL condition in `_use_cbm_for_request`; non-forced CBM ⇒ unsensed
  - [x] 3.6 `sensing_overhead_ms` accumulator + warn threshold; exposed in status
  - [x] 3.7 Config keys (SENSING_*); arm/disarm hooks in profile activate/deactivate + SAE detach
  - [x] 3.8 Unit tests: config build, context slicing edges (pos 0, end, early stop), summary, ambient rules, lifecycle

- [x] 4.0 API + WS surface (covers FR-11.4; SEN-P3, SEN-P4)
  - [x] 4.1 `api/schemas/sensing.py` + `routes/management/sensing.py` (status/events/enable/disable/clear)
  - [x] 4.2 DI + router registration; route tests (envelope, filters, enable toggles column + live arm)
  - [x] 4.3 `emit_sensing_event` in sockets/progress.py (payload w/o context_text) + emission test

- [x] 5.0 Clusters-page sensing UI (covers SEN-P5)
  - [x] 5.1 `services/sensing.ts` + `useSensing.ts` + WS subscription (live prepend)
  - [x] 5.2 SensingPanel (status strip, event list) + SensingEventDetail (member table, highlighted context)
  - [x] 5.3 Wire SensingToggle (Feature 8 stub) to enable/disable
  - [x] 5.4 Vitest: panel render, toggle flow, live event prepend

- [x] 6.0 Integration verification (covers FR-11.1..11.5 end-to-end)
  - [x] 6.1 `test_sensing_workflow.py`: arm→generate on a known co-firing fixture→events with correct spans/quorum/context
  - [ ] 6.2 Streaming early-stop context correctness; prefill event context
  - [x] 6.3 Lifecycle: enable/disable live, SAE detach disarms, profile delete cascades events
  - [x] 6.4 Safety: serial forcing asserted; CBM-unsensed; overhead accumulator populated; un-armed zero-delta smoke
  - [x] 6.5 WS emission observed end-to-end

- [ ] 7.0 Feature Acceptance (per instruct 008)
  - [ ] 7.1 Verify FPRD §9 criteria 1–5 + all US/EC boxes one-by-one
  - [x] 7.2 Manual: sensing semantics section (fired/attribution/alone-within caveat/retention+privacy)
  - [ ] 7.3 Full suite green; update CLAUDE.md Document Inventory + Current Status

## Coverage Audit
- FR-11.1→2.0/6.1; FR-11.2→3.4/6.1; FR-11.3→3.3/6.2; FR-11.4→1.0/4.0/6.3/6.5; FR-11.5→3.5/3.6/6.4 ✓
- US-11.1→3.7/5.3/6.3; US-11.2→4.x/5.x/6.1; US-11.3→1.3/3.6/6.4 — implementing + testing sub-tasks each ✓
- EC-11.1→2.3/2.5; EC-11.2→2.4/2.5; EC-11.3→3.5/6.4; EC-11.4→3.1/2.5(mode); EC-11.5→3.3; EC-11.6→2.2/2.5 ✓
- TDD/TID sections all mapped (detection→2.x, lifecycle/flush→3.x, persistence→1.x, API/WS→4.x, UI→5.x) ✓
- Open questions: none — no spike tasks ✓
- Final task is Feature Acceptance ✓
