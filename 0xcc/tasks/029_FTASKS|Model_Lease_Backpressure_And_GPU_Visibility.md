# Feature 29: Model Lease, Backpressure and GPU Visibility — Task List

**Status:** Planned (2026-10-06). Clarifying rounds and the "Go" pause were **waived** by the
coordinator for this increment; parent tasks and sub-tasks were generated in one pass.
**Inputs:** `029_FPRD` v1.1 · `029_FTDD` v1.0 · `029_FTID` v1.0 · BRD-04 §5.10–§5.12 (R-04.38 –
R-04.45) · PADR v1.5 §10 · Feature-PRD decisions P-05, X-01, X-08, default TTL · register T-66,
T-84 – T-90
**Build order:** first in BRD-04 with Feature 25 (BRD-04 RSK-09). Feature 26 builds on the lease API,
`RequestQueue.holding_count` and the backlog provider named here (026 FTDD §2, §7).
**Co-release:** task 10.4 (MCP contract) unblocks miStudio `034` FR-18 lease tools and FR-19.

## Relevant Files
- `millm/services/model_lease.py`: registry, records, startup clear · `tests/unit/services/test_model_lease.py`
- `millm/services/model_service.py`: lease API, `_refuse_if_leased`, `lease_id` on load/unload/wait ·
  `tests/unit/services/test_lease_enforcement.py`
- `millm/core/errors.py`: five lease errors · `millm/core/config.py`, `.env.example`: lease, queue and `RETRY_AFTER_*` settings
- `millm/core/backpressure.py`: `retry_after_for`, estimate, backlog provider · `tests/unit/api/test_retry_after.py`
- `millm/api/retry_after.py`: `RetryAfterMiddleware` · `millm/main.py`: middleware registration, `clear_leases_on_startup()` call
- `millm/api/routes/openai/errors.py`: map rows, `retry_after` on builders, `model_not_resident_error` · `tests/unit/api/test_openai_errors.py`
- `millm/api/exception_handlers.py`: `Retry-After` on every 503 · `tests/unit/api/test_exception_handlers.py`
- `millm/api/routes/openai/load_policy.py` + `chat.py`, `completions.py`, `embeddings.py` · `tests/unit/api/test_load_policy.py`
- `millm/api/routes/management/models.py`: lease routes, header pass-through, `_with_lease` · `tests/unit/api/test_lease_routes.py`
- `millm/api/schemas/lease.py`, `millm/api/schemas/model.py` (`ModelResponse.lease`)
- `millm/api/routes/system/health.py`: `InferenceState`, `lease`, `/gpus`, readiness header ·
  `tests/unit/api/test_health_queue_state.py`, `tests/unit/api/test_gpu_memory_route.py`
- `millm/services/request_queue.py`: holding counter, durations · `tests/unit/services/test_request_queue_holding.py`
- `millm/services/inference_service.py`: `_stream_error_event` `retry_after`
- `millm/services/gpu_memory.py`, `millm/ml/nvidia_smi.py` (`query_compute_apps`), `millm/ml/model_loader.py` (touched set) ·
  `tests/unit/services/test_gpu_memory.py`
- `tests/unit/test_startup_reset.py`: lease reconciliation tests
- Admin UI: `admin-ui/src/types/api.ts`, `components/models/LeaseBadge.tsx` (+ `__tests__/LeaseBadge.test.tsx`),
  `components/models/index.ts`, `pages/ModelsPage.tsx`, `components/models/ModelDetailsModal.tsx`,
  `components/models/LoadedModelCard.tsx`, `hooks/useModels.ts`
- Docs: `docs/mcp-contract.md`, `manual/docs/api/models.md`, `manual/docs/api/management-api.md`,
  `manual/docs/reference/error-codes.md`, `manual/docs/reference/configuration.md`,
  `manual/docs/features/model-management.md`
- Review record (new): `0xcc/reviews/review_feature029_model_lease_<date>.md`

### Notes
- Tests: `pytest tests/unit` (backend), `cd admin-ui && npm test`; `ruff`, `mypy millm/`,
  `npm run lint`, `npm run typecheck`.
- **Reachability is a shipping gate** (miLLM `CLAUDE.md`; PPRD FR-20.3): every wiring line needs a
  test that fails when it is removed, asserting payload and call count. Every lease-enforcement path
  has a mutation control (FTID §8.4, M1–M18).
- **Re-verify every line number from the FTID before editing** (task 0.1). Features 025–028 and 030
  edit the same route files.
- A test that reads `docs/mcp-contract.md` or any `0xcc/` file must skip loudly when the file is
  absent: the public mirror strips `0xcc/` and, in miLLM, everything under `docs/` except `schemas/`.

### Category Checklist Results
- **Data layer:** N/A — no table and no migration; leases are process memory by decision X-01, and
  `models.locked` is untouched (C8). The in-memory registry is task 2.x.
- **Backend/API:** 5.x (lease routes, headers, refuse policy), 6.x (`Retry-After`), 7.x (health),
  8.x (GPU route)
- **Frontend/UI:** 9.x
- **Business logic:** 2.x (registry, service rules), 3.x (enforcement), 7.2 (estimate)
- **Integration wiring:** 3.x (load paths), 4.x (`lifespan`), 5.3–5.5 (routes), 6.3–6.5 (builders,
  middleware, stream), 7.4 (Feature 26 provider), 10.4 (miStudio contract)
- **Error handling and logging:** 1.2–1.3, 2.6, 5.7, 6.x, 12-row table in FTID §12 → 11.1
- **Testing:** throughout; integration 5.8, 11.2; hardware 11.3–11.5; mutation controls 11.6
- **Performance and security:** 2.4 (digest only, no ID in logs), 5.6 (no-disclosure), 8.3 (no
  CUDA context), 7.5 (no `nvidia-smi` on the health path)
- **Configuration/deployment:** 1.1 (settings, `.env.example`); no Kubernetes manifest change (FTDD §11)
- **Documentation:** 10.x

## Tasks

- [ ] 0.0 Spikes and preconditions (covers FR-29.8, T-89; all FRs for 0.1)
  - [x] 0.1 Re-verify every `path:line` cited in the FTDD and FTID at the current HEAD. Record moved
        lines in the review record before editing anything.
  - [?] 0.2 **T-89 spike, on the node.** _needs hardware — operator session._ (a) Inside the miLLM pod, run
        `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits`
        with a model loaded; record whether miLLM's process appears and with which PID. (b) Measure
        the CUDA context size: after an unload, record `torch.cuda.memory_reserved(i)` and the
        process's `used_memory` from the node's `nvidia-smi`; the difference is `cuda_context_mb`.
        Record both in the review; acceptance 16 (11.4) uses them.
  - [x] 0.3 Confirm with the 026 FTDD (§2, §7) that the names in FTID §10 are the ones its runner
        calls: `acquire_lease`, `renew_lease`, `release_lease`, `resolve_lease`, `holding_count`,
        `register_backlog_provider`. Record any drift as a task here, not in 026's files.

- [x] 1.0 Errors, settings and the value policy (covers FR-29.1.2, FR-29.2.3, FR-29.6.3)
  - [x] 1.1 Add the 14 settings (FTID §9) with the startup validator
        `LEASE_DEFAULT_TTL_SECONDS <= LEASE_MAX_TTL_SECONDS`; document each in `.env.example`.
  - [x] 1.2 Add `ModelLeasedError`, `ModelNotResidentError`, `LeaseNotFoundError`,
        `LeaseExpiredError`, `InvalidLeaseRequestError` (FTID §5). `ModelLeasedError` does **not**
        subclass `ModelLockedError`.
  - [x] 1.3 Add `MODEL_LEASED` and `MODEL_NOT_RESIDENT` to `ERROR_STATUS_MAP`; add
        `model_not_resident_error`.
  - [x] 1.4 Write `backpressure.retry_after_for` with the T-88 table (FTDD §5.3), including the
        unload-vs-load distinction on `MODEL_BUSY` and the breaker remainder for `HUB_UNAVAILABLE`.
  - [x] 1.5 Tests: one value per code; integer ≥ 1; `QUEUE_FULL` clamp at 60 and default without an
        estimate; breaker remainder with an injected clock; a `ModelBusyError` with and without
        `unloading_model_id` gives 5 and 15; `ModelLeasedError` maps to `model_leased`, never
        `model_locked`.

- [x] 2.0 Lease registry and service API (covers FR-29.1, FR-29.1.1 – FR-29.1.11)
  - [x] 2.1 Implement `LeaseRegistry` (FTID §3.1): digest storage, injectable clocks, a lock, ended
        memory of 64, `grant`, `renew`, `release`, `resolve`, `current`, `matches`, `end_for_model`,
        `last_ended`, `clear`.
  - [x] 2.2 Implement `ModelService.acquire_lease`, `renew_lease`, `release_lease`, `get_lease`,
        `resolve_lease` with the precondition order of FTID §3.2 (P-05: no approval step).
  - [x] 2.3 Default TTL 7200 when omitted; refuse 0, negative, above 7200 and non-integers with
        `400 INVALID_LEASE_REQUEST` naming field and limit (FR-29.1.2). Strip and bound `holder`
        (128) and `reason` (512).
  - [x] 2.4 Lease ID = `token_urlsafe(24)`, returned only in the grant; store its SHA-256; log
        `lease_ref` only (FR-29.1.6, FR-29.1.11).
  - [x] 2.5 Grant refuses: non-resident model (`409 MODEL_NOT_RESIDENT`, T-85), load or unload in
        progress (`503 MODEL_BUSY`), a live lease on the model (`409 MODEL_LEASED`, X-08).
  - [x] 2.6 Lazy expiry with `<=` and once-only `lease_expired` log (FR-29.1.10); renew sets expiry
        from now (FR-29.1.7); unknown ID `404` with the restart sentence; ended ID `409` with
        `end_reason`.
  - [x] 2.7 Tests (registry): grant/conflict/renew/release/expiry by clock; ended reasons;
        `resolve` returns live only; two holders with the same string are two callers.
  - [x] 2.8 Tests (service): each grant refusal, with fixtures where the requested model differs
        from the resident one; grant during a load (`_loading_model_id` set) refused (control M18);
        capture logs and assert no event contains the lease ID.

- [x] 3.0 Enforcement on every load path (covers FR-29.2, FR-29.2.1 – FR-29.2.6, FR-29.3,
      FR-29.3.1 – FR-29.3.4)
  - [x] 3.1 Implement `_refuse_if_leased` (FTDD §7.2) reading the resident model's lease;
        constant-time match; `lease_header_unmatched` warning for a wrong ID that needed no lift.
  - [x] 3.2 Call it in `load_model` immediately before the slot check (FTID §3.3), no `await`
        between; add keyword `lease_id`.
  - [x] 3.3 Pass `lease_id` to the internal unload a load performs.
  - [x] 3.4 Call it in `unload_model` after the not-loaded check and before the unloading check;
        add `lease_id`.
  - [x] 3.5 Call it in `load_model_and_wait` after the early return and before the `locked` check;
        pass `lease_id` to `load_model` (FPRD D9: lease first).
  - [x] 3.6 End the lease on unload success, beside the auto-unlock (FR-29.1.9, FR-29.3.4).
  - [x] 3.7 Tests through the real service: foreign lease refuses management load, management unload
        and auto-load with holder, reason and `expires_at` in the payload, and `_load_worker` call
        count 0; the right header proceeds; the holder's swap succeeds and ends the lease; lease and
        `locked` together → `MODEL_LEASED`; a refused load claims no slot and moves no row
        (FR-29.2.6); a request naming the leased resident model proceeds without a header
        (FR-29.2.4).
  - [x] 3.8 AST call tests: `_refuse_if_leased` in each of the three method bodies.

- [x] 4.0 Restart reconciliation and the self-healing read (covers FR-29.1.9; X-01; T-66)
  - [x] 4.1 Implement `clear_leases_on_startup()` in `model_lease.py`: clears the registry with
        reason `restart`, logs `leases_cleared_on_startup` with the count, never raises.
  - [x] 4.2 Call it from `lifespan` beside `disarm_probes_on_startup` (`main.py:379`).
  - [x] 4.3 Implement the residency branch of `current()`: a lease whose model is no longer the
        loader's resident model ends with `model_unloaded`.
  - [x] 4.4 **Startup reconciliation test (required):** in `tests/unit/test_startup_reset.py`, fill
        the registry, run `clear_leases_on_startup()` for real, assert empty and `end_reason ==
        "restart"`; a second test walks `lifespan`'s AST for the call (memory
        `startup-reset-lists-hide-omissions`).
  - [x] 4.5 Test the self-healing read: empty the loader without calling `unload_model` (the forced
        path), assert the lease reads as none.

- [ ] 5.0 Lease routes, headers and the refuse-load policy (covers FR-29.1.5 – FR-29.1.8, FR-29.3,
      FR-29.4, FR-29.4.1 – FR-29.4.6, FR-29.5.3)
  - [ ] 5.1 Add `millm/api/schemas/lease.py` (`extra="forbid"`; no `lease_id` on any response but the
        grant) and the single `lease_summary` serialiser.
  - [ ] 5.2 Add the four lease routes on the management models router (FTDD §5.1); lease ID read from
        `X-miLLM-Lease` on renew and release; a lease ID for another model answers `404`.
  - [ ] 5.3 Pass `X-miLLM-Lease` from `POST /api/models/{id}/load` and `/unload` to the service.
  - [ ] 5.4 Implement `load_policy.parse_load_policy` and `apply_load_policy` (FTID §3.4): `auto`
        default; invalid value `400`; not resident → `409 model_not_resident` with the resident
        model and its lease summary; model loading now → `503 model_loading` with `Retry-After`.
  - [ ] 5.5 Wire both headers into the three `/v1` routes; call `apply_load_policy` after the
        existing pre-load refusals and before the auto-load; pass `lease_id` to
        `load_model_and_wait`.
  - [ ] 5.6 Add `ModelResponse.lease` and `_with_lease` on the list and single-model routes.
  - [ ] 5.7 Tests (routes): 201 grant carries `lease_id`; `GET`, renew, release, `ModelResponse` and
        health never carry it (control M12); renew sets new expiry; release ends; 404/409 bodies;
        `400` for each bad TTL and text; reachability in `app.openapi()["paths"]`.
  - [ ] 5.8 Tests (policy, integration): on each of the three routes, `refuse` + non-resident → `409`
        and load call count 0; `refuse` + loading → `503` with `Retry-After`; absent header
        auto-loads as today; invalid header `400`; the GGUF and embedding-only refusals still come
        first; AST test that each route calls `apply_load_policy` before `load_model_and_wait`.
        Management load ignores the policy header (FR-29.4.6).

- [ ] 6.0 `Retry-After` on every 503 (covers FR-29.6, FR-29.6.1 – FR-29.6.6)
  - [ ] 6.1 `create_openai_error` gains `retry_after`; `model_not_loaded_error`, `model_busy_error`
        and `load_refused_error` pass it; `MODEL_NOT_LOADED`'s message adds the "only after a model
        is loaded" sentence.
  - [ ] 6.2 `millm_error_handler` sets the header for every 503 on both API families.
  - [ ] 6.3 Readiness probe header (`health.py:261-262`).
  - [ ] 6.4 `_stream_error_event` adds `retry_after` when the code maps to 503 on `/v1`.
  - [ ] 6.5 Add `RetryAfterMiddleware` and register it in `create_app`.
  - [ ] 6.6 Tests: parametrised over `QUEUE_FULL` (eleven concurrent requests, BRD-04 acceptance 15),
        `MODEL_BUSY` (load and unload), `MODEL_LOADING` (via 5.4), `MODEL_NOT_LOADED`,
        `INSUFFICIENT_MEMORY`, `HUB_UNAVAILABLE`, readiness — each with its own value and **no**
        `retry_after_defaulted` log; a synthetic bare-503 route gets 10 and the warning; envelope
        and codes unchanged against a snapshot; in-stream error carries `retry_after`; middleware in
        `app.user_middleware` (controls M14, M15).

- [ ] 7.0 Queue state and the health contract (covers FR-29.5.1, FR-29.7, FR-29.7.1 – FR-29.7.6)
  - [ ] 7.1 `RequestQueue`: `_holding`, the duration window, `holding_count`, `median_hold_seconds`
        (FTID §3.6). `pending_count` unchanged.
  - [ ] 7.2 `backpressure.estimate_wait_seconds` (FTDD §7.3); `null` below 3 samples or under CBM.
  - [ ] 7.3 Replace the untyped `inference` dict with `InferenceState`; always present, `error` on
        failure; `in_flight`/`queue_waiting`/estimate `null` while `cbm_running` (T-90).
  - [ ] 7.4 `register_backlog_provider` / `backlog_rows` (`null` when none registered; logs and
        `null` when the provider raises). Feature 26 registers its provider (026 FR-26.4.7).
  - [ ] 7.5 Add `lease: LeaseStatusResponse | None` to `DetailedHealthResponse`; no `nvidia-smi` on
        this path.
  - [ ] 7.6 Tests: one holding and two waiting → `in_flight` 1, `queue_waiting` 2; the idle cache
        release counts as holding; CBM running → `null`s; backlog `null` unregistered (control M16)
        and the provider's value when registered; estimate formula with fixed durations; field-set
        pin for `inference` and `lease`; `test_every_request_queue_slot_is_taken_through_admission`
        stays green.

- [ ] 8.0 GPU memory endpoint (covers FR-29.8, FR-29.8.1 – FR-29.8.8)
  - [ ] 8.1 `nvidia_smi.query_compute_apps()` with parser and 5-second timeout.
  - [ ] 8.2 Touched-card set in `model_loader.py`, recorded in `LoadedModelState.set()` for
        transformers models.
  - [ ] 8.3 `gpu_memory.read_gpu_memory()` (FTID §7): torch read only on touched cards
        (`torch_measured`), `null` otherwise; `engine_memory` for a resident GGUF model; processes by
        UUID or `null` with a reason; `cards: []` + `reason` without `nvidia-smi`.
  - [ ] 8.4 `GET /api/health/gpus` via `asyncio.to_thread`; no request-queue slot.
  - [ ] 8.5 Tests: one- and two-card fixtures; a card torch cannot see listed with `torch_index:
        null`; untouched card never calls `memory_reserved` (control M17); GGUF label; absent
        `nvidia-smi`; route reachable; read runs in a thread (patched `to_thread` call count 1).

- [ ] 9.0 Admin UI lease visibility (covers FR-29.5.2 – FR-29.5.5)
  - [ ] 9.1 `LeaseSummary` on `ModelInfo` in `types/api.ts`; no `lease_id` in any type.
  - [ ] 9.2 `LeaseBadge` (FTID §6): amber, text label and `aria-label`, 1-second local countdown,
        hides past expiry.
  - [ ] 9.3 Place it on `ModelsPage` beside the lock icon, in `ModelDetailsModal` and in
        `LoadedModelCard`.
  - [ ] 9.4 `useModels` polls every 10 seconds while any model is leased.
  - [ ] 9.5 Vitest: holder, reason and remaining time render; badge disappears after expiry with fake
        timers; lock tooltip still reads "Locked for steering"; no lease action rendered (T-86);
        `refetchInterval` returns 10,000 with a lease.

- [ ] 10.0 Documentation and the cross-repo contract (covers FR-29.7.6 and the API surface of
      FR-29.1 – FR-29.8)
  - [ ] 10.1 `manual/docs/api/models.md`: lease routes, headers, codes, examples.
  - [ ] 10.2 `manual/docs/api/management-api.md`: `InferenceState` fields with the estimate formula
        and the `queue_pending` meaning; `lease`; `/api/health/gpus`.
  - [ ] 10.3 `manual/docs/reference/error-codes.md` (`MODEL_LEASED`, `MODEL_NOT_RESIDENT`,
        `LEASE_NOT_FOUND`, `LEASE_EXPIRED`, `INVALID_LEASE_REQUEST`, `Retry-After`) and
        `configuration.md` (14 settings); `features/model-management.md` (badge, restart ends every
        lease).
  - [ ] 10.4 `docs/mcp-contract.md`: next additive version; lease routes, both headers, codes, health
        fields, GPU route; the `GET /api/health/detailed` row (`docs/mcp-contract.md:92`) gains the
        new fields. Notify miStudio 034 FR-18/FR-19 owners.
  - [ ] 10.5 Tests: `test_manual_pages_are_reachable` and `test_mcp_contract_consistency` green; any
        new test reading `docs/` skips loudly when the file is absent (mirror).

- [ ] 11.0 Feature Acceptance
  - [ ] 11.1 Walk each FPRD edge case (§2 table) and success criterion (§11) against its test; record
        the test name per row in the review record.
  - [ ] 11.2 Full suites: `pytest tests/unit`, the same with `0xcc/` hidden (the mirror's view),
        `admin-ui` Vitest, `tsc`, lint, `mypy millm/`.
  - [ ] 11.3 **Hardware — lease (BRD-04 acceptance 14, success criteria 1–3).** On the node, with
        JEV-9B-decision or LFM2.5-1.2B resident: take a lease as `midataworks` with TTL 120; a chat
        request naming another model → `409 model_leased` naming holder and expiry;
        `POST /api/models/{other}/load` without the header → `409 MODEL_LEASED`; with the header →
        proceeds and the lease ends; re-lease, wait past 120 s without renewing → the same load
        succeeds. A `refuse` request for a non-resident model → `409 model_not_resident` and no load
        in the logs.
  - [ ] 11.4 **Hardware — GPU visibility (BRD-04 acceptance 16, success criterion 7).** Load and
        unload a model; read `/api/health/gpus`; on each card miLLM touched, compare
        `millm_reserved_mb + cuda_context_mb` (from 0.2) with the node's per-process
        `used_memory`, within 256 MiB (T-89). Confirm no new CUDA context appears on the card miLLM
        did not touch (its process list unchanged).
  - [ ] 11.5 **Hardware — backpressure and restart (success criteria 4–6, X-01).** Eleven concurrent
        requests → a `503 queue_full` with `Retry-After`; `/api/health/detailed` shows `in_flight`,
        `queue_waiting` and an estimate during the burst. Take a lease, restart the pod, confirm the
        lease is gone (`GET` none, renew `404` with the restart sentence) and that the Admin UI
        badge is gone.
  - [ ] 11.6 **Mutation controls M1–M18** (FTID §8.4): run each, show it red, restore, verify the
        restore with `git diff` and a re-grep of the mutated line. A survivor gets a test, then the
        mutation is re-run as a negative control. Record all in the review record.
  - [ ] 11.7 Record the tracked debt (management load/unload ignore `locked`; single-process
        registry) in the review record; update miLLM's Document Inventory and Current Status. Do not
        mark the feature ✅ until 11.3–11.5 pass.

## Coverage Audit
- **FRs → tasks:** 29.1 → 2.x, 4.x, 5.1–5.2, 5.7 · 29.2 → 3.1–3.5, 3.7–3.8 · 29.3 → 3.1, 3.3, 3.6,
  3.7, 5.3, 5.5 · 29.4 → 5.4, 5.5, 5.8 · 29.5 → 5.6, 7.5, 9.x · 29.6 → 1.4, 6.x · 29.7 → 7.x, 10.2,
  10.4 · 29.8 → 0.2, 8.x, 11.4. All eight covered; every refined item FR-29.x.y maps to the parent
  task that cites its FR.
- **Edge cases (FPRD §2), implement / test:** live lease conflict 2.5 / 2.8 · non-resident 2.5 / 2.8 ·
  grant during load or unload 2.5 / 2.8 · bad TTL 2.3 / 5.7 · wrong or expired ID 2.6 / 5.7 · leased
  model used without header 3.1 / 3.7 · refuse + loading 5.4 / 5.8 · 503 in a committed stream 6.4 /
  6.6 · restart 4.1–4.2 / 4.4, 11.5 · `nvidia-smi` absent 8.3 / 8.5.
- **Success criteria (FPRD §11), implement / test:** 1 → 3.x / 3.7, 11.3 · 2 → 3.3, 3.6 / 3.7 ·
  3 → 5.4 / 5.8, 11.3 · 4 → 6.x / 6.6, 11.5 · 5 → 6.x / 6.6 · 6 → 7.x / 7.6, 11.5 · 7 → 8.x / 8.5,
  11.4 · 8 → 5.6, 7.5, 9.x / 5.7, 9.5.
- **TDD/TID sections:** Data Design → N/A (checklist) and 2.1 · API Design → 5.x, 6.x, 7.x, 8.x ·
  Component Architecture → 2.x–8.x · State Management → 2.6, 4.x, 7.1, 9.4 · Security → 2.4, 5.7 ·
  Performance → 7.5, 8.4 · Testing → every parent, 11.6 · Deployment & DevOps → 1.1, 10.4, 11.5 ·
  TID Configuration → 1.1 · Integration → 0.3, 3.x, 7.4, 10.4 · Error handling → 1.2–1.3, 2.6, 6.x.
- **Open questions:** none for the operator (FPRD v1.1 §14). The one technical unknown, in-pod
  `--query-compute-apps` visibility and the CUDA context size (T-89), is spike 0.2, placed before
  11.4 which depends on it.
- **The final parent task is Feature Acceptance.** ✔
