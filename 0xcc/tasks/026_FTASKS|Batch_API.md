# Feature 26: Batch API — Task List

**Status:** Planned (2026-10-06). Generated from the FPRD, FTDD and FTID in one pass. **The "Go" pause between parent tasks and sub-tasks was waived** by the coordinator's instruction for this increment; parent tasks and sub-tasks were produced together.
**Inputs:** `026_FPRD|Batch_API.md` v1.1 · `026_FTDD|Batch_API.md` · `026_FTID|Batch_API.md` · BRD-04 §5.5 · PADR v1.5 §10
**Build order:** after Features 25 and 29 (PPRD; BRD-04 RSK-09) and after Feature 27 (T-71). Feature 28 before phase 6; Feature 30 before embedding packing (6.6).
**Co-release:** miStudio's batch tools (`034_FTASKS` phase 6) start only after task 8.2 publishes the contract section.

## Relevant Files

- `millm/services/request_queue.py` — `acquire_background`, counters, `occupied_count` · `tests/unit/services/test_request_queue_background.py`
- `millm/services/inference_service.py` — `_admit(background=)`, owner re-entry, `_score_prompts`, packed gather, batch-row CBM clause · `tests/unit/services/test_admit_reentry.py`, `tests/unit/services/batch/test_packed_scoring.py`
- `millm/services/model_service.py` — unload drain on `occupied_count` (`:1258`)
- `millm/services/batch/{__init__,state,files,validator,executors,runner,retention,reconcile}.py` · `tests/unit/services/batch/test_*.py`
- `millm/db/models/batch.py`, `millm/db/models/__init__.py`, `millm/db/repositories/batch_repository.py`, `millm/db/migrations/versions/0NN_add_batch_api.py` · `tests/unit/db/test_batch_models.py`
- `millm/db/models/probe.py`, `millm/db/repositories/probe_repository.py`, `millm/services/probe_event_service.py` — batch origin · `tests/unit/services/test_probe_events_batch_origin.py`
- `millm/api/schemas/batch.py`, `millm/api/routes/openai/files.py`, `millm/api/routes/openai/batches.py`, `millm/api/routes/openai/__init__.py` · `tests/unit/api/test_batch_routes.py`
- `millm/api/provenance.py`; `millm/api/routes/openai/{chat,completions,embeddings}.py` (validate extraction, headers) · `tests/unit/api/test_provenance.py`
- `millm/sockets/progress.py` — `emit_batch_progress` · `tests/unit/sockets/test_batch_progress.py`
- `millm/main.py` — lifespan wiring · `tests/unit/test_batch_lifespan_wiring.py`
- `millm/core/config.py`, `.env.example`, `k8s/base/backend.yaml`, `docker-compose.yml`
- `tests/unit/services/test_unload_admission.py` (guard extended), `tests/unit/test_mcp_tool_paths_are_real.py`
- `tests/integration/test_batch_workflow.py`, `tests/performance/test_batch_throughput.py` (hardware)
- `docs/mcp-contract.md`, `manual/docs/api/batches.md` (+ sidebar entry)

### Notes

- Tests: `pytest tests/unit` (backend), `ruff`, `mypy millm/`. No frontend change in this feature.
- **Reachability is a shipping gate.** Every route must be present in the live app's `app.openapi()["paths"]` and called through `TestClient` with its payload asserted. Every wiring line (router include, lifespan call, emit, lease acquire/release, `_admit` use) needs a test that fails when it is removed, asserting payload **and** call count.
- **Mutation controls** on load-bearing lines are listed in FTID §8 (M1–M14). Back up, edit one line, run, restore, confirm `git diff` clean and re-grep the line before the next. Record each control in the review notes. A control that "survives" may not have landed — verify the edit took.
- **Registries, not hand-kept lists:** served endpoints from `app.openapi()`, statuses from one enum, transitions from one function.
- Re-verify every line number from the FTID before editing; sibling features move them.

### Category Checklist Results

- **Data layer:** 1.x
- **Backend/API:** 3.x, 5.x
- **Frontend/UI:** N/A — FPRD §4 specifies no Admin UI surface; progress is a Socket.IO event (5.8).
- **Business logic:** 4.x (validation), 5.x (runner), 6.x (packing), 7.x (probe rows)
- **Integration wiring:** 2.x (queue, admission, unload drain), 4.2 (route `validate_*`), 6.4 (provenance), 7.x (probe events), 5.10 (lifespan), 8.3 (health backlog)
- **Error handling & logging:** 3.4, 4.5, 5.6, 5.7, 6.3
- **Testing:** sub-tasks throughout; integration 9.2; performance/hardware 9.3–9.5; mutation controls 9.6
- **Performance & security:** 3.3 (caps while streaming), 6.3 (pack bounds, OOM halving), 3.6 (path safety), 9.3 (throughput)
- **Configuration/deployment:** 8.1 (config, `.env.example`, k8s, compose)
- **Documentation:** 8.2 (MCP contract), 8.4 (manual API page)

## Tasks

- [ ] 0.0 Gates and spikes (covers the FTDD §14 open items; no product question remains open — FPRD §14)
  - [x] 0.1 **Gate: Feature 25 contract.** Confirm 025 shipped `request_policy.evaluate(request, endpoint, engine, strict=...)`, the JSON-schema subset check and `_score_prompts` as its FTDD (§1, §5) specifies. Blocks 4.x and 6.1.
    - *Verified 2026-10-06 at `6cce090`:* `millm/api/request_policy.py` `evaluate(request, endpoint, engine, *, strict)` (keyword-only `strict`), `json_schema_subset.check(schema)`, and `InferenceService._score_prompts(texts, *, add_special_tokens, allowed, temperature, top_k, …)` all present. **Code wins:** the scorer takes ONE option set for all texts, not per-spec options; the packed path (6.x) therefore adds a per-spec function beside it rather than widening this signature.
  - [x] 0.2 **Gate: Feature 29 contract.** Confirm `ModelService` offers in-process `acquire_lease`, `renew_lease`, `release_lease`, `resolve_lease` and `get_lease`, the errors `ModelLeasedError`, `ModelNotResidentError`, `LeaseNotFoundError` and `LeaseExpiredError`, plus `RequestQueue.holding_count` and `register_backlog_provider`, as 029 FTDD §2 and §6 fix them; and that a restart ends every lease (029 FR-29.1.9). (Stage 3, 2026-10-06, requested by 029) Blocks 5.4.
    - *Verified 2026-10-06:* `ModelService.acquire_lease(model_id, holder, reason, ttl_seconds)`, `renew_lease`, `release_lease`, `get_lease`, `resolve_lease(lease_id)`; `ModelLeasedError`, `ModelNotResidentError`, `LeaseNotFoundError`, `LeaseExpiredError` in `core/errors.py`; `RequestQueue.holding_count`; `clear_leases_on_startup()` called in `lifespan`. **Code wins:** `register_backlog_provider` lives in `millm/core/backpressure.py`, not on `ModelService`; `acquire_lease` also raises `ModelBusyError` during a load/unload (FTDD §7 step 2 maps it to `model_not_resident`, as implemented).
  - [x] 0.3 **Gate: Feature 27 and 28 contracts.** Confirm 027's scoring service function and 028's steering-value function (FR-28.3.10). Blocks 7.x and 6.4.
    - *Verified 2026-10-06:* 027's scoring service is `ProbeScoringService(repository, inference).score(request, session)` behind `POST /api/probes/score`; batch probe events plumb through `ProbeEventService.record`. **Feature 28 is NOT in the tree** (no steering-value function, no `X-miLLM-Steering` header outside scoring-with-activations). FR-26.10.1 is implemented over the headers the synchronous routes set TODAY (`provenance.py`); the steering value joins when 028 lands — recorded as a deviation. **Feature 30 is NOT in the tree**, so embedding lines stay `packed: false` (6.6, as the FTASKS foresaw). **`ProbeScoreRequest` has no `model` field (`extra="forbid"`)**, so FR-26.2.7's "every line names `body.model`" cannot hold for probe lines: a probe batch's model is the resident model at validation — recorded discrepancy.
  - [ ] 0.4 **Spike: sensing events from batch generation rows** (FTDD §14 item 1). Measure whether a 5,000-row generation batch with sensing armed evicts live sensing history. If it does, extend 7.2's `origin` marking to sensing and circuit-edge sensing events and record the decision; if not, record the measurement.
  - [x] 0.5 **Spike: `logits_to_keep` index tensor on served architectures.** On LFM2, Gemma, Llama and Granite model classes in the installed transformers, confirm the forward accepts a tensor; list any that need the full-logits fallback (FTID §7).
    - *Read 2026-10-06 against the installed transformers (5.x in `~/app/miLLM/venv`):* `slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep` in `llama`, `lfm2`, `gemma2`, `gemma3`, `granite`, `granitemoehybrid` (`grep` of each `modeling_*.py`); a tensor is accepted by all six. The packed path keeps the `TypeError` fallback to full logits for any class that rejects it, and a test runs the packed gather on a tiny real Llama with a tensor `logits_to_keep`.

- [x] 1.0 Data layer (covers FR-26.3.1, FR-26.3.7, FR-26.4.8, FR-26.9)
  - [x] 1.1 `millm/db/models/batch.py`: `BatchFile`, `Batch`, `BatchRow` per FTDD §4; register in `millm/db/models/__init__.py`.
  - [x] 1.2 `probe_events` columns `origin`, `batch_id`, `batch_line` and the partial unique index in `millm/db/models/probe.py`.
  - [x] 1.3 Migration `0NN_add_batch_api.py` (next free number), with downgrade. → `019_add_batch_api.py`; `tests/schema` (alembic drift ratchet + migration round trip) green against a local PostgreSQL 15.
  - [x] 1.4 `BatchRepository` methods (FTID §4), every criteria delete/update with `synchronize_session=False`; `record_chunk` guarded by `state='pending'`.
  - [x] 1.5 Tests: status CHECK rejects a ninth value; `(batch_id, line_no)` uniqueness; `record_chunk` refuses to overwrite a recorded row; cascade; schema parity covers the new tables; existing `probe_events` rows read `origin='live'`. → `tests/unit/db/test_batch_models.py` (13). **Code wins:** `batch_rows.state` is `pending|invalid|done|failed`; FTDD §4's `cancelled`/`expired` row states are not stored, because unrun rows are written to the error file at assembly and the rows are deleted in the same transaction (FTDD §7) — a stored state nobody reads would be a second source of truth.

- [ ] 2.0 Queue and admission (covers FR-26.4.1 – FR-26.4.4)
  - [ ] 2.1 `RequestQueue.acquire_background` with `_background_waiting`, `_background_holding`, `_holding`, a `Condition`, interactive priority, and the `_idle` rule (FTID §3).
  - [ ] 2.2 `occupied_count` property; switch the unload drain (`model_service.py:1258`) and the idle-release checks (`inference_service.py:725`, `:750`) to it. Leave `pending_count` and the `QUEUE_FULL` check (`chat.py:244`) unchanged.
  - [ ] 2.3 `_admit(background=True)` and owner re-entry through `_SLOT_OWNER` (FTID §3).
  - [ ] 2.4 Extend `test_every_request_queue_slot_is_taken_through_admission` (`tests/unit/services/test_unload_admission.py:451`) so `acquire_background` has exactly one caller, `_admit`.
  - [ ] 2.5 Tests: a background holder does not raise `pending_count` and never gets `QUEUE_FULL`; with a batch running, the 11th interactive request still gets `QUEUE_FULL` exactly as before (threshold unchanged); an interactive waiter runs before the next chunk; `wait_idle` waits for a background holder; nested `_admit` in the owner task re-enters; a child task created inside the slot queues; the unloading refusal still fires on re-entry.
  - [ ] 2.6 Edge case: unload during a chunk — the drain waits for the chunk; the next chunk's `_admit` refusal makes the batch wait, not fail. Implement in 5.6; test here with a fake model service.

- [ ] 3.0 Files surface, limits and retention (covers FR-26.1.1, FR-26.6.5, FR-26.6.9, FR-26.6.10, FR-26.8, FR-26.9)
  - [ ] 3.1 `FileStore` (`write`, `read_line`, `assemble`, `delete`, orphan sweep); generated paths under `BATCH_FILES_DIR`.
  - [ ] 3.2 Routes `POST /v1/files`, `GET /v1/files`, `GET /v1/files/{id}`, `GET /v1/files/{id}/content` (`application/jsonl`), `DELETE /v1/files/{id}`; include `files_router` in `openai_router`.
  - [ ] 3.3 Limits: `Content-Length` pre-check, byte cap while copying, row count; refusal names limit and measured value; nothing stored.
  - [ ] 3.4 Errors: bad `purpose` → `400`; unknown id → `404`; expired/deleted content → `404 file_expired|file_deleted`; delete of a referenced file → `409 file_in_use` naming the batch.
  - [ ] 3.5 Retention: `expires_at` on every file (30 days input — T-68; output/error by `output_expires_after`, default 30 days); `prune_expired_files(now)` skips files referenced by a non-terminal batch; hourly loop plus startup run.
  - [ ] 3.6 Security: client filename never reaches the filesystem; per-line cap.
  - [ ] 3.7 Tests: OpenAI file-object fields; list shape and paging; over-row and over-byte uploads refused, nothing on disk; content streams with the stated media type; delete refused while referenced, allowed after; prune with an injected clock removes only expired, unreferenced files; orphan sweep. **Reachability:** each route present in `create_app().openapi()["paths"]` and exercised via `TestClient`; removing the router include turns these red (M12).

- [ ] 4.0 Validation before any row runs (covers FR-26.1.2 – FR-26.1.6, FR-26.2, FR-26.7.3)
  - [ ] 4.1 Schemas in `millm/api/schemas/batch.py`: create request (incl. `pack`, `output_expires_after`, `metadata`), batch object with `millm` extension, list object.
  - [ ] 4.2 Extract `validate_<endpoint>(request, model_row)` from chat, completions and embeddings routes; routes call it; snapshot test that synchronous responses are unchanged.
  - [ ] 4.3 `POST /v1/batches`: file checks, `served_batch_endpoints(app)` from `app.openapi()["paths"]`, `completion_window` `^([1-9][0-9]*)h$` up to the configured maximum (T-65), `output_expires_after` bounds, lease check → `409 model_leased` unless `X-miLLM-Lease` matches; unknown fields per Feature 25; tolerate `X-miLLM-Load-Policy`.
  - [ ] 4.4 `BatchValidator` per FTID §7 (parse, shape, duplicate `custom_id`, `stream`, schema, strict, model resolution, refusals, kind); single-model rule; zero-valid → `failed` (T-67); bulk row insert; `in_progress` queued; cancel flag every 1,000 lines.
  - [ ] 4.5 Validation errors: error-file line per invalid line with line number and reason; `errors.data` in OpenAI's shape, first `BATCH_ERRORS_SHOWN`; counters (`total` = lines, invalid in `failed`).
  - [ ] 4.6 Tests (one per edge case, implement + test): malformed line; wrong `url`; duplicate `custom_id`; `stream: true`; schema error; strict unused field named; output-changing refusal with the synchronous message; scoring on a GGUF row refused; two models → `failed` listing counts; zero valid → `failed`, no lease taken, no row run; `/api/probes/score` accepted only when its route is registered (remove the route in a test app → refused); `24h`, `72h`, `0h`, `200h`, `1d`; lease held by another → `409`; validation takes no slot (queue spy).

- [ ] 5.0 Runner, lease, cancel, expiry and resume (covers FR-26.3, FR-26.4.5 – FR-26.4.7, FR-26.6.1 – FR-26.6.4, FR-26.6.6 – FR-26.6.8, FR-26.7)
  - [ ] 5.1 `BatchRunner` loop (FTID §3): FIFO, chunk rule (`chunk_rows`), `_admit(background=True)`, `BATCH_ROW` context per row, `record_chunk` per chunk.
  - [ ] 5.2 Executors (single-row): generation via `create_chat_completion` / `create_text_completion`; scoring via `_score_prompts(pack_size=1)`; embeddings via the synchronous body; error lines via `create_openai_error`.
  - [ ] 5.3 Batch-row clause in `_use_cbm_for_request` (`inference_service.py:958`): no row on the continuous batching manager.
  - [ ] 5.4 Lease handling (FTDD §7): own lease acquire/renew at a third of TTL/release in `finally`; caller lease from create header or `POST /v1/batches/{id}/lease`, renewed and never released (T-64); re-acquire or wait after restart (X-01, T-66); `waiting_reason` `queued|model_not_resident|lease_unavailable`.
  - [ ] 5.5 Cancel: `POST /v1/batches/{id}/cancel` → `cancelling`; no new row; check between rows of an unpacked chunk; finish the running forward; cancelled while validating → `cancelled`; terminal → `409`.
  - [ ] 5.6 Expiry and waiting: `expires_at = created_at + window`; expired → assemble with unrun rows as `batch_expired`; unloading refusal mid-run → wait, not fail; unexpected exception → `failed` with `errors`, lease released.
  - [ ] 5.7 Finalisation: `.partial` → fsync → rename → one transaction (files, terminal, delete rows); unrun cancelled rows as `batch_cancelled`; empty error file → null id; output in input-line order.
  - [ ] 5.8 `ProgressEmitter.emit_batch_progress` (`batch:progress`, `{id, status, request_counts}`), throttled, on every chunk and transition.
  - [ ] 5.9 `reconcile_batches_on_startup`: `validating` re-validates; `in_progress` resumes; `finalizing` re-assembles; `cancelling` → `cancelled`; stray `.partial` removed; input sha256 checked.
  - [ ] 5.10 Lifespan (`millm/main.py:277`): reconcile → `BatchRunner.start()` → retention loop, after `disarm_probes_on_startup` (`main.py:379`); runner stopped on shutdown.
  - [ ] 5.11 `BatchRunner.backlog_rows()`, registered through Feature 29's `register_backlog_provider` at runner start, for `batch_backlog_rows` (FR-29.7.3); `background_holding_count` added to `in_flight`. (Stage 3, 2026-10-06, requested by 029)
  - [ ] 5.12 Tests: status transitions only along the FR-26.3.5 table; output-line and error-line shapes; counters per chunk; **exactly-once** across an injected crash mid-chunk and between chunks (fresh runner + reconcile); each reconciliation branch; cancel at a row boundary keeps completed rows; expiry path; own lease released on success, cancel, expiry, failure and exception; caller lease never released; after a simulated restart the batch waits with `lease_unavailable` until the lease frees, and runs under a handed-over caller lease; a batch never calls `load_model`/`unload_model` (spy, call count 0); interactive chat proceeds while the batch holds the lease; progress emitted with payload and count; **wiring:** removing the lifespan reconcile call, the runner start or the emit turns a test red (M11).

- [ ] 6.0 Packing and provenance (covers FR-26.5, FR-26.10.1)
  - [ ] 6.1 Add `pack_size` to Feature 25's `_score_prompts`; the synchronous text and chat scorers call it with `pack_size=1`.
  - [ ] 6.2 Packed forward: right padding per call, `logits_to_keep` index tensor, per-row gather, unsteered; full-logits fallback per 0.5.
  - [ ] 6.3 Pack bounds (`BATCH_PACK_MAX_ROWS`, `BATCH_PACK_MAX_TOKENS`) and OOM halving to single; single-row OOM becomes the row's error line.
  - [ ] 6.4 `millm/api/provenance.py`; routes and executors both call it; Feature 28's steering value and Feature 25's seed/constrained values come from it; output lines carry `response.millm.headers` and `packed`.
  - [ ] 6.5 `pack` default from `BATCH_PACK_DEFAULT` (T-63); generation and probe rows always single; llama.cpp rows single.
  - [ ] 6.6 Embedding packing after Feature 30's mask-aware pooling; until then embedding lines say `packed: false`.
  - [ ] 6.7 Tests: synchronous scoring bit-identical before and after the refactor; `pack: false` row equals the synchronous request (token ids and logprobs); mixed-length packed rows score their own last real token on a tiny real Llama (M13); probe and generation rows never packed regardless of `pack`; OOM halving retries unrecorded rows only; a batch line's `millm.headers` equals the synchronous response's `X-miLLM-*` headers for the same request (steering none/profile/inline, seed).

- [ ] 7.0 Probe rows and batch-marked probe events (covers FR-26.1.4 for `/api/probes/score`, FR-26.4.8, FR-26.5.3)
  - [ ] 7.1 `ProbeExecutor` calling Feature 27's scoring service; always single; result body is the endpoint's `ApiResponse`.
  - [ ] 7.2 `_probe_record` (`inference_service.py:2615`) passes `BATCH_ROW`; `ProbeEventService.record` sets `origin='batch'`, `batch_id`, `batch_line`; `prune_to_cap` (`probe_repository.py:180`) filters by origin with `PROBE_MAX_BATCH_EVENTS_PER_PROBE`; conflict skip; no live emit.
  - [ ] 7.3 Tests: a batch of more than `PROBE_MAX_EVENTS_PER_PROBE` generation rows leaves every live event in place (M10); a re-run row records no second event; batch events never reach the socket; probe-score rows produce no `probe_events` (027 FR-27.4f) and run one input at a time.

- [ ] 8.0 Configuration, contract and documentation (covers FR-26.8.1, FR-26.10.2)
  - [ ] 8.1 `BATCH_*` and `PROBE_MAX_BATCH_EVENTS_PER_PROBE` in `millm/core/config.py` and `.env.example`; `BATCH_FILES_DIR` and the `mkdir -p` entry in `k8s/base/backend.yaml` (`:36`, `:55`); `batch_files` volume in `docker-compose.yml`. Boolean settings fail to their default.
  - [ ] 8.2 `docs/mcp-contract.md`: section for files and batches routes — paths, payloads, statuses, limits, `completion_window` extension, `/lease` extension, media type `application/jsonl`, `millm` extension objects, packed-versus-single measurement (filled by 9.4). Add the batch routers to `tests/unit/test_mcp_tool_paths_are_real.py`'s served set.
  - [ ] 8.3 Feature 29 health: `batch_backlog_rows` reads `backlog_rows()`; test that it is a number (not `null`) once this feature is live and counts pending rows.
  - [ ] 8.4 `manual/docs/api/batches.md` (registered in the sidebar): the journey, limits, retention, packing and its measured difference, resume and lease behaviour, the waiting reasons.

- [ ] 9.0 Feature Acceptance (covers all FRs; BRD-04 acceptance 7, 8, 9)
  - [ ] 9.1 Walk FPRD FR-26.1 – FR-26.10 and §11 one by one; tick each against a passing test.
  - [ ] 9.2 Integration (`tests/integration/test_batch_workflow.py`): upload → create → validate → run → finalize → download, against the real app and a tiny real model; cancel mid-run; restart mid-run.
  - [ ] 9.3 **Hardware — acceptance 7.** On the node with JEV-9B-decision in bfloat16: a 10,000-row scoring batch with `pack: false` completes at **≥ 19 rows per second**, with `batch:progress` and `GET /v1/batches/{id}` counts visible throughout. Repeat packed; record the packed rate.
  - [ ] 9.4 **Hardware — packing difference (FR-26.5.8, T-63).** Same 10,000 rows packed and single: maximum absolute logprob difference, top-token agreement rate, embedding cosine on an embedding batch. Publish in the API reference and contract. **If any top-token label differs, set `BATCH_PACK_DEFAULT=false`** in config, `.env.example` and k8s, and record it.
  - [ ] 9.5 **Hardware — acceptance 8 and 9.** Cancel at about 3,000 rows: every completed row kept, none run twice, unrun rows `batch_cancelled`. Second batch: restart the pod mid-run; the lease is gone (X-01); the batch re-acquires one when the model is resident and resumes with no recorded row repeated; with another holder's lease live, it waits (`lease_unavailable`) and runs no row. Send a chat request during a running batch; it is answered within one chunk's duration.
  - [ ] 9.6 Run and record mutation controls M1–M14 (FTID §8); every one must turn the suite red. Re-mutate any fix a review round makes.
  - [ ] 9.7 Full suite: `pytest tests/unit`, integration, `ruff`, `mypy millm/`; also with `0xcc/` hidden (the public mirror's checkout).
  - [ ] 9.8 Update the PPRD Feature 26 status and CLAUDE.md inventory in the implementation session (not in this planning pass); file follow-ups (0.4 outcome) as tasks.

## Coverage Audit

- **FRs → tasks:** 26.1 → 3.2, 4.1, 4.3, 7.1 · 26.2 → 4.4, 4.5 · 26.3 → 1.x, 5.1, 5.6, 5.7, 5.9 · 26.4 → 2.x, 5.1–5.3, 5.11, 7.2 · 26.5 → 6.1–6.3, 6.5, 6.6, 7.1, 9.4 · 26.6 → 3.2, 5.5, 5.7, 5.8, 3.4 (list/delete 26.6.9–26.6.10) · 26.7 → 4.3, 5.4 · 26.8 → 3.3, 8.1 · 26.9 → 3.5 · 26.10 → 6.4, 8.2. **Every FR group is cited by a parent task.**
- **Acceptance criteria (implement + test):** 7 → 5.x, 6.x / 9.3, 9.4 · 8 → 5.4, 5.5, 5.9 / 5.12, 9.5 · 9 → 2.1 / 2.5, 9.5.
- **Edge cases (implement + test):** malformed / no-valid / two-model files → 4.4 / 4.6 · lease held → 4.3, 5.4 / 4.6, 5.12 · model not resident → 5.4, 5.6 / 5.12 · over-limit upload → 3.3 / 3.7 · restart in each state → 5.9 / 5.12 · row-level failure → 5.2 / 5.12 · OOM in a pack → 6.3 / 6.7 · unload during chunk → 5.6 / 2.6.
- **TDD sections:** Data 1.x · API 3.x, 4.x · Components 5.x, 6.x · State 2.x, 5.x · Security 3.6 · Performance 6.3, 9.3 · Testing throughout, 9.6 · Deployment 8.1. **TID sections:** files 1–8 above; frontend N/A (checklist); config 8.1; integration 2.2, 4.2, 6.4, 7.2, 8.3; errors 3.4, 4.5, 5.6.
- **Open questions:** none open in the FPRD (all decided, §14). The FTDD's two technical open items → 0.4 and 0.1–0.3.
- **The final parent task is Feature Acceptance.** ✔
