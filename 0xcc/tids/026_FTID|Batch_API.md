# Technical Implementation Document: Batch API

## miLLM Feature 26

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**Inputs:** `026_FPRD|Batch_API.md` v1.1 · `026_FTDD|Batch_API.md` v1.0 · PADR v1.5 §10
Code references are to miLLM at `7aa659c`, verified 2026-10-06. **Re-verify every line number before editing**; Features 25, 27, 28 and 29 land first and will move them.

---

## 1. Implementation Overview

Build a durable batch job on top of the code that already serves synchronous requests. Four principles:

1. **One path to the model.** Every row runs through the synchronous service method for its endpoint. The runner never calls the model (FR-26.4.5).
2. **One admission path.** Every slot comes from `_admit()` (`millm/services/inference_service.py:649`). The batch's slot is a background slot; nested `_admit()` calls in the same task re-enter it (FTDD §7).
3. **Recorded means committed.** A row is done only when its chunk's transaction has committed (FR-26.3.2).
4. **State from the registry, never a hand-kept list.** Served endpoints come from `app.openapi()["paths"]`; the status set from one enum; the transition table from one function.

Integration points: `RequestQueue` (`millm/services/request_queue.py`), `InferenceService._admit`, `_score_text_completion` (`:4930`), `_next_token_logits` (`:5043`), `create_embeddings` (`:5071`), `ModelService.unload_model` drain (`millm/services/model_service.py:1258`), `ProbeEventService.record` (`millm/services/probe_event_service.py:48`), `ProgressEmitter` (`millm/sockets/progress.py:124`), `lifespan` (`millm/main.py:277`).

## 2. File Structure and Organization

**New:**
- `millm/services/batch/__init__.py` — exports `BatchRunner`, `get_batch_runner`.
- `millm/services/batch/state.py` — `BatchStatus`, `RowState`, `RowKind` enums; `transition(batch, to)`; `batch_object(batch)`, `file_object(file)`, `list_object(...)`.
- `millm/services/batch/files.py` — `FileStore`.
- `millm/services/batch/validator.py` — `BatchValidator`.
- `millm/services/batch/executors.py` — `ScoringExecutor`, `GenerationExecutor`, `EmbeddingExecutor`, `ProbeExecutor`.
- `millm/services/batch/runner.py` — `BatchRunner`.
- `millm/services/batch/retention.py` — `prune_expired_files`, `retention_loop`.
- `millm/services/batch/reconcile.py` — `reconcile_batches_on_startup`.
- `millm/db/models/batch.py` — `BatchFile`, `Batch`, `BatchRow`.
- `millm/db/repositories/batch_repository.py` — `BatchRepository`.
- `millm/db/migrations/versions/0NN_add_batch_api.py` — next free number at implementation time.
- `millm/api/schemas/batch.py` — request/response models (OpenAI shapes plus `millm` extension objects).
- `millm/api/routes/openai/files.py`, `millm/api/routes/openai/batches.py`.
- `millm/api/provenance.py` — `response_provenance(...)`.
- Tests under `tests/unit/services/batch/`, `tests/unit/api/test_batch_routes.py`, `tests/unit/db/test_batch_models.py`, `tests/integration/test_batch_workflow.py`, `tests/performance/test_batch_throughput.py` (hardware-marked).

**Modified:**
- `millm/services/request_queue.py` — `acquire_background`, counters, `occupied_count`, `_idle` rule.
- `millm/services/inference_service.py` — `_admit(background=...)`, owner context variable, `_score_prompts`, packed `_next_token_logits_at`, batch-row clause in `_use_cbm_for_request` (`:958`).
- `millm/services/model_service.py` — drain uses `occupied_count` (`:1258`).
- `millm/services/probe_event_service.py`, `millm/db/repositories/probe_repository.py`, `millm/db/models/probe.py` — batch origin.
- `millm/api/routes/openai/__init__.py` — include `files_router`, `batches_router`.
- `millm/api/routes/openai/chat.py`, `completions.py`, `embeddings.py` — extract `validate_<endpoint>`; header building moves to `provenance.py`.
- `millm/sockets/progress.py` — `emit_batch_progress`.
- `millm/main.py` — lifespan wiring.
- `millm/core/config.py`, `.env.example`, `k8s/base/backend.yaml`, `docker-compose.yml`.
- `docs/mcp-contract.md`, `manual/docs/api/` (batch page), `tests/unit/services/test_unload_admission.py`.

Imports: `services/batch` imports from `services/inference_service` and `db`; routes import `services/batch`; nothing in `inference_service` imports `services/batch` except the `BATCH_ROW` context variable, which lives in `services/batch/state.py` to avoid a cycle.

## 3. Component Implementation Hints

**`RequestQueue.acquire_background`** — mirror `acquire`'s release-only-what-was-acquired discipline (`request_queue.py:118-170`):

```python
@asynccontextmanager
async def acquire_background(self):
    async with self._cond:                      # Condition over self._lock
        self._background_waiting += 1
        self._idle.clear()
        await self._cond.wait_for(lambda: self._pending - self._holding == 0)
    acquired = False
    try:
        await self._semaphore.acquire()
        acquired = True
        async with self._cond:
            self._background_waiting -= 1; self._background_holding += 1
        yield
    finally:
        async with self._cond:
            if acquired:
                self._background_holding -= 1
                self._semaphore.release()
            else:
                self._background_waiting -= 1
            self._set_idle_if_empty()
            self._cond.notify_all()
```

`acquire` gains `_holding` (incremented after its semaphore acquire, decremented in `finally`) and calls `notify_all` when it releases or when a waiter leaves, so a background waiter wakes when the last interactive waiter has its slot. A race where interactive arrives between `wait_for` and the semaphore is acceptable (it waits one chunk), and is what the FPRD's "within one chunk" allows.

**Re-entrant `_admit`:**

```python
_SLOT_OWNER: ContextVar[Optional[asyncio.Task]] = ContextVar("millm_slot_owner", default=None)

@asynccontextmanager
async def _admit(self, raise_refusal=True, background=False):
    if _SLOT_OWNER.get() is asyncio.current_task() and _SLOT_OWNER.get() is not None:
        refusal = self._unloading_refusal()
        if refusal is not None and raise_refusal: raise refusal
        yield refusal; return
    acquire = self._request_queue.acquire_background if background else self._request_queue.acquire
    ...existing body, using `acquire()`; inside the held slot: token = _SLOT_OWNER.set(current_task) ... finally _SLOT_OWNER.reset(token)
```

Keep the existing double check of the unloading refusal and the idle-release bookkeeping unchanged for the outer acquisition. The re-entry branch does no bookkeeping.

**`BatchRunner`** — one instance per process (`get_batch_runner()`), `start()` creates the task, `stop()` cancels it at shutdown. Loop:

```python
while True:
    batch = await repo.next_runnable()            # in_progress, FIFO by created_at
    if batch is None: await self._wake.wait(); continue
    if not await self._ensure_lease(batch): await asyncio.sleep(poll); continue
    rows = await repo.next_pending(batch.id, n=chunk_rows(batch))
    if not rows: await self._finalize(batch, BatchStatus.COMPLETED); continue
    if self._cancelled(batch) : await self._finalize(batch, BatchStatus.CANCELLED); continue
    if now() >= batch.expires_at: await self._finalize(batch, BatchStatus.EXPIRED); continue
    async with inference._admit(background=True):
        results = await executor_for(batch).run(rows, pack=batch.pack, cancel=self._cancel_event(batch))
    await repo.record_chunk(batch.id, results)    # one transaction
    self._emit(batch)
```

`run` checks the cancel event between rows of an unpacked chunk (FR-26.6.4: row boundary). Each row runs inside `BATCH_ROW.set((batch_id, line_no))`.

**Executors.** Parse the row's line from the input file (`FileStore.read_line(offset, length)`), build the endpoint's request model, call the service, and wrap the result as an OpenAI output line. A raised `MiLLMError` becomes an error line with the status code and OpenAI error body the route's error mapper produces (reuse `create_openai_error` from `millm/api/routes/openai/errors.py`, not a copy).

## 4. Database Implementation Approach

- Models in `millm/db/models/batch.py`, registered in `millm/db/models/__init__.py`. Use `JSONVariant` exactly as `millm/db/models/probe.py` does (JSON with a JSONB variant on PostgreSQL).
- `batches.status` CHECK constraint over the eight values; `batch_rows` composite primary key `(batch_id, line_no)`; index `(batch_id, state, line_no)`; foreign keys `batch_rows.batch_id → batches.id ON DELETE CASCADE`, `batches.*_file_id → batch_files.id ON DELETE SET NULL`.
- `probe_events`: `origin` NOT NULL server default `'live'`; `batch_id`, `batch_line` nullable; partial unique index (PostgreSQL `postgresql_where`, SQLite `sqlite_where`).
- Repository methods: `create_file`, `create_batch`, `bulk_insert_rows` (1,000 per statement), `next_runnable`, `next_pending(batch_id, n)`, `record_chunk(batch_id, results)` (updates rows by primary key and the three counters in one transaction, guarded by `state='pending'` in the WHERE so a row already recorded cannot be overwritten), `set_status`, `finalize(batch_id, output_file, error_file)` (insert files, set terminal, delete rows), `expired_files(now)`, `referenced_by_active(file_id)`.
- Use `synchronize_session=False` on every criteria `DELETE`/`UPDATE` (the reason is recorded at `millm/db/repositories/probe_repository.py:158-170`).
- The schema-parity check (`millm/db/schema_parity.py`) must cover the new tables.
- Downgrade: drop index, columns, tables in reverse order.

## 5. API Implementation Strategy

- Two routers, `files.py` and `batches.py`, included in `openai_router` (`millm/api/routes/openai/__init__.py`). They inherit the `/v1` prefix from `register_routes` (`millm/api/routes/__init__.py`).
- `POST /v1/files`: read `Content-Length`; over `BATCH_MAX_FILE_BYTES` plus a small multipart allowance → `400` before reading the body. Then `FileStore.write(upload.file)` copies in 1 MiB chunks, counting bytes, newlines and sha256; over a cap → delete the partial file, `400` naming limit and measured value. Starlette spools large uploads to a temporary file, not memory.
- `GET /v1/files/{id}/content`: `StreamingResponse` over the stored file, `media_type="application/jsonl"`; `404` with code `file_expired` or `file_deleted` when the bytes are gone.
- `POST /v1/batches`: synchronous checks only — input file exists, purpose `batch`, not expired; `endpoint in served_batch_endpoints(request.app)`; `completion_window` matches `^([1-9][0-9]*)h$` within the configured maximum; `output_expires_after` within 3,600–2,592,000 seconds; lease: if Feature 29 reports a live lease and the request's `X-miLLM-Lease` does not match it → `409 model_leased` (`MODEL_LEASED`). The match uses `resolve_lease(lease_id)` (029 FTDD §2). (Stage 3, 2026-10-06, requested by 029) Then insert `validating`, start the validator task, return the batch object.
- `X-miLLM-Lease` on create, when valid, is stored in the runner's memory against the batch (`lease_mode='caller'`).
- `POST /v1/batches/{id}/lease`: requires a live `X-miLLM-Lease` for the batch's model and a non-terminal batch; sets caller mode and wakes the runner. `resolve_lease(lease_id)` returns the live `LeaseRecord` (`model_id`, holder, `expires_at`) or `None`. `None`, or a lease on another model, is `404 LEASE_NOT_FOUND`, as Feature 29's routes answer an ID that does not match (029 FTDD §5.1). (Stage 3, 2026-10-06, requested by 029)
- Unknown request fields on these routes follow Feature 25's mechanism (FR-26.1.6).
- Errors use the existing envelope helpers (`create_openai_error`, `validation_error`).

## 6. Frontend Implementation Approach

N/A. The FPRD specifies no Admin UI surface (FPRD §4). Progress reaches clients through the `batch:progress` Socket.IO event, emitted by `ProgressEmitter`; no admin-ui component subscribes in this feature.

## 7. Business Logic Implementation Hints

**Validation (`BatchValidator.validate`, run with `asyncio.to_thread`).**
For each line (iterate the file, tracking byte offsets):
1. Over `BATCH_MAX_LINE_BYTES` → invalid `line_too_large`.
2. `json.loads` → invalid `invalid_json` on failure; not an object → `invalid_line`.
3. Shape: `custom_id` string (≤ 512), `method == "POST"`, `url == batch.endpoint`, `body` object; duplicate `custom_id` → `duplicate_custom_id`.
4. `body.stream` true → `stream_not_supported`.
5. Endpoint schema: build the request model (`ChatCompletionRequest` etc. from `millm/api/schemas/openai.py`); a pydantic error → `invalid_request` with the field path.
6. Strict: `request_policy.evaluate(request, endpoint, engine, strict=True)` (025 FTDD §5) → `UNUSED_FIELDS_REFUSED` naming every location (FR-26.2.3).
7. Model: resolve `body.model` with `ModelService.find_model_by_name` (`model_service.py:1523`), cached per name; unknown → `model_not_found`.
8. Refusals: the same `evaluate` call returns `FIELD_NOT_HONOURED` for the output-changing table (engine rows included); then the JSON-schema subset check and the route's `validate_<endpoint>(request, model_row)` → the same code and message as the synchronous route.
9. Classify `kind`: scoring (`wants_scores()` on completions; Feature 25's predicate on chat), generation, embedding, probe.
Then: all valid lines must resolve to one model id, else `failed` with `errors.data` listing `model → count`. Zero valid → `failed` (T-67). Otherwise bulk-insert rows, set counters, set `in_progress` with `waiting_reason='queued'`, wake the runner. Check the cancel flag every 1,000 lines.

**Packed scoring (`_score_prompts`).** Feature 25 extracts `_score_text_completion`'s per-prompt loop into `_score_prompts` (025 FTDD §1). This feature adds a `pack_size` parameter to it, where each spec carries the prompt text, `add_special_tokens`, `allowed_token_ids`, `temperature`, `logprobs`, `return_tokens_as_token_ids`. The synchronous path calls it with `pack_size=1`, so its behaviour and its error messages are unchanged. For a pack:
1. Tokenise each spec separately (special tokens per spec), check each with `_check_context_length`.
2. Right-pad to the pack's longest (`padding_side="right"` passed per call — never mutate `self._tokenizer.padding_side`; the reason is at `inference_service.py:3186-3193`).
3. `last = lengths - 1`; `keep = torch.unique(last)`; forward with `logits_to_keep=keep`, `use_cache=False`, inside `_unsteered()`; gather row `i` at the index of `last[i]` within `keep`. On `TypeError` mentioning `logits_to_keep`, forward without it and gather from full logits — allowed only because the pack token budget bounds the size.
4. Apply the existing NaN/inf checks and `next_token_scores` per row, unchanged.
5. A CUDA out-of-memory error: release memory as `_next_token_logits` does, halve the pack, retry; at size 1 the row gets the out-of-memory error line.
Chat scoring rows (Feature 25) render the template first, then enter the same function.

**Embedding packing.** Only after Feature 30's mask-aware pooling exists. Until then the executor calls the synchronous embedding body one row at a time and writes `packed: false`. llama.cpp embedding rows are never packed (FR-26.5.7).

**Generation rows.** Call `create_chat_completion` / `create_text_completion` with the parsed request. The batch-row clause in `_use_cbm_for_request` returns False while `BATCH_ROW` is set, so a row never runs outside the slot on the continuous batching manager.

**Probe rows.** Call Feature 27's scoring service with the parsed body. It runs one input at a time and re-enters `_admit` per input (027 FR-27.6a, b). Always `packed: false`.

**Probe events (T-70).** `_probe_record` (`inference_service.py:2615`) passes `BATCH_ROW.get()` into `ProbeEventService.record`. When set, rows carry `origin='batch'`, `batch_id`, `batch_line`; `prune_to_cap` (`probe_repository.py:180`) filters by origin and uses `PROBE_MAX_BATCH_EVENTS_PER_PROBE` for batch events; insertion skips a conflicting `(probe_id, batch_id, batch_line, window)`; `_emit_events` is skipped for batch events.

**Provenance.** `response_provenance(endpoint, request, response, service)` returns the `X-miLLM-*` dict a route sets after generation. Routes call it and copy into `response.headers`; executors put it under `response.millm.headers`. Feature 28's steering-value function and Feature 25's seed/constrained values are called from here, once.

**Finalisation.** `FileStore.assemble(batch)` streams rows ordered by `line_no`; `done` rows to the output file, `invalid`/`failed`/unrun rows to the error file (unrun rows get `batch_cancelled`/`batch_expired`). Write `.partial`, fsync, rename, then the database transaction. Empty error file → `error_file_id` null, as OpenAI.

**Expiry and retention.** `expires_at = created_at + window`. Output and error files: `expires_at = created_at + output_expires_after` (default 30 days). Input files: 30 days (T-68). `prune_expired_files(now)` deletes bytes of expired files not referenced by a non-terminal batch, marks rows `expired`, then sweeps orphan paths.

## 8. Testing Implementation Approach

- **Organisation:** `tests/unit/services/batch/test_{state,files,validator,runner,executors,retention,reconcile}.py`; `tests/unit/services/test_request_queue_background.py`; `tests/unit/services/test_admit_reentry.py`; `tests/unit/api/test_batch_routes.py`; `tests/unit/db/test_batch_models.py`; `tests/integration/test_batch_workflow.py`.
- **Isolation:** each test gets its own `BATCH_FILES_DIR` (`tmp_path`) and database; never two pytest runs at once against one database.
- **Real model for packing:** build a tiny real `LlamaForCausalLM` from a config (no download). Rows must differ in length by at least two tokens, so a gather off by one cannot match by coincidence.
- **Clock injection:** retention and expiry take a `now` callable.
- **Crash simulation:** an executor that raises `SystemExit`-like cancellation after N rows; then construct a fresh runner and run reconciliation against the same database.
- **Reachability:** every route asserted present in `app.openapi()["paths"]` of the real `create_app()` and exercised through `TestClient` with payload checks; the lifespan wiring asserted by patching `reconcile_batches_on_startup` and `BatchRunner.start` and asserting call count 1 with the session factory argument.

**Mutation controls to run and record** (each must turn the suite red; restore and confirm `git diff` clean before the next):

| # | Mutation | Expected red |
|---|---|---|
| M1 | `acquire_background` increments `_pending` | pending-count test; interactive `QUEUE_FULL` threshold test |
| M2 | interactive-priority `wait_for` removed | chunk-boundary priority test |
| M3 | owner check in `_admit` compares context only (drop `is current_task()`) | child-task-queues test |
| M4 | `record_chunk` commits per row instead of per chunk | atomic-chunk crash test |
| M5 | `state='pending'` guard dropped from `record_chunk` | exactly-once test |
| M6 | cancel check between rows removed | cancel-at-row-boundary test |
| M7 | own-lease release removed from `finally` | lease released on exception test |
| M8 | caller lease released at end | T-64 test |
| M9 | retention `referenced_by_active` check removed | running-batch file survives prune test |
| M10 | `origin='batch'` not set on batch probe events | live-cap eviction test |
| M11 | `reconcile_batches_on_startup` call removed from lifespan | wiring test |
| M12 | `files_router` include removed | route reachability test |
| M13 | packed gather uses `-1` instead of `last[i]` | mixed-length packed test |
| M14 | `occupied_count` reverted to `pending_count` in the unload drain | drain-waits-for-chunk test |

## 9. Configuration and Environment Strategy

Add to `millm/core/config.py` (one `BATCH_*` block beside the `PROBE_*` block at `:241-243`) and `.env.example`:

| Setting | Default | Purpose |
|---|---|---|
| `BATCH_FILES_DIR` | `/app/batch_files` | bytes; k8s `/data/batch_files` |
| `BATCH_MAX_ROWS` | 50000 | FR-26.8.1 |
| `BATCH_MAX_FILE_BYTES` | 209715200 | FR-26.8.1 |
| `BATCH_MAX_LINE_BYTES` | 1048576 | per-line cap |
| `BATCH_PACK_DEFAULT` | `true` | T-63 switch |
| `BATCH_CHUNK_ROWS` | 8 | unpacked chunk |
| `BATCH_PACK_MAX_ROWS` / `BATCH_PACK_MAX_TOKENS` | 16 / 16384 | pack bounds |
| `BATCH_MAX_COMPLETION_WINDOW_HOURS` | 168 | T-65 |
| `BATCH_FILE_RETENTION_DAYS` | 30 | checkpoint; T-68 |
| `BATCH_LEASE_TTL_S` | 900 | own lease TTL, ≤ `LEASE_MAX_TTL_SECONDS` |
| `BATCH_WAIT_POLL_S` | 10 | waiting retry |
| `BATCH_PROGRESS_MIN_INTERVAL_S` | 1 | socket throttle |
| `BATCH_ERRORS_SHOWN` | 100 | `errors.data` length |
| `PROBE_MAX_BATCH_EVENTS_PER_PROBE` | 50000 | T-70 cap |

A boolean setting fails to its default, not to False (this suite's lesson for `dry_run`-style flags). `k8s/base/backend.yaml`: add `BATCH_FILES_DIR` beside `MODEL_CACHE_DIR` (`:55`) and add `/data/batch_files` to the init container's `mkdir -p` (`:36`). `docker-compose.yml`: named volume `batch_files`.

## 10. Integration Strategy

| Existing code | Change | Compatibility |
|---|---|---|
| `RequestQueue` | new method and counters; `_idle` includes background | `acquire` unchanged for callers |
| `_admit` | `background` kwarg; owner re-entry | default path byte-for-byte equivalent for interactive callers |
| unload drain, idle release | `occupied_count` | identical when no batch runs |
| `_score_prompts` (from Feature 25) | gains `pack_size`; synchronous callers pass 1 | bit-identical output (test) |
| `_use_cbm_for_request` | batch-row clause | no effect outside batches |
| `ProbeEventService` | origin plumbing | live events unchanged; default `origin='live'` |
| routes chat/completions/embeddings | `validate_<endpoint>`, `provenance.py` | headers unchanged (snapshot test before/after) |
| `lifespan` | reconcile → runner start → retention loop; runner stop on shutdown | after `disarm_probes_on_startup` (`main.py:379`) |
| Feature 29 health | `register_backlog_provider(runner.backlog_rows)` at runner start; `in_flight` adds `background_holding_count` | `null` before this feature, per 029 FR-29.7.3 (Stage 3, 2026-10-06, requested by 029) |
| `docs/mcp-contract.md` | new section for files and batches routes | additive minor version |
| `tests/unit/test_mcp_tool_paths_are_real.py` | the batch routers join the served-router set that miStudio's tool paths are checked against | needed by miStudio BR-013 |

## 11. Utilities and Helpers Design

- `served_batch_endpoints(app) -> frozenset[str]` in `services/batch/state.py`.
- `openai_list(data, first_id, last_id, has_more)` — shared by files and batches lists.
- `epoch(dt) -> int | None` — OpenAI timestamps are Unix seconds.
- `FileStore.read_line(file, offset, length) -> bytes` — `os.pread`, no full read.
- `chunk_rows(batch, kind) -> int` — the chunk-size rule (FTDD §9).
- `error_line(custom_id, line_no, status, code, message) -> dict` — one shape for validation, row and unrun errors.

## 12. Error Handling and Logging Strategy

| Category | Handling |
|---|---|
| Upload over cap, bad purpose, bad window, unserved endpoint | `400`, field or limit named |
| Lease conflict | `409 model_leased` with holder, reason, expiry (Feature 29 body) |
| Terminal-state cancel / delete referenced file | `409 batch_state_conflict` / `file_in_use` naming the batch |
| Invalid line | error-file line; batch continues |
| Row failure (`MiLLMError`) | error-file line with the synchronous status and body; batch continues |
| Unloading refusal inside a chunk | chunk not recorded; batch waits (`model_not_resident`) |
| Unexpected exception in a chunk | chunk not recorded; batch `failed` with `errors` naming the exception class; lease released |
| Input file hash changed | batch `failed`, `input_file_changed` |

Log events (structured, no row text, no lease ID): `batch_file_uploaded`, `batch_created`, `batch_validated` (valid, invalid, seconds), `batch_waiting` (reason), `batch_chunk_recorded` (rows, packed, seconds), `batch_lease` (acquired/renewed/released/lost, holder), `batch_terminal` (status, counts), `batch_files_pruned`.

## 13. Performance Implementation Hints

- Validation: one pass over the file; cache model resolution per name; insert rows in 1,000-row statements.
- Chunk loop: fetch the next chunk's rows while the current chunk's forward runs is **not** needed at 19 rows per second; keep it simple until acceptance 7 shows otherwise. **Benchmark the path changed** — measure with the runner, not a bare forward.
- Assemble files by streaming rows with `yield_per(1000)`.
- Socket emits throttled; health's backlog computed from counters, not a scan.

## 14. Code Quality and Standards

- Black (100), Ruff, MyPy strict, Google docstrings (PADR Appendix A).
- Comments state *why*, especially at the owner check, the interactive-priority wait, the `state='pending'` guard and the right-padding choice.
- No source-scrape guards: wiring tests assert calls, registries and payloads, not text.
- No `assert X or True`; no `getattr` default standing in for a real attribute.
- Every review round re-mutates the previous round's fixes.

## 15. Decisions from Clarifying Questions

Rounds waived; answers from sources.

| # | Question | Answer | Source |
|---|---|---|---|
| ID1 | Module layout | `millm/services/batch/` package | FTDD §6 |
| ID2 | Nested admission | Owner-task re-entry | FTDD TD2 |
| ID3 | Background waiting | Separate counters, interactive priority via `Condition` | FTDD TD3 |
| ID4 | Pad side | Right, gather with index tensor | FTDD TD5 |
| ID5 | Synchronous scoring after refactor | `pack_size=1`, bit-identical | FPRD FR-26.5.4 |
| ID6 | Lease ID storage | Memory only | 029 FR-29.1.6 |
| ID7 | Migration number | Next free at implementation | Siblings land first |
| ID8 | Media type | `application/jsonl` | FTDD TD6 |
| ID9 | Probe-event dedupe | Partial unique index, skip on conflict | FTDD TD12 |
| ID10 | Defaults for chunk and pack | 8 / 16 / 16,384, confirmed by acceptance 7 | FTDD §9 |

**Open items:** the two technical items in FTDD §14 (sensing events from batch rows; sibling contract names).
