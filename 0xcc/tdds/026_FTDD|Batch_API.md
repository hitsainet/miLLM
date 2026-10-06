# Technical Design: Batch API

## miLLM Feature 26

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** `026_FPRD|Batch_API.md` v1.1 · BRD-04 §5.5 (R-04.16 – R-04.23) · PPRD v1.5 Feature 26 · PADR v1.5 §10 "Batch runner inside the admission path vs a separate batch worker" and "Packed scoring by default vs one row at a time" · checkpoint decisions and Feature-PRD decisions (`~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`) · register T-63 – T-71 (`fprd-open-questions-2026-10-06.md`)
**Siblings:** 025 (validation; its FTDD exists and is followed here), 027 (probe scoring), 028 (steering value), 029 (lease, health), 030 (embedding pooling)
**Clients:** miStudio `034_FTDD` §6.5 (`get_jsonl`, `upload`); miDataworks `005_FTDD` §9 (batch transport under its shared lease)

Code references are to miLLM at `7aa659c`, verified 2026-10-06. Clarifying rounds were waived; every decision is in §14 with its source.

---

## 1. Executive Summary

Feature 26 turns a JSON Lines (JSONL) file of OpenAI-shaped requests into one durable job. The business goal is a 50,000-row labelling run that survives a rollout, reports progress and can be cancelled, without starving interactive chat (FPRD §1).

| Area | Decision | Why |
|---|---|---|
| Where it runs | One `BatchRunner` asyncio task in the API process | PADR: one resident model; a second path to the model is how batched chat skipped probes |
| Admission | `_admit(background=True)`: one slot per chunk, through a new `RequestQueue.acquire_background()` | R-04.19: single admission path, not counted as pending, interactive first |
| Nested slots | `_admit` becomes **re-entrant for the task that holds the slot** (a context variable naming the owner task) | Rows run through the unchanged synchronous service methods, each of which takes `_admit()`; without re-entry they deadlock at concurrency 1 |
| Persistence | PostgreSQL for batches, files and one row per input line; file bytes on the data volume | R-04.18; keeps nightly database dumps small |
| Resume | Startup reconciliation function + per-chunk atomic commits + unique `(batch_id, line_no)` | No recorded row runs twice (FR-26.3.4) |
| Packing | Right-padded packs, logits gathered at each row's last real token (`logits_to_keep` as an index tensor) | Padding-safe on causal attention, convolution and recurrent mixers alike; no position-id shift |
| Lease | Batch takes Feature 29's lease in-process, renews it, re-acquires after restart or waits | R-04.22; X-01; T-64 |
| Results | `GET /v1/files/{id}/content` with media type `application/jsonl` | miStudio's typed JSONL path needs one stated type (034 FR-22) |
| Retention | 30 days for every batch file; hourly prune + at startup | Checkpoint default; T-68 |

## 2. System Architecture

```
client ──multipart──► POST /v1/files ──► FileStore.write (byte cap, line count) ──► batch_files row
client ──JSON──────► POST /v1/batches ──► lease check (409?) ──► batches row (validating)
                                               │
                         BatchValidator (thread) ── per line: parse → shape → endpoint schema →
                         Feature 25 unused-field + refusal list → model resolve
                         → batch_rows (pending | invalid) → status in_progress (queued) | failed
                                               │
BatchRunner (one asyncio task, FIFO by created_at)
   loop: pick next runnable batch → ensure lease (own or caller's) → next chunk of pending rows
         async with inference._admit(background=True):      # one slot, not "pending"
             set BATCH_ROW context var (batch_id, line)
             scoring/embedding rows → _score_prompts / _embed_inputs (packed or single)
             generation rows       → create_chat_completion / create_text_completion (re-enter _admit)
             probe-score rows      → Feature 27 scoring service (re-enters _admit per input)
         commit chunk results (one transaction) → emit batch:progress → check cancel/expiry
   end:  finalizing → FileStore.assemble(output, error) → completed | cancelled | expired → release lease
Startup (lifespan): reconcile_batches_on_startup() → BatchRunner.start() → retention loop
```

**Integration points:**
- `InferenceService._admit` (`millm/services/inference_service.py:649`) and `RequestQueue.acquire` (`millm/services/request_queue.py:77`).
- `ModelService.unload_model` drain (`millm/services/model_service.py:1258`) — must see background slot holders.
- `ModelService.find_model_by_name` (`millm/services/model_service.py:1523`) — line model resolution.
- `_score_text_completion` (`inference_service.py:4930`) and `_next_token_logits` (`inference_service.py:5043`) — packed scoring.
- `create_embeddings` (`inference_service.py:5071`) — packed embeddings after Feature 30.
- `ProbeEventService.record` (`millm/services/probe_event_service.py:48`) and `ProbeEventRepository.prune` (`millm/db/repositories/probe_repository.py:206`) — batch-marked events.
- `ProgressEmitter` (`millm/sockets/progress.py:124`, instance at `:670`) — `batch:progress`.
- `lifespan` (`millm/main.py:277`) — reconciliation, runner, retention.
- Feature 29 lease service; Feature 25 validation helpers; Feature 28 steering value; Feature 27 scoring service.

## 3. Technical Stack

- FastAPI routes under the existing `/v1` router (`millm/api/routes/openai/__init__.py`), multipart through `python-multipart` (`pyproject.toml:24`).
- SQLAlchemy 2 async models and one Alembic migration (`millm/db/migrations/versions/`).
- asyncio for the runner; `asyncio.to_thread` for validation, file assembly and forward passes, as today.
- torch / transformers `>=5.15.1,<6` (`pyproject.toml:52`). Packed scoring relies on `logits_to_keep` accepting an index tensor: `slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep` (installed `transformers/models/llama/modeling_llama.py:479`, same at `lfm2/modeling_lfm2.py:595`). Models whose forward rejects a tensor fall back to full logits for the pack, bounded by the pack token budget.
- No new dependency.

## 4. Data Design

Three new tables and three new columns, in one migration at the next free number (the last today is `018_add_probe_threshold_revision.py`; Features 25, 27 and 29 may take numbers first).

**`batch_files`**
| Column | Type | Notes |
|---|---|---|
| `id` | String(32) PK | `file-` + 24 hex |
| `purpose` | String(16) | `batch`, `batch_output` |
| `filename` | String(255) | as uploaded, or `batch_<id>_output.jsonl` |
| `bytes`, `line_count` | BigInteger, Integer | measured at write |
| `storage_path` | String(512) | relative to `BATCH_FILES_DIR`; never returned by the API |
| `sha256` | String(64) | of the stored bytes; checked before a run reads them |
| `status` | String(16) | `processed`, `error`, `deleted`, `expired` (OpenAI's three plus two for retention) |
| `created_at`, `expires_at`, `deleted_at` | DateTime(tz) | |

**`batches`**
| Column | Type | Notes |
|---|---|---|
| `id` | String(32) PK | `batch_` + 24 hex |
| `endpoint`, `completion_window` | String | |
| `model_id`, `model_name` | Integer FK `models.id` (SET NULL), String | set at end of validation |
| `status` | String(16) | the eight OpenAI values; CHECK constraint |
| `input_file_id`, `output_file_id`, `error_file_id` | FK `batch_files.id` | |
| `pack` | Boolean | resolved at create from the request or `BATCH_PACK_DEFAULT` |
| `metadata` | JSON | OpenAI's |
| `errors` | JSON | OpenAI's `{object: "list", data: [...]}`, first `BATCH_ERRORS_SHOWN` |
| `request_total`, `request_completed`, `request_failed` | Integer | |
| `output_expires_after_s` | Integer | default 2,592,000 |
| `waiting_reason` | String(48) nullable | `queued`, `model_not_resident`, `lease_unavailable`, null while running |
| `lease_mode` | String(8) nullable | `own` or `caller`; the lease ID itself is never persisted (029 FR-29.1.6) |
| every OpenAI timestamp | DateTime(tz) | `created_at` … `cancelled_at`, `expires_at` |

**`batch_rows`** — one per input line.
| Column | Type | Notes |
|---|---|---|
| `batch_id`, `line_no` | composite PK | the uniqueness that enforces "never twice" |
| `custom_id` | String(512) nullable | null when unreadable |
| `kind` | String(12) | `scoring`, `generation`, `embedding`, `probe` |
| `byte_offset`, `byte_length` | BigInteger, Integer | where the line sits in the input file |
| `state` | String(12) | `pending`, `invalid`, `done`, `failed`, `cancelled`, `expired` |
| `result` | JSON nullable | the finished output or error line |
| `packed` | Boolean nullable | |
| `chunk_seq` | Integer nullable | which chunk recorded it |

Index `(batch_id, state, line_no)` serves "first pending rows in order". `batch_rows` for a batch are deleted in the same transaction that marks its files assembled and the batch terminal (§7).

**`probe_events`** (Feature 24 table) gains `origin` String(8) NOT NULL `server_default='live'`, `batch_id` String(32) nullable, `batch_line` Integer nullable, and a partial unique index on `(probe_id, batch_id, batch_line, window)` where `batch_id IS NOT NULL`. A row re-run after a crash then cannot record a second event (FR-26.4.8).

**Validation strategy:** status transitions are enforced by one function (`batch_state.transition`) holding the FPRD FR-26.3.5 table; the CHECK constraint guards the value set. **Migration:** additive only; existing `probe_events` rows become `origin='live'` by server default. Downgrade drops the new tables and columns.

## 5. API Design

All routes live under `/v1`, return OpenAI shapes, and use the existing OpenAI error envelope (`millm/api/routes/openai/errors.py`). No authentication (BRD-04 §3).

| Route | Behaviour | FR |
|---|---|---|
| `POST /v1/files` | multipart `file`, `purpose`; cap checked from `Content-Length`, then while copying; lines counted | 26.1.1, 26.8 |
| `GET /v1/files` | newest first, `purpose`, `limit`, `after`, `order` | 26.6.9 |
| `GET /v1/files/{id}` | file object | 26.6.5 |
| `GET /v1/files/{id}/content` | streamed bytes, `Content-Type: application/jsonl` | 26.6.5 |
| `DELETE /v1/files/{id}` | `409` while a non-terminal batch references it | 26.6.10 |
| `POST /v1/batches` | create; synchronous checks only (file exists and is `batch`, endpoint served, window, lease) | 26.1.2 – 26.1.6, 26.7.3 |
| `GET /v1/batches`, `GET /v1/batches/{id}` | OpenAI list and object | 26.6.1, 26.6.3 |
| `POST /v1/batches/{id}/cancel` | `cancelling`; `409` on a terminal batch | 26.6.4 |
| `POST /v1/batches/{id}/lease` | **miLLM extension.** Header `X-miLLM-Lease` required. Hands the caller's live lease to a waiting batch | 26.7.7 |

**Extension fields.** The batch object carries one extension object `millm: {pack, waiting_reason, lease_mode, completion_window_extension}`. Each output line's `response` carries `millm: {packed, headers: {...}}` where `headers` holds every `X-miLLM-*` value the synchronous route would have set (FR-26.10.1). Extensions sit in their own object so an OpenAI client ignores them.

**Endpoint set.** `served_batch_endpoints(app)` reads `app.openapi()["paths"]`, which is how this suite lists served routes (memory: `app.routes` is not a route list), and intersects it with the four supported paths. `/api/probes/score` is accepted the moment Feature 27's route is registered (FR-26.1.4).

**Error handling.** Synchronous refusals: `400` (`invalid_request_error`) naming the field, limit or endpoint; `404` unknown id; `409` lease (`model_leased`, Feature 29's code) or state conflict (`batch_state_conflict`); `413` is not used — R-04.23 asks for a refusal at upload, and `400` with the limit named is what the FPRD fixed. Per-row errors never fail the batch; they become error-file lines carrying the status code and OpenAI error body the synchronous route would have returned.

**Validation helpers from Feature 25.** 025's FTDD (§5, `millm/api/request_policy.py`) defines `request_policy.evaluate(request, endpoint, engine, strict=...)` over the output-changing table and unused fields, plus the JSON-schema subset check (`millm/api/json_schema_subset.py`). The validator calls `evaluate(..., strict=True)` on every line (FR-26.2.3) and the subset check, in 025's evaluation order (its §5.2, steps 3–6), so a line is refused with the same code (`FIELD_NOT_HONOURED`, `UNUSED_FIELDS_REFUSED`) the synchronous route returns.
Route pre-service checks that today live inline in the route (stream refused on completions, embedding-only model, GGUF scoring) move into `validate_<endpoint>(request, model_row)` functions in the route modules, called by the route and the validator. One copy, two callers.

**Performance principles.** Validation never takes a slot (FR-26.2.9). Content is streamed, never read whole. Request counts come from the `batches` row, updated in the chunk transaction.

## 6. Component Architecture

```
millm/services/batch/
  __init__.py
  state.py        transition table, status/kind enums, OpenAI object serialisers
  files.py        FileStore: write (cap, count, sha256), read_line(offset,len), assemble(), delete()
  validator.py    BatchValidator.validate(batch_id) — thread-safe, pure over (bytes, model rows)
  runner.py       BatchRunner: start/stop, FIFO pick, lease handling, chunk loop, cancel/expiry
  executors.py    row executors per kind (scoring, generation, embedding, probe)
  retention.py    prune_expired_files(now) + the hourly loop
  reconcile.py    reconcile_batches_on_startup(session_factory)
millm/db/models/batch.py, millm/db/repositories/batch_repository.py
millm/api/routes/openai/files.py, batches.py; millm/api/schemas/batch.py
millm/api/provenance.py   response_provenance(endpoint, request, response, service) -> dict[str,str]
```

- **Executors share the synchronous code.** Generation rows call `create_chat_completion` / `create_text_completion` unchanged. Scoring rows call Feature 25's `_score_prompts`, extended here with `pack_size`; the synchronous scorers call it with `pack_size=1`. Embedding rows call the embedding body that Feature 30 makes mask-aware. Probe rows call Feature 27's service function. No executor talks to the model directly (FR-26.4.5).
- **Provenance in one place.** Header construction moves out of the route bodies into `millm/api/provenance.py`. Routes and executors both call it. Feature 28's steering-value function (FR-28.3.10) and Feature 25's seed and constrained values feed it.
- **Separation:** routes do HTTP and synchronous checks; `services/batch` owns state and execution; `InferenceService` owns the model. The runner holds no model reference.

## 7. State Management

**Batch lifecycle.** `validating → in_progress → finalizing → completed`, with `failed`, `cancelling → cancelled` and `expired` (FPRD FR-26.3.5). `waiting_reason` is set while `in_progress` and not running.

**Runner state (in memory only, rebuilt at startup):** the current batch id, its lease handle (`own` lease object, or the caller's lease ID held in memory), cancel events per batch, and the last emit time per batch.

**Chunk transaction.** One database transaction per chunk: update each row's `state`, `result`, `packed`, `chunk_seq`; bump the batch counters; commit. Nothing is visible before commit, so a crash mid-chunk leaves those rows `pending` and they run again — none was recorded (FR-26.3.2, 26.3.4).

**Finalisation.** Assemble output and error files from `batch_rows` in `line_no` order into `<file>.partial`, fsync, rename, insert `batch_files` rows, set the batch terminal and delete its `batch_rows`, all in one transaction after the rename. A crash before commit leaves `finalizing`; reconciliation re-assembles (the rename is idempotent; a stray `.partial` is removed).

**Unrun rows at the end.** Cancelled and expired batches write each unrun valid row to the error file with code `batch_cancelled` or `batch_expired`, as OpenAI does, so every `custom_id` appears exactly once across the two files.

**Lease handling (FR-26.7, X-01, T-64):**
1. Before each chunk, `ensure_lease()`: an `own` lease is renewed when a third of its TTL has passed; a `caller` lease is renewed the same way using the in-memory ID.
2. No live lease → try `acquire(model_id, holder="millm-batch:<id>", ttl=BATCH_LEASE_TTL_S, reason="batch <id>")`. Fails because the model is not resident → `waiting_reason=model_not_resident`; because another lease is live → `lease_unavailable`. No row runs. Retry every `BATCH_WAIT_POLL_S`.
3. `POST /v1/batches/{id}/lease` with a live `X-miLLM-Lease` whose model is the batch's model switches the batch to `caller` mode.
4. On any terminal status: release an `own` lease (in `finally`); never release a `caller` lease (T-64).
A restart forgets every lease, as Feature 29 does (029 FR-29.1.9), so a resumed batch always enters step 2.

**Side effects.** Socket emission and probe-event writes are fire-and-forget and never block a chunk commit. File deletion happens after the database commit that marks a file deleted; a failed unlink is logged and retried by the retention loop (the row says `deleted`, the bytes are orphans the loop sweeps by scanning `BATCH_FILES_DIR` for paths no row owns).

**Admission state (the two constraints):**

*Constraint 1 — nested slots would deadlock.* A chunk holds the only slot. A generation row calls `create_chat_completion`, which enters `async with self._admit()` and waits for a slot its own task holds. **Resolution:** `_admit` keeps a context variable `_SLOT_OWNER` set to the task that holds the slot. On entry, if `_SLOT_OWNER.get() is asyncio.current_task()`, `_admit` re-checks the unloading refusal and yields without acquiring. A child task created inside the slot inherits the variable but is a *different* task, so it acquires normally and cannot run concurrently by mistake. Rejected alternative: extracting a slot-free body from every synchronous method. It touches every chat, completion, scoring, embedding and probe path, and each extraction is a chance to create the second path FR-26.4.5 forbids.

*Constraint 2 — waiters count as pending.* `RequestQueue.acquire` increments `_pending` before waiting and refuses past `max_pending` (`request_queue.py:107-117`). **Resolution:** `RequestQueue.acquire_background()`:
- never checks `max_pending` and never touches `_pending`;
- counts itself in `_background_waiting` / `_background_holding`;
- before taking the semaphore, waits on an `asyncio.Condition` until no interactive request is waiting (`_pending - _holding == 0`). Interactive requests therefore always go first at a chunk boundary, whatever the interpreter's semaphore fairness. (Python 3.12's `Semaphore.locked()` already queues a newcomer behind waiters — measured here: `['chunk0', 'chat', 'chunk1', 'chunk2']` — but the image runs `python:3.11-slim`, `Dockerfile:7`, so the design does not rely on it.)
- `_idle` is set only when interactive and background counts are both zero, so `wait_idle` drains a running chunk.
- New property `occupied_count` = interactive pending + background waiting + background holding. The unload drain (`model_service.py:1258`) and the idle-cache release checks (`inference_service.py:725`, `:750`) switch to it. `pending_count` keeps its meaning, so the `QUEUE_FULL` check at `chat.py:244` and the health field stay interactive-only.

`_admit(background=True)` is the only caller of `acquire_background`; the guard `test_every_request_queue_slot_is_taken_through_admission` (`tests/unit/services/test_unload_admission.py:451`) extends to the new method.

## 8. Security Considerations

- **No authentication**, as every miLLM route (BRD-04 §3, decision 7). The lease ID is the only proof of holding (029 FR-29.1.6): it is held in memory, never persisted, never logged, never echoed.
- **Path safety.** Stored paths are generated (`<BATCH_FILES_DIR>/<yyyy-mm>/<file-id>.jsonl`). The client filename is metadata only and never touches the filesystem.
- **Input hostility.** Byte and row caps before parse; each line parsed independently with a per-line size cap (`BATCH_MAX_LINE_BYTES`); JSON parse depth bounded by the standard library's recursion limit, caught per line.
- **Privacy.** No log line carries row text. Batch probe events are not emitted on the live feed (FR-26.4.8); `context_text` follows Feature 24's rule (never on the socket).
- **Integrity.** The input file's sha256 is checked before validation and before resume; a mismatch fails the batch (`input_file_changed`), since a different file would break "first row without a recorded result".

## 9. Performance & Scalability

- **Throughput target:** unpacked ≥ 19 rows per second on JEV-9B-decision (BRD acceptance 7). In-process single forwards with no HTTP round trip and one slot per `BATCH_CHUNK_ROWS` rows should exceed the client loop's 19–21.
- **Chunk sizing:** scoring/embedding chunk = one pack when packed, or `BATCH_CHUNK_ROWS` single forwards (default 8); generation chunk = 1 row; probe chunk = 1 line. Interactive wait ≤ one chunk (acceptance 9). Defaults are confirmed or changed by the acceptance measurement.
- **Pack bounds:** `BATCH_PACK_MAX_ROWS` (default 16) and `BATCH_PACK_MAX_TOKENS` (default 16,384 padded tokens). An out-of-memory error halves the pack and retries the same unrecorded rows, down to 1; a single-row out-of-memory error becomes that row's error line, as `_chunk_batch_for_memory` already prefers a slow answer to a 500 (`inference_service.py:3349`).
- **Database:** bulk insert of `batch_rows` at validation (50,000 rows in batches of 1,000); per-chunk updates by primary key; counters on the batch row, no `COUNT(*)` on read.
- **Disk:** bounded by 200 MB × live batches plus outputs, pruned at 30 days.
- **Socket:** `batch:progress` at most once per `BATCH_PROGRESS_MIN_INTERVAL_S` (1 s) per batch, plus every transition.

## 10. Testing Strategy

- **Unit:** state transitions; validator per failure kind (parse, shape, schema, strict, refusal, model mismatch, duplicate `custom_id`, `stream: true`); limits; serialisers against OpenAI field lists; retention with an injected clock; file assembly order and idempotence.
- **Admission (load-bearing):** background acquisition not counted as pending and never `QUEUE_FULL`; an interactive waiter beats the next chunk; `wait_idle` waits for a chunk; re-entry only for the owner task (a child task must queue); the admission guard extended.
- **Resume:** crash injected after N chunks and mid-chunk; restart; exactly-once `custom_id`s; reconciliation of each non-terminal state; the startup call removed → red.
- **Packing:** `pack: false` bit-identical to the synchronous endpoint; packed rows of mixed lengths gather the right position (a fixture whose rows differ in length by more than one, so a wrong index cannot pass by coincidence); probe rows never packed; embedding rows unpacked until Feature 30.
- **Lease:** own lease taken, renewed, released on every terminal path including an exception; caller lease never released; restart → re-acquire or wait; `/lease` hand-over.
- **Reachability:** every route present in `app.openapi()["paths"]` and called through `TestClient` with payload asserted; runner start and reconciliation asserted by call and argument; `emit_batch_progress` asserted with payload and count.
- **Mutation controls** (FTID §8): admission exemption, owner check, chunk atomicity, cancel check, lease release, retention reference guard, probe-event origin.
- **Fixtures:** a tiny real Llama for packing (the `test_residual_hook_is_resid_post` style), never a stub that agrees with the code by construction; SQLite plus the PostgreSQL schema-parity test (`millm/db/schema_parity.py`).
- **Hardware:** BRD acceptance 7, 8, 9 on the node.

## 11. Deployment & DevOps

- **Configuration** (`millm/core/config.py`, `.env.example`): `BATCH_FILES_DIR` (`/app/batch_files`; k8s `/data/batch_files`), `BATCH_MAX_ROWS` 50,000, `BATCH_MAX_FILE_BYTES` 209,715,200, `BATCH_MAX_LINE_BYTES` 1,048,576, `BATCH_PACK_DEFAULT` true (T-63 may flip it), `BATCH_CHUNK_ROWS` 8, `BATCH_PACK_MAX_ROWS` 16, `BATCH_PACK_MAX_TOKENS` 16,384, `BATCH_MAX_COMPLETION_WINDOW_HOURS` 168, `BATCH_FILE_RETENTION_DAYS` 30, `BATCH_LEASE_TTL_S` 900, `BATCH_WAIT_POLL_S` 10, `BATCH_PROGRESS_MIN_INTERVAL_S` 1, `BATCH_ERRORS_SHOWN` 100, `PROBE_MAX_BATCH_EVENTS_PER_PROBE` 50,000.
- **Kubernetes:** `k8s/base/backend.yaml` gains `BATCH_FILES_DIR=/data/batch_files`, and the init container's `mkdir -p` list gains it. **docker-compose:** a named volume `batch_files:/app/batch_files`.
- **Monitoring:** structured log events `batch_created`, `batch_validated`, `batch_chunk_recorded` (rows, seconds, packed), `batch_waiting` (reason), `batch_terminal`, `batch_files_pruned`. Feature 29's `batch_backlog_rows` reads `BatchRunner.backlog_rows()`.
- **Rollout:** the migration is additive; batches started on the new image survive later rollouts by design. **A rollout still interrupts the chunk in flight** — expected, and the reason for row-level resume.
- **Rollback:** an older image ignores the new tables. Batches left `in_progress` resume when the new image returns. Downgrading the migration deletes batch state; take the nightly-style manual dump first.

## 12. Risk Assessment

| Risk | Mitigation |
|---|---|
| Re-entrant `_admit` lets two coroutines share a slot | Owner is a task, not a context; a child task test; mutation control on the owner check |
| Interactive starvation | Interactive-first condition; acceptance 9 |
| Batch starvation under constant chat | Accepted: interactive is bounded by `MAX_PENDING_REQUESTS`; `batch_waiting` would show it |
| Packed scores differ from single (bfloat16) | Right-padding gather; measured difference published; T-63 flip |
| Padding unsafe on hybrid models | Right padding is causal-safe for attention, convolution and recurrence; a test on a tiny real hybrid if one is constructible, otherwise recorded |
| A route header lost in batch lines | `provenance.py` shared by route and executor; a test compares a batch line with the synchronous response's headers |
| Lease lost mid-run (caller releases) | Checked before every chunk; batch waits, never runs unleased |
| Sibling helpers not yet designed (027, 028, 029; 025 is designed) | Contracts stated in §5 and §7; FTASKS phase gates on each |
| Disk fill | Caps, retention, delete route |

**Complexity:** high. **Alternatives considered:** a separate worker (rejected by PADR); extracting slot-free bodies (rejected, §7); left padding with position ids (rejected: unsafe for convolution mixers such as LFM2 and needs per-model position handling); storing file bytes in PostgreSQL (rejected: 200 MB files in nightly dumps).

## 13. Development Phases

| Phase | Content | Depends on |
|---|---|---|
| 1 | Data layer: models, migration, repository, `probe_events` columns | — |
| 2 | Queue and admission: `acquire_background`, re-entrant `_admit`, `occupied_count` call sites | — |
| 3 | Files: store, routes, limits, list/delete, retention | 1 |
| 4 | Validation: helpers from 025, route `validate_*` extraction, validator | 1, Feature 25 |
| 5 | Runner and executors (single-row), lease handling, cancel, expiry, reconciliation | 2, 4, Feature 29 |
| 6 | Packed scoring and embeddings, provenance module | 5, Features 28, 30 |
| 7 | Probe rows and batch-marked probe events | 5, Feature 27 |
| 8 | Contract and docs; hardware acceptance; T-63 setting | all |

## 14. Decisions from Clarifying Questions

Rounds waived; answers from sources.

| # | Question | Answer | Source |
|---|---|---|---|
| TD1 | Separate worker or in-process runner? | In-process `BatchRunner` | PADR "Batch runner inside the admission path" |
| TD2 | How do rows avoid deadlock on nested `_admit`? | Re-entrant for the owner task | §7; FPRD FR-26.4.5 (no second path) |
| TD3 | How do batches avoid the pending count? | `acquire_background` with separate counters and interactive priority | R-04.19; FR-26.4.3, FR-26.4.4 |
| TD4 | Where do file bytes live? | Data volume; metadata and rows in PostgreSQL | R-04.18; backup size (`k8s/base/db-backup.yaml`) |
| TD5 | Pad side for packing | Right padding, gather at last real token | Causal safety; `logits_to_keep` tensor support verified |
| TD6 | Results media type | `application/jsonl` | miStudio 034 FR-22, `034_FTDD` §6.5 |
| TD7 | Lease across restart | Re-acquire or wait; never run unleased | X-01, T-66 |
| TD8 | Caller's lease | Run under it, renew, never release | T-64, X-08 |
| TD9 | Re-attaching a caller lease after restart | `POST /v1/batches/{id}/lease` extension | FPRD FR-26.7.7 |
| TD10 | `completion_window` range | 1–168 hours, configurable | T-65 |
| TD11 | Packing default | Config `BATCH_PACK_DEFAULT`, true until acceptance 7 says otherwise | T-63 |
| TD12 | Probe events from batch rows | `origin='batch'`, own cap, unique per (probe, batch, line, window), no live emit | T-70 |
| TD13 | Retention | 30 days all batch files; hourly + startup prune | Checkpoint default; T-68 |
| TD14 | Unrun rows at cancel/expiry | Error-file lines `batch_cancelled` / `batch_expired` | OpenAI behaviour; FR-26.3.4 exactly-once |
| TD15 | Endpoint set | From `app.openapi()["paths"]` | FR-26.1.4; "registries, not hand-kept lists" |

**Open items (technical, not product):**
1. **Sensing and circuit-edge sensing events from batch generation rows.** T-70 covers probe events only. As designed, batch generation rows flow through the synchronous path and record sensing events as live traffic. Phase 7 measures whether a 50,000-row generation batch evicts live sensing history (`SENSING_MAX_AGE_DAYS` and caps, `millm/core/config.py:156`) and, if so, applies the same `origin` marking. It needs no product decision unless the marking changes what an operator sees.
2. **Sibling contracts.** Feature 27's scoring function, Feature 28's steering value and Feature 29's lease service are specified here by contract only; their FTDDs fix names. FTASKS gates each phase on them.
