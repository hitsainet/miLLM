# Feature 26 (Batch API) — implementation controls and review notes

**Branch:** `feat/026-batch-api` (worktree `~/app/miLLM-026`, cut from `main` at `6cce090`).
**Dates:** 2026-10-06 → 2026-10-07. **Author:** implementation agent (Claude Opus 5.5).
**Method:** every control backs the file up, changes ONE line, runs the affected suite with `-x`,
restores from the backup, then verifies the restore by sha256 AND re-greps that the mutated text is
gone (`scratchpad/impl-026/mut/mutate.py`; results in `results.jsonl`). A control that "survives"
was first checked to have LANDED (`landed: true` for all 32). A hang is recorded as a hang, never as
a red: the runner times out at 600 s.

## 1. Controls

FTID §8 M1–M14 plus the coordinator's extras (re-entrancy, `max_pending`, interactive priority,
lease renewal, restart re-acquire, packing switch, the T-67 end state) and further wiring lines.

| # | File | Mutation | Suite | First run | Final | Restore |
|---|---|---|---|---|---|---|
| M1 | `millm/services/request_queue.py` | acquire_background increments _pending | test_request_queue_background.py | red | RED (1 failed in 12.32s) | sha ✓ / re-grep ✓ |
| M2 | `millm/services/request_queue.py` | interactive-priority wait dropped (only the free-slot half kept) | test_request_queue_background.py | **survived** | RED (1 failed, 7 passed in 2.39s) | sha ✓ / re-grep ✓ |
| M3 | `millm/services/inference_service.py` | owner check compares context only (drops `is current_task()`) | test_admit_reentry.py | red | RED (1 failed, 2 passed in 2.74s) | sha ✓ / re-grep ✓ |
| M4 | `millm/db/repositories/batch_repository.py` | record_chunk commits per row | test_batch_models.py | red | RED (1 failed, 8 passed in 2.65s) | sha ✓ / re-grep ✓ |
| M5 | `millm/db/repositories/batch_repository.py` | state='pending' guard dropped | test_batch_models.py, test_runner.py | red | RED (1 failed, 7 passed in 2.56s) | sha ✓ / re-grep ✓ |
| M6 | `millm/services/batch/executors.py` | cancel check between rows removed | test_runner.py | red | RED (1 failed, 8 passed in 31.12s) | sha ✓ / re-grep ✓ |
| M7 | `millm/services/batch/runner.py` | own-lease release removed from finally | test_runner.py | red | RED (1 failed, 11 passed in 46.01s) | sha ✓ / re-grep ✓ |
| M8 | `millm/services/batch/runner.py` | caller lease released at end | test_runner.py | red | RED (1 failed, 9 passed in 60.91s (0:01:00)) | sha ✓ / re-grep ✓ |
| M9 | `millm/services/batch/retention.py` | retention reference guard removed | test_retention.py | red | RED (1 failed, 2 passed in 9.63s) | sha ✓ / re-grep ✓ |
| M10 | `millm/services/probe_event_service.py` | origin='batch' not set on batch probe events | test_probe_events_batch_origin.py | red | RED (1 failed in 5.57s) | sha ✓ / re-grep ✓ |
| M11 | `millm/main.py` | start_batch_api (reconcile+runner) call removed from lifespan | test_batch_lifespan_wiring.py | red | RED (1 failed in 4.95s) | sha ✓ / re-grep ✓ |
| M11b | `millm/services/batch/reconcile.py` | reconcile call removed from start_batch_api | test_batch_lifespan_wiring.py | red | RED (1 failed, 1 passed in 4.99s) | sha ✓ / re-grep ✓ |
| M11c | `millm/services/batch/runner.py` | batch:progress emit call removed | test_runner.py | red | RED (1 failed, 21 passed in 77.23s (0:01:17)) | sha ✓ / re-grep ✓ |
| M12 | `millm/api/routes/openai/__init__.py` | files_router include removed | test_batch_file_routes.py | red | RED (1 failed in 3.05s) | sha ✓ / re-grep ✓ |
| M12b | `millm/api/routes/openai/__init__.py` | batches_router include removed | test_batch_contract_rows_are_served.py | red | RED (1 failed in 3.00s) | sha ✓ / re-grep ✓ |
| M13 | `millm/services/inference_service.py` | packed gather uses -1 instead of last[i] | test_packed_scoring.py | red | RED (1 failed, 1 passed in 5.23s) | sha ✓ / re-grep ✓ |
| M14 | `millm/services/model_service.py` | occupied_count reverted to pending_count in unload drain | test_unload_admission.py | red | RED (1 failed, 12 passed in 16.20s) | sha ✓ / re-grep ✓ |
| X1 | `millm/services/request_queue.py` | acquire_background checks max_pending (QUEUE_FULL) | test_request_queue_background.py | **survived** | RED (1 failed, 10 passed in 2.36s) | sha ✓ / re-grep ✓ |
| X2 | `millm/services/batch/runner.py` | lease renewal call removed | test_runner.py | red | RED (1 failed, 15 passed in 56.32s) | sha ✓ / re-grep ✓ |
| X3 | `millm/services/batch/runner.py` | restart re-acquire skipped: runs without a lease | test_runner.py | red | RED (1 failed, 11 passed in 43.04s) | sha ✓ / re-grep ✓ |
| X4 | `millm/api/routes/openai/batches.py` | pack default ignores BATCH_PACK_DEFAULT | test_validation.py | red | RED (1 failed, 24 passed in 84.57s (0:01:24)) | sha ✓ / re-grep ✓ |
| X5 | `millm/services/batch/executors.py` | packing switch ignores pack=false | test_packed_scoring.py | red | RED (1 failed, 6 passed in 11.96s) | sha ✓ / re-grep ✓ |
| X6 | `millm/services/batch/runner.py` | T-67: zero-valid file does not end failed | test_validation.py | red | RED (1 failed, 5 passed in 23.60s) | sha ✓ / re-grep ✓ |
| X7 | `millm/services/inference_service.py` | batch-row clause in _use_cbm_for_request removed | test_runner.py | red | RED (1 failed, 24 passed in 69.38s (0:01:09)) | sha ✓ / re-grep ✓ |
| X8 | `millm/services/inference_service.py` | _probe_record does not pass BATCH_ROW | test_probe_events_batch_origin.py | red | RED (1 failed, 4 passed in 10.40s) | sha ✓ / re-grep ✓ |
| X9 | `millm/services/batch/runner.py` | backlog provider not registered at start | test_runner.py | red | RED (1 failed, 23 passed in 71.47s (0:01:11)) | sha ✓ / re-grep ✓ |
| X10 | `millm/db/repositories/sensing_repository.py` | sensing cap ignores origin | test_sensing_batch_origin.py | red | RED (1 failed in 4.08s) | sha ✓ / re-grep ✓ |
| X11 | `millm/services/batch/validator.py` | strict mode not applied to lines | test_validation.py | red | RED (1 failed, 3 passed in 12.97s) | sha ✓ / re-grep ✓ |
| X12 | `millm/core/config.py` | BATCH_PACK_DEFAULT fails to False | test_batch_lifespan_wiring.py | red | RED (1 failed, 7 passed, 2 warnings in 4.56s) | sha ✓ / re-grep ✓ |
| X13 | `millm/services/batch/runner.py` | unloading refusal fails the batch instead of waiting | test_runner.py | red | RED (1 failed, 19 passed in 59.01s) | sha ✓ / re-grep ✓ |
| X14 | `millm/services/batch/files.py` | byte cap not enforced while copying | test_batch_file_routes.py | red | RED (1 failed, 4 passed in 11.42s) | sha ✓ / re-grep ✓ |
| X15 | `millm/api/routes/openai/batches.py` | create ignores a foreign lease | test_validation.py | red | RED (1 failed, 21 passed in 67.67s (0:01:07)) | sha ✓ / re-grep ✓ |

**32 controls; 3 did not turn red the first time** — every one a TEST gap, every one closed with a
test and re-run as a negative control (red):

1. **M1 HUNG instead of failing** (first run, killed by hand). With `_pending` incremented by the
   chunk, the chunk's own priority predicate never becomes true and every queue test waited
   forever — the suite never went red, it never finished. Fix: every test in
   `test_request_queue_background.py` and `test_admit_reentry.py` runs under a 10 s bound
   (`@bounded`), so a deadlocked slot FAILS. Re-run: red in 12 s.
2. **M2 SURVIVED** — removing the interactive-priority half of the predicate left all 9 queue tests
   green. The fixture meant to reproduce Python 3.11's unfair `Semaphore` subclassed 3.12's and
   overrode only `locked()`; 3.12's `release()` hands the permit to the woken waiter, so the order
   was right with no priority rule at all. Fix: a full 3.11-semantics semaphore in the test, and a
   test where the batch releases and re-requests WITHOUT yielding while a chat was just woken
   (`test_a_chunk_re_requested_at_once_still_lets_a_woken_chat_go_first`). Re-run: red.
3. **X1 SURVIVED** — making `acquire_background` raise `QueueFullError` at `max_pending` left the
   suite green: no test requested a chunk while the INTERACTIVE queue was full. Fix:
   `test_a_chunk_requested_while_the_interactive_queue_is_full_is_not_refused`. Re-run: red.

## 2. Defects found during implementation (before any control)

- **`logger.info("batch_lease", event=...)`** collides with structlog's positional `event` and
  raised `TypeError` on every lease acquisition: the batch would never have run a row. Caught by
  the first end-to-end validation test.
- **The FTID's `acquire_background` sketch leaks a count on cancellation**: `_background_waiting`
  is incremented outside the `try`, so a waiter cancelled during `wait_for` leaves the count up
  forever and `_idle` never sets again (the unload drain would always time out). Fixed and tested
  (`test_a_cancelled_background_waiter_leaves_no_count_behind`).
- **The FTID's predicate (`_pending - _holding == 0`) alone lets the chunk queue on the semaphore**,
  where a chat request arriving next lands behind it. The predicate also requires a free slot.
- **The SAE detach drain read `pending_count` only** (latent, touched by this feature): it would
  remove hooks under a running batch generation row. Now `pending_count + background_holding_count`.
  Tracked debt: a CONTINUOUSLY running batch still lets detach proceed between chunks — the same
  pre-existing race an arriving interactive request has; the durable fix is for detach to take a
  slot.
- **An hourly orphan sweep with no age floor** would delete an upload whose bytes were renamed into
  place before its row committed. The sweep spares files younger than 10 minutes.
- **The lifespan test would have run the startup resets against whatever `DATABASE_URL` names on the
  machine** (here: a sibling project's PostgreSQL on `localhost:5432`). The test points
  `async_session_factory` at its own SQLite file.

## 3. Code wins over the documents (discrepancies recorded)

| Document says | Code | Decision |
|---|---|---|
| FR-26.2.7: every line names `body.model` | 027's `ProbeScoreRequest` has no `model` and is `extra="forbid"` | A probe batch's model is the resident model at validation |
| FTDD §4 row states include `cancelled`, `expired` | Unrun rows are written to the error file at assembly; rows are deleted in that transaction | Stored states are `pending, invalid, done, failed` |
| 029 FTDD: `register_backlog_provider` on `ModelService` | It is `millm.core.backpressure.register_backlog_provider` | Used as it is |
| FTID: `_score_prompts` gains `pack_size` with per-spec options | The scorer takes one option set for all texts | `pack_size` delegates to a per-spec `_score_specs_packed`; response builders extracted and shared |
| FTID §5: upload checks `Content-Length` before reading | FastAPI parses declared `UploadFile` params before the handler | The route parses multipart itself (`request.form()`) after the check |
| FTDD §7: caller lease renewed "the same way" | `BATCH_LEASE_TTL_S` (900 s) would shorten a caller's 2-hour lease | A caller lease is renewed with its OWN TTL |
| FR-26.6.2: emit after each chunk | FTDD §9 throttles | Chunks throttled to `BATCH_PROGRESS_MIN_INTERVAL_S`; transitions always emit |
| FPRD/FTASKS assume Feature 28 present (steering header) | 028 is not in the tree | `provenance.py` carries today's headers; 028 adds the steering value there |
| Feature 30 mask-aware pooling | Not in the tree | Embedding rows run single, `packed: false` (6.6 as foreseen) |
| FTASKS 0.4: spike "measure" | The answer is a property of the retention code | Answered without hardware: a batch WOULD evict live sensing history; marking extended (migration 020) |

## 4. Reachability

- Every new route is asserted present in `create_app().openapi()["paths"]` and called through the
  real app (`test_batch_file_routes.py`, `test_validation.py`, `test_runner.py`); the §4f contract
  table is asserted EQUAL to the served batch routes in both directions
  (`test_batch_contract_rows_are_served.py`). M12/M12b remove each router include: red.
- `lifespan` is RUN with spies: `start_batch_api` called once with the session factory, stopped
  once (M11); `start_batch_api` reconciles before it starts the runner, then prunes (M11b).
- `acquire_background` has exactly one caller, `_admit` (AST guard over `inference_service` and a
  repo-wide scan), and every new function has a production caller (grep, 2026-10-07).
- `batch:progress` is asserted with payload and exact count (M11c); the backlog provider is
  asserted through `read_inference_state` (X9).

## 5. Suites

Baseline at `6cce090` (`millm.__file__` = `~/app/miLLM-026/millm/__init__.py`): `tests/unit`
**4474 passed / 3 skipped**; admin-ui **471 passed**. Final counts are in the FTASKS 9.7 entry.
`tests/schema` (alembic drift ratchet, migration round trip) green against a local PostgreSQL 15
with migrations 019 and 020. `ruff` and `mypy` are not installed in `~/app/miLLM/venv` — not run.

## 6. Needs hardware — operator session

- 9.3 acceptance 7: 10,000-row JEV-9B-decision scoring batch, unpacked ≥ 19 rows/s, packed rate
  (`tests/performance/test_batch_throughput.py`, skipped without `MILLM_BATCH_HW_URL`).
- 9.4 packed-vs-single difference (max |Δ logprob|, top-token agreement, embedding cosine) → fill
  §4f of the contract and the manual; flip `BATCH_PACK_DEFAULT` if any top token differs (T-63).
- 9.5 acceptance 8 and 9: cancel at ~3,000 rows; pod restart mid-run with and without another
  holder's lease; a chat answered within one chunk during a batch.
- Re-run with `MILLM_REQUIRE_CROSS_REPO_CHECKS=1` once miStudio 034 phase 6 ships `millm_batches.py`.
