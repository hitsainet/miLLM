# Feature PRD: Batch API

**Document ID:** 026_FPRD|Batch_API
**Version:** 1.1 (planned)
**Status:** Planned. Written 2026-10-06 from BRD-04 and the 2026-10-06 checkpoint. v1.1 (2026-10-06) applies the operator's Feature-PRD decisions (X-01, X-08) and the register's technical defaults T-63 – T-71 (`~/app/miDataworks/0xcc/docs/fprd-open-questions-2026-10-06.md`); every former open question is now decided (§14).
**Source BRD:** BRD-04 (miLLM — Dataworks Support) §5.5, requirements R-04.16 – R-04.23; acceptance criteria 7, 8 and 9.
**PPRD:** Feature 26, requirement group FR-26.x, Project Product Requirements Document (PPRD) v1.5.
**PADR:** v1.5 §10 "Dataworks Support" — trade-offs *"Batch runner inside the admission path vs a separate batch worker"* and *"Packed scoring by default vs one row at a time"*.
**Binding decisions:** `~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`, "Checkpoint decisions — 2026-10-06" (C8 and the technical defaults) and "Feature-PRD decisions — 2026-10-06" (X-01, X-08); register `fprd-open-questions-2026-10-06.md` T-63 – T-71.
**Depends on:** Feature 29 (model lease) — hard. Feature 25 (strict validation, output-changing field list, chat scoring) — hard. Feature 28 (`X-miLLM-Steering`) — for output-line provenance. Feature 27 (`POST /api/probes/score`) — built first, so probe-score lines are accepted at release (T-71); FR-26.1.4 still derives the endpoint set from served routes. Feature 30 — embedding packing needs its mask-aware pooling (FR-30.2.3).
**Co-release:** miStudio BRD-MIS-DATAWORKS-001 batch tools (`millm_upload_batch_file`, `millm_submit_batch`, `millm_batch_status`, `millm_cancel_batch`, `millm_batch_results`, miStudio `034_FPRD` FR-22) are built only once these routes are served and documented in miLLM `docs/mcp-contract.md`.
**Consumers:** miDataworks BRD-03 R-03.26 ("label runs at scale") through miDataworks `005_FPRD` FR-005.35; miStudio BRD-MIS-DATAWORKS-001 BR-017 and BR-018.

---

## 1. Feature Overview

**Name:** Batch API.

**What it is:** an OpenAI-shaped batch interface. A caller uploads a JSON Lines (JSONL) file of requests. It starts a batch, watches its progress, cancels it if needed, and downloads the results as JSONL. miLLM stores every batch in PostgreSQL. Each row reaches the model through the same admission slot as interactive traffic. A batch survives a pod restart and resumes where it stopped.

**Problem:** on 2026-10-04/05 miLLM labelled 25,000 rows at 19–21 rows per second. Every row was a separate HTTP request in a client loop (BRD-04 §1). A prompt list on `/v1/completions` runs inside one request, with no server-side progress, cancel or resume. A 50,000-row run is 50,000 requests. In this suite a rollout takes in-flight graphics processing unit (GPU) work with it, and nothing requeues it (BRD-04 R-04.18 *Why*).

**Goals:**
- one job, not 50,000 requests: server-side progress, cancel and resume;
- no recorded row ever runs twice, and none is lost to a restart;
- interactive chat is never starved by a batch;
- every output line says how it was produced, including whether it was packed;
- a batch never loads, unloads or swaps a model.

**Connection to the project:** Feature 26 turns the scoring, generation and embedding surfaces of Features 25, 27, 28 and 30 into a durable job. It is the "at scale" half of BRD-03 R-03.26. It sits on the single admission path `_admit()` (`millm/services/inference_service.py:649`), which keeps steering, sensing and probes correct by serial execution.

## 2. User Stories & Scenarios

**US-1: A labelling run is one job.** The miDataworks operator has 50,000 rows to score with JEV-9B-decision.
*Acceptance:* they upload one JSONL file, start one batch and see request counts rise. They download one output file whose lines carry each row's `custom_id`. (R-04.16, R-04.21; BRD acceptance 7)

**US-2: A rollout does not lose the run.** The pod restarts at row 30,000.
*Acceptance:* after restart the batch resumes from the first row without a recorded result. No row recorded before the restart appears twice in the output. (R-04.18; BRD acceptance 8)

**US-3: Cancel keeps the work done.** The operator cancels at about row 3,000.
*Acceptance:* the batch ends `cancelled`. The output file holds every completed row. No new row starts after the cancel. (R-04.21; BRD acceptance 8)

**US-4: Chat keeps working during a batch.** An Open WebUI user chats while a batch runs.
*Acceptance:* the chat request is answered within one chunk's duration. (R-04.19; BRD acceptance 9)

**US-5: An agent hands large work to a batch.** A miStudio Model Context Protocol (MCP) agent needs 10,000 probe scores.
*Acceptance:* the agent uploads, submits, polls and reads results through the miStudio batch tools. The results route returns JSONL with a documented content type, so miStudio's typed path can check it. (miStudio BR-017, BR-018; miStudio `034_FPRD` US-6)

**US-6: Reproducible scores when needed.** A detector operator needs scores identical to the synchronous endpoint.
*Acceptance:* with `pack: false` each scoring row gives the same logprobs as the same request sent synchronously. Every output line states whether it was packed. (R-04.20)

**Edge and error scenarios:**
- A file with some malformed lines: valid lines run, invalid lines go to the error file with line number and reason (R-04.17).
- A file with no valid line: the batch ends `failed` and no row runs (R-04.17; FR-26.2.6).
- Lines naming two different models: the batch ends `failed`, listing the models (FR-26.2.7).
- Another holder has the model lease: the batch does not start, `409 MODEL_LEASED` (R-04.22).
- The named model is not resident: the batch does not load it (R-04.22).
- A file over the row or byte limit: refused at upload, never truncated (R-04.23).
- A pod restart during `validating`, `in_progress`, `finalizing` or `cancelling`: each is reconciled at startup (FR-26.3.3).
- A row that fails on its own merits (for example, longer than the context window): it goes to the error file and the batch continues (FR-26.6.7).

**User journey (happy path):**
1. `POST /v1/files` with the JSONL file and `purpose: "batch"` → file object.
2. `POST /v1/batches` with `input_file_id`, `endpoint`, `completion_window` → batch in `validating`.
3. Validation ends → lease taken → `in_progress`. Progress on `GET /v1/batches/{id}` and Socket.IO.
4. Last row recorded → `finalizing` → output and error files assembled → `completed`.
5. `GET /v1/files/{output_file_id}/content` → JSONL results.

## 3. Functional Requirements

Requirement IDs FR-26.1 – FR-26.8 are the PPRD v1.5 IDs, refined here into testable sub-requirements. FR-26.9 and FR-26.10 are added by this document; their sources are stated.

### FR-26.1 The files and batches surface (R-04.16)

- **FR-26.1.1** `POST /v1/files` SHALL accept a multipart upload with `file` and `purpose`. `purpose` other than `"batch"` SHALL return `400` naming `purpose`. The response SHALL be OpenAI's file object: `id`, `object: "file"`, `bytes`, `created_at`, `filename`, `purpose`, `status`, `expires_at`.
- **FR-26.1.2** `POST /v1/batches` SHALL accept `input_file_id`, `endpoint` and `completion_window` (required), `metadata` (optional, as OpenAI), `output_expires_after` (FR-26.9.5) and the miLLM extension `pack` (FR-26.5). The response SHALL be OpenAI's batch object: `id`, `object: "batch"`, `endpoint`, `model`, `errors`, `input_file_id`, `completion_window`, `status`, `output_file_id`, `error_file_id`, `created_at`, `in_progress_at`, `expires_at`, `finalizing_at`, `completed_at`, `failed_at`, `expired_at`, `cancelling_at`, `cancelled_at`, `request_counts {total, completed, failed}`, `metadata`.
- **FR-26.1.3** `completion_window` SHALL accept `"24h"`, OpenAI's only value, and, as a documented miLLM extension, any whole number of hours `"<N>h"` from 1 up to a configured maximum. Any other value SHALL return `400` naming the field and the accepted range. *Decided:* T-65 — a 50,000-row generation batch needs about 3.5 days, so longer windows are accepted.
- **FR-26.1.4** `endpoint` SHALL be one of `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and `/api/probes/score`. An endpoint is accepted only while miLLM serves its route. The accepted set SHALL be derived from the served routes, not from a hand-kept list. So `/api/probes/score` is refused with `400` until Feature 27 ships, and accepted from then on with no change here. Any other endpoint SHALL return `400` naming the supported set.
- **FR-26.1.5** Each input line SHALL be one JSON object with `custom_id` (string), `method` (`"POST"`), `url` (equal to the batch's `endpoint`) and `body` (that endpoint's request). A duplicate `custom_id`, another `method` or a mismatched `url` SHALL make the line invalid (FR-26.2).
- **FR-26.1.6** The create request itself SHALL follow Feature 25's unknown-field rule: reported in `X-miLLM-Ignored-Fields`, refused under `X-miLLM-Strict: true` (R-04.1, R-04.2).
- **FR-26.1.7** Every batch and file route SHALL accept, and never be broken by, `X-miLLM-Load-Policy` and `X-miLLM-Lease`. miStudio sends both on every batch request (miStudio `034_FPRD` FR-18, FR-19).

### FR-26.2 Validation before any row runs (R-04.17)

- **FR-26.2.1** A new batch SHALL start in `validating`. No row SHALL reach the model until validation ends.
- **FR-26.2.2** Every line SHALL be validated against its endpoint's request schema. A line that fails JSON parsing, the line shape (FR-26.1.5) or the schema is invalid.
- **FR-26.2.3** Strict mode (R-04.2) SHALL apply to every line, whatever headers the create request carried. A field the endpoint would ignore makes the line invalid, naming the field.
- **FR-26.2.4** Feature 25's output-changing field list (R-04.3) and every refusal decidable from the model row (for example, scoring or `response_format` on a GGUF model, R-04.7, R-04.12) SHALL be applied at validation. Such a line is invalid with the same reason the synchronous endpoint gives.
- **FR-26.2.5** Each invalid line SHALL produce one error-file line carrying its 1-based input line number, its `custom_id` when readable, and an error `{code, message}`. The batch's `errors` object SHALL list validation errors in OpenAI's shape (`code`, `line`, `message`, `param`).
- **FR-26.2.6** A file with no valid line SHALL end the batch `failed`, with `failed_at` and `errors` set. No row runs and no lease is taken. *Decided:* T-67 (OpenAI's behaviour).
- **FR-26.2.7** Every valid line SHALL name the same model in `body.model`. Lines naming more than one model SHALL end the batch `failed`, listing each model with its line count. The batch's `model` field is that single model (R-04.22).
- **FR-26.2.8** `request_counts.total` SHALL equal the number of input lines. Invalid lines SHALL count in `failed` from the end of validation.
- **FR-26.2.9** Validation SHALL take no request slot. It is CPU work and must not delay interactive traffic.

### FR-26.3 Persistence and resume (R-04.18)

- **FR-26.3.1** Batches, files and per-row results SHALL be persisted in PostgreSQL. Batch status SHALL take exactly OpenAI's values: `validating`, `in_progress`, `finalizing`, `completed`, `failed`, `cancelling`, `cancelled`, `expired`. Each transition SHALL set its OpenAI timestamp.
- **FR-26.3.2** Rows SHALL be recorded per chunk, atomically: a chunk's results are recorded whole or not at all. A row counts in `request_counts.completed` or `failed` only once recorded.
- **FR-26.3.3** At startup, a named reconciliation function SHALL handle every non-terminal batch: `validating` restarts validation; `in_progress` resumes from the first row without a recorded result; `finalizing` re-assembles its files; `cancelling` ends `cancelled`. A test SHALL fail when the call to it is removed from startup.
- **FR-26.3.4** No recorded row SHALL run twice. A test SHALL interrupt a batch mid-chunk and assert every `custom_id` appears exactly once in the final output and error files.
- **FR-26.3.5** Only these transitions SHALL be allowed: `validating → in_progress | failed | cancelling`; `in_progress → finalizing | cancelling | expired | failed`; `finalizing → completed | failed`; `cancelling → cancelled`. Cancel on a terminal batch SHALL return `409` naming the current status.
- **FR-26.3.6** A resumed batch SHALL NOT load its model. A restart ends every lease (029 FR-29.1.9), so a resumed batch SHALL re-acquire a lease before its next row, and SHALL run no row while it cannot: its model is not resident, or another holder has the lease. It stays `in_progress`, reports why it is waiting, keeps trying, and ends `expired` at `expires_at` with completed rows kept. *Decided:* X-01 (Feature 29 wins) and T-66.
- **FR-26.3.7** Uploaded and assembled file bytes SHALL survive a pod restart.

### FR-26.4 Admission through `_admit()` (R-04.19)

- **FR-26.4.1** Every forward pass for a batch row SHALL run inside `_admit()`. The existing guard `test_every_request_queue_slot_is_taken_through_admission` (`tests/unit/services/test_unload_admission.py:451`) SHALL keep passing with the batch runner present.
- **FR-26.4.2** A batch SHALL hold one slot per chunk and release it between chunks.
- **FR-26.4.3** When an interactive request is waiting at a chunk boundary, it SHALL run before the batch's next chunk.
- **FR-26.4.4** A batch's slot request SHALL NOT count against `MAX_PENDING_REQUESTS` (`millm/core/config.py:256`). A batch SHALL never receive `QUEUE_FULL`. An interactive request's `QUEUE_FULL` threshold SHALL be unchanged while a batch runs.
- **FR-26.4.5** Each row SHALL be executed by the same service code the synchronous endpoint uses. The runner SHALL add no second scoring, generation or embedding path. Feature 27's discovery test over generation entry points (FR-27.9) SHALL cover the runner.
- **FR-26.4.6** Chunk size SHALL be configurable, with a stated default chosen by measurement so that acceptance criterion 9 holds on the reference model.
- **FR-26.4.7** The runner SHALL supply the remaining row count of active batches to Feature 29's batch backlog field in `/api/health/detailed` (FR-29.7).
- **FR-26.4.8** Probe events written for batch generation rows SHALL be marked as batch events, carrying the batch id and input line. They SHALL be kept outside the live-traffic cap (`PROBE_MAX_EVENTS_PER_PROBE`, `millm/core/config.py:241`), so a batch never evicts live-traffic events, and SHALL NOT be emitted on the live probe feed. *Decided:* T-70.

### FR-26.5 Packing (R-04.20)

- **FR-26.5.1** `pack` SHALL default to `true` (R-04.20). It SHALL affect only scoring rows (chat or text completions in scoring mode) and embedding rows. *Decided:* T-63 — if the acceptance-7 measurement shows any packed row whose label (top token) differs from its single-row label, the default becomes `false`. The default is therefore a configuration value, set from that measurement.
- **FR-26.5.2** Generation rows SHALL always run with single-request semantics, whatever `pack` says.
- **FR-26.5.3** `/api/probes/score` rows SHALL always run one input at a time, whatever `pack` says. This is the checkpoint default ("Probe scoring of stored text runs one input at a time, matching miStudio's parity method").
- **FR-26.5.4** With `pack: false`, a scoring row SHALL return the same token IDs and logprobs as the same request sent to the synchronous endpoint.
- **FR-26.5.5** A packed scoring row SHALL score its own last real token, never a padding position. A test with rows of different lengths SHALL fail if padding lands between a prompt and its scored position.
- **FR-26.5.6** Every output line SHALL state whether its row was packed. The batch object SHALL report `pack`.
- **FR-26.5.7** Rows on the llama.cpp engine (GGUF embeddings) SHALL NOT be packed, and their lines SHALL say so.
- **FR-26.5.8** The packed-versus-single difference SHALL be measured on JEV-9B-decision in bfloat16 and stated in the API reference: maximum absolute logprob difference, top-token agreement rate, and embedding cosine. The rule that flips the default to off is T-63 (FR-26.5.1).

### FR-26.6 Status, progress, cancel, list and content (R-04.21)

- **FR-26.6.1** `GET /v1/batches/{id}` SHALL return the batch object, with `request_counts` current to the last recorded chunk.
- **FR-26.6.2** A Socket.IO progress event SHALL be emitted after each recorded chunk and on each status transition. Its payload SHALL carry `id`, `status` and `request_counts`. The event name follows the existing `noun:verb` convention (for example `model:download:progress`, `millm/sockets/progress.py:181`) and is fixed in the technical design.
- **FR-26.6.3** `GET /v1/batches` SHALL list batches newest first in OpenAI's list shape (`object: "list"`, `data`, `first_id`, `last_id`, `has_more`), paged by `after` and `limit`.
- **FR-26.6.4** `POST /v1/batches/{id}/cancel` SHALL set `cancelling` and start no new row. A forward pass already running SHALL finish and its rows SHALL be recorded. The batch then ends `cancelled` with output and error files holding every recorded row. A batch cancelled while `validating` ends `cancelled` without running.
- **FR-26.6.5** `GET /v1/files/{id}` SHALL return the file object. `GET /v1/files/{id}/content` SHALL return the file's bytes. Output and error files SHALL be served with one stated JSON Lines content type, documented in the API reference and in `docs/mcp-contract.md`.
- **FR-26.6.6** An output line SHALL take OpenAI's shape: `id`, `custom_id`, `response {status_code, request_id, body}`, `error: null`. `body` SHALL be what the synchronous endpoint would have returned for that row.
- **FR-26.6.7** A row that fails on its own merits SHALL produce an error-file line: `id`, `custom_id`, `response` (with the status code the synchronous endpoint would return) and `error {code, message}`. The batch SHALL continue.
- **FR-26.6.8** Output and error lines SHALL be written in input-line order.
- **FR-26.6.9** `GET /v1/files` SHALL list files newest first in OpenAI's list shape, filterable by `purpose`. *Decided:* T-69.
- **FR-26.6.10** `DELETE /v1/files/{id}` SHALL delete a file's bytes and mark its record deleted, returning OpenAI's `{id, object: "file", deleted: true}`. A file referenced by a non-terminal batch SHALL be refused with `409` naming the batch. *Decided:* T-69 (disk pressure, RSK-06).

### FR-26.7 One model, held under the lease (R-04.22)

- **FR-26.7.1** A batch SHALL never load, unload or swap a model, whatever `X-miLLM-Load-Policy` says.
- **FR-26.7.2** When validation ends and the batch's model is not resident, the batch SHALL end `failed` with an error naming the batch's model and the resident one.
- **FR-26.7.3** `POST /v1/batches` SHALL return `409 MODEL_LEASED`, naming holder and expiry, when a lease is live and the request does not present it. One lease exists per server, on the resident model (029 FR-29.1.3, FR-29.1.4), so this check needs no line to be read. The check SHALL run again at the move to `in_progress`; a lease lost in between ends the batch `failed` with the same code.
- **FR-26.7.4** On entering `in_progress` the batch SHALL acquire the lease with a holder naming the batch. It SHALL renew the lease while running, within the 2-hour maximum time to live (TTL) (checkpoint default). It SHALL release the lease on every terminal status, and on `failed` reached by any path.
- **FR-26.7.5** A batch submitted with a valid `X-miLLM-Lease` for its model is the holder's own request (R-04.40) and SHALL NOT be refused as another holder. The batch runs under that lease, renews it while running and does not release it at the end. *Decided:* T-64. This is how miDataworks submits, since it holds one lease per model (X-08).
- **FR-26.7.6** While a batch holds the lease, interactive inference on the same model SHALL proceed. The lease blocks model changes, not requests (R-04.39).
- **FR-26.7.7** A resumed batch re-acquires a lease of its own (FR-26.3.6). Where another caller holds the model's lease — for example miDataworks after re-taking its lease (X-08) — that caller SHALL be able to hand its live lease to a waiting batch of its own, so the batch runs under it as in FR-26.7.5. *Derived:* X-01 and X-08 together; without it, a miDataworks batch would wait behind miDataworks' own lease until it expired. The route is fixed in the FTDD.

### FR-26.8 Limits (R-04.23)

- **FR-26.8.1** The maximum rows per file and bytes per file SHALL be configurable. Their defaults SHALL be OpenAI's documented limits, 50,000 requests and 200 MB, and SHALL be stated in the API reference.
- **FR-26.8.2** A file over either limit SHALL be refused at upload with `400` naming the limit and the measured value. Nothing SHALL be stored, and nothing is ever truncated.
- **FR-26.8.3** The byte limit SHALL be enforced while the upload is read, so an oversized upload is never held whole in memory.

### FR-26.9 Retention (checkpoint default; BRD-04 §9 question 4; RSK-06)

- **FR-26.9.1** Output and error files SHALL expire 30 days after creation. This is the checkpoint default ("Batch result files are kept 30 days"). `expires_at` on the file object SHALL state it.
- **FR-26.9.2** Input files SHALL also expire 30 days after creation, stated in `expires_at`. *Decided:* T-68.
- **FR-26.9.3** An expired file's content route SHALL return `404` with a code saying the file expired. The batch record SHALL remain.
- **FR-26.9.4** Pruning SHALL never remove a file referenced by a non-terminal batch. It SHALL run periodically and at startup.
- **FR-26.9.5** `output_expires_after` (OpenAI's create field) SHALL be honoured within OpenAI's bounds of 3,600 seconds to 2,592,000 seconds (30 days). A value outside them SHALL return `400`.

### FR-26.10 Provenance and contract (BRD-04 §2; R-04.33 batch clause; miStudio BR-013, BR-017)

- **FR-26.10.1** Each output line SHALL carry, in an extension object, every `X-miLLM-*` provenance value the synchronous endpoint would have set for that row. This includes the steering state (FR-28.3) and, where they apply, the seed and constrained-format values (FR-25.12, FR-25.13), plus the packed flag (FR-26.5.6).
- **FR-26.10.2** miLLM `docs/mcp-contract.md` SHALL gain a section for the batch and file routes: paths, payloads, status values, limits, the results content type and the packed-versus-single measurement. miStudio's tools are built only against this section (miStudio BR-013).

### Coverage of BRD-04 requirements

| BRD-04 requirement | PPRD FR | Refined here | Acceptance |
|---|---|---|---|
| R-04.16 files and batches, line shape, endpoints | FR-26.1 | FR-26.1.1 – FR-26.1.7 | 7 |
| R-04.17 validation before any row, strict per line | FR-26.2 | FR-26.2.1 – FR-26.2.9 | 7 |
| R-04.18 PostgreSQL persistence, OpenAI statuses, resume, no row twice | FR-26.3 | FR-26.3.1 – FR-26.3.7 | 8 |
| R-04.19 only through `_admit()`, slot per chunk, not pending | FR-26.4 | FR-26.4.1 – FR-26.4.8 | 9 |
| R-04.20 packing on by default, `pack: false`, measured difference | FR-26.5 | FR-26.5.1 – FR-26.5.8 | 7 |
| R-04.21 counts, Socket.IO progress, cancel, list, content | FR-26.6 | FR-26.6.1 – FR-26.6.10 | 7, 8 |
| R-04.22 one model, lease for the whole run, never load | FR-26.7 | FR-26.7.1 – FR-26.7.7 | 8 (a restart ends the lease; the resumed batch re-acquires one or does not run — X-01) |
| R-04.23 configurable limits, refused at upload | FR-26.8 | FR-26.8.1 – FR-26.8.3 | — (unit) |

**8 of 8** R-04 requirements owned by Feature 26 are covered. FR-26.9 (retention) comes from the checkpoint default and BRD-04 §9 question 4. FR-26.10 (provenance and contract) comes from BRD-04 §2, the batch clause of R-04.33 (owned by Feature 28) and miStudio BR-013 / BR-017.

## 4. User Experience Requirements

There is no new Admin UI page. The surface is the OpenAI-shaped API plus a Socket.IO event.
- Error responses use the existing OpenAI error envelope (`millm/api/routes/openai/errors.py`).
- Every refusal names the field, limit, model or holder that caused it.
- The API reference documents every route, the extension fields (`pack`, the provenance object), limits, retention and the packed-versus-single measurement.

## 5. Data Requirements

- **Batch record:** id, endpoint, model, status, every OpenAI timestamp, request counts, `pack`, metadata, `errors`, input/output/error file ids, the lease id it holds, and a waiting reason (FR-26.3.6).
- **File record:** id, purpose (`batch`, `batch_output`), filename, bytes, row count, created and expiry times, and where its bytes are stored.
- **Row result record:** batch id, input line number, `custom_id`, outcome, the result line, packed flag, and the chunk that recorded it. Uniqueness on (batch id, line number) enforces FR-26.3.4 in the database.
- **Migrations:** one Alembic migration at the next free number in `millm/db/migrations/versions/` (the last is `018_add_probe_threshold_revision.py`; Features 25, 27 and 29 may take numbers first).
- **Byte storage:** PostgreSQL or the `/data` volume. Both survive a pod restart. The choice is the technical design's, weighed against nightly backup size (`k8s/base/db-backup.yaml`).

## 6. Technical Constraints

- `MAX_CONCURRENT_REQUESTS` stays 1 (`millm/core/config.py:255`; BRD-04 §7). Steering, sensing and probes depend on serial execution.
- One resident model; the batch runs inside the API process (PADR trade-off "Batch runner inside the admission path vs a separate batch worker").
- The request queue counts every waiter as pending (`millm/services/request_queue.py`, `acquire`). FR-26.4.4 needs a way to wait for a slot without that count; it must still go through `_admit()`.
- Each synchronous service method takes its own slot today (for example `_score_text_completion` at `millm/services/inference_service.py:4930`). The runner's chunk slot and the per-row methods must not both acquire it, or the runner deadlocks at concurrency 1.
- bfloat16 is not batch-invariant: up to 0.177 between batched and single probe scores (PADR, citing miStudio `0xcc/reviews/native_dtype_2026-10-03.md`).
- Continuous batching stays off in production (BRD-04 §3, §7).
- `python-multipart` is already a dependency (`pyproject.toml:24`).

## 7. API/Integration Specifications

| Method | Path | Purpose | Requirement |
|---|---|---|---|
| POST | `/v1/files` | Upload a JSONL file, `purpose: "batch"` | FR-26.1.1, FR-26.8 |
| GET | `/v1/files/{id}` | File object | FR-26.6.5 |
| GET | `/v1/files/{id}/content` | File bytes; JSONL with a stated content type | FR-26.6.5 |
| POST | `/v1/batches` | Create a batch | FR-26.1.2 – FR-26.1.6 |
| GET | `/v1/batches/{id}` | Batch object and counts | FR-26.6.1 |
| GET | `/v1/batches` | List batches, `after` / `limit` | FR-26.6.3 |
| POST | `/v1/batches/{id}/cancel` | Cancel | FR-26.6.4 |
| Socket.IO | progress event (name fixed in design) | Progress and transitions | FR-26.6.2 |

- No authentication, as for every miLLM route (BRD-04 §3, decision 7).
- Integration: Feature 29 lease routes, Feature 25 validation helpers, Feature 28 steering-state value, Feature 27 probe scoring, Feature 29 health fields.

## 8. Non-Functional Requirements

- **Throughput:** an unpacked 10,000-row JEV-9B-decision scoring batch SHALL run at 19 rows per second or more (BRD acceptance 7). The packed rate SHALL be recorded.
- **Interactive latency:** a chat request during a batch is answered within one chunk's duration (BRD acceptance 9).
- **Durability:** no recorded row lost or repeated across a restart (BRD acceptance 8).
- **Disk:** bounded by FR-26.8 limits and FR-26.9 retention (RSK-06).
- **Privacy:** batch rows write no new log lines carrying row text.

## 9. Feature Boundaries (Non-Goals)

- A separate batch worker or a second resident model (PADR trade-off; BRD-04 §7).
- Packing generation rows (R-04.20 covers scoring and embeddings only).
- Packing probe-score rows (checkpoint default).
- Probe scoring of each row inside a batched chat request (BRD-04 §7).
- Authentication (decision 7). MCP tools (miStudio owns them).
- An Admin UI page for batches.

## 10. Dependencies

- **Feature 29** (hard): the lease (R-04.38 – R-04.40) and health fields (R-04.44).
- **Feature 25** (hard): strict mode, the output-changing field list, scoring-mode chat.
- **Feature 28:** `X-miLLM-Steering` for FR-26.10.1.
- **Feature 27:** `/api/probes/score`. Built before Feature 26 (T-71); FR-26.1.4 still derives acceptance from the served route.
- **Feature 30:** mask-aware pooling (FR-30.2.3); until it ships, embedding rows run unpacked and say so.
- **Feature 24:** probe behaviour on generation rows follows the synchronous path, with events marked as batch (FR-26.4.8, T-70).
- Hardware: the RTX 3090 node and JEV-9B-decision for acceptance 7 and FR-26.5.8.

## 11. Success Criteria

BRD-04 acceptance criteria owned by this feature:
1. **Acceptance 7 (hardware).** A 10,000-row JEV-9B-decision scoring batch completes, with progress visible throughout. Unpacked throughput is 19 rows per second or more. Packed rate and packed-versus-single difference are recorded.
2. **Acceptance 8.** Cancel at about 3,000 rows keeps every completed row and runs none twice. A pod restart during a second batch resumes it with no recorded row repeated.
3. **Acceptance 9.** A chat request sent during a running batch is answered within one chunk's duration.

Plus: every wiring item (route registration, startup reconciliation call, lease acquire and release, `_admit()` use, progress emission) is accepted only by a test that fails when the line is removed. The test asserts payload and call count (PPRD FR-20.3).

## 12. Testing Requirements

- **Unit:** line validation (each failure kind, strict mode, model mismatch); status transitions; limits; retention pruning with a fake clock; output-line shape; input-line ordering.
- **Admission:** `test_every_request_queue_slot_is_taken_through_admission` stays green; a waiting interactive request beats the next chunk; batch waits do not raise `QUEUE_FULL` and do not change the interactive threshold.
- **Resume:** interrupt mid-chunk, restart, assert exactly-once `custom_id`s. Remove the startup reconciliation call and require a red.
- **Packing:** `pack: false` equals the synchronous endpoint bit for bit; mixed-length packed rows score their own last real token; probe rows never packed.
- **Lease:** another holder → `409`; lease released on every terminal path (including a raised exception); the batch never calls a load or unload.
- **Mutation controls:** break the per-chunk atomic record, the pending-count exemption, the lease release and the progress emission, one at a time. Each must turn the suite red.
- **Hardware:** acceptance 7, 8, 9 on the node; FR-26.5.8 measurement.

## 13. Implementation Considerations

- **Complexity:** high. Durable job state, restart reconciliation and a new admission mode are each easy to get subtly wrong.
- **Approach:** a runner task in the API process, started at startup. It reads the next unrecorded chunk, takes one slot, runs the rows through factored-out slot-free service bodies, records the chunk in one transaction, then releases the slot.
- **Challenges:**
  - factoring slot acquisition out of the synchronous methods without creating a second path (FR-26.4.5, §6);
  - waiting for a slot without counting as pending, while staying inside `_admit()`;
  - fairness at the chunk boundary (FR-26.4.3), which a naive release-and-reacquire may not give;
  - packed scoring: `_next_token_logits` reads `[0, -1]` only (`inference_service.py:5037-5065`), so packing needs left padding or explicit last-token indices, as batched generation already uses (`inference_service.py:3186-3193`);
  - an out-of-memory error in a packed chunk should not fail rows that would succeed alone. The existing batched path shrinks chunks rather than refusing (`_chunk_batch_for_memory`, `inference_service.py:3349-3388`).
- **Order:** after Features 25 and 29 (PPRD; BRD-04 RSK-09).

## 14. Open Questions

Every question in v1.0 is now decided. Each decided item is also written into the requirement it changes.

| # | Former question | Decision | Source | Changes |
|---|---|---|---|---|
| 1 | Packing threshold on JEV-9B-decision | `pack: true` stays the default unless acceptance 7 finds any packed row whose label differs from its single-row label; then the default becomes `false` | T-63 | FR-26.5.1 |
| 2 | A batch under the caller's lease | The batch renews it while running and does not release it | T-64 | FR-26.7.5 |
| 3 | `completion_window` beyond 24 hours | Accepted as a documented extension, whole hours up to a configured maximum | T-65 | FR-26.1.3 |
| 4 | Resume without the model; lease across restart | A restart ends every lease. The resumed batch re-acquires one when its model is resident, runs no row while it cannot, and ends `expired` at `expires_at` | X-01, T-66 | FR-26.3.6, FR-26.7.7; coverage row R-04.22 |
| 5 | File with no valid line | `failed` after `validating` | T-67 | FR-26.2.6 |
| 6 | Input-file retention | 30 days | T-68 | FR-26.9.2 |
| 7 | `GET /v1/files` and `DELETE /v1/files/{id}` | Serve both | T-69 | FR-26.6.9, FR-26.6.10 |
| 8 | Probe events from batch generation rows | Written, marked as batch, outside the live-traffic cap | T-70 | FR-26.4.8 |
| 9 | Build order with Feature 27 | Feature 27 first, so probe-score lines are accepted at release | T-71 | Header, §10 |

**Still open (not product questions; recorded for the FTDD):**
- Sensing and circuit-edge sensing events from batch generation rows. T-70 covers probe events only. The FTDD records what batch rows do to sensing and whether the same marking applies.

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived. Each question is answered from a cited source; genuinely undecided points are in §14.

| # | Question | Answer | Source |
|---|---|---|---|
| D1 | Priority | Important; after Features 25 and 29 | PPRD Feature 26; BRD-04 RSK-09 |
| D2 | Where does the runner live? | Inside the API process, through `_admit()` | PADR "Batch runner inside the admission path vs a separate batch worker"; R-04.19 |
| D3 | API shape | OpenAI's files and batches objects, statuses and line shapes | R-04.16, R-04.18; OpenAI SDK types (`batch.py`, `batch_create_params.py`, `batch_error.py`, `file_object.py`) |
| D4 | Pack by default? | Yes, for scoring and embedding rows; `pack: false` available | R-04.20; PADR "Packed scoring by default vs one row at a time" |
| D5 | Are probe-score rows packed? | No; one input at a time | Checkpoint technical default; FR-26.5.3 |
| D6 | Result-file retention | 30 days | Checkpoint technical default |
| D7 | Lease relationship | Batch takes the lease; the lease sits beside `locked` | R-04.22; checkpoint C8 |
| D8 | Maximum lease TTL | 2 hours, renewable | Checkpoint technical default |
| D9 | Row and byte limit defaults | 50,000 requests, 200 MB | R-04.23 requires stated defaults; values are OpenAI's documented limits |
| D10 | Results format | JSONL from `/v1/files/{id}/content` with a stated content type | R-04.16, R-04.21; miStudio BR-017, `034_FPRD` FR-22 and D12 |
| D11 | Strict validation per line | Always, whatever the request headers | R-04.17 |
| D12 | Who builds the miStudio tools? | miStudio, after this route is served and documented | BRD-04 §4; miStudio BR-013 |
| D13 | Does a batch ever load a model? | Never | R-04.22; BRD-03 R-03.27 |
| D14 | Output-line order | Input-line order | This document: makes resume and file assembly deterministic; OpenAI does not promise an order, so it is compatible |
| D15 | Lease across a restart | Ended by the restart; re-acquired, or the batch does not run | X-01 (Feature 29 wins) |
| D16 | Leases held by miDataworks | One per model; its batches run under it | X-08; T-64 |
| D17 | Packing default | `true`, flipped to `false` by configuration if acceptance 7 finds a label difference | T-63 |
| D18 | `completion_window` | `24h` plus whole hours up to a configured maximum | T-65 |
| D19 | All-invalid file | `failed` after `validating` | T-67 |
| D20 | Input-file retention | 30 days | T-68 |
| D21 | File list and delete | Served | T-69 |
| D22 | Probe events from batch rows | Marked as batch, outside the live cap, not on the live feed | T-70 |
| D23 | Build order | Feature 27 before Feature 26 | T-71 |
