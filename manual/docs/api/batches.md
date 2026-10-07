---
sidebar_position: 3
title: Batch API
---

# Batch API

The Batch API turns a JSON Lines (JSONL) file of OpenAI-shaped requests into **one durable job**.
You upload the file, start a batch, watch it, cancel it if you need to, and download the results
as JSONL. The batch lives in PostgreSQL, so a pod restart does not lose it: it resumes from the
first row that has no recorded result, and **no recorded row ever runs twice**.

Every row reaches the model through the same admission slot and the same service code as the
synchronous endpoint. Interactive chat always goes first at a chunk boundary, so a batch never
starves it. A batch **never loads, unloads or swaps a model**.

## The journey

```bash
# 1. Upload the requests
curl -F purpose=batch -F file=@labels.jsonl http://millm/v1/files
# → {"id": "file-…", "object": "file", "bytes": …, "status": "processed", "expires_at": …}

# 2. Start the batch
curl -H 'Content-Type: application/json' http://millm/v1/batches -d '{
  "input_file_id": "file-…", "endpoint": "/v1/completions", "completion_window": "24h"}'
# → {"id": "batch_…", "status": "validating", …}

# 3. Watch it (or listen for the batch:progress Socket.IO event)
curl http://millm/v1/batches/batch_…
# → "status": "in_progress", "request_counts": {"total": 50000, "completed": 1200, "failed": 3}

# 4. Download the results
curl http://millm/v1/files/<output_file_id>/content     # Content-Type: application/jsonl
```

Each input line is `{"custom_id": "...", "method": "POST", "url": "<the batch endpoint>",
"body": {...}}`. The endpoint is one of `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings` or `/api/probes/score` (probe-score lines name no model; the resident model is
used).

## Routes

| Route | What it does |
|---|---|
| `POST /v1/files` | Upload (`multipart/form-data`: `file`, `purpose=batch`) |
| `GET /v1/files` | List, newest first (`purpose`, `limit`, `after`, `order`) |
| `GET /v1/files/{id}` | The file object |
| `GET /v1/files/{id}/content` | The bytes, `application/jsonl` |
| `DELETE /v1/files/{id}` | Delete (refused with `409 file_in_use` while a running batch needs it) |
| `POST /v1/batches` | Create (`input_file_id`, `endpoint`, `completion_window`, optional `metadata`, `output_expires_after`, `pack`) |
| `GET /v1/batches`, `GET /v1/batches/{id}` | List / one batch |
| `POST /v1/batches/{id}/cancel` | Cancel |
| `POST /v1/batches/{id}/lease` | miLLM extension: hand your live lease (`X-miLLM-Lease`) to a waiting batch |

## Validation before anything runs

A new batch starts in `validating`. Every line is checked — JSON, shape, a unique `custom_id`,
`url` equal to the batch's endpoint, no `stream: true`, the endpoint's own schema, and
**strict mode on every line** whatever headers you sent: a field the endpoint would ignore makes
the line invalid, naming the field. Refusals carry the same code and message as the synchronous
endpoint. Invalid lines go to the error file with their line number; valid lines run.

The batch ends `failed` instead of running when no line is valid, when the valid lines name more
than one model, when its model is not the resident one, or when another holder leases the model.
Validation takes no request slot.

## Limits and retention

- 50,000 lines and 200 MB (209,715,200 bytes) per file, 1 MiB per line (`BATCH_MAX_ROWS`,
  `BATCH_MAX_FILE_BYTES`, `BATCH_MAX_LINE_BYTES`). An oversize upload is refused at upload with
  the limit and the measured value; nothing is stored or truncated.
- `completion_window`: `"24h"`, or whole hours `"1h"`–`"168h"` (miLLM extension). A batch still
  running at `expires_at` ends `expired`; completed rows are kept and unrun rows become
  `batch_expired` error lines.
- Every batch file expires 30 days after creation (outputs follow `output_expires_after`,
  3,600–2,592,000 seconds). An expired file's content answers `404 file_expired`; its record stays.
  A file a running batch references is never pruned.

## Results

Output lines are in input order:

```json
{"id": "batch_req_…", "custom_id": "row-17",
 "response": {"status_code": 200, "request_id": "req_…", "body": { … the synchronous response … },
              "millm": {"packed": true, "headers": {"X-miLLM-Backend": "serial", "X-miLLM-Seed": "7;scope=\"request\""}}},
 "error": null}
```

`millm.headers` carries every `X-miLLM-*` value the synchronous route would have set. A row that
fails on its own merits (too long for the context, for example) goes to the error file with the
status code and body the synchronous endpoint would have returned, and the batch goes on.

## Packing

Scoring rows (chat or text completions in scoring mode) are scored one at a time by default
(`pack` omitted → `BATCH_PACK_DEFAULT`, false). Send `pack: true` to score them in right-padded packs. Each row is gathered at its own last real
token. Send `pack: false` when scores must equal the synchronous endpoint bit for bit: bfloat16 is
not batch-invariant. Generation and probe-score rows always run one at a time; embedding rows do
too until mask-aware pooling (Feature 30) lands. Every output line says whether it was packed.

**Measured packed-versus-single difference on JEV-9B-decision (bfloat16, RTX 3090, 10,000 rows, 2026-10-07):** single 23.2 rows/s, packed 63.7 rows/s (2.7x); **57 of 10,000 rows (0.57%) changed their top token**, largest logprob difference 0.349. By the rule the feature shipped with, packing therefore defaults to off. Use `pack: true` only where a small fraction of changed labels is acceptable for the speed.

## Leases, resume and waiting

A batch holds the model lease while it runs (holder `millm-batch:<id>`), renews it, and releases it
when it ends. Submit with your own `X-miLLM-Lease` and the batch runs under **your** lease, renews
it and never releases it. A restart ends every lease: a resumed batch re-acquires one when its
model is resident, and otherwise waits — running no row — until `expires_at`. Hand it your lease
with `POST /v1/batches/{id}/lease` if you hold the model.

While a batch is `in_progress` but not running, `millm.waiting_reason` says why:

| `waiting_reason` | Meaning |
|---|---|
| `queued` | Validated; waiting for its turn |
| `model_not_resident` | Its model is not loaded, or is being loaded/unloaded |
| `lease_unavailable` | Another holder leases the model |

`GET /api/health/detailed` reports `inference.batch_backlog_rows`: rows not yet run across active
batches.
