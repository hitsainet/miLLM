---
sidebar_position: 3
title: Models API
---

# Models API

Model lifecycle management at `/api/models`. All responses use the [management envelope](/api/overview#the-management-envelope); examples below show the `data` payload.

## List models

```bash
curl http://localhost:8000/api/models
```

Returns every model in the registry with its status (`downloading`, `ready`, `loading`, `loaded`, `error`), quantization, sizes, and lock state:

```json
{
  "models": [{
    "id": 1,
    "name": "gemma-2-2b",
    "source": "huggingface",
    "repo_id": "google/gemma-2-2b",
    "quantization": "FP16",
    "params": "2.6B",
    "architecture": "Gemma2ForCausalLM",
    "disk_size_mb": 5240,
    "estimated_memory_mb": 6100,
    "status": "loaded",
    "locked": true,
    "device": "cuda:0",
    "dtype": "torch.bfloat16",
    "loaded_at": "2026-07-11T12:19:40Z"
  }],
  "total": 1
}
```

## Preview a model

Check size, architecture, and per-quantization memory estimates **before** downloading:

```bash
curl -X POST http://localhost:8000/api/models/preview \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "google/gemma-2-2b", "hf_token": "hf_..."}'
```

Gated repos without a valid token return `401 GATED_MODEL_NO_TOKEN`.

## Download a model

```bash
curl -X POST http://localhost:8000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "source": "huggingface",
    "repo_id": "google/gemma-2-2b",
    "quantization": "FP16",
    "trust_remote_code": false,
    "hf_token": "hf_...",
    "custom_name": null
  }'
```

| Field | Notes |
|-------|-------|
| `source` | `huggingface` or `local` |
| `repo_id` | Required for `huggingface` |
| `local_path` | Required for `local`; system directories are rejected |
| `quantization` | `FP16`, `Q8`, `Q4`, `Q2` — the precision the model is **loaded** at. The download stores the repository's checkpoint as published whatever you pick (a `Q4` row downloads full-precision weights), and bitsandbytes quantizes it on every load. A checkpoint that is already quantized (GPTQ, AWQ, bitsandbytes, BitNet) loads at its own precision and is sized from the weights it stores, or from what transformers loads it as when that is larger: an FP8 checkpoint is loaded as bfloat16 on a card below compute capability 8.9, such as an RTX 3090. A `Q2` transformers checkpoint that is not already quantized is refused at load (`400 UNSUPPORTED_QUANTIZATION`): bitsandbytes has no 2-bit mode |
| `trust_remote_code` | Explicit opt-in per download |
| `hf_token` | Never logged or persisted |
| `gguf_label` | The exact GGUF quantization to fetch, e.g. `IQ4_XS`. Names the model `repo:LABEL` so several quantizations of one repository can coexist — see [Model Management](/features/model-management#gguf-models-and-quantization). |

Returns `202` with the created model record; progress streams via [`model:download:progress`](/api/websockets) WebSocket events. Cancel with `POST /api/models/{id}/cancel`.

## Load / unload

```bash
curl -X POST http://localhost:8000/api/models/1/load
curl -X POST http://localhost:8000/api/models/1/unload
```

- Loading another model first unloads the current one (one model resident at a time)
- Memory is estimated and checked against free VRAM before load
- Unload is graceful — waits up to `GRACEFUL_UNLOAD_TIMEOUT` for in-flight requests
- While another holder **leases** the resident model, a load of any other model and an unload
  are refused with `409 MODEL_LEASED`, naming the holder and expiry. The holder sends its lease
  ID in `X-miLLM-Lease` and proceeds; the lease then ends with the unloaded model. See
  [Model lease](#model-lease)
- ⚠ These two routes do **not** read `locked`: only the `/v1` auto-load does. A steering model
  can be unloaded here while locked. This is tracked debt, to be retired with `locked` in favour
  of the lease
- `X-miLLM-Load-Policy` has no effect here: these routes load by explicit request

## Lock / unlock

```bash
curl -X POST http://localhost:8000/api/models/1/lock
curl -X POST http://localhost:8000/api/models/1/unlock
```

Locking stops a `/v1` request from auto-loading another model over the locked one (it answers
`409 model_locked`). Attaching an SAE locks automatically; detaching unlocks. The management load
and unload above do not read it (tracked debt); use a [lease](#model-lease) to pin a model against
every load path.

## Delete

```bash
curl -X DELETE http://localhost:8000/api/models/1
```

Hard delete: removes weights from disk and the registry entry. Refused while loaded or locked.

## Model lease

A lease pins the **resident** model for a named holder until its time to live (TTL) runs out.
While it is live, nobody else can load another model, unload it, or swap it — through the
management routes or a `/v1` auto-load. Requests that **use** the leased model keep working
without the lease ID: a lease blocks swaps, not use. No approval is needed to take one.

| Route | Body / headers | Success | Refusals |
|---|---|---|---|
| `POST /api/models/{id}/lease` | `{holder, reason, ttl_seconds?}` | `201`, the lease **with `lease_id`** | `400 INVALID_LEASE_REQUEST`, `404 MODEL_NOT_FOUND`, `409 MODEL_NOT_RESIDENT`, `409 MODEL_LEASED`, `409 MODEL_BUSY` (a load or unload is running) |
| `GET /api/models/{id}/lease` | — | `200`, `{lease, last_ended}` | `404 MODEL_NOT_FOUND` |
| `POST /api/models/{id}/lease/renew` | header `X-miLLM-Lease`; `{ttl_seconds?}` | `200`, the lease | `400`, `404 LEASE_NOT_FOUND`, `409 LEASE_EXPIRED` |
| `DELETE /api/models/{id}/lease` | header `X-miLLM-Lease` | `200`, the ended lease with `end_reason` | `404 LEASE_NOT_FOUND`, `409 LEASE_EXPIRED` |

```bash
# Take a lease on the resident model (id 3) for two hours
curl -X POST http://localhost:8000/api/models/3/lease \
  -H 'Content-Type: application/json' \
  -d '{"holder": "midataworks", "reason": "label run 7", "ttl_seconds": 7200}'
# → 201 {"success": true, "data": {"lease_id": "Xq…", "model_id": 3, "holder": "midataworks",
#        "expires_at": "2026-10-06T14:00:00Z", "ttl_seconds": 7200, "seconds_remaining": 7200, …}}

# Renew: the new expiry is NOW + ttl_seconds, not the old expiry + ttl_seconds
curl -X POST http://localhost:8000/api/models/3/lease/renew \
  -H 'X-miLLM-Lease: Xq…' -H 'Content-Type: application/json' -d '{"ttl_seconds": 3600}'

# Swap the model as the holder: proceeds, and the lease ends with the unloaded model
curl -X POST http://localhost:8000/api/models/5/load -H 'X-miLLM-Lease: Xq…'

# Release
curl -X DELETE http://localhost:8000/api/models/3/lease -H 'X-miLLM-Lease: Xq…'
```

- **`lease_id` is returned once**, by the grant. No read returns it — not `GET`, not the model
  list, not `/api/health/detailed`, not the Admin UI, not a log line (logs carry an 8-character
  `lease_ref`). It travels in the `X-miLLM-Lease` header, never in a path, because paths reach
  access logs.
- **`holder`** (1–128 characters) and **`reason`** (1–512) are required free text. `holder` is a
  label, not an identity: two callers sending the same string are two callers.
- **`ttl_seconds`** is an integer from 1 to 7200, default 7200. A value outside that range is
  refused with `400 INVALID_LEASE_REQUEST` naming the field and the limit — never clamped.
- **Only the resident, loaded model can be leased**, and at most one lease is live. A lease ends
  when its TTL passes (checked on every read; no background task needed), when the holder
  releases it, when its model stops being resident (`end_reason: model_unloaded`), and **when
  miLLM restarts** — a restart ends every lease, and renew then answers `404 LEASE_NOT_FOUND`
  "unknown lease; a restart ends every lease". Re-acquire once the model is resident.
- A lease ID for a different model than the path answers `404`, like an unknown one.
- A wrong lease ID on a request that needed no lift is ignored (logged as a warning).
- The lease and `locked` are independent. When both refuse a `/v1` auto-load the answer is
  `409 model_leased`, because it names a holder and an expiry.

### `X-miLLM-Load-Policy` on `/v1`

`/v1/chat/completions`, `/v1/completions` and `/v1/embeddings` load the model a request names.
`X-miLLM-Load-Policy: refuse` (case-insensitive) promises the request never causes a load:

| Situation | Answer |
|---|---|
| The named model is resident | Served as usual |
| It is not resident | `409 model_not_resident`, naming the requested and the resident model ("none"), plus the resident model's lease when one is held. Nothing loads |
| It is being loaded right now | `503 model_loading` with `Retry-After` |

Absent, or `auto`, keeps today's auto-load. Any other value is `400 invalid_parameter` with
`param: "X-miLLM-Load-Policy"`. The existing refusals (unknown model, embedding-only model, the
request policy's GGUF refusals) still come first. `/v1` routes also read `X-miLLM-Lease`: the
holder's own auto-load swap proceeds; anyone else's is `409 model_leased`.

## GGUF model names

A downloaded GGUF model is named `repo:QUANT`, and each quantization appears separately in `/v1/models`. A **bare** repository name resolves while exactly one quantization exists; once a second is downloaded the bare name returns **400 `AMBIGUOUS_MODEL_NAME`**, listing the tags that exist. Requests should name the tag exactly.

Model records also carry `supports_embeddings`. It is `false` when the architecture refused the pooling mode embeddings require and miLLM loaded the model without them — serving the model matters more than embedding it, and the flag says so up front rather than leaving a caller to discover it at `/v1/embeddings`.

## Common errors

| Code | Status | When |
|------|--------|------|
| `MODEL_NOT_FOUND` | 404 | Unknown ID |
| `MODEL_ALREADY_LOADED` / `MODEL_NOT_LOADED` | 400 | Load/unload state mismatch |
| `MODEL_LOCKED` | 409 | Lock requested while another model is locked |
| `MODEL_LEASED` | 409 | Load, unload or swap while another holder leases the resident model; `details` name holder, reason, `expires_at` |
| `MODEL_NOT_RESIDENT` | 409 | Lease requested on a model that is not the resident, loaded one |
| `LEASE_NOT_FOUND` / `LEASE_EXPIRED` | 404 / 409 | Renew or release with an unknown, or an ended, lease ID |
| `INVALID_LEASE_REQUEST` | 400 | `holder`, `reason`, `ttl_seconds` or the `X-miLLM-Lease` header out of bounds |
| `INSUFFICIENT_MEMORY` / `INSUFFICIENT_DISK` | 507 | Resource checks failed |
| `REPO_NOT_FOUND` | 404 | Bad `repo_id` |
| `GATED_MODEL_NO_TOKEN` / `INVALID_HF_TOKEN` | 401 | HuggingFace auth |

Full list: [Error Codes](/reference/error-codes).
