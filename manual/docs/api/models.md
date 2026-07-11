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
| `quantization` | `FP16`, `Q8`, `Q4`, `Q2` — applied at download time, weights saved quantized |
| `trust_remote_code` | Explicit opt-in per download |
| `hf_token` | Never logged or persisted |

Returns `202` with the created model record; progress streams via [`model:download:progress`](/api/websockets) WebSocket events. Cancel with `POST /api/models/{id}/cancel`.

## Load / unload

```bash
curl -X POST http://localhost:8000/api/models/1/load
curl -X POST http://localhost:8000/api/models/1/unload
```

- Loading another model first unloads the current one (one model resident at a time)
- Memory is estimated and checked against free VRAM before load
- Unload is graceful — waits up to `GRACEFUL_UNLOAD_TIMEOUT` for in-flight requests
- Unloading a **locked** model returns `409 MODEL_LOCKED` (models auto-lock while an SAE is attached)

## Lock / unlock

```bash
curl -X POST http://localhost:8000/api/models/1/lock
curl -X POST http://localhost:8000/api/models/1/unlock
```

Locking prevents unload/delete. Attaching an SAE locks automatically; detaching unlocks.

## Delete

```bash
curl -X DELETE http://localhost:8000/api/models/1
```

Hard delete: removes weights from disk and the registry entry. Refused while loaded or locked.

## Common errors

| Code | Status | When |
|------|--------|------|
| `MODEL_NOT_FOUND` | 404 | Unknown ID |
| `MODEL_ALREADY_LOADED` / `MODEL_NOT_LOADED` | 400 | Load/unload state mismatch |
| `MODEL_LOCKED` | 409 | Unload/delete while locked |
| `INSUFFICIENT_MEMORY` / `INSUFFICIENT_DISK` | 507 | Resource checks failed |
| `REPO_NOT_FOUND` | 404 | Bad `repo_id` |
| `GATED_MODEL_NO_TOKEN` / `INVALID_HF_TOKEN` | 401 | HuggingFace auth |

Full list: [Error Codes](/reference/error-codes).
