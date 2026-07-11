---
sidebar_position: 4
title: SAEs & Steering API
---

# SAEs & Steering API

SAE lifecycle, attachment, and feature steering at `/api/saes`. All responses use the [management envelope](/api/overview#the-management-envelope).

## SAE lifecycle

### Preview a repository

Browse SAE files in a HuggingFace repo without downloading — essential for GemmaScope repos, which contain hundreds of SAEs:

```bash
curl -X POST http://localhost:8000/api/saes/preview \
  -H "Content-Type: application/json" \
  -d '{"repository_id": "google/gemma-scope-2b-pt-res", "hf_token": "hf_..."}'
```

Returns `files[]` with `path`, `size_bytes`, and the `layer`/`width` parsed from each path.

### Download

```bash
curl -X POST http://localhost:8000/api/saes/download \
  -H "Content-Type: application/json" \
  -d '{
    "repository_id": "google/gemma-scope-2b-pt-res",
    "file_path": "layer_12/width_16k/average_l0_82/params.npz",
    "hf_token": "hf_..."
  }'
```

`file_path` selects one SAE from the repo (omit to download the whole repo — usually not what you want for GemmaScope). Returns `202` with a status of `downloading`, `cached`, `attached`, or `already_downloading` — the call is idempotent per repo+revision+path. Progress arrives via [`sae:download:progress`](/api/websockets). Cancel with `POST /api/saes/{id}/cancel`.

### List / get / delete

```bash
curl http://localhost:8000/api/saes            # all SAEs + current attachment status
curl http://localhost:8000/api/saes/{sae_id}   # one SAE's metadata
curl -X DELETE http://localhost:8000/api/saes/{sae_id}   # refuses 409 while attached
```

## Attachment

### Attach

```bash
curl -X POST http://localhost:8000/api/saes/{sae_id}/attach \
  -H "Content-Type: application/json" \
  -d '{"layer": 12}'
```

```json title="data"
{
  "status": "attached",
  "sae_id": "sae_a1b2c3...",
  "layer": 12,
  "memory_usage_mb": 289,
  "warnings": [],
  "layer_module_path": "model.layers.12"
}
```

- Compatibility is checked first: `d_in` mismatch → `400 SAE_INCOMPATIBLE`; layer/model-family mismatches come back as `warnings`
- `layer_module_path` names the exact module hooked — verify it on unusual architectures
- Attaching locks the model; only one SAE attaches at a time (`409 SAE_ALREADY_ATTACHED`)

There's also a dry-run check: `GET /api/saes/{sae_id}/compatibility?layer=12`.

### Detach & status

```bash
curl -X POST http://localhost:8000/api/saes/{sae_id}/detach
curl http://localhost:8000/api/saes/attachment
```

```json title="GET /api/saes/attachment → data"
{
  "is_attached": true,
  "sae_id": "sae_a1b2c3...",
  "layer": 12,
  "memory_usage_mb": 289,
  "steering_enabled": true,
  "monitoring_enabled": false,
  "steering_apply_count": 4182
}
```

`steering_apply_count` counts forward passes where the steering delta was actually applied — the [verification diagnostic](/concepts/steering#verifying-steering-is-active). Detach drains in-flight requests (serial **and** continuous-batching), clears steering values, and unlocks the model.

## Steering

All steering state is global and takes effect on the next forward pass. Feature indices are validated against the attached SAE's `d_sae`; values must be in **[−200, 200]**.

### Get current state

```bash
curl http://localhost:8000/api/saes/steering
```

```json title="data"
{"enabled": true, "values": {"12082": 40.0, "4517": -15.0}}
```

### Set a single feature

```bash
curl -X POST http://localhost:8000/api/saes/steering \
  -H "Content-Type: application/json" \
  -d '{"feature_idx": 12082, "value": 40}'
```

Setting any value **auto-enables** steering.

### Set many features

```bash
curl -X POST http://localhost:8000/api/saes/steering/batch \
  -H "Content-Type: application/json" \
  -d '{"steering": {"12082": 40, "4517": -15, "9001": 8.5}}'
```

Up to 1000 features per call. Validation is all-or-nothing — one bad index rejects the whole batch with `400 INVALID_FEATURE_INDEX` and no partial application.

### Enable / disable / clear

```bash
curl -X POST   http://localhost:8000/api/saes/steering/enable    # apply configured values
curl -X POST   http://localhost:8000/api/saes/steering/disable   # keep values, stop applying
curl -X DELETE http://localhost:8000/api/saes/steering/12082     # remove one feature
curl -X DELETE http://localhost:8000/api/saes/steering           # remove all + disable
```

Clearing all values also disables steering; removing the last remaining feature does too — the reported `enabled` state always matches what the hook is actually doing.

### Monitoring shortcut

`POST /api/saes/monitoring` configures activation monitoring (`{"enabled": true, "features": [..]}`); it is equivalent to [`POST /api/monitoring/configure`](/api/monitoring) and kept for convenience.

## Common errors

| Code | Status | When |
|------|--------|------|
| `SAE_NOT_FOUND` | 404 | Unknown SAE ID |
| `SAE_NOT_ATTACHED` | 400 | Steering/monitoring calls with nothing attached |
| `SAE_ALREADY_ATTACHED` | 409 | Second attach, or delete while attached |
| `SAE_INCOMPATIBLE` | 400 | Dimension mismatch with the loaded model |
| `INVALID_FEATURE_INDEX` | 400 | Index ≥ `d_sae`, or steering value out of range |
| `MODEL_NOT_LOADED` | 400 | Attach without a loaded model |
