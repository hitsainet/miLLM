---
sidebar_position: 2
title: Error Codes
---

# Error Codes

Machine-readable error codes returned by the management API in the [error envelope](/api/overview#the-management-envelope) (`error.code`), with their HTTP status. The `/v1` OpenAI-compatible endpoints translate the same conditions into OpenAI-format errors.

## Model errors

| Code | HTTP | Meaning |
|------|------|---------|
| `MODEL_NOT_FOUND` | 404 | No model with that ID |
| `MODEL_ALREADY_EXISTS` | 409 | Same repo + quantization already downloaded |
| `MODEL_LOAD_FAILED` | 500 | Load crashed — see server logs |
| `MODEL_NOT_LOADED` | 400 | Operation needs a loaded model |
| `MODEL_ALREADY_LOADED` | 400 | Load called on the loaded model |
| `MODEL_BUSY` | 409 | Operation conflicts with one in progress |
| `MODEL_LOCKED` | 409 | Unload/delete refused; detach the SAE or unlock first |

## Resource errors

| Code | HTTP | Meaning |
|------|------|---------|
| `INSUFFICIENT_MEMORY` | 507 | Estimated VRAM exceeds what's free |
| `INSUFFICIENT_DISK` | 507 | Not enough disk for the download |

## Download errors

| Code | HTTP | Meaning |
|------|------|---------|
| `DOWNLOAD_FAILED` | 502 | Upstream (HuggingFace) failure |
| `DOWNLOAD_CANCELLED` | 499 | Cancelled by user |
| `REPO_NOT_FOUND` | 404 | Repository doesn't exist |
| `GATED_MODEL_NO_TOKEN` | 401 | Gated repo, no token — accept the license and supply `hf_token` |
| `INVALID_HF_TOKEN` | 401 | Token rejected by HuggingFace |
| `INVALID_LOCAL_PATH` | 400 | Local import path missing, malformed, or in a protected system directory |

## SAE & steering errors

| Code | HTTP | Meaning |
|------|------|---------|
| `SAE_NOT_FOUND` | 404 | No SAE with that ID |
| `SAE_NOT_ATTACHED` | 400 | Steering/monitoring requires an attached SAE |
| `SAE_ALREADY_ATTACHED` | 409 | One SAE at a time; also raised deleting an attached SAE |
| `SAE_INCOMPATIBLE` | 400 | `d_in` doesn't match the loaded model's hidden size |
| `SAE_LOAD_FAILED` | 500 | Weight loading crashed |
| `INVALID_FEATURE_INDEX` | 400 | Feature index outside `[0, d_sae)`, or steering value outside ±200 — `details` names the offender |

## Profile errors

| Code | HTTP | Meaning |
|------|------|---------|
| `PROFILE_NOT_FOUND` | 404 | Unknown profile ID/name (including the per-request `profile` parameter) |
| `PROFILE_ALREADY_EXISTS` | 409 | Duplicate name |
| `PROFILE_INCOMPATIBLE` | 400 | Profile can't apply to the current configuration |
| `INVALID_PROFILE_FORMAT` | 400 | Malformed import payload |

## General

| Code | HTTP | Meaning |
|------|------|---------|
| `VALIDATION_ERROR` | 422 | Request body failed schema validation (also FastAPI's native 422s) |

## Handling errors in code

```python
r = requests.post(f"{MILLM}/api/saes/steering",
                  json={"feature_idx": 999999, "value": 40})
body = r.json()
if not body["success"]:
    code = body["error"]["code"]          # "INVALID_FEATURE_INDEX"
    details = body["error"]["details"]    # {"feature_idx": 999999, "d_sae": 16384}
```

For `/v1` endpoints, OpenAI SDKs raise their native exceptions (`NotFoundError`, `RateLimitError`, …) based on the HTTP status.
