---
sidebar_position: 1
title: API Overview
---

# API Overview

miLLM exposes two HTTP API surfaces and a WebSocket layer on the same port (default `8000`):

| Surface | Base path | Purpose | Error format |
|---------|-----------|---------|--------------|
| [OpenAI-compatible](/api/openai-compatible) | `/v1` | Inference: chat, completions, embeddings, model list | OpenAI-style `{"error": {...}}` |
| Management | `/api` | Models, SAEs, steering, monitoring, profiles, health | Envelope (below) |
| [WebSocket (Socket.IO)](/api/websockets) | `/socket.io` | Progress, metrics, live activations | — |

There is **no authentication** — miLLM is designed to run on trusted networks. Put it behind a reverse proxy with auth if you expose it further. CORS origins are configured via [`CORS_ORIGINS`](/reference/configuration).

## The management envelope

Every `/api/*` endpoint returns the same wrapper:

```json title="Success"
{
  "success": true,
  "data": { "...": "endpoint-specific payload" },
  "error": null
}
```

```json title="Failure"
{
  "success": false,
  "data": null,
  "error": {
    "code": "SAE_NOT_ATTACHED",
    "message": "No SAE attached",
    "details": {}
  }
}
```

`error.code` is machine-readable and stable — the full list with HTTP status mappings is in the [Error Codes reference](/reference/error-codes). `details` carries context (e.g. the offending `feature_idx` and the SAE's `d_sae` on an index-validation failure).

`/v1/*` endpoints return OpenAI-format errors instead, so OpenAI SDKs raise their native exception types:

```json
{
  "error": {
    "message": "No model is currently loaded",
    "type": "invalid_request_error",
    "code": "model_not_loaded"
  }
}
```

## Endpoint map

| Area | Base | Reference |
|------|------|-----------|
| Chat/completions/embeddings | `/v1/...` | [OpenAI-Compatible API](/api/openai-compatible) |
| Model lifecycle | `/api/models` | [Models](/api/models) |
| SAEs, attachment & steering | `/api/saes` | [SAEs & Steering](/api/saes) |
| Activation monitoring | `/api/monitoring` | [Monitoring](/api/monitoring) |
| Steering profiles | `/api/profiles` | [Profiles](/api/profiles) |
| Health & operations | `/api/health` | below |
| Live events | Socket.IO | [WebSocket Events](/api/websockets) |

## Health & operations endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Liveness — process is up |
| `/api/health/ready` | GET | Readiness — dependencies (DB, GPU) are usable |
| `/api/health/detailed` | GET | Per-component health breakdown |
| `/api/health/inference` | GET | Active inference backend (`serial` vs `cbm`), its capabilities and limitations |
| `/api/health/metrics` | GET | Application metrics (requests, latency, GPU) |
| `/api/health/metrics/prometheus` | GET | Prometheus exposition format |
| `/api/health/circuits` | GET | Circuit-breaker states (HuggingFace calls) |
| `/api/health/circuits/{name}/reset` | POST | Manually reset a tripped breaker |
| `/api/health/version` | GET | Application version |

```bash
curl -s http://localhost:8000/api/health
```

```json
{"status": "healthy", "version": "0.5.0", "timestamp": "2026-07-11T12:20:05Z", "uptime_seconds": 512.4}
```

## Conventions

- **Content type** is `application/json` for all request bodies
- **IDs**: models use integer IDs; SAEs and profiles use string IDs (`sae_...`-style hashes, `prof_...`)
- **Queueing**: generation requests beyond the concurrency limit wait in a bounded queue; when full, requests are rejected with `429`
- **Timestamps** are ISO-8601 UTC
