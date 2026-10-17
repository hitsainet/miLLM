---
sidebar_position: 5
title: Monitoring API
---

# Monitoring API

Feature-activation monitoring at `/api/monitoring`. Requires an attached SAE. Capture semantics — what a record actually represents — are explained in [Concepts: Monitoring](/concepts/monitoring).

## State & configuration

### Get state

```bash
curl http://localhost:8000/api/monitoring
```

```json title="data"
{
  "enabled": true,
  "sae_attached": true,
  "sae_id": "sae_a1b2c3...",
  "monitored_features": [12082, 4517],
  "history_size": 100,
  "history_count": 37,
  "top_k": 10
}
```

### Configure

```bash
curl -X POST http://localhost:8000/api/monitoring/configure \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "features": [12082, 4517, 9001],
    "history_size": 200,
    "top_k": 10
  }'
```

| Field | Default | Notes |
|-------|---------|-------|
| `enabled` | `true` | Master switch |
| `features` | `null` | `null` = monitor **all** features; a list restricts capture to those indices (validated against `d_sae` — out-of-range → `400 INVALID_FEATURE_INDEX`) |
| `history_size` | `100` | Ring-buffer capacity; existing entries are kept up to the new size |
| `top_k` | `10` | Top features highlighted per record and in WebSocket events (1–1000) |

### Toggle without reconfiguring

```bash
curl -X POST http://localhost:8000/api/monitoring/enable \
  -H "Content-Type: application/json" -d '{"enabled": false}'
```

Configuration (watched features, top-k) is preserved while disabled.

## History

```bash
curl "http://localhost:8000/api/monitoring/history?limit=5&request_id=chatcmpl-9f3a2b..."
```

```json title="data (one record)"
{
  "records": [{
    "timestamp": "2026-07-11T12:31:07Z",
    "request_id": "chatcmpl-9f3a2b...",
    "token_position": 0,
    "activations": [
      {"feature_index": 12082, "activation": 14.2},
      {"feature_index": 771, "activation": 9.8}
    ],
    "top_k": [
      {"feature_index": 12082, "activation": 14.2},
      {"feature_index": 771, "activation": 9.8}
    ]
  }],
  "total": 1
}
```

- Newest first; `limit` 1–1000 (default 50); optional `request_id` filter for correlating with completion IDs
- Each record reflects the **final forward pass** of a generation (last generated token) — not a per-token trace
- `DELETE /api/monitoring/history` clears the buffer

## Statistics

Aggregated per-feature statistics across all recorded activations since the last reset:

```bash
curl "http://localhost:8000/api/monitoring/statistics?features=12082,4517"
```

```json title="data (one feature)"
{
  "features": [{
    "feature_idx": 12082,
    "count": 231,
    "mean": 6.4113,
    "std": 4.0281,
    "min": 0.0,
    "max": 19.7734,
    "active_ratio": 0.8442
  }],
  "total_activations": 412,
  "since": "2026-07-11T12:20:00Z"
}
```

Omit `features` for all observed features. `DELETE /api/monitoring/statistics` resets.

### Top features by metric

```bash
curl -X POST http://localhost:8000/api/monitoring/statistics/top \
  -H "Content-Type: application/json" \
  -d '{"k": 10, "metric": "mean"}'
```

`metric` ∈ `mean` | `max` | `active_ratio` | `count`. The quickest answer to "what does my workload activate most?"

## Live events

With monitoring enabled, each recorded activation also emits a throttled **`monitoring:activation`** WebSocket event carrying the timestamp, request ID, and top features — see [WebSocket Events](/api/websockets).
