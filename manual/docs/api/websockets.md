---
sidebar_position: 7
title: WebSocket Events
---

# WebSocket Events

miLLM streams real-time events over **Socket.IO** mounted at the standard `/socket.io` path on the same port as the HTTP APIs. The Admin UI is the primary consumer, but any Socket.IO client can subscribe — useful for dashboards, experiment loggers, or progress bars in scripts.

```python
import socketio

sio = socketio.Client()
sio.connect("http://localhost:8000")

@sio.on("monitoring:activation")
def on_activation(data):
    print("features:", data["features"])

@sio.on("system:metrics")
def on_metrics(data):
    print("gpu:", data)

sio.wait()
```

## Event catalog

### System

| Event | Payload highlights | Notes |
|-------|--------------------|-------|
| `system:metrics` | GPU utilization, VRAM used/total, temperature | Periodic broadcast; also sent immediately on `system:join` |

### Model lifecycle

| Event | Payload highlights |
|-------|--------------------|
| `model:download:progress` | `model_id`, `percent`, bytes progress |
| `model:download:complete` | `model_id` |
| `model:download:error` | `model_id`, `error` |
| `model:load:progress` | `model_id`, stage/percent |
| `model:load:complete` | `model_id` |
| `model:load:error` | `model_id`, `error` |
| `model:unload:complete` | `model_id` |

### SAE lifecycle

| Event | Payload highlights |
|-------|--------------------|
| `sae:download:progress` | `sae_id`, `percent` |
| `sae:download:complete` | `sae_id` |
| `sae:download:error` | `sae_id`, `error` |
| `sae:attached` | `sae_id`, `layer` |
| `sae:detached` | `sae_id` |

### Steering & monitoring

| Event | Payload highlights | Notes |
|-------|--------------------|-------|
| `steering:update` | `enabled`, `values` (feature → strength), `active_count` | Emitted on every steering change from any source |
| `monitoring:state` | monitoring configuration | On configure/toggle |
| `monitoring:activation` | `timestamp`, `request_id`, `features` (top-k `[index, activation]` pairs), `position` | Throttled (~10/sec max); one per recorded generation |
| `sensing:event` | One persisted **cluster** co-activation event: `id`, `profile_id`, `phase`, span, `fired_members`, `score`, `summary` — **context text is never sent over WS** (fetch `/api/sensing/events/{id}`) | Throttled (max 5 per request flush, min 100 ms between flushes); the DB is complete regardless |
| `circuit:sensing:event` | One persisted **circuit edge** firing (Feature 15): `id`, `circuit_id`, `request_id`, `phase`, `edge_key`, `token_lag`, `edge_rung`, `summary` — **context text is never sent over WS** (fetch `/api/circuit-sensing/events/{id}`) | Same throttle discipline; the payload is built with `include_context=False`, so no decoded prompt text leaves the box |

## Patterns

**Experiment logging** — subscribe to `monitoring:activation` and `steering:update` and append both to your run log; you get a timeline of *what the steering was* and *what fired* without polling.

**Progress bars** — the `*:download:progress` events carry percentages; downloads can be driven headlessly via the [Models](/api/models)/[SAEs](/api/saes) APIs with progress rendered from these events.

**Reconnection** — all events are fire-and-forget state deltas. After a reconnect, resync authoritative state from the REST endpoints (`/api/saes/attachment`, `/api/saes/steering`, `/api/monitoring`) rather than assuming you saw every event.
