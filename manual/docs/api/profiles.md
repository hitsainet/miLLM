---
sidebar_position: 6
title: Profiles API
---

# Profiles API

Steering profile management at `/api/profiles`. Profiles store a named steering configuration (feature → strength map) plus the model/SAE context it was built for. See [Profiles](/features/profiles) for concepts and the per-request usage pattern.

## List / get

```bash
curl http://localhost:8000/api/profiles           # all profiles
curl http://localhost:8000/api/profiles/active    # currently active profile (or null)
curl http://localhost:8000/api/profiles/prof_1a2b3c
```

```json title="data (one profile)"
{
  "id": "prof_1a2b3c4d5e6f",
  "name": "dogs-40",
  "description": "GemmaScope L12/16k feature 12082 @ 40",
  "model_id": "google/gemma-2-2b",
  "sae_id": "sae_a1b2c3...",
  "layer": 12,
  "steering": {"12082": 40.0},
  "is_active": false,
  "created_at": "2026-07-11T12:40:00Z",
  "updated_at": "2026-07-11T12:40:00Z"
}
```

Note: `steering` keys are stringified feature indices (JSON object keys are strings).

## Create

```bash
curl -X POST http://localhost:8000/api/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "name": "dogs-40",
    "description": "dogs feature at moderate strength",
    "steering": {"12082": 40.0, "4517": -10.0},
    "sae_id": null, "model_id": null, "layer": null
  }'
```

Names must be unique (`409 PROFILE_ALREADY_EXISTS`). An **empty** `steering` map is valid — activating such a profile is a supported way to switch to a clean unsteered state.

### Save the current steering as a profile

```bash
curl -X POST http://localhost:8000/api/profiles/save-current \
  -H "Content-Type: application/json" \
  -d '{"name": "session-2026-07-11", "description": "whatever is dialed in right now"}'
```

Captures the live steering values plus the attached SAE/layer context. Requires an attached SAE.

## Update

```bash
curl -X PUT http://localhost:8000/api/profiles/prof_1a2b3c \
  -H "Content-Type: application/json" \
  -d '{"steering": {"12082": 55.0}}'
```

Partial updates; `steering`, when provided, **replaces** the map.

## Activate / deactivate

```bash
curl -X POST http://localhost:8000/api/profiles/prof_1a2b3c/activate
curl -X POST http://localhost:8000/api/profiles/prof_1a2b3c/deactivate
```

Activation **replaces** the current steering values with the profile's and enables steering:

- Requires an attached SAE when the profile has steering values (`400 SAE_NOT_ATTACHED` otherwise)
- Indices are validated against the attached SAE first — out-of-range indices fail with `400` BEFORE any live steering is touched (nothing is partially applied). Values are scaled by the profile's intensity (λ) and clamped to ±200 at apply time
- Activating an empty-steering profile clears existing steering
- Deactivation (optionally `?clear_steering=false`) clears the steering values by default

## Per-request activation

Any saved profile can be applied for a **single** chat-completion request via the `profile` parameter on [`POST /v1/chat/completions`](/api/openai-compatible#per-request-steering-with-profile) — no global state change, previous steering restored afterwards.

## Export / import

```bash
curl http://localhost:8000/api/profiles/prof_1a2b3c/export > dogs-40.json
curl -X POST http://localhost:8000/api/profiles/import \
  -H "Content-Type: application/json" \
  --data-binary @dogs-40.json
```

The export format includes model/SAE provenance (see [Profiles](/features/profiles#import--export)). Imports are validated — malformed entries return `400 INVALID_PROFILE_FORMAT` rather than being silently dropped. Feature indices are only meaningful with the same SAE the profile was built for.

## Delete

```bash
curl -X DELETE http://localhost:8000/api/profiles/prof_1a2b3c
```

Deleting the active profile deactivates it first.

## Common errors

| Code | Status | When |
|------|--------|------|
| `PROFILE_NOT_FOUND` | 404 | Unknown ID or (for per-request use) unknown name |
| `PROFILE_ALREADY_EXISTS` | 409 | Duplicate name on create/rename |
| `INVALID_PROFILE_FORMAT` | 400 | Malformed import payload |
| `SAE_NOT_ATTACHED` | 400 | Activate/save-current without an attached SAE |
| `INVALID_FEATURE_INDEX` | 400 | Profile steering doesn't fit the attached SAE |
