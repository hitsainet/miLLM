---
sidebar_position: 8
title: Endpoint Index
---

# Management API — Endpoint Index

A single-page index of every `/api` endpoint. Each area has a dedicated page with request/response examples — follow the section links.

All endpoints return the [management envelope](/api/overview#the-management-envelope): `{success, data, error}`.

## Models (`/api/models`) — [full reference](/api/models)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List all models |
| `/api/models` | POST | Download a model (HF or local path) |
| `/api/models/preview` | POST | Preview model metadata & memory estimates |
| `/api/models/{id}` | GET | Get model details |
| `/api/models/{id}` | DELETE | Delete model from disk |
| `/api/models/{id}/load` | POST | Load model to GPU |
| `/api/models/{id}/unload` | POST | Unload model from GPU |
| `/api/models/{id}/lock` | POST | Lock model (prevent unload) |
| `/api/models/{id}/unlock` | POST | Unlock model |
| `/api/models/{id}/cancel` | POST | Cancel in-progress download |

## SAEs (`/api/saes`) — [full reference](/api/saes)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/saes` | GET | List SAEs + attachment status |
| `/api/saes/download` | POST | Download SAE from HuggingFace |
| `/api/saes/preview` | POST | Preview SAE repository files |
| `/api/saes/attachment` | GET | Attachment status (incl. `steering_apply_count`) |
| `/api/saes/{id}` | GET / DELETE | Get / delete an SAE |
| `/api/saes/{id}/compatibility` | GET | Dry-run compatibility check |
| `/api/saes/{id}/attach` | POST | Attach to loaded model |
| `/api/saes/{id}/detach` | POST | Detach |
| `/api/saes/{id}/cancel` | POST | Cancel in-progress download |
| `/api/saes/monitoring` | POST | Configure monitoring (alias of `/api/monitoring/configure`) |

## Steering (`/api/saes/steering`) — [full reference](/api/saes#steering)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/saes/steering` | GET | Get steering status and values |
| `/api/saes/steering` | POST | Set single feature value |
| `/api/saes/steering/batch` | POST | Set multiple feature values |
| `/api/saes/steering/enable` | POST | Enable steering |
| `/api/saes/steering/disable` | POST | Disable steering (keep values) |
| `/api/saes/steering/{idx}` | DELETE | Remove single feature |
| `/api/saes/steering` | DELETE | Clear all values & disable |

## Monitoring (`/api/monitoring`) — [full reference](/api/monitoring)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/monitoring` | GET | Get monitoring state |
| `/api/monitoring/configure` | POST | Configure (features, history size, top-k) |
| `/api/monitoring/enable` | POST | Toggle without reconfiguring |
| `/api/monitoring/history` | GET / DELETE | Activation history / clear |
| `/api/monitoring/statistics` | GET / DELETE | Feature statistics / reset |
| `/api/monitoring/statistics/top` | POST | Top features by metric |

## Profiles (`/api/profiles`) — [full reference](/api/profiles)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/profiles` | GET / POST | List / create |
| `/api/profiles/save-current` | POST | Save live steering as a profile |
| `/api/profiles/active` | GET | Get active profile |
| `/api/profiles/{id}` | GET / PUT / DELETE | Get / update / delete |
| `/api/profiles/{id}/activate` | POST | Activate (replaces current steering) |
| `/api/profiles/{id}/deactivate` | POST | Deactivate |
| `/api/profiles/{id}/export` | GET | Export as JSON |
| `/api/profiles/import` | POST | Import from JSON |

## Sensing (`/api/sensing`) — cluster co-activation events ([concepts](/features/clusters#co-activation-sensing))

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sensing/status` | GET | Armed state, threshold mode, per-request overhead, retention limits, and `enabled_clusters` (persistent intent, distinct from `armed`) |
| `/api/sensing/events` | GET | Events newest-first (`?profile_id=`, `?limit=`, `?since=`); age-expired rows are pruned on read |
| `/api/sensing/events/{id}` | GET | Event detail incl. the ±K context window (404 when pruned/cleared) |
| `/api/sensing/events` | DELETE | Clear events (`?profile_id=` scopes to one cluster) |
| `/api/sensing/{profile_id}/enable` | POST | Enable sensing for a cluster (arms live when that cluster is active with an SAE attached) |
| `/api/sensing/{profile_id}/disable` | POST | Disable sensing (disarms live) |

## Health & operations (`/api/health`) — [full reference](/api/overview#health--operations-endpoints)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Liveness |
| `/api/health/ready` | GET | Readiness |
| `/api/health/detailed` | GET | Component breakdown |
| `/api/health/inference` | GET | Active backend & capabilities |
| `/api/health/metrics` | GET | App metrics |
| `/api/health/metrics/prometheus` | GET | Prometheus format |
| `/api/health/circuits` | GET | Circuit breakers |
| `/api/health/circuits/{name}/reset` | POST | Reset a breaker |
| `/api/health/version` | GET | Version |
