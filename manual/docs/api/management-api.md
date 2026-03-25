---
sidebar_position: 2
title: Management API
---

# Management API

The management API at `/api` provides endpoints for model, SAE, steering, monitoring, and profile operations.

## Models (`/api/models`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List all models |
| `/api/models` | POST | Download a model |
| `/api/models/{id}` | GET | Get model details |
| `/api/models/{id}/load` | POST | Load model to GPU |
| `/api/models/{id}/unload` | POST | Unload model from GPU |
| `/api/models/{id}/lock` | POST | Lock model (prevent unload) |
| `/api/models/{id}/unlock` | POST | Unlock model |
| `/api/models/{id}` | DELETE | Delete model from cache |
| `/api/models/preview` | POST | Preview model metadata from HuggingFace |

## SAEs (`/api/saes`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/saes` | GET | List all SAEs with attachment status |
| `/api/saes/download` | POST | Download SAE from HuggingFace |
| `/api/saes/preview` | POST | Preview SAE repository files |
| `/api/saes/{id}/attach` | POST | Attach SAE to loaded model |
| `/api/saes/{id}/detach` | POST | Detach SAE |
| `/api/saes/{id}/cancel` | POST | Cancel in-progress download |
| `/api/saes/{id}` | DELETE | Delete SAE from cache |
| `/api/saes/attachment` | GET | Get current attachment status |

## Steering (`/api/saes/steering`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/saes/steering` | GET | Get steering status and values |
| `/api/saes/steering` | POST | Set single feature value |
| `/api/saes/steering/batch` | POST | Set multiple feature values |
| `/api/saes/steering/enable` | POST | Enable steering |
| `/api/saes/steering/disable` | POST | Disable steering |
| `/api/saes/steering/{idx}` | DELETE | Remove single feature |

## Monitoring (`/api/monitoring`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/monitoring` | GET | Get monitoring state |
| `/api/monitoring/configure` | POST | Configure monitoring parameters |
| `/api/monitoring/enable` | POST | Enable/disable monitoring |
| `/api/monitoring/history` | GET | Get activation history |
| `/api/monitoring/statistics` | GET | Get feature statistics |
| `/api/monitoring/statistics/top` | POST | Get top features by metric |

## Profiles (`/api/profiles`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/profiles` | GET | List all profiles |
| `/api/profiles` | POST | Create profile |
| `/api/profiles/save-current` | POST | Save current steering as profile |
| `/api/profiles/active` | GET | Get active profile |
| `/api/profiles/{id}/activate` | POST | Activate profile |
| `/api/profiles/{id}/deactivate` | POST | Deactivate profile |
| `/api/profiles/{id}/export` | GET | Export as JSON |
| `/api/profiles/import` | POST | Import from JSON |
