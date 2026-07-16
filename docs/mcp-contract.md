# miLLM ↔ Unified MCP Server Contract

**Status:** Normative for miLLM Feature 9 (Unified MCP). **Version:** 1.0 (2026-07-16)
**Consumer:** the unified MCP server that ships in the miStudio repo
(`backend/src/mcp_server/`), exposing `millm_runtime` / `millm_clusters` /
`millm_sensing` tool categories against a miLLM deployment.

## 1. Versioning rule

This contract is **additive-only**: miLLM may add endpoints, response fields, and
error codes; it must not rename or remove anything listed here, change field
types, or change status-code semantics without a new contract version. The MCP
server must tolerate unknown fields everywhere.

## 2. Response envelope

Every management endpoint (everything under `/api/`) returns:

```json
{ "success": true,  "data": <payload>, "error": null }
{ "success": false, "data": null, "error": { "code": "UPPER_SNAKE", "message": "…", "details": { } } }
```

- **The envelope is authoritative for machine handling** — unwrap in the
  client, never in tools. HTTP status *usually* mirrors the error class
  (400/404/409/422/503), but NOT always: the cluster import route returns
  some refusals as **HTTP 200 with `success: false`** (`PAYLOAD_TOO_LARGE`,
  `UNKNOWN_KIND`, contract `VALIDATION_ERROR`, `NO_ACTIVE_CLUSTER`) — house
  style for handler-level failures. Never branch on status alone.
- Non-envelope endpoints: `GET /api/clusters/{id}/export` returns the RAW
  portable cluster-definition document (the response *is* the artifact);
  `GET /api/health`, `/api/health/detailed`, and `/api/health/ready` return
  bare DTOs (no envelope); `/v1/*` endpoints speak the OpenAI error shape.
- FastAPI *request-validation* failures on management routes (bad query/body
  types, out-of-range values) return the default 422
  `{"detail": [...]}` shape — no envelope, no `error.code`. Clients should
  validate tool arguments before calling.

## 3. Health-gate contract

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /api/health` | **Gate hot path.** Cheap liveness: `{status, version, timestamp, uptime_seconds}` | No DB read. Poll ≤ 1/10 s (gate TTL). 3 s timeout recommended |
| `GET /api/health/detailed` | One-call status for `millm_status` | Includes `model_loaded`, `model_name`, `sae_attached`, `sae_id`, `inference`, and `active_profile` |

**`active_profile`** (added for this contract):
`{id, name, source_kind: "manual"|"cluster", intensity, sensing_enabled} | null`.

**Gate semantics:** available ⇔ **2xx AND `status != "unhealthy"`** —
`degraded` IS available (miLLM with no model loaded must still accept cluster
imports and report status). Connection failure, timeout, non-2xx (including
3xx redirects, which the gate does not follow), or a 2xx body reporting
`status: "unhealthy"` (reserved; today's liveness endpoint only ever reports
healthy) mark the product unavailable; tools then return a structured
`{"unavailable": "millm", "reason": …}` result and are **never unregistered**
(MCP clients cache tool lists).

## 4. Endpoint inventory consumed by the MCP tools

### `millm_runtime`
| Tool | Endpoint |
|---|---|
| `millm_status` | `GET /api/health/detailed` |
| `millm_list_profiles` | `GET /api/profiles` |
| `millm_activate_profile` | `POST /api/profiles/{id}/activate` |
| `millm_deactivate_profile` | `POST /api/profiles/{id}/deactivate` |
| `millm_set_intensity` | `PUT /api/clusters/active/intensity` (`{intensity, reapply}`) |

### `millm_clusters`
| Tool | Endpoint |
|---|---|
| `millm_list_clusters` | `GET /api/clusters` |
| `millm_import_cluster` (inline) | `POST /api/clusters/import?activate=&on_conflict=` (body = raw v1 document; `on_conflict`: `rename` (default) \| `fail`) |
| `millm_import_cluster` (hub) | `POST /api/clusters/hub/import` (`{repo_id, filename, revision?, activate?, on_conflict?}`) |
| `millm_hub_search` | `GET /api/clusters/hub/search?q=&base_model=&limit=` |
| `millm_activate_cluster` | `POST /api/clusters/{id}/activate` |
| `millm_deactivate_cluster` | `POST /api/clusters/{id}/deactivate` |
| `millm_export_cluster` | `GET /api/clusters/{id}/export` (raw document — no envelope) |

### `millm_sensing`
| Tool | Endpoint |
|---|---|
| `millm_sensing_status` | `GET /api/sensing/status` |
| `millm_sensing_events` | `GET /api/sensing/events?profile_id=&limit=&since=` (list rows include context fields) |
| `millm_sensing_enable` / `_disable` | `POST /api/sensing/{profile_id}/enable` / `/disable` |

Notes:
- `repo_id` contains a slash; miLLM's hub routes declare `{repo_id:path}` —
  clients must NOT URL-encode the slash.
- Cluster activation enforces the declared-feature-space gate server-side
  (422 with a human-readable reason); imports of incompatible definitions
  succeed as **unbound** and refuse only at activation.
- Payload caps: import documents ≤ 1 MB, ≤ 20 members/definition,
  ≤ 50 definitions/bundle.

## 5. Error codes the MCP client must map

`VALIDATION_ERROR` (422), `PROFILE_NOT_FOUND` (404), `MODEL_NOT_LOADED`
(**400** on management routes; the 503 mapping applies only to `/v1/*`),
`SAE_NOT_ATTACHED` (409/400 family), `PAYLOAD_TOO_LARGE` (200+envelope),
`UNKNOWN_KIND` (200+envelope), `HUB_UNAVAILABLE` (503, circuit open),
`NO_ACTIVE_CLUSTER` (200+envelope), `INTERNAL_ERROR` (500).
(`SENSING_EVENT_NOT_FOUND` (404) exists on the event-detail route, which no
MCP tool currently consumes.) Unknown codes: surface `error.message`
verbatim — messages are written to be user/agent-safe.

## 6. Auth posture & deployment guidance (Task 1.3)

miLLM's management API is **unauthenticated by design** in the current
release. The supported topology is **same-network-segment deployment**: the
MCP server and miLLM must be reachable only on a trusted segment (cluster
namespace / LAN); do not expose `/api/*` to untrusted networks. The MCP
server's own bearer-token auth protects the agent-facing surface; it does
NOT add auth to miLLM. If a future miLLM release adds a management bearer
token, it will arrive as an additive `Authorization` requirement announced in
a new contract version; the client should already send a configurable
optional bearer header to be forward-compatible.

Deployment wiring (miStudio side): set `MILLM_API_URL` (e.g.
`http://millm-backend.millm.svc.cluster.local:8000`) and opt in via
`MCP_TOOL_CATEGORIES=...,millm_runtime,millm_clusters,millm_sensing`. With
`MILLM_API_URL` unset, the millm_* categories are skipped at registration
(logged once) — miStudio-only deployments are unaffected.

## 7. Cross-product agent flow (reference)

```
miStudio: export_cluster_definition(profile_id)     → v1 document
miLLM:    millm_import_cluster(definition=…, activate=true)
miLLM:    millm_set_intensity(1.2)
miLLM:    millm_sensing_enable(profile_id)
miLLM:    millm_sensing_events(profile_id, limit=20)
```
