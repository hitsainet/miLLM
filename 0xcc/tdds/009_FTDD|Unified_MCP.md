# Technical Design Document: Unified MCP

## miLLM Feature 9

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**References:** `009_FPRD|Unified_MCP.md` · miStudio `backend/src/mcp_server/{server.py,config.py,client.py,tools/}`

---

## 1. Executive Summary

The unified server is the existing miStudio MCP server plus: a second backend client, three new
health-gated tool categories, and a tiny health-gate helper. miLLM contributes a contract document and
one additive health field. Everything follows patterns the server already has (category gating,
bearer auth, audit logging, streamable HTTP).

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Server home | miStudio repo `backend/src/mcp_server/` (cross-repo) | User decision; reuses auth/gating/audit |
| Availability | Tools registered always; per-call HealthGate check | No tool-list churn; agents can retry |
| Gating | `millm_*` categories opt-in, refuse registration without `MILLM_API_URL` | Mirrors existing category-gating pattern |
| Envelope | `MiLLMClient` unwraps `ApiResponse{success,data,error}` | miLLM's house envelope differs from miStudio's |
| miLLM change | One additive field in detailed health | Keeps millm_status a single round-trip |

## 2. System Architecture

```
            ┌────────────────────────── unified MCP server (miStudio repo) ─────────────────────────┐
 agent ───► │ FastMCP (streamable HTTP, bearer)                                                      │
            │  categories: read/groups/steering/labeling/experiments/profiles/jobs [existing]        │
            │              millm_runtime / millm_clusters / millm_sensing [NEW, opt-in]              │
            │  HealthGate ── TTL 10 s ──► GET mistudio/api/v1/... health                             │
            │        │                └─► GET {MILLM_API_URL}/api/health                             │
            │  MiStudioClient ──► miStudio REST      MiLLMClient ──► miLLM /api/* (ApiResponse)      │
            └────────────────────────────────────────────────────────────────────────────────────────┘
```

## 3. Cross-Repo Design (miStudio `backend/src/mcp_server/`)

```python
# config.py (MOD)
VALID_CATEGORIES |= {"millm_runtime", "millm_clusters", "millm_sensing"}
# NOT added to DEFAULT_CATEGORIES — opt-in via MCP_TOOL_CATEGORIES
millm_api_url: str = Field(default="", alias="MILLM_API_URL")
# server.py registration loop: skip millm_* modules when millm_api_url is empty (log once)
```

```python
# millm_client.py (NEW) — sibling of client.py:MiStudioClient
class MiLLMClient:
    def __init__(self, base_url: str, timeout: float = 30.0): ...
    async def get(self, path: str, **params) -> Any: ...
    async def post(self, path: str, json_body: dict | None = None) -> Any: ...
    async def put(self, path: str, json_body: dict | None = None) -> Any: ...
    # All methods unwrap miLLM's envelope:
    #   {"success": true, "data": {...}}      -> return data
    #   {"success": false, "error": {...}}    -> raise BackendError(code, message)
```

```python
# health_gate.py (NEW)
class HealthGate:
    """TTL-cached per-product availability. available ⇔ HTTP 200 ∧ status != 'unhealthy'."""
    def __init__(self, ttl_s: float = 10.0): ...
    async def millm_available(self) -> tuple[bool, str]: ...     # (ok, reason)
    async def mistudio_available(self) -> tuple[bool, str]: ...

def gated(product: str):
    """Tool decorator: on unavailable, return {'unavailable': product, 'reason': ...} instead of calling."""
```

```python
# tools/millm_runtime.py (NEW) — register(mcp, millm_client, gate, settings)
millm_status()                     # GET /api/health/detailed  (incl. active_profile)
millm_list_profiles()              # GET /api/profiles
millm_activate_profile(profile_id) # POST /api/profiles/{id}/activate
millm_set_intensity(intensity)     # PUT /api/clusters/active/intensity  [documented GLOBAL]

# tools/millm_clusters.py (NEW)
millm_list_clusters()
millm_import_cluster(definition: dict | None, repo_id: str | None, filename: str | None,
                     activate: bool = False)   # inline XOR hub — exactly one source required
millm_hub_search(query, base_model, limit)
millm_activate_cluster(profile_id)
millm_export_cluster(profile_id)

# tools/millm_sensing.py (NEW)
millm_sensing_status(); millm_get_sensing_events(profile_id, limit, since)
millm_enable_sensing(profile_id); millm_disable_sensing(profile_id)
```

`SERVER_INSTRUCTIONS` addition: the cross-product loop —
`export_cluster_definition → millm_import_cluster → millm_activate_cluster → millm_set_intensity →
millm_enable_sensing → millm_get_sensing_events`.

## 4. miLLM-Side Design (this repo)

```python
# millm/api/routes/system/health.py (MOD — DetailedHealthResponse, :77 area)
class ActiveProfileInfo(BaseModel):
    id: str
    name: str
    source_kind: str            # 'manual' | 'cluster'
    intensity: float

class DetailedHealthResponse(BaseModel):
    ...existing fields...
    active_profile: ActiveProfileInfo | None = None
```
Populated from `ProfileRepository.get_active()` — one extra indexed query on the detailed endpoint only
(the basic `/api/health` used by the gate is untouched).

`docs/mcp-contract.md` (NEW): normative endpoint inventory (FPRD §5 table), envelope semantics,
health-gate contract, auth posture, versioning note (contract additions are appended, never mutated).

## 5. API Design
No new miLLM routes. Contract table in FPRD §5 is the interface.

## 6. Deployment
Server continues to deploy from the miStudio pipeline (mistudio namespace) with new env
`MILLM_API_URL=http://millm-backend.millm.svc.cluster.local:8000` (cluster-internal) and
`MCP_TOOL_CATEGORIES+=,millm_runtime,millm_clusters,millm_sensing`. Optional future co-location
manifest documented in the contract doc, not shipped.

## 7. Testing Strategy

### miLLM (this repo)
- `tests/unit/api/test_health_active_profile.py`: null when none; populated block when a cluster is active.

### miStudio (cross-repo, flagged)
- Client: envelope unwrap (success/data, success:false→BackendError, non-JSON, timeout).
- HealthGate: TTL caching, degraded=available, unhealthy/refused=unavailable with reason.
- Tools: smoke per tool with mocked MiLLMClient; import XOR-source validation.
- Topology matrix integration: both / miStudio-only (MILLM_API_URL unset ⇒ categories absent) /
  miLLM-down (tools return structured unavailable).

## 8. Risks
- Release-train coupling → contract-first (`docs/mcp-contract.md` is normative; additive-only).
- Unauthenticated miLLM management API → same-segment deployment documented; optional bearer later.
- Envelope drift → client unit tests pinned to miLLM's `ApiResponse` schema.
