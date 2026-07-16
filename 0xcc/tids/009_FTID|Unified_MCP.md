# Technical Implementation Document: Unified MCP

## miLLM Feature 9

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**References:** `009_FPRD|Unified_MCP.md` · `009_FTDD|Unified_MCP.md`

---

## 1. File Structure

```
miLLM repo (this repo):
├── millm/api/routes/system/health.py            (MOD — ActiveProfileInfo + active_profile field)
├── docs/mcp-contract.md                         (NEW — normative contract)
└── tests/unit/api/test_health_active_profile.py (NEW)

miStudio repo (CROSS-REPO — /home/x-sean/app/miStudio/backend/src/mcp_server/):
├── config.py                                    (MOD — categories + MILLM_API_URL)
├── millm_client.py                              (NEW)
├── health_gate.py                               (NEW)
├── server.py                                    (MOD — client/gate wiring, skip-when-unconfigured)
├── tools/__init__.py                            (MOD — CATEGORY_MODULES += 3)
├── tools/millm_runtime.py / millm_clusters.py / millm_sensing.py   (NEW)
└── backend/tests/unit/test_mcp_millm_*.py       (NEW)
```

## 2. Load-Bearing Implementation Points

- **miStudio server patterns to copy exactly** (all verified in the live repo):
  - Category registry: `tools/__init__.py::CATEGORY_MODULES` dict, `register(mcp, client, settings)`
    per module — the profiles category (added 2026-07-16) is the freshest template.
  - Gating: `config.py::VALID_CATEGORIES` / `enabled_categories` validation; the millm_* skip mirrors
    how empty-token refusal works (log once, continue).
  - Client: `client.py::MiStudioClient` — reuse its httpx setup/timeout/error style; only the envelope
    unwrap differs.
- **miLLM envelope** (`millm/api/schemas/common.py::ApiResponse`): `{success: bool, data: T | null,
  error: {code, message, details?} | null}` — unwrap in ONE place (client), never in tools.
- **Health field** (`millm/api/routes/system/health.py:77` DetailedHealthResponse): populate via the
  existing DI session + `ProfileRepository.get_active()` (profile_repository.py:115). Keep the BASIC
  `/api/health` untouched — it's the gate's hot path.
- **repo_id slashes**: millm_hub_search/import pass repo_id through — the miLLM route is `{repo_id:path}`;
  client must not double-encode.

## 3. Key Implementations

```python
# miStudio: millm_client.py — envelope unwrap core
class MiLLMClient:
    async def _request(self, method: str, path: str, **kw) -> Any:
        resp = await self._http.request(method, f"{self._base}{path}", **kw)
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict) or "success" not in body:
            return body                     # non-envelope endpoints (none expected; defensive)
        if body.get("success"):
            return body.get("data")
        err = body.get("error") or {}
        raise BackendError(err.get("code", "MILLM_ERROR"),
                           err.get("message", "miLLM request failed"))
```

```python
# miStudio: health_gate.py
class HealthGate:
    def __init__(self, millm_url: str, mistudio_client, ttl_s: float = 10.0):
        self._cache: dict[str, tuple[float, bool, str]] = {}
    async def check(self, product: str) -> tuple[bool, str]:
        now = time.monotonic()
        hit = self._cache.get(product)
        if hit and now - hit[0] < self._ttl:
            return hit[1], hit[2]
        ok, reason = await self._probe(product)     # millm: GET /api/health, 3 s timeout
        self._cache[product] = (now, ok, reason)
        return ok, reason
# unavailable result convention (uniform across all millm_* tools):
# {"unavailable": "millm", "reason": "connection refused (http://…/api/health)"}
```

```python
# miStudio: tools/millm_clusters.py — import tool source validation
@mcp.tool()
async def millm_import_cluster(definition: dict | None = None,
                               repo_id: str | None = None, filename: str | None = None,
                               activate: bool = False) -> Any:
    """Import a mistudio.cluster-definition/v1 into miLLM (inline JSON XOR hub repo+file)."""
    if bool(definition) == bool(repo_id):
        return {"error": "provide exactly one source: `definition` OR `repo_id`+`filename`"}
    ok, reason = await gate.check("millm")
    if not ok:
        return {"unavailable": "millm", "reason": reason}
    if definition:
        return await millm.post(f"/api/clusters/import?activate={str(activate).lower()}",
                                json_body=definition)
    return await millm.post("/api/clusters/hub/import",
                            json_body={"repo_id": repo_id, "filename": filename,
                                       "activate": activate})
```

```python
# miLLM: health.py active_profile population
active = await profile_repo.get_active()
detailed.active_profile = (ActiveProfileInfo(
    id=active.id, name=active.name,
    source_kind=active.source_kind, intensity=active.intensity)
    if active else None)
```

## 4. Implementation Pitfalls

1. **Do not unregister tools on outage** — the gate returns structured unavailable; MCP clients cache
   tool lists and churn confuses agents.
2. **Basic vs detailed health**: the gate polls `/api/health` (cheap); `millm_status` calls
   `/api/health/detailed`. Don't swap them.
3. **`degraded` is AVAILABLE** — miLLM with no model loaded must still accept imports and report status.
4. **Envelope unwrap belongs in the client only** — tools returning raw envelopes leak
   `{"success": true, ...}` wrappers into agent results.
5. **Category names are load-bearing config** — compose/k8s `MCP_TOOL_CATEGORIES` strings must match
   `VALID_CATEGORIES` exactly or the server refuses startup (existing behavior).
6. **Cross-repo commits**: miStudio-side work commits in the miStudio repo with its own conventions;
   the miLLM FTASKS tracks them as flagged tasks, not as miLLM commits.

## 5. Config

- miStudio server: `MILLM_API_URL` (empty ⇒ millm_* skipped), `MCP_TOOL_CATEGORIES` gains the three
  names where enabled (compose + k8s env, both repos' manifests documented in the contract doc).
- miLLM: none.
