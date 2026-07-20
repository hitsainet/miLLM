# Technical Implementation Document: Circuit Import, Slice-Fallback & Evidence Ladder

## miLLM Feature 13

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Core (Increment: Circuit Runtime)
**References:** `013_FPRD|Circuit_Import.md` · `013_FTDD|Circuit_Import.md` · `docs/mcp-contract.md` (v1.1)

---

## 1. File Structure

```
millm/
├── api/
│   ├── schemas/circuit.py                        (NEW — v1 circuit contract mirror + API DTOs)
│   ├── routes/management/circuits.py             (NEW — /api/circuits router)
│   └── dependencies.py                           (MOD — CircuitServiceDep)
├── services/
│   └── circuit_service.py                        (NEW — import, per-SAE compat, activate, slice-fallback, export)
├── core/circuit_evidence.py                      (NEW — EvidenceRung + RUNG_LANGUAGE + circuit_rung; single vocabulary)
├── db/
│   ├── models/circuit.py                         (NEW — circuits table model + serveable property)
│   ├── repositories/circuit_repository.py        (NEW — CRUD + single-active guard)
│   └── migrations/versions/011_add_circuits_table.py  (NEW — next free after 010 on disk)
docs/schemas/circuit-definition-v1.json           (NEW — vendored, frozen, copied from miStudio)
admin-ui/src/
├── pages/CircuitsPage.tsx                        (NEW)  + App.tsx/Sidebar.tsx/pages/index.ts (MOD)
├── components/circuits/{CircuitCard,CircuitImportDialog,CircuitActivateControl}.tsx (NEW)
├── services/circuits.ts, hooks/useCircuits.ts, types/circuits.ts (NEW)
tests/unit/{api,core,services}/test_circuit_*.py  (NEW)
tests/integration/test_circuit_import_workflow.py (NEW)
```

## 2. Load-Bearing Implementation Points (verified against live code + contracts)

- **Reuse the cluster import path for slices, don't fork it:** `ClusterService.import_definition`
  (`millm/services/cluster_service.py`) + `POST /api/clusters/import` already validate, compat-check,
  materialize a profile, and activate a `cluster-definition/v1`. A per-layer circuit slice IS a valid
  `cluster-definition/v1` — feed it straight in. `CircuitService._activate_slice_fallback` calls
  `ClusterService.import_definition(slice_doc, activate=True)` per bound layer. No new fallback machinery.
- **Vendored schema + sync test mirror the cluster pattern exactly:** copy
  `miStudio/docs/schemas/circuit-definition-v1.json` → `millm/docs/schemas/circuit-definition-v1.json`
  (frozen); mirror it in `api/schemas/circuit.py`; add `tests/unit/api/test_circuit_schema_sync.py`
  cloned from `tests/unit/api/test_cluster_schema_sync.py` (pydantic ⇄ JSON schema field/constraint parity).
- **Rung vocabulary is a SINGLE source module:** `millm/core/circuit_evidence.py` mirrors miStudio's
  `evidence_ladder.py` verbatim — `EvidenceRung` (0 MINED, 1 ATTRIBUTION_SUPPORTED, 2 CAUSALLY_VALIDATED,
  3 FAITHFULNESS_TESTED), `RUNG_LANGUAGE` (0→"associated", 1→"suggested (attribution-supported)",
  2→"causally validated (edge)", 3→"faithfulness-tested (circuit)"), `circuit_rung = MIN(edges)` (empty ⇒ 0).
  Every surface (service, route, UI type) reads `rung_language` from here; no per-surface phrasing.
- **Full serving delegates to Feature 12:** `MultiSAEService.apply_circuit(circuit)` (Feature 12) applies
  each member through its own layer's SAE at authored per-layer budgets under one λ. Feature 13 only
  gates (all SAEs bound?) and delegates — no steering math here.
- **Envelope + route style:** all management routes return `ApiResponse` (`millm/api/schemas/common.py`)
  via `ApiResponse.ok(data)` / `.fail(code, message)`; mirror `routes/management/clusters.py`
  (prefix `/api/circuits`, tags `["circuits"]`). The export route returns the RAW document (no envelope),
  exactly like `GET /api/clusters/{id}/export`.
- **Router registration:** `millm/api/routes/__init__.py::register_routes` — add
  `from .management import circuits` + `app.include_router(circuits.router)`.
- **Single-active spans three kinds now:** activation must deactivate any active profile OR cluster OR
  circuit. The `circuits.uq_circuits_active` partial index guards circuit↔circuit; the service coordinates
  circuit↔profile/cluster via `ProfileRepository.set_active(None)` before circuit activation and vice versa.

## 3. Key Implementations

```python
# millm/core/circuit_evidence.py — the one vocabulary (import everywhere; never re-phrase)
CAUSAL_MIN_RUNG = 2  # the word "causal" is FORBIDDEN below this rung (copy-audit enforced)

def circuit_rung(edges) -> "EvidenceRung":
    """Circuit rung = MIN over edges; empty edges ⇒ MINED (0)."""
    return EvidenceRung(min((int(e["rung"]) for e in edges), default=0))
```

```python
# millm/services/circuit_service.py — per-SAE compat + serveable
def _assess_sae(self, ref, attached: dict[int, "AttachedSAE"]) -> dict:
    at = attached.get(ref.layer)
    if at is None:
        return {"layer": ref.layer, "state": "unbound", "reason": "no SAE attached at layer"}
    if ref.n_features is not None and at.d_sae != ref.n_features:
        return {"layer": ref.layer, "state": "block",
                "code": "INCOMPATIBLE_FEATURE_SPACE",
                "reason": f"n_features {ref.n_features} != attached d_sae {at.d_sae}"}
    warns = []
    if ref.mistudio_sae_id and at.model_id and ref.model and ref.model != at.model_id:
        warns.append("model differs")
    return {"layer": ref.layer, "state": "warn" if warns else "bind", "warnings": warns}

def _serveable(self, per_sae) -> bool:
    return all(s["state"] in ("bind", "warn") for s in per_sae)
```

```python
# millm/services/circuit_service.py — activation gate (rung → SAE set → delegate)
async def activate(self, circuit_id, *, acknowledge_unvalidated=False):
    circ = await self.circuit_repo.get(circuit_id)
    if circ is None:
        return ActivationResult.fail("CIRCUIT_NOT_FOUND")           # 404
    if circ.rung < CAUSAL_MIN_RUNG and not acknowledge_unvalidated:
        return ActivationResult.fail("UNVALIDATED_CIRCUIT")         # 200 + success:false
    per_sae = self._per_sae_now(circ)
    missing = [s for s in per_sae if s["state"] in ("unbound", "block")]
    await self._deactivate_any_active()                            # cross-kind single-active
    if not missing:
        return await self.multisae.apply_circuit(circ)             # serving_mode="full"
    return await self._activate_slice_fallback(circ, per_sae)      # serving_mode="slice_fallback"
```

```python
# millm/services/circuit_service.py — slice fallback via the Feature 8 cluster path (unchanged)
async def _activate_slice_fallback(self, circ, per_sae):
    bound = [s["layer"] for s in per_sae if s["state"] in ("bind", "warn")]
    if not bound:
        return ActivationResult.fail("SAE_SET_INCOMPLETE",
                                     details={"missing": [s for s in per_sae if s["layer"] not in bound]})
    for layer in bound:
        slice_doc = self._to_layer_slice(circ.circuit_meta, layer)  # valid cluster-definition/v1
        await self.cluster_service.import_definition(
            ClusterDefinitionV1.model_validate(slice_doc), activate=True)
    await self.circuit_repo.set_serving_mode(circ.id, "slice_fallback", bound_layers=bound)
    return ActivationResult.ok(serving_mode="slice_fallback", bound_layers=bound)
```

```python
# millm/api/routes/management/circuits.py — import route skeleton
@router.post("/import")
async def import_circuit(payload: dict = Body(...),
                         on_conflict: str = Query("rename", pattern="^(rename|fail)$"),
                         activate: bool = Query(False),
                         acknowledge_unvalidated: bool = Query(False),
                         service: CircuitServiceDep = None) -> ApiResponse:
    if len(json.dumps(payload)) > 1_048_576:
        return ApiResponse.fail(code="PAYLOAD_TOO_LARGE", message="Import exceeds 1 MB")
    if payload.get("kind") != "mistudio.circuit-definition":
        return ApiResponse.fail(code="UNKNOWN_KIND",
                                message=f"kind {payload.get('kind')!r} is not a circuit document")
    defn = CircuitDefinitionV1.model_validate(payload)     # major-version + caps + no-paths
    result = await service.import_definition(
        defn, on_conflict=on_conflict, activate=activate,
        acknowledge_unvalidated=acknowledge_unvalidated)
    return ApiResponse.ok(result)
```

```typescript
// admin-ui/src/services/circuits.ts — client shape
export const circuitsApi = {
  list: (p?) => request<CircuitListResponse>('/circuits' + qs(p)),
  active: () => request<ActiveCircuit | null>('/circuits/active'),
  import: (payload: unknown, opts?) => request('/circuits/import', {method:'POST', body: payload, ...}),
  activate: (id, ack?) => request(`/circuits/${id}/activate?acknowledge_unvalidated=${!!ack}`, {method:'POST'}),
  deactivate: (id) => request(`/circuits/${id}/deactivate`, {method:'POST'}),
  setIntensity: (intensity) => request('/circuits/active/intensity', {method:'PUT', body:{intensity, reapply:true}}),
  export: (id) => request<CircuitDefinitionV1>(`/circuits/${id}/export`),
};
```

## 4. Implementation Pitfalls

1. **Never re-phrase the rung** — all user/agent-facing rung text comes from `rung_language` /
   `RUNG_LANGUAGE`. Hand-writing "causal", "validated", etc. per surface is the exact overclaim the
   copy-audit test forbids. Below rung 2 the string "causal" must never render.
2. **A slice is NOT the circuit** — every slice-fallback render must carry the partial-rendering marker
   (name suffix ` [L{n} slice]` + `provenance.source_note`) and `serving_mode: "slice_fallback"`. Never
   surface a bound-layer projection as the whole circuit.
3. **Don't fork the cluster path for slices** — `_activate_slice_fallback` calls
   `ClusterService.import_definition(...)`; a per-layer slice is an ordinary `cluster-definition/v1`.
   Duplicating validation/materialization is a divergence bug waiting to happen.
4. **Gate the SAE set BEFORE delegating to Feature 12** — `apply_circuit` must only ever see a fully-bound
   set. A single unbound/block layer routes to slice-fallback, never a partial full-serve.
5. **Schema is FROZEN** — never "fix" the vendored circuit JSON; the sync test exists to catch accidental
   divergence of the pydantic mirror. Tier-2.5 nullable fields (position/attention) must survive
   round-trip via `extra="allow"`.
6. **rung is stored int, rendered via the module** — persist `rung` (0–3) on the row; compute
   `rung_language` at the boundary. Do not persist the phrase (it would drift from the vocabulary source).
7. **Single-active is cross-kind now** — activating a circuit must deactivate any active manual profile
   or cluster (and vice versa). The partial unique index only guards circuit↔circuit.
8. **`export` returns the raw document** — no `ApiResponse` wrapper (matches the cluster export; the
   response IS the artifact).

## 5. Config Additions (millm/core/config.py)

```python
CIRCUIT_HUB_TAG: str = "mistudio-circuit-definition"   # hub search (owned by later features; declared here)
CIRCUIT_MAX_LAYERS: int = 16
CIRCUIT_MAX_EDGES: int = 200
CIRCUIT_MAX_MEMBERS_PER_LAYER: int = 20
```
