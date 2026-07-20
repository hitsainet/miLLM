# Technical Design Document: Circuit-Aware OWUI Dial

## miLLM Feature 14

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Runtime)
**References:** `BRD-MILLM-CIRCUITS-001.md` · `000_PPRD|miLLM.md` (v1.2, FR-14.x) · `000_PADR|miLLM.md` (v1.2) · `docs/mcp-contract.md` (v1.1) · `010_FTDD|OWUI_Cluster_Dial.md` (the machinery reused)

---

## 1. Executive Summary

The circuit dial rides the exact machinery Feature 10 proved: request field → serial routing → apply
inside the queue semaphore → restore in `finally`. The generalization is small and precise: when the
active steering target is a **circuit** (not a cluster), one resolved λ scales EVERY layer's authored
strengths together (`_apply_request_steering` already iterates a members dict — the circuit path feeds it
all layers at once), and the OWUI filter gains a best-effort **circuit-status probe** so its status line
names the active circuit and surfaces its evidence `rung_language` verbatim. No new DB table, no
migration, no new management route — the global circuit dial (`PUT /api/circuits/active/intensity`) is
Features 12/13's.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Transport | Reuse Feature 10's `steering_intensity` field + validator verbatim | `extra="ignore"` both directions; no new schema |
| Scaling | One λ scales ALL circuit layers together (per-layer strengths × λ, clamped ±200 per member) | BR-006 locked; `_apply_request_steering` already iterates the members dict |
| Resolution | Symbolic λ resolved server-side against the ACTIVE circuit's intensity semantics; cluster fallback when the active target is a cluster | Filter stays semantics-ignorant |
| Rung surfacing | Filter probes `GET /api/circuits/active` and renders `rung_language` verbatim; rung<2 → "unvalidated" | §4a: never "causal" below rung 2; render, never re-phrase |
| Header echo | `X-miLLM-Circuit-Rung` beside the reused `X-miLLM-Steering-Intensity` | Observability without a body-shape change |
| No new route | Global dial is `PUT /api/circuits/active/intensity` (Features 12/13) | This feature is per-request + filter only |

## 2. System Architecture

```
 OWUI chat ──► Filter.inlet:
                 (best-effort) GET /api/circuits/active → {name, rung, rung_language, serving_mode}
                 body["steering_intensity"]="max"
                 status: "miLLM circuit «Induction-L10→L13» max — rung 1: suggested (attribution-supported) · UNVALIDATED"
                        │
                        ▼  POST /v1/chat/completions
      ┌─────────────────────────────────────────────────────────────────┐
      │ InferenceService (serial queue semaphore) — Feature 10 path      │
      │  resolve_request_intensity(request) → λ (circuit range | config) │
      │  _apply_request_steering(profile?, λ):                           │
      │    base = ACTIVE circuit's per-layer members (all layers)        │
      │    λ==0 → save + enable_steering(False)                          │
      │    else → save + set_steering_batch({(layer,i): clamp(s·λ)})     │
      │  … generate …                                                    │
      │  finally: _restore_request_profile(saved)  (incl. disconnect)    │
      └─────────────────────────────────────────────────────────────────┘
```

The circuit path differs from the cluster path only in the **base**: the active-circuit members span
multiple (layer, feature) keys; every one is scaled by the same λ. The save/restore shape is identical
(`{values, enabled}` over the attached multi-SAE steering state).

## 3. Data Model

**No new DB table. No migration.** Reads Features 12/13's active-circuit state (per-layer budgets, rung,
serving_mode). The per-request λ is transient (never persisted); the persisted global λ is the circuit
row's, owned by Features 12/13.

## 4. Request Schema (reused, not changed)

```python
# millm/api/schemas/openai.py — the field ALREADY ships (Feature 10, :63-84).
# steering_intensity: Optional[Union[float, Literal["off","min","max"]]]
# bool-reject + [0,2] validators reused verbatim. NO schema change for circuits.
```

The field's meaning generalizes with the active target: cluster active ⇒ scales the cluster (Feature 10);
circuit active ⇒ scales all circuit layers (this feature). The client sends the same field either way.

## 5. Inference Service Design

```python
# millm/services/inference_service.py
# resolve_request_intensity(request, ensure_named_profile=...) ALREADY exists (Feature 10);
# generalize its range source: when the ACTIVE steering target is a circuit, resolve
# min/max from the circuit's intensity semantics (Features 12/13), else the active
# cluster's range, else config fallback (CLUSTER_INTENSITY_MIN/MAX).

async def _apply_request_steering(self, profile_name, intensity_raw):
    """Feature 10's generalized apply — REUSED. Base selection now includes the
    active CIRCUIT: when a circuit is active and no named profile is given, base =
    the circuit's per-layer members (all (layer, feature_idx): signed_strength).
    Saves current multi-SAE {values, enabled}; λ==0 → enable_steering(False);
    else set_steering_batch({key: clamp_steering(strength * λ)}) across ALL layers.
    Restore via the existing _restore_request_profile in finally (same saved shape)."""
```
- **Call sites unchanged:** both generation paths already wrap `_apply_request_steering` in the
  `try/finally` that restores on completion and client disconnect (Feature 10).

> **AS-BUILT AMENDMENT (2026-07-20, R3).** The design above specifies a single base-selection branch
> inside the reused `_apply_request_steering`. **What shipped is a separate
> `_apply_request_circuit_steering`**, with its own saved-state shape
> (`{"circuit": True, "layers": [...]}` vs `{"values", "enabled"}`), its own resolver
> (`_resolve_circuit_intensity` vs `_plan_effective_intensity`, which differ on floor clamping and on
> the configured envelope), and a demultiplexing branch in `_restore_request_profile`.
>
> **Why:** Feature 10's base is single-SAE — it saves and restores exactly one SAE, reachable only as
> `layers[0]`. A circuit spans layers, so reusing that base would leave every other layer permanently
> dialled. The branch could not be "only base selection" without generalizing Feature 10's save/restore
> to a list first.
>
> **Known cost, recorded rather than hidden:** this is two parallel derivations of one concept, and
> five of the worst defects across R1–R3 were consequences of it (wrong-basis rescale, snapshot keyed
> off the wrong source, rung/λ echo divergence, λ=0 clear divergence, duplicated envelope logic). The
> right long-term shape is to make the PROFILE path a degenerate one-member case of the circuit path,
> collapsing to one saved shape, one restore loop, and one resolver. That refactor is deferred with the
> steering-epoch work (R3 deferred items A and B) because it also spans Feature 10.
- **All-layers-under-one-λ (DIAL-A1/A3):** the members dict spans layers; a single λ multiplies every
  authored strength; each product is clamped ±200 per member via the shared `clamp_steering` helper.
- **Routing:** `steering_intensity is not None` already forces serial (Feature 10, `_use_cbm_for_request`).
  A dialed circuit must never hit CBM (shared multi-SAE state). No change.
- **Slice_fallback (EC-14.3):** when the active circuit serves in slice_fallback, the base is the bound
  per-layer slice's members; λ scales those. No special-casing beyond reading the fallback base.

## 6. API Design

FPRD §5. **No management routes added** — the global circuit dial is
`PUT /api/circuits/active/intensity` (Features 12/13). Header echo added in `chat.py`:

```python
# millm/api/routes/openai/chat.py — beside the existing X-miLLM-Steering-Intensity echo (:87-95, :118-131)
# When a circuit is active, also echo its rung so observability tools can pin the
# evidence grade the request steered under. Best-effort, same as the λ echo.
circuit_rung = await inference.active_circuit_rung()   # None when the active target is not a circuit
if circuit_rung is not None:
    headers["X-miLLM-Circuit-Rung"] = str(circuit_rung)
```

## 7. OWUI Filter Extension Design

The shipped filter (`millm_dial_filter.py`, v1.3.0 — `class Filter`, `Valves`/`UserValves`, `inlet`,
toggle chip, `__event_emitter__` status) is **extended, not replaced**. The dial valves are unchanged; the
only additions are (a) a best-effort circuit-status probe and (b) rung-aware status copy.

```python
# integrations/openwebui/millm_dial_filter.py — additions (bumped to v1.4.0)

# RUNG_LANGUAGE mirrors docs/mcp-contract.md §4a VERBATIM. NEVER paraphrase; NEVER
# say "causal" for a rung below 2. The server sends rung_language; the map is a
# fallback for when the probe returns only a bare rung int.
RUNG_LANGUAGE = {
    0: "associated",
    1: "suggested (attribution-supported)",
    2: "causally validated (edge)",
    3: "faithfulness-tested (circuit)",
}

class Valves(BaseModel):
    # ...existing (enabled, default_dial, default_custom_lambda, show_status)...
    show_circuit_rung: bool = Field(
        default=True,
        description="Probe the active circuit and show its identity + evidence rung in the status line.",
    )

def _circuit_status(self, base_url, headers) -> Optional[dict]:
    """Best-effort GET /api/circuits/active → {name, rung, rung_language, serving_mode}
    or None. A failed/absent probe (older runtime, no circuit) MUST degrade silently
    to Feature 10 cluster behavior — a probe error never breaks chat."""
    ...

# in inlet(), after resolving `dial` and BEFORE emitting status:
#   if self.valves.show_circuit_rung and dial is not None:
#       c = self._circuit_status(...)
#       if c:
#           lang = c.get("rung_language") or RUNG_LANGUAGE.get(c.get("rung", 0), "associated")
#           mark = " · UNVALIDATED" if (c.get("rung", 0) < 2) else ""
#           slice_ = " (slice)" if c.get("serving_mode") == "slice_fallback" else ""
#           text = f"miLLM circuit «{c['name']}»{slice_} {dial} — rung {c['rung']}: {lang}{mark}"
```
No `outlet`: restoration is server-side per request (DIAL-F3). The probe is best-effort so the filter
still works against a runtime with no circuit endpoint (EC-14.5). The base URL/headers for the probe reuse
whatever the filter already knows about the miLLM backend (documented in the file header, like v1.3.0's
compatibility notes).

## 8. Testing Strategy

### Unit (`tests/unit/services/test_request_intensity.py` ext, `tests/unit/api/test_openai_schemas.py`)
- Circuit λ resolution: range present / absent (config fallback) / active target is cluster vs circuit.
- `_apply_request_steering` circuit base: all-layers scaling; λ=0 disable; clamp parity per member; saved/restored multi-SAE shape.
- Header echo: `X-miLLM-Circuit-Rung` present only when a circuit is active.
- RUNG_LANGUAGE map exactness — no "causal" string for keys 0/1.

### Integration (`tests/integration/api/test_chat_completions.py` ext)
- Field over an active circuit, streaming + non-streaming; serial routing asserted; global steering byte-identical before/after.
- Cluster-active fallback (EC-14.1); no-active no-op with logged notice (EC-14.2); slice_fallback scaling (EC-14.3).

### Filter unit (`tests/unit/integrations/test_dial_filter.py` ext or standalone)
- Circuit probe → status copy; rung<2 → "unvalidated"; "causal" never in output for rung<2; probe failure → clean degradation.

### E2E (post-deploy)
- Scripted identical-prompt off/min/max on a serveable circuit; OWUI dial walkthrough per the manual.

## 9. Risks

- **Rung lost/paraphrased at the filter (RSK-004).** Mitigation: render `rung_language` verbatim; the
  RUNG_LANGUAGE map keys 0/1 are asserted to contain no "causal" string; rung<2 forced to "unvalidated".
- **Probe latency in the inlet path.** Mitigation: best-effort, short-timeout, silent-degrade probe; the
  dial injection never waits on it (status emitted after, chat never blocked).
- **Circuit vs cluster base-selection drift.** Mitigation as designed was one shared branch; **as built**
  the two paths are separate (see the amendment above), so the mitigation is instead: one
  `_steering_circuit()` predicate answering "is a circuit steering" for every surface (apply, λ echo,
  rung echo, and the `steering` field clients read), plus unit tests pinning cluster-active vs
  circuit-active vs no-active bases and integration tests asserting global state is byte-identical
  across interleaved requests. This risk MATERIALISED three times during review — it is the single
  most productive risk row in the document.
- **Cross-layer over-steering under one λ (RSK-002).** Out of scope to correct here (surfaced by Features
  12/13); the dial can always reach off, and default λ stays conservative.
