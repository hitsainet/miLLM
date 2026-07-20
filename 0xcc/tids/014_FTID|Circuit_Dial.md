# Technical Implementation Document: Circuit-Aware OWUI Dial

## miLLM Feature 14

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Runtime)
**References:** `BRD-MILLM-CIRCUITS-001.md` · `000_PPRD|miLLM.md` (v1.2, FR-14.x) · `000_PADR|miLLM.md` (v1.2) · `docs/mcp-contract.md` (v1.1) · `014_FPRD|Circuit_Dial.md` · `014_FTDD|Circuit_Dial.md`

---

## 1. File Structure

```
millm/
├── api/schemas/openai.py                (NO CHANGE — steering_intensity field reused verbatim)
├── api/routes/openai/chat.py            (MOD — X-miLLM-Circuit-Rung echo beside the λ echo)
├── services/inference_service.py        (MOD — circuit base-selection branch in _apply_request_steering;
│                                         circuit range in resolve_request_intensity; active_circuit_rung())
integrations/openwebui/millm_dial_filter.py   (MOD — v1.4.0: circuit-status probe + rung status copy)
manual/docs/tutorials/open-webui.md      (MOD — circuit dial + rung/unvalidated marker)
tests/unit/services/test_request_intensity.py (MOD — circuit base + resolution)
tests/unit/api/test_openai_schemas.py    (unchanged — field already covered by Feature 10)
tests/unit/integrations/test_dial_filter.py (MOD/NEW — probe→copy, rung<2 marker, no-"causal")
tests/integration/api/test_chat_completions.py (MOD — circuit dial cases)
```

No migration, no new model, no new route module. Everything is additive over Features 10/12/13.

## 2. Load-Bearing Implementation Points (verified against live code)

- **The proven template is Feature 10's `_apply_request_steering`** (`inference_service.py`): applied
  inside the request-queue semaphore, saves `{values, enabled}`, restored via `_restore_request_profile`
  in `finally` — already covers client disconnect mid-stream. The circuit dial **adds a base-selection
  branch**, not a parallel mechanism. Do NOT build a second apply path.
- **The `finally` restore already covers disconnect** (`chat.py` wraps generation; the finally placement
  from Feature 10 is what restores on cancellation). Keep the circuit branch inside the SAME try/finally.
- **The request field + validators already ship** (`openai.py:63-84`): `steering_intensity`, bool-reject,
  `[0,2]` bound, `model_config = {"extra": "ignore"}`. **Do not touch them** — the circuit meaning is
  server-side base selection, not a schema change.
- **The echo path already resolves λ once** (`chat.py:87-95`): `resolve_request_intensity(request,
  ensure_named_profile=bool(request.stream))` doubles as the streaming pre-commit profile check. Add the
  `X-miLLM-Circuit-Rung` echo beside `X-miLLM-Steering-Intensity` (`chat.py:118-131` non-stream,
  `:118-125` stream headers) — best-effort, only when a circuit is active.
- **Serial routing already keys on the field** (`_use_cbm_for_request`, Feature 10): `steering_intensity
  is not None` forces serial. A dialed circuit must never see CBM (shared multi-SAE state). No change.
- **The global circuit dial is Features 12/13's** `PUT /api/circuits/active/intensity`
  (`set_active_intensity`, analogue of `clusters.py:212 set_active_intensity`). This feature adds NO route.
- **The shipped filter is v1.3.0** (`millm_dial_filter.py`): `class Filter`, `Valves`/`UserValves` with a
  `dial` field (`default|server|off|min|max|custom`), `inlet`, toggle chip (`self.toggle`, `self.icon`),
  `__event_emitter__` status via `_status`, and `_resolve`. Extend it to v1.4.0 by adding a
  `show_circuit_rung` valve, a `_circuit_status` probe, and rung-aware status copy — reuse `_resolve`,
  `_status`, `_read` untouched.

## 3. Key Implementations

```python
# inference_service.py — base selection inside the REUSED _apply_request_steering
# (only the base branch is new; save/scale/restore are Feature 10's)
active_circuit = await self._circuit_repo.get_active()   # Features 12/13
if profile_name is None and active_circuit is not None:
    # circuit base: ALL layers at once, one λ scales every member
    base = active_circuit.member_strengths()   # {(layer, feature_idx): signed_strength}
else:
    base = ...  # Feature 10 cluster/live base (unchanged)
# λ==0 → enable_steering(False); else set_steering_batch across every key:
sae_set.set_steering_batch({k: clamp_steering(s * lam) for k, s in base.items()})
```

```python
# inference_service.py — rung for the header echo
async def active_circuit_rung(self) -> int | None:
    c = await self._circuit_repo.get_active()
    return c.rung if c is not None else None   # None when the active target is a cluster/none
```

```python
# millm_dial_filter.py (v1.4.0) — RUNG_LANGUAGE mirrors §4a VERBATIM
RUNG_LANGUAGE = {
    0: "associated",
    1: "suggested (attribution-supported)",
    2: "causally validated (edge)",
    3: "faithfulness-tested (circuit)",
}
# in inlet(), when a dial resolves and show_circuit_rung is on:
c = self._circuit_status(...)          # best-effort; None → degrade to Feature 10 copy
if c:
    lang = c.get("rung_language") or RUNG_LANGUAGE.get(c.get("rung", 0), "associated")
    mark = " · UNVALIDATED" if c.get("rung", 0) < 2 else ""
    slice_ = " (slice)" if c.get("serving_mode") == "slice_fallback" else ""
    await self._status(__event_emitter__,
                       f"miLLM circuit «{c['name']}»{slice_} {dial} — rung {c['rung']}: {lang}{mark}")
```

## 4. Implementation Pitfalls

1. **One λ, ALL layers — never per-layer.** The circuit base spans multiple (layer, feature) keys; a
   single resolved λ multiplies every one. Do not expose or accept a per-layer λ (BR-006 locked).
2. **Request λ OVERRIDES the stored global λ** — inherited from Feature 10. Do not multiply the request λ
   by the circuit's persisted intensity; the request dial is absolute for that request.
3. **λ=None ≠ λ=1.0** — None means "field absent, leave live steering untouched"; the no-op path must not
   save/restore needlessly (Feature 10 pitfall, still applies).
4. **NEVER say "causal" below rung 2.** Render `rung_language` verbatim; keys 0/1 of RUNG_LANGUAGE must
   contain no "causal" string (assert it in tests). rung<2 must read "UNVALIDATED". This is a product
   constraint (BRD evidence-integrity policy), not a copy preference.
5. **The circuit-status probe is best-effort.** A failed/absent probe (older runtime, no active circuit)
   MUST degrade silently to Feature 10 cluster copy — a probe error never breaks chat. Short timeout;
   never block the dial injection on it.
6. **Restore must run on stream cancellation** — keep the circuit base branch inside the existing
   try/finally; do not move the apply out of it.
7. **Do not import miLLM types into the filter** — the probe uses only stdlib/`urllib` (or the HTTP client
   OWUI already provides); the file must run in OWUI's sandbox with no miLLM dependency (v1.3.0 property).
8. **Cluster-active must still work** — when the active target is a cluster, `active_circuit` is None and
   the code falls through to Feature 10's exact behavior (EC-14.1). Pin this with a test.

## 5. Config Additions
None new. Reuses Feature 8/10's `CLUSTER_INTENSITY_MIN/MAX` fallbacks and Features 12/13's circuit
intensity semantics. The filter's `show_circuit_rung` is a valve default (`True`), not a server config key.
