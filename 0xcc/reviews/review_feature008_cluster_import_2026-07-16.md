# Multi-Agent Review — Feature 8: Cluster Import

**Date:** 2026-07-16 · **Scope:** feature (commits `5a46d65..HEAD`, miLLM)
**Rounds:** 1 = multi-angle /code-review (2 parallel finder agents + inline angles),
2 = post-fix verification + fresh angles, 3 = this 4-perspective /review.
**Goal gate:** ≥10 findings per round → **R1: 24, R2: 12, R3: 12 — 48 total, 44 fixed, 4 documented.**

---

## Round ledger

### Round 1 (24 — 20 fixed, 4 documented)
Critical: cluster gate bypassable via the generic profiles route AND `clear_steering` ran before
validation (failed activation wiped live steering) → gate moved into
`ProfileService._validate_activation`, the single choke point, validating BEFORE any mutation.
Deactivating a non-active profile cleared the active one's steering → scoped to `profile.is_active`.
Unbound clusters could never bind through the UI → Activate enabled (activation IS the binding
mechanism). Cluster steering editable via `update_profile` (double-scaling + export divergence) →
guarded. Flat export/import of cluster rows (λ silently dropped / empty-profile trap) → guarded with
pointers to the cluster endpoints. Hub typos tripped the SHARED HF breaker (blocking model/SAE
downloads) → dedicated breaker. `set_intensity` persisted λ before re-apply → rollback. Import
warnings mutated post-flush (never persisted) → explicit update. `extra="ignore"` stripped unknown
fields → raw-payload storage. Manifest `AttributeError` 500s, unbounded hub cache, unhandled UI
rejections, Profiles-page cluster rows unguarded, manual claims stale (reject→clamp) — all fixed.
Documented: 200-status envelope for size cap (house style), dead `blocked` status (contract parity).

### Round 2 (12 — all addressed)
The round-1 lossless fix was HALF-wired: export re-validated through the stripping mirror →
export now returns the raw stored dict (route drops `response_model`); regression test extended to
the full round-trip. The breaker "exclusion" happened AFTER failure recording → `CircuitBreaker`
gains `excluded_exceptions`; `CircuitOpenError` → 503 `HUB_UNAVAILABLE`. The UI reduced every gate
message to "Validation error" (assumed FastAPI's 422 shape) → envelope-first parsing. The
per-request OpenAI `profile` param missed the cluster n_features gate → added. Also: validation
hoisted above the steering branch, rollback guarded against masking, EFFECTIVE intensity bounds as
the single envelope (three fallback disagreements), colors keyed to feature identity (rank-keyed
hues flickered), slider dirty-guard + honest ±200 defaults, `intensity_range` destructure guard,
size-gate comment honesty.

### Round 3 (12 — this review)
1. **[Product→fixed]** FPRD CLI-U3: budget block (B, formula id) missing from ClusterCard → added.
2. **[Product→fixed]** CLI-H1's base-model narrowing had no UI control → "Only packs for
   {loaded model}" checkbox.
3. **[QA→fixed]** `get_user_friendly_message` REPLACED every crafted gate message with a generic
   sentence — users still never saw them despite the round-2 client fix. All `ValidationError`
   messages audited (user-appropriate, no internals) → generic override removed for that code.
4. **[Test→fixed]** The round-1 bypass was regression-tested only at the service layer → HTTP-level
   test through the real profiles route + `millm_error_handler` (asserts the 422 envelope carries
   the real message).
5. **[Architect→fixed]** `PUT /active/intensity` built N summaries to find one row →
   `get_active_cluster()` repo lookup.
6. **[Product→fixed]** Manual page for the Clusters feature was missing (FPRD §14) →
   `manual/docs/features/clusters.md` + sidebar (manual builds).
7. **[QA→verified-safe]** `MiLLMError` handler maps `ValidationError`→422 management envelope on
   cluster routes (no 500s); OpenAI routes get the OpenAI error shape.
8. **[Architect→verified-safe]** Route precedence (`/active/intensity` before `/{id}/intensity`;
   `/hub/search` before the greedy path route) — pinned by tests.
9. **[Test→debt]** `request()` envelope-first 422 parsing has no frontend unit test (api.ts is
   untested generally — pre-existing) — noted, not blocking.
10. **[Architect→debt]** True pre-parse body limiting belongs in middleware/ingress; in-handler
    gates bound downstream work only (comment now says so honestly).
11. **[Product→debt]** `blocked` import status is never produced by miLLM (activation is the hard
    gate) — kept for miStudio import-matrix contract parity, documented in code.
12. **[QA→debt]** Hub browse is anonymous-only by design (public packs); private-pack support would
    need token plumbing — future consideration, matches the BRD's consume-only scope.

## Perspective summaries

- **Product:** All 8 FR groups (FR-8.1–8.8) implemented and traceable; US-8.1..8.5 acceptance
  criteria verified against tests/UI; the two UI gaps found this round are fixed. Import→activate→
  steer works on a real 19-member miStudio export (fixture-based integration test).
- **QA:** Hostile-payload caps tested; no execution of imported content; path traversal blocked;
  breaker isolation; error messages now reach users end-to-end; validation-before-mutation invariant
  pinned by regression tests.
- **Architect:** Single activation choke point (`_validate_activation`) serves all four paths
  (Profiles route, Clusters route, per-request param, MCP-future); raw-dict storage/export keeps the
  frozen contract honest; `excluded_exceptions` is a general breaker capability, not a special case.
- **Test:** 61 feature-scoped backend tests (schema 21, service 17+9 regressions, hub 10, routes 15,
  workflow 6+1 HTTP) + 10 frontend; suites: 857 backend / 198 frontend green.

## Gate

**SHIP** — implementation, three deep review rounds, and acceptance evidence complete.
Deploy + live E2E ride the increment's rollout (GitOps auto-deploys pushed commits).
