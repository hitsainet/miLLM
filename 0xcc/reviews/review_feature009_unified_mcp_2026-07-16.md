# Multi-Agent Review — Feature 9: Unified MCP (cross-repo)

**Date:** 2026-07-16 · **Scope:** miLLM `0020dce`+`c8ec5b2` · miStudio `40f55f2`+`b78b47a`
**Goal gate:** ≥10 findings per round, 3 rounds.

---

## Round 1 (1 finder agent, cross-repo contract audit) — 14 findings: 12 fixed, 2 documented

**Critical, fixed:**
1. **`millm_activate_profile` 422'd on every real call** (empirically confirmed on miLLM's
   FastAPI version) — the route declares a required request body; the tool sent none. Smoke tests
   mocked the client, so nothing caught it. → sends `{apply_steering}` (flag exposed); regression
   pin asserts the body; test-gap finding (#14) addressed with route-shape pins.
2. **The normative contract was wrong on its most import-critical paths** — import-route refusals
   (`PAYLOAD_TOO_LARGE`, `UNKNOWN_KIND`, `VALIDATION_ERROR`, `NO_ACTIVE_CLUSTER`) arrive as
   HTTP 200 + `success:false`, and health endpoints + FastAPI 422s are non-envelope; §5 listed
   `MODEL_NOT_LOADED` as 503 (400 on management routes). → contract §2/§5 rewritten honestly
   ("never branch on status alone"; non-envelope endpoints enumerated).
3. **US-9.4's per-product `/health` was missing and miStudio itself had no gate product** →
   `/health` now reports `{products: {mistudio, millm}}` via the TTL gate; the gate knows the
   `mistudio` product (`/api/v1/system/health`).

**Also fixed:** gate hardening (strict 2xx — a 3xx from an ingress fronting a dead backend no
longer counts as available; 200+`status:"unhealthy"` refuses per the contract's reserved
semantics; ONE long-lived probe client instead of per-probe construction + `aclose`);
`millm_sensing_events` gained the contract-documented `since` param and an honest docstring
(context fields ARE in list rows); import XOR by presence not truthiness (empty-dict definitions
reach miLLM's real validator) + `on_conflict` exposed and validated; `reapply: null` no longer
inverts to False; `raw_get` failures surface the structured `{code,message,details}` instead of an
escaped envelope string; 10 new regression pins (51 MCP tests green).

**Documented:**
- The production k8s manifest opts into the millm categories by default — FPRD said "existing
  deployments unchanged", but FTASKS task 5.1 explicitly directs the deployment wiring and miLLM
  IS deployed in this cluster (verified live). The SERVER defaults still exclude millm_*
  (pinned by test); accepted as the increment's rollout.
- `MiLLMClient` isn't closed at server shutdown — pre-existing pattern (`MiStudioClient` isn't
  either); the gate's client now has `aclose`. Process-lifetime leak only; refactor debt.
- miStudio's own 38 tools remain ungated (raw BackendError on backend outage) — the gate now
  supports the product; wiring all tool modules through it is follow-on work, noted.

## Round 2 (fix verification + fresh angles, 1 finder agent) — 13 findings: 11 fixed, 2 documented · 12 R1-fix verifications (all functionally landed)

**Critical, fixed:**
1. **The R1 per-product `/health` could take the MCP server down** — it awaited live probes while
   k8s probes default to a 1 s timeout with periods exceeding the gate TTL: a HUNG dependency
   (not a clean refusal) would fail readiness (~45 s to out-of-rotation) and restart-loop
   liveness (~90 s cycles). → `/health` now serves gate SNAPSHOTS (never blocks; background
   refresh); probes gain `timeoutSeconds: 5`; per-product single-flight kills the
   TTL-expiry thundering herd; the cache is stamped AFTER the probe (a 3 s probe was eating
   3 s of every 10 s window).
2. **Unauthenticated topology leak** — `/health` (excepted from bearer auth, exposed via ingress)
   returned reasons containing internal cluster DNS names and raw exception text. → coarse
   public categories (`unreachable` / `not configured` / `error response`); detailed reasons stay
   in logs and authenticated tool results.
3. **`on_conflict` silently dropped on the hub import path** (validated, then not sent; miLLM's
   hub route didn't support it) → `HubImportRequest.on_conflict` added in miLLM (`1010830`) +
   forwarded by the tool; contract §4 updated.

**Also fixed:** contract §3 contradicted the shipped gate (unhealthy-body refusal was
undocumented) → §3 rewritten to `2xx AND status != "unhealthy"`, 3xx-refusal noted, and the
reserved-semantics caveat made explicit; `/health` omits-vs-reports millm keyed off requested
categories too (misconfigured URL is now self-diagnosing); `raw_get` 2xx-non-JSON and
non-envelope-4xx stay structured `BackendError`s; `since` validated locally (naive timestamps
rejected — the server would silently assume UTC, shifting polling windows); dead-API `aclose`
made real (`mcp.close_backend_clients` closes gate + clients; gate client lazily built — tests
leaked one per instantiation); test debt (dead helper, unused import, real-network probe in a
unit test) cleaned; 8 new pins (59 MCP tests green).

**Documented:**
- The unhealthy-body branch is defensive: today's liveness endpoints hardcode healthy statuses
  (comment + contract note the reserved semantics).
- miStudio's own tools remain ungated (the gate knows the product; wiring 38 tools is follow-on).

## Round 3 (/review, 4 perspectives) — 14 findings: 9 fixed, 5 assessed/documented

**Headline verification:** the cross-product pipe is verbatim-compatible end-to-end — miStudio's
export returns the RAW v1 document, both repos' vendored contracts are byte-identical (now
pinned to EACH OTHER by a cross-repo test, not just to their own mirrors), and
`millm_import_cluster(definition=…)` posts it unmodified into miLLM's validator.

**Fixed:** `activate=true` on unbound imports was silently dropped (the flagship flow's agent
believed the cluster live while traffic ran unsteered) → explicit persisted warning; EC-9.3
mid-TTL outages now return structured unavailable on THIS call and invalidate the stale gate
entry (`invalidate` was dead API); snapshot background refreshes keep strong task refs with
exception retrieval + per-product dedup (fire-and-forget tasks were GC-collectable);
`public_reason` anchored (an 'unhealthy' HOSTNAME miscategorized an HTTP-status reason);
timeouts labeled as timeouts (a slow hub import read as a connectivity failure); the malformed
`_locks` annotation (MyPy-strict violation); `__main__` closes backend clients after uvicorn
stops (the R2 hook had no production caller — the R2 record overstated); the sensing tool
docstring names `ambient_fired_count` as the alone-vs-within signal (US-9.3 discoverability);
FPRD amended with shipped tool/route names; both named mutation survivors pinned
(`request()` non-envelope 4xx, `public_reason` HTTP prefix) + audit-triple-wrap invocation test.

**Assessed sound / documented:** validate-then-gate ordering in `millm_import_cluster`
(deliberate — argument errors are actionable regardless of backend state); single-loop gate
affinity (docstring note added); env-property config (matches worker convention; k8s env is
immutable per pod); split `MILLM_CATEGORY_MODULES` registry (different register arity);
`definition`+`filename` still silently ignores `filename` (R1-noted; XOR pairs the sources).

## Live E2E (tasks 4.2/5.2 — deployed pair, 2026-07-16)

- **Deployed unified server**: `/health` reports all 10 categories incl. the 3 millm ones and
  per-product availability (`mistudio: ok, millm: ok`) via non-blocking snapshots.
- **MCP tool call through the real path** (initialize → `tools/call millm_status` with bearer
  auth): returned miLLM's detailed health including `active_profile` — US-9.2's one-call status.
- **Flow**: import (bound — after the live-E2E-caught `sae_id` VARCHAR(50) truncation fix +
  migration 009) → activate → per-request dial off/min/max produced three observably different
  outputs with correct `X-miLLM-Steering-Intensity` echoes (0 / 0.5 / 1.5) → sensing enabled →
  real co-activation event captured (quorum 2/3, prefill span, score 2.5×θ, ±8-token context
  decoded) → warm sensing overhead 3.9 ms (< 5 ms budget; cold first pass 34.9 ms = kernel
  warmup) → arm-refusal path verified live with the actionable message → all artifacts cleaned.

## Gate
**SHIP** — 41 findings across 3 rounds (32 fixed, 9 assessed/documented), live E2E complete on
the deployed pair. Suites: miStudio MCP 67 green (backend suite pytest exit 0), miLLM 1084+.
