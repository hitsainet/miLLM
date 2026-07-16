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
