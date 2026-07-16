# Feature PRD: OWUI Cluster Dial

## miLLM Feature 10

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Cluster Runtime)
**References:** `BRD-MILLM-CLUSTERS-001.md` · `000_PPRD|miLLM.md` (v1.1, FR-10.x) · `008_FPRD|Cluster_Import.md`

---

## 1. Feature Overview

### Feature Name
OWUI Cluster Dial — live, per-user cluster intensity control in real chat.

### Brief Description
A per-request `steering_intensity` extension on `/v1/chat/completions` (numeric λ or symbolic
off/min/max, resolved server-side against the active cluster's intensity range) plus a single-file
Open WebUI **Filter Function** that exposes the dial as a per-user valve. Users compare identical
prompts at off / min / max within one chat session; every request's steering is applied and restored
inside the request boundary, so concurrent users never interfere.

### Problem Statement
The OpenAI-compatible API has no native control channel for steering intensity. Today a user must edit
the active profile globally (affecting every consumer of the server) to feel a cluster's influence —
impossible to do mid-conversation, unsafe under concurrency, and invisible from inside Open WebUI.

### Feature Goals
1. Dial semantics: off / min / max (and raw λ) on identical prompts in-session (BR-010).
2. Concurrency safety: request-scoped apply/restore; no cross-user interference (FR-10.2).
3. Zero client contamination: stock clients ignore the extension; older miLLM ignores the field.
4. Native OWUI UX: one installable Function, per-user valve, no OWUI fork.

### User Value Proposition
"Same prompt, dial off → min → max, three visibly different answers — inside my normal chat window."

### Connection to Project Objectives
Implements the BRD's "put a cluster's influence under the end user's hand" objective; inherits λ
semantics from Feature 8 (intensity stored raw, scaled at apply, clamped ±200).

---

## 2. User Stories & Scenarios

#### US-10.1: Dial in chat
**As an** Open WebUI user with the miLLM Function installed
**I want to** set the dial valve (default/off/min/max) per chat
**So that** my next messages generate under that cluster intensity.

**Acceptance Criteria:**
- [ ] Valve values default/off/min/max; default sends no extension field (server state governs)
- [ ] off ⇒ steering disabled for the request; min/max ⇒ λ from the active cluster's intensity_range
- [ ] Responses to identical prompts differ observably across dial positions (given a validated cluster)

#### US-10.2: API caller
**As an** API integrator
**I want to** pass `steering_intensity` (float 0..2 or "off"/"min"/"max") directly
**So that** any client — not just OWUI — can dial.

**Acceptance Criteria:**
- [ ] Numeric λ accepted (0..2, validated); symbolic values resolved server-side
- [ ] Works on streaming and non-streaming paths
- [ ] Response header `X-miLLM-Steering-Intensity` echoes the effective λ (observability)

#### US-10.3: Concurrency isolation
**As a** second user chatting concurrently
**I want** my requests unaffected by another user's dial
**So that** the dial is safe on a shared server.

**Acceptance Criteria:**
- [ ] Apply/restore happens inside the request boundary (serial queue); post-request global state is unchanged
- [ ] Requests carrying the field route serial (never CBM)

#### Edge Cases
**EC-10.1: No active cluster** — **Trigger:** dial set but active profile is manual or none.
**Behavior:** manual profile: λ scales its strengths the same way; none: field is a no-op with a
logged notice (not an error — chat must not break).
**EC-10.2: Definition without intensity_range** — **Behavior:** min/max resolve from config
`CLUSTER_INTENSITY_MIN/MAX` (0.5/1.5).
**EC-10.3: Invalid value** — **Trigger:** λ<0, λ>2, unknown symbol. **Behavior:** OpenAI-style 400
validation error (not a silent default).
**EC-10.4: Older miLLM** — **Behavior:** `extra="ignore"` drops the field; chat works unsteered-dial —
documented rollout property.

---

## 3. Functional Requirements

### API Extension (FR-10.1, FR-10.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| DIAL-A1 | `/v1/chat/completions` shall accept optional `steering_intensity: float(0..2) \| "off"\|"min"\|"max"` | Must |
| DIAL-A2 | Symbolic values shall resolve server-side: off→0.0; min/max→active cluster's `budget.intensity_range`, fallback config | Must |
| DIAL-A3 | Effective per-request steering = base strengths × λ, clamped ±200 (shared helper with Feature 8); λ=0 ⇒ steering disabled for the request | Must |
| DIAL-A4 | Apply/restore shall occur within the request boundary (serial queue semaphore); global state unchanged afterward | Must |
| DIAL-A5 | Requests carrying the field shall route serial (extend the existing has_profile condition) | Must |
| DIAL-A6 | Both streaming and non-streaming paths supported; effective λ echoed in `X-miLLM-Steering-Intensity` | Must |
| DIAL-A7 | The field shall compose with the existing `profile` per-request override (profile resolved first, then λ) | Should |

### OWUI Function (FR-10.3, FR-10.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| DIAL-F1 | Ship `integrations/openwebui/millm_dial_filter.py` — a single-file OWUI Filter Function | Must |
| DIAL-F2 | Per-user valve `dial: default\|off\|min\|max`; non-default injects `body["steering_intensity"]` | Must |
| DIAL-F3 | No outlet/restore logic in the plugin (restoration is server-side) | Must |
| DIAL-F4 | Install/usage documented in the manual's Open WebUI tutorial | Must |

---

## 4. Data Requirements
None. No migration; no new tables. (λ persistence is Feature 8's `profiles.intensity`; the dial is
request-scoped.)

## 5. API Specifications

#### POST /v1/chat/completions (extension)
```json
{ "model": "...", "messages": [...], "steering_intensity": "max" }
```
Validation: number in [0,2] or one of off|min|max, else 400. Response includes
`X-miLLM-Steering-Intensity: 1.5` (effective resolved λ; absent when field not sent).

#### Existing (Feature 8, referenced): PUT /api/clusters/active/intensity — the GLOBAL dial
(Admin-UI/MCP); this feature adds no management routes.

## 6. UI Requirements
None in Admin UI (the global slider ships with Feature 8). The OWUI-side surface is the Function's
valve UI, rendered natively by Open WebUI.

## 7. Non-Functional Requirements
- Dial resolution + scaling adds no measurable latency (dict math on ≤20 entries).
- Concurrency: correctness guaranteed by the existing serial routing; documented CBM bypass.

## 8. Dependencies
- Feature 8: cluster profiles, `profiles.intensity`, `intensity_range` in cluster_meta, clamp helper.
- Existing per-request machinery: `ChatCompletionRequest.profile` → `_apply_request_profile`
  save/restore + serial routing.
- Open WebUI ≥ the Functions/Filter plugin API (verified compatible client since v1.0).

## 9. Success Criteria
1. Same-prompt off/min/max comparison produces observably different outputs on a validated cluster
   (manual E2E via OWUI + scripted E2E via API).
2. Two concurrent sessions with different dials produce independent, correct results; global state
   unchanged after each.
3. Plugin installs on a stock Open WebUI and the valve round-trips.
4. All EC behaviors verified by tests.

## 10. Testing Requirements
- Unit: symbolic resolution (range present/absent), numeric validation bounds, λ composition with
  `profile` param, clamp parity with Feature 8, header echo.
- Integration: streaming + non-streaming with the field; serial routing assertion; save/restore
  invariant (global steering values identical before/after); no-active-cluster no-op.
- E2E (post-deploy): OWUI plugin flow; scripted identical-prompt comparison.

## 11. Rollout & Migration
Purely additive; `extra="ignore"` makes old/new client/server combinations safe in both directions.

## 12. Out of Scope
Per-message UI affordances inside OWUI beyond the valve; multiple simultaneous cluster dials;
persisting per-user dial positions server-side.

## 13. Open Questions
None blocking (mechanism decided — see §15).

## 14. Documentation Requirements
Manual: extend `manual/docs/tutorials/open-webui.md` with Function install + dial usage; OpenAI-API
reference gains the extension field.

## 15. Decisions from Clarifying Questions
1. **Mechanism: OWUI Filter/Function plugin** (user decision 2026-07-16) — native valve UX; synthetic
   model variants and raw request-param-only approaches rejected (PADR v1.1).
2. **Per-request field is primary** (design decision from concurrency analysis): a global-only dial
   mutates other users' in-flight generations; the management endpoint remains as the documented-global
   secondary.
3. **Symbolic resolution server-side** — the plugin stays range-ignorant; definitions carry their own
   intensity_range.
