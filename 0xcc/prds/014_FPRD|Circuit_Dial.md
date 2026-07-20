# Feature PRD: Circuit-Aware OWUI Dial

## miLLM Feature 14

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Runtime)
**References:** `BRD-MILLM-CIRCUITS-001.md` · `000_PPRD|miLLM.md` (v1.2, FR-14.x) · `000_PADR|miLLM.md` (v1.2) · `docs/mcp-contract.md` (v1.1)

---

## 1. Feature Overview

### Feature Name
Circuit-Aware OWUI Dial — live, per-user influence control over a whole active CIRCUIT in real chat,
with its evidence rung shown honestly.

### Brief Description
Extends Feature 10's per-request `steering_intensity` dial from a single-layer cluster to a whole
multi-layer **circuit**: one global λ scales ALL layers of the active circuit together (never a
per-layer entry), resolved server-side, applied and restored inside the request boundary. The shipped
Open WebUI Filter Function is extended so the dial surfaces the active circuit's identity and its
**evidence rung** verbatim (`rung_language`); a rung<2 circuit is visibly marked **unvalidated** — the
word "causal" never appears below rung 2. Users compare identical prompts at off / min / max within one
chat session and can see whether what they are steering is validated.

### Problem Statement
Feature 10 dials one cluster (one layer). A circuit lives across several layers with per-layer budgets;
the runtime can now serve it (Features 12/13), but the end user has no live control over its influence
and — critically — no signal at the point of influence about whether the circuit is causally validated
or merely mined. A mined (rung<2) circuit dialed to max in a live chat, presented as if causal,
overclaims exactly where it matters most.

### Feature Goals
1. Circuit dial semantics: off / min / max (and raw λ) scale ALL layers of the active circuit under one λ (BR-006; FR-14.1).
2. Per-request isolation: apply/restore within the request boundary incl. client disconnect; concurrency-safe (FR-14.2).
3. Evidence honesty at the dial: surface the active circuit's identity + `rung_language` verbatim; rung<2 visibly marked unvalidated; never say "causal" below rung 2 (FR-14.3).
4. Same-prompt off/min/max comparison in one chat session (FR-14.4).

### User Value Proposition
"Same prompt, dial a whole circuit off → min → max — three visibly different answers, inside my normal
chat window — and I can see whether this circuit is actually validated."

### Connection to Project Objectives
Delivers BRD-MILLM-CIRCUITS-001's "put a whole circuit's influence under the end user's hand" objective
and its evidence-integrity policy at the frontend. Reuses Feature 10's proven per-request machinery;
the circuit generalization is a scale-all-layers λ plus rung surfacing.

### BRD Traceability

| BR | Requirement (abbrev.) | Coverage IDs |
|----|-----------------------|--------------|
| BR-006 | End user dials an imported circuit off/min/max (all layers under one λ), compares identical prompts in-session, per-request isolated + restored incl. client disconnect | DIAL-A1, DIAL-A2, DIAL-A3, DIAL-A4, DIAL-F1, DIAL-F2, DIAL-F3 |
| BR-005 (dial surface only) | Surface each circuit's EvidenceRung verbatim wherever steering state shows; never "causal" below rung 2 | DIAL-A5, DIAL-F2, DIAL-F4 |

*(BR-005's full ladder-surfacing obligation is owned by Features 12/13 across MCP/Admin UI; this feature
covers only its Open WebUI dial surface.)*

---

## 2. User Stories & Scenarios

#### US-14.1: Dial a circuit in chat
**As an** Open WebUI user with the miLLM Function installed and an active circuit
**I want to** set the dial (default/off/min/max) per chat
**So that** my next messages generate under that circuit's influence, all layers together.

**Acceptance Criteria:**
- [ ] Dial off/min/max sends `steering_intensity`; the server scales ALL layers of the active circuit under one λ
- [ ] off ⇒ circuit steering disabled for the request; min/max ⇒ λ from the circuit's intensity semantics
- [ ] Identical prompts differ observably across dial positions (given a serveable circuit)

#### US-14.2: See the circuit's evidence rung
**As an** Open WebUI user
**I want** the dial to tell me which circuit is active and its evidence rung
**So that** I never mistake a mined circuit for a validated one.

**Acceptance Criteria:**
- [ ] The status line names the active circuit and shows its `rung_language` verbatim
- [ ] A rung<2 circuit is visibly marked **unvalidated**; the word "causal" never appears below rung 2
- [ ] When no circuit is active the dial says so (falls through to cluster/server state, never errors)

#### US-14.3: API caller
**As an** API integrator
**I want to** pass `steering_intensity` (float 0..2 or off/min/max) and have it scale the whole active circuit
**So that** any client — not just OWUI — can dial a circuit.

**Acceptance Criteria:**
- [ ] Numeric λ (0..2) and symbolic values resolve server-side against the active circuit
- [ ] Streaming + non-streaming; `X-miLLM-Steering-Intensity` echoes effective λ (reused from Feature 10)
- [ ] `X-miLLM-Circuit-Rung` echoes the active circuit's rung when a circuit is active

#### US-14.4: Concurrency isolation
**As a** second user chatting concurrently
**I want** my request unaffected by another user's circuit dial
**So that** the dial is safe on a shared server.

**Acceptance Criteria:**
- [ ] Apply/restore inside the request boundary (serial queue); global steering unchanged afterward
- [ ] Restore runs on client disconnect mid-stream (reuses Feature 10's finally placement)

#### Edge Cases
**EC-14.1: Active is a cluster, not a circuit** — **Behavior:** the dial degrades to Feature 10 behavior
(scales the active cluster); status line reflects cluster identity, no circuit rung.
**EC-14.2: No active steering at all** — **Behavior:** field is a no-op with a logged notice; chat never
breaks (parity with EC-10.1).
**EC-14.3: Circuit in slice_fallback** — **Behavior:** λ scales the bound per-layer slice; the status line
marks it a slice (a projection, not the whole circuit); rung still surfaced.
**EC-14.4: Invalid λ** — **Behavior:** OpenAI-style 400 (λ<0, λ>2, unknown symbol) — reuses Feature 10's validator.
**EC-14.5: Older miLLM / no circuit runtime** — **Behavior:** `extra="ignore"` drops the field; the filter's
circuit status probe returns nothing and the dial degrades to cluster behavior — documented rollout property.

---

## 3. Functional Requirements

### Per-Request Circuit Dial (FR-14.1, FR-14.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| DIAL-A1 | `/v1/chat/completions` `steering_intensity` shall, when a circuit is active, scale ALL layers of that circuit under one λ (never a per-layer entry) | Must |
| DIAL-A2 | Symbolic values resolve server-side against the active circuit's intensity semantics (off→0; min/max→bounds); numeric passthrough; config fallback when unspecified | Must |
| DIAL-A3 | Effective per-request circuit steering = per-layer authored strengths × λ, clamped ±200 per member (shared clamp helper); λ=0 ⇒ circuit steering disabled for the request | Must |
| DIAL-A4 | Apply/restore within the request boundary (serial queue semaphore); restored on completion incl. client disconnect; global state unchanged afterward (reuses Feature 10's `finally`) | Must |
| DIAL-A5 | `X-miLLM-Circuit-Rung` echoed when a circuit is active; `X-miLLM-Steering-Intensity` echo reused verbatim | Should |

### OWUI Filter Extension — Circuit Identity & Rung (FR-14.3, FR-14.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| DIAL-F1 | Extend `integrations/openwebui/millm_dial_filter.py` (do NOT fork) — same dial valves scale a circuit when one is active | Must |
| DIAL-F2 | The filter's status line surfaces the active circuit's identity + `rung_language` verbatim; a rung<2 circuit is visibly marked **unvalidated** | Must |
| DIAL-F3 | No outlet/restore logic in the filter (restoration is server-side, per request) | Must |
| DIAL-F4 | The filter NEVER emits the word "causal" for a rung<2 circuit; RUNG_LANGUAGE strings mirror the contract verbatim | Must |

---

## 4. Data Requirements
None. **No new DB table; no migration.** This is a dial extension over Features 12/13's active-circuit
state. λ is request-scoped; the persisted global λ lives on the circuit row Features 12/13 own.

## 5. API Specifications

#### POST /v1/chat/completions (extension — reused)
```json
{ "model": "...", "messages": [...], "steering_intensity": "max" }
```
Same field, same validator as Feature 10. When a circuit is active, λ scales all its layers. Response
headers: `X-miLLM-Steering-Intensity: 1.5` (effective λ) and `X-miLLM-Circuit-Rung: 2` (when a circuit is active).

#### Existing (Features 12/13, referenced): PUT /api/circuits/active/intensity — the GLOBAL dial
`{intensity, reapply}`; one global λ scales all layers of the active circuit (analogue of Feature 8's
`PUT /api/clusters/active/intensity`). Admin-UI/MCP surface. **This feature adds no management routes.**

## 6. UI Requirements
None in Admin UI (the global circuit slider ships with Features 12/13). The OWUI surface is the extended
Filter's valve + status line, rendered natively by Open WebUI.

## 7. Non-Functional Requirements
- Dial resolution + all-layer scaling adds no measurable latency (dict math over the circuit's members).
- Concurrency correctness inherited from Feature 10's serial routing; documented CBM bypass unchanged.
- Evidence-integrity: rung surfacing is a first-class product constraint, not a UI nicety (BRD policy).

## 8. Dependencies
- **Feature 10** (extended): `steering_intensity` field + validator, `resolve_request_intensity`,
  `_apply_request_steering` save/restore-in-finally, serial routing, header echo, the shipped OWUI filter.
- **Features 12/13**: active-circuit state, per-layer budgets, `PUT /api/circuits/active/intensity`,
  `GET /api/circuits/active` (identity + rung + serving_mode), rung vocabulary.
- **`docs/mcp-contract.md` §4a**: verbatim RUNG_LANGUAGE and the "never causal below rung 2" rule.
- Open WebUI Filter surface (Filter/Valves/inlet + toggle chip + `__event_emitter__`, as already shipped).

## 9. Success Criteria
1. Same-prompt off/min/max on a serveable circuit produces observably different outputs, all layers scaling (manual OWUI E2E + scripted API E2E).
2. The dial status line names the active circuit and shows its rung verbatim; a rung<2 circuit reads "unvalidated"; "causal" appears for no rung<2 circuit anywhere.
3. Two concurrent sessions with different dials produce independent, correct results; global state unchanged after each; restore fires on disconnect.
4. All EC behaviors verified by tests; the filter degrades cleanly against a runtime with no circuit (EC-14.1/14.5).

## 10. Testing Requirements
- Unit: circuit λ resolution (range present/absent, config fallback), all-layer scaling + clamp parity, λ=0 disable, rung header echo, RUNG_LANGUAGE map exactness (no "causal" below rung 2).
- Integration: streaming + non-streaming with the field over an active circuit; serial routing asserted; global steering byte-identical before/after; cluster-active fallback (EC-14.1); no-active no-op (EC-14.2); slice_fallback scaling (EC-14.3).
- Filter unit: circuit-status probe → status-line copy; rung<2 → "unvalidated"; "causal" never emitted; degradation when the probe returns nothing.
- E2E (post-deploy): OWUI circuit dial walkthrough; scripted identical-prompt off/min/max comparison.

## 11. Rollout & Migration
Purely additive. `extra="ignore"` keeps old/new client/server combinations safe both directions. The
filter's circuit-status probe is best-effort: on a runtime without the circuit runtime it degrades to
Feature 10 cluster behavior (EC-14.5). No migration.

## 12. Out of Scope
- Per-layer dials (one global λ only — BR-006 locked).
- Multiple simultaneous circuit dials.
- Persisting per-user dial positions server-side.
- Admin UI circuit slider (Features 12/13); the unvalidated-activation gate (Features 12/13 own activation).

## 13. Open Questions
None blocking. Circuit intensity semantics + `GET /api/circuits/active` shape are fixed by Features 12/13
and §4a of the contract.

## 14. Documentation Requirements
Manual: extend `manual/docs/tutorials/open-webui.md` — the dial now targets a whole circuit and shows its
rung; document the "unvalidated" marker and the global-vs-per-request distinction. OpenAI-API reference
gains the `X-miLLM-Circuit-Rung` header note.

## 15. Decisions from Clarifying Questions
1. **One global λ for the whole circuit** (BRD locked decision (1)/(6)) — per-layer caps rejected; all
   layers scale together, per-request λ overrides the stored global λ for one request (Feature 10 semantics).
2. **Extend the shipped filter, do not fork** (BRD locked decision (3)) — same `steering_intensity`
   transport; the circuit-awareness is a status-line probe + rung copy, not a new plugin.
3. **Rung surfaced verbatim at the dial** (BRD policy / §4a) — the filter renders `rung_language` as sent
   by the server; rung<2 is marked unvalidated; the filter never re-phrases or says "causal" below rung 2.
