# Feature PRD: Circuit Edge Sensing

## miLLM Feature 15

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Runtime)
**References:** `BRD-MILLM-CIRCUITS-001.md` · `000_PPRD|miLLM.md` (v1.2, FR-15.x) · `000_PADR|miLLM.md` (v1.2) · `docs/mcp-contract.md` (v1.1)

---

## 1. Feature Overview

### Feature Name
Circuit Edge Sensing — observe a circuit's EDGES firing (upstream→downstream) in production traffic.

### Brief Description
When armed for an imported circuit (opt-in, off by default), miLLM detects — per forward pass — the
moments when an upstream member fires and its DOWNSTREAM partner fires within a configurable token-lag
window, records each ordered pair as a bounded, persistent edge event carrying the upstream/downstream
activations, an alone-vs-within-larger-set side channel, a ±K window of surrounding token context, and
the edge's evidence rung (surfaced verbatim — never "causal" below rung 2), and surfaces events via
API, WebSocket, and an edge-sensing panel on the Circuits page.

### Problem Statement
Feature 11 senses a cluster co-firing as a UNIT (all members fire at the same position). A CIRCUIT is
directional: its meaning is the EDGE — an upstream feature firing and then, a token or two later, its
downstream partner firing. Nothing today observes that ordering. Cluster sensing's simultaneous
all-fire predicate cannot answer the authoring question "which edges actually fire in production?"
because it collapses ordering and has no notion of an up→down partner or a lag window.

### Feature Goals
1. Catch each upstream→downstream edge firing across the generation, within a token-lag window (BR-007).
2. Alone-vs-within distinction recorded honestly (best-effort v1, never fabricated) (BR-007).
3. ±K tokens of context per event so events are interpretable (BR-008).
4. Bounded, queryable persistence carrying the edge's rung; API/UI/WS (BR-008).
5. New additive `/api/circuits/*` + `millm_circuits` MCP surface, no serving degradation (BR-009, BR-012).

### User Value Proposition
"I armed my promoted circuit; overnight traffic produced 9 edge events — here's exactly where L10→L13
fired in order, the sentences around each, and the honest note that this edge is only 'suggested'."

### Connection to Project Objectives
Closes the BRD's authoring-loop objective ("close the loop with EDGE-level co-activation sensing —
upstream→downstream — feeding edge-level observation back into authoring").

### BRD Traceability

| BR | Requirement (abbrev.) | Coverage IDs |
|----|-----------------------|--------------|
| BR-007 | Record edge co-activation (up-fire followed by down-fire); alone/within; opt-in, off by default | EDGE-D1, EDGE-D2, EDGE-D3, EDGE-R1, EDGE-S1 |
| BR-008 | Edge events retrievable (API+UI) with context (timestamp, request, up/down activations, ±K context, alone/within) | EDGE-R2, EDGE-R3, EDGE-P1, EDGE-P3, EDGE-P5 |
| BR-009 | Unified MCP gains a circuit category (status, list, import, activate/deactivate, edge sensing readout); health-gated | EDGE-P3, EDGE-P4 |
| BR-012 | New endpoints/tools additive-only, tracked in `docs/mcp-contract.md` v1.1; circuit error family | EDGE-P3, EDGE-P6 |

---

## 2. User Stories & Scenarios

#### US-15.1: Arm a circuit's edges
**As a** user with an imported circuit
**I want to** toggle edge sensing on for it
**So that** subsequent traffic is watched for its edges firing upstream→downstream.

**Acceptance Criteria:**
- [ ] Edge-sensing toggle on the Circuits page (per circuit); off by default
- [ ] Arming activates on circuit activation and disarms on deactivation / SAE-set detach
- [ ] Status endpoint reports the armed circuit, per-edge thresholds, lag window, and overhead

#### US-15.2: Review edge events
**As a** researcher
**I want to** list edge events with the up/down member, activations, flags, rung, and token context
**So that** I can learn which of a circuit's edges actually fire in real traffic.

**Acceptance Criteria:**
- [ ] Events carry: timestamp, request id, phase, upstream member + activation, downstream member +
      activation, token lag, alone/within field (or null), context parts, edge rung + rung_language, summary
- [ ] API supports circuit filter, limit, since; UI panel lists newest-first with live WS updates
- [ ] The edge's rung is surfaced verbatim; the word "causal" never appears for rung < 2

#### US-15.3: Bounded and safe
**As an** operator
**I want** retention caps and observable overhead
**So that** edge sensing can run indefinitely without harming serving.

**Acceptance Criteria:**
- [ ] Per-circuit event cap + age pruning enforced automatically
- [ ] `sensing_overhead_ms` visible in status; warn logged above threshold
- [ ] Un-armed requests: zero added work on the hot path

#### Edge Cases
**EC-15.1: Upstream fires, downstream never** — **Trigger:** upstream member fires, partner silent for the
rest of the window. **Behavior:** NO event (an edge is up→down; a lone upstream firing is not an edge event).
**EC-15.2: Downstream precedes upstream** — **Trigger:** the partner fires BEFORE the upstream member.
**Behavior:** NOT an edge event — ordering is directional; only up-then-down within the window records.
**EC-15.3: Long co-firing / repeated re-fires** — **Trigger:** the edge re-fires at many positions.
**Behavior:** per-request edge-event cap (default 20); evaluation stops, `truncated` flag on the last event.
**EC-15.4: Circuit in slice-fallback** — **Trigger:** the active circuit is serving via a per-layer
cluster slice (not all SAEs attached). **Behavior:** edge sensing requires BOTH endpoint layers' SAEs
attached; edges whose endpoints span an unattached layer are UNSENSABLE and reported so in status —
never approximated through the wrong decoder.
**EC-15.5: CBM-routed request** — **Behavior:** edge-sensing-armed ⇒ forced serial (default); if the
operator disables forcing, CBM requests are simply NOT sensed — never approximated.
**EC-15.6: Members without max_activation** — **Behavior:** the per-member threshold falls back exactly as
in Feature 11 (infinite threshold; edge unsensable unless a positive floor governs); reported in status.
**EC-15.7: context_tokens=0** — **Behavior:** events persist without text (metadata only).

---

## 3. Functional Requirements

### Edge Detection (FR-15.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| EDGE-D1 | Detection shall run per forward pass over ALL token positions, only when armed; extends the shipped `_sense` path (member-only encode, ≤20 columns) | Must |
| EDGE-D2 | 'Fired' per member: identical predicate to Feature 11 — act > max(θ_floor, ε·max_activation); per-circuit overrides via circuit meta | Must |
| EDGE-D3 | Edge event: an upstream member firing at position p FOLLOWED BY its downstream partner firing at a position in (p, p+L], L = CIRCUIT_SENSING_LAG_TOKENS (default 8, hard max 64); ordering strict (up before down) | Must |
| EDGE-D4 | Edges evaluated from the circuit's declared edges (upstream_feature_idx/layer → downstream_feature_idx/layer); only edges whose both endpoint SAEs are attached are sensed (EC-15.4) | Must |
| EDGE-D5 | Per-request edge-event cap (default 20) with truncation flag; detection cost decoupled from monitoring; armed-only | Must |

### Recording (FR-15.2, FR-15.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| EDGE-R1 | Each event records upstream member+peak activation, downstream member+peak activation, token lag, and edge rung + rung_language (verbatim from the definition) | Must |
| EDGE-R2 | Alone-vs-within: `ambient_fired_count` populated when full-width monitoring co-runs; NULL otherwise (documented best-effort — same rule as Feature 11) | Must |
| EDGE-R3 | ±K token context (CIRCUIT_SENSING_CONTEXT_TOKENS=16 default, hard max 64; K=0 ⇒ no text) as `context_parts{before, span, after}` decoded OFF the hot path; span covers the up→down positions | Must |
| EDGE-R4 | Each event carries a ≤300-char human summary; the summary NEVER uses "causal" for an edge whose rung < 2 | Must |

### Persistence & Surfacing (FR-15.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| EDGE-P1 | Events persist in `circuit_edge_sensing_events` (migration 012) with FK CASCADE to circuit ownership | Must |
| EDGE-P2 | Retention: per-circuit cap (default 1000) + age pruning (default 7 days), enforced on flush and on read (throttled) | Must |
| EDGE-P3 | API: `/api/circuits/sensing/status`, `/api/circuits/sensing/events` (circuit_id/limit/since), `/api/circuits/{circuit_id}/sensing/enable|disable`, clear — per `docs/mcp-contract.md` §4 `millm_circuits` | Must |
| EDGE-P4 | `millm_circuit_sensing_status` / `_events` / `_enable` / `_disable` MCP tools consume the routes above, health-gated | Must |
| EDGE-P5 | WS event `circuit:sensing:event` on each recorded event (throttled; mirrors the cluster `sensing:event` channel; payload excludes context text) | Must |
| EDGE-P6 | `NO_ACTIVE_CIRCUIT` (200+envelope) on sensing calls with no active circuit; reuse the sensing envelope/error conventions | Must |
| EDGE-P7 | Circuits-page edge-sensing panel: toggle, live event list, event detail (up/down member table, lag, rung badge, context) | Must |

### Safety (FR-15.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| EDGE-S1 | Armed requests route serial (CIRCUIT_SENSING_FORCE_SERIAL=true default); non-forced ⇒ CBM requests unsensed, never approximated | Must |
| EDGE-S2 | `sensing_overhead_ms` accumulated per request, surfaced in status; warn above CIRCUIT_SENSING_MAX_OVERHEAD_MS=5.0; stays within the CBM latency budget (NFR-1.5) | Must |
| EDGE-S3 | Zero hot-path work when un-armed (single boolean check in the hook); one member-only matmul per pass when armed | Must |

---

## 4. Data Requirements

Migration `012_add_circuit_edge_sensing.py` (chained after Feature 13's `011_add_circuits_table.py`;
does NOT collide with 008 sensing-events or the circuits table Feature 13 owns):

```sql
CREATE TABLE circuit_edge_sensing_events (
  id                   SERIAL PRIMARY KEY,
  circuit_id           VARCHAR(50) NOT NULL,            -- owning circuit (CASCADE via circuit ownership)
  request_id           VARCHAR(64) NOT NULL,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  phase                VARCHAR(10) NOT NULL,            -- 'prefill' | 'decode'
  edge_key             VARCHAR(80) NOT NULL,            -- '{up_idx}@{up_layer}->{down_idx}@{down_layer}'
  up_feature_idx       INTEGER NOT NULL,
  up_layer             INTEGER NOT NULL,
  up_pos               INTEGER NOT NULL,
  up_peak_act          FLOAT NOT NULL,
  down_feature_idx     INTEGER NOT NULL,
  down_layer           INTEGER NOT NULL,
  down_pos             INTEGER NOT NULL,
  down_peak_act        FLOAT NOT NULL,
  token_lag            INTEGER NOT NULL,                -- down_pos - up_pos (1..L)
  edge_rung            INTEGER NOT NULL,                -- 0..3, verbatim from the definition
  edge_rung_language   VARCHAR(80) NOT NULL,            -- server-rendered phrase (never re-worded)
  ambient_fired_count  INTEGER NULL,                    -- alone/within side channel (best-effort)
  truncated            BOOLEAN NOT NULL DEFAULT FALSE,
  context_parts        JSONB NULL,                      -- {before, span, after} decoded segments
  context_token_ids    JSONB NULL,
  summary              VARCHAR(300) NOT NULL
);
CREATE INDEX idx_cese_circuit_created ON circuit_edge_sensing_events (circuit_id, created_at);
CREATE INDEX idx_cese_request ON circuit_edge_sensing_events (request_id);
```

The model mirrors `db/models/sensing_event.py` (JSONVariant for JSONB/SQLite, `to_dict(include_context)`
excluding context on WS payloads). CASCADE follows the circuit-ownership row Feature 13 provides.

## 5. API Specifications

Router `/api/circuits/sensing` + circuit-scoped toggles (`millm/api/routes/management/circuit_sensing.py`),
shaped exactly like `management/sensing.py`:
- `GET /api/circuits/sensing/status` — armed circuit, per-edge thresholds+mode, lag window, sensable/unsensable
  edges (EC-15.4/15.6), overhead stats, plus persistent per-circuit intent (enabled flags) reported distinctly
- `GET /api/circuits/sensing/events?circuit_id=&limit=&since=` — newest-first (rows include context + rung)
- `GET /api/circuits/sensing/events/{event_id}` — detail (context); 404 `CIRCUIT_SENSING_EVENT_NOT_FOUND`
- `POST /api/circuits/{circuit_id}/sensing/enable` · `POST .../disable` — sets the circuit's edge-sensing intent;
  arms/disarms live if that circuit is active. `NO_ACTIVE_CIRCUIT` (200+envelope) when applicable.
- `DELETE /api/circuits/sensing/events?circuit_id=` — clear

WS: `circuit:sensing:event` broadcast (payload = event summary fields incl. edge rung; NOT full context text).
All additive to `docs/mcp-contract.md` v1.1; the `millm_circuit_sensing_*` tools consume these verbatim.

## 6. UI Requirements
The Circuits page (Features 13/14) gains: an EdgeSensingToggle wired to enable/disable, an edge-sensing panel
(event list w/ live updates, phase/lag/rung chips, expandable detail with up→down member table + `context_parts`
highlighting the span), and a status strip (armed circuit + overhead + unsensable-edge notes). The rung badge
renders `rung_language` verbatim. Components under `components/circuits/sensing/`.

## 7. Non-Functional Requirements
- Un-armed: one boolean per hook call. Armed: one (seq,d_in)×(d_in,≤20) matmul per pass, per attached SAE the
  circuit references (edges span layers; the encode is per-SAE, member columns only).
- Context decode strictly post-generation.
- Event context text is user content: retention caps are the privacy control; documented (NFR).
- Edge sensing stays within the CBM/speculative-decoding latency budget (NFR-1.5).

## 8. Dependencies
- Feature 13: circuit rows (edges with upstream/downstream feature+layer, per-edge rung, sensing-intent flag).
- Feature 12: multi-SAE attach (edge endpoints span layers; both endpoints' SAEs must be attached to sense).
- Feature 11 heritage: the shipped `_sense` path, `SensingConfig`/`SensedHit`, `context_parts`, WS emitter,
  serial-forcing, retention repository — all EXTENDED, not re-built.
- Serial request queue (request boundaries for begin/collect).

## 9. Success Criteria
1. Scripted prompt panel with known up→down ground truth: 100% edge-event capture, correct lag/ordering.
2. EC-15.1/15.2 honored: lone-upstream and reversed-order sequences produce NO event.
3. Alone/within field correct when monitoring co-runs; NULL otherwise.
4. Context windows match expected tokens; span covers the up→down positions.
5. Overhead within budget on armed requests; zero measurable delta un-armed; retention caps enforced.
6. Every surfaced rung is verbatim; no "causal" string for any rung < 2 edge (asserted in tests).

## 10. Testing Requirements
- Unit: per-member threshold reuse, up→down ordering + lag-window matching, EC-15.1/15.2 negatives, per-request
  cap + truncated, unsensable-edge exclusion (EC-15.4/15.6), summary builder (no "causal" below rung 2),
  repository retention.
- Integration: arm→generate→edge events persisted with correct lag/context/rung; WS emission; enable/disable
  lifecycle incl. SAE-set detach; serial forcing; CBM-unsensed path; overhead accumulator; latency-budget assert.
- E2E (post-deploy): Circuits-page edge-sensing panel live flow.

## 11. Rollout & Migration
Migration 011 additive. Feature dormant until a circuit is armed. `/api/circuits/*` additive to the contract v1.1.

## 12. Out of Scope
Simultaneous multi-circuit arming (v1: the active circuit only); per-token attribution of SAMPLED tokens;
edge sensing on CBM batches; transitive multi-hop edge chains (v1: direct declared edges only); feeding events
back into miStudio automatically (future); attribution-tier edge evidence.

## 13. Open Questions
None blocking. Lag-window default (8) and retention (per-circuit cap + age) decided — see §15.

## 14. Documentation Requirements
Manual: Circuits page edge-sensing section (semantics: what an EDGE event means, up→down ordering + lag window,
attribution convention, alone/within caveat, rung verbatim rule, retention + privacy note).

## 15. Decisions from Clarifying Questions
1. **Edge = up-fire then down-fire within a token-lag window** (BRD open question resolved 2026-07-20);
   default lag L=8 tokens, hard max 64; strict ordering; reversed/lone firings are non-events.
2. **Per-event summaries with bounded retention** (mirrors Feature 11) — per-circuit cap 1000 + age 7 days.
3. **Rung carried verbatim per edge**; "causal" forbidden below rung 2 in every surfaced string (BR-005 policy).
4. **Serial-only v1 with forced serial default** (CBM batch-attribution limit, inherited from Feature 11).
5. **Attribution convention: token being READ at each endpoint position**; sampled-token attribution out of scope.
