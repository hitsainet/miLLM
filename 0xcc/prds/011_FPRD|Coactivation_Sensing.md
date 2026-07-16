# Feature PRD: Co-Activation Sensing

## miLLM Feature 11

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Cluster Runtime)
**References:** `BRD-MILLM-CLUSTERS-001.md` · `000_PPRD|miLLM.md` (v1.1, FR-11.x) · `008_FPRD|Cluster_Import.md` · `005_FPRD|Feature_Monitoring.md`

---

## 1. Feature Overview

### Feature Name
Co-Activation Sensing — observe a cluster firing together in production traffic.

### Brief Description
When armed for an imported cluster (opt-in, off by default), miLLM detects — per forward pass, at every
token position — the moments when the cluster's members co-fire (per-member thresholds, quorum), records
each moment as a bounded, persistent event with an alone-vs-within-larger-set side channel and a
configurable window of surrounding token context, and surfaces events via API, WebSocket, and a panel on
the Clusters page.

### Problem Statement
Authoring asks "what patterns should we monitor for?" — but nothing today observes a cluster as a UNIT
in real traffic. Existing Feature Monitoring reports per-feature activations of only the last forward
pass of a generation; mid-generation co-activations are invisible, there is no cluster-level predicate,
no persistence, and no surrounding text to interpret an event.

### Feature Goals
1. Catch every co-activation moment across the whole generation, not just the final token (BR-011).
2. Alone-vs-within distinction recorded honestly (best-effort v1, never fabricated) (BR-011).
3. ±K tokens of context per event so events are interpretable (BR-012 + user addition).
4. Bounded, queryable persistence; retrievable via API/UI/WS (BR-012).
5. Observable, bounded overhead; opt-in; never degrades un-armed serving (BR-013).

### User Value Proposition
"I armed my 'newsletter' cluster; overnight traffic produced 14 events — here's exactly where all five
members fired together, with the sentences around each moment."

### Connection to Project Objectives
Closes the BRD's authoring loop objective ("sense when members co-fire in real traffic — feeding
observation back into authoring").

---

## 2. User Stories & Scenarios

#### US-11.1: Arm a cluster
**As a** user with an imported cluster
**I want to** toggle sensing on for it
**So that** subsequent traffic is watched for co-activation.

**Acceptance Criteria:**
- [ ] Sensing toggle on the Clusters page (per cluster); off by default
- [ ] Arming activates on cluster activation and disarms on deactivation/SAE detach
- [ ] Status endpoint reports armed cluster, thresholds, and overhead

#### US-11.2: Review events
**As a** researcher
**I want to** list events with fired members, flags, and token context
**So that** I can learn the cluster's real-world firing patterns.

**Acceptance Criteria:**
- [ ] Events carry: timestamp, request id, prefill/decode phase, token span, fired members with peak
      activations, quorum stats, alone/within field (or null), context text, human summary
- [ ] API supports profile filter, limit, since; UI panel lists newest-first with live WS updates

#### US-11.3: Bounded and safe
**As an** operator
**I want** retention caps and observable overhead
**So that** sensing can run indefinitely without harming serving.

**Acceptance Criteria:**
- [ ] Per-cluster event cap + age pruning enforced automatically
- [ ] `sensing_overhead_ms` visible in status; warn logged above threshold
- [ ] Un-armed requests: zero added work on the hot path

#### Edge Cases
**EC-11.1: Long co-firing span** — **Trigger:** members co-fire for 30 consecutive positions.
**Behavior:** debounced into ONE event with a token span (start..end) and peak stats.
**EC-11.2: Event flood** — **Trigger:** pathological prompt fires the cluster everywhere.
**Behavior:** per-request event cap (default 20); evaluation stops, flag `truncated` on the last event.
**EC-11.3: CBM-routed request** — **Behavior:** sensing-armed ⇒ forced serial (default); if the
operator disables forcing, CBM requests are simply NOT sensed — never approximated.
**EC-11.4: Members without max_activation** — **Behavior:** threshold falls back to act>θ_floor
(degenerate act>0 when floor=0); recorded in status so results are interpretable.
**EC-11.5: context_tokens=0** — **Behavior:** events persist without text (metadata only).
**EC-11.6: Embeddings requests** — **Behavior:** already excluded (suppressed hook context).

---

## 3. Functional Requirements

### Detection (FR-11.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| SEN-D1 | Detection shall run per forward pass over ALL token positions (prefill + decode + speculative verification), only when armed | Must |
| SEN-D2 | 'Fired' per member: act > max(θ_floor, ε·max_activation) with ε=0.1, θ_floor=0.0 defaults; per-cluster overrides via cluster_meta.sensing | Must |
| SEN-D3 | Event: fired_count ≥ min_k (default max(2, ceil(0.3·m))); consecutive positions debounce to spans | Must |
| SEN-D4 | Detection cost decoupled from monitoring: member-only encode (≤20 columns), armed-only | Must |
| SEN-D5 | Per-request event cap (default 20) with truncation flag | Must |

### Recording (FR-11.2, FR-11.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| SEN-R1 | Alone-vs-within: `ambient_fired_count` populated when full-width monitoring co-runs; NULL otherwise (documented best-effort) | Must |
| SEN-R2 | ±K token context (SENSING_CONTEXT_TOKENS=16 default, hard max 64; K=0 ⇒ no text) decoded OFF the hot path post-generation | Must |
| SEN-R3 | Event attaches to the token being READ at the position (sampled-token attribution explicitly not claimed) | Must |
| SEN-R4 | Each event carries a ≤300-char human summary | Should |

### Persistence & Surfacing (FR-11.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| SEN-P1 | Events persist in `sensing_events` (migration 008) with FK CASCADE to profiles | Must |
| SEN-P2 | Retention: per-cluster cap (default 1000) + age pruning (default 7 days), enforced on flush and read | Must |
| SEN-P3 | API: status, list (profile_id/limit/since), enable/disable per cluster, clear | Must |
| SEN-P4 | WS event `sensing:event` on each recorded event (throttled like monitoring) | Must |
| SEN-P5 | Clusters-page sensing panel: toggle, live event list, event detail (members, context) | Must |

### Safety (FR-11.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| SEN-S1 | Armed requests route serial (SENSING_FORCE_SERIAL=true default); non-forced ⇒ CBM requests unsensed, never approximated | Must |
| SEN-S2 | `sensing_overhead_ms` accumulated per request, surfaced in status; warn above SENSING_MAX_OVERHEAD_MS=5.0 | Must |
| SEN-S3 | Zero hot-path work when un-armed (single boolean check in the hook) | Must |

---

## 4. Data Requirements

Migration `008_create_sensing_events_table.py`:

```sql
CREATE TABLE sensing_events (
  id            VARCHAR(50) PRIMARY KEY,          -- 'sev_' + hex
  profile_id    VARCHAR(50) NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  request_id    VARCHAR(64) NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  phase         VARCHAR(10) NOT NULL,             -- 'prefill' | 'decode'
  token_start   INTEGER NOT NULL,
  token_end     INTEGER NOT NULL,
  fired_members JSONB NOT NULL,                   -- [{feature_idx, peak_act, label?}]
  fired_count   INTEGER NOT NULL,
  member_count  INTEGER NOT NULL,
  score         FLOAT NOT NULL,                   -- max(act/θ) over fired members
  ambient_fired_count INTEGER NULL,               -- alone/within side channel (best-effort)
  truncated     BOOLEAN NOT NULL DEFAULT FALSE,
  context_text  TEXT NULL,
  context_token_ids JSONB NULL,
  summary       VARCHAR(300) NOT NULL
);
CREATE INDEX idx_sensing_events_profile ON sensing_events (profile_id, created_at DESC);
CREATE INDEX idx_sensing_events_request ON sensing_events (request_id);
```

## 5. API Specifications

New router `/api/sensing` (`millm/api/routes/management/sensing.py`):
- `GET /api/sensing/status` — armed cluster, thresholds (incl. fallback mode per EC-11.4), overhead stats
- `GET /api/sensing/events?profile_id=&limit=&since=` — newest-first
- `POST /api/sensing/clusters/{profile_id}/enable` · `POST .../disable` — sets `profiles.sensing_enabled`; arms/disarms live if that cluster is active
- `DELETE /api/sensing/events?profile_id=` — clear
WS: `sensing:event` broadcast (payload = event summary fields, not full context text).

## 6. UI Requirements
Clusters page (Feature 8) gains: SensingToggle wired to enable/disable, a sensing panel (event list w/
live updates, phase/span/score chips, expandable detail with member table + context text), and status
strip (armed cluster + overhead). Components under `components/clusters/sensing/`.

## 7. Non-Functional Requirements
- Un-armed: one boolean per hook call. Armed: one (seq,d_in)×(d_in,≤20) matmul per pass.
- Context decode strictly post-generation.
- Event payload text is user content: retention caps are the privacy control; documented.

## 8. Dependencies
- Feature 8: cluster rows (`sensing_enabled`, members with max_activation in cluster_meta).
- Feature 7/5 heritage: SAE hook (`sae_hooker.hook_fn`), suppressed() context, WS emitter pattern.
- Serial request queue (request boundaries for begin/collect).

## 9. Success Criteria
1. Scripted prompt panel with known co-firing ground truth: 100% event capture, correct spans/quorum.
2. Alone/within field correct when monitoring co-runs; NULL otherwise.
3. Context windows match expected tokens at prefill and decode positions (incl. early-stop streams).
4. Overhead within budget on armed requests; zero measurable delta un-armed.
5. Retention caps enforced under sustained event generation.

## 10. Testing Requirements
- Unit: threshold/quorum math incl. fallbacks, debounce spans, per-request cap + truncated flag,
  offset accounting (prefill/decode/speculative shapes), summary builder, repository retention.
- Integration: arm→generate→events persisted with correct spans + context; WS emission; enable/disable
  lifecycle incl. SAE detach; serial forcing; CBM-unsensed path; overhead accumulator.
- E2E (post-deploy): Clusters-page sensing panel live flow.

## 11. Rollout & Migration
Migration 008 additive. Feature dormant until a cluster is armed.

## 12. Out of Scope
Full-encode exclusivity tier (precise alone/within); per-token attribution of SAMPLED tokens;
sensing on CBM; multi-cluster simultaneous arming (v1: the active cluster only); feeding events back
into miStudio automatically (future).

## 13. Open Questions
None blocking (granularity/retention/context decided — see §15).

## 14. Documentation Requirements
Manual: Clusters page sensing section (semantics: what 'fired' means, attribution convention,
alone/within caveat, retention + privacy note).

## 15. Decisions from Clarifying Questions
1. **Granularity/retention: per-event summaries with bounded retention** (user decision 2026-07-16);
   full per-token traces and memory-ring-only rejected.
2. **Token context: configurable ±K window per event** (user addition, 2026-07-16) — default 16, max 64,
   K=0 supported.
3. Serial-only v1 with forced serial default (design consequence of CBM batch-attribution limits).
4. Attribution convention: token being READ; sampled-token attribution explicitly out of scope
   (honesty-first design default).
