# Feature PRD: Feature Monitoring

## miLLM Feature 5

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**Feature Priority:** Secondary (P2)
**References:**
- Project PRD: `000_PPRD|miLLM.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Feature Overview

### Feature Name
**Feature Monitoring** - Real-time observation of SAE feature activations during inference.

### Brief Description
Feature Monitoring enables users to observe which SAE features activate during model inference. This provides visibility into the model's internal representations, helping users understand model behavior, verify steering effects, and identify relevant features for experimentation.

### Problem Statement
Users working with SAE steering need visibility into what's happening:
- Which features activate for different inputs?
- Is steering actually changing the activations?
- What features correlate with certain behaviors?
- How strong are feature activations?

### Feature Goals
1. **Real-Time Visibility:** Stream feature activations during inference
2. **Selective Monitoring:** Choose specific features to watch
3. **Activation Statistics:** Min/max/average for monitored features
4. **Historical View:** See activation history for analysis
5. **Embeddings Support:** Monitor activations on embeddings endpoint

### User Value Proposition
Users can see exactly which features activate during inference, providing insight into model behavior and verification that steering is working as intended. This closes the feedback loop between steering adjustments and model outputs.

### Connection to Project Objectives
- **BO-4:** Support interpretability research
- **FR-4.1 through FR-4.4:** Direct implementation requirements

---

## 2. User Stories & Scenarios

### Primary User Stories

#### US-5.1: Monitor Specific Features
**As a** researcher testing a hypothesis
**I want to** watch specific features during generation
**So that** I can verify my assumptions about feature behavior

**Acceptance Criteria:**
- [ ] Add feature indices to watch list
- [ ] See real-time activation values during generation
- [ ] Values update with each token
- [ ] Can add/remove features from watch list

#### US-5.2: View Activation History
**As a** user analyzing feature patterns
**I want to** see activation history over time
**So that** I can identify patterns and correlations

**Acceptance Criteria:**
- [ ] Display activation values for last N tokens
- [ ] Show timeline visualization
- [ ] Export history as JSON/CSV
- [ ] Clear history option

#### US-5.3: Monitor on Embeddings
**As a** user building semantic search
**I want to** capture feature activations on embeddings
**So that** I can analyze document representations

**Acceptance Criteria:**
- [ ] Embeddings endpoint triggers monitoring
- [ ] Activations captured for input text
- [ ] Can view activations for each embedding request
- [ ] Monitoring doesn't slow embedding generation

#### US-5.4: Activation Statistics
**As a** user understanding feature ranges
**I want to** see activation statistics
**So that** I can set appropriate steering values

**Acceptance Criteria:**
- [ ] Show min, max, mean for each feature
- [ ] Update statistics as new activations arrive
- [ ] Reset statistics option
- [ ] Per-session statistics

#### US-5.5: Pause/Resume Monitoring
**As a** user managing resource usage
**I want to** pause monitoring
**So that** I can reduce overhead when not needed

**Acceptance Criteria:**
- [ ] Toggle button for monitoring on/off
- [ ] Pausing stops activation capture
- [ ] Resuming continues capture
- [ ] Configuration preserved when paused

### Edge Cases and Error Scenarios

#### EC-5.1: No SAE Attached
- **Trigger:** Try to monitor without SAE
- **Behavior:** Disable monitoring UI, show message
- **Message:** "Attach an SAE to enable feature monitoring"

#### EC-5.2: No Features Selected
- **Trigger:** Enable monitoring with empty watch list
- **Behavior:** Allow but show warning
- **Message:** "No features selected. Add features to monitor."

#### EC-5.3: Too Many Features
- **Trigger:** Select >100 features to monitor
- **Behavior:** Show warning about performance
- **Message:** "Monitoring many features may impact performance"

#### EC-5.4: High-Frequency Updates
- **Trigger:** Fast token generation
- **Behavior:** Throttle WebSocket updates
- **Strategy:** Batch updates, max 10 per second

---

## 3. Functional Requirements

### Activation Capture (FR-4.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| MON-C1 | System shall capture feature activations during inference | Must |
| MON-C2 | System shall capture activations for selected features only | Must |
| MON-C3 | System shall capture per-token activations for streaming | Should |
| MON-C4 | System shall capture mean activation across sequence | Must |

### Monitoring API (FR-4.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| MON-A1 | System shall expose activation data via WebSocket | Must |
| MON-A2 | System shall support REST endpoint for current values | Should |
| MON-A3 | System shall batch updates for performance | Must |
| MON-A4 | System shall throttle updates to max 10/second | Must |

### Embeddings Monitoring (FR-4.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| MON-E1 | System shall capture activations on embeddings requests | Must |
| MON-E2 | System shall store embeddings activations separately | Should |
| MON-E3 | System shall not block embeddings response | Must |

### Feature Selection (FR-4.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| MON-S1 | System shall allow selecting specific features to monitor | Must |
| MON-S2 | System shall support selecting features by index | Must |
| MON-S3 | System shall support selecting all features (with warning) | Should |
| MON-S4 | System shall persist selection across requests | Must |

### Input/Output Specifications

#### Set Monitored Features Request
```typescript
interface SetMonitoringRequest {
  features: number[];           // Feature indices to monitor
  enabled?: boolean;            // Enable/disable monitoring
  capture_all?: boolean;        // Monitor all features (expensive!)
}
```

#### Activation Update (WebSocket)
```typescript
interface ActivationUpdate {
  timestamp: string;            // ISO timestamp
  request_id: string;           // Associated request
  features: Record<number, number>;  // feature_index → activation value
  position?: number;            // Token position (if streaming)
  type: "token" | "mean" | "embeddings";
}
```

#### Monitoring State Response
```typescript
interface MonitoringState {
  enabled: boolean;
  monitored_features: number[];
  capture_all: boolean;
  history_size: number;
  statistics: Record<number, {
    min: number;
    max: number;
    mean: number;
    count: number;
  }>;
}
```

---

## 4. User Experience Requirements

### Monitoring Interface

#### Feature Selection
- Multi-select input for feature indices
- Range input (e.g., "100-200")
- Quick presets for common ranges
- Clear all button

#### Activation Display
- Real-time value update
- Color-coded by magnitude
- Sparkline for recent history
- Expandable detail view

#### Statistics Panel
- Per-feature min/max/mean
- Global statistics
- Update frequency indicator
- Reset button

### Visual Design

- Live indicator (pulsing dot when active)
- Throttle indicator if updates limited
- Memory usage display
- Historical chart (last 100 values)

---

## 5. Technical Constraints

### From ADR
- **Protocol:** WebSocket for real-time updates
- **Batching:** Required to prevent flooding
- **Throttling:** Max 10 updates per second

### Performance Impact
- Monitoring captures activations in SAE forward pass
- Selected features only (not full tensor copy)
- Detached tensors (no gradient tracking)
- CPU copy for WebSocket serialization

### Memory Constraints
- History limited to configurable size (default 100)
- Statistics use running computation (O(1) memory)
- Per-request activation stored temporarily

---

## 6. API Specifications

### Management API Endpoints

#### GET /api/monitoring
Get current monitoring state.
```json
Response: {
  "enabled": true,
  "monitored_features": [100, 200, 300],
  "capture_all": false,
  "history_size": 45,
  "statistics": {
    "100": {"min": 0.1, "max": 5.2, "mean": 2.1, "count": 45}
  }
}
```

#### POST /api/monitoring/configure
Configure monitoring settings.
```json
Request: {
  "features": [100, 200, 300],
  "enabled": true
}

Response: {
  "enabled": true,
  "monitored_features": [100, 200, 300]
}
```

#### POST /api/monitoring/enable
Enable/disable monitoring.
```json
Request: {"enabled": true}
Response: {"enabled": true}
```

#### DELETE /api/monitoring/history
Clear activation history.
```json
Response: {"cleared": true, "cleared_count": 45}
```

### WebSocket Events

#### activation_update
```json
{
  "event": "activation_update",
  "data": {
    "timestamp": "2026-01-30T12:00:00Z",
    "request_id": "req-abc123",
    "features": {"100": 2.5, "200": -0.3, "300": 4.1},
    "type": "token",
    "position": 5
  }
}
```

---

## 7. Non-Functional Requirements

### Performance

| Requirement | Target |
|-------------|--------|
| Capture overhead | <5% of inference time |
| WebSocket latency | <50ms |
| Update throttle | Max 10/second |
| Memory per feature | <1KB |

### Reliability

| Requirement | Target |
|-------------|--------|
| Data accuracy | Exact values (no approximation) |
| No inference impact | Monitoring failures don't affect output |
| Graceful degradation | Disable if resource constrained |

---

## 8. Feature Boundaries (Non-Goals)

### NOT Included in v1.0

| Non-Goal | Rationale |
|----------|-----------|
| Feature correlation analysis | Analysis tool, not monitoring |
| Automatic feature discovery | Requires separate tooling |
| Long-term persistence | Keep lightweight, export for analysis |
| Cross-request comparison | UI complexity |

---

## 9. Dependencies

### Feature Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| SAE Management | Internal | Feature 3 (Required) |
| LoadedSAE monitoring methods | Internal | Feature 3 |

---

## 10. Success Criteria

### Completion Criteria
- [ ] Can select features to monitor
- [ ] Activations stream in real-time
- [ ] Statistics computed correctly
- [ ] Embeddings monitoring works
- [ ] WebSocket updates throttled
- [ ] History exportable

---

**Document Status:** Complete
**Next Document:** `005_FTDD|Feature_Monitoring.md`
