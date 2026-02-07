# Feature PRD: Feature Steering

## miLLM Feature 4

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**Feature Priority:** Core/MVP (P1)
**References:**
- Project PRD: `000_PPRD|miLLM.md`
- ADR: `000_PADR|miLLM.md`
- BRD: `0xcc/docs/miLLM_BRD_v1.0.md`

---

## 1. Feature Overview

### Feature Name
**Feature Steering** - Real-time adjustment of SAE feature activation strengths to influence model behavior.

### Brief Description
Feature Steering is the core differentiator of miLLM. It allows users to modify model outputs by adjusting the strength of specific SAE features during inference. This enables fine-grained behavioral control without prompt engineering or fine-tuning - simply amplify or suppress features like "formality," "technical language," or "yelling" to shape model responses.

### Problem Statement
Users wanting to influence LLM behavior face limited options:
- System prompts consume context window and are inconsistently followed
- Fine-tuning requires data, compute, and creates inflexible static models
- RLHF is opaque - users can't target specific behaviors
- No existing tool allows real-time, reversible behavioral modification

### Feature Goals
1. **Precise Control:** Adjust individual features with granular strength values
2. **Real-Time Application:** Changes take effect immediately without restart
3. **Reversible:** Enable/disable steering instantly, no permanent model changes
4. **Multi-Feature:** Adjust multiple features simultaneously for complex modifications
5. **Observable:** See how steering affects outputs in real-time

### User Value Proposition
Users can directly influence model behavior by adjusting specific interpretable features. Want the model to be more formal? Increase the formality feature. Want less verbose responses? Decrease verbosity. This provides unprecedented control over LLM outputs with instant feedback.

### Connection to Project Objectives
- **BO-1:** Enable practical SAE steering in local inference
- **BO-2:** Reduce dependency on system prompts for behavioral control
- **FR-3.1 through FR-3.5:** Direct implementation requirements

---

## 2. User Stories & Scenarios

### Primary User Stories

#### US-4.1: Adjust Single Feature
**As a** user experimenting with steering
**I want to** increase a feature's strength
**So that** I can see how it affects model output

**Acceptance Criteria:**
- [ ] Enter feature index (e.g., 1234)
- [ ] Adjust strength slider (-200.0 to +200.0)
- [ ] Changes apply to next generation
- [ ] UI shows current strength value
- [ ] Can reset to default (0.0)

#### US-4.2: Adjust Multiple Features
**As a** user creating a specific persona
**I want to** adjust several features at once
**So that** I can create complex behavioral modifications

**Acceptance Criteria:**
- [ ] Add multiple features to adjustment list
- [ ] Set different strengths for each
- [ ] All adjustments apply simultaneously
- [ ] Can add/remove features from list
- [ ] Total active features displayed

#### US-4.3: Toggle Steering On/Off
**As a** user comparing steered vs unsteered output
**I want to** quickly toggle steering
**So that** I can see the effect of my configuration

**Acceptance Criteria:**
- [ ] Single toggle switch for all steering
- [ ] Toggle is instant (no reload)
- [ ] Steering configuration preserved when off
- [ ] Visual indicator shows steering state
- [ ] Keyboard shortcut available

#### US-4.4: Search/Browse Features
**As a** user looking for specific behaviors
**I want to** search features by name/description
**So that** I can find features to adjust

**Acceptance Criteria:**
- [ ] Search by feature index
- [ ] Search by feature name (if available from Neuronpedia)
- [ ] Browse features by activation strength
- [ ] Show feature metadata (description, examples)
- [ ] Quick-add feature to adjustment list

#### US-4.5: Observe Steering Effects
**As a** user adjusting features
**I want to** see real-time effects on output
**So that** I can fine-tune my settings

**Acceptance Criteria:**
- [ ] Test prompt input area
- [ ] Generate button triggers inference
- [ ] Output shows with steering applied
- [ ] Side-by-side comparison (steered vs unsteered) optional
- [ ] Generation uses current steering values

### Secondary User Scenarios

#### US-4.6: Feature Strength Presets
**Scenario:** User wants quick strength options
- Preset buttons: Off (0), Low (±10), Medium (±50), High (±100)
- Click to set strength
- Custom value still available via slider

#### US-4.7: Feature Suggestions
**Scenario:** User discovers related features
- System suggests related features
- Based on co-activation patterns
- Easy to add to adjustment list

### Edge Cases and Error Scenarios

#### EC-4.1: No SAE Attached
- **Trigger:** Attempt steering without SAE
- **Behavior:** Disable steering controls, show message
- **Message:** "Attach an SAE to enable feature steering"

#### EC-4.2: Invalid Feature Index
- **Trigger:** Enter index > SAE feature count
- **Behavior:** Validation error, prevent addition
- **Message:** "Feature index must be 0-{max_index}"

#### EC-4.3: Extreme Steering Values
- **Trigger:** Set very high steering (>100 or <-100)
- **Behavior:** Show warning, allow with confirmation
- **Message:** "Extreme values may cause unpredictable outputs"
- **Note:** Values outside -200 to +200 are rejected by backend Pydantic validation

#### EC-4.4: Steering During Generation
- **Trigger:** Change steering while generating
- **Behavior:** Queue change, apply to next token
- **Message:** None (seamless application)

#### EC-4.5: SAE Detached During Steering
- **Trigger:** SAE detached while steering active
- **Behavior:** Disable steering, clear values, notify
- **Message:** "SAE detached. Steering has been disabled."

---

## 3. Functional Requirements

### Feature Adjustment (FR-3.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| ST-A1 | System shall allow adjustment of individual feature strengths by index | Must |
| ST-A2 | System shall support strength values from -200.0 to +200.0 | Must |
| ST-A3 | System shall support 0.1 precision for strength values | Should |
| ST-A4 | System shall display current strength for each adjusted feature | Must |
| ST-A5 | System shall allow resetting feature to default (0.0) | Must |

### Multiple Features (FR-3.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| ST-M1 | System shall support simultaneous adjustment of multiple features | Must |
| ST-M2 | System shall display count of active adjustments | Should |
| ST-M3 | System shall support batch update of multiple features | Should |
| ST-M4 | System shall allow clearing all adjustments at once | Must |

### Steering Application (FR-3.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| ST-P1 | System shall apply steering to all inference requests | Must |
| ST-P2 | System shall apply steering during token generation | Must |
| ST-P3 | System shall support steering in both streaming and non-streaming | Must |
| ST-P4 | System shall maintain steering across multiple requests | Must |

### Real-Time Adjustment (FR-3.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| ST-R1 | System shall apply steering changes without server restart | Must |
| ST-R2 | System shall apply changes to next generation request | Must |
| ST-R3 | System shall provide enable/disable toggle for all steering | Must |
| ST-R4 | System shall preserve settings when steering is disabled | Must |

### Positive/Negative Values (FR-3.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| ST-V1 | System shall support positive values for amplification | Must |
| ST-V2 | System shall support negative values for suppression | Must |
| ST-V3 | System shall clearly indicate sign in UI | Must |
| ST-V4 | System shall support zero value (no effect) | Must |

### Input/Output Specifications

#### Set Feature Steering Request
```typescript
interface SetSteeringRequest {
  feature_index: number;      // 0 to d_sae-1
  value: number;              // -200.0 to +200.0
}
```

#### Batch Steering Request
```typescript
interface BatchSteeringRequest {
  steering: Array<{
    feature_index: number;
    value: number;              // -200.0 to +200.0
  }>;
}
```

#### Steering State Response
```typescript
interface SteeringState {
  enabled: boolean;
  active_count: number;
  values: Record<number, number>;  // feature_index → value
  sae_id: string;
  sae_feature_count: number;
}
```

#### Toggle Steering Request
```typescript
interface ToggleSteeringRequest {
  enabled: boolean;
}
```

---

## 4. User Experience Requirements

### Steering Interface Design

#### Feature Entry
- Numeric input for feature index
- Autocomplete for known feature names (future)
- Validation feedback on invalid indices
- Quick-add button

#### Strength Adjustment
- Horizontal slider from -200 to +200 with 0.1 precision
- Numeric input for precise values
- Color coding: red (negative), green (positive), gray (zero)
- Preset buttons for common values
- Follows Neuronpedia-compatible strength semantics (typical values: +/-50-100 for strong observable effects)

#### Feature List
- List of currently adjusted features
- Each row shows: index, name (if known), strength, remove button
- Sortable by index or strength
- Clear all button

### Visual Feedback

#### Steering Status
- Global indicator: "Steering Active" / "Steering Off"
- Count badge: "5 features adjusted"
- Color: Green when active, gray when off

#### Per-Feature Indicators
- Strength bar visualization
- Direction arrow (↑ amplify, ↓ suppress)
- Tooltip with feature description (if available)

### Performance Requirements

| Requirement | Target |
|-------------|--------|
| Steering toggle response | <100ms |
| Feature add/remove | <100ms |
| Strength adjustment | <100ms |
| Generation with steering | Same as without steering |

---

## 5. Data Requirements

### Steering Configuration (In-Memory)

Steering values are stored in-memory on the LoadedSAE instance:
```python
# No database persistence for transient steering
# Profiles (Feature 6) handle persistence

class LoadedSAE:
    _steering_values: Dict[int, float] = {}
    _steering_enabled: bool = False
```

### Optional Request Logging

```sql
-- Log steering usage for analysis
CREATE TABLE steering_logs (
    id SERIAL PRIMARY KEY,
    sae_id VARCHAR(50) NOT NULL,
    feature_indices INTEGER[] NOT NULL,
    feature_values FLOAT[] NOT NULL,
    request_id VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Data Validation Rules

| Field | Validation |
|-------|------------|
| feature_index | Integer, 0 to sae.d_sae - 1 |
| value | Float, -200.0 to +200.0 (validated by Pydantic) |
| precision | 0.1 minimum step |

---

## 6. Technical Constraints

### From ADR
- **Backend:** Python 3.11+ / FastAPI
- **Steering Mechanism:** Applied in SAE forward pass
- **Real-Time:** No restart required for changes

### Steering Application

```
Forward Pass with Steering:

Input Activations (from model layer)
        │
        ▼
    SAE Encode
        │
        ▼
Feature Activations ──► Add Steering Vector
        │
        ▼
    SAE Decode
        │
        ▼
Output Activations (back to model)
```

### Performance Considerations
- Steering vector is pre-computed (sparse tensor)
- Only non-zero values stored
- Vector addition is O(1) with broadcasting
- No overhead when steering disabled

### Thread Safety
- Steering values modified via service methods
- Pre-computed vector rebuilt atomically
- Forward pass uses immutable vector snapshot

---

## 7. API Specifications

### Management API Endpoints

#### GET /api/steering
Get current steering state.
```json
Response: {
  "enabled": true,
  "active_count": 3,
  "values": {
    "1234": 5.0,
    "5678": -3.5,
    "9012": 2.0
  },
  "sae_id": "sae-abc123",
  "sae_feature_count": 16384
}
```

#### POST /api/steering/enable
Enable or disable steering.
```json
Request: {
  "enabled": true
}

Response: {
  "enabled": true,
  "active_count": 3
}
```

#### POST /api/steering/features
Set steering for a feature.
```json
Request: {
  "feature_index": 1234,
  "value": 5.0
}

Response: {
  "feature_index": 1234,
  "value": 5.0,
  "active_count": 4
}
```

#### POST /api/steering/features/batch
Set multiple features at once.
```json
Request: {
  "steering": [
    {"feature_index": 1234, "value": 5.0},
    {"feature_index": 5678, "value": -3.5}
  ]
}

Response: {
  "updated_count": 2,
  "active_count": 5
}
```

#### DELETE /api/steering/features/{index}
Remove steering for a feature.
```json
Response: {
  "feature_index": 1234,
  "removed": true,
  "active_count": 2
}
```

#### DELETE /api/steering/features
Clear all steering values.
```json
Response: {
  "cleared_count": 5,
  "active_count": 0
}
```

### WebSocket Events

#### Steering State Changed
```json
{
  "event": "steering_changed",
  "data": {
    "enabled": true,
    "active_count": 3,
    "changed_feature": 1234,
    "new_value": 5.0
  }
}
```

---

## 8. Non-Functional Requirements

### Performance

| Requirement | Target |
|-------------|--------|
| Steering application | <1ms overhead per forward pass |
| Value update | <10ms |
| Enable/disable toggle | <10ms |
| Batch update (100 features) | <50ms |

### Reliability

| Requirement | Target |
|-------------|--------|
| Steering consistency | All tokens use same values within request |
| Recovery from SAE detach | Graceful disable, values preserved |
| Concurrent access | Thread-safe operations |

### Usability

| Requirement | Target |
|-------------|--------|
| Learning curve | Immediate for basic use |
| Value feedback | Real-time slider updates |
| Error messages | Clear validation errors |

---

## 9. Feature Boundaries (Non-Goals)

### Explicitly NOT Included in v1.0

| Non-Goal | Rationale |
|----------|-----------|
| Feature name lookup | Requires Neuronpedia integration |
| Automatic feature suggestions | Requires analysis tooling |
| Steering strength optimization | Research feature |
| Per-request steering via API | Use profiles instead |
| Conditional steering (if/then) | Complexity |

### Future Enhancements (Post v1.0)
- Neuronpedia integration for feature names
- Feature search by activation patterns
- Steering presets library
- A/B testing framework
- Automated steering optimization

---

## 10. Dependencies

### Feature Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| SAE Management | Internal | Feature 3 (Required) |
| LoadedSAE wrapper | Internal | Feature 3 (Required) |
| Model loaded | Internal | Feature 1 (Required) |

### Technical Dependencies
- SAE must be attached before steering
- LoadedSAE implements steering methods
- InferenceService applies steering during generation

---

## 11. Success Criteria

### Quantitative Metrics

| Metric | Target |
|--------|--------|
| Steering overhead | <1ms per token |
| UI responsiveness | <100ms for all operations |
| Value precision | ±0.01 accuracy |

### User Satisfaction Indicators
- Users can adjust features within 30 seconds of learning
- Steering effects are observable in output
- Toggle provides clear before/after comparison
- Error messages enable self-service troubleshooting

### Completion Criteria
- [ ] Can set individual feature values
- [ ] Can set multiple features simultaneously
- [ ] Toggle enables/disables all steering
- [ ] Values persist until cleared
- [ ] Steering applies to all inference
- [ ] UI shows current steering state
- [ ] Validation prevents invalid inputs

---

## 12. Testing Requirements

### Unit Testing
- Steering vector computation
- Value validation
- Thread-safe operations
- Enable/disable toggle

### Integration Testing
- Steering with inference
- Multiple features combined
- SAE detach handling
- API endpoint validation

### Manual Testing Scenarios

```
Scenario: Yelling Demo
1. Load gemma-2-2b model
2. Attach Gemma-Scope SAE
3. Set feature #1234 (capitalization) to +5.0
4. Enable steering
5. Generate response to "Tell me about the weather"
6. Verify response contains more capitalization
7. Set feature to -5.0
8. Generate again
9. Verify response has less capitalization
10. Toggle steering off
11. Generate again
12. Verify normal output
```

---

## 13. Implementation Considerations

### Complexity Assessment

| Component | Complexity | Risk |
|-----------|------------|------|
| Steering value storage | Low | None |
| Vector computation | Low | None |
| API endpoints | Low | None |
| Thread safety | Medium | Race conditions |
| UI controls | Medium | UX polish |

### Recommended Implementation Order
1. Steering methods in LoadedSAE (already in Feature 3)
2. SteeringService wrapping SAE operations
3. API endpoints
4. WebSocket events
5. UI components

### Technical Challenges
- Ensuring thread-safe steering updates
- Efficient sparse vector computation
- Clear feedback for subtle steering effects

---

## 14. Open Questions

### Resolved
| Question | Resolution |
|----------|------------|
| Value range? | -200.0 to +200.0, aligned with Neuronpedia-compatible semantics (typical strong effects at +/-50-100) |
| Persistence? | In-memory only; profiles handle persistence |

### Questions for TDD
1. Should we support steering during streaming mid-generation?
2. How to handle steering when model layer dimensions don't match?
3. Should we provide steering strength recommendations?

---

**Document Status:** Complete
**Next Document:** `004_FTDD|Feature_Steering.md` (Technical Design Document)
**Instruction File:** `@0xcc/instruct/004_create-tdd.md`
