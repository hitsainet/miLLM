# Feature PRD: SAE Management

## miLLM Feature 3

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
**SAE Management** - Download, attach, and manage Sparse Autoencoders for model interpretability.

### Brief Description
SAE Management enables users to download Sparse Autoencoders from Hugging Face, attach them to specific layers of loaded models, and manage the SAE lifecycle. This feature is the bridge between having a running model and being able to apply feature steering and monitoring - the core value proposition of miLLM.

### Problem Statement
Users who want to apply SAE steering to their models face several challenges:
- SAEs must be downloaded from various sources and properly cached
- Attaching an SAE to a model requires understanding layer architecture
- Model-SAE compatibility must be verified before attachment
- Dynamically attaching/detaching SAEs without server restarts is complex
- Memory management becomes critical with model + SAE in VRAM

### Feature Goals
1. **Seamless SAE Acquisition:** Download SAEs from Hugging Face with minimal friction
2. **Safe Attachment:** Validate SAE-model compatibility before attachment
3. **Dynamic Operations:** Attach/detach SAEs without server restarts
4. **Resource Awareness:** Track and display SAE memory usage
5. **Layer Targeting:** Support attachment to specific model layers

### User Value Proposition
Users can easily download compatible SAEs and attach them to their running models, enabling the feature steering and monitoring capabilities that make miLLM unique. The process is streamlined with automatic compatibility checking and clear feedback.

### Connection to Project Objectives
- **BO-1:** Enable practical SAE steering in local inference
- **BO-4:** Support interpretability research
- **FR-2.1 through FR-2.6:** Direct implementation requirements

---

## 2. User Stories & Scenarios

### Primary User Stories

#### US-3.1: Download SAE from Hugging Face
**As a** researcher wanting to experiment with steering
**I want to** download an SAE from Hugging Face
**So that** I can attach it to my loaded model

**Acceptance Criteria:**
- [ ] Enter Hugging Face SAE repository ID (e.g., `jbloom/gemma-2-2b-res-jb`)
- [ ] System validates the repository exists
- [ ] Download progress is displayed in real-time
- [ ] SAE is cached locally after download
- [ ] SAE metadata is displayed (feature count, layer, dimensions)

#### US-3.2: Attach SAE to Model
**As a** user with a loaded model
**I want to** attach a downloaded SAE to a specific layer
**So that** I can enable feature steering

**Acceptance Criteria:**
- [ ] Select from downloaded SAEs
- [ ] Choose target layer for attachment
- [ ] System validates model-SAE compatibility
- [ ] Attachment succeeds with confirmation message
- [ ] Model continues to function during attachment
- [ ] UI shows SAE status as "Attached"

#### US-3.3: Detach SAE from Model
**As a** user who wants to compare steered vs unsteered outputs
**I want to** detach the SAE without restarting
**So that** I can see baseline model behavior

**Acceptance Criteria:**
- [ ] Click detach button on attached SAE
- [ ] SAE removes its hooks from model
- [ ] Model continues functioning normally
- [ ] Memory is freed (VRAM reduction visible)
- [ ] Can re-attach the same SAE later

#### US-3.4: View SAE Metadata
**As a** user selecting an SAE
**I want to** see detailed information about the SAE
**So that** I can verify it's appropriate for my use case

**Acceptance Criteria:**
- [ ] Display feature count (e.g., 16,384)
- [ ] Display trained layer (e.g., layer 12)
- [ ] Display input/output dimensions
- [ ] Display training model (e.g., gemma-2-2b)
- [ ] Display estimated memory requirement
- [ ] Display SAE format (SAELens, etc.)

#### US-3.5: Delete Cached SAE
**As a** user managing disk space
**I want to** delete downloaded SAEs I no longer need
**So that** I can free up storage

**Acceptance Criteria:**
- [ ] Cannot delete SAE while attached
- [ ] Confirmation dialog before deletion
- [ ] SAE files removed from cache
- [ ] SAE removed from downloaded list
- [ ] Disk space freed

### Secondary User Scenarios

#### US-3.6: Resume Interrupted Download
**Scenario:** Download interrupted due to network issue
- Download progress is saved
- Resuming shows previous progress
- Only remaining data downloaded
- Corrupted partial files detected and cleaned up

#### US-3.7: Memory Warning on Attachment
**Scenario:** SAE attachment would exceed available VRAM
- System estimates total memory need
- Warning displayed with current/required VRAM
- User can proceed or cancel
- Suggestion to use quantization if applicable

### Edge Cases and Error Scenarios

#### EC-3.1: No Model Loaded
- **Trigger:** Attempt to attach SAE with no model loaded
- **Behavior:** Disable attach button, show message
- **Message:** "Load a model before attaching an SAE"

#### EC-3.2: Incompatible SAE Dimensions
- **Trigger:** SAE trained on different model architecture
- **Behavior:** Block attachment, show error
- **Message:** "SAE input dimension (2048) does not match model layer dimension (3072)"

#### EC-3.3: Layer Out of Range
- **Trigger:** Specified layer doesn't exist in model
- **Behavior:** Block attachment, show error
- **Message:** "Layer 50 does not exist. Model has 26 layers (0-25)"

#### EC-3.4: SAE Already Attached
- **Trigger:** Attempt to attach second SAE (v1.0 limitation)
- **Behavior:** Block attachment, offer to replace
- **Message:** "Only one SAE can be attached at a time. Detach current SAE first or replace it."

#### EC-3.5: OOM During Attachment
- **Trigger:** Insufficient VRAM for SAE
- **Behavior:** Rollback attachment, clear SAE from memory
- **Message:** "Insufficient GPU memory for SAE. Try a quantized model or smaller SAE."

#### EC-3.6: Corrupted SAE File
- **Trigger:** Downloaded SAE file is corrupted
- **Behavior:** Detect and report, offer redownload
- **Message:** "SAE file appears corrupted. Delete and redownload?"

---

## 3. Functional Requirements

### SAE Download (FR-2.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| SAE-D1 | System shall download SAEs from Hugging Face by repository ID | Must |
| SAE-D2 | System shall display download progress in real-time | Must |
| SAE-D3 | System shall support resume for interrupted downloads | Should |
| SAE-D4 | System shall cache downloaded SAEs locally | Must |
| SAE-D5 | System shall validate SAE file integrity after download | Must |
| SAE-D6 | System shall support private repositories via HF_TOKEN | Should |

### SAE Attachment (FR-2.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| SAE-A1 | System shall attach SAE to specified model layer | Must |
| SAE-A2 | System shall validate SAE-model compatibility before attachment | Must |
| SAE-A3 | System shall only allow one SAE attachment at a time (v1.0) | Must |
| SAE-A4 | System shall maintain model inference during attachment | Should |
| SAE-A5 | System shall track and display SAE attachment status | Must |

### SAE Detachment (FR-2.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| SAE-T1 | System shall detach SAE without server restart | Must |
| SAE-T2 | System shall free GPU memory on detachment | Must |
| SAE-T3 | System shall gracefully handle in-flight requests during detachment | Must |
| SAE-T4 | System shall allow re-attachment after detachment | Must |

### SAE Caching (FR-2.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| SAE-C1 | System shall store SAEs in configurable cache directory | Must |
| SAE-C2 | System shall list all cached SAEs | Must |
| SAE-C3 | System shall allow deletion of cached SAEs | Must |
| SAE-C4 | System shall display SAE cache disk usage | Should |
| SAE-C5 | System shall validate cached SAEs on startup | Should |

### SAE Format Support (FR-2.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| SAE-F1 | System shall support SAELens format SAEs | Must |
| SAE-F2 | System shall parse SAE configuration metadata | Must |
| SAE-F3 | System shall detect SAE format automatically | Should |
| SAE-F4 | Architecture shall allow future format additions | Should |

### Future Multi-SAE Architecture (FR-2.6)

| ID | Requirement | Priority |
|----|-------------|----------|
| SAE-M1 | Architecture shall support future multi-SAE attachment | Should |
| SAE-M2 | Database schema shall allow multiple SAE-layer mappings | Should |

### Input/Output Specifications

#### Download SAE Request
```typescript
interface DownloadSAERequest {
  repository_id: string;      // e.g., "jbloom/gemma-2-2b-res-jb"
  revision?: string;          // Git revision, default: "main"
  force_redownload?: boolean; // Re-download even if cached
}
```

#### SAE Metadata Response
```typescript
interface SAEMetadata {
  id: string;                 // Unique identifier
  repository_id: string;      // HuggingFace repo
  name: string;               // Display name
  format: "saelens" | "other";
  d_in: number;               // Input dimension
  d_sae: number;              // SAE hidden dimension (feature count)
  trained_on: string;         // Model trained on
  trained_layer: number;      // Layer trained on
  file_size_bytes: number;
  cache_path: string;         // Local path
  created_at: string;         // ISO timestamp
}
```

#### Attach SAE Request
```typescript
interface AttachSAERequest {
  sae_id: string;             // ID of downloaded SAE
  layer: number;              // Target layer (0-indexed)
  validate?: boolean;         // Run compatibility check (default: true)
}
```

#### Attachment Status Response
```typescript
interface AttachmentStatus {
  is_attached: boolean;
  sae_id: string | null;
  layer: number | null;
  attached_at: string | null; // ISO timestamp
  memory_usage_mb: number | null;
}
```

---

## 4. User Experience Requirements

### SAE Download Experience

#### Progress Feedback
- Show download percentage with progress bar
- Display download speed (MB/s)
- Show estimated time remaining
- Allow cancellation during download

#### Post-Download
- Automatic metadata extraction
- Display extracted information immediately
- Clear success/failure indication

### SAE Attachment Experience

#### Pre-Attachment Checks
- Memory availability indicator
- Compatibility validation result
- Clear proceed/cancel options

#### During Attachment
- Loading indicator with status messages
- "Loading SAE weights..."
- "Installing hooks..."
- "Validating attachment..."

#### Post-Attachment
- Success message with SAE name
- Updated status indicators in UI
- Memory usage display updated

### SAE List/Management Experience

#### SAE Cards
Each cached SAE displays:
- Name and repository
- Feature count (d_sae)
- Trained layer
- Status badge (Attached/Detached/Downloading)
- Size on disk
- Actions (Attach/Detach/Delete)

#### Empty State
When no SAEs downloaded:
- Helpful message explaining SAEs
- Quick link to popular SAE repositories
- Example repository IDs

---

## 5. Data Requirements

### Database Schema

```sql
-- SAE registry table
CREATE TABLE saes (
    id VARCHAR(50) PRIMARY KEY,
    repository_id VARCHAR(200) NOT NULL,
    revision VARCHAR(100) DEFAULT 'main',
    name VARCHAR(200) NOT NULL,
    format VARCHAR(50) DEFAULT 'saelens',
    d_in INTEGER NOT NULL,
    d_sae INTEGER NOT NULL,
    trained_on VARCHAR(200),
    trained_layer INTEGER,
    file_size_bytes BIGINT,
    cache_path VARCHAR(500) NOT NULL,
    status VARCHAR(20) DEFAULT 'cached',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT status_check CHECK (status IN ('downloading', 'cached', 'attached', 'error'))
);

-- Index for common queries
CREATE INDEX idx_saes_status ON saes(status);
CREATE INDEX idx_saes_repository ON saes(repository_id);

-- SAE attachment tracking
CREATE TABLE sae_attachments (
    id SERIAL PRIMARY KEY,
    sae_id VARCHAR(50) REFERENCES saes(id),
    model_id VARCHAR(50) REFERENCES models(id),
    layer INTEGER NOT NULL,
    attached_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    detached_at TIMESTAMP WITH TIME ZONE,
    memory_usage_mb INTEGER,
    is_active BOOLEAN DEFAULT TRUE
);

-- Only one active attachment allowed in v1.0
CREATE UNIQUE INDEX idx_active_attachment ON sae_attachments(is_active) WHERE is_active = TRUE;
```

### Data Validation Rules

| Field | Validation |
|-------|------------|
| repository_id | Non-empty, valid HuggingFace format (user/repo) |
| layer | 0 to model.num_layers - 1 |
| d_in | Must match model's hidden dimension at layer |
| d_sae | Positive integer |
| format | Must be supported format |

### Memory Management

- Track SAE memory usage during attachment
- Report total VRAM (model + SAE)
- Trigger warning at 90% VRAM utilization
- Provide recommendations when near limit

---

## 6. Technical Constraints

### From ADR
- **Backend:** Python 3.11+ / FastAPI
- **SAE Format:** SAELens as primary format
- **ML Framework:** PyTorch, Transformers integration
- **Cache:** Configurable directory, follows HuggingFace conventions

### Hooking Mechanism
- Use PyTorch forward hooks for activation interception
- Hook installed at specified layer's output
- Hook must be removable for detachment
- Thread-safe for concurrent inference

### Memory Constraints
- SAE typically requires 2x d_in × d_sae × 4 bytes (encoder + decoder, FP32)
- For Gemma-2-2b with 16K features: ~500MB
- Must fit alongside model in VRAM
- Consider FP16 SAE loading option

### Compatibility Matrix

| SAE Source | Model | Compatible |
|------------|-------|------------|
| gemma-2-2b SAE | gemma-2-2b | Yes |
| gemma-2-2b SAE | gemma-2-9b | No (dim mismatch) |
| Layer 12 SAE | Layer 12 | Yes (default) |
| Layer 12 SAE | Layer 20 | Yes (with warning) |

---

## 7. API Specifications

### Management API Endpoints

#### GET /api/saes
List all cached SAEs.
```json
Response: {
  "saes": [
    {
      "id": "sae-gemma-2-2b-layer12",
      "name": "Gemma 2 2B Layer 12 SAE",
      "repository_id": "jbloom/gemma-2-2b-res-jb",
      "d_sae": 16384,
      "trained_layer": 12,
      "status": "cached",
      "file_size_mb": 512.5
    }
  ],
  "attachment": {
    "is_attached": false,
    "sae_id": null
  }
}
```

#### POST /api/saes/download
Download SAE from HuggingFace.
```json
Request: {
  "repository_id": "jbloom/gemma-2-2b-res-jb",
  "revision": "main"
}

Response: {
  "status": "downloading",
  "sae_id": "sae-abc123",
  "progress_key": "download-sae-abc123"
}
```

#### POST /api/saes/{sae_id}/attach
Attach SAE to loaded model.
```json
Request: {
  "layer": 12
}

Response: {
  "status": "attached",
  "sae_id": "sae-abc123",
  "layer": 12,
  "memory_usage_mb": 512.5
}
```

#### POST /api/saes/{sae_id}/detach
Detach SAE from model.
```json
Response: {
  "status": "detached",
  "memory_freed_mb": 512.5
}
```

#### DELETE /api/saes/{sae_id}
Delete cached SAE.
```json
Response: {
  "status": "deleted",
  "disk_freed_mb": 512.5
}
```

#### GET /api/saes/attachment
Get current attachment status.
```json
Response: {
  "is_attached": true,
  "sae_id": "sae-abc123",
  "layer": 12,
  "attached_at": "2026-01-30T10:00:00Z",
  "memory_usage_mb": 512.5
}
```

### WebSocket Events

#### Download Progress
```json
{
  "event": "sae_download_progress",
  "data": {
    "sae_id": "sae-abc123",
    "percent": 45,
    "downloaded_mb": 230,
    "total_mb": 512,
    "speed_mbps": 15.2
  }
}
```

#### Attachment Status Change
```json
{
  "event": "sae_attachment_changed",
  "data": {
    "action": "attached",
    "sae_id": "sae-abc123",
    "layer": 12
  }
}
```

---

## 8. Non-Functional Requirements

### Performance

| Requirement | Target |
|-------------|--------|
| SAE download speed | Network-limited |
| SAE attachment time | <5 seconds |
| SAE detachment time | <1 second |
| Memory tracking accuracy | ±10% |

### Reliability

| Requirement | Target |
|-------------|--------|
| Download resume | Support interruption recovery |
| Attachment rollback | Clean state on failure |
| File integrity | SHA256 verification |

### Scalability (Future-Ready)
- Architecture supports multiple SAE attachments
- Database schema allows SAE-layer mappings
- Hooking mechanism extensible to multi-layer

---

## 9. Feature Boundaries (Non-Goals)

### Explicitly NOT Included in v1.0

| Non-Goal | Rationale |
|----------|-----------|
| Multiple simultaneous SAEs | Complexity, defer to v2.0 |
| SAE training within miLLM | Out of scope, use external tools |
| Automatic layer selection | Requires research, keep manual |
| SAE format conversion | Support SAELens only initially |
| Custom SAE loading from local files | Focus on HuggingFace workflow |

### Future Enhancements (Post v1.0)
- Multi-layer SAE attachment
- SAE format auto-conversion
- Local SAE file upload
- SAE performance profiling
- Neuronpedia metadata fetch

---

## 10. Dependencies

### Feature Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| Model Management | Internal | Feature 1 (Required) |
| Database/Persistence | Internal | Core infrastructure |
| HuggingFace Download | External | huggingface_hub library |

### Library Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| huggingface_hub | >=0.20 | SAE downloading |
| safetensors | >=0.4 | SAE weight loading |
| torch | >=2.0 | Tensor operations, hooks |
| sae-lens | >=2.0 | SAE format support |

---

## 11. Success Criteria

### Quantitative Metrics

| Metric | Target |
|--------|--------|
| SAE attachment success rate | >99% |
| Attachment time | <5 seconds |
| Memory estimation accuracy | ±10% |
| Download reliability | Resume from any interruption |

### User Satisfaction Indicators
- Users can download and attach SAE in <2 minutes
- Compatibility errors prevented before attachment
- Clear memory usage visibility
- Detachment is instantaneous and clean

### Completion Criteria
- [ ] Can download SAE from HuggingFace
- [ ] Can attach SAE to any layer
- [ ] Compatibility validation works
- [ ] Detachment frees memory
- [ ] SAE cache management works
- [ ] Real-time download progress
- [ ] Error messages are actionable

---

## 12. Testing Requirements

### Unit Testing
- SAE metadata parsing
- Compatibility validation logic
- Hook installation/removal
- Cache path management

### Integration Testing
- Full download → attach → detach flow
- Memory tracking accuracy
- Database persistence
- WebSocket progress events

### Manual Testing Scenarios

```
Scenario: Complete SAE Workflow
1. Start miLLM with no SAEs cached
2. Load gemma-2-2b model
3. Download jbloom/gemma-2-2b-res-jb SAE
4. Verify download progress updates
5. Verify SAE appears in list
6. Attach SAE to layer 12
7. Verify attachment status
8. Detach SAE
9. Verify memory freed
10. Re-attach SAE
11. Delete SAE (after detach)
```

---

## 13. Implementation Considerations

### Complexity Assessment

| Component | Complexity | Risk |
|-----------|------------|------|
| HuggingFace download | Low | Network reliability |
| SAE loading | Medium | Format variations |
| Hook mechanism | Medium | Thread safety |
| Compatibility validation | Low | Dimension checks |
| Memory management | Medium | Accurate tracking |

### Recommended Implementation Order
1. SAE download and caching
2. SAE metadata parsing
3. Compatibility validation
4. Hook mechanism (attach/detach)
5. Database persistence
6. WebSocket progress events
7. Memory tracking

### Technical Challenges
- Thread-safe hook installation during inference
- Accurate VRAM tracking (PyTorch allocator)
- SAELens format compatibility
- Graceful failure during attachment

---

## 14. Open Questions

### Resolved
| Question | Resolution |
|----------|------------|
| Layer numbering (0 or 1 indexed)? | 0-indexed, matching PyTorch |
| SAE storage format? | SafeTensors, matches SAELens |

### Questions for TDD
1. How to handle SAE format variations within SAELens?
2. Should we support loading SAE to different layer than trained?
3. How to handle graceful degradation if SAE causes inference errors?

---

**Document Status:** Complete
**Next Document:** `003_FTDD|SAE_Management.md` (Technical Design Document)
**Instruction File:** `@0xcc/instruct/004_create-tdd.md`
