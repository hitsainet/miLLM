# Feature PRD: Model Management

## miLLM Feature 1

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**Feature Priority:** Core/MVP (P1)
**References:**
- Project PRD: `000_PPRD|miLLM.md`
- ADR: `000_PADR|miLLM.md`
- BRD: `0xcc/docs/miLLM_BRD_v1.0.md`
- UI Mockup: `0xcc/spec/miLLM_UI.jsx`

---

## 1. Feature Overview

### Feature Name
**Model Management** - Download, configure, load, and manage LLM models from HuggingFace and local sources.

### Brief Description
Model Management provides the foundational capability for miLLM to acquire, store, and load large language models. Users can download models from HuggingFace with configurable quantization, load them into GPU memory for inference, and manage their local model cache. This feature is the prerequisite for all other miLLM functionality.

### Problem Statement
Users running local LLMs face several challenges:
- Downloading models requires command-line tools and manual configuration
- Quantization for memory optimization requires separate tooling
- No unified interface to manage multiple downloaded models
- Loading/unloading models requires server restarts in many tools
- Memory requirements are often unclear until loading fails

### Feature Goals
1. **Simplify Model Acquisition:** One-click download from HuggingFace with automatic quantization
2. **Enable Memory Optimization:** Support 4-bit and 8-bit quantization via bitsandbytes
3. **Provide Visibility:** Display memory estimates before loading to prevent OOM errors
4. **Support Flexibility:** Allow both HuggingFace downloads and local model paths
5. **Ensure Reliability:** Graceful handling of downloads, loads, and errors

### User Value Proposition
Users can easily acquire and manage models without leaving the miLLM interface, with clear visibility into memory requirements and the ability to optimize for their hardware through quantization.

### Connection to Project Objectives
- **BO-1:** Foundation for SAE steering (requires loaded model)
- **BO-3:** Enables OpenAI API compatibility (requires model to serve)
- **NFR-3.1:** Supports Docker deployment (model caching)
- **FR-1.1 through FR-1.6:** Direct implementation of model management requirements

---

## 2. User Stories & Scenarios

### Primary User Stories

#### US-1.1: Download Model from HuggingFace
**As a** developer/researcher
**I want to** download a model by entering its HuggingFace repository ID
**So that** I can use it for inference without leaving the miLLM interface

**Acceptance Criteria:**
- [ ] Can enter HuggingFace repository ID (e.g., "google/gemma-2-2b")
- [ ] Can select quantization format (Q4, Q8, FP16)
- [ ] Can optionally provide HF access token for gated models
- [ ] Can enable/disable trust_remote_code per download
- [ ] Download shows real-time progress percentage
- [ ] Can cancel download in progress (partial files cleaned up)
- [ ] Success notification when download completes
- [ ] Model appears in "Your Models" list after download

#### US-1.2: Load Model into Memory
**As a** developer/researcher
**I want to** load a downloaded model into GPU memory
**So that** I can use it for inference

**Acceptance Criteria:**
- [ ] Can see estimated memory requirement before loading
- [ ] Can see current GPU memory availability
- [ ] Warning displayed if estimated memory exceeds available
- [ ] Loading shows progress indicator
- [ ] Model status changes to "Loaded" when complete
- [ ] Previously loaded model automatically unloaded (graceful)
- [ ] Server status bar updates to show loaded model name

#### US-1.3: Unload Model from Memory
**As a** developer/researcher
**I want to** unload the current model from GPU memory
**So that** I can free memory for a different model or SAE

**Acceptance Criteria:**
- [ ] Can click "Unload" on currently loaded model
- [ ] System waits for pending inference requests to complete
- [ ] GPU memory released after unload
- [ ] Model status changes to "Ready"
- [ ] Server status bar updates to show no model loaded

#### US-1.4: Load Model from Local Path
**As a** developer/researcher
**I want to** load a model from a local filesystem path
**So that** I can use models I've already downloaded or custom fine-tuned models

**Acceptance Criteria:**
- [ ] Can enter absolute path to local model directory
- [ ] System validates path exists and contains valid model files
- [ ] Can select quantization to apply at load time
- [ ] Model appears in list with "Local" source indicator
- [ ] Loading behavior same as HuggingFace models

#### US-1.5: Delete Model from Cache
**As a** developer/researcher
**I want to** delete a downloaded model from disk
**So that** I can free disk space

**Acceptance Criteria:**
- [ ] Can click delete button on any non-loaded model
- [ ] Confirmation dialog shown before deletion
- [ ] Model files removed from disk
- [ ] Model removed from "Your Models" list
- [ ] Cannot delete currently loaded model (must unload first)

#### US-1.6: Preview Model Before Download
**As a** developer/researcher
**I want to** see model details before downloading
**So that** I can verify it's the correct model and check requirements

**Acceptance Criteria:**
- [ ] Can click "Preview" to fetch model info from HuggingFace
- [ ] Shows model name, parameter count, architecture
- [ ] Shows estimated disk space and memory requirements per quantization
- [ ] Shows whether trust_remote_code is required
- [ ] Shows whether model is gated (requires token)

### Secondary User Scenarios

#### US-1.7: Handle Gated Models
**Scenario:** User attempts to download a gated model (e.g., Llama 3)
- System detects gated model status
- Prompts for HuggingFace access token
- Validates token before starting download
- Stores token securely for session (not persisted)

#### US-1.8: Resume After Server Restart
**Scenario:** Server restarts while model was loaded
- By default, previously loaded model is not auto-loaded
- Model remains in "Ready" state
- User must manually load model again
- All downloaded models preserved in cache
- **AUTO_LOAD_MODEL:** If the `AUTO_LOAD_MODEL` environment variable is set to a model name, miLLM automatically loads that model on server startup. This is useful for headless/Docker deployments where the admin UI is not immediately available.

### Edge Cases and Error Scenarios

#### EC-1.1: Insufficient Disk Space
- **Trigger:** Download started with insufficient disk space
- **Behavior:** Fail fast with clear error before download starts
- **Message:** "Insufficient disk space. Need X GB, only Y GB available."

#### EC-1.2: Insufficient GPU Memory
- **Trigger:** Load attempted with insufficient VRAM
- **Behavior:** Show warning, allow user to proceed or cancel
- **Message:** "Warning: Model requires ~X GB VRAM, only Y GB available. Loading may fail or cause system instability."

#### EC-1.3: Network Failure During Download
- **Trigger:** Network disconnects mid-download
- **Behavior:** Retry with exponential backoff (3 attempts), then fail
- **Message:** "Download failed: Network error. Please check connection and try again."
- **Cleanup:** Partial download files removed

#### EC-1.4: Invalid Repository ID
- **Trigger:** User enters non-existent HuggingFace repo
- **Behavior:** Fail on preview/download with clear error
- **Message:** "Model not found: 'repo/name' does not exist on HuggingFace."

#### EC-1.5: Invalid Local Path
- **Trigger:** User enters path that doesn't exist or isn't a valid model
- **Behavior:** Validation error before load attempt
- **Message:** "Invalid model path: Directory does not exist or does not contain valid model files."

#### EC-1.6: Model Load OOM
- **Trigger:** GPU runs out of memory during load
- **Behavior:** Catch OOM, clean up partial load, report error
- **Message:** "Out of memory loading model. Try a smaller model or higher quantization (Q4)."

#### EC-1.7: Unload with Pending Requests
- **Trigger:** Unload requested while inference in progress
- **Behavior:** Wait for pending requests (max 30s timeout), then unload
- **Message:** "Waiting for X pending requests to complete..."

---

## 3. Functional Requirements

### Model Download (FR-1.1, FR-1.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| MM-D1 | System shall download models from HuggingFace by repository ID | Must |
| MM-D2 | System shall support local filesystem paths as model source | Must |
| MM-D3 | System shall display download progress in real-time | Must |
| MM-D4 | System shall allow cancellation of in-progress downloads | Must |
| MM-D5 | System shall clean up partial files on download cancel/failure | Must |
| MM-D6 | System shall validate HuggingFace repository exists before download | Must |
| MM-D7 | System shall support HF_TOKEN for gated model access | Must |
| MM-D8 | System shall support trust_remote_code flag (opt-in per download) | Must |

### Model Quantization (FR-1.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| MM-Q1 | System shall support 4-bit quantization via bitsandbytes | Must |
| MM-Q2 | System shall support 8-bit quantization via bitsandbytes | Must |
| MM-Q3 | System shall support FP16 (full precision) loading | Must |
| MM-Q4 | System shall apply quantization at download time (save quantized weights) | Must |
| MM-Q5 | System shall display memory savings for each quantization option | Should |

### Model Caching (FR-1.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| MM-C1 | System shall cache downloaded models in configurable directory | Must |
| MM-C2 | System shall persist model metadata in database | Must |
| MM-C3 | System shall detect and use existing cached models | Must |
| MM-C4 | System shall support hard deletion of cached models | Must |
| MM-C5 | System shall track disk space used by model cache | Should |

### Model Loading (FR-1.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| MM-L1 | System shall load models into GPU memory on demand | Must |
| MM-L2 | System shall support only one loaded model at a time | Must |
| MM-L3 | System shall automatically unload previous model when loading new | Must |
| MM-L4 | System shall display loading progress indicator | Must |
| MM-L5 | System shall report load failures with actionable messages | Must |

### Memory Estimation (FR-1.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| MM-M1 | System shall estimate memory requirements before loading | Must |
| MM-M2 | System shall display current GPU memory availability | Must |
| MM-M3 | System shall warn if estimated memory exceeds available | Must |
| MM-M4 | System shall verify actual memory availability before load | Must |
| MM-M5 | Memory estimates shall be within 20% of actual usage | Should |

### Model Unloading

| ID | Requirement | Priority |
|----|-------------|----------|
| MM-U1 | System shall unload model from GPU memory on request | Must |
| MM-U2 | System shall wait for pending requests before unload (graceful) | Must |
| MM-U3 | System shall timeout graceful unload after 30 seconds | Must |
| MM-U4 | System shall release GPU memory after unload | Must |

### Input/Output Specifications

#### Download Request
```typescript
interface ModelDownloadRequest {
  source: 'huggingface' | 'local';
  repo_id?: string;           // For HuggingFace: "google/gemma-2-2b"
  local_path?: string;        // For local: "/path/to/model" (absolute path)
  quantization: 'Q4' | 'Q8' | 'FP16';
  trust_remote_code: boolean; // Default: false
  hf_token?: string;          // Optional, for gated models
  custom_name?: string;       // Optional display name
}
```

**Local model path notes:**
- When `source` is `"local"`, the `local_path` parameter must be provided as an absolute filesystem path.
- The system validates the path exists and contains valid model files (e.g., `config.json`).
- Local models are registered in the database but files remain at the original path (no copying).

#### Model Response
```typescript
interface Model {
  id: number;
  name: string;
  source: 'huggingface' | 'local';
  repo_id: string | null;
  local_path: string | null;
  params: string;             // "2.5B", "9B", etc.
  quantization: 'Q4' | 'Q8' | 'FP16';
  disk_size_mb: number;
  estimated_memory_mb: number;
  status: 'downloading' | 'ready' | 'loading' | 'loaded' | 'error';
  download_progress?: number; // 0-100, only when downloading
  error_message?: string;     // Only when status is 'error'
  created_at: string;
  loaded_at?: string;
}
```

### Business Logic and Validation Rules

1. **Repository ID Validation:** Must match pattern `owner/repo-name`
2. **Local Path Validation:** Must be absolute path, must exist, must contain config.json or model files
3. **Quantization Compatibility:** Q4/Q8 require CUDA GPU; FP16 can run on CPU (slow)
4. **Single Model Loaded:** Loading new model triggers unload of current
5. **Cannot Delete Loaded:** Must unload before deletion allowed
6. **Graceful Unload Timeout:** 30 seconds max wait for pending requests

---

## 4. User Experience Requirements

### UI Components (Reference: UI Mockup)

#### Models Page Layout
- **Download Section:** Card at top with input fields and buttons
- **Model List:** Below download section, shows all downloaded models
- **Each Model Card:** Icon, name, metadata, status badge, action buttons

#### Download Form Fields
1. **Repository ID Input:** Text field with placeholder "e.g., google/gemma-2-2b"
2. **Quantization Dropdown:** Q4 (default, recommended), Q8, FP16
3. **Access Token Input:** Password field, optional
4. **Trust Remote Code Checkbox:** Unchecked by default, with warning text
5. **Preview Button:** Secondary style
6. **Download Button:** Primary style

#### Model Card Elements
- **Icon:** Server icon in cyan circle
- **Model Name:** Bold, primary text
- **Metadata Line 1:** "2.5B params • Q4 quantization • 1.8 GB memory"
- **Metadata Line 2:** Repository ID or local path (monospace, smaller)
- **Status Badge:** "Loaded" (green), "Ready" (cyan), "Downloading" (yellow)
- **Action Buttons:**
  - Loaded: "Unload" (red outline)
  - Ready: "Load" (cyan), Delete icon
  - Downloading: Progress bar, "Cancel" button

### Interaction Patterns

#### Download Flow
1. User enters repository ID
2. User selects quantization (default Q4)
3. User optionally enters HF token
4. User optionally checks trust_remote_code
5. User clicks "Preview" (optional) → Shows model info modal
6. User clicks "Download" → Button becomes disabled, shows spinner
7. Progress bar appears in new model card
8. On complete: Success toast, card shows "Ready" status

#### Load Flow
1. User clicks "Load" on a "Ready" model
2. System shows memory estimate vs available
3. If warning needed, confirmation dialog appears
4. Loading indicator appears
5. If another model loaded, shows "Unloading previous model..."
6. On complete: Status changes to "Loaded", status bar updates

#### Unload Flow
1. User clicks "Unload" on "Loaded" model
2. If pending requests, shows "Waiting for X requests..."
3. On complete: Status changes to "Ready", status bar updates

### Responsive Design
- Download form: Stack vertically on narrow screens
- Model cards: Single column on mobile, two columns on tablet+
- Action buttons: Icon-only on mobile, icon+text on desktop

### Accessibility Requirements
- All form inputs have visible labels
- Status badges have text, not just color
- Progress bars have aria-valuenow
- Keyboard navigation for all actions
- Focus management after modal close

### Error State Display
- Inline validation errors below inputs (red text)
- Toast notifications for transient errors
- Error badge on model card for persistent errors
- "Retry" action available for failed downloads

---

## 5. Data Requirements

### Database Schema

#### models Table
```sql
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    source VARCHAR(20) NOT NULL CHECK (source IN ('huggingface', 'local')),
    repo_id VARCHAR(255),
    local_path VARCHAR(500),
    params VARCHAR(50),
    architecture VARCHAR(100),
    quantization VARCHAR(10) NOT NULL CHECK (quantization IN ('Q4', 'Q8', 'FP16')),
    disk_size_mb INTEGER,
    estimated_memory_mb INTEGER,
    cache_path VARCHAR(500) NOT NULL,
    config_json JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'ready',
    error_message TEXT,
    trust_remote_code BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT unique_repo_quant UNIQUE (repo_id, quantization),
    CONSTRAINT unique_local_path UNIQUE (local_path)
);

CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_repo_id ON models(repo_id);
```

### Data Validation Rules

| Field | Validation |
|-------|------------|
| name | Required, 1-255 chars, no special chars except `-_` |
| repo_id | Pattern: `^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$` |
| local_path | Absolute path, exists, contains model files |
| quantization | Enum: Q4, Q8, FP16 |
| disk_size_mb | Positive integer |
| estimated_memory_mb | Positive integer |

### Data Persistence

- **Model metadata:** PostgreSQL (persistent across restarts)
- **Download progress:** Redis (ephemeral, real-time updates)
- **Load state:** In-memory (not persisted; models not auto-loaded on restart)
- **Model files:** Local filesystem (configurable MODEL_CACHE_DIR)

### File Storage Structure
```
$MODEL_CACHE_DIR/
├── huggingface/
│   ├── google--gemma-2-2b--Q4/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer.json
│   └── meta-llama--Llama-3.2-3B--Q8/
│       └── ...
└── local/
    └── [symlinks to local paths for consistency]
```

---

## 6. Technical Constraints

### From ADR

- **Backend:** Python 3.11+ / FastAPI
- **ML Libraries:** PyTorch 2.0+, Transformers 4.36+, bitsandbytes 0.42+
- **Database:** PostgreSQL 14+ for metadata, Redis for real-time state
- **Async:** All I/O operations must be async
- **Error Handling:** Use MiLLMError hierarchy, structured logging

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| MODEL_CACHE_DIR | No | Directory for cached model files (default: `./data/models`) |
| HF_TOKEN | No | HuggingFace access token for gated models |
| AUTO_LOAD_MODEL | No | Model name to automatically load on server startup (e.g., `gemma-2-2b`) |

### Technology Stack Constraints

- **Quantization:** bitsandbytes requires NVIDIA GPU with CUDA
- **Model Formats:** Support safetensors and pytorch (.bin) via Transformers
- **HuggingFace:** Use huggingface_hub library for downloads
- **Memory:** torch.cuda operations for memory queries

### Performance Requirements

| Metric | Target |
|--------|--------|
| Download start latency | <2s after click |
| Model load time (2B Q4) | <30s |
| Model load time (9B Q4) | <90s |
| Memory estimate accuracy | Within 20% of actual |
| Unload completion | <5s (excluding graceful wait) |

### Security Requirements

- HF tokens stored in memory only (not persisted to database)
- trust_remote_code requires explicit opt-in per model
- Local paths validated to prevent directory traversal
- No execution of downloaded code without trust_remote_code flag

---

## 7. API/Integration Specifications

### Management API Endpoints

#### List Models
```
GET /api/models
Response: { success: true, data: Model[] }
```

#### Get Model
```
GET /api/models/{id}
Response: { success: true, data: Model }
```

#### Download Model
```
POST /api/models
Body: ModelDownloadRequest
Response: { success: true, data: Model }  // Status: downloading
```

#### Preview Model (HuggingFace)
```
POST /api/models/preview
Body: { repo_id: string, hf_token?: string }
Response: {
  success: true,
  data: {
    name: string,
    params: string,
    architecture: string,
    requires_trust_remote_code: boolean,
    is_gated: boolean,
    estimated_sizes: {
      Q4: { disk_mb: number, memory_mb: number },
      Q8: { disk_mb: number, memory_mb: number },
      FP16: { disk_mb: number, memory_mb: number }
    }
  }
}
```

#### Load Model
```
POST /api/models/{id}/load
Response: { success: true, data: Model }  // Status: loading → loaded
```

#### Unload Model
```
POST /api/models/{id}/unload
Response: { success: true, data: Model }  // Status: ready
```

#### Cancel Download
```
POST /api/models/{id}/cancel
Response: { success: true, data: null }
```

#### Delete Model
```
DELETE /api/models/{id}
Response: { success: true, data: null }
```

### WebSocket Events (Socket.IO)

#### Namespace: /progress

**Server → Client Events:**
```typescript
// Download progress
'model:download:progress': {
  model_id: number,
  progress: number,      // 0-100
  bytes_downloaded: number,
  total_bytes: number,
  speed_mbps: number
}

// Download complete
'model:download:complete': {
  model_id: number,
  model: Model
}

// Download error
'model:download:error': {
  model_id: number,
  error: { code: string, message: string }
}

// Load progress
'model:load:progress': {
  model_id: number,
  stage: 'loading_weights' | 'quantizing' | 'moving_to_gpu',
  progress: number       // 0-100
}

// Load complete
'model:load:complete': {
  model_id: number,
  model: Model
}
```

### OpenAI API Integration

Model Management populates the `/v1/models` endpoint:
```
GET /v1/models
Response: {
  object: "list",
  data: [
    {
      id: "gemma-2-2b",           // Uses model.name
      object: "model",
      created: 1706627200,
      owned_by: "miLLM"
    }
  ]
}
```

Only the currently loaded model appears in this list.

### Error Response Format
```typescript
{
  success: false,
  data: null,
  error: {
    code: string,          // e.g., "MODEL_NOT_FOUND"
    message: string,       // Human-readable message
    details: {             // Optional additional info
      model_id?: number,
      required_memory?: number,
      available_memory?: number
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| MODEL_NOT_FOUND | 404 | Model ID doesn't exist |
| MODEL_ALREADY_EXISTS | 409 | Model with same repo+quantization exists |
| MODEL_DOWNLOAD_FAILED | 502 | HuggingFace download failed |
| MODEL_LOAD_FAILED | 500 | Failed to load model into memory |
| MODEL_ALREADY_LOADED | 400 | Attempting to load already-loaded model |
| MODEL_NOT_LOADED | 400 | Attempting to unload non-loaded model |
| INSUFFICIENT_MEMORY | 507 | Not enough GPU memory |
| INSUFFICIENT_DISK | 507 | Not enough disk space |
| INVALID_REPO_ID | 400 | Repository ID format invalid |
| INVALID_LOCAL_PATH | 400 | Local path doesn't exist or invalid |
| REPO_NOT_FOUND | 404 | HuggingFace repository doesn't exist |
| GATED_MODEL_NO_TOKEN | 401 | Gated model requires HF token |
| INVALID_HF_TOKEN | 401 | HF token is invalid or lacks access |

---

## 8. Non-Functional Requirements

### Performance

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| API response (list/get) | <100ms | p95 latency |
| Download start | <2s | Time to first progress event |
| Load time (2B model) | <30s | Wall clock time |
| Memory query | <500ms | torch.cuda operations |
| Unload time | <5s | Excluding graceful wait |

### Scalability

- **Concurrent downloads:** Support 1 active download at a time (queue additional)
- **Model storage:** Limited by disk space only
- **Database:** Model count limited to ~1000 (practical limit)

### Reliability

| Requirement | Target |
|-------------|--------|
| Download retry | 3 attempts with exponential backoff (circuit breaker pattern) |
| Partial download cleanup | 100% cleanup on failure/cancel |
| Load failure recovery | Clean state, no GPU memory leak |
| Graceful unload timeout | 30 seconds max |

### Security

- No persistent storage of HF tokens
- trust_remote_code explicit opt-in
- Path traversal prevention for local paths
- No shell command injection in paths

---

## 9. Feature Boundaries (Non-Goals)

### Explicitly NOT Included in v1.0

| Non-Goal | Rationale |
|----------|-----------|
| Multiple simultaneous downloads | Complexity; single download sufficient for v1.0 |
| Download queue management | Out of scope; single download at a time |
| Model format conversion | Use HuggingFace formats only |
| GGUF format support | Requires different inference engine |
| Model fine-tuning | Delegated to external tools |
| Automatic model updates | Manual re-download if needed |
| Model version management | Single version per repo+quantization |
| Pause/resume downloads | Cancel only; restart from beginning |

### Future Enhancements (Post v1.0)

- Multiple concurrent downloads with queue
- Pause/resume large downloads
- GGUF and other format support
- Automatic model updates/version checking
- Model comparison tools
- Download scheduling

### Related Features Handled Separately

- **SAE Management:** Different feature, depends on Model Management
- **OpenAI API:** Uses loaded model, doesn't manage models
- **Admin UI:** Consumes Model Management API

---

## 10. Dependencies

### Feature Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| PostgreSQL database | Infrastructure | Available (from ADR) |
| Redis cache | Infrastructure | Available (from ADR) |
| NVIDIA GPU | Hardware | Required for quantization |
| HuggingFace Hub | External | Internet access required |

### Library Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| transformers | ≥4.36.0 | Model loading |
| huggingface_hub | ≥0.20.0 | Model downloads |
| bitsandbytes | ≥0.42.0 | Quantization |
| torch | ≥2.0 | GPU operations |
| safetensors | ≥0.4.0 | Model format |

### Infrastructure Dependencies

- **GPU:** NVIDIA with CUDA support (for Q4/Q8)
- **Disk:** Sufficient space for model cache (10GB+ recommended)
- **Network:** Internet access for HuggingFace downloads
- **Memory:** Sufficient VRAM for target models

### Timeline Dependencies

- Model Management must be complete before:
  - SAE Management (requires loaded model)
  - OpenAI API (requires loaded model)
  - Feature Steering (requires model + SAE)

---

## 11. Success Criteria

### Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Download success rate | >95% | (successful / attempted) |
| Load success rate | >99% | When memory sufficient |
| Memory estimate accuracy | Within 20% | (estimated - actual) / actual |
| UI responsiveness | <100ms | Interaction latency |
| API response time | <100ms | p95 for list/get operations |

### User Satisfaction Indicators

- Users can download their first model within 5 minutes of starting miLLM
- Users understand memory requirements before attempting to load
- Users can recover from errors without server restart
- Download cancellation works reliably without leaving partial files

### Completion Criteria

- [ ] All functional requirements (MM-*) implemented and tested
- [ ] All user stories have passing acceptance tests
- [ ] API endpoints documented in OpenAPI spec
- [ ] Error handling covers all edge cases
- [ ] 80%+ backend code coverage
- [ ] E2E test for download → load → unload workflow
- [ ] UI matches mockup specifications
- [ ] Performance targets met

---

## 12. Testing Requirements

### Unit Testing

#### Backend (pytest)
- `test_model_service.py`: Service layer logic
  - Download initiation and cancellation
  - Load/unload state management
  - Memory estimation calculations
  - Error handling for each error code

- `test_model_loader.py`: ML loading logic
  - Quantization application
  - Model file validation
  - GPU memory operations (mocked)

- `test_model_repository.py`: Database operations
  - CRUD operations
  - Query filters
  - Unique constraints

#### Frontend (Vitest)
- `modelStore.test.ts`: Zustand store actions
- `useModels.test.ts`: Custom hook behavior
- `ModelCard.test.tsx`: Component rendering and interactions
- `DownloadForm.test.tsx`: Form validation and submission

### Integration Testing

- **API Integration:** Test all endpoints with real database
- **HuggingFace Integration:** Test with real HF API (use small test model)
- **WebSocket Integration:** Test progress events flow correctly
- **Database Integration:** Test model persistence across server restart

### End-to-End Testing (Playwright)

```typescript
test('complete model workflow', async ({ page }) => {
  // Navigate to models page
  await page.goto('/models');

  // Download a model (use small test model)
  await page.fill('[data-testid="repo-input"]', 'hf-internal-testing/tiny-random-gpt2');
  await page.click('[data-testid="download-btn"]');

  // Wait for download to complete
  await expect(page.locator('[data-testid="model-status"]')).toHaveText('Ready', { timeout: 60000 });

  // Load the model
  await page.click('[data-testid="load-btn"]');
  await expect(page.locator('[data-testid="model-status"]')).toHaveText('Loaded', { timeout: 30000 });

  // Verify status bar shows model
  await expect(page.locator('[data-testid="status-bar-model"]')).toContainText('tiny-random-gpt2');

  // Unload the model
  await page.click('[data-testid="unload-btn"]');
  await expect(page.locator('[data-testid="model-status"]')).toHaveText('Ready');

  // Delete the model
  await page.click('[data-testid="delete-btn"]');
  await page.click('[data-testid="confirm-delete-btn"]');
  await expect(page.locator('[data-testid="model-card"]')).toHaveCount(0);
});
```

### Performance Testing

- Load time benchmarks for different model sizes
- Memory estimation accuracy validation
- Concurrent request handling during load/unload
- Download speed measurement

---

## 13. Implementation Considerations

### Complexity Assessment

| Component | Complexity | Risk |
|-----------|------------|------|
| HuggingFace download | Medium | Network failures, large files |
| Quantization | Medium | bitsandbytes quirks, GPU compat |
| Memory estimation | Low | Simple calculation |
| GPU memory management | High | OOM handling, cleanup |
| Progress streaming | Medium | WebSocket reliability |
| Graceful unload | Medium | Request tracking |

### Recommended Implementation Order

1. **Database schema and repository** - Foundation
2. **Model service skeleton** - Business logic structure
3. **HuggingFace download** - Core download functionality
4. **Progress WebSocket** - Real-time updates
5. **Model loading (FP16)** - Basic load without quantization
6. **Quantization integration** - Q4/Q8 support
7. **Memory estimation** - Pre-load checks
8. **Local path support** - Alternative source
9. **UI components** - Frontend implementation
10. **Error handling polish** - Edge cases

### Potential Technical Challenges

1. **bitsandbytes compatibility:** May have issues with certain GPU architectures
2. **Large file downloads:** Need chunked download with retry
3. **GPU memory fragmentation:** May need explicit cache clearing
4. **Transformers version conflicts:** Pin versions carefully
5. **Progress tracking accuracy:** HuggingFace download progress can be inaccurate

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| bitsandbytes fails | Fallback to FP16 with warning |
| Download corrupts | Checksum verification |
| OOM on load | Pre-check memory, clear caches |
| WebSocket disconnect | Polling fallback, reconnection |
| HuggingFace API instability | Circuit breaker pattern on HF API calls: tracks failures and opens circuit after repeated failures to prevent cascading errors. Automatically recovers after cooldown period. |

---

## 14. Open Questions

### Resolved During PRD Creation

| Question | Resolution |
|----------|------------|
| Support local paths? | Yes, in addition to HuggingFace |
| Quantization timing? | At download time (save quantized) |
| Multiple models on disk? | Yes, but one loaded at a time |
| Delete behavior? | Hard delete (remove files) |

### Questions for TDD/Implementation

1. **Checksum verification:** Should we verify downloaded files against HuggingFace checksums?
2. **Cache directory structure:** Exact directory naming scheme for cached models?
3. **Config.json handling:** How much model metadata to extract and store?
4. **Tokenizer handling:** Download and cache tokenizer with model?

### Questions Requiring User Feedback

1. **Default quantization:** Is Q4 the right default, or should we detect GPU and suggest?
2. **Model naming:** Auto-generate from repo, or require user input?
3. **Cache location:** Should users be able to change cache dir from UI, or env var only?

---

## Appendix A: UI Mockup Reference

From `0xcc/spec/miLLM_UI.jsx`, the Models page includes:

- Download card with repository input, quantization select, token input, trust checkbox
- Preview and Download buttons
- Model list showing name, params, quantization, memory, repo, status
- Action buttons: Load/Unload, Delete
- Status badges: Loaded (green), Ready (cyan)

---

## Appendix B: Related BRD Requirements

| BRD ID | Requirement | PRD Coverage |
|--------|-------------|--------------|
| FR-1.1 | Download from HuggingFace | MM-D1 |
| FR-1.2 | Load Transformers format | MM-L1 |
| FR-1.3 | 4-bit/8-bit quantization | MM-Q1, MM-Q2 |
| FR-1.4 | Cache models locally | MM-C1, MM-C2 |
| FR-1.5 | Display memory requirements | MM-M1, MM-M2 |
| FR-1.6 | Extensible format support | Via Transformers library |

---

**Document Status:** Complete
**Next Document:** `001_FTDD|Model_Management.md` (Technical Design Document)
**Instruction File:** `@0xcc/instruct/004_create-tdd.md`
