# Task List: SAE Management

## miLLM Feature 3

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `003_FPRD|SAE_Management.md`
- Feature TDD: `003_FTDD|SAE_Management.md`
- Feature TID: `003_FTID|SAE_Management.md`

---

## Relevant Files

### Backend - Database

- `millm/db/models/sae.py` - SAE and SAEAttachment SQLAlchemy models
- `millm/db/repositories/sae_repository.py` - Database operations for SAEs
- `migrations/versions/xxx_add_sae_tables.py` - Alembic migration for SAE tables

### Backend - ML Components

- `millm/ml/sae_config.py` - SAE configuration parsing (SAELens format)
- `millm/ml/sae_loader.py` - SAE weight loading from SafeTensors
- `millm/ml/sae_wrapper.py` - LoadedSAE class with encode/decode/steering
- `millm/ml/sae_hooker.py` - PyTorch forward hook management
- `millm/ml/sae_downloader.py` - HuggingFace download with progress

### Backend - API

- `millm/api/routes/management/saes.py` - SAE management REST endpoints
- `millm/api/schemas/sae.py` - Pydantic schemas for SAE API

### Backend - Services

- `millm/services/sae_service.py` - Main SAE service coordinating all operations

### Backend - Errors

- `millm/core/errors.py` - Add SAE-specific error classes

### Tests - Unit

- `tests/unit/ml/test_sae_config.py` - Config parsing tests
- `tests/unit/ml/test_sae_loader.py` - SAE loading tests
- `tests/unit/ml/test_sae_wrapper.py` - LoadedSAE tests
- `tests/unit/ml/test_sae_hooker.py` - Hook installation tests
- `tests/unit/services/test_sae_service.py` - Service unit tests
- `tests/unit/db/test_sae_repository.py` - Repository tests

### Tests - Integration

- `tests/integration/services/test_sae_service_integration.py` - Full flow tests
- `tests/integration/api/test_sae_routes.py` - API endpoint tests

### Test Fixtures

- `tests/fixtures/sae/sample_config.json` - Sample SAE configuration
- `tests/fixtures/sae/sample_weights.safetensors` - Small test SAE weights

### Notes

- SAE tests require small test fixtures (not real SAEs)
- Integration tests mock HuggingFace downloads
- Use `pytest tests/unit/ml/` to run ML-specific tests
- Use `pytest tests/integration/` for full flow tests

---

## Tasks

### Phase 1: Database Infrastructure

- [ ] 1.0 Create SAE database models
  - [ ] 1.1 Create `millm/db/models/sae.py` file
  - [ ] 1.2 Implement SAE model with all fields (id, repository_id, dimensions, status, etc.)
  - [ ] 1.3 Add status CHECK constraint for valid states
  - [ ] 1.4 Implement SAEAttachment model with relationship
  - [ ] 1.5 Add partial unique index for single active attachment
  - [ ] 1.6 Export models in `millm/db/models/__init__.py`

- [ ] 2.0 Create Alembic migration for SAE tables
  - [ ] 2.1 Generate new migration with `alembic revision --autogenerate`
  - [ ] 2.2 Review and adjust generated migration
  - [ ] 2.3 Add indexes for common queries (status, repository_id)
  - [ ] 2.4 Test migration up and down
  - [ ] 2.5 Apply migration to development database

- [ ] 3.0 Implement SAE repository
  - [ ] 3.1 Create `millm/db/repositories/sae_repository.py` file
  - [ ] 3.2 Implement `get_all()` method
  - [ ] 3.3 Implement `get(sae_id)` method
  - [ ] 3.4 Implement `get_by_repository()` method
  - [ ] 3.5 Implement `create_downloading()` method
  - [ ] 3.6 Implement `update_downloaded()` method
  - [ ] 3.7 Implement `update_status()` method
  - [ ] 3.8 Implement `delete()` method
  - [ ] 3.9 Implement attachment methods (get_active, create, deactivate)
  - [ ] 3.10 Write unit tests for repository

### Phase 2: SAE Configuration

- [ ] 4.0 Implement SAE configuration parsing
  - [ ] 4.1 Create `millm/ml/sae_config.py` file
  - [ ] 4.2 Implement SAEConfig dataclass with all fields
  - [ ] 4.3 Implement `from_json()` class method
  - [ ] 4.4 Handle SAELens format variations (d_in, d_model, input_dim)
  - [ ] 4.5 Handle different config file names (cfg.json, config.json)
  - [ ] 4.6 Implement `estimate_memory_mb()` method
  - [ ] 4.7 Write unit tests for config parsing
  - [ ] 4.8 Create test fixture: `tests/fixtures/sae/sample_config.json`

### Phase 3: SAE Downloading

- [ ] 5.0 Implement SAE downloader
  - [ ] 5.1 Create `millm/ml/sae_downloader.py` file
  - [ ] 5.2 Implement SAEDownloader class with cache_dir initialization
  - [ ] 5.3 Implement async `download()` method
  - [ ] 5.4 Implement sync `_download_sync()` using snapshot_download
  - [ ] 5.5 Add repository validation before download
  - [ ] 5.6 Implement progress callback support
  - [ ] 5.7 Implement `delete()` method for cache cleanup
  - [ ] 5.8 Implement `get_cache_size()` utility method
  - [ ] 5.9 Write unit tests with mocked HuggingFace API
  - [ ] 5.10 Test resume functionality

### Phase 4: SAE Loading

- [ ] 6.0 Implement SAE loader
  - [ ] 6.1 Create `millm/ml/sae_loader.py` file
  - [ ] 6.2 Implement SAELoader class
  - [ ] 6.3 Implement `load_config()` method
  - [ ] 6.4 Implement `load()` method for weights
  - [ ] 6.5 Implement `_find_weights_file()` helper
  - [ ] 6.6 Handle weight tensor name variations (W_enc, encoder.weight, etc.)
  - [ ] 6.7 Support dtype conversion (float32, float16)
  - [ ] 6.8 Create test fixture: small safetensors file
  - [ ] 6.9 Write unit tests for loading

### Phase 5: SAE Wrapper

- [ ] 7.0 Implement LoadedSAE wrapper
  - [ ] 7.1 Create `millm/ml/sae_wrapper.py` file
  - [ ] 7.2 Implement LoadedSAE class with weight tensors
  - [ ] 7.3 Implement `forward()` method (encode → steer → decode)
  - [ ] 7.4 Implement `encode()` and `decode()` methods
  - [ ] 7.5 Implement steering methods (set, clear, enable, get_values)
  - [ ] 7.6 Implement `_rebuild_steering_vector()` for efficiency
  - [ ] 7.7 Implement monitoring methods (enable, get_last_activations)
  - [ ] 7.8 Implement `_capture_activations()` with optional feature selection
  - [ ] 7.9 Implement memory management (estimate_memory_mb, to_device, to_cpu)
  - [ ] 7.10 Add dimension validation in constructor
  - [ ] 7.11 Write comprehensive unit tests

### Phase 6: Model Hooking

- [ ] 8.0 Implement SAE hooker
  - [ ] 8.1 Create `millm/ml/sae_hooker.py` file
  - [ ] 8.2 Implement SAEHooker class
  - [ ] 8.3 Implement `install()` method returning RemovableHandle
  - [ ] 8.4 Implement `_create_hook_fn()` for SAE application
  - [ ] 8.5 Handle tuple outputs (hidden_states, ...) in hook
  - [ ] 8.6 Implement `remove()` method
  - [ ] 8.7 Implement `_get_layer()` with architecture detection
  - [ ] 8.8 Support multiple architectures (Gemma, Llama, GPT-2 style)
  - [ ] 8.9 Implement `get_layer_count()` utility
  - [ ] 8.10 Write unit tests with mock models

### Phase 7: SAE Service

- [ ] 9.0 Implement SAE service core
  - [ ] 9.1 Create `millm/services/sae_service.py` file
  - [ ] 9.2 Implement constructor with dependencies
  - [ ] 9.3 Initialize downloader, loader, hooker components
  - [ ] 9.4 Set up attachment lock for thread safety
  - [ ] 9.5 Implement attachment state tracking properties

- [ ] 10.0 Implement SAE listing methods
  - [ ] 10.1 Implement `list_saes()` returning SAEListResponse
  - [ ] 10.2 Implement `get_sae()` by ID
  - [ ] 10.3 Implement `get_attachment_status()` method
  - [ ] 10.4 Implement `_to_metadata()` conversion helper

- [ ] 11.0 Implement SAE download methods
  - [ ] 11.1 Implement `start_download()` method
  - [ ] 11.2 Generate unique SAE ID from repository
  - [ ] 11.3 Check for existing cached SAE
  - [ ] 11.4 Create downloading record in database
  - [ ] 11.5 Implement `_download_task()` background task
  - [ ] 11.6 Update database on download completion
  - [ ] 11.7 Handle download errors with status update

- [ ] 12.0 Implement SAE attachment methods
  - [ ] 12.1 Implement `check_compatibility()` method
  - [ ] 12.2 Validate layer range against model
  - [ ] 12.3 Validate dimension match (sae.d_in == model.hidden_size)
  - [ ] 12.4 Generate warnings for layer mismatch
  - [ ] 12.5 Implement `attach_sae()` method with lock
  - [ ] 12.6 Validate preconditions (model loaded, SAE cached, none attached)
  - [ ] 12.7 Load SAE weights
  - [ ] 12.8 Install hook on model
  - [ ] 12.9 Update state and persist to database
  - [ ] 12.10 Return memory usage information

- [ ] 13.0 Implement SAE detachment methods
  - [ ] 13.1 Implement `detach_sae()` method with lock
  - [ ] 13.2 Remove hook from model
  - [ ] 13.3 Move SAE to CPU and delete
  - [ ] 13.4 Clear CUDA cache
  - [ ] 13.5 Update state and database
  - [ ] 13.6 Return freed memory information

- [ ] 14.0 Implement SAE deletion
  - [ ] 14.1 Implement `delete_sae()` method
  - [ ] 14.2 Prevent deletion of attached SAE
  - [ ] 14.3 Delete cache files
  - [ ] 14.4 Delete database records
  - [ ] 14.5 Return freed disk space

### Phase 8: API Routes

- [ ] 15.0 Implement Pydantic schemas
  - [ ] 15.1 Create `millm/api/schemas/sae.py` file
  - [ ] 15.2 Implement SAEMetadata schema
  - [ ] 15.3 Implement DownloadSAERequest schema with validation
  - [ ] 15.4 Implement AttachSAERequest schema
  - [ ] 15.5 Implement AttachmentStatus schema
  - [ ] 15.6 Implement SAEListResponse schema
  - [ ] 15.7 Implement CompatibilityResult schema

- [ ] 16.0 Implement SAE API routes
  - [ ] 16.1 Create `millm/api/routes/management/saes.py` file
  - [ ] 16.2 Implement GET /api/saes (list all)
  - [ ] 16.3 Implement GET /api/saes/attachment (current status)
  - [ ] 16.4 Implement POST /api/saes/download (start download)
  - [ ] 16.5 Implement GET /api/saes/{sae_id} (get one)
  - [ ] 16.6 Implement POST /api/saes/{sae_id}/attach
  - [ ] 16.7 Implement POST /api/saes/{sae_id}/detach
  - [ ] 16.8 Implement DELETE /api/saes/{sae_id}
  - [ ] 16.9 Implement GET /api/saes/{sae_id}/compatibility
  - [ ] 16.10 Mount router in main app

### Phase 9: Error Handling

- [ ] 17.0 Add SAE-specific errors
  - [ ] 17.1 Add SAEError base class to `millm/core/errors.py`
  - [ ] 17.2 Add SAENotFoundError
  - [ ] 17.3 Add SAEIncompatibleError
  - [ ] 17.4 Add SAEAlreadyAttachedError
  - [ ] 17.5 Add SAENotAttachedError
  - [ ] 17.6 Add SAEDownloadError
  - [ ] 17.7 Register error handlers for SAE errors

### Phase 10: Unit Tests

- [ ] 18.0 Write ML component unit tests
  - [ ] 18.1 Create `tests/unit/ml/test_sae_config.py`
  - [ ] 18.2 Create `tests/unit/ml/test_sae_loader.py`
  - [ ] 18.3 Create `tests/unit/ml/test_sae_wrapper.py`
  - [ ] 18.4 Create `tests/unit/ml/test_sae_hooker.py`
  - [ ] 18.5 Test forward pass preserves shape
  - [ ] 18.6 Test steering modifies output
  - [ ] 18.7 Test monitoring captures activations
  - [ ] 18.8 Test feature index validation

- [ ] 19.0 Write service unit tests
  - [ ] 19.1 Create `tests/unit/services/test_sae_service.py`
  - [ ] 19.2 Test compatibility checking logic
  - [ ] 19.3 Test attachment state management
  - [ ] 19.4 Test concurrent access with lock

- [ ] 20.0 Write repository unit tests
  - [ ] 20.1 Create `tests/unit/db/test_sae_repository.py`
  - [ ] 20.2 Test CRUD operations
  - [ ] 20.3 Test attachment tracking
  - [ ] 20.4 Test single-active constraint

### Phase 11: Integration Tests

- [ ] 21.0 Write integration tests
  - [ ] 21.1 Create `tests/integration/services/test_sae_service_integration.py`
  - [ ] 21.2 Test full download → attach → detach flow
  - [ ] 21.3 Test compatibility validation
  - [ ] 21.4 Test cannot attach second SAE
  - [ ] 21.5 Test memory tracking accuracy

- [ ] 22.0 Write API route tests
  - [ ] 22.1 Create `tests/integration/api/test_sae_routes.py`
  - [ ] 22.2 Test all endpoints with mocked services
  - [ ] 22.3 Test error responses match format
  - [ ] 22.4 Test validation errors

### Phase 12: Test Fixtures

- [ ] 23.0 Create test fixtures
  - [ ] 23.1 Create `tests/fixtures/sae/` directory
  - [ ] 23.2 Create `sample_config.json` with valid SAELens format
  - [ ] 23.3 Create small `sample_weights.safetensors` (64 x 128 dimensions)
  - [ ] 23.4 Create pytest fixtures for easy access
  - [ ] 23.5 Document fixture usage in README

### Phase 13: Integration and Polish

- [ ] 24.0 Integrate with existing systems
  - [ ] 24.1 Update app lifespan to initialize SAEService
  - [ ] 24.2 Add SAE service dependency
  - [ ] 24.3 Verify WebSocket progress events work
  - [ ] 24.4 Test with actual model loaded

- [ ] 25.0 Documentation and cleanup
  - [ ] 25.1 Add docstrings to all public methods
  - [ ] 25.2 Update OpenAPI documentation
  - [ ] 25.3 Review error messages for clarity
  - [ ] 25.4 Remove debug code
  - [ ] 25.5 Run full test suite and fix issues

---

## Notes

### Development Order Recommendation

1. Database models and repository (Tasks 1-3) - foundation
2. SAE configuration parsing (Task 4) - needed for validation
3. SAE wrapper and hooker (Tasks 7-8) - core functionality
4. SAE loader (Task 6) - connects config to wrapper
5. SAE downloader (Task 5) - acquisition system
6. SAE service (Tasks 9-14) - orchestration layer
7. API routes and schemas (Tasks 15-16) - external interface
8. Error handling (Task 17) - proper error responses
9. Tests (Tasks 18-23) - validation
10. Integration and polish (Tasks 24-25) - finishing touches

### Key Dependencies

- Requires Model Management (Feature 1) to be complete
- Model must be loaded before SAE attachment
- SAELens format SAEs from HuggingFace

### Testing Notes

- Create small test SAE (64 x 128) for fast unit tests
- Mock HuggingFace downloads in integration tests
- Real SAE tests only in separate slow test suite
- Test with real model only in manual/E2E tests

### Memory Tracking

- SAE memory = (d_in × d_sae + d_sae + d_sae × d_in + d_in) × bytes_per_param
- For Gemma-2-2b with 16K features: ~500MB in float32
- Track both GPU memory and disk usage

---

**Document Status:** Complete
**Total Tasks:** 25 parent tasks, 150+ sub-tasks
**Estimated Timeline:** 2-3 weeks
**Next Feature:** Feature 4 - Feature Steering
