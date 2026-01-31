# Task List: SAE Management

## miLLM Feature 3

**Document Version:** 1.2
**Created:** January 30, 2026
**Status:** Complete
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

- [x] 1.0 Create SAE database models
  - [x] 1.1 Create `millm/db/models/sae.py` file
  - [x] 1.2 Implement SAE model with all fields (id, repository_id, dimensions, status, etc.)
  - [x] 1.3 Add status CHECK constraint for valid states
  - [x] 1.4 Implement SAEAttachment model with relationship
  - [x] 1.5 Add partial unique index for single active attachment
  - [x] 1.6 Export models in `millm/db/models/__init__.py`

- [x] 2.0 Create Alembic migration for SAE tables
  - [x] 2.1 Generate new migration with `alembic revision --autogenerate`
  - [x] 2.2 Review and adjust generated migration
  - [x] 2.3 Add indexes for common queries (status, repository_id)
  - [x] 2.4 Test migration up and down
  - [x] 2.5 Apply migration to development database

- [x] 3.0 Implement SAE repository
  - [x] 3.1 Create `millm/db/repositories/sae_repository.py` file
  - [x] 3.2 Implement `get_all()` method
  - [x] 3.3 Implement `get(sae_id)` method
  - [x] 3.4 Implement `get_by_repository()` method
  - [x] 3.5 Implement `create_downloading()` method
  - [x] 3.6 Implement `update_downloaded()` method
  - [x] 3.7 Implement `update_status()` method
  - [x] 3.8 Implement `delete()` method
  - [x] 3.9 Implement attachment methods (get_active, create, deactivate)
  - [x] 3.10 Write unit tests for repository

### Phase 2: SAE Configuration

- [x] 4.0 Implement SAE configuration parsing
  - [x] 4.1 Create `millm/ml/sae_config.py` file
  - [x] 4.2 Implement SAEConfig dataclass with all fields
  - [x] 4.3 Implement `from_json()` class method
  - [x] 4.4 Handle SAELens format variations (d_in, d_model, input_dim)
  - [x] 4.5 Handle different config file names (cfg.json, config.json)
  - [x] 4.6 Implement `estimate_memory_mb()` method
  - [x] 4.7 Write unit tests for config parsing
  - [x] 4.8 Create test fixture: `tests/fixtures/sae/sample_config.json`

### Phase 3: SAE Downloading

- [x] 5.0 Implement SAE downloader
  - [x] 5.1 Create `millm/ml/sae_downloader.py` file
  - [x] 5.2 Implement SAEDownloader class with cache_dir initialization
  - [x] 5.3 Implement async `download()` method
  - [x] 5.4 Implement sync `_download_sync()` using snapshot_download
  - [x] 5.5 Add repository validation before download
  - [x] 5.6 Implement progress callback support
  - [x] 5.7 Implement `delete()` method for cache cleanup
  - [x] 5.8 Implement `get_cache_size()` utility method
  - [x] 5.9 Write unit tests with mocked HuggingFace API
  - [x] 5.10 Test resume functionality

### Phase 4: SAE Loading

- [x] 6.0 Implement SAE loader
  - [x] 6.1 Create `millm/ml/sae_loader.py` file
  - [x] 6.2 Implement SAELoader class
  - [x] 6.3 Implement `load_config()` method
  - [x] 6.4 Implement `load()` method for weights
  - [x] 6.5 Implement `_find_weights_file()` helper
  - [x] 6.6 Handle weight tensor name variations (W_enc, encoder.weight, etc.)
  - [x] 6.7 Support dtype conversion (float32, float16)
  - [x] 6.8 Create test fixture: small safetensors file
  - [x] 6.9 Write unit tests for loading

### Phase 5: SAE Wrapper

- [x] 7.0 Implement LoadedSAE wrapper
  - [x] 7.1 Create `millm/ml/sae_wrapper.py` file
  - [x] 7.2 Implement LoadedSAE class with weight tensors
  - [x] 7.3 Implement `forward()` method (encode → steer → decode)
  - [x] 7.4 Implement `encode()` and `decode()` methods
  - [x] 7.5 Implement steering methods (set, clear, enable, get_values)
  - [x] 7.6 Implement `_rebuild_steering_vector()` for efficiency
  - [x] 7.7 Implement monitoring methods (enable, get_last_activations)
  - [x] 7.8 Implement `_capture_activations()` with optional feature selection
  - [x] 7.9 Implement memory management (estimate_memory_mb, to_device, to_cpu)
  - [x] 7.10 Add dimension validation in constructor
  - [x] 7.11 Write comprehensive unit tests

### Phase 6: Model Hooking

- [x] 8.0 Implement SAE hooker
  - [x] 8.1 Create `millm/ml/sae_hooker.py` file
  - [x] 8.2 Implement SAEHooker class
  - [x] 8.3 Implement `install()` method returning RemovableHandle
  - [x] 8.4 Implement `_create_hook_fn()` for SAE application
  - [x] 8.5 Handle tuple outputs (hidden_states, ...) in hook
  - [x] 8.6 Implement `remove()` method
  - [x] 8.7 Implement `_get_layer()` with architecture detection
  - [x] 8.8 Support multiple architectures (Gemma, Llama, GPT-2 style)
  - [x] 8.9 Implement `get_layer_count()` utility
  - [x] 8.10 Write unit tests with mock models

### Phase 7: SAE Service

- [x] 9.0 Implement SAE service core
  - [x] 9.1 Create `millm/services/sae_service.py` file
  - [x] 9.2 Implement constructor with dependencies
  - [x] 9.3 Initialize downloader, loader, hooker components
  - [x] 9.4 Set up attachment lock for thread safety
  - [x] 9.5 Implement attachment state tracking properties

- [x] 10.0 Implement SAE listing methods
  - [x] 10.1 Implement `list_saes()` returning SAEListResponse
  - [x] 10.2 Implement `get_sae()` by ID
  - [x] 10.3 Implement `get_attachment_status()` method
  - [x] 10.4 Implement `_to_metadata()` conversion helper

- [x] 11.0 Implement SAE download methods
  - [x] 11.1 Implement `start_download()` method
  - [x] 11.2 Generate unique SAE ID from repository
  - [x] 11.3 Check for existing cached SAE
  - [x] 11.4 Create downloading record in database
  - [x] 11.5 Implement `_download_task()` background task
  - [x] 11.6 Update database on download completion
  - [x] 11.7 Handle download errors with status update

- [x] 12.0 Implement SAE attachment methods
  - [x] 12.1 Implement `check_compatibility()` method
  - [x] 12.2 Validate layer range against model
  - [x] 12.3 Validate dimension match (sae.d_in == model.hidden_size)
  - [x] 12.4 Generate warnings for layer mismatch
  - [x] 12.5 Implement `attach_sae()` method with lock
  - [x] 12.6 Validate preconditions (model loaded, SAE cached, none attached)
  - [x] 12.7 Load SAE weights
  - [x] 12.8 Install hook on model
  - [x] 12.9 Update state and persist to database
  - [x] 12.10 Return memory usage information

- [x] 13.0 Implement SAE detachment methods
  - [x] 13.1 Implement `detach_sae()` method with lock
  - [x] 13.2 Remove hook from model
  - [x] 13.3 Move SAE to CPU and delete
  - [x] 13.4 Clear CUDA cache
  - [x] 13.5 Update state and database
  - [x] 13.6 Return freed memory information

- [x] 14.0 Implement SAE deletion
  - [x] 14.1 Implement `delete_sae()` method
  - [x] 14.2 Prevent deletion of attached SAE
  - [x] 14.3 Delete cache files
  - [x] 14.4 Delete database records
  - [x] 14.5 Return freed disk space

### Phase 8: API Routes

- [x] 15.0 Implement Pydantic schemas
  - [x] 15.1 Create `millm/api/schemas/sae.py` file
  - [x] 15.2 Implement SAEMetadata schema
  - [x] 15.3 Implement DownloadSAERequest schema with validation
  - [x] 15.4 Implement AttachSAERequest schema
  - [x] 15.5 Implement AttachmentStatus schema
  - [x] 15.6 Implement SAEListResponse schema
  - [x] 15.7 Implement CompatibilityResult schema

- [x] 16.0 Implement SAE API routes
  - [x] 16.1 Create `millm/api/routes/management/saes.py` file
  - [x] 16.2 Implement GET /api/saes (list all)
  - [x] 16.3 Implement GET /api/saes/attachment (current status)
  - [x] 16.4 Implement POST /api/saes/download (start download)
  - [x] 16.5 Implement GET /api/saes/{sae_id} (get one)
  - [x] 16.6 Implement POST /api/saes/{sae_id}/attach
  - [x] 16.7 Implement POST /api/saes/{sae_id}/detach
  - [x] 16.8 Implement DELETE /api/saes/{sae_id}
  - [x] 16.9 Implement GET /api/saes/{sae_id}/compatibility
  - [x] 16.10 Mount router in main app

### Phase 9: Error Handling

- [x] 17.0 Add SAE-specific errors
  - [x] 17.1 Add SAEError base class to `millm/core/errors.py`
  - [x] 17.2 Add SAENotFoundError
  - [x] 17.3 Add SAEIncompatibleError
  - [x] 17.4 Add SAEAlreadyAttachedError
  - [x] 17.5 Add SAENotAttachedError
  - [x] 17.6 Add SAEDownloadError
  - [x] 17.7 Register error handlers for SAE errors

### Phase 10: Unit Tests

- [x] 18.0 Write ML component unit tests
  - [x] 18.1 Create `tests/unit/ml/test_sae_config.py`
  - [x] 18.2 Create `tests/unit/ml/test_sae_loader.py`
  - [x] 18.3 Create `tests/unit/ml/test_sae_wrapper.py`
  - [x] 18.4 Create `tests/unit/ml/test_sae_hooker.py`
  - [x] 18.5 Test forward pass preserves shape
  - [x] 18.6 Test steering modifies output
  - [x] 18.7 Test monitoring captures activations
  - [x] 18.8 Test feature index validation

- [x] 19.0 Write service unit tests
  - [x] 19.1 Create `tests/unit/services/test_sae_service.py`
  - [x] 19.2 Test compatibility checking logic
  - [x] 19.3 Test attachment state management
  - [x] 19.4 Test concurrent access with lock

- [x] 20.0 Write repository unit tests
  - [x] 20.1 Create `tests/unit/db/test_sae_repository.py`
  - [x] 20.2 Test CRUD operations
  - [x] 20.3 Test attachment tracking
  - [x] 20.4 Test single-active constraint

### Phase 11: Integration Tests

- [x] 21.0 Write integration tests
  - [x] 21.1 Create `tests/integration/services/test_sae_service_integration.py`
  - [x] 21.2 Test full download → attach → detach flow
  - [x] 21.3 Test compatibility validation
  - [x] 21.4 Test cannot attach second SAE
  - [x] 21.5 Test memory tracking accuracy

- [x] 22.0 Write API route tests
  - [x] 22.1 Create `tests/integration/api/test_sae_routes.py`
  - [x] 22.2 Test all endpoints with mocked services
  - [x] 22.3 Test error responses match format
  - [x] 22.4 Test validation errors

### Phase 12: Test Fixtures

- [x] 23.0 Create test fixtures
  - [x] 23.1 Create `tests/fixtures/sae/` directory
  - [x] 23.2 Create `sample_config.json` with valid SAELens format
  - [x] 23.3 Create small `sample_weights.safetensors` (64 x 128 dimensions)
  - [x] 23.4 Create pytest fixtures for easy access
  - [x] 23.5 Document fixture usage in README

### Phase 13: Integration and Polish

- [x] 24.0 Integrate with existing systems
  - [x] 24.1 Update app lifespan to initialize SAEService
  - [x] 24.2 Add SAE service dependency
  - [x] 24.3 Verify WebSocket progress events work
  - [x] 24.4 Test with actual model loaded

- [x] 25.0 Documentation and cleanup
  - [x] 25.1 Add docstrings to all public methods
  - [x] 25.2 Update OpenAPI documentation
  - [x] 25.3 Review error messages for clarity
  - [x] 25.4 Remove debug code
  - [x] 25.5 Run full test suite and fix issues

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
