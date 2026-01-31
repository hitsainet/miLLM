# Task List: Model Management

## miLLM Feature 1

**Document Version:** 1.1
**Created:** January 30, 2026
**Status:** Complete
**References:**
- Feature PRD: `001_FPRD|Model_Management.md`
- TDD: `001_FTDD|Model_Management.md`
- TID: `001_FTID|Model_Management.md`

---

## Relevant Files

### Backend - Core Infrastructure
- `millm/__init__.py` - Package initialization
- `millm/main.py` - FastAPI application entry point with Socket.IO integration
- `millm/core/__init__.py` - Core package initialization
- `millm/core/config.py` - Pydantic settings for environment configuration
- `millm/core/errors.py` - MiLLMError exception hierarchy
- `millm/core/logging.py` - Structlog setup for structured logging

### Backend - Database Layer
- `millm/db/__init__.py` - Database package initialization
- `millm/db/base.py` - SQLAlchemy async engine and session setup
- `millm/db/models/__init__.py` - Models package initialization
- `millm/db/models/model.py` - Model ORM class with status/source/quantization enums
- `millm/db/repositories/__init__.py` - Repositories package initialization
- `millm/db/repositories/model_repository.py` - ModelRepository CRUD operations
- `millm/db/migrations/env.py` - Alembic migration environment
- `millm/db/migrations/versions/001_create_models_table.py` - Initial models table migration

### Backend - API Layer
- `millm/api/__init__.py` - API package initialization
- `millm/api/dependencies.py` - FastAPI dependency injection
- `millm/api/exception_handlers.py` - MiLLMError exception handler
- `millm/api/routes/__init__.py` - Route registration function
- `millm/api/routes/management/__init__.py` - Management routes package
- `millm/api/routes/management/models.py` - /api/models/* endpoints
- `millm/api/routes/system/__init__.py` - System routes package
- `millm/api/routes/system/health.py` - Health check endpoint
- `millm/api/schemas/__init__.py` - Schemas package initialization
- `millm/api/schemas/common.py` - ApiResponse, ErrorDetails schemas
- `millm/api/schemas/model.py` - Model request/response schemas

### Backend - Service Layer
- `millm/services/__init__.py` - Services package initialization
- `millm/services/model_service.py` - ModelService business logic

### Backend - ML Layer
- `millm/ml/__init__.py` - ML package initialization
- `millm/ml/model_loader.py` - ModelLoader, LoadedModelState, ModelLoadContext
- `millm/ml/model_downloader.py` - ModelDownloader for HuggingFace downloads
- `millm/ml/memory_utils.py` - GPU memory utilities

### Backend - WebSocket
- `millm/sockets/__init__.py` - Sockets package initialization
- `millm/sockets/progress.py` - Socket.IO progress event handlers

### Backend - Tests
- `tests/__init__.py` - Tests package initialization
- `tests/conftest.py` - Shared pytest fixtures
- `tests/unit/__init__.py` - Unit tests package
- `tests/unit/services/__init__.py` - Service unit tests package
- `tests/unit/services/test_model_service.py` - ModelService unit tests
- `tests/unit/ml/__init__.py` - ML unit tests package
- `tests/unit/ml/test_model_loader.py` - ModelLoader unit tests
- `tests/unit/ml/test_model_downloader.py` - ModelDownloader unit tests
- `tests/unit/ml/test_memory_utils.py` - Memory utilities unit tests
- `tests/unit/db/__init__.py` - Database unit tests package
- `tests/unit/db/test_model_repository.py` - ModelRepository unit tests
- `tests/integration/__init__.py` - Integration tests package
- `tests/integration/test_model_api.py` - Model API integration tests
- `tests/integration/test_model_workflow.py` - Download→Load workflow tests

### Frontend - Types
- `frontend/src/types/index.ts` - Type re-exports
- `frontend/src/types/model.ts` - Model, DownloadRequest, etc. interfaces
- `frontend/src/types/api.ts` - ApiResponse, ErrorDetails types

### Frontend - Services
- `frontend/src/services/api.ts` - Axios instance configuration
- `frontend/src/services/modelService.ts` - Model API client
- `frontend/src/services/socketService.ts` - Socket.IO client

### Frontend - State
- `frontend/src/stores/modelStore.ts` - Zustand store for model state

### Frontend - Components
- `frontend/src/components/common/Button.tsx` - Reusable button component
- `frontend/src/components/common/Card.tsx` - Reusable card component
- `frontend/src/components/common/Input.tsx` - Reusable input component
- `frontend/src/components/common/Select.tsx` - Reusable select component
- `frontend/src/components/common/Badge.tsx` - Status badge component
- `frontend/src/components/common/ProgressBar.tsx` - Progress bar component
- `frontend/src/components/models/ModelCard.tsx` - Single model display card
- `frontend/src/components/models/ModelList.tsx` - Grid of model cards
- `frontend/src/components/models/DownloadForm.tsx` - Download input form
- `frontend/src/components/models/MemoryEstimate.tsx` - Memory display component

### Frontend - Pages
- `frontend/src/pages/ModelsPage.tsx` - Models tab page

### Frontend - Tests
- `frontend/src/stores/modelStore.test.ts` - Zustand store tests
- `frontend/src/services/modelService.test.ts` - API client tests
- `frontend/src/components/models/ModelCard.test.tsx` - ModelCard component tests
- `frontend/src/components/models/DownloadForm.test.tsx` - DownloadForm component tests
- `frontend/src/pages/ModelsPage.test.tsx` - ModelsPage integration tests

### Configuration
- `pyproject.toml` - Python project configuration (dependencies, pytest, ruff)
- `alembic.ini` - Alembic migration configuration
- `.env.example` - Environment variable template
- `docker-compose.yml` - Development environment (PostgreSQL, Redis)
- `Dockerfile` - Backend container definition

---

## Notes

### Testing
- Backend: Run `pytest` from project root, `pytest --cov=millm` for coverage
- Frontend: Run `npm test` from frontend directory, `npm run test:coverage` for coverage
- Integration tests require database: `docker-compose up -d db` first
- Use `pytest -k test_name` to run specific tests

### Development Setup
1. Copy `.env.example` to `.env` and configure
2. Run `docker-compose up -d` for PostgreSQL and Redis
3. Run `alembic upgrade head` to apply migrations
4. Run `uvicorn millm.main:app --reload` for backend
5. Run `npm run dev` from frontend directory for frontend

### Code Quality
- Backend: `ruff check .` for linting, `mypy millm/` for type checking
- Frontend: `npm run lint` for linting, `npm run typecheck` for types
- All tests must pass before committing

---

## Tasks

### Phase 1: Core Infrastructure

- [x] 1.0 Set up backend project structure and core configuration
  - [x] 1.1 Create `millm/__init__.py` with version info
  - [x] 1.2 Create `millm/core/__init__.py` package
  - [x] 1.3 Create `millm/core/config.py` with Settings class (DATABASE_URL, MODEL_CACHE_DIR, etc.)
  - [x] 1.4 Create `millm/core/errors.py` with MiLLMError hierarchy (ModelNotFoundError, ModelAlreadyExistsError, InsufficientMemoryError, etc.)
  - [x] 1.5 Create `millm/core/logging.py` with structlog configuration
  - [x] 1.6 Create `.env.example` with all environment variables
  - [x] 1.7 Create `pyproject.toml` with dependencies and tool config

- [x] 2.0 Set up database layer with SQLAlchemy async
  - [x] 2.1 Create `millm/db/__init__.py` package
  - [x] 2.2 Create `millm/db/base.py` with async engine, session factory, and get_db dependency
  - [x] 2.3 Create `millm/db/models/__init__.py` package
  - [x] 2.4 Create `millm/db/models/model.py` with Model ORM class, ModelStatus, ModelSource, QuantizationType enums
  - [x] 2.5 Create `alembic.ini` configuration
  - [x] 2.6 Create `millm/db/migrations/env.py` for async migrations
  - [x] 2.7 Create `millm/db/migrations/versions/001_create_models_table.py` migration
  - [x] 2.8 Create `millm/db/repositories/__init__.py` package
  - [x] 2.9 Create `millm/db/repositories/model_repository.py` with CRUD operations
  - [x] 2.10 Write unit tests for ModelRepository in `tests/unit/db/test_model_repository.py`

- [x] 3.0 Create API schemas and common response format
  - [x] 3.1 Create `millm/api/__init__.py` package
  - [x] 3.2 Create `millm/api/schemas/__init__.py` package
  - [x] 3.3 Create `millm/api/schemas/common.py` with ApiResponse and ErrorDetails
  - [x] 3.4 Create `millm/api/schemas/model.py` with ModelDownloadRequest, ModelResponse, ModelPreviewRequest, ModelPreviewResponse
  - [x] 3.5 Create `millm/api/exception_handlers.py` with millm_error_handler
  - [x] 3.6 Create `millm/api/dependencies.py` with get_db, get_model_service dependencies

- [x] 4.0 Create API route skeleton and health endpoint
  - [x] 4.1 Create `millm/api/routes/__init__.py` with register_routes function
  - [x] 4.2 Create `millm/api/routes/system/__init__.py` package
  - [x] 4.3 Create `millm/api/routes/system/health.py` with health check endpoint
  - [x] 4.4 Create `millm/api/routes/management/__init__.py` package
  - [x] 4.5 Create `millm/api/routes/management/models.py` with endpoint stubs (list, get, download, load, unload, delete, cancel, preview)
  - [x] 4.6 Create `millm/main.py` with FastAPI app creation and route registration
  - [x] 4.7 Verify API starts and health endpoint returns 200 (runtime verification - use `docker-compose up` or `uvicorn millm.main:app`)

### Phase 2: Download System

- [x] 5.0 Implement model downloader for HuggingFace
  - [x] 5.1 Create `millm/ml/__init__.py` package
  - [x] 5.2 Create `millm/ml/model_downloader.py` with ModelDownloader class
  - [x] 5.3 Implement `download()` method using huggingface_hub.snapshot_download
  - [x] 5.4 Implement `get_model_info()` method for model preview
  - [x] 5.5 Add progress callback support for download tracking
  - [x] 5.6 Handle HuggingFace errors (RepositoryNotFoundError, GatedRepoError)
  - [x] 5.7 Implement cache path generation (huggingface/owner--repo--quantization format)
  - [x] 5.8 Write unit tests in `tests/unit/ml/test_model_downloader.py`

- [x] 6.0 Implement WebSocket progress events
  - [x] 6.1 Create `millm/sockets/__init__.py` package
  - [x] 6.2 Create `millm/sockets/progress.py` with register_handlers function
  - [x] 6.3 Define Socket.IO namespace /progress
  - [x] 6.4 Implement model:download:progress event emission
  - [x] 6.5 Implement model:download:complete event emission
  - [x] 6.6 Implement model:download:error event emission
  - [x] 6.7 Update `millm/main.py` to create combined FastAPI + Socket.IO app

- [x] 7.0 Implement ModelService download functionality
  - [x] 7.1 Create `millm/services/__init__.py` package
  - [x] 7.2 Create `millm/services/model_service.py` with ModelService class
  - [x] 7.3 Implement `download_model()` method with validation and DB record creation
  - [x] 7.4 Implement `_download_worker()` method for ThreadPoolExecutor
  - [x] 7.5 Implement download progress callback that emits Socket.IO events
  - [x] 7.6 Implement `cancel_download()` method
  - [x] 7.7 Implement `list_models()` and `get_model()` methods
  - [x] 7.8 Implement `preview_model()` method
  - [x] 7.9 Write unit tests in `tests/unit/services/test_model_service.py`

- [x] 8.0 Wire up download API endpoints
  - [x] 8.1 Implement POST /api/models endpoint in models.py
  - [x] 8.2 Implement GET /api/models endpoint in models.py
  - [x] 8.3 Implement GET /api/models/{id} endpoint in models.py
  - [x] 8.4 Implement POST /api/models/{id}/cancel endpoint in models.py
  - [x] 8.5 Implement POST /api/models/preview endpoint in models.py
  - [x] 8.6 Write integration tests in `tests/integration/test_model_api.py`

### Phase 3: Load/Unload System

- [x] 9.0 Implement memory utilities
  - [x] 9.1 Create `millm/ml/memory_utils.py` with parse_params function
  - [x] 9.2 Implement estimate_memory_mb function (formula: params * bytes_per_quant * 1.2)
  - [x] 9.3 Implement get_available_memory_mb using torch.cuda.mem_get_info
  - [x] 9.4 Implement get_total_memory_mb function
  - [x] 9.5 Implement verify_memory_available function
  - [x] 9.6 Write unit tests in `tests/unit/ml/test_memory_utils.py`

- [x] 10.0 Implement model loader
  - [x] 10.1 Create `millm/ml/model_loader.py` with LoadedModel dataclass
  - [x] 10.2 Implement LoadedModelState singleton class with thread-safe access
  - [x] 10.3 Implement ModelLoadContext context manager for safe loading
  - [x] 10.4 Implement ModelLoader class with load() method
  - [x] 10.5 Add BitsAndBytesConfig support for Q4 and Q8 quantization
  - [x] 10.6 Implement unload() method with GPU memory cleanup
  - [x] 10.7 Write unit tests in `tests/unit/ml/test_model_loader.py`

- [x] 11.0 Implement ModelService load/unload functionality
  - [x] 11.1 Add `load_model()` method to ModelService
  - [x] 11.2 Implement auto-unload of previous model before loading new
  - [x] 11.3 Add `_load_worker()` method for ThreadPoolExecutor
  - [x] 11.4 Implement load progress events (model:load:progress, model:load:complete)
  - [x] 11.5 Add `unload_model()` method with graceful unload (wait for pending requests)
  - [x] 11.6 Add `delete_model()` method (validate not loaded, remove files, delete DB)
  - [x] 11.7 Update unit tests in `tests/unit/services/test_model_service.py`

- [x] 12.0 Wire up load/unload API endpoints
  - [x] 12.1 Implement POST /api/models/{id}/load endpoint
  - [x] 12.2 Implement POST /api/models/{id}/unload endpoint
  - [x] 12.3 Implement DELETE /api/models/{id} endpoint
  - [x] 12.4 Write integration tests for load/unload workflow
  - [x] 12.5 Write integration tests for delete operation

### Phase 4: Frontend Implementation

- [x] 13.0 Set up frontend project structure and types
  - [x] 13.1 Create `frontend/src/types/index.ts` with re-exports
  - [x] 13.2 Create `frontend/src/types/model.ts` with Model, ModelStatus, DownloadRequest, etc.
  - [x] 13.3 Create `frontend/src/types/api.ts` with ApiResponse, ErrorDetails
  - [x] 13.4 Create `frontend/src/services/api.ts` with Axios instance

- [x] 14.0 Implement model API service
  - [x] 14.1 Create `frontend/src/services/modelService.ts` with ModelService class
  - [x] 14.2 Implement listModels(), getModel(), downloadModel() methods
  - [x] 14.3 Implement loadModel(), unloadModel(), deleteModel() methods
  - [x] 14.4 Implement cancelDownload(), previewModel() methods
  - [x] 14.5 Write tests in `frontend/src/services/modelService.test.ts`

- [x] 15.0 Implement Socket.IO client service
  - [x] 15.1 Create `frontend/src/services/socketService.ts` with SocketService class
  - [x] 15.2 Implement connect() and disconnect() methods
  - [x] 15.3 Add event handlers for model:download:progress
  - [x] 15.4 Add event handlers for model:download:complete and model:download:error
  - [x] 15.5 Add event handlers for model:load:complete

- [x] 16.0 Implement Zustand store for models
  - [x] 16.1 Create `frontend/src/stores/modelStore.ts` with useModelStore
  - [x] 16.2 Implement state: models, loadedModelId, isLoading, downloadProgress, error
  - [x] 16.3 Implement fetchModels action
  - [x] 16.4 Implement downloadModel, loadModel, unloadModel, deleteModel actions
  - [x] 16.5 Implement cancelDownload action
  - [x] 16.6 Implement socket event handlers (handleDownloadProgress, etc.)
  - [x] 16.7 Write tests in `frontend/src/stores/modelStore.test.ts`

- [x] 17.0 Create common UI components
  - [x] 17.1 Create `frontend/src/components/common/Button.tsx`
  - [x] 17.2 Create `frontend/src/components/common/Card.tsx`
  - [x] 17.3 Create `frontend/src/components/common/Input.tsx`
  - [x] 17.4 Create `frontend/src/components/common/Select.tsx`
  - [x] 17.5 Create `frontend/src/components/common/Badge.tsx`
  - [x] 17.6 Create `frontend/src/components/common/ProgressBar.tsx`

- [x] 18.0 Create model-specific components
  - [x] 18.1 Create `frontend/src/components/models/DownloadForm.tsx` with validation
  - [x] 18.2 Create `frontend/src/components/models/ModelCard.tsx` with status display
  - [x] 18.3 Create `frontend/src/components/models/ModelList.tsx` grid layout
  - [x] 18.4 Create `frontend/src/components/models/MemoryEstimate.tsx`
  - [x] 18.5 Write component tests for DownloadForm
  - [x] 18.6 Write component tests for ModelCard

- [x] 19.0 Create Models page
  - [x] 19.1 Create `frontend/src/pages/ModelsPage.tsx`
  - [x] 19.2 Wire up DownloadForm with store actions
  - [x] 19.3 Wire up ModelList with store data and actions
  - [x] 19.4 Initialize Socket.IO connection on mount
  - [x] 19.5 Fetch models on mount
  - [x] 19.6 Write page integration tests

### Phase 5: Integration and Polish

- [x] 20.0 End-to-end testing
  - [x] 20.1 Create E2E test: Download model from HuggingFace (use tiny test model)
  - [x] 20.2 Create E2E test: Load model and verify status change
  - [x] 20.3 Create E2E test: Unload model and verify memory release
  - [x] 20.4 Create E2E test: Delete model and verify removal
  - [x] 20.5 Create E2E test: Cancel download in progress
  - [x] 20.6 Create E2E test: Error handling (invalid repo, insufficient memory)

- [x] 21.0 Error handling polish
  - [x] 21.1 Verify all error codes are returned correctly (14 error codes from PRD)
  - [x] 21.2 Add user-friendly error messages for all error scenarios
  - [x] 21.3 Implement retry logic for download failures (3 attempts, exponential backoff)
  - [x] 21.4 Add partial download cleanup on failure/cancel
  - [x] 21.5 Add graceful unload timeout (30 seconds)

- [x] 22.0 Performance and documentation
  - [x] 22.1 Verify API response times (<100ms for list/get)
  - [x] 22.2 Verify download start latency (<2s)
  - [x] 22.3 Add OpenAPI documentation for all endpoints
  - [x] 22.4 Verify memory estimation accuracy (within 20%)
  - [x] 22.5 Run full test suite and ensure 80%+ coverage

- [x] 23.0 Docker and deployment setup
  - [x] 23.1 Create `docker-compose.yml` with PostgreSQL and Redis services
  - [x] 23.2 Create `Dockerfile` for backend with NVIDIA runtime support
  - [x] 23.3 Create volume mount for model cache directory
  - [x] 23.4 Verify Docker deployment works end-to-end
  - [x] 23.5 Document deployment steps in README

---

## Implementation Notes

### Priority Order
1. Tasks 1-4 (Core Infrastructure) - Foundation, must be done first
2. Tasks 5-8 (Download System) - Core functionality
3. Tasks 9-12 (Load/Unload System) - Completes backend
4. Tasks 13-19 (Frontend) - Can partially parallel with backend
5. Tasks 20-23 (Integration) - Final polish

### Dependencies
- Tasks 2.x depend on 1.x (database needs config)
- Tasks 3.x depend on 2.x (schemas need models)
- Tasks 4.x depend on 3.x (routes need schemas)
- Tasks 5-8 depend on 4.x (download needs API skeleton)
- Tasks 9-12 depend on 5-8 (load needs download complete)
- Tasks 13-19 depend on 4.x (frontend needs API endpoints)
- Tasks 20-23 depend on all previous tasks

### Testing Strategy
- Write unit tests alongside implementation (sub-task pattern: implement, then test)
- Integration tests after each phase
- E2E tests in Phase 5

### Estimated Timeline
- Phase 1 (Core Infrastructure): 3-4 days
- Phase 2 (Download System): 3-4 days
- Phase 3 (Load/Unload System): 3-4 days
- Phase 4 (Frontend): 4-5 days
- Phase 5 (Integration): 2-3 days
- **Total: ~3-4 weeks**

---

**Document Status:** Complete
**Next Step:** Begin implementation with Task 1.0
**Instruction File for Execution:** `@0xcc/instruct/007_process-task-list.md`
