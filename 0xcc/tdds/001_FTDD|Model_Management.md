# Technical Design Document: Model Management

## miLLM Feature 1

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `001_FPRD|Model_Management.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Executive Summary

Model Management provides the foundational capability for miLLM to acquire, store, and load LLM models. The design follows an **"Ollama-simple"** philosophy: when no SAE is attached and no steering is configured, miLLM should behave exactly like Ollama - download models, load them, serve inference requests.

### Design Principles
1. **Simplicity First:** No over-engineering. Simple status fields, not state machines.
2. **Predictable Behavior:** Download → Ready → Load → Serving. Clear states.
3. **Single User Assumption:** v1.0 targets local deployment with one user.
4. **Automatic Cleanup:** Context managers handle resource cleanup.
5. **Standard Tools:** Use huggingface_hub, bitsandbytes as designed.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Threading | ThreadPoolExecutor | Simple, no Celery overhead |
| State | Status enum field | Business logic validates transitions |
| Progress | Broadcast to all clients | Single user assumption |
| Cache | Hierarchical directories | Clean, browsable |
| Cleanup | Context managers | Automatic, Pythonic |
| Quantization | bitsandbytes auto | Load with config, cache quantized |

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ ModelsPage   │  │ modelStore   │  │ socketService        │  │
│  │ (UI)         │◄─┤ (Zustand)    │◄─┤ (Socket.IO client)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/WS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│                                                                  │
│  ┌─────────────────┐     ┌─────────────────────────────────┐   │
│  │ API Routes      │     │ Socket.IO Server                │   │
│  │ /api/models/*   │     │ /progress namespace             │   │
│  └────────┬────────┘     └────────────────┬────────────────┘   │
│           │                               │                     │
│           ▼                               │                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   ModelService                           │   │
│  │  - download_model()  - load_model()  - unload_model()   │   │
│  │  - list_models()     - delete_model() - get_model()     │   │
│  └────────┬─────────────────────┬──────────────────────────┘   │
│           │                     │                               │
│           ▼                     ▼                               │
│  ┌─────────────────┐   ┌─────────────────┐                     │
│  │ ModelRepository │   │ ModelLoader     │                     │
│  │ (PostgreSQL)    │   │ (ML Operations) │                     │
│  └─────────────────┘   └─────────────────┘                     │
│                               │                                 │
│                               ▼                                 │
│                      ┌─────────────────┐                       │
│                      │ GPU Memory      │                       │
│                      │ (Loaded Model)  │                       │
│                      └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      External Services                           │
│  ┌─────────────────┐   ┌─────────────────┐                     │
│  │ HuggingFace Hub │   │ Local Filesystem│                     │
│  │ (Model Source)  │   │ (Model Cache)   │                     │
│  └─────────────────┘   └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Component Relationships

1. **Frontend → API:** REST calls for CRUD, WebSocket for real-time progress
2. **API → Service:** Thin routes delegate to ModelService
3. **Service → Repository:** Database operations for metadata
4. **Service → Loader:** ML operations for download/load
5. **Loader → HuggingFace:** Downloads via huggingface_hub
6. **Loader → GPU:** Model loading with bitsandbytes

### Data Flow: Download Model

```
User clicks Download
        │
        ▼
POST /api/models ─────────► ModelService.download_model()
        │                           │
        │                           ├─► Validate request
        │                           ├─► Create DB record (status: downloading)
        │                           ├─► Submit to ThreadPoolExecutor
        │                           └─► Return immediately (202 Accepted)
        │
        │   (In background thread)
        │                           │
        │                           ├─► huggingface_hub.hf_hub_download()
        │                           │       └─► Progress callback
        │                           │               └─► sio.emit('model:download:progress')
        │                           │
        │                           ├─► Apply quantization (if not FP16)
        │                           ├─► Update DB (status: ready)
        │                           └─► sio.emit('model:download:complete')
        │
        ▼
WebSocket ◄──────────────── Progress events to frontend
```

### Data Flow: Load Model

```
User clicks Load
        │
        ▼
POST /api/models/{id}/load ─► ModelService.load_model()
        │                           │
        │                           ├─► Check current loaded model
        │                           │       └─► If loaded: graceful_unload()
        │                           │
        │                           ├─► Verify GPU memory
        │                           ├─► Update DB (status: loading)
        │                           ├─► Submit to ThreadPoolExecutor
        │                           └─► Return immediately (202 Accepted)
        │
        │   (In background thread)
        │                           │
        │                           ├─► with ModelLoadContext():
        │                           │       ├─► AutoModelForCausalLM.from_pretrained()
        │                           │       ├─► Move to GPU
        │                           │       └─► Store reference in LoadedModelState
        │                           │
        │                           ├─► Update DB (status: loaded)
        │                           └─► sio.emit('model:load:complete')
        │
        ▼
WebSocket ◄──────────────── Load complete event
```

---

## 3. Technical Stack

### Backend Dependencies

```python
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.0
python-socketio>=5.10.0

# Database
sqlalchemy>=2.0
asyncpg>=0.29.0
alembic>=1.13.0

# ML/AI
torch>=2.0
transformers>=4.36.0
huggingface-hub>=0.20.0
bitsandbytes>=0.42.0
accelerate>=0.25.0

# Utilities
structlog>=24.1.0
python-dotenv>=1.0.0
```

### Frontend Dependencies

```json
{
  "react": "^18.2.0",
  "zustand": "^4.x",
  "@tanstack/react-query": "^5.x",
  "socket.io-client": "^4.x"
}
```

### Technology Justification

| Technology | Purpose | Why |
|------------|---------|-----|
| ThreadPoolExecutor | Background tasks | Simple, no Celery overhead for single-user |
| huggingface_hub | Downloads | Native progress callbacks, handles auth |
| bitsandbytes | Quantization | Industry standard, Transformers integrated |
| Socket.IO | Real-time | Already in stack for monitoring |
| Pydantic v2 | Validation | Fast, type-safe, OpenAPI generation |

---

## 4. Data Design

### Database Schema

```python
# millm/db/models/model.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Enum
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import enum

class ModelStatus(str, enum.Enum):
    DOWNLOADING = "downloading"
    READY = "ready"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"

class ModelSource(str, enum.Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class QuantizationType(str, enum.Enum):
    Q4 = "Q4"
    Q8 = "Q8"
    FP16 = "FP16"

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    source = Column(Enum(ModelSource), nullable=False)
    repo_id = Column(String(255), nullable=True)
    local_path = Column(String(500), nullable=True)

    # Model metadata
    params = Column(String(50))  # "2.5B", "9B"
    architecture = Column(String(100))
    quantization = Column(Enum(QuantizationType), nullable=False)

    # Storage info
    disk_size_mb = Column(Integer)
    estimated_memory_mb = Column(Integer)
    cache_path = Column(String(500), nullable=False)

    # Configuration
    config_json = Column(JSONB)
    trust_remote_code = Column(Boolean, default=False)

    # State
    status = Column(Enum(ModelStatus), default=ModelStatus.READY)
    error_message = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    loaded_at = Column(DateTime, nullable=True)

    # Constraints
    __table_args__ = (
        # Unique constraint: same repo with same quantization
        UniqueConstraint('repo_id', 'quantization', name='uq_repo_quantization'),
        # Unique local path
        UniqueConstraint('local_path', name='uq_local_path'),
    )
```

### Cache Directory Structure

```
$MODEL_CACHE_DIR/
├── huggingface/
│   ├── google--gemma-2-2b--Q4/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── .millm_meta.json          # miLLM metadata
│   │
│   └── meta-llama--Llama-3.2-3B--Q8/
│       └── ...
│
└── local/
    └── my-finetuned-model/           # Symlink to original path
        └── ...
```

### Metadata File (.millm_meta.json)

```json
{
  "model_id": 1,
  "downloaded_at": "2026-01-30T12:00:00Z",
  "source": "huggingface",
  "repo_id": "google/gemma-2-2b",
  "quantization": "Q4",
  "original_size_bytes": 5368709120,
  "quantized_size_bytes": 1879048192,
  "transformers_version": "4.36.0",
  "bitsandbytes_version": "0.42.0"
}
```

### Data Validation Strategy

```python
# Pydantic schemas handle validation

class ModelDownloadRequest(BaseModel):
    source: ModelSource
    repo_id: Optional[str] = Field(None, pattern=r'^[\w-]+/[\w.-]+$')
    local_path: Optional[str] = None
    quantization: QuantizationType = QuantizationType.Q4
    trust_remote_code: bool = False
    hf_token: Optional[str] = None
    custom_name: Optional[str] = None

    @model_validator(mode='after')
    def validate_source_fields(self):
        if self.source == ModelSource.HUGGINGFACE and not self.repo_id:
            raise ValueError("repo_id required for HuggingFace source")
        if self.source == ModelSource.LOCAL and not self.local_path:
            raise ValueError("local_path required for local source")
        return self
```

---

## 5. API Design

### Route Structure

```python
# millm/api/routes/management/models.py

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("")
async def list_models() -> ApiResponse[List[ModelResponse]]:
    """List all downloaded models."""

@router.post("")
async def download_model(request: ModelDownloadRequest) -> ApiResponse[ModelResponse]:
    """Start model download. Returns immediately with status: downloading."""

@router.get("/{model_id}")
async def get_model(model_id: int) -> ApiResponse[ModelResponse]:
    """Get single model details."""

@router.delete("/{model_id}")
async def delete_model(model_id: int) -> ApiResponse[None]:
    """Delete model from disk and database."""

@router.post("/{model_id}/load")
async def load_model(model_id: int) -> ApiResponse[ModelResponse]:
    """Load model into GPU memory."""

@router.post("/{model_id}/unload")
async def unload_model(model_id: int) -> ApiResponse[ModelResponse]:
    """Unload model from GPU memory."""

@router.post("/{model_id}/cancel")
async def cancel_download(model_id: int) -> ApiResponse[None]:
    """Cancel in-progress download."""

@router.post("/preview")
async def preview_model(request: ModelPreviewRequest) -> ApiResponse[ModelPreviewResponse]:
    """Get model info from HuggingFace without downloading."""
```

### Response Format

```python
# millm/api/schemas/common.py

class ErrorDetails(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[ErrorDetails] = None

    @classmethod
    def ok(cls, data: T) -> "ApiResponse[T]":
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, code: str, message: str, details: Dict = None) -> "ApiResponse[T]":
        return cls(success=False, error=ErrorDetails(code=code, message=message, details=details))
```

### Error Handling Strategy

```python
# millm/core/errors.py

class MiLLMError(Exception):
    """Base exception for miLLM."""
    code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(self, message: str, details: Dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class ModelNotFoundError(MiLLMError):
    code = "MODEL_NOT_FOUND"
    status_code = 404

class ModelAlreadyExistsError(MiLLMError):
    code = "MODEL_ALREADY_EXISTS"
    status_code = 409

class InsufficientMemoryError(MiLLMError):
    code = "INSUFFICIENT_MEMORY"
    status_code = 507

class DownloadFailedError(MiLLMError):
    code = "DOWNLOAD_FAILED"
    status_code = 502

# Exception handler in FastAPI
@app.exception_handler(MiLLMError)
async def millm_error_handler(request: Request, exc: MiLLMError):
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse.fail(exc.code, exc.message, exc.details).model_dump()
    )
```

---

## 6. Component Architecture

### Backend Components

```
millm/
├── api/
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── management/
│   │   │   ├── __init__.py
│   │   │   └── models.py          # Model CRUD endpoints
│   │   └── system/
│   │       └── health.py          # Health check
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── common.py              # ApiResponse, ErrorDetails
│   │   └── model.py               # Model request/response schemas
│   └── dependencies.py            # Dependency injection
│
├── services/
│   ├── __init__.py
│   └── model_service.py           # Business logic
│
├── ml/
│   ├── __init__.py
│   ├── model_loader.py            # Load models into GPU
│   ├── model_downloader.py        # Download from HuggingFace
│   └── memory_utils.py            # GPU memory utilities
│
├── db/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py               # SQLAlchemy model
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── model_repository.py    # Database operations
│   └── migrations/
│       └── versions/
│
├── sockets/
│   ├── __init__.py
│   └── progress.py                # Progress event handlers
│
├── core/
│   ├── __init__.py
│   ├── config.py                  # Settings management
│   ├── errors.py                  # Custom exceptions
│   └── logging.py                 # Structured logging setup
│
└── main.py                        # Application entry point
```

### Service Layer Design

```python
# millm/services/model_service.py

class ModelService:
    """
    Model management business logic.

    Orchestrates between repository (DB), loader (ML), and events (Socket.IO).
    All long-running operations submitted to thread pool.
    """

    def __init__(
        self,
        repository: ModelRepository,
        loader: ModelLoader,
        downloader: ModelDownloader,
        sio: AsyncServer,
        executor: ThreadPoolExecutor
    ):
        self.repository = repository
        self.loader = loader
        self.downloader = downloader
        self.sio = sio
        self.executor = executor
        self._loaded_model: Optional[LoadedModel] = None
        self._pending_downloads: Dict[int, asyncio.Future] = {}

    async def download_model(self, request: ModelDownloadRequest) -> Model:
        """
        Start model download in background.
        Returns immediately with model in 'downloading' status.
        """
        # 1. Validate not duplicate
        existing = await self.repository.find_by_repo_quantization(
            request.repo_id, request.quantization
        )
        if existing:
            raise ModelAlreadyExistsError(f"Model already exists with id={existing.id}")

        # 2. Create DB record
        model = await self.repository.create(
            name=request.custom_name or self._derive_name(request),
            source=request.source,
            repo_id=request.repo_id,
            quantization=request.quantization,
            status=ModelStatus.DOWNLOADING,
            # ... other fields
        )

        # 3. Submit download to thread pool
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            self._download_worker,
            model.id,
            request
        )
        self._pending_downloads[model.id] = future

        return model

    def _download_worker(self, model_id: int, request: ModelDownloadRequest):
        """
        Runs in thread pool. Downloads and optionally quantizes model.
        Emits progress via Socket.IO.
        """
        try:
            # Download with progress callback
            def progress_callback(progress: float, downloaded: int, total: int):
                # Socket.IO emit from thread
                asyncio.run(self.sio.emit(
                    'model:download:progress',
                    {'model_id': model_id, 'progress': progress, ...},
                    namespace='/progress'
                ))

            cache_path = self.downloader.download(
                repo_id=request.repo_id,
                quantization=request.quantization,
                progress_callback=progress_callback,
                token=request.hf_token
            )

            # Update DB
            asyncio.run(self.repository.update(model_id,
                status=ModelStatus.READY,
                cache_path=cache_path
            ))

            # Emit completion
            asyncio.run(self.sio.emit(
                'model:download:complete',
                {'model_id': model_id},
                namespace='/progress'
            ))

        except Exception as e:
            # Update DB with error
            asyncio.run(self.repository.update(model_id,
                status=ModelStatus.ERROR,
                error_message=str(e)
            ))
            asyncio.run(self.sio.emit(
                'model:download:error',
                {'model_id': model_id, 'error': str(e)},
                namespace='/progress'
            ))
```

### Frontend Components

```
src/
├── components/
│   └── models/
│       ├── ModelCard.tsx           # Single model display
│       ├── ModelList.tsx           # List of models
│       ├── DownloadForm.tsx        # Download input form
│       ├── ProgressBar.tsx         # Download/load progress
│       └── MemoryEstimate.tsx      # Memory usage display
│
├── pages/
│   └── ModelsPage.tsx              # Models tab page
│
├── stores/
│   └── modelStore.ts               # Zustand store
│
├── services/
│   └── modelService.ts             # API client
│
├── hooks/
│   └── useModels.ts                # Custom hook
│
└── types/
    └── model.ts                    # TypeScript interfaces
```

---

## 7. State Management

### Backend State: Loaded Model Singleton

```python
# millm/ml/model_loader.py

@dataclass
class LoadedModel:
    """Represents a model loaded in GPU memory."""
    model_id: int
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    config: PretrainedConfig
    loaded_at: datetime

class LoadedModelState:
    """
    Singleton managing the currently loaded model.
    Only one model can be loaded at a time.
    """
    _instance: Optional['LoadedModelState'] = None
    _loaded: Optional[LoadedModel] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'LoadedModelState':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def current(self) -> Optional[LoadedModel]:
        return self._loaded

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def set(self, model: LoadedModel):
        with self._lock:
            self._loaded = model

    def clear(self):
        with self._lock:
            if self._loaded:
                # Explicit cleanup
                del self._loaded.model
                del self._loaded.tokenizer
                torch.cuda.empty_cache()
                gc.collect()
            self._loaded = None
```

### Frontend State: Zustand Store

```typescript
// src/stores/modelStore.ts

interface ModelState {
  // Data
  models: Model[];
  loadedModelId: number | null;

  // UI State
  isLoading: boolean;
  downloadProgress: Record<number, DownloadProgress>;
  error: string | null;

  // Actions
  fetchModels: () => Promise<void>;
  downloadModel: (request: ModelDownloadRequest) => Promise<void>;
  loadModel: (id: number) => Promise<void>;
  unloadModel: (id: number) => Promise<void>;
  deleteModel: (id: number) => Promise<void>;
  cancelDownload: (id: number) => Promise<void>;

  // Socket handlers
  handleDownloadProgress: (data: DownloadProgressEvent) => void;
  handleDownloadComplete: (data: DownloadCompleteEvent) => void;
  handleLoadComplete: (data: LoadCompleteEvent) => void;
}

export const useModelStore = create<ModelState>((set, get) => ({
  models: [],
  loadedModelId: null,
  isLoading: false,
  downloadProgress: {},
  error: null,

  fetchModels: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await modelService.listModels();
      if (response.success) {
        set({ models: response.data, isLoading: false });
        // Find loaded model
        const loaded = response.data.find(m => m.status === 'loaded');
        set({ loadedModelId: loaded?.id ?? null });
      }
    } catch (e) {
      set({ error: e.message, isLoading: false });
    }
  },

  loadModel: async (id: number) => {
    set({ isLoading: true, error: null });
    try {
      await modelService.loadModel(id);
      // Actual state update happens via WebSocket
    } catch (e) {
      set({ error: e.message, isLoading: false });
    }
  },

  handleDownloadProgress: (data) => {
    set((state) => ({
      downloadProgress: {
        ...state.downloadProgress,
        [data.model_id]: data
      }
    }));
  },

  handleLoadComplete: (data) => {
    set((state) => ({
      isLoading: false,
      loadedModelId: data.model_id,
      models: state.models.map(m => ({
        ...m,
        status: m.id === data.model_id ? 'loaded' :
                m.status === 'loaded' ? 'ready' : m.status
      }))
    }));
  },

  // ... other actions
}));
```

### WebSocket Integration

```typescript
// src/services/socketService.ts

class SocketService {
  private socket: Socket;

  connect() {
    this.socket = io('/progress', {
      transports: ['websocket'],
    });

    this.socket.on('model:download:progress', (data) => {
      useModelStore.getState().handleDownloadProgress(data);
    });

    this.socket.on('model:download:complete', (data) => {
      useModelStore.getState().handleDownloadComplete(data);
    });

    this.socket.on('model:load:complete', (data) => {
      useModelStore.getState().handleLoadComplete(data);
    });
  }
}
```

---

## 8. Security Considerations

### Input Validation

```python
# All inputs validated via Pydantic

class ModelDownloadRequest(BaseModel):
    repo_id: Optional[str] = Field(
        None,
        pattern=r'^[\w-]+/[\w.-]+$',  # Prevent path traversal
        max_length=255
    )
    local_path: Optional[str] = Field(None, max_length=500)

    @field_validator('local_path')
    @classmethod
    def validate_local_path(cls, v):
        if v is None:
            return v
        path = Path(v)
        # Must be absolute
        if not path.is_absolute():
            raise ValueError("local_path must be absolute")
        # Must exist
        if not path.exists():
            raise ValueError("local_path does not exist")
        # Resolve to prevent symlink attacks
        resolved = path.resolve()
        # Check it's a directory
        if not resolved.is_dir():
            raise ValueError("local_path must be a directory")
        return str(resolved)
```

### Trust Remote Code

```python
# Explicit opt-in required, logged

def load_model(self, cache_path: str, trust_remote_code: bool = False):
    if trust_remote_code:
        logger.warning(
            "Loading model with trust_remote_code=True",
            cache_path=cache_path
        )

    return AutoModelForCausalLM.from_pretrained(
        cache_path,
        trust_remote_code=trust_remote_code,
        # ... other args
    )
```

### HuggingFace Token Handling

```python
# Token only in memory, never persisted

class ModelDownloadRequest(BaseModel):
    hf_token: Optional[str] = Field(
        None,
        exclude=True  # Exclude from serialization/logging
    )

# In downloader
def download(self, repo_id: str, token: Optional[str] = None):
    # Token passed directly to huggingface_hub
    # Never stored in DB or cache
    hf_hub_download(
        repo_id=repo_id,
        token=token,  # Used only for this request
        ...
    )
```

---

## 9. Performance & Scalability

### Performance Targets

| Operation | Target | Strategy |
|-----------|--------|----------|
| List models | <50ms | Simple DB query, indexed |
| Download start | <2s | Immediate return, background thread |
| Model load (2B Q4) | <30s | Direct GPU load |
| Memory query | <500ms | torch.cuda calls |
| Unload | <5s | Reference deletion + gc |

### Memory Estimation

```python
# millm/ml/memory_utils.py

def estimate_memory_mb(params_str: str, quantization: QuantizationType) -> int:
    """
    Estimate VRAM needed for model.

    Rough formula:
    - FP16: params * 2 bytes
    - Q8: params * 1 byte
    - Q4: params * 0.5 bytes
    Plus ~20% overhead for KV cache, activations
    """
    # Parse params (e.g., "2.5B" -> 2.5e9)
    params = parse_params(params_str)

    bytes_per_param = {
        QuantizationType.FP16: 2.0,
        QuantizationType.Q8: 1.0,
        QuantizationType.Q4: 0.5,
    }

    base_bytes = params * bytes_per_param[quantization]
    with_overhead = base_bytes * 1.2  # 20% overhead

    return int(with_overhead / (1024 * 1024))  # Convert to MB

def get_available_memory_mb() -> int:
    """Get available GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0

    free, total = torch.cuda.mem_get_info()
    return int(free / (1024 * 1024))

def verify_memory_available(required_mb: int) -> Tuple[bool, int]:
    """
    Check if enough memory available.
    Returns (is_available, available_mb).
    """
    available = get_available_memory_mb()
    return (available >= required_mb, available)
```

### Graceful Unload

```python
# millm/services/model_service.py

async def graceful_unload(self, timeout: float = 30.0) -> bool:
    """
    Wait for pending requests, then unload.
    Returns True if unloaded, False if timed out.
    """
    if not self._loaded_model:
        return True

    # Wait for pending inference requests
    start = time.time()
    while self._pending_requests > 0:
        if time.time() - start > timeout:
            logger.warning("Graceful unload timed out, forcing unload")
            break
        await asyncio.sleep(0.1)

    # Clear loaded model
    LoadedModelState.get_instance().clear()

    # Update DB
    await self.repository.update(
        self._loaded_model.model_id,
        status=ModelStatus.READY,
        loaded_at=None
    )

    return True
```

---

## 10. Testing Strategy

### Unit Tests

```python
# tests/services/test_model_service.py

class TestModelService:
    @pytest.fixture
    def service(self, mock_repository, mock_loader, mock_sio):
        return ModelService(
            repository=mock_repository,
            loader=mock_loader,
            downloader=MockDownloader(),
            sio=mock_sio,
            executor=ThreadPoolExecutor(max_workers=1)
        )

    async def test_download_model_creates_record(self, service, mock_repository):
        """Download should create DB record immediately."""
        request = ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE,
            repo_id="google/gemma-2-2b",
            quantization=QuantizationType.Q4
        )

        model = await service.download_model(request)

        assert model.status == ModelStatus.DOWNLOADING
        mock_repository.create.assert_called_once()

    async def test_load_unloads_previous(self, service):
        """Loading new model should unload previous."""
        # Setup: model 1 already loaded
        service._loaded_model = LoadedModel(model_id=1, ...)

        await service.load_model(model_id=2)

        # Previous model should be unloaded
        assert service._loaded_model.model_id == 2
```

### Integration Tests

```python
# tests/integration/test_model_api.py

class TestModelAPI:
    @pytest.fixture
    def client(self, app):
        return TestClient(app)

    async def test_download_then_load_workflow(self, client, test_db):
        """Full workflow: download → ready → load."""
        # Start download (use tiny test model)
        response = client.post("/api/models", json={
            "source": "huggingface",
            "repo_id": "hf-internal-testing/tiny-random-gpt2",
            "quantization": "FP16"
        })
        assert response.status_code == 200
        model_id = response.json()["data"]["id"]

        # Wait for download (poll status)
        for _ in range(60):
            response = client.get(f"/api/models/{model_id}")
            if response.json()["data"]["status"] == "ready":
                break
            time.sleep(1)

        # Load model
        response = client.post(f"/api/models/{model_id}/load")
        assert response.status_code == 200

        # Verify loaded
        response = client.get(f"/api/models/{model_id}")
        assert response.json()["data"]["status"] == "loaded"
```

### Mock Strategy

```python
# tests/conftest.py

@pytest.fixture
def mock_huggingface(mocker):
    """Mock HuggingFace hub operations."""
    mock = mocker.patch('huggingface_hub.hf_hub_download')
    mock.return_value = "/tmp/test-model"
    return mock

@pytest.fixture
def mock_torch_cuda(mocker):
    """Mock CUDA operations for testing without GPU."""
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.mem_get_info', return_value=(8*1024**3, 24*1024**3))
    mocker.patch('torch.cuda.empty_cache')

@pytest.fixture
def mock_transformers(mocker):
    """Mock model loading."""
    mock_model = MagicMock()
    mock_model.config.name_or_path = "test-model"
    mocker.patch(
        'transformers.AutoModelForCausalLM.from_pretrained',
        return_value=mock_model
    )
```

---

## 11. Deployment & DevOps

### Environment Configuration

```bash
# .env

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/millm

# Model cache
MODEL_CACHE_DIR=/data/models

# HuggingFace (optional, for gated models)
HF_TOKEN=hf_xxxxxxxxxxxx

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # or "console" for development

# Server
HOST=0.0.0.0
PORT=8000
```

### Docker Compose

```yaml
# docker-compose.yml

services:
  millm:
    build: .
    runtime: nvidia
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/millm
      - MODEL_CACHE_DIR=/data/models
    volumes:
      - model-cache:/data/models
    ports:
      - "8000:8000"
    depends_on:
      - db
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=millm
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  model-cache:
  postgres-data:
```

### Monitoring & Logging

```python
# millm/core/logging.py

import structlog

def setup_logging():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()  # or ConsoleRenderer() for dev
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )

# Usage in code
logger = structlog.get_logger()

logger.info("model_download_started",
    model_id=model.id,
    repo_id=request.repo_id,
    quantization=request.quantization
)
```

---

## 12. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| bitsandbytes GPU compatibility | Medium | High | Test on target GPUs; fallback to FP16 |
| HuggingFace download failures | Medium | Medium | Retry logic; clear error messages |
| OOM during load | Medium | Medium | Pre-check memory; context manager cleanup |
| Thread pool exhaustion | Low | Medium | Max workers = 2; queue downloads |
| Socket.IO disconnect | Low | Low | Polling fallback; reconnection |

### Mitigation Strategies

```python
# bitsandbytes fallback
def load_with_fallback(cache_path: str, quantization: QuantizationType):
    """Attempt quantized load, fallback to FP16 if fails."""
    if quantization in (QuantizationType.Q4, QuantizationType.Q8):
        try:
            return load_quantized(cache_path, quantization)
        except Exception as e:
            logger.warning("Quantized load failed, falling back to FP16", error=str(e))
            return load_fp16(cache_path)
    return load_fp16(cache_path)
```

### Alternative Approaches Considered

| Approach | Considered For | Why Rejected |
|----------|----------------|--------------|
| Celery workers | Background tasks | Overkill for single-user inference |
| State machine lib | Status management | Simple enum sufficient |
| Redis pub/sub | Progress events | Adds complexity for single-user |
| Subprocess loading | Error isolation | Complexity, IPC overhead |

---

## 13. Development Phases

### Phase 1: Core Infrastructure (3-4 days)
- [ ] Database schema and migrations
- [ ] Model repository (CRUD)
- [ ] Core error classes
- [ ] API route skeleton
- [ ] Basic tests

### Phase 2: Download System (3-4 days)
- [ ] HuggingFace downloader
- [ ] Progress tracking
- [ ] Socket.IO integration
- [ ] Cancel functionality
- [ ] Download tests

### Phase 3: Load/Unload System (3-4 days)
- [ ] Model loader with quantization
- [ ] Memory estimation
- [ ] Graceful unload
- [ ] Loaded state management
- [ ] Load tests

### Phase 4: Frontend (4-5 days)
- [ ] Zustand store
- [ ] API service
- [ ] Socket.IO client
- [ ] UI components
- [ ] Page assembly

### Phase 5: Integration & Polish (2-3 days)
- [ ] End-to-end testing
- [ ] Error handling polish
- [ ] Documentation
- [ ] Performance optimization

### Milestone Definitions

| Milestone | Definition | Estimated |
|-----------|------------|-----------|
| M1: API Complete | All endpoints working, tested | Week 1 |
| M2: Backend Complete | Download + Load working E2E | Week 2 |
| M3: Frontend Complete | UI functional, connected | Week 3 |
| M4: Feature Complete | All requirements met | Week 3-4 |

---

## Appendix A: Model Load Context Manager

```python
# millm/ml/model_loader.py

class ModelLoadContext:
    """
    Context manager for safe model loading.
    Ensures cleanup on any failure.
    """

    def __init__(self, loader: 'ModelLoader', model_id: int):
        self.loader = loader
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Error occurred - cleanup
            logger.error("Model load failed, cleaning up",
                model_id=self.model_id,
                error=str(exc_val)
            )

            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer

            torch.cuda.empty_cache()
            gc.collect()

        return False  # Don't suppress exception

    def load(self, cache_path: str, **kwargs) -> LoadedModel:
        """Load model within context."""
        self.tokenizer = AutoTokenizer.from_pretrained(cache_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            **kwargs
        )

        return LoadedModel(
            model_id=self.model_id,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.model.config,
            loaded_at=datetime.utcnow()
        )
```

---

## Appendix B: API Response Examples

### List Models
```json
GET /api/models

{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "gemma-2-2b",
      "source": "huggingface",
      "repo_id": "google/gemma-2-2b",
      "params": "2.5B",
      "quantization": "Q4",
      "disk_size_mb": 1800,
      "estimated_memory_mb": 2160,
      "status": "loaded",
      "created_at": "2026-01-30T12:00:00Z",
      "loaded_at": "2026-01-30T12:05:00Z"
    }
  ]
}
```

### Download Error
```json
POST /api/models

{
  "success": false,
  "data": null,
  "error": {
    "code": "REPO_NOT_FOUND",
    "message": "HuggingFace repository 'invalid/repo' not found",
    "details": {
      "repo_id": "invalid/repo"
    }
  }
}
```

---

**Document Status:** Complete
**Next Document:** `001_FTID|Model_Management.md` (Technical Implementation Document)
**Instruction File:** `@0xcc/instruct/005_create-tid.md`
