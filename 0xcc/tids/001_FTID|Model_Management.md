# Technical Implementation Document: Model Management

## miLLM Feature 1

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `001_FPRD|Model_Management.md`
- TDD: `001_FTDD|Model_Management.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Implementation Overview

### Summary
Model Management is implemented as a greenfield feature following the "Ollama-simple" philosophy. The implementation consists of a FastAPI backend with SQLAlchemy/PostgreSQL persistence, a Python ML layer using huggingface_hub and bitsandbytes, and a React frontend with Zustand state management.

### Key Implementation Principles
1. **Simplicity Over Abstraction:** Direct implementations, minimal inheritance hierarchies
2. **Async-First Backend:** All I/O operations use async/await
3. **Type Safety:** Pydantic v2 for API validation, SQLAlchemy 2.0 type hints
4. **Fail Fast:** Validate early, return clear errors
5. **Resource Cleanup:** Context managers for GPU memory, explicit cleanup paths

### Integration Points
- **Database:** PostgreSQL via SQLAlchemy async
- **Cache:** Redis for ephemeral download progress (optional, can use in-memory for v1.0)
- **Real-time:** Socket.IO for progress events
- **ML Libraries:** Transformers, huggingface_hub, bitsandbytes

### Implementation Notes (Post-Implementation)

**AUTO_LOAD_MODEL:** The `main.py` lifespan handler supports an `AUTO_LOAD_MODEL` setting (in `core/config.py`). When set, the `_auto_load_model()` function runs during startup, looking up a model by numeric ID or name, then calling `service.load_model()` with a polling loop (up to 120s) to wait for the load to complete.

**Local Model Path Support:** When `source == ModelSource.LOCAL`, the `download_model()` method in `ModelService` validates the directory exists, calculates disk size from the local path, and marks the model as `ready` immediately (no download needed). The `local_path` is used directly as `cache_path`.

**Circuit Breaker Pattern:** HuggingFace API calls are protected by a circuit breaker implemented in `core/resilience.py`. The `CircuitBreaker` class has `CLOSED`, `OPEN`, and `HALF_OPEN` states with configurable failure threshold and recovery timeout. The health endpoint at `/api/health` exposes circuit breaker status.

---

## 2. File Structure and Organization

### Backend Directory Structure

```
millm/
├── __init__.py
├── main.py                          # FastAPI app entry point
│
├── api/
│   ├── __init__.py
│   ├── dependencies.py              # Dependency injection (get_db, get_service)
│   ├── routes/
│   │   ├── __init__.py              # Route registration
│   │   ├── management/
│   │   │   ├── __init__.py
│   │   │   └── models.py            # /api/models/* endpoints
│   │   └── system/
│   │       ├── __init__.py
│   │       └── health.py            # /api/health endpoint
│   └── schemas/
│       ├── __init__.py
│       ├── common.py                # ApiResponse, ErrorDetails
│       └── model.py                 # Model request/response schemas
│
├── services/
│   ├── __init__.py
│   └── model_service.py             # ModelService class
│
├── ml/
│   ├── __init__.py
│   ├── model_loader.py              # ModelLoader, LoadedModelState, ModelLoadContext
│   ├── model_downloader.py          # ModelDownloader class
│   └── memory_utils.py              # GPU memory utilities
│
├── db/
│   ├── __init__.py
│   ├── base.py                      # SQLAlchemy Base, engine, session
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py                 # Model ORM class
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── model_repository.py      # ModelRepository class
│   └── migrations/
│       ├── env.py
│       └── versions/
│           └── 001_create_models_table.py
│
├── sockets/
│   ├── __init__.py
│   └── progress.py                  # Socket.IO namespace and events
│
└── core/
    ├── __init__.py
    ├── config.py                    # Settings class (pydantic-settings)
    ├── errors.py                    # MiLLMError hierarchy
    └── logging.py                   # structlog setup
```

### Frontend Directory Structure

```
src/
├── App.tsx
├── main.tsx
│
├── components/
│   ├── common/
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Input.tsx
│   │   ├── Select.tsx
│   │   ├── Badge.tsx
│   │   └── ProgressBar.tsx
│   └── models/
│       ├── ModelCard.tsx            # Single model display
│       ├── ModelList.tsx            # Grid of ModelCards
│       ├── DownloadForm.tsx         # Download input form
│       ├── MemoryEstimate.tsx       # Memory display component
│       └── ModelPreviewModal.tsx    # Preview before download
│
├── pages/
│   └── ModelsPage.tsx               # Models tab content
│
├── stores/
│   └── modelStore.ts                # Zustand store
│
├── services/
│   ├── api.ts                       # Axios instance config
│   ├── modelService.ts              # Model API client
│   └── socketService.ts             # Socket.IO client
│
├── hooks/
│   └── useModels.ts                 # React Query hooks
│
├── types/
│   ├── index.ts                     # Re-exports
│   ├── model.ts                     # Model interfaces
│   └── api.ts                       # API response types
│
└── utils/
    ├── format.ts                    # Formatting helpers (bytes, params)
    └── validation.ts                # Form validation
```

### File Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Python modules | snake_case | `model_service.py` |
| Python classes | PascalCase | `ModelService` |
| React components | PascalCase | `ModelCard.tsx` |
| TypeScript modules | camelCase | `modelStore.ts` |
| TypeScript interfaces | PascalCase, I-prefix optional | `Model`, `ApiResponse` |
| Test files | `test_*.py` / `*.test.ts` | `test_model_service.py` |

### Import Organization

**Python:**
```python
# 1. Standard library
import asyncio
from datetime import datetime
from typing import Optional, List

# 2. Third-party
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

# 3. Local application
from millm.core.errors import ModelNotFoundError
from millm.db.repositories import ModelRepository
from millm.api.schemas import ModelResponse
```

**TypeScript:**
```typescript
// 1. React/External libraries
import React, { useState, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';

// 2. Internal services/stores
import { useModelStore } from '@/stores/modelStore';
import { modelService } from '@/services/modelService';

// 3. Components
import { Button, Card } from '@/components/common';

// 4. Types
import type { Model, DownloadRequest } from '@/types';
```

---

## 3. Component Implementation Hints

### Backend Service Layer

**ModelService Pattern:**
```python
# millm/services/model_service.py

class ModelService:
    """
    Thin orchestration layer. Delegates to specialized components.
    No complex business logic - just coordination.
    """

    def __init__(
        self,
        repository: ModelRepository,
        loader: ModelLoader,
        downloader: ModelDownloader,
        sio: AsyncServer,
    ):
        # Store injected dependencies
        self.repository = repository
        self.loader = loader
        self.downloader = downloader
        self.sio = sio

        # Single shared executor for background tasks
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="millm-")

        # Track in-flight downloads for cancellation
        self._active_downloads: Dict[int, Future] = {}
```

**Key Method Patterns:**

```python
async def download_model(self, request: ModelDownloadRequest) -> Model:
    """
    Pattern: Validate → Create record → Submit background → Return immediately
    """
    # 1. Validate uniqueness
    existing = await self.repository.find_by_repo_quantization(...)
    if existing:
        raise ModelAlreadyExistsError(...)

    # 2. Create DB record in "downloading" state
    model = await self.repository.create(status=ModelStatus.DOWNLOADING, ...)

    # 3. Submit to executor (fire-and-forget)
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(
        self._executor,
        self._download_worker,
        model.id,
        request
    )
    self._active_downloads[model.id] = future

    # 4. Return immediately - client gets updates via WebSocket
    return model
```

### Backend Repository Layer

**Repository Pattern:**
```python
# millm/db/repositories/model_repository.py

class ModelRepository:
    """
    Simple CRUD operations. No business logic.
    Each method does one thing.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Model:
        model = Model(**kwargs)
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def get_by_id(self, model_id: int) -> Optional[Model]:
        return await self.session.get(Model, model_id)

    async def get_all(self) -> List[Model]:
        result = await self.session.execute(select(Model).order_by(Model.created_at.desc()))
        return result.scalars().all()

    async def update(self, model_id: int, **kwargs) -> Optional[Model]:
        model = await self.get_by_id(model_id)
        if model:
            for key, value in kwargs.items():
                setattr(model, key, value)
            model.updated_at = datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(model)
        return model

    async def delete(self, model_id: int) -> bool:
        model = await self.get_by_id(model_id)
        if model:
            await self.session.delete(model)
            await self.session.commit()
            return True
        return False
```

### Frontend Component Patterns

**Presentational Component:**
```tsx
// src/components/models/ModelCard.tsx

interface ModelCardProps {
  model: Model;
  onLoad: () => void;
  onUnload: () => void;
  onDelete: () => void;
}

export function ModelCard({ model, onLoad, onUnload, onDelete }: ModelCardProps) {
  // Pure display logic - no data fetching
  const isLoaded = model.status === 'loaded';
  const isDownloading = model.status === 'downloading';

  return (
    <Card>
      {/* Render based on props only */}
    </Card>
  );
}
```

**Container Component (Page):**
```tsx
// src/pages/ModelsPage.tsx

export function ModelsPage() {
  // Connect to store and services
  const { models, loadedModelId, downloadProgress } = useModelStore();
  const { loadModel, unloadModel, deleteModel } = useModelStore();

  // Fetch on mount
  useEffect(() => {
    useModelStore.getState().fetchModels();
  }, []);

  return (
    <div>
      <DownloadForm />
      <ModelList
        models={models}
        onLoad={loadModel}
        onUnload={unloadModel}
        onDelete={deleteModel}
      />
    </div>
  );
}
```

---

## 4. Database Implementation Approach

### SQLAlchemy Model Definition

```python
# millm/db/models/model.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Enum, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
import enum

from millm.db.base import Base


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

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Core fields
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[ModelSource] = mapped_column(Enum(ModelSource), nullable=False)
    repo_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    local_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Model metadata
    params: Mapped[str | None] = mapped_column(String(50))
    architecture: Mapped[str | None] = mapped_column(String(100))
    quantization: Mapped[QuantizationType] = mapped_column(Enum(QuantizationType), nullable=False)

    # Storage
    disk_size_mb: Mapped[int | None] = mapped_column(Integer)
    estimated_memory_mb: Mapped[int | None] = mapped_column(Integer)
    cache_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Configuration
    config_json: Mapped[dict | None] = mapped_column(JSONB)
    trust_remote_code: Mapped[bool] = mapped_column(Boolean, default=False)

    # State
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus), default=ModelStatus.READY
    )
    error_message: Mapped[str | None] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    loaded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint('repo_id', 'quantization', name='uq_repo_quantization'),
        UniqueConstraint('local_path', name='uq_local_path'),
    )
```

### Migration Strategy

```python
# millm/db/migrations/versions/001_create_models_table.py

"""Create models table

Revision ID: 001
Create Date: 2026-01-30
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


def upgrade():
    op.create_table(
        'models',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('source', sa.Enum('huggingface', 'local', name='modelsource'), nullable=False),
        sa.Column('repo_id', sa.String(255), nullable=True),
        sa.Column('local_path', sa.String(500), nullable=True),
        sa.Column('params', sa.String(50), nullable=True),
        sa.Column('architecture', sa.String(100), nullable=True),
        sa.Column('quantization', sa.Enum('Q4', 'Q8', 'FP16', name='quantizationtype'), nullable=False),
        sa.Column('disk_size_mb', sa.Integer(), nullable=True),
        sa.Column('estimated_memory_mb', sa.Integer(), nullable=True),
        sa.Column('cache_path', sa.String(500), nullable=False),
        sa.Column('config_json', JSONB(), nullable=True),
        sa.Column('trust_remote_code', sa.Boolean(), default=False),
        sa.Column('status', sa.Enum('downloading', 'ready', 'loading', 'loaded', 'error', name='modelstatus'), default='ready'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('loaded_at', sa.DateTime(), nullable=True),
        sa.UniqueConstraint('repo_id', 'quantization', name='uq_repo_quantization'),
        sa.UniqueConstraint('local_path', name='uq_local_path'),
    )

    op.create_index('idx_models_status', 'models', ['status'])
    op.create_index('idx_models_repo_id', 'models', ['repo_id'])


def downgrade():
    op.drop_index('idx_models_repo_id')
    op.drop_index('idx_models_status')
    op.drop_table('models')
    op.execute('DROP TYPE IF EXISTS modelsource')
    op.execute('DROP TYPE IF EXISTS quantizationtype')
    op.execute('DROP TYPE IF EXISTS modelstatus')
```

### Database Session Management

```python
# millm/db/base.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from millm.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """Dependency for FastAPI routes."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
```

---

## 5. API Implementation Strategy

### Route Implementation

```python
# millm/api/routes/management/models.py

from fastapi import APIRouter, Depends, HTTPException
from typing import List

from millm.api.dependencies import get_model_service
from millm.api.schemas.common import ApiResponse
from millm.api.schemas.model import (
    ModelResponse,
    ModelDownloadRequest,
    ModelPreviewRequest,
    ModelPreviewResponse,
)
from millm.services.model_service import ModelService
from millm.core.errors import MiLLMError

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ApiResponse[List[ModelResponse]])
async def list_models(
    service: ModelService = Depends(get_model_service)
):
    """List all downloaded models."""
    models = await service.list_models()
    return ApiResponse.ok([ModelResponse.from_orm(m) for m in models])


@router.post("", response_model=ApiResponse[ModelResponse], status_code=202)
async def download_model(
    request: ModelDownloadRequest,
    service: ModelService = Depends(get_model_service)
):
    """
    Start model download. Returns immediately with status: downloading.
    Progress updates sent via WebSocket.
    """
    model = await service.download_model(request)
    return ApiResponse.ok(ModelResponse.from_orm(model))


@router.get("/{model_id}", response_model=ApiResponse[ModelResponse])
async def get_model(
    model_id: int,
    service: ModelService = Depends(get_model_service)
):
    """Get single model details."""
    model = await service.get_model(model_id)
    return ApiResponse.ok(ModelResponse.from_orm(model))


@router.delete("/{model_id}", response_model=ApiResponse[None])
async def delete_model(
    model_id: int,
    service: ModelService = Depends(get_model_service)
):
    """Delete model from disk and database."""
    await service.delete_model(model_id)
    return ApiResponse.ok(None)


@router.post("/{model_id}/load", response_model=ApiResponse[ModelResponse], status_code=202)
async def load_model(
    model_id: int,
    service: ModelService = Depends(get_model_service)
):
    """Load model into GPU memory."""
    model = await service.load_model(model_id)
    return ApiResponse.ok(ModelResponse.from_orm(model))


@router.post("/{model_id}/unload", response_model=ApiResponse[ModelResponse])
async def unload_model(
    model_id: int,
    service: ModelService = Depends(get_model_service)
):
    """Unload model from GPU memory."""
    model = await service.unload_model(model_id)
    return ApiResponse.ok(ModelResponse.from_orm(model))


@router.post("/{model_id}/cancel", response_model=ApiResponse[None])
async def cancel_download(
    model_id: int,
    service: ModelService = Depends(get_model_service)
):
    """Cancel in-progress download."""
    await service.cancel_download(model_id)
    return ApiResponse.ok(None)


@router.post("/preview", response_model=ApiResponse[ModelPreviewResponse])
async def preview_model(
    request: ModelPreviewRequest,
    service: ModelService = Depends(get_model_service)
):
    """Get model info from HuggingFace without downloading."""
    preview = await service.preview_model(request)
    return ApiResponse.ok(preview)
```

### Schema Definitions

```python
# millm/api/schemas/model.py

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class QuantizationType(str, Enum):
    Q4 = "Q4"
    Q8 = "Q8"
    FP16 = "FP16"


class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ModelStatus(str, Enum):
    DOWNLOADING = "downloading"
    READY = "ready"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class ModelDownloadRequest(BaseModel):
    source: ModelSource
    repo_id: Optional[str] = Field(
        None,
        pattern=r'^[\w-]+/[\w.-]+$',
        max_length=255,
        examples=["google/gemma-2-2b", "meta-llama/Llama-3.2-3B"]
    )
    local_path: Optional[str] = Field(None, max_length=500)
    quantization: QuantizationType = QuantizationType.Q4
    trust_remote_code: bool = False
    hf_token: Optional[str] = Field(None, exclude=True)  # Never log
    custom_name: Optional[str] = Field(None, max_length=100)

    @model_validator(mode='after')
    def validate_source_fields(self):
        if self.source == ModelSource.HUGGINGFACE and not self.repo_id:
            raise ValueError("repo_id required for HuggingFace source")
        if self.source == ModelSource.LOCAL and not self.local_path:
            raise ValueError("local_path required for local source")
        return self

    @field_validator('local_path')
    @classmethod
    def validate_local_path(cls, v):
        if v is None:
            return v
        from pathlib import Path
        path = Path(v)
        if not path.is_absolute():
            raise ValueError("local_path must be absolute")
        return str(path.resolve())


class ModelPreviewRequest(BaseModel):
    repo_id: str = Field(..., pattern=r'^[\w-]+/[\w.-]+$')
    hf_token: Optional[str] = Field(None, exclude=True)


class SizeEstimate(BaseModel):
    disk_mb: int
    memory_mb: int


class ModelPreviewResponse(BaseModel):
    name: str
    params: str
    architecture: str
    requires_trust_remote_code: bool
    is_gated: bool
    estimated_sizes: Dict[str, SizeEstimate]  # Q4, Q8, FP16


class ModelResponse(BaseModel):
    id: int
    name: str
    source: ModelSource
    repo_id: Optional[str]
    local_path: Optional[str]
    params: Optional[str]
    architecture: Optional[str]
    quantization: QuantizationType
    disk_size_mb: Optional[int]
    estimated_memory_mb: Optional[int]
    status: ModelStatus
    error_message: Optional[str]
    created_at: datetime
    loaded_at: Optional[datetime]

    class Config:
        from_attributes = True
```

### Error Handling

```python
# millm/core/errors.py

from typing import Dict, Any, Optional


class MiLLMError(Exception):
    """Base exception for all miLLM errors."""
    code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ModelNotFoundError(MiLLMError):
    code = "MODEL_NOT_FOUND"
    status_code = 404


class ModelAlreadyExistsError(MiLLMError):
    code = "MODEL_ALREADY_EXISTS"
    status_code = 409


class ModelLoadError(MiLLMError):
    code = "MODEL_LOAD_FAILED"
    status_code = 500


class InsufficientMemoryError(MiLLMError):
    code = "INSUFFICIENT_MEMORY"
    status_code = 507


class InsufficientDiskError(MiLLMError):
    code = "INSUFFICIENT_DISK"
    status_code = 507


class DownloadFailedError(MiLLMError):
    code = "DOWNLOAD_FAILED"
    status_code = 502


class RepoNotFoundError(MiLLMError):
    code = "REPO_NOT_FOUND"
    status_code = 404


class GatedModelError(MiLLMError):
    code = "GATED_MODEL_NO_TOKEN"
    status_code = 401


class InvalidTokenError(MiLLMError):
    code = "INVALID_HF_TOKEN"
    status_code = 401


class ModelNotLoadedError(MiLLMError):
    code = "MODEL_NOT_LOADED"
    status_code = 400


class ModelAlreadyLoadedError(MiLLMError):
    code = "MODEL_ALREADY_LOADED"
    status_code = 400


class InvalidLocalPathError(MiLLMError):
    code = "INVALID_LOCAL_PATH"
    status_code = 400
```

### Exception Handler

```python
# millm/api/exception_handlers.py

from fastapi import Request
from fastapi.responses import JSONResponse

from millm.api.schemas.common import ApiResponse
from millm.core.errors import MiLLMError


async def millm_error_handler(request: Request, exc: MiLLMError):
    """Convert MiLLMError to standard API response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse.fail(
            code=exc.code,
            message=exc.message,
            details=exc.details
        ).model_dump()
    )
```

---

## 6. Frontend Implementation Approach

### Zustand Store Implementation

```typescript
// src/stores/modelStore.ts

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { modelService } from '@/services/modelService';
import type { Model, ModelDownloadRequest, DownloadProgress } from '@/types';

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
  downloadModel: (request: ModelDownloadRequest) => Promise<Model>;
  loadModel: (id: number) => Promise<void>;
  unloadModel: (id: number) => Promise<void>;
  deleteModel: (id: number) => Promise<void>;
  cancelDownload: (id: number) => Promise<void>;

  // Socket event handlers
  handleDownloadProgress: (data: DownloadProgressEvent) => void;
  handleDownloadComplete: (data: DownloadCompleteEvent) => void;
  handleDownloadError: (data: DownloadErrorEvent) => void;
  handleLoadComplete: (data: LoadCompleteEvent) => void;

  // Internal
  clearError: () => void;
}

export const useModelStore = create<ModelState>()(
  immer((set, get) => ({
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
          set({
            models: response.data,
            loadedModelId: response.data.find(m => m.status === 'loaded')?.id ?? null,
            isLoading: false
          });
        } else {
          set({ error: response.error?.message, isLoading: false });
        }
      } catch (e) {
        set({ error: (e as Error).message, isLoading: false });
      }
    },

    downloadModel: async (request) => {
      set({ error: null });
      try {
        const response = await modelService.downloadModel(request);
        if (response.success) {
          set((state) => {
            state.models.push(response.data);
          });
          return response.data;
        } else {
          throw new Error(response.error?.message);
        }
      } catch (e) {
        set({ error: (e as Error).message });
        throw e;
      }
    },

    loadModel: async (id) => {
      set({ isLoading: true, error: null });
      try {
        const response = await modelService.loadModel(id);
        if (response.success) {
          // State will be updated via WebSocket event
          set((state) => {
            const model = state.models.find(m => m.id === id);
            if (model) model.status = 'loading';
          });
        } else {
          set({ error: response.error?.message, isLoading: false });
        }
      } catch (e) {
        set({ error: (e as Error).message, isLoading: false });
      }
    },

    unloadModel: async (id) => {
      set({ isLoading: true, error: null });
      try {
        const response = await modelService.unloadModel(id);
        if (response.success) {
          set((state) => {
            const model = state.models.find(m => m.id === id);
            if (model) {
              model.status = 'ready';
              model.loaded_at = undefined;
            }
            state.loadedModelId = null;
            state.isLoading = false;
          });
        } else {
          set({ error: response.error?.message, isLoading: false });
        }
      } catch (e) {
        set({ error: (e as Error).message, isLoading: false });
      }
    },

    deleteModel: async (id) => {
      try {
        const response = await modelService.deleteModel(id);
        if (response.success) {
          set((state) => {
            state.models = state.models.filter(m => m.id !== id);
            delete state.downloadProgress[id];
          });
        } else {
          set({ error: response.error?.message });
        }
      } catch (e) {
        set({ error: (e as Error).message });
      }
    },

    cancelDownload: async (id) => {
      try {
        await modelService.cancelDownload(id);
        set((state) => {
          state.models = state.models.filter(m => m.id !== id);
          delete state.downloadProgress[id];
        });
      } catch (e) {
        set({ error: (e as Error).message });
      }
    },

    // Socket event handlers
    handleDownloadProgress: (data) => {
      set((state) => {
        state.downloadProgress[data.model_id] = {
          progress: data.progress,
          bytesDownloaded: data.bytes_downloaded,
          totalBytes: data.total_bytes,
          speedMbps: data.speed_mbps,
        };
      });
    },

    handleDownloadComplete: (data) => {
      set((state) => {
        const model = state.models.find(m => m.id === data.model_id);
        if (model) {
          Object.assign(model, data.model);
        }
        delete state.downloadProgress[data.model_id];
      });
    },

    handleDownloadError: (data) => {
      set((state) => {
        const model = state.models.find(m => m.id === data.model_id);
        if (model) {
          model.status = 'error';
          model.error_message = data.error.message;
        }
        delete state.downloadProgress[data.model_id];
        state.error = data.error.message;
      });
    },

    handleLoadComplete: (data) => {
      set((state) => {
        // Unmark previous loaded model
        const prevLoaded = state.models.find(m => m.id === state.loadedModelId);
        if (prevLoaded) {
          prevLoaded.status = 'ready';
          prevLoaded.loaded_at = undefined;
        }

        // Mark new model as loaded
        const model = state.models.find(m => m.id === data.model_id);
        if (model) {
          model.status = 'loaded';
          model.loaded_at = new Date().toISOString();
        }

        state.loadedModelId = data.model_id;
        state.isLoading = false;
      });
    },

    clearError: () => set({ error: null }),
  }))
);
```

### API Service

```typescript
// src/services/modelService.ts

import { api } from './api';
import type {
  Model,
  ModelDownloadRequest,
  ModelPreviewRequest,
  ModelPreviewResponse,
  ApiResponse
} from '@/types';

class ModelService {
  async listModels(): Promise<ApiResponse<Model[]>> {
    const response = await api.get<ApiResponse<Model[]>>('/api/models');
    return response.data;
  }

  async getModel(id: number): Promise<ApiResponse<Model>> {
    const response = await api.get<ApiResponse<Model>>(`/api/models/${id}`);
    return response.data;
  }

  async downloadModel(request: ModelDownloadRequest): Promise<ApiResponse<Model>> {
    const response = await api.post<ApiResponse<Model>>('/api/models', request);
    return response.data;
  }

  async loadModel(id: number): Promise<ApiResponse<Model>> {
    const response = await api.post<ApiResponse<Model>>(`/api/models/${id}/load`);
    return response.data;
  }

  async unloadModel(id: number): Promise<ApiResponse<Model>> {
    const response = await api.post<ApiResponse<Model>>(`/api/models/${id}/unload`);
    return response.data;
  }

  async deleteModel(id: number): Promise<ApiResponse<null>> {
    const response = await api.delete<ApiResponse<null>>(`/api/models/${id}`);
    return response.data;
  }

  async cancelDownload(id: number): Promise<ApiResponse<null>> {
    const response = await api.post<ApiResponse<null>>(`/api/models/${id}/cancel`);
    return response.data;
  }

  async previewModel(request: ModelPreviewRequest): Promise<ApiResponse<ModelPreviewResponse>> {
    const response = await api.post<ApiResponse<ModelPreviewResponse>>('/api/models/preview', request);
    return response.data;
  }
}

export const modelService = new ModelService();
```

### Socket.IO Client

```typescript
// src/services/socketService.ts

import { io, Socket } from 'socket.io-client';
import { useModelStore } from '@/stores/modelStore';

class SocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect(): void {
    if (this.socket?.connected) return;

    this.socket = io('/progress', {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });

    this.socket.on('connect', () => {
      console.log('Socket.IO connected');
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket.IO disconnected:', reason);
    });

    this.socket.on('connect_error', (error) => {
      console.error('Socket.IO connection error:', error);
      this.reconnectAttempts++;
    });

    // Model events
    this.socket.on('model:download:progress', (data) => {
      useModelStore.getState().handleDownloadProgress(data);
    });

    this.socket.on('model:download:complete', (data) => {
      useModelStore.getState().handleDownloadComplete(data);
    });

    this.socket.on('model:download:error', (data) => {
      useModelStore.getState().handleDownloadError(data);
    });

    this.socket.on('model:load:progress', (data) => {
      // Optional: could add load progress handling
    });

    this.socket.on('model:load:complete', (data) => {
      useModelStore.getState().handleLoadComplete(data);
    });
  }

  disconnect(): void {
    this.socket?.disconnect();
    this.socket = null;
  }

  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }
}

export const socketService = new SocketService();
```

### UI Components

```tsx
// src/components/models/DownloadForm.tsx

import React, { useState } from 'react';
import { Button, Input, Select, Checkbox } from '@/components/common';
import { useModelStore } from '@/stores/modelStore';
import type { QuantizationType, ModelSource } from '@/types';

export function DownloadForm() {
  const [repoId, setRepoId] = useState('');
  const [quantization, setQuantization] = useState<QuantizationType>('Q4');
  const [hfToken, setHfToken] = useState('');
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const downloadModel = useModelStore(state => state.downloadModel);
  const error = useModelStore(state => state.error);
  const clearError = useModelStore(state => state.clearError);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    setIsSubmitting(true);

    try {
      await downloadModel({
        source: 'huggingface' as ModelSource,
        repo_id: repoId,
        quantization,
        trust_remote_code: trustRemoteCode,
        hf_token: hfToken || undefined,
      });

      // Clear form on success
      setRepoId('');
      setHfToken('');
      setTrustRemoteCode(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  const quantizationOptions = [
    { value: 'Q4', label: 'Q4 (4-bit, smallest, recommended)' },
    { value: 'Q8', label: 'Q8 (8-bit, balanced)' },
    { value: 'FP16', label: 'FP16 (full precision, largest)' },
  ];

  return (
    <form onSubmit={handleSubmit} className="space-y-4 p-4 bg-gray-900 rounded-lg">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">
          HuggingFace Repository
        </label>
        <Input
          value={repoId}
          onChange={(e) => setRepoId(e.target.value)}
          placeholder="e.g., google/gemma-2-2b"
          required
          pattern="[\w-]+/[\w.-]+"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">
          Quantization
        </label>
        <Select
          value={quantization}
          onChange={(value) => setQuantization(value as QuantizationType)}
          options={quantizationOptions}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">
          Access Token (for gated models)
        </label>
        <Input
          type="password"
          value={hfToken}
          onChange={(e) => setHfToken(e.target.value)}
          placeholder="hf_..."
        />
      </div>

      <div className="flex items-center">
        <Checkbox
          checked={trustRemoteCode}
          onChange={setTrustRemoteCode}
          id="trust-remote-code"
        />
        <label htmlFor="trust-remote-code" className="ml-2 text-sm text-gray-300">
          Trust remote code (required for some models)
        </label>
      </div>

      {error && (
        <div className="text-red-400 text-sm p-2 bg-red-900/20 rounded">
          {error}
        </div>
      )}

      <div className="flex gap-2">
        <Button type="submit" disabled={isSubmitting || !repoId}>
          {isSubmitting ? 'Starting...' : 'Download'}
        </Button>
      </div>
    </form>
  );
}
```

```tsx
// src/components/models/ModelCard.tsx

import React from 'react';
import { Card, Badge, Button, ProgressBar } from '@/components/common';
import { MemoryEstimate } from './MemoryEstimate';
import type { Model, DownloadProgress } from '@/types';

interface ModelCardProps {
  model: Model;
  downloadProgress?: DownloadProgress;
  onLoad: () => void;
  onUnload: () => void;
  onDelete: () => void;
  onCancel: () => void;
}

export function ModelCard({
  model,
  downloadProgress,
  onLoad,
  onUnload,
  onDelete,
  onCancel,
}: ModelCardProps) {
  const isLoaded = model.status === 'loaded';
  const isDownloading = model.status === 'downloading';
  const isLoading = model.status === 'loading';
  const hasError = model.status === 'error';

  const statusColors: Record<string, string> = {
    loaded: 'bg-green-500',
    ready: 'bg-cyan-500',
    downloading: 'bg-yellow-500',
    loading: 'bg-yellow-500',
    error: 'bg-red-500',
  };

  return (
    <Card className="p-4">
      <div className="flex justify-between items-start mb-2">
        <div>
          <h3 className="font-bold text-white">{model.name}</h3>
          <p className="text-sm text-gray-400">
            {model.params} params • {model.quantization} •
            {model.estimated_memory_mb && ` ~${(model.estimated_memory_mb / 1024).toFixed(1)} GB`}
          </p>
          <p className="text-xs text-gray-500 font-mono">
            {model.repo_id || model.local_path}
          </p>
        </div>
        <Badge className={statusColors[model.status]}>
          {model.status.charAt(0).toUpperCase() + model.status.slice(1)}
        </Badge>
      </div>

      {isDownloading && downloadProgress && (
        <div className="my-3">
          <ProgressBar value={downloadProgress.progress} max={100} />
          <p className="text-xs text-gray-400 mt-1">
            {downloadProgress.progress.toFixed(1)}% •
            {(downloadProgress.speedMbps).toFixed(1)} MB/s
          </p>
        </div>
      )}

      {hasError && model.error_message && (
        <div className="my-2 p-2 bg-red-900/20 rounded text-sm text-red-400">
          {model.error_message}
        </div>
      )}

      <div className="flex gap-2 mt-3">
        {isLoaded && (
          <Button variant="outline" color="red" onClick={onUnload}>
            Unload
          </Button>
        )}

        {model.status === 'ready' && (
          <>
            <Button variant="primary" onClick={onLoad}>
              Load
            </Button>
            <Button variant="ghost" color="red" onClick={onDelete}>
              Delete
            </Button>
          </>
        )}

        {isDownloading && (
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        )}

        {isLoading && (
          <span className="text-sm text-gray-400">Loading...</span>
        )}
      </div>
    </Card>
  );
}
```

---

## 7. Business Logic Implementation Hints

### Model Downloader

```python
# millm/ml/model_downloader.py

import os
import shutil
from pathlib import Path
from typing import Callable, Optional
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
import structlog

from millm.core.config import settings
from millm.core.errors import RepoNotFoundError, GatedModelError, DownloadFailedError

logger = structlog.get_logger()


class ModelDownloader:
    """
    Downloads models from HuggingFace Hub.
    Uses huggingface_hub's snapshot_download for reliable downloads.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_api = HfApi()

    def download(
        self,
        repo_id: str,
        quantization: str,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> str:
        """
        Download model to cache directory.

        Args:
            repo_id: HuggingFace repo (e.g., "google/gemma-2-2b")
            quantization: Q4, Q8, or FP16
            progress_callback: Called with (progress_pct, downloaded_bytes, total_bytes)
            token: HuggingFace access token for gated models
            trust_remote_code: Whether model requires trust_remote_code

        Returns:
            Path to downloaded model directory
        """
        # Generate cache path: huggingface/owner--repo--quantization
        safe_name = repo_id.replace("/", "--") + f"--{quantization}"
        local_dir = self.cache_dir / "huggingface" / safe_name

        logger.info("download_started", repo_id=repo_id, local_dir=str(local_dir))

        try:
            # Download all model files
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                token=token,
                # Progress tracking happens via tqdm internally
                # For custom progress, we'd need to wrap or use alternative
            )

            logger.info("download_complete", repo_id=repo_id, local_dir=str(local_dir))
            return str(local_dir)

        except RepositoryNotFoundError:
            raise RepoNotFoundError(f"Repository '{repo_id}' not found on HuggingFace")
        except GatedRepoError:
            raise GatedModelError(f"Model '{repo_id}' is gated. Please provide access token.")
        except Exception as e:
            # Clean up partial download
            if local_dir.exists():
                shutil.rmtree(local_dir)
            raise DownloadFailedError(f"Download failed: {str(e)}")

    def get_model_info(self, repo_id: str, token: Optional[str] = None) -> dict:
        """Get model info without downloading."""
        try:
            info = self.hf_api.model_info(repo_id, token=token)
            return {
                "name": info.modelId.split("/")[-1],
                "params": self._extract_params(info),
                "architecture": getattr(info, "pipeline_tag", "unknown"),
                "is_gated": info.gated,
                "requires_trust_remote_code": self._check_trust_remote_code(info),
            }
        except RepositoryNotFoundError:
            raise RepoNotFoundError(f"Repository '{repo_id}' not found")

    def _extract_params(self, info) -> str:
        """Extract parameter count from model info."""
        # Try to get from model card or name
        safetensors = getattr(info, "safetensors", None)
        if safetensors and hasattr(safetensors, "total"):
            params = safetensors.total
            if params > 1e9:
                return f"{params/1e9:.1f}B"
            elif params > 1e6:
                return f"{params/1e6:.0f}M"
        return "unknown"

    def _check_trust_remote_code(self, info) -> bool:
        """Check if model requires trust_remote_code."""
        # Check config.json for custom code indicators
        files = [f.rfilename for f in info.siblings] if info.siblings else []
        return any(f.endswith("_utils.py") or f == "modeling.py" for f in files)
```

### Model Loader

```python
# millm/ml/model_loader.py

import gc
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import structlog

from millm.core.errors import ModelLoadError, InsufficientMemoryError
from millm.ml.memory_utils import estimate_memory_mb, get_available_memory_mb

logger = structlog.get_logger()


@dataclass
class LoadedModel:
    """Represents a model loaded in GPU memory."""
    model_id: int
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    loaded_at: datetime


class LoadedModelState:
    """
    Singleton managing the currently loaded model.
    Thread-safe for access from executor threads.
    """
    _instance: Optional['LoadedModelState'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loaded = None
        return cls._instance

    @property
    def current(self) -> Optional[LoadedModel]:
        return self._loaded

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def loaded_model_id(self) -> Optional[int]:
        return self._loaded.model_id if self._loaded else None

    def set(self, model: LoadedModel):
        with self._lock:
            self._loaded = model

    def clear(self):
        with self._lock:
            if self._loaded:
                del self._loaded.model
                del self._loaded.tokenizer
                self._loaded = None
            torch.cuda.empty_cache()
            gc.collect()


class ModelLoadContext:
    """
    Context manager for safe model loading.
    Ensures cleanup on any failure.
    """

    def __init__(self, model_id: int):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error("model_load_failed", model_id=self.model_id, error=str(exc_val))
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        return False  # Don't suppress exception

    def load(
        self,
        cache_path: str,
        quantization: str,
        trust_remote_code: bool = False,
        device: str = "cuda"
    ) -> LoadedModel:
        """Load model with quantization config."""
        logger.info("model_load_started",
            model_id=self.model_id,
            cache_path=cache_path,
            quantization=quantization
        )

        # Configure quantization
        quantization_config = None
        torch_dtype = torch.float16

        if quantization == "Q4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "Q8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer first (small)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cache_path,
            trust_remote_code=trust_remote_code,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )

        logger.info("model_load_complete", model_id=self.model_id)

        return LoadedModel(
            model_id=self.model_id,
            model=self.model,
            tokenizer=self.tokenizer,
            loaded_at=datetime.utcnow()
        )


class ModelLoader:
    """High-level model loading operations."""

    def __init__(self):
        self.state = LoadedModelState()

    def load(
        self,
        model_id: int,
        cache_path: str,
        quantization: str,
        estimated_memory_mb: int,
        trust_remote_code: bool = False,
    ) -> LoadedModel:
        """
        Load a model into GPU memory.
        Verifies memory availability before loading.
        """
        # Check memory
        available_mb = get_available_memory_mb()
        if available_mb < estimated_memory_mb:
            raise InsufficientMemoryError(
                f"Not enough GPU memory. Need ~{estimated_memory_mb}MB, have {available_mb}MB",
                details={
                    "required_mb": estimated_memory_mb,
                    "available_mb": available_mb,
                }
            )

        # Load with context manager for cleanup
        with ModelLoadContext(model_id) as ctx:
            loaded = ctx.load(
                cache_path=cache_path,
                quantization=quantization,
                trust_remote_code=trust_remote_code,
            )
            self.state.set(loaded)
            return loaded

    def unload(self):
        """Unload current model and free GPU memory."""
        self.state.clear()
```

### Memory Utilities

```python
# millm/ml/memory_utils.py

import re
from typing import Tuple
import torch


def parse_params(params_str: str) -> int:
    """
    Parse parameter string to number.
    Examples: "2.5B" -> 2_500_000_000, "350M" -> 350_000_000
    """
    if not params_str or params_str == "unknown":
        return 0

    match = re.match(r'^([\d.]+)([BMK]?)$', params_str.upper())
    if not match:
        return 0

    value = float(match.group(1))
    suffix = match.group(2)

    multipliers = {"B": 1e9, "M": 1e6, "K": 1e3, "": 1}
    return int(value * multipliers.get(suffix, 1))


def estimate_memory_mb(params_str: str, quantization: str) -> int:
    """
    Estimate VRAM needed for model.

    Formula:
    - FP16: params * 2 bytes
    - Q8: params * 1 byte
    - Q4: params * 0.5 bytes
    Plus ~20% overhead for KV cache, activations
    """
    params = parse_params(params_str)
    if params == 0:
        return 0

    bytes_per_param = {
        "FP16": 2.0,
        "Q8": 1.0,
        "Q4": 0.5,
    }

    base_bytes = params * bytes_per_param.get(quantization, 2.0)
    with_overhead = base_bytes * 1.2  # 20% overhead

    return int(with_overhead / (1024 * 1024))


def get_available_memory_mb() -> int:
    """Get available GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0

    free, total = torch.cuda.mem_get_info()
    return int(free / (1024 * 1024))


def get_total_memory_mb() -> int:
    """Get total GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0

    free, total = torch.cuda.mem_get_info()
    return int(total / (1024 * 1024))


def verify_memory_available(required_mb: int) -> Tuple[bool, int]:
    """
    Check if enough memory available.
    Returns (is_available, available_mb).
    """
    available = get_available_memory_mb()
    return (available >= required_mb, available)
```

---

## 8. Testing Implementation Approach

### Test Organization

```
tests/
├── conftest.py                      # Shared fixtures
├── unit/
│   ├── services/
│   │   └── test_model_service.py
│   ├── ml/
│   │   ├── test_model_loader.py
│   │   ├── test_model_downloader.py
│   │   └── test_memory_utils.py
│   └── db/
│       └── test_model_repository.py
├── integration/
│   ├── test_model_api.py
│   └── test_model_workflow.py
└── e2e/
    └── test_model_e2e.py
```

### Fixture Strategy

```python
# tests/conftest.py

import pytest
from unittest.mock import MagicMock, AsyncMock
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from fastapi.testclient import TestClient

from millm.db.base import Base
from millm.main import create_app


# Database fixtures
@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession)
    async with async_session() as session:
        yield session


@pytest.fixture
def mock_model_loader():
    """Mock ModelLoader for unit tests."""
    loader = MagicMock()
    loader.load.return_value = MagicMock(model_id=1)
    loader.unload.return_value = None
    loader.state.is_loaded = False
    loader.state.loaded_model_id = None
    return loader


@pytest.fixture
def mock_model_downloader():
    """Mock ModelDownloader for unit tests."""
    downloader = MagicMock()
    downloader.download.return_value = "/tmp/test-model"
    downloader.get_model_info.return_value = {
        "name": "test-model",
        "params": "2B",
        "architecture": "causal-lm",
        "is_gated": False,
        "requires_trust_remote_code": False,
    }
    return downloader


@pytest.fixture
def mock_sio():
    """Mock Socket.IO server."""
    sio = AsyncMock()
    return sio


@pytest.fixture
def mock_torch_cuda(mocker):
    """Mock CUDA operations for testing without GPU."""
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.mem_get_info', return_value=(8 * 1024**3, 24 * 1024**3))
    mocker.patch('torch.cuda.empty_cache')


@pytest.fixture
def client(test_db, mock_model_loader, mock_model_downloader, mock_sio):
    """FastAPI test client with mocked dependencies."""
    app = create_app()
    # Override dependencies
    app.dependency_overrides[...] = lambda: ...
    return TestClient(app)
```

### Unit Test Examples

```python
# tests/unit/services/test_model_service.py

import pytest
from unittest.mock import AsyncMock

from millm.services.model_service import ModelService
from millm.api.schemas.model import ModelDownloadRequest, ModelSource, QuantizationType
from millm.db.models.model import ModelStatus
from millm.core.errors import ModelAlreadyExistsError, ModelNotFoundError


class TestModelServiceDownload:
    @pytest.fixture
    def service(self, test_db, mock_model_loader, mock_model_downloader, mock_sio):
        repo = AsyncMock()
        repo.find_by_repo_quantization.return_value = None
        repo.create.return_value = MagicMock(id=1, status=ModelStatus.DOWNLOADING)

        return ModelService(
            repository=repo,
            loader=mock_model_loader,
            downloader=mock_model_downloader,
            sio=mock_sio,
        )

    async def test_download_creates_record_in_downloading_state(self, service):
        """Download should create DB record immediately in downloading state."""
        request = ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE,
            repo_id="google/gemma-2-2b",
            quantization=QuantizationType.Q4,
        )

        model = await service.download_model(request)

        assert model.status == ModelStatus.DOWNLOADING
        service.repository.create.assert_called_once()

    async def test_download_rejects_duplicate(self, service):
        """Download should reject if same repo+quantization exists."""
        service.repository.find_by_repo_quantization.return_value = MagicMock(id=1)

        request = ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE,
            repo_id="google/gemma-2-2b",
            quantization=QuantizationType.Q4,
        )

        with pytest.raises(ModelAlreadyExistsError):
            await service.download_model(request)


class TestModelServiceLoad:
    @pytest.fixture
    def service(self, test_db, mock_model_loader, mock_model_downloader, mock_sio):
        repo = AsyncMock()
        return ModelService(
            repository=repo,
            loader=mock_model_loader,
            downloader=mock_model_downloader,
            sio=mock_sio,
        )

    async def test_load_unloads_previous_model(self, service):
        """Loading new model should unload previous."""
        service._loaded_model = MagicMock(model_id=1)
        service.repository.get_by_id.return_value = MagicMock(
            id=2,
            status=ModelStatus.READY,
            cache_path="/tmp/model",
            quantization=QuantizationType.Q4,
        )

        await service.load_model(model_id=2)

        service.loader.unload.assert_called_once()

    async def test_load_not_found_raises(self, service):
        """Load should raise if model doesn't exist."""
        service.repository.get_by_id.return_value = None

        with pytest.raises(ModelNotFoundError):
            await service.load_model(model_id=999)
```

### Integration Test Examples

```python
# tests/integration/test_model_api.py

import pytest
from fastapi.testclient import TestClient


class TestModelAPI:
    def test_list_models_returns_empty_initially(self, client):
        """List models should return empty array when no models."""
        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []

    def test_download_model_returns_202(self, client):
        """Download should return 202 Accepted with downloading status."""
        response = client.post("/api/models", json={
            "source": "huggingface",
            "repo_id": "hf-internal-testing/tiny-random-gpt2",
            "quantization": "FP16",
        })

        assert response.status_code == 202
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "downloading"

    def test_get_model_not_found(self, client):
        """Get nonexistent model should return 404."""
        response = client.get("/api/models/999")

        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "MODEL_NOT_FOUND"
```

---

## 9. Configuration and Environment Strategy

### Settings Management

```python
# millm/core/config.py

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/millm"

    # Model cache
    MODEL_CACHE_DIR: str = "/data/models"

    # HuggingFace
    HF_TOKEN: Optional[str] = None

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "console"

    # Threading
    MAX_DOWNLOAD_WORKERS: int = 2

    # Timeouts
    GRACEFUL_UNLOAD_TIMEOUT: float = 30.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

### Environment Files

```bash
# .env.example

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/millm

# Model cache (ensure directory exists and has write permission)
MODEL_CACHE_DIR=/data/models

# HuggingFace token (optional, for gated models)
# HF_TOKEN=hf_xxxxxxxxxxxx

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=console
```

---

## 10. Integration Strategy

### Application Assembly

```python
# millm/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

from millm.api.routes import register_routes
from millm.api.exception_handlers import millm_error_handler
from millm.core.config import settings
from millm.core.errors import MiLLMError
from millm.core.logging import setup_logging
from millm.sockets.progress import create_socket_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    setup_logging()
    yield
    # Shutdown
    # Cleanup any loaded model
    from millm.ml.model_loader import LoadedModelState
    LoadedModelState().clear()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="miLLM",
        description="Mechanistic Interpretability LLM Server",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(MiLLMError, millm_error_handler)

    # Routes
    register_routes(app)

    return app


def create_combined_app():
    """Create combined FastAPI + Socket.IO app."""
    fastapi_app = create_app()
    sio = socketio.AsyncServer(
        async_mode='asgi',
        cors_allowed_origins='*',
    )
    combined = socketio.ASGIApp(sio, fastapi_app)

    # Store sio reference for dependency injection
    fastapi_app.state.sio = sio

    # Register socket handlers
    from millm.sockets.progress import register_handlers
    register_handlers(sio)

    return combined


app = create_combined_app()
```

### Dependency Injection

```python
# millm/api/dependencies.py

from functools import lru_cache
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.base import get_db
from millm.db.repositories.model_repository import ModelRepository
from millm.services.model_service import ModelService
from millm.ml.model_loader import ModelLoader
from millm.ml.model_downloader import ModelDownloader


@lru_cache()
def get_model_loader() -> ModelLoader:
    """Singleton model loader."""
    return ModelLoader()


@lru_cache()
def get_model_downloader() -> ModelDownloader:
    """Singleton model downloader."""
    return ModelDownloader()


async def get_model_repository(
    session: AsyncSession = Depends(get_db)
) -> ModelRepository:
    """Per-request repository."""
    return ModelRepository(session)


async def get_model_service(
    repository: ModelRepository = Depends(get_model_repository),
    loader: ModelLoader = Depends(get_model_loader),
    downloader: ModelDownloader = Depends(get_model_downloader),
) -> ModelService:
    """Per-request service with injected dependencies."""
    from fastapi import Request
    # Get sio from app state (need to pass request)
    # This is simplified - actual implementation needs request context
    sio = None  # Will be set from app.state
    return ModelService(
        repository=repository,
        loader=loader,
        downloader=downloader,
        sio=sio,
    )
```

---

## 11. Utilities and Helpers Design

### Formatting Utilities

```typescript
// src/utils/format.ts

/**
 * Format bytes to human readable string.
 * @example formatBytes(1536) => "1.5 KB"
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';

  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/**
 * Format memory in MB to human readable string.
 * @example formatMemory(2048) => "2.0 GB"
 */
export function formatMemory(mb: number): string {
  if (mb < 1024) return `${mb} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

/**
 * Format parameter count string.
 * @example formatParams("2500000000") => "2.5B"
 */
export function formatParams(params: string | number): string {
  const num = typeof params === 'string' ? parseInt(params, 10) : params;

  if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(0)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(0)}K`;
  return String(num);
}

/**
 * Format download speed.
 * @example formatSpeed(10.5) => "10.5 MB/s"
 */
export function formatSpeed(mbps: number): string {
  return `${mbps.toFixed(1)} MB/s`;
}
```

### Validation Utilities

```typescript
// src/utils/validation.ts

/**
 * Validate HuggingFace repository ID format.
 */
export function isValidRepoId(repoId: string): boolean {
  const pattern = /^[\w-]+\/[\w.-]+$/;
  return pattern.test(repoId);
}

/**
 * Validate local file path format.
 */
export function isValidLocalPath(path: string): boolean {
  // Must be absolute path
  return path.startsWith('/') || /^[A-Z]:\\/.test(path);
}

/**
 * Get validation error message for repo ID.
 */
export function getRepoIdError(repoId: string): string | null {
  if (!repoId) return 'Repository ID is required';
  if (!isValidRepoId(repoId)) {
    return 'Invalid format. Use owner/repo-name (e.g., google/gemma-2-2b)';
  }
  return null;
}
```

---

## 12. Error Handling and Logging Strategy

### Structured Logging Setup

```python
# millm/core/logging.py

import logging
import sys
import structlog

from millm.core.config import settings


def setup_logging():
    """Configure structured logging."""

    # Determine renderer based on settings
    if settings.LOG_FORMAT == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set log level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )
```

### Logging Patterns

```python
# Example logging in service layer

logger = structlog.get_logger()

async def download_model(self, request: ModelDownloadRequest) -> Model:
    logger.info("model_download_requested",
        repo_id=request.repo_id,
        quantization=request.quantization,
        trust_remote_code=request.trust_remote_code,
    )

    try:
        model = await self._do_download(request)
        logger.info("model_download_started",
            model_id=model.id,
            repo_id=request.repo_id,
        )
        return model
    except ModelAlreadyExistsError:
        logger.warning("model_download_duplicate",
            repo_id=request.repo_id,
            quantization=request.quantization,
        )
        raise
    except Exception as e:
        logger.error("model_download_failed",
            repo_id=request.repo_id,
            error=str(e),
            exc_info=True,
        )
        raise
```

---

## 13. Performance Implementation Hints

### Async Best Practices

```python
# Use async for all I/O operations

async def list_models(self) -> List[Model]:
    # Good: async database query
    return await self.repository.get_all()

def _download_worker(self, model_id: int, request: ModelDownloadRequest):
    # This runs in thread pool - use sync code here
    # Don't mix async/await in thread pool workers
    cache_path = self.downloader.download(
        repo_id=request.repo_id,
        quantization=request.quantization,
    )

    # To call async from thread, use asyncio.run()
    # But prefer structuring code to minimize this
```

### Memory Management

```python
# Always clean up GPU memory explicitly

def unload(self):
    """Unload model with explicit cleanup."""
    if self._loaded:
        # Delete model references
        del self._loaded.model
        del self._loaded.tokenizer
        self._loaded = None

        # Force GPU memory release
        torch.cuda.empty_cache()

        # Force Python garbage collection
        gc.collect()
```

### Database Query Optimization

```python
# Use indexes for common queries
# Already defined in migration:
# - idx_models_status
# - idx_models_repo_id

# For list queries, select only needed columns if performance is concern
async def get_all_summary(self) -> List[ModelSummary]:
    result = await self.session.execute(
        select(
            Model.id,
            Model.name,
            Model.status,
            Model.quantization,
            Model.estimated_memory_mb,
        )
    )
    return [ModelSummary(**row._asdict()) for row in result]
```

---

## 14. Code Quality and Standards

### Type Hints

```python
# Always use type hints for function signatures

from typing import Optional, List, Dict, Any

async def download_model(
    self,
    request: ModelDownloadRequest,
) -> Model:
    ...

def estimate_memory_mb(
    params_str: str,
    quantization: str,
) -> int:
    ...
```

### Docstrings

```python
# Use Google-style docstrings for public methods

def download(
    self,
    repo_id: str,
    quantization: str,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download model to cache directory.

    Args:
        repo_id: HuggingFace repository ID (e.g., "google/gemma-2-2b")
        quantization: Quantization level (Q4, Q8, or FP16)
        progress_callback: Optional callback for progress updates.
            Called with (progress_pct, downloaded_bytes, total_bytes)
        token: Optional HuggingFace access token for gated models

    Returns:
        Path to downloaded model directory

    Raises:
        RepoNotFoundError: If repository doesn't exist
        GatedModelError: If model is gated and no token provided
        DownloadFailedError: If download fails for other reasons
    """
    ...
```

### Code Organization

- One class per file for major components
- Group related functions in modules
- Keep files under 300 lines when possible
- Extract complex logic to helper functions

---

## Appendix A: File Creation Checklist

### Backend Files to Create

| File | Type | Priority |
|------|------|----------|
| `millm/__init__.py` | Package | P1 |
| `millm/main.py` | Entry point | P1 |
| `millm/core/__init__.py` | Package | P1 |
| `millm/core/config.py` | Settings | P1 |
| `millm/core/errors.py` | Exceptions | P1 |
| `millm/core/logging.py` | Logging setup | P1 |
| `millm/db/__init__.py` | Package | P1 |
| `millm/db/base.py` | SQLAlchemy setup | P1 |
| `millm/db/models/__init__.py` | Package | P1 |
| `millm/db/models/model.py` | ORM model | P1 |
| `millm/db/repositories/__init__.py` | Package | P1 |
| `millm/db/repositories/model_repository.py` | Repository | P1 |
| `millm/db/migrations/env.py` | Alembic env | P1 |
| `millm/db/migrations/versions/001_*.py` | Migration | P1 |
| `millm/api/__init__.py` | Package | P1 |
| `millm/api/dependencies.py` | DI | P1 |
| `millm/api/exception_handlers.py` | Error handling | P1 |
| `millm/api/routes/__init__.py` | Route registration | P1 |
| `millm/api/routes/management/__init__.py` | Package | P1 |
| `millm/api/routes/management/models.py` | Model endpoints | P1 |
| `millm/api/routes/system/__init__.py` | Package | P2 |
| `millm/api/routes/system/health.py` | Health check | P2 |
| `millm/api/schemas/__init__.py` | Package | P1 |
| `millm/api/schemas/common.py` | Common schemas | P1 |
| `millm/api/schemas/model.py` | Model schemas | P1 |
| `millm/services/__init__.py` | Package | P1 |
| `millm/services/model_service.py` | Service | P1 |
| `millm/ml/__init__.py` | Package | P1 |
| `millm/ml/model_loader.py` | Model loading | P1 |
| `millm/ml/model_downloader.py` | Downloads | P1 |
| `millm/ml/memory_utils.py` | Memory utils | P1 |
| `millm/sockets/__init__.py` | Package | P1 |
| `millm/sockets/progress.py` | Socket handlers | P1 |

### Frontend Files to Create

| File | Type | Priority |
|------|------|----------|
| `src/types/model.ts` | Types | P1 |
| `src/types/api.ts` | Types | P1 |
| `src/types/index.ts` | Re-exports | P1 |
| `src/services/api.ts` | Axios config | P1 |
| `src/services/modelService.ts` | API client | P1 |
| `src/services/socketService.ts` | Socket client | P1 |
| `src/stores/modelStore.ts` | Zustand store | P1 |
| `src/utils/format.ts` | Formatters | P2 |
| `src/utils/validation.ts` | Validators | P2 |
| `src/hooks/useModels.ts` | Custom hook | P2 |
| `src/components/common/Button.tsx` | Component | P1 |
| `src/components/common/Card.tsx` | Component | P1 |
| `src/components/common/Input.tsx` | Component | P1 |
| `src/components/common/Select.tsx` | Component | P1 |
| `src/components/common/Badge.tsx` | Component | P1 |
| `src/components/common/ProgressBar.tsx` | Component | P1 |
| `src/components/models/ModelCard.tsx` | Component | P1 |
| `src/components/models/ModelList.tsx` | Component | P1 |
| `src/components/models/DownloadForm.tsx` | Component | P1 |
| `src/components/models/MemoryEstimate.tsx` | Component | P2 |
| `src/components/models/ModelPreviewModal.tsx` | Component | P2 |
| `src/pages/ModelsPage.tsx` | Page | P1 |

---

**Document Status:** Complete
**Next Document:** `001_FTASKS|Model_Management.md` (Task List)
**Instruction File:** `@0xcc/instruct/006_generate-tasks.md`
