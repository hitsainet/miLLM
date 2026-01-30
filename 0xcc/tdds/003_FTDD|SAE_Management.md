# Technical Design Document: SAE Management

## miLLM Feature 3

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `003_FPRD|SAE_Management.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Executive Summary

SAE Management provides the infrastructure for downloading, caching, and attaching Sparse Autoencoders to loaded models. The design prioritizes safe attachment with compatibility validation, clean detachment with memory recovery, and a robust caching system for SAE files.

### Design Principles
1. **Safe by Default:** Validate compatibility before any attachment attempt
2. **Memory-Aware:** Track and report SAE memory usage accurately
3. **Clean Operations:** Attachment and detachment leave system in consistent state
4. **Future-Ready:** Architecture supports multi-SAE (v2.0) without redesign

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Hooking | PyTorch forward hooks | Native, well-supported, removable |
| Format | SAELens via sae-lens library | Standard format, active maintenance |
| Storage | SafeTensors | Fast loading, safe format |
| Caching | HuggingFace-style hub | Familiar pattern, proven reliability |
| Locking | Per-attachment lock | Thread-safe operations |

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Admin UI / Management API                   │
│                                                                  │
│    [SAE List]    [Download]    [Attach/Detach]    [Status]      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ REST + WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    SAE Routes                             │   │
│  │  /api/saes        /api/saes/download    /api/saes/attach │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    SAE Service                            │   │
│  │  - download_sae()    - attach_sae()    - detach_sae()    │   │
│  │  - list_saes()       - validate_compatibility()          │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                  │
│            ┌──────────────────┼──────────────────┐              │
│            ▼                  ▼                  ▼              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  SAE Downloader │ │   SAE Loader    │ │  SAE Hooker     │   │
│  │  - HF download  │ │  - Load weights │ │  - Install hook │   │
│  │  - Progress     │ │  - Parse config │ │  - Remove hook  │   │
│  │  - Caching      │ │  - Validate     │ │  - Apply SAE    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    SAE Repository                         │   │
│  │              Database persistence for SAE metadata        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Loaded Model (from Feature 1)                 │
│                                                                  │
│    Layer 0  →  Layer 1  →  ...  →  [Layer N with SAE Hook]     │
│                                          ↓                       │
│                                    ┌─────────┐                   │
│                                    │   SAE   │                   │
│                                    │ Encoder │                   │
│                                    │ Decoder │                   │
│                                    └─────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Relationships

1. **Routes → SAEService:** Routes validate requests, delegate to service
2. **SAEService → SAEDownloader:** Handles HuggingFace downloads
3. **SAEService → SAELoader:** Loads SAE weights and config
4. **SAEService → SAEHooker:** Manages model hooks
5. **SAEService → SAERepository:** Persists metadata to database
6. **SAEHooker → Model:** Installs/removes forward hooks

### Data Flow: SAE Attachment

```
Client requests POST /api/saes/{id}/attach
        │
        ▼
SAE Routes ────────────► Validate request
        │                 - SAE exists in cache
        │                 - Model is loaded
        │                 - Layer in valid range
        │
        ▼
SAE Service ───────────► Compatibility check
        │                 - Load SAE config
        │                 - Check d_in matches layer dim
        │                 - Estimate memory requirement
        │
        ▼
Memory Check ──────────► Verify VRAM available
        │                 - Get current GPU usage
        │                 - Add SAE memory estimate
        │                 - Warn if >90% utilization
        │
        ▼
SAE Loader ────────────► Load SAE weights
        │                 - Load encoder/decoder tensors
        │                 - Move to GPU
        │                 - Create SAE wrapper
        │
        ▼
SAE Hooker ────────────► Install forward hook
        │                 - Get target layer reference
        │                 - Register hook function
        │                 - Store hook handle
        │
        ▼
SAE Repository ────────► Persist attachment
        │                 - Update SAE status
        │                 - Create attachment record
        │
        ▼
Return success with memory usage
```

---

## 3. Technical Stack

### Dependencies

```python
# Core
fastapi>=0.109.0
pydantic>=2.0
sqlalchemy>=2.0

# SAE/ML
torch>=2.0
sae-lens>=2.0
safetensors>=0.4.0

# HuggingFace
huggingface_hub>=0.20.0

# Async
aiofiles>=23.0
```

### Technology Justification

| Technology | Purpose | Why |
|------------|---------|-----|
| sae-lens | SAE loading | Standard format, active development |
| safetensors | Weight storage | Fast, safe, memory-mapped |
| huggingface_hub | Downloads | Resume support, caching |
| PyTorch hooks | Model integration | Native, clean removal |

---

## 4. Data Design

### Database Schema

```python
# millm/db/models/sae.py

from sqlalchemy import Column, String, Integer, BigInteger, Boolean, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from datetime import datetime

from millm.db.base import Base


class SAE(Base):
    """SAE metadata and cache information."""
    __tablename__ = "saes"

    id = Column(String(50), primary_key=True)
    repository_id = Column(String(200), nullable=False)
    revision = Column(String(100), default="main")
    name = Column(String(200), nullable=False)
    format = Column(String(50), default="saelens")

    # Dimensions
    d_in = Column(Integer, nullable=False)
    d_sae = Column(Integer, nullable=False)

    # Training info
    trained_on = Column(String(200))
    trained_layer = Column(Integer)

    # Storage
    file_size_bytes = Column(BigInteger)
    cache_path = Column(String(500), nullable=False)

    # Status
    status = Column(String(20), default="cached")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    attachments = relationship("SAEAttachment", back_populates="sae")

    __table_args__ = (
        CheckConstraint(
            status.in_(["downloading", "cached", "attached", "error"]),
            name="sae_status_check"
        ),
    )


class SAEAttachment(Base):
    """Track SAE-model attachments."""
    __tablename__ = "sae_attachments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sae_id = Column(String(50), ForeignKey("saes.id"), nullable=False)
    model_id = Column(String(50), ForeignKey("models.id"), nullable=False)
    layer = Column(Integer, nullable=False)

    attached_at = Column(DateTime, default=datetime.utcnow)
    detached_at = Column(DateTime)
    memory_usage_mb = Column(Integer)
    is_active = Column(Boolean, default=True)

    # Relationships
    sae = relationship("SAE", back_populates="attachments")
```

### Pydantic Schemas

```python
# millm/api/schemas/sae.py

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class SAEMetadata(BaseModel):
    """SAE metadata for API responses."""
    id: str
    repository_id: str
    name: str
    format: Literal["saelens", "other"]
    d_in: int
    d_sae: int
    trained_on: Optional[str]
    trained_layer: Optional[int]
    file_size_mb: float
    status: Literal["downloading", "cached", "attached", "error"]
    created_at: datetime

    class Config:
        from_attributes = True


class DownloadSAERequest(BaseModel):
    """Request to download SAE from HuggingFace."""
    repository_id: str = Field(..., description="HuggingFace repo (e.g., jbloom/gemma-2-2b-res-jb)")
    revision: str = Field(default="main", description="Git revision")
    force_redownload: bool = Field(default=False, description="Re-download even if cached")


class AttachSAERequest(BaseModel):
    """Request to attach SAE to model."""
    layer: int = Field(..., ge=0, description="Target layer (0-indexed)")
    validate: bool = Field(default=True, description="Run compatibility check")


class AttachmentStatus(BaseModel):
    """Current SAE attachment status."""
    is_attached: bool
    sae_id: Optional[str]
    sae_name: Optional[str]
    layer: Optional[int]
    attached_at: Optional[datetime]
    memory_usage_mb: Optional[float]


class SAEListResponse(BaseModel):
    """List of SAEs with attachment status."""
    saes: list[SAEMetadata]
    attachment: AttachmentStatus


class CompatibilityResult(BaseModel):
    """Result of SAE-model compatibility check."""
    compatible: bool
    sae_d_in: int
    model_layer_dim: int
    layer: int
    warnings: list[str]
    errors: list[str]
```

### SAE Configuration Schema (SAELens)

```python
# millm/ml/sae_config.py

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class SAEConfig:
    """
    SAE configuration from SAELens format.

    Parsed from cfg.json in SAE repository.
    """
    d_in: int                    # Input dimension (model hidden dim)
    d_sae: int                   # SAE hidden dimension (feature count)
    model_name: str              # Trained on model
    hook_name: str               # Layer hook point
    hook_layer: int              # Layer index
    dtype: str = "float32"       # Weight dtype
    device: str = "cuda"         # Default device

    # Optional metadata
    neuronpedia_id: Optional[str] = None
    normalize_activations: str = "none"

    @classmethod
    def from_json(cls, path: str) -> "SAEConfig":
        """Load config from SAELens cfg.json."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            d_in=data["d_in"],
            d_sae=data["d_sae"],
            model_name=data.get("model_name", "unknown"),
            hook_name=data.get("hook_name", ""),
            hook_layer=data.get("hook_layer", 0),
            dtype=data.get("dtype", "float32"),
            neuronpedia_id=data.get("neuronpedia_id"),
            normalize_activations=data.get("normalize_activations", "none"),
        )

    def estimate_memory_mb(self) -> float:
        """
        Estimate SAE memory usage in MB.

        SAE has encoder (d_in × d_sae) and decoder (d_sae × d_in) matrices.
        Plus biases: encoder_bias (d_sae), decoder_bias (d_in).
        """
        bytes_per_param = 4 if self.dtype == "float32" else 2

        encoder_params = self.d_in * self.d_sae + self.d_sae  # W_enc + b_enc
        decoder_params = self.d_sae * self.d_in + self.d_in   # W_dec + b_dec

        total_bytes = (encoder_params + decoder_params) * bytes_per_param
        return total_bytes / (1024 * 1024)
```

---

## 5. API Design

### Route Structure

```python
# millm/api/routes/management/saes.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Optional

from millm.api.schemas.sae import (
    SAEMetadata,
    SAEListResponse,
    DownloadSAERequest,
    AttachSAERequest,
    AttachmentStatus,
    CompatibilityResult,
)
from millm.services.sae_service import SAEService
from millm.api.dependencies import get_sae_service

router = APIRouter(prefix="/api/saes", tags=["SAE Management"])


@router.get("", response_model=SAEListResponse)
async def list_saes(
    sae_service: SAEService = Depends(get_sae_service),
):
    """List all cached SAEs with current attachment status."""
    return await sae_service.list_saes()


@router.get("/attachment", response_model=AttachmentStatus)
async def get_attachment_status(
    sae_service: SAEService = Depends(get_sae_service),
):
    """Get current SAE attachment status."""
    return await sae_service.get_attachment_status()


@router.post("/download")
async def download_sae(
    request: DownloadSAERequest,
    background_tasks: BackgroundTasks,
    sae_service: SAEService = Depends(get_sae_service),
):
    """
    Start SAE download from HuggingFace.

    Returns immediately with progress tracking key.
    Download continues in background.
    """
    sae_id = await sae_service.start_download(
        repository_id=request.repository_id,
        revision=request.revision,
        force=request.force_redownload,
        background_tasks=background_tasks,
    )

    return {
        "status": "downloading",
        "sae_id": sae_id,
        "progress_key": f"sae-download-{sae_id}"
    }


@router.get("/{sae_id}", response_model=SAEMetadata)
async def get_sae(
    sae_id: str,
    sae_service: SAEService = Depends(get_sae_service),
):
    """Get SAE metadata by ID."""
    sae = await sae_service.get_sae(sae_id)
    if not sae:
        raise HTTPException(status_code=404, detail="SAE not found")
    return sae


@router.post("/{sae_id}/attach")
async def attach_sae(
    sae_id: str,
    request: AttachSAERequest,
    sae_service: SAEService = Depends(get_sae_service),
):
    """
    Attach SAE to loaded model at specified layer.

    Validates compatibility before attachment.
    """
    result = await sae_service.attach_sae(
        sae_id=sae_id,
        layer=request.layer,
        validate=request.validate,
    )
    return result


@router.post("/{sae_id}/detach")
async def detach_sae(
    sae_id: str,
    sae_service: SAEService = Depends(get_sae_service),
):
    """Detach SAE from model, freeing GPU memory."""
    result = await sae_service.detach_sae(sae_id)
    return result


@router.delete("/{sae_id}")
async def delete_sae(
    sae_id: str,
    sae_service: SAEService = Depends(get_sae_service),
):
    """Delete cached SAE. Must be detached first."""
    result = await sae_service.delete_sae(sae_id)
    return result


@router.get("/{sae_id}/compatibility")
async def check_compatibility(
    sae_id: str,
    layer: int,
    sae_service: SAEService = Depends(get_sae_service),
) -> CompatibilityResult:
    """Check SAE-model compatibility for given layer."""
    return await sae_service.check_compatibility(sae_id, layer)
```

---

## 6. Component Architecture

### Backend Structure

```
millm/
├── api/
│   ├── routes/
│   │   └── management/
│   │       └── saes.py              # SAE routes
│   └── schemas/
│       └── sae.py                   # Pydantic schemas
│
├── services/
│   └── sae_service.py               # Main SAE service
│
├── ml/
│   ├── sae_config.py                # SAE configuration
│   ├── sae_loader.py                # SAE weight loading
│   ├── sae_hooker.py                # Model hook management
│   └── sae_downloader.py            # HuggingFace download
│
└── db/
    ├── models/
    │   └── sae.py                   # SQLAlchemy models
    └── repositories/
        └── sae_repository.py        # Database operations
```

### SAE Service Design

```python
# millm/services/sae_service.py

from typing import Optional
import asyncio
import logging

from millm.ml.sae_downloader import SAEDownloader
from millm.ml.sae_loader import SAELoader
from millm.ml.sae_hooker import SAEHooker
from millm.db.repositories.sae_repository import SAERepository
from millm.services.model_service import ModelService
from millm.api.schemas.sae import (
    SAEListResponse,
    AttachmentStatus,
    SAEMetadata,
    CompatibilityResult,
)

logger = logging.getLogger(__name__)


class SAEService:
    """
    Service for SAE management operations.

    Thread-safety: Uses lock for attachment operations.
    """

    def __init__(
        self,
        repository: SAERepository,
        model_service: ModelService,
        cache_dir: str,
    ):
        self._repository = repository
        self._model_service = model_service
        self._downloader = SAEDownloader(cache_dir)
        self._loader = SAELoader()
        self._hooker = SAEHooker()
        self._attachment_lock = asyncio.Lock()

        # Current attachment state
        self._attached_sae_id: Optional[str] = None
        self._attached_layer: Optional[int] = None
        self._hook_handle: Optional[object] = None
        self._loaded_sae: Optional[object] = None

    async def list_saes(self) -> SAEListResponse:
        """List all cached SAEs with attachment status."""
        saes = await self._repository.get_all()
        attachment = await self.get_attachment_status()

        return SAEListResponse(
            saes=[self._to_metadata(s) for s in saes],
            attachment=attachment,
        )

    async def get_attachment_status(self) -> AttachmentStatus:
        """Get current SAE attachment status."""
        if not self._attached_sae_id:
            return AttachmentStatus(
                is_attached=False,
                sae_id=None,
                sae_name=None,
                layer=None,
                attached_at=None,
                memory_usage_mb=None,
            )

        sae = await self._repository.get(self._attached_sae_id)
        attachment = await self._repository.get_active_attachment()

        return AttachmentStatus(
            is_attached=True,
            sae_id=self._attached_sae_id,
            sae_name=sae.name if sae else None,
            layer=self._attached_layer,
            attached_at=attachment.attached_at if attachment else None,
            memory_usage_mb=attachment.memory_usage_mb if attachment else None,
        )

    async def start_download(
        self,
        repository_id: str,
        revision: str = "main",
        force: bool = False,
        background_tasks=None,
    ) -> str:
        """
        Start SAE download in background.

        Returns SAE ID for tracking.
        """
        # Generate ID from repo
        sae_id = self._generate_sae_id(repository_id, revision)

        # Check if already cached
        existing = await self._repository.get(sae_id)
        if existing and not force:
            if existing.status == "cached":
                logger.info(f"SAE {sae_id} already cached")
                return sae_id
            elif existing.status == "downloading":
                logger.info(f"SAE {sae_id} already downloading")
                return sae_id

        # Create initial record
        await self._repository.create_downloading(
            sae_id=sae_id,
            repository_id=repository_id,
            revision=revision,
        )

        # Start background download
        if background_tasks:
            background_tasks.add_task(
                self._download_task,
                sae_id,
                repository_id,
                revision,
            )

        return sae_id

    async def _download_task(
        self,
        sae_id: str,
        repository_id: str,
        revision: str,
    ):
        """Background download task."""
        try:
            # Download SAE files
            cache_path = await self._downloader.download(
                repository_id=repository_id,
                revision=revision,
                progress_callback=lambda p: self._emit_progress(sae_id, p),
            )

            # Load config to get metadata
            config = self._loader.load_config(cache_path)

            # Update database
            await self._repository.update_downloaded(
                sae_id=sae_id,
                cache_path=cache_path,
                config=config,
            )

            logger.info(f"SAE {sae_id} downloaded successfully")

        except Exception as e:
            logger.error(f"SAE download failed: {e}")
            await self._repository.update_status(sae_id, "error")
            raise

    async def attach_sae(
        self,
        sae_id: str,
        layer: int,
        validate: bool = True,
    ) -> dict:
        """
        Attach SAE to model at specified layer.

        Thread-safe via lock.
        """
        async with self._attachment_lock:
            # Check preconditions
            if not self._model_service.is_loaded():
                raise ValueError("No model loaded")

            sae = await self._repository.get(sae_id)
            if not sae:
                raise ValueError(f"SAE {sae_id} not found")

            if sae.status != "cached":
                raise ValueError(f"SAE is in {sae.status} state, cannot attach")

            if self._attached_sae_id:
                raise ValueError(
                    f"SAE {self._attached_sae_id} already attached. "
                    "Detach it first or use replace."
                )

            # Validate compatibility
            if validate:
                compat = await self.check_compatibility(sae_id, layer)
                if not compat.compatible:
                    raise ValueError(
                        f"SAE incompatible: {', '.join(compat.errors)}"
                    )

            # Load SAE
            loaded_sae = self._loader.load(sae.cache_path)
            memory_mb = loaded_sae.estimate_memory_mb()

            # Install hook
            model = self._model_service.get_model()
            self._hook_handle = self._hooker.install(
                model=model,
                layer=layer,
                sae=loaded_sae,
            )

            # Update state
            self._attached_sae_id = sae_id
            self._attached_layer = layer
            self._loaded_sae = loaded_sae

            # Persist
            await self._repository.update_status(sae_id, "attached")
            await self._repository.create_attachment(
                sae_id=sae_id,
                model_id=self._model_service.get_current_model().id,
                layer=layer,
                memory_usage_mb=int(memory_mb),
            )

            return {
                "status": "attached",
                "sae_id": sae_id,
                "layer": layer,
                "memory_usage_mb": memory_mb,
            }

    async def detach_sae(self, sae_id: str) -> dict:
        """Detach SAE from model, freeing memory."""
        async with self._attachment_lock:
            if self._attached_sae_id != sae_id:
                raise ValueError(f"SAE {sae_id} is not attached")

            # Remove hook
            if self._hook_handle:
                self._hooker.remove(self._hook_handle)
                self._hook_handle = None

            # Free memory
            memory_freed = 0
            if self._loaded_sae:
                memory_freed = self._loaded_sae.estimate_memory_mb()
                self._loaded_sae.to_cpu()  # Move to CPU first
                del self._loaded_sae
                self._loaded_sae = None
                import torch
                torch.cuda.empty_cache()

            # Update state
            old_id = self._attached_sae_id
            self._attached_sae_id = None
            self._attached_layer = None

            # Persist
            await self._repository.update_status(old_id, "cached")
            await self._repository.deactivate_attachment(old_id)

            return {
                "status": "detached",
                "memory_freed_mb": memory_freed,
            }

    async def check_compatibility(
        self,
        sae_id: str,
        layer: int,
    ) -> CompatibilityResult:
        """Check if SAE is compatible with model at given layer."""
        sae = await self._repository.get(sae_id)
        if not sae:
            return CompatibilityResult(
                compatible=False,
                sae_d_in=0,
                model_layer_dim=0,
                layer=layer,
                warnings=[],
                errors=["SAE not found"],
            )

        model = self._model_service.get_model()
        model_config = model.config

        # Get model's hidden dimension at layer
        # This varies by architecture
        model_dim = getattr(model_config, "hidden_size", None)
        if model_dim is None:
            model_dim = getattr(model_config, "d_model", 2048)

        # Check layer range
        num_layers = getattr(model_config, "num_hidden_layers", 26)
        errors = []
        warnings = []

        if layer < 0 or layer >= num_layers:
            errors.append(
                f"Layer {layer} out of range. Model has {num_layers} layers (0-{num_layers-1})"
            )

        # Check dimension match
        if sae.d_in != model_dim:
            errors.append(
                f"SAE input dimension ({sae.d_in}) does not match "
                f"model hidden dimension ({model_dim})"
            )

        # Check if layer matches trained layer
        if sae.trained_layer is not None and sae.trained_layer != layer:
            warnings.append(
                f"SAE was trained on layer {sae.trained_layer}, "
                f"attaching to layer {layer} may produce unexpected results"
            )

        return CompatibilityResult(
            compatible=len(errors) == 0,
            sae_d_in=sae.d_in,
            model_layer_dim=model_dim,
            layer=layer,
            warnings=warnings,
            errors=errors,
        )

    async def delete_sae(self, sae_id: str) -> dict:
        """Delete cached SAE. Must be detached first."""
        if self._attached_sae_id == sae_id:
            raise ValueError("Cannot delete attached SAE. Detach first.")

        sae = await self._repository.get(sae_id)
        if not sae:
            raise ValueError(f"SAE {sae_id} not found")

        # Delete files
        disk_freed = await self._downloader.delete(sae.cache_path)

        # Delete from database
        await self._repository.delete(sae_id)

        return {
            "status": "deleted",
            "disk_freed_mb": disk_freed,
        }

    def _generate_sae_id(self, repository_id: str, revision: str) -> str:
        """Generate unique SAE ID from repository."""
        import hashlib
        key = f"{repository_id}@{revision}"
        hash_part = hashlib.sha256(key.encode()).hexdigest()[:8]
        safe_name = repository_id.replace("/", "-")
        return f"sae-{safe_name}-{hash_part}"

    def _to_metadata(self, sae) -> SAEMetadata:
        """Convert database model to API schema."""
        return SAEMetadata(
            id=sae.id,
            repository_id=sae.repository_id,
            name=sae.name,
            format=sae.format,
            d_in=sae.d_in,
            d_sae=sae.d_sae,
            trained_on=sae.trained_on,
            trained_layer=sae.trained_layer,
            file_size_mb=sae.file_size_bytes / (1024 * 1024) if sae.file_size_bytes else 0,
            status=sae.status,
            created_at=sae.created_at,
        )

    def _emit_progress(self, sae_id: str, progress: dict):
        """Emit download progress via WebSocket."""
        # Implemented via ProgressService (shared with model downloads)
        pass
```

---

## 7. SAE Hooking Mechanism

### Hook Implementation

```python
# millm/ml/sae_hooker.py

from typing import Optional, Callable
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class SAEHooker:
    """
    Manages PyTorch forward hooks for SAE attachment.

    Thread-safety: Hook function must be thread-safe.
    The hook is called during every forward pass.
    """

    def install(
        self,
        model: nn.Module,
        layer: int,
        sae: "LoadedSAE",
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Install forward hook at specified layer.

        Returns handle for later removal.
        """
        # Get the target layer
        target_layer = self._get_layer(model, layer)

        # Create hook function
        def hook_fn(module, input, output):
            """
            Forward hook that applies SAE encoding/decoding.

            For steering: output = output + steering_vector
            For monitoring: capture activations
            """
            # Get activations (output of layer)
            activations = output[0] if isinstance(output, tuple) else output

            # Apply SAE
            # 1. Encode: activations → feature activations
            # 2. Apply steering (if any)
            # 3. Decode: feature activations → reconstructed
            reconstructed = sae.forward(activations)

            # Return modified output
            if isinstance(output, tuple):
                return (reconstructed,) + output[1:]
            return reconstructed

        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)

        logger.info(f"Installed SAE hook at layer {layer}")
        return handle

    def remove(self, handle: torch.utils.hooks.RemovableHandle):
        """Remove a previously installed hook."""
        handle.remove()
        logger.info("Removed SAE hook")

    def _get_layer(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """
        Get the layer module at specified index.

        Handles different model architectures.
        """
        # Try common attribute names
        for attr_name in ["model.layers", "layers", "transformer.h", "h"]:
            try:
                layers = self._get_nested_attr(model, attr_name)
                if isinstance(layers, nn.ModuleList) and len(layers) > layer_idx:
                    return layers[layer_idx]
            except AttributeError:
                continue

        # Fallback: try to find layers dynamically
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > layer_idx:
                return module[layer_idx]

        raise ValueError(
            f"Could not find layer {layer_idx} in model. "
            f"Model architecture may not be supported."
        )

    def _get_nested_attr(self, obj, attr_path: str):
        """Get nested attribute by dot-separated path."""
        parts = attr_path.split(".")
        for part in parts:
            obj = getattr(obj, part)
        return obj
```

### Loaded SAE Wrapper

```python
# millm/ml/sae_wrapper.py

from typing import Optional, Dict
import torch
from torch import nn, Tensor
import logging

logger = logging.getLogger(__name__)


class LoadedSAE:
    """
    Loaded SAE with encoder and decoder.

    Supports:
    - Basic forward pass (encode → decode)
    - Steering (modify feature activations)
    - Feature capture for monitoring
    """

    def __init__(
        self,
        W_enc: Tensor,      # (d_in, d_sae)
        b_enc: Tensor,      # (d_sae,)
        W_dec: Tensor,      # (d_sae, d_in)
        b_dec: Tensor,      # (d_in,)
        device: str = "cuda",
    ):
        self.W_enc = W_enc.to(device)
        self.b_enc = b_enc.to(device)
        self.W_dec = W_dec.to(device)
        self.b_dec = b_dec.to(device)
        self.device = device

        self.d_in = W_enc.shape[0]
        self.d_sae = W_enc.shape[1]

        # Steering configuration
        self._steering_values: Dict[int, float] = {}
        self._steering_enabled: bool = False

        # Monitoring
        self._last_feature_acts: Optional[Tensor] = None
        self._monitoring_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: encode, apply steering, decode.

        Args:
            x: Input activations (batch, seq_len, d_in)

        Returns:
            Reconstructed activations (batch, seq_len, d_in)
        """
        # Encode
        feature_acts = self.encode(x)

        # Capture for monitoring if enabled
        if self._monitoring_enabled:
            self._last_feature_acts = feature_acts.detach().clone()

        # Apply steering if enabled
        if self._steering_enabled and self._steering_values:
            feature_acts = self._apply_steering(feature_acts)

        # Decode
        reconstructed = self.decode(feature_acts)

        return reconstructed

    def encode(self, x: Tensor) -> Tensor:
        """Encode activations to feature space."""
        # x @ W_enc + b_enc with ReLU activation
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, feature_acts: Tensor) -> Tensor:
        """Decode feature activations to original space."""
        # feature_acts @ W_dec + b_dec
        return feature_acts @ self.W_dec + self.b_dec

    def set_steering(self, feature_idx: int, value: float):
        """Set steering value for a feature."""
        self._steering_values[feature_idx] = value

    def clear_steering(self, feature_idx: Optional[int] = None):
        """Clear steering for one or all features."""
        if feature_idx is None:
            self._steering_values.clear()
        elif feature_idx in self._steering_values:
            del self._steering_values[feature_idx]

    def enable_steering(self, enabled: bool = True):
        """Enable/disable steering."""
        self._steering_enabled = enabled

    def enable_monitoring(self, enabled: bool = True):
        """Enable/disable feature activation monitoring."""
        self._monitoring_enabled = enabled

    def get_last_feature_activations(self) -> Optional[Tensor]:
        """Get feature activations from last forward pass."""
        return self._last_feature_acts

    def _apply_steering(self, feature_acts: Tensor) -> Tensor:
        """Apply steering values to feature activations."""
        steered = feature_acts.clone()

        for idx, value in self._steering_values.items():
            # Add steering value to feature activation
            steered[..., idx] = steered[..., idx] + value

        return steered

    def estimate_memory_mb(self) -> float:
        """Estimate GPU memory usage in MB."""
        params = (
            self.W_enc.numel() +
            self.b_enc.numel() +
            self.W_dec.numel() +
            self.b_dec.numel()
        )
        bytes_per_param = 4 if self.W_enc.dtype == torch.float32 else 2
        return (params * bytes_per_param) / (1024 * 1024)

    def to_cpu(self):
        """Move all tensors to CPU."""
        self.W_enc = self.W_enc.cpu()
        self.b_enc = self.b_enc.cpu()
        self.W_dec = self.W_dec.cpu()
        self.b_dec = self.b_dec.cpu()
        self.device = "cpu"
```

---

## 8. SAE Downloader

```python
# millm/ml/sae_downloader.py

from typing import Optional, Callable
import os
import asyncio
from pathlib import Path
import logging

from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)


class SAEDownloader:
    """
    Download SAEs from HuggingFace Hub.

    Supports resume and progress tracking.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def download(
        self,
        repository_id: str,
        revision: str = "main",
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """
        Download SAE repository.

        Returns local cache path.
        """
        loop = asyncio.get_event_loop()

        # Run download in executor (blocking operation)
        cache_path = await loop.run_in_executor(
            None,
            self._download_sync,
            repository_id,
            revision,
            progress_callback,
        )

        return cache_path

    def _download_sync(
        self,
        repository_id: str,
        revision: str,
        progress_callback: Optional[Callable],
    ) -> str:
        """Synchronous download implementation."""
        try:
            # Download entire repository
            local_path = snapshot_download(
                repo_id=repository_id,
                revision=revision,
                cache_dir=str(self.cache_dir),
                resume_download=True,
            )

            logger.info(f"Downloaded SAE to {local_path}")
            return local_path

        except HfHubHTTPError as e:
            if "404" in str(e):
                raise ValueError(f"SAE repository not found: {repository_id}")
            raise

    async def delete(self, cache_path: str) -> float:
        """
        Delete cached SAE.

        Returns freed disk space in MB.
        """
        import shutil

        path = Path(cache_path)
        if not path.exists():
            return 0

        # Calculate size before deletion
        size_bytes = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

        # Delete
        shutil.rmtree(path)

        return size_bytes / (1024 * 1024)
```

---

## 9. Testing Strategy

### Unit Tests

```python
# tests/unit/ml/test_sae_wrapper.py

import pytest
import torch
from millm.ml.sae_wrapper import LoadedSAE


@pytest.fixture
def sample_sae():
    """Create small SAE for testing."""
    d_in, d_sae = 64, 128
    return LoadedSAE(
        W_enc=torch.randn(d_in, d_sae),
        b_enc=torch.zeros(d_sae),
        W_dec=torch.randn(d_sae, d_in),
        b_dec=torch.zeros(d_in),
        device="cpu",
    )


class TestLoadedSAE:
    def test_forward_pass_shape(self, sample_sae):
        """Forward pass preserves shape."""
        x = torch.randn(2, 10, 64)  # batch, seq, d_in
        out = sample_sae.forward(x)
        assert out.shape == x.shape

    def test_steering_modifies_output(self, sample_sae):
        """Steering changes output."""
        x = torch.randn(1, 5, 64)

        # Without steering
        sample_sae.enable_steering(False)
        out_normal = sample_sae.forward(x)

        # With steering
        sample_sae.enable_steering(True)
        sample_sae.set_steering(0, 5.0)
        out_steered = sample_sae.forward(x)

        assert not torch.allclose(out_normal, out_steered)

    def test_monitoring_captures_activations(self, sample_sae):
        """Monitoring captures feature activations."""
        x = torch.randn(1, 5, 64)

        sample_sae.enable_monitoring(True)
        sample_sae.forward(x)

        acts = sample_sae.get_last_feature_activations()
        assert acts is not None
        assert acts.shape == (1, 5, 128)  # batch, seq, d_sae
```

### Integration Tests

```python
# tests/integration/services/test_sae_service.py

class TestSAEService:
    async def test_download_and_attach_flow(self, sae_service, mock_model_service):
        """Full download → attach → detach flow."""
        # Download
        sae_id = await sae_service.start_download(
            repository_id="test/test-sae",
        )
        assert sae_id is not None

        # Wait for download (in tests, this is mocked to be instant)
        await asyncio.sleep(0.1)

        # Check compatibility
        compat = await sae_service.check_compatibility(sae_id, layer=12)
        assert compat.compatible

        # Attach
        result = await sae_service.attach_sae(sae_id, layer=12)
        assert result["status"] == "attached"

        # Verify status
        status = await sae_service.get_attachment_status()
        assert status.is_attached
        assert status.sae_id == sae_id

        # Detach
        result = await sae_service.detach_sae(sae_id)
        assert result["status"] == "detached"

        # Verify detached
        status = await sae_service.get_attachment_status()
        assert not status.is_attached
```

---

## 10. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SAE format variations | Medium | Medium | Start with SAELens, document requirements |
| Hook installation fails | Low | High | Comprehensive model architecture detection |
| Memory tracking inaccurate | Medium | Low | Conservative estimates, manual verification |
| Concurrent attachment issues | Low | High | Lock-based serialization |

### Mitigation Strategies

```python
# Safe hook installation with rollback
async def attach_sae_safe(self, sae_id: str, layer: int):
    """Attach with automatic rollback on failure."""
    loaded_sae = None
    hook_handle = None

    try:
        loaded_sae = self._loader.load(sae.cache_path)
        hook_handle = self._hooker.install(model, layer, loaded_sae)
        # ... update state
    except Exception as e:
        # Rollback
        if hook_handle:
            self._hooker.remove(hook_handle)
        if loaded_sae:
            del loaded_sae
            torch.cuda.empty_cache()
        raise
```

---

## 11. Development Phases

### Phase 1: Core Infrastructure (2-3 days)
- [ ] SAE database models and repository
- [ ] SAE schemas
- [ ] Basic routes structure

### Phase 2: Download System (2 days)
- [ ] SAEDownloader implementation
- [ ] Progress tracking
- [ ] Config parsing

### Phase 3: Attachment System (2-3 days)
- [ ] SAELoader implementation
- [ ] LoadedSAE wrapper
- [ ] SAEHooker implementation

### Phase 4: Service Layer (2 days)
- [ ] SAEService complete implementation
- [ ] Compatibility validation
- [ ] State management

### Phase 5: Testing & Polish (2 days)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Error handling refinement

---

**Document Status:** Complete
**Next Document:** `003_FTID|SAE_Management.md` (Technical Implementation Document)
**Instruction File:** `@0xcc/instruct/005_create-tid.md`
