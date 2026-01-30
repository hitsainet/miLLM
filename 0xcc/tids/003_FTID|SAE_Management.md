# Technical Implementation Document: SAE Management

## miLLM Feature 3

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `003_FPRD|SAE_Management.md`
- Feature TDD: `003_FTDD|SAE_Management.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Overview

This Technical Implementation Document provides specific implementation guidance for Feature 3: SAE Management. It translates the technical design into actionable coding patterns, file structures, and implementation hints.

### Implementation Philosophy
- **Safety First:** Always validate before modifying system state
- **Clean State Transitions:** Attachment/detachment leave system consistent
- **Memory Awareness:** Track GPU memory accurately throughout operations
- **Thread Safety:** Lock critical sections for concurrent access

---

## 2. File Structure

### Backend Organization

```
millm/
├── api/
│   ├── routes/
│   │   └── management/
│   │       └── saes.py                  # SAE management routes
│   └── schemas/
│       └── sae.py                       # SAE Pydantic schemas
│
├── services/
│   └── sae_service.py                   # Main SAE service
│
├── ml/
│   ├── sae_config.py                    # SAE configuration parsing
│   ├── sae_loader.py                    # SAE weight loading
│   ├── sae_wrapper.py                   # LoadedSAE wrapper class
│   ├── sae_hooker.py                    # Model hook management
│   └── sae_downloader.py                # HuggingFace download
│
├── db/
│   ├── models/
│   │   └── sae.py                       # SQLAlchemy models
│   └── repositories/
│       └── sae_repository.py            # Database operations
│
└── core/
    └── errors.py                        # Add SAE-specific errors
```

### Test Organization

```
tests/
├── unit/
│   ├── ml/
│   │   ├── test_sae_wrapper.py          # LoadedSAE tests
│   │   ├── test_sae_hooker.py           # Hook installation tests
│   │   ├── test_sae_loader.py           # SAE loading tests
│   │   └── test_sae_config.py           # Config parsing tests
│   └── services/
│       └── test_sae_service.py          # Service unit tests
│
├── integration/
│   └── services/
│       └── test_sae_service_integration.py  # Full flow tests
│
└── fixtures/
    └── sae/
        ├── sample_config.json           # Sample SAE config
        └── sample_weights.safetensors   # Small test weights
```

---

## 3. Database Models Implementation

### SQLAlchemy Models

```python
# millm/db/models/sae.py

"""
SAE database models.

Implementation notes:
1. SAE status uses CHECK constraint for valid values
2. Only one active attachment allowed via partial unique index
3. Cascade delete from SAE to attachments
"""

from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean,
    DateTime, ForeignKey, CheckConstraint, Index
)
from sqlalchemy.orm import relationship
from datetime import datetime

from millm.db.base import Base


class SAE(Base):
    """
    SAE metadata and cache information.

    Status flow:
    - downloading: Download in progress
    - cached: Downloaded and ready to attach
    - attached: Currently attached to model
    - error: Download or validation failed
    """
    __tablename__ = "saes"

    id = Column(String(50), primary_key=True)
    repository_id = Column(String(200), nullable=False, index=True)
    revision = Column(String(100), default="main")
    name = Column(String(200), nullable=False)
    format = Column(String(50), default="saelens")

    # Dimensions
    d_in = Column(Integer, nullable=False)
    d_sae = Column(Integer, nullable=False)

    # Training metadata
    trained_on = Column(String(200))
    trained_layer = Column(Integer)

    # Storage
    file_size_bytes = Column(BigInteger)
    cache_path = Column(String(500), nullable=False)

    # Status
    status = Column(String(20), default="cached", index=True)
    error_message = Column(String(500))  # Store error details
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    attachments = relationship(
        "SAEAttachment",
        back_populates="sae",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('downloading', 'cached', 'attached', 'error')",
            name="sae_status_check"
        ),
    )


class SAEAttachment(Base):
    """
    Track SAE-model attachments.

    Only one active attachment allowed at a time (v1.0).
    Partial unique index enforces this constraint.
    """
    __tablename__ = "sae_attachments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sae_id = Column(String(50), ForeignKey("saes.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String(50), ForeignKey("models.id"), nullable=False)
    layer = Column(Integer, nullable=False)

    attached_at = Column(DateTime, default=datetime.utcnow)
    detached_at = Column(DateTime)
    memory_usage_mb = Column(Integer)
    is_active = Column(Boolean, default=True)

    # Relationships
    sae = relationship("SAE", back_populates="attachments")

    __table_args__ = (
        # Only one active attachment allowed
        Index(
            "idx_single_active_attachment",
            "is_active",
            unique=True,
            postgresql_where=(is_active == True)
        ),
    )
```

### Repository Implementation

```python
# millm/db/repositories/sae_repository.py

"""
SAE repository for database operations.

Implementation notes:
1. All methods are async for consistency with rest of codebase
2. Use select() for reads, not session.query()
3. Always commit after writes
"""

from typing import Optional, List
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from millm.db.models.sae import SAE, SAEAttachment
from millm.ml.sae_config import SAEConfig


class SAERepository:
    """Repository for SAE database operations."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_all(self) -> List[SAE]:
        """Get all cached SAEs."""
        result = await self._session.execute(
            select(SAE).order_by(SAE.created_at.desc())
        )
        return result.scalars().all()

    async def get(self, sae_id: str) -> Optional[SAE]:
        """Get SAE by ID."""
        result = await self._session.execute(
            select(SAE).where(SAE.id == sae_id)
        )
        return result.scalar_one_or_none()

    async def get_by_repository(self, repository_id: str, revision: str = "main") -> Optional[SAE]:
        """Get SAE by repository and revision."""
        result = await self._session.execute(
            select(SAE).where(
                SAE.repository_id == repository_id,
                SAE.revision == revision
            )
        )
        return result.scalar_one_or_none()

    async def create_downloading(
        self,
        sae_id: str,
        repository_id: str,
        revision: str = "main",
    ) -> SAE:
        """Create SAE record in downloading state."""
        sae = SAE(
            id=sae_id,
            repository_id=repository_id,
            revision=revision,
            name=repository_id.split("/")[-1],  # Use repo name as initial name
            status="downloading",
            d_in=0,  # Updated after download
            d_sae=0,
            cache_path="",  # Updated after download
        )
        self._session.add(sae)
        await self._session.commit()
        return sae

    async def update_downloaded(
        self,
        sae_id: str,
        cache_path: str,
        config: SAEConfig,
        file_size_bytes: int = 0,
    ):
        """Update SAE after successful download."""
        await self._session.execute(
            update(SAE)
            .where(SAE.id == sae_id)
            .values(
                status="cached",
                cache_path=cache_path,
                d_in=config.d_in,
                d_sae=config.d_sae,
                trained_on=config.model_name,
                trained_layer=config.hook_layer,
                file_size_bytes=file_size_bytes,
                name=f"{config.model_name} Layer {config.hook_layer} SAE",
                updated_at=datetime.utcnow(),
            )
        )
        await self._session.commit()

    async def update_status(self, sae_id: str, status: str, error_message: Optional[str] = None):
        """Update SAE status."""
        values = {
            "status": status,
            "updated_at": datetime.utcnow(),
        }
        if error_message:
            values["error_message"] = error_message

        await self._session.execute(
            update(SAE)
            .where(SAE.id == sae_id)
            .values(**values)
        )
        await self._session.commit()

    async def delete(self, sae_id: str):
        """Delete SAE and all attachments."""
        await self._session.execute(
            delete(SAE).where(SAE.id == sae_id)
        )
        await self._session.commit()

    # Attachment operations

    async def get_active_attachment(self) -> Optional[SAEAttachment]:
        """Get currently active attachment."""
        result = await self._session.execute(
            select(SAEAttachment).where(SAEAttachment.is_active == True)
        )
        return result.scalar_one_or_none()

    async def create_attachment(
        self,
        sae_id: str,
        model_id: str,
        layer: int,
        memory_usage_mb: int,
    ) -> SAEAttachment:
        """Create new attachment record."""
        attachment = SAEAttachment(
            sae_id=sae_id,
            model_id=model_id,
            layer=layer,
            memory_usage_mb=memory_usage_mb,
            is_active=True,
        )
        self._session.add(attachment)
        await self._session.commit()
        return attachment

    async def deactivate_attachment(self, sae_id: str):
        """Mark attachment as inactive."""
        await self._session.execute(
            update(SAEAttachment)
            .where(
                SAEAttachment.sae_id == sae_id,
                SAEAttachment.is_active == True
            )
            .values(
                is_active=False,
                detached_at=datetime.utcnow(),
            )
        )
        await self._session.commit()
```

---

## 4. SAE Loading Implementation

### SAE Config Parser

```python
# millm/ml/sae_config.py

"""
SAE configuration parsing.

Supports SAELens format cfg.json files.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SAEConfig:
    """
    SAE configuration from SAELens format.

    Expected cfg.json structure:
    {
        "d_in": 2304,
        "d_sae": 16384,
        "model_name": "google/gemma-2-2b",
        "hook_name": "blocks.12.hook_resid_post",
        "hook_layer": 12,
        "dtype": "float32"
    }
    """
    d_in: int
    d_sae: int
    model_name: str
    hook_name: str
    hook_layer: int
    dtype: str = "float32"
    normalize_activations: str = "none"

    @classmethod
    def from_json(cls, path: str) -> "SAEConfig":
        """
        Load config from SAELens cfg.json.

        Handles variations in field naming.
        """
        config_path = Path(path) / "cfg.json"
        if not config_path.exists():
            # Try alternative locations
            for alt_name in ["config.json", "sae_cfg.json"]:
                alt_path = Path(path) / alt_name
                if alt_path.exists():
                    config_path = alt_path
                    break
            else:
                raise FileNotFoundError(
                    f"No configuration file found in {path}. "
                    "Expected cfg.json, config.json, or sae_cfg.json"
                )

        with open(config_path) as f:
            data = json.load(f)

        # Handle SAELens format variations
        d_in = data.get("d_in") or data.get("d_model") or data.get("input_dim")
        d_sae = data.get("d_sae") or data.get("d_hidden") or data.get("hidden_dim")

        if d_in is None or d_sae is None:
            raise ValueError(
                f"Config missing required dimensions. Found keys: {list(data.keys())}"
            )

        return cls(
            d_in=d_in,
            d_sae=d_sae,
            model_name=data.get("model_name", "unknown"),
            hook_name=data.get("hook_name", ""),
            hook_layer=data.get("hook_layer", 0),
            dtype=data.get("dtype", "float32"),
            normalize_activations=data.get("normalize_activations", "none"),
        )

    def estimate_memory_mb(self) -> float:
        """
        Estimate SAE memory usage.

        Memory = (encoder + decoder) * bytes_per_param
        encoder: d_in × d_sae + d_sae (bias)
        decoder: d_sae × d_in + d_in (bias)
        """
        bytes_per_param = 4 if self.dtype in ("float32", "fp32") else 2

        encoder_params = self.d_in * self.d_sae + self.d_sae
        decoder_params = self.d_sae * self.d_in + self.d_in

        total_bytes = (encoder_params + decoder_params) * bytes_per_param
        return total_bytes / (1024 * 1024)
```

### SAE Loader

```python
# millm/ml/sae_loader.py

"""
SAE weight loading.

Loads weights from SafeTensors files.
"""

from pathlib import Path
from typing import Optional
import logging

import torch
from safetensors.torch import load_file

from millm.ml.sae_config import SAEConfig
from millm.ml.sae_wrapper import LoadedSAE

logger = logging.getLogger(__name__)


class SAELoader:
    """
    Loads SAE weights from disk.

    Supports SAELens SafeTensors format.
    """

    def load_config(self, cache_path: str) -> SAEConfig:
        """Load SAE configuration from cache path."""
        return SAEConfig.from_json(cache_path)

    def load(
        self,
        cache_path: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> LoadedSAE:
        """
        Load SAE weights and create wrapper.

        Args:
            cache_path: Path to downloaded SAE directory
            device: Target device (cuda/cpu)
            dtype: Override weight dtype (None = use config)

        Returns:
            LoadedSAE wrapper ready for use
        """
        path = Path(cache_path)

        # Load config
        config = self.load_config(cache_path)

        # Find weights file
        weights_path = self._find_weights_file(path)

        logger.info(f"Loading SAE from {weights_path}")

        # Load weights
        state_dict = load_file(weights_path)

        # Extract tensors (handle naming variations)
        W_enc = self._get_tensor(state_dict, ["W_enc", "encoder.weight", "W_e"])
        b_enc = self._get_tensor(state_dict, ["b_enc", "encoder.bias", "b_e"])
        W_dec = self._get_tensor(state_dict, ["W_dec", "decoder.weight", "W_d"])
        b_dec = self._get_tensor(state_dict, ["b_dec", "decoder.bias", "b_d"])

        # Convert dtype if specified
        target_dtype = dtype or self._str_to_dtype(config.dtype)
        W_enc = W_enc.to(target_dtype)
        b_enc = b_enc.to(target_dtype)
        W_dec = W_dec.to(target_dtype)
        b_dec = b_dec.to(target_dtype)

        # Create wrapper
        loaded_sae = LoadedSAE(
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=b_dec,
            config=config,
            device=device,
        )

        logger.info(
            f"Loaded SAE: d_in={config.d_in}, d_sae={config.d_sae}, "
            f"memory={loaded_sae.estimate_memory_mb():.1f}MB"
        )

        return loaded_sae

    def _find_weights_file(self, path: Path) -> Path:
        """Find SAE weights file in directory."""
        # Check common names
        for name in ["sae_weights.safetensors", "model.safetensors", "weights.safetensors"]:
            weights_path = path / name
            if weights_path.exists():
                return weights_path

        # Look for any .safetensors file
        safetensors_files = list(path.glob("*.safetensors"))
        if safetensors_files:
            return safetensors_files[0]

        raise FileNotFoundError(
            f"No weights file found in {path}. "
            "Expected .safetensors file."
        )

    def _get_tensor(self, state_dict: dict, names: list) -> torch.Tensor:
        """Get tensor from state dict, trying multiple names."""
        for name in names:
            if name in state_dict:
                return state_dict[name]

        raise KeyError(
            f"Could not find tensor. Tried: {names}. "
            f"Available: {list(state_dict.keys())}"
        )

    def _str_to_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(dtype_str.lower(), torch.float32)
```

---

## 5. SAE Wrapper Implementation

```python
# millm/ml/sae_wrapper.py

"""
LoadedSAE wrapper for inference.

Handles encoding, decoding, steering, and monitoring.
"""

from typing import Optional, Dict, List
import torch
from torch import Tensor
import logging

from millm.ml.sae_config import SAEConfig

logger = logging.getLogger(__name__)


class LoadedSAE:
    """
    Loaded SAE with encoder and decoder weights.

    Thread-safety notes:
    - Forward pass is thread-safe (no mutation)
    - Steering modification should use lock if concurrent
    - Monitoring capture creates new tensor (safe)

    Memory layout:
    - W_enc: (d_in, d_sae) - encoder weights
    - b_enc: (d_sae,) - encoder bias
    - W_dec: (d_sae, d_in) - decoder weights
    - b_dec: (d_in,) - decoder bias
    """

    def __init__(
        self,
        W_enc: Tensor,
        b_enc: Tensor,
        W_dec: Tensor,
        b_dec: Tensor,
        config: SAEConfig,
        device: str = "cuda",
    ):
        self.W_enc = W_enc.to(device)
        self.b_enc = b_enc.to(device)
        self.W_dec = W_dec.to(device)
        self.b_dec = b_dec.to(device)
        self.config = config
        self.device = device

        self.d_in = W_enc.shape[0]
        self.d_sae = W_enc.shape[1]

        # Validate dimensions match config
        assert self.d_in == config.d_in, f"d_in mismatch: {self.d_in} vs {config.d_in}"
        assert self.d_sae == config.d_sae, f"d_sae mismatch: {self.d_sae} vs {config.d_sae}"

        # Steering state
        self._steering_values: Dict[int, float] = {}
        self._steering_enabled: bool = False

        # Pre-computed steering vector (optimization)
        self._steering_vector: Optional[Tensor] = None

        # Monitoring state
        self._monitoring_enabled: bool = False
        self._monitored_features: Optional[List[int]] = None
        self._last_feature_acts: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through SAE.

        Args:
            x: Input activations (batch, seq_len, d_in)

        Returns:
            Reconstructed activations (batch, seq_len, d_in)
        """
        # Encode: x @ W_enc + b_enc with ReLU
        feature_acts = torch.relu(x @ self.W_enc + self.b_enc)

        # Capture for monitoring
        if self._monitoring_enabled:
            self._capture_activations(feature_acts)

        # Apply steering
        if self._steering_enabled and self._steering_vector is not None:
            feature_acts = feature_acts + self._steering_vector

        # Decode: feature_acts @ W_dec + b_dec
        reconstructed = feature_acts @ self.W_dec + self.b_dec

        return reconstructed

    def encode(self, x: Tensor) -> Tensor:
        """Encode activations to feature space (for monitoring)."""
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, feature_acts: Tensor) -> Tensor:
        """Decode feature activations."""
        return feature_acts @ self.W_dec + self.b_dec

    # === Steering Methods ===

    def set_steering(self, feature_idx: int, value: float):
        """
        Set steering value for a feature.

        Args:
            feature_idx: Feature index (0 to d_sae-1)
            value: Steering strength (positive=amplify, negative=suppress)
        """
        if not 0 <= feature_idx < self.d_sae:
            raise ValueError(f"Feature index {feature_idx} out of range [0, {self.d_sae})")

        self._steering_values[feature_idx] = value
        self._rebuild_steering_vector()

    def set_steering_batch(self, steering: Dict[int, float]):
        """Set multiple steering values at once."""
        for idx, value in steering.items():
            if not 0 <= idx < self.d_sae:
                raise ValueError(f"Feature index {idx} out of range")
        self._steering_values.update(steering)
        self._rebuild_steering_vector()

    def clear_steering(self, feature_idx: Optional[int] = None):
        """Clear steering for one or all features."""
        if feature_idx is None:
            self._steering_values.clear()
        elif feature_idx in self._steering_values:
            del self._steering_values[feature_idx]
        self._rebuild_steering_vector()

    def get_steering_values(self) -> Dict[int, float]:
        """Get current steering values."""
        return dict(self._steering_values)

    def enable_steering(self, enabled: bool = True):
        """Enable or disable steering."""
        self._steering_enabled = enabled

    def _rebuild_steering_vector(self):
        """Rebuild pre-computed steering vector from values."""
        if not self._steering_values:
            self._steering_vector = None
            return

        # Create sparse steering vector
        vector = torch.zeros(self.d_sae, device=self.device, dtype=self.W_enc.dtype)
        for idx, value in self._steering_values.items():
            vector[idx] = value

        self._steering_vector = vector

    # === Monitoring Methods ===

    def enable_monitoring(self, enabled: bool = True, features: Optional[List[int]] = None):
        """
        Enable feature activation monitoring.

        Args:
            enabled: Whether to capture activations
            features: Specific features to monitor (None = all)
        """
        self._monitoring_enabled = enabled
        self._monitored_features = features

    def get_last_feature_activations(self) -> Optional[Tensor]:
        """Get feature activations from last forward pass."""
        return self._last_feature_acts

    def _capture_activations(self, feature_acts: Tensor):
        """Capture activations for monitoring."""
        if self._monitored_features is not None:
            # Only capture selected features
            self._last_feature_acts = feature_acts[..., self._monitored_features].detach().clone()
        else:
            # Capture all (may be memory intensive)
            self._last_feature_acts = feature_acts.detach().clone()

    # === Memory Management ===

    def estimate_memory_mb(self) -> float:
        """Estimate current GPU memory usage."""
        return self.config.estimate_memory_mb()

    def to_device(self, device: str):
        """Move all tensors to device."""
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_dec = self.b_dec.to(device)
        if self._steering_vector is not None:
            self._steering_vector = self._steering_vector.to(device)
        self.device = device

    def to_cpu(self):
        """Move to CPU (for cleanup)."""
        self.to_device("cpu")
```

---

## 6. SAE Hooker Implementation

```python
# millm/ml/sae_hooker.py

"""
Model hook management for SAE attachment.

Installs forward hooks to intercept and modify activations.
"""

from typing import Optional, Tuple, Callable
import torch
from torch import nn, Tensor
from torch.utils.hooks import RemovableHandle
import logging

from millm.ml.sae_wrapper import LoadedSAE

logger = logging.getLogger(__name__)


class SAEHooker:
    """
    Manages PyTorch forward hooks for SAE attachment.

    Hook function signature:
        hook(module, input, output) -> modified_output

    Thread safety: Hook functions are called during forward pass.
    Ensure SAE forward is thread-safe.
    """

    def install(
        self,
        model: nn.Module,
        layer: int,
        sae: LoadedSAE,
    ) -> RemovableHandle:
        """
        Install forward hook at specified layer.

        Args:
            model: The loaded model
            layer: Target layer index (0-indexed)
            sae: Loaded SAE to apply

        Returns:
            Hook handle for later removal
        """
        # Get target layer
        target_layer = self._get_layer(model, layer)

        # Create hook function
        hook_fn = self._create_hook_fn(sae)

        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)

        logger.info(f"Installed SAE hook at layer {layer}")
        return handle

    def remove(self, handle: RemovableHandle):
        """Remove a previously installed hook."""
        handle.remove()
        logger.info("Removed SAE hook")

    def _create_hook_fn(self, sae: LoadedSAE) -> Callable:
        """
        Create the hook function for SAE.

        The hook intercepts layer output, applies SAE, returns modified output.
        """
        def hook_fn(
            module: nn.Module,
            input: Tuple[Tensor, ...],
            output: Tensor,
        ) -> Tensor:
            """
            Forward hook that applies SAE.

            Handles different output formats:
            - Tuple (hidden_states, ...) - common in transformers
            - Single tensor
            """
            # Handle tuple output (common for transformer layers)
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Apply SAE
                modified = sae.forward(hidden_states)
                # Return with same structure
                return (modified,) + output[1:]
            else:
                # Single tensor output
                return sae.forward(output)

        return hook_fn

    def _get_layer(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """
        Get the layer module at specified index.

        Handles different transformer architectures.
        """
        # Architecture-specific layer access patterns
        layer_access_patterns = [
            # Gemma, Llama style
            lambda m: m.model.layers[layer_idx],
            # GPT-2 style
            lambda m: m.transformer.h[layer_idx],
            # Some other styles
            lambda m: m.layers[layer_idx],
            lambda m: m.encoder.layer[layer_idx],
        ]

        for accessor in layer_access_patterns:
            try:
                layer = accessor(model)
                logger.debug(f"Found layer via {accessor}")
                return layer
            except (AttributeError, IndexError, TypeError):
                continue

        # Fallback: search for ModuleList
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList):
                if len(module) > layer_idx:
                    logger.debug(f"Found layer via ModuleList: {name}")
                    return module[layer_idx]

        raise ValueError(
            f"Could not find layer {layer_idx}. "
            f"Model architecture may not be supported. "
            f"Try checking model.named_modules() to find layer structure."
        )

    def get_layer_count(self, model: nn.Module) -> int:
        """Get total number of layers in model."""
        # Try config first
        if hasattr(model, "config"):
            for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
                if hasattr(model.config, attr):
                    return getattr(model.config, attr)

        # Count layers
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return len(module)

        raise ValueError("Could not determine layer count")
```

---

## 7. SAE Downloader Implementation

```python
# millm/ml/sae_downloader.py

"""
Download SAEs from HuggingFace Hub.

Uses huggingface_hub for reliable downloading with resume support.
"""

from typing import Optional, Callable, Dict
from pathlib import Path
import asyncio
import shutil
import logging

from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


class SAEDownloader:
    """
    Download SAEs from HuggingFace Hub.

    Features:
    - Resume interrupted downloads
    - Progress tracking via callback
    - Local caching with configurable directory
    """

    def __init__(self, cache_dir: str):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory for SAE cache (e.g., ~/.cache/millm/saes)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api = HfApi()

    async def download(
        self,
        repository_id: str,
        revision: str = "main",
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ) -> str:
        """
        Download SAE repository.

        Args:
            repository_id: HuggingFace repo (e.g., "jbloom/gemma-2-2b-res-jb")
            revision: Git revision (branch, tag, commit)
            progress_callback: Called with progress updates

        Returns:
            Local cache path
        """
        loop = asyncio.get_event_loop()

        # Run in executor (HF download is blocking)
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
        logger.info(f"Downloading SAE from {repository_id}@{revision}")

        try:
            # Validate repository exists
            try:
                self._api.repo_info(repository_id, revision=revision)
            except RepositoryNotFoundError:
                raise ValueError(f"SAE repository not found: {repository_id}")

            # Download with huggingface_hub
            local_path = snapshot_download(
                repo_id=repository_id,
                revision=revision,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                # Only download essential files
                ignore_patterns=["*.md", "*.txt", ".git*"],
            )

            logger.info(f"SAE downloaded to {local_path}")

            # Emit completion
            if progress_callback:
                progress_callback({
                    "status": "complete",
                    "percent": 100,
                    "path": local_path,
                })

            return local_path

        except HfHubHTTPError as e:
            logger.error(f"Download failed: {e}")
            raise

    async def delete(self, cache_path: str) -> float:
        """
        Delete cached SAE directory.

        Args:
            cache_path: Path to SAE cache directory

        Returns:
            Freed disk space in MB
        """
        path = Path(cache_path)
        if not path.exists():
            return 0.0

        # Calculate size
        size_bytes = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

        # Delete directory
        shutil.rmtree(path)

        freed_mb = size_bytes / (1024 * 1024)
        logger.info(f"Deleted SAE cache: {cache_path} ({freed_mb:.1f}MB)")

        return freed_mb

    def get_cache_size(self) -> float:
        """Get total SAE cache size in MB."""
        if not self.cache_dir.exists():
            return 0.0

        size_bytes = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("**/*")
            if f.is_file()
        )
        return size_bytes / (1024 * 1024)
```

---

## 8. Error Handling

```python
# millm/core/errors.py (add to existing)

"""
SAE-specific error classes.
"""


class SAEError(MiLLMError):
    """Base class for SAE errors."""
    pass


class SAENotFoundError(SAEError):
    """SAE not found in cache."""
    def __init__(self, sae_id: str):
        super().__init__(
            code="SAE_NOT_FOUND",
            message=f"SAE '{sae_id}' not found"
        )


class SAEIncompatibleError(SAEError):
    """SAE incompatible with model."""
    def __init__(self, reason: str):
        super().__init__(
            code="SAE_INCOMPATIBLE",
            message=f"SAE incompatible: {reason}"
        )


class SAEAlreadyAttachedError(SAEError):
    """SAE already attached."""
    def __init__(self, sae_id: str):
        super().__init__(
            code="SAE_ALREADY_ATTACHED",
            message=f"SAE '{sae_id}' is already attached. Detach first."
        )


class SAENotAttachedError(SAEError):
    """SAE not attached."""
    def __init__(self, sae_id: str):
        super().__init__(
            code="SAE_NOT_ATTACHED",
            message=f"SAE '{sae_id}' is not attached"
        )


class SAEDownloadError(SAEError):
    """SAE download failed."""
    def __init__(self, repository_id: str, reason: str):
        super().__init__(
            code="SAE_DOWNLOAD_FAILED",
            message=f"Failed to download SAE '{repository_id}': {reason}"
        )
```

---

## 9. Testing Patterns

### Unit Test Examples

```python
# tests/unit/ml/test_sae_wrapper.py

import pytest
import torch
from millm.ml.sae_wrapper import LoadedSAE
from millm.ml.sae_config import SAEConfig


@pytest.fixture
def small_sae():
    """Create small SAE for testing."""
    d_in, d_sae = 64, 128
    config = SAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        model_name="test",
        hook_name="test",
        hook_layer=0,
    )
    return LoadedSAE(
        W_enc=torch.randn(d_in, d_sae),
        b_enc=torch.zeros(d_sae),
        W_dec=torch.randn(d_sae, d_in),
        b_dec=torch.zeros(d_in),
        config=config,
        device="cpu",
    )


class TestLoadedSAE:
    def test_forward_preserves_shape(self, small_sae):
        """Forward pass should preserve input shape."""
        x = torch.randn(2, 10, 64)  # batch=2, seq=10, d_in=64
        out = small_sae.forward(x)
        assert out.shape == x.shape

    def test_encode_produces_features(self, small_sae):
        """Encode should produce d_sae features."""
        x = torch.randn(1, 5, 64)
        features = small_sae.encode(x)
        assert features.shape == (1, 5, 128)

    def test_steering_modifies_output(self, small_sae):
        """Steering should modify output."""
        x = torch.randn(1, 5, 64)

        # Without steering
        small_sae.enable_steering(False)
        out_baseline = small_sae.forward(x.clone())

        # With steering
        small_sae.enable_steering(True)
        small_sae.set_steering(0, 10.0)
        out_steered = small_sae.forward(x.clone())

        assert not torch.allclose(out_baseline, out_steered)

    def test_clear_steering(self, small_sae):
        """Clearing steering should return to baseline."""
        x = torch.randn(1, 5, 64)

        small_sae.enable_steering(True)
        small_sae.set_steering(0, 10.0)
        out_steered = small_sae.forward(x.clone())

        small_sae.clear_steering()
        out_cleared = small_sae.forward(x.clone())

        # After clearing, should be same as no steering
        small_sae.enable_steering(False)
        out_baseline = small_sae.forward(x.clone())

        assert torch.allclose(out_cleared, out_baseline)

    def test_monitoring_captures_activations(self, small_sae):
        """Monitoring should capture feature activations."""
        x = torch.randn(1, 5, 64)

        small_sae.enable_monitoring(True)
        small_sae.forward(x)

        acts = small_sae.get_last_feature_activations()
        assert acts is not None
        assert acts.shape == (1, 5, 128)

    def test_monitoring_specific_features(self, small_sae):
        """Monitoring specific features only."""
        x = torch.randn(1, 5, 64)

        small_sae.enable_monitoring(True, features=[0, 1, 2])
        small_sae.forward(x)

        acts = small_sae.get_last_feature_activations()
        assert acts.shape == (1, 5, 3)  # Only 3 features

    def test_feature_index_validation(self, small_sae):
        """Should reject invalid feature indices."""
        with pytest.raises(ValueError):
            small_sae.set_steering(-1, 1.0)

        with pytest.raises(ValueError):
            small_sae.set_steering(128, 1.0)  # d_sae=128, max index is 127
```

### Integration Test Examples

```python
# tests/integration/services/test_sae_service_integration.py

@pytest.mark.asyncio
class TestSAEServiceIntegration:
    async def test_full_attachment_flow(
        self,
        sae_service,
        mock_model_service,
        test_sae_repo,
    ):
        """Test complete download → attach → detach flow."""
        # Download
        sae_id = await sae_service.start_download(
            repository_id=test_sae_repo,
        )

        # Wait for download (mocked to be instant in tests)
        await asyncio.sleep(0.1)

        # Verify cached
        sae = await sae_service.get_sae(sae_id)
        assert sae.status == "cached"

        # Check compatibility
        compat = await sae_service.check_compatibility(sae_id, layer=12)
        assert compat.compatible

        # Attach
        result = await sae_service.attach_sae(sae_id, layer=12)
        assert result["status"] == "attached"
        assert result["memory_usage_mb"] > 0

        # Verify attached
        status = await sae_service.get_attachment_status()
        assert status.is_attached
        assert status.sae_id == sae_id
        assert status.layer == 12

        # Detach
        result = await sae_service.detach_sae(sae_id)
        assert result["status"] == "detached"
        assert result["memory_freed_mb"] > 0

        # Verify detached
        status = await sae_service.get_attachment_status()
        assert not status.is_attached

    async def test_cannot_attach_second_sae(
        self,
        sae_service,
        mock_model_service,
    ):
        """Should prevent attaching second SAE."""
        # Attach first
        await sae_service.attach_sae("sae-1", layer=12)

        # Try to attach second - should fail
        with pytest.raises(ValueError, match="already attached"):
            await sae_service.attach_sae("sae-2", layer=12)
```

---

## 10. Common Patterns and Anti-Patterns

### DO: Use Lock for State Mutations

```python
# Correct: Lock for thread-safe operations
async with self._attachment_lock:
    self._attached_sae_id = sae_id
    self._hook_handle = hook_handle

# Incorrect: No lock
self._attached_sae_id = sae_id  # Race condition!
```

### DO: Clean Up on Failure

```python
# Correct: Rollback on error
try:
    loaded_sae = self._loader.load(path)
    hook = self._hooker.install(model, layer, loaded_sae)
except Exception:
    if loaded_sae:
        del loaded_sae
    torch.cuda.empty_cache()
    raise

# Incorrect: No cleanup
loaded_sae = self._loader.load(path)  # If install fails, SAE stays in memory
hook = self._hooker.install(model, layer, loaded_sae)
```

### DO: Validate Before Action

```python
# Correct: Validate first
if not self._model_service.is_loaded():
    raise ValueError("No model loaded")

compat = await self.check_compatibility(sae_id, layer)
if not compat.compatible:
    raise SAEIncompatibleError(compat.errors[0])

# Now safe to attach
loaded_sae = self._loader.load(...)

# Incorrect: Validate after loading (wasted work)
loaded_sae = self._loader.load(...)  # Expensive!
if not self._model_service.is_loaded():  # Too late
    raise ValueError(...)
```

### DON'T: Leave Hooks Dangling

```python
# Incorrect: Forgot to remove hook
async def detach_sae(self):
    self._attached_sae_id = None  # Hook still installed!

# Correct: Always remove hook
async def detach_sae(self):
    if self._hook_handle:
        self._hooker.remove(self._hook_handle)
        self._hook_handle = None
    self._attached_sae_id = None
```

---

**Document Status:** Complete
**Next Document:** `003_FTASKS|SAE_Management.md` (Task List)
**Instruction File:** `@0xcc/instruct/006_generate-tasks.md`
