# Technical Design Document: Feature Steering

## miLLM Feature 4

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `004_FPRD|Feature_Steering.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Executive Summary

Feature Steering provides the mechanism for users to modify model behavior by adjusting SAE feature activation strengths during inference. The design builds on the LoadedSAE wrapper from Feature 3, adding a service layer for state management and API exposure.

### Design Principles
1. **Zero Overhead When Disabled:** No performance impact when steering is off
2. **Atomic Updates:** Steering changes are thread-safe and consistent
3. **Immediate Effect:** Changes apply to the next token generation
4. **Stateless API:** Each request is independent; state lives in service

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Storage | In-memory on LoadedSAE | Fast access, no persistence needed |
| Vector | Pre-computed sparse tensor | O(1) application during inference |
| Updates | Rebuild on change | Simple, fast for typical counts |
| Thread Safety | Copy-on-read for vector | No locks during inference |
| Range | -200.0 to +200.0 | Neuronpedia-compatible strength semantics: 0=none, 1=1x multiplier, typical strong effects at +/-50-100. Backend validates range -200 to +200 via Pydantic. |

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Admin UI / Management API                     │
│                                                                  │
│    [Feature List]    [Add Feature]    [Slider]    [Toggle]      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ REST + WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          Steering Routes (in saes.py router)              │   │
│  │  /api/saes/steering/*  (embedded in SAE management routes)│   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  SAEService (steering methods)             │   │
│  │  - set_steering()  - enable_steering()  - get_steering()  │   │
│  │  - clear_steering() - set_steering_batch()                │   │
│  │  NOTE: No separate SteeringService. Steering is managed   │   │
│  │  through SAEService and LoadedSAE methods directly.       │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    LoadedSAE                              │   │
│  │  _steering_values: Dict[int, float]                      │   │
│  │  _steering_enabled: bool                                  │   │
│  │  _steering_vector: Tensor (pre-computed)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ Applied during inference
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Forward Pass                            │
│                                                                  │
│    **Direct Residual Stream Steering** (no SAE reconstruction)   │
│    Layer Output ──► modified = original + Σ(strength_i × W_dec[i,:]) │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Setting a Feature Value

```
User adjusts slider for feature 1234 to +5.0
        │
        ▼
Steering Route (saes.py) ► Validate request
        │                 - Check SAE attached
        │                 - Validate feature index
        │                 - Validate value range (-200 to +200)
        │
        ▼
SAEService methods ────► Update state
        │                 - Call LoadedSAE.set_steering(1234, 5.0)
        │                 - Emit via ProgressEmitter.emit_steering_changed()
        │
        ▼
LoadedSAE ─────────────► Rebuild steering vector
        │                 - Update _steering_values dict
        │                 - Call _rebuild_steering_vector()
        │                 - Pre-compute sparse tensor
        │
        ▼
Return success with new state
```

### Data Flow: Steering Applied During Inference

> **Implementation Note:** Steering uses **direct residual stream modification**, not
> SAE encode-steer-decode. The formula is:
>
> `modified_activations = original_activations + Σ(strength_i × decoder_direction_i)`
>
> where `decoder_direction_i = W_dec[feature_idx, :]` (the SAE decoder row for that feature).
> This is applied uniformly to ALL token positions without SAE reconstruction, making it
> compatible with miStudio/Neuronpedia conventions.

```
Token generation request
        │
        ▼
Model forward pass reaches hooked layer
        │
        ▼
SAE Hook intercepts activations (sae_hooker.py)
        │
        ▼
    if steering_enabled and steering_vector is not None:
        activations = activations + steering_vector   ◄── Pre-computed!
        │                                                 steering_vector = Σ(strength_i × W_dec[i,:])
        │                                                 Shape: (d_model,), added to residual stream
        ▼
    (Optional: monitoring capture of feature_acts)
        │
        ▼
Return modified activations to model
```

---

## 3. Technical Stack

### Dependencies

```python
# Core (from Feature 3)
torch>=2.0

# API
fastapi>=0.109.0
pydantic>=2.0

# WebSocket (for real-time updates)
python-socketio>=5.0  # Already in use for progress
```

### No New Dependencies
Feature Steering leverages existing infrastructure from Features 1-3.

---

## 4. Data Design

### In-Memory State (LoadedSAE)

```python
# millm/ml/sae_wrapper.py (from Feature 3)

class LoadedSAE:
    # Steering state
    _steering_values: Dict[int, float] = {}  # feature_idx → value
    _steering_enabled: bool = False
    _steering_vector: Optional[Tensor] = None  # Pre-computed
```

### Pydantic Schemas

> **Frontend Note:** The admin UI slider uses `step=0.1` (not integer steps),
> allowing fine-grained control within the -200.0 to +200.0 range.

```python
# millm/api/schemas/steering.py

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional


class SetFeatureRequest(BaseModel):
    """Request to set a single feature's steering value."""
    feature_index: int = Field(..., ge=0, description="Feature index")
    value: float = Field(..., ge=-200.0, le=200.0, description="Steering strength")

    @field_validator("value")
    def round_value(cls, v):
        """Round to 1 decimal place for consistency."""
        return round(v, 1)


class BatchSetRequest(BaseModel):
    """Request to set multiple features at once."""
    steering: List[SetFeatureRequest]


class ToggleRequest(BaseModel):
    """Request to enable/disable steering."""
    enabled: bool


class SteeringState(BaseModel):
    """Current steering state."""
    enabled: bool
    active_count: int
    values: Dict[str, float]  # String keys for JSON compatibility
    sae_id: Optional[str]
    sae_feature_count: Optional[int]

    @classmethod
    def from_service(cls, enabled, values, sae_id, feature_count):
        return cls(
            enabled=enabled,
            active_count=len(values),
            values={str(k): v for k, v in values.items()},
            sae_id=sae_id,
            sae_feature_count=feature_count,
        )


class SetFeatureResponse(BaseModel):
    """Response after setting a feature."""
    feature_index: int
    value: float
    active_count: int


class ClearResponse(BaseModel):
    """Response after clearing features."""
    cleared_count: int
    active_count: int
```

---

## 5. API Design

### Route Structure

> **Implementation Note:** Steering routes are embedded in the SAE management router
> (`millm/api/routes/management/saes.py`) under the `/api/saes/steering/*` prefix.
> There is no separate `steering.py` route file or `SteeringService` class. Steering
> is managed through `SAEService` methods (`set_steering()`, `enable_steering()`,
> `get_steering_values()`) and `LoadedSAE` methods directly.

```python
# millm/api/routes/management/saes.py (steering routes section)

from fastapi import APIRouter, Depends, HTTPException

from millm.api.schemas.steering import (
    SetFeatureRequest,
    BatchSetRequest,
    ToggleRequest,
    SteeringState,
    SetFeatureResponse,
    ClearResponse,
)
from millm.services.sae_service import SAEService
from millm.api.dependencies import get_sae_service

router = APIRouter(prefix="/api/saes", tags=["SAE Management & Steering"])


@router.get("/steering", response_model=SteeringState)
async def get_steering_state(
    sae_service: SAEService = Depends(get_sae_service),
):
    """Get current steering state including all active features."""
    return sae_service.get_steering_state()


@router.post("/steering/enable", response_model=SteeringState)
async def toggle_steering(
    request: ToggleRequest,
    sae_service: SAEService = Depends(get_sae_service),
):
    """Enable or disable steering. Values are preserved when disabled."""
    return sae_service.enable_steering(request.enabled)


@router.post("/steering/features", response_model=SetFeatureResponse)
async def set_feature(
    request: SetFeatureRequest,
    sae_service: SAEService = Depends(get_sae_service),
):
    """Set steering value for a single feature."""
    return sae_service.set_steering(
        request.feature_index,
        request.value,
    )


@router.post("/steering/features/batch", response_model=SteeringState)
async def batch_set_features(
    request: BatchSetRequest,
    sae_service: SAEService = Depends(get_sae_service),
):
    """Set multiple feature values at once."""
    return sae_service.set_steering_batch(
        {r.feature_index: r.value for r in request.steering}
    )


@router.delete("/steering/features/{feature_index}", response_model=SetFeatureResponse)
async def remove_feature(
    feature_index: int,
    sae_service: SAEService = Depends(get_sae_service),
):
    """Remove steering for a specific feature (set to 0)."""
    return sae_service.clear_steering_feature(feature_index)


@router.delete("/steering/features", response_model=ClearResponse)
async def clear_all_features(
    sae_service: SAEService = Depends(get_sae_service),
):
    """Clear all steering values."""
    return sae_service.clear_all_steering()
```

---

## 6. Component Architecture

### Steering Logic (in SAEService)

> **Implementation Note:** There is no separate `SteeringService` class. Steering is
> managed through `SAEService` methods (`set_steering()`, `enable_steering()`,
> `get_steering_values()`) and `LoadedSAE` methods. Steering API routes are embedded
> in `/api/saes/steering/*` within the SAE management router (`saes.py`).

The SAEService exposes steering methods that delegate to the `LoadedSAE` instance:

```python
# millm/services/sae_service.py (steering-related methods)

class SAEService:
    """SAE management including steering operations."""

    def set_steering(self, feature_index: int, value: float):
        """Set steering value for a feature. Delegates to LoadedSAE."""
        sae = self._require_attached_sae()
        sae.set_steering(feature_index, value)
        self._emitter.emit_steering_changed(...)

    def enable_steering(self, enabled: bool):
        """Enable or disable steering. Delegates to LoadedSAE."""
        sae = self._require_attached_sae()
        sae.enable_steering(enabled)
        self._emitter.emit_steering_changed(...)

    def get_steering_values(self) -> Dict[int, float]:
        """Get current steering values. Delegates to LoadedSAE."""
        sae = self._require_attached_sae()
        return sae.get_steering_values()

    def set_steering_batch(self, steering: Dict[int, float]):
        """Set multiple features at once. Delegates to LoadedSAE."""
        sae = self._require_attached_sae()
        sae.set_steering_batch(steering)
        self._emitter.emit_steering_changed(...)

    def clear_all_steering(self):
        """Clear all steering values. Delegates to LoadedSAE."""
        sae = self._require_attached_sae()
        sae.clear_steering()
        self._emitter.emit_steering_changed(...)
```

---

## 7. LoadedSAE Steering Implementation

The steering implementation is already part of Feature 3's LoadedSAE wrapper. Here's the complete steering section for reference:

```python
# millm/ml/sae_wrapper.py (steering methods)

class LoadedSAE:
    """LoadedSAE with steering support."""

    def __init__(self, ...):
        # Steering state
        self._steering_values: Dict[int, float] = {}
        self._steering_enabled: bool = False
        self._steering_vector: Optional[Tensor] = None

    def apply_steering(self, activations: Tensor) -> Tensor:
        """Apply steering directly to residual stream activations.

        Uses direct residual stream steering (not SAE encode-steer-decode):
        modified = original + Σ(strength_i × W_dec[feature_idx, :])

        The steering_vector is pre-computed as Σ(strength_i × W_dec[i,:])
        with shape (d_model,), applied uniformly to all token positions.
        """
        if self._steering_enabled and self._steering_vector is not None:
            # Broadcasting: (batch, seq, d_model) + (d_model,)
            activations = activations + self._steering_vector

        return activations

    def set_steering(self, feature_idx: int, value: float):
        """Set steering value for a feature."""
        if not 0 <= feature_idx < self.d_sae:
            raise ValueError(f"Feature index {feature_idx} out of range")
        self._steering_values[feature_idx] = value
        self._rebuild_steering_vector()

    def set_steering_batch(self, steering: Dict[int, float]):
        """Set multiple steering values at once."""
        self._steering_values.update(steering)
        # Remove zeros
        self._steering_values = {
            k: v for k, v in self._steering_values.items() if v != 0
        }
        self._rebuild_steering_vector()

    def clear_steering(self, feature_idx: Optional[int] = None):
        """Clear one or all steering values."""
        if feature_idx is None:
            self._steering_values.clear()
        elif feature_idx in self._steering_values:
            del self._steering_values[feature_idx]
        self._rebuild_steering_vector()

    def get_steering_values(self) -> Dict[int, float]:
        """Get copy of current steering values."""
        return dict(self._steering_values)

    def enable_steering(self, enabled: bool = True):
        """Enable or disable steering."""
        self._steering_enabled = enabled

    def _rebuild_steering_vector(self):
        """Rebuild pre-computed steering vector in residual stream space.

        Computes: steering_vector = Σ(strength_i × W_dec[feature_idx, :])
        Result shape: (d_model,) — added directly to residual stream.
        """
        if not self._steering_values:
            self._steering_vector = None
            return

        # Compute weighted sum of decoder directions
        vector = torch.zeros(
            self.d_model,
            device=self.device,
            dtype=self.W_dec.dtype,
        )
        for idx, strength in self._steering_values.items():
            vector += strength * self.W_dec[idx, :]

        self._steering_vector = vector
```

---

## 8. WebSocket Events

> **Implementation Note:** `steering:update` events are emitted via
> `ProgressEmitter.emit_steering_changed()` (from `millm/sockets/progress.py`),
> not via standalone functions. This is called from all steering mutation endpoints
> in `saes.py` (set_feature, batch_set, enable, clear, etc.).

```python
# millm/sockets/progress.py (ProgressEmitter class)

class ProgressEmitter:
    """Centralized WebSocket event emitter."""

    def emit_steering_changed(self, **kwargs):
        """Emit steering state change event."""
        self._emit(
            "steering_changed",
            kwargs,
        )


# Event data examples:
# {"enabled": True}
# {"feature_index": 1234, "value": 5.0}
# {"feature_index": 1234, "removed": True}
# {"batch": True, "count": 5}
# {"cleared": True, "count": 10}
```

---

## 9. Error Handling

```python
# millm/core/errors.py (add steering errors)

class SteeringError(MiLLMError):
    """Base class for steering errors."""
    pass


class NoSAEAttachedError(SteeringError):
    """No SAE attached for steering."""
    def __init__(self):
        super().__init__(
            code="NO_SAE_ATTACHED",
            message="No SAE attached. Attach an SAE to use steering."
        )


class InvalidFeatureIndexError(SteeringError):
    """Feature index out of range."""
    def __init__(self, index: int, max_index: int):
        super().__init__(
            code="INVALID_FEATURE_INDEX",
            message=f"Feature index {index} out of range (0-{max_index})"
        )


class InvalidSteeringValueError(SteeringError):
    """Steering value out of range."""
    def __init__(self, value: float):
        super().__init__(
            code="INVALID_STEERING_VALUE",
            message=f"Steering value {value} out of range (-200.0 to +200.0)"
        )
```

---

## 10. Testing Strategy

### Unit Tests

> **Note:** Tests target SAEService steering methods (no separate SteeringService).

```python
# tests/unit/services/test_sae_service_steering.py

import pytest
from unittest.mock import Mock, MagicMock

from millm.services.sae_service import SAEService


@pytest.fixture
def mock_sae():
    """Create mock LoadedSAE with steering support."""
    sae = MagicMock()
    sae.d_sae = 1000
    sae._steering_enabled = False
    sae._steering_values = {}
    sae.get_steering_values.return_value = {}
    return sae


@pytest.fixture
def sae_service(mock_sae):
    service = SAEService(...)
    service._loaded_sae = mock_sae
    return service


class TestSAEServiceSteering:
    async def test_get_state_no_sae(self, sae_service):
        """Should return empty state when no SAE attached."""
        sae_service._loaded_sae = None

        state = sae_service.get_steering_state()

        assert not state.enabled
        assert state.active_count == 0

    async def test_set_feature_validates_index(self, sae_service):
        """Should reject out-of-range feature index."""
        with pytest.raises(ValueError, match="out of range"):
            sae_service.set_steering(9999, 5.0)

    async def test_batch_validates_all_first(self, sae_service):
        """Should validate all indices before applying any."""
        with pytest.raises(ValueError):
            sae_service.set_steering_batch({
                100: 5.0,
                9999: 3.0,  # Invalid
            })

        # Verify nothing was set
        sae_service._loaded_sae.set_steering_batch.assert_not_called()
```

### Integration Tests

```python
# tests/integration/api/test_steering_routes.py

class TestSteeringRoutes:
    async def test_get_state_endpoint(self, client, attached_sae):
        """GET /api/saes/steering returns current state."""
        response = await client.get("/api/saes/steering")

        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "active_count" in data
        assert data["sae_id"] is not None

    async def test_set_feature_endpoint(self, client, attached_sae):
        """POST /api/saes/steering/features sets a feature."""
        response = await client.post(
            "/api/saes/steering/features",
            json={"feature_index": 100, "value": 5.0}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["feature_index"] == 100
        assert data["value"] == 5.0
        assert data["active_count"] == 1

    async def test_no_sae_returns_error(self, client):
        """Should return error when no SAE attached."""
        response = await client.post(
            "/api/saes/steering/features",
            json={"feature_index": 100, "value": 5.0}
        )

        assert response.status_code == 400
        assert "No SAE attached" in response.json()["detail"]
```

---

## 11. Performance Considerations

### Vector Computation Efficiency

```python
# Optimized steering vector rebuild (residual stream space)

def _rebuild_steering_vector(self):
    """Rebuild with minimal allocations.

    Computes Σ(strength_i × W_dec[i,:]) in d_model space.
    """
    if not self._steering_values:
        self._steering_vector = None
        return

    # Pre-allocate on correct device (d_model, not d_sae)
    if self._steering_vector is None or \
       self._steering_vector.device != self.device:
        self._steering_vector = torch.zeros(
            self.d_model,
            device=self.device,
            dtype=self.W_dec.dtype,
        )
    else:
        # Reuse existing tensor
        self._steering_vector.zero_()

    # Batch compute: gather decoder rows and weight them
    indices = list(self._steering_values.keys())
    strengths = torch.tensor(
        list(self._steering_values.values()),
        device=self.device,
        dtype=self.W_dec.dtype,
    )
    # W_dec[indices] shape: (n_features, d_model)
    # strengths shape: (n_features,)
    self._steering_vector = (strengths.unsqueeze(1) * self.W_dec[indices]).sum(dim=0)
```

### Thread Safety During Inference

```python
# Copy-on-read for thread safety during inference

def apply_steering(self, activations: Tensor) -> Tensor:
    """Thread-safe direct residual stream steering."""
    # Capture local reference (atomic read)
    steering_enabled = self._steering_enabled
    steering_vector = self._steering_vector

    # Apply steering if enabled
    if steering_enabled and steering_vector is not None:
        # steering_vector shape: (d_model,)
        # activations shape: (batch, seq, d_model)
        activations = activations + steering_vector

    return activations
```

---

## 12. Development Phases

### Phase 1: SAEService Steering Methods (1 day)
- [ ] Add steering methods to SAEService
- [ ] Validation methods
- [ ] State management

### Phase 2: API Routes (1 day)
- [ ] Pydantic schemas
- [ ] Route handlers
- [ ] Error handling

### Phase 3: WebSocket Events (0.5 days)
- [ ] Event emission
- [ ] Client notification

### Phase 4: Testing (1 day)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

---

**Document Status:** Complete
**Next Document:** `004_FTID|Feature_Steering.md` (Technical Implementation Document)
**Instruction File:** `@0xcc/instruct/005_create-tid.md`
