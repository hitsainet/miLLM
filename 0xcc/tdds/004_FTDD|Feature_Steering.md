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
| Range | -10.0 to +10.0 | Research-based reasonable range |

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
│  │                  Steering Routes                          │   │
│  │  /api/steering   /api/steering/features   /api/steering/* │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  SteeringService                          │   │
│  │  - get_state()   - set_feature()   - toggle()            │   │
│  │  - clear_all()   - batch_set()     - validate()          │   │
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
│    Layer Output ──► SAE Encode ──► [+Steering] ──► SAE Decode   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Setting a Feature Value

```
User adjusts slider for feature 1234 to +5.0
        │
        ▼
Steering Route ────────► Validate request
        │                 - Check SAE attached
        │                 - Validate feature index
        │                 - Validate value range
        │
        ▼
Steering Service ──────► Update state
        │                 - Call LoadedSAE.set_steering(1234, 5.0)
        │                 - Emit WebSocket event
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

```
Token generation request
        │
        ▼
Model forward pass reaches hooked layer
        │
        ▼
SAE Hook intercepts output
        │
        ▼
LoadedSAE.forward() called ──────────────────────────────────┐
        │                                                     │
        ▼                                                     │
    Encode: x @ W_enc + b_enc → feature_acts                 │
        │                                                     │
        ▼                                                     │
    if steering_enabled and steering_vector:                  │
        feature_acts = feature_acts + steering_vector   ◄────┘
        │                                                 Pre-computed!
        ▼
    Decode: feature_acts @ W_dec + b_dec → output
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

```python
# millm/api/schemas/steering.py

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional


class SetFeatureRequest(BaseModel):
    """Request to set a single feature's steering value."""
    feature_index: int = Field(..., ge=0, description="Feature index")
    value: float = Field(..., ge=-10.0, le=10.0, description="Steering strength")

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

```python
# millm/api/routes/management/steering.py

from fastapi import APIRouter, Depends, HTTPException

from millm.api.schemas.steering import (
    SetFeatureRequest,
    BatchSetRequest,
    ToggleRequest,
    SteeringState,
    SetFeatureResponse,
    ClearResponse,
)
from millm.services.steering_service import SteeringService
from millm.api.dependencies import get_steering_service

router = APIRouter(prefix="/api/steering", tags=["Feature Steering"])


@router.get("", response_model=SteeringState)
async def get_steering_state(
    steering: SteeringService = Depends(get_steering_service),
):
    """Get current steering state including all active features."""
    return await steering.get_state()


@router.post("/enable", response_model=SteeringState)
async def toggle_steering(
    request: ToggleRequest,
    steering: SteeringService = Depends(get_steering_service),
):
    """Enable or disable steering. Values are preserved when disabled."""
    return await steering.set_enabled(request.enabled)


@router.post("/features", response_model=SetFeatureResponse)
async def set_feature(
    request: SetFeatureRequest,
    steering: SteeringService = Depends(get_steering_service),
):
    """Set steering value for a single feature."""
    return await steering.set_feature(
        request.feature_index,
        request.value,
    )


@router.post("/features/batch", response_model=SteeringState)
async def batch_set_features(
    request: BatchSetRequest,
    steering: SteeringService = Depends(get_steering_service),
):
    """Set multiple feature values at once."""
    return await steering.batch_set(
        {r.feature_index: r.value for r in request.steering}
    )


@router.delete("/features/{feature_index}", response_model=SetFeatureResponse)
async def remove_feature(
    feature_index: int,
    steering: SteeringService = Depends(get_steering_service),
):
    """Remove steering for a specific feature (set to 0)."""
    return await steering.clear_feature(feature_index)


@router.delete("/features", response_model=ClearResponse)
async def clear_all_features(
    steering: SteeringService = Depends(get_steering_service),
):
    """Clear all steering values."""
    return await steering.clear_all()
```

---

## 6. Component Architecture

### SteeringService Design

```python
# millm/services/steering_service.py

from typing import Dict, Optional
import logging

from millm.services.sae_service import SAEService
from millm.api.schemas.steering import (
    SteeringState,
    SetFeatureResponse,
    ClearResponse,
)
from millm.sockets.events import emit_steering_changed

logger = logging.getLogger(__name__)


class SteeringService:
    """
    Service for managing feature steering.

    Wraps LoadedSAE steering methods with validation
    and state tracking.
    """

    def __init__(self, sae_service: SAEService):
        self._sae_service = sae_service

    @property
    def _sae(self):
        """Get currently attached SAE."""
        return self._sae_service._loaded_sae

    def _require_sae(self):
        """Raise if no SAE attached."""
        if not self._sae_service._attached_sae_id:
            raise ValueError("No SAE attached. Attach an SAE to use steering.")

    def _validate_feature_index(self, index: int):
        """Validate feature index is in range."""
        self._require_sae()
        if not 0 <= index < self._sae.d_sae:
            raise ValueError(
                f"Feature index {index} out of range. "
                f"SAE has {self._sae.d_sae} features (0-{self._sae.d_sae - 1})"
            )

    async def get_state(self) -> SteeringState:
        """Get current steering state."""
        if not self._sae_service._attached_sae_id:
            return SteeringState(
                enabled=False,
                active_count=0,
                values={},
                sae_id=None,
                sae_feature_count=None,
            )

        return SteeringState.from_service(
            enabled=self._sae._steering_enabled,
            values=self._sae.get_steering_values(),
            sae_id=self._sae_service._attached_sae_id,
            feature_count=self._sae.d_sae,
        )

    async def set_enabled(self, enabled: bool) -> SteeringState:
        """Enable or disable steering."""
        self._require_sae()

        self._sae.enable_steering(enabled)

        logger.info(f"Steering {'enabled' if enabled else 'disabled'}")
        emit_steering_changed(enabled=enabled)

        return await self.get_state()

    async def set_feature(
        self,
        feature_index: int,
        value: float,
    ) -> SetFeatureResponse:
        """Set steering value for a feature."""
        self._validate_feature_index(feature_index)

        # Round to 1 decimal
        value = round(value, 1)

        # Set on SAE
        if value == 0:
            self._sae.clear_steering(feature_index)
        else:
            self._sae.set_steering(feature_index, value)

        logger.info(f"Set feature {feature_index} to {value}")
        emit_steering_changed(
            feature_index=feature_index,
            value=value,
        )

        return SetFeatureResponse(
            feature_index=feature_index,
            value=value,
            active_count=len(self._sae.get_steering_values()),
        )

    async def batch_set(
        self,
        steering: Dict[int, float],
    ) -> SteeringState:
        """Set multiple features at once."""
        self._require_sae()

        # Validate all indices first
        for index in steering:
            self._validate_feature_index(index)

        # Apply all
        rounded = {k: round(v, 1) for k, v in steering.items()}
        self._sae.set_steering_batch(rounded)

        logger.info(f"Batch set {len(steering)} features")
        emit_steering_changed(batch=True, count=len(steering))

        return await self.get_state()

    async def clear_feature(self, feature_index: int) -> SetFeatureResponse:
        """Clear steering for a specific feature."""
        self._require_sae()

        self._sae.clear_steering(feature_index)

        logger.info(f"Cleared feature {feature_index}")
        emit_steering_changed(feature_index=feature_index, removed=True)

        return SetFeatureResponse(
            feature_index=feature_index,
            value=0.0,
            active_count=len(self._sae.get_steering_values()),
        )

    async def clear_all(self) -> ClearResponse:
        """Clear all steering values."""
        self._require_sae()

        values = self._sae.get_steering_values()
        cleared_count = len(values)

        self._sae.clear_steering()

        logger.info(f"Cleared all {cleared_count} steering values")
        emit_steering_changed(cleared=True, count=cleared_count)

        return ClearResponse(
            cleared_count=cleared_count,
            active_count=0,
        )
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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional steering."""
        # Encode
        feature_acts = torch.relu(x @ self.W_enc + self.b_enc)

        # Apply steering if enabled
        if self._steering_enabled and self._steering_vector is not None:
            # Broadcasting: (batch, seq, d_sae) + (d_sae,)
            feature_acts = feature_acts + self._steering_vector

        # Decode
        return feature_acts @ self.W_dec + self.b_dec

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
        """Rebuild pre-computed steering vector."""
        if not self._steering_values:
            self._steering_vector = None
            return

        # Create sparse vector on same device as weights
        vector = torch.zeros(
            self.d_sae,
            device=self.device,
            dtype=self.W_enc.dtype,
        )
        for idx, value in self._steering_values.items():
            vector[idx] = value

        self._steering_vector = vector
```

---

## 8. WebSocket Events

```python
# millm/sockets/events.py (add steering events)

from millm.sockets.manager import socket_manager


def emit_steering_changed(**kwargs):
    """Emit steering state change event."""
    socket_manager.emit(
        "steering_changed",
        kwargs,
        room="admin",  # Only admin clients
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
            message=f"Steering value {value} out of range (-10.0 to +10.0)"
        )
```

---

## 10. Testing Strategy

### Unit Tests

```python
# tests/unit/services/test_steering_service.py

import pytest
from unittest.mock import Mock, MagicMock

from millm.services.steering_service import SteeringService


@pytest.fixture
def mock_sae_service():
    """Create mock SAE service with attached SAE."""
    service = Mock()
    service._attached_sae_id = "test-sae"
    service._loaded_sae = MagicMock()
    service._loaded_sae.d_sae = 1000
    service._loaded_sae._steering_enabled = False
    service._loaded_sae._steering_values = {}
    service._loaded_sae.get_steering_values.return_value = {}
    return service


@pytest.fixture
def steering_service(mock_sae_service):
    return SteeringService(mock_sae_service)


class TestSteeringService:
    async def test_get_state_no_sae(self, mock_sae_service):
        """Should return empty state when no SAE attached."""
        mock_sae_service._attached_sae_id = None
        service = SteeringService(mock_sae_service)

        state = await service.get_state()

        assert not state.enabled
        assert state.active_count == 0
        assert state.sae_id is None

    async def test_set_feature_validates_index(self, steering_service):
        """Should reject out-of-range feature index."""
        with pytest.raises(ValueError, match="out of range"):
            await steering_service.set_feature(9999, 5.0)

    async def test_set_feature_rounds_value(self, steering_service):
        """Should round value to 1 decimal place."""
        result = await steering_service.set_feature(100, 5.55)

        steering_service._sae.set_steering.assert_called_with(100, 5.6)

    async def test_clear_on_zero_value(self, steering_service):
        """Setting value to 0 should clear the feature."""
        await steering_service.set_feature(100, 0.0)

        steering_service._sae.clear_steering.assert_called_with(100)

    async def test_batch_validates_all_first(self, steering_service):
        """Should validate all indices before applying any."""
        with pytest.raises(ValueError):
            await steering_service.batch_set({
                100: 5.0,
                9999: 3.0,  # Invalid
            })

        # Verify nothing was set
        steering_service._sae.set_steering_batch.assert_not_called()
```

### Integration Tests

```python
# tests/integration/api/test_steering_routes.py

class TestSteeringRoutes:
    async def test_get_state_endpoint(self, client, attached_sae):
        """GET /api/steering returns current state."""
        response = await client.get("/api/steering")

        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "active_count" in data
        assert data["sae_id"] is not None

    async def test_set_feature_endpoint(self, client, attached_sae):
        """POST /api/steering/features sets a feature."""
        response = await client.post(
            "/api/steering/features",
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
            "/api/steering/features",
            json={"feature_index": 100, "value": 5.0}
        )

        assert response.status_code == 400
        assert "No SAE attached" in response.json()["detail"]
```

---

## 11. Performance Considerations

### Vector Computation Efficiency

```python
# Optimized steering vector rebuild

def _rebuild_steering_vector(self):
    """Rebuild with minimal allocations."""
    if not self._steering_values:
        self._steering_vector = None
        return

    # Pre-allocate on correct device
    if self._steering_vector is None or \
       self._steering_vector.device != self.device:
        self._steering_vector = torch.zeros(
            self.d_sae,
            device=self.device,
            dtype=self.W_enc.dtype,
        )
    else:
        # Reuse existing tensor
        self._steering_vector.zero_()

    # Set values (fast indexed assignment)
    indices = list(self._steering_values.keys())
    values = list(self._steering_values.values())
    self._steering_vector[indices] = torch.tensor(
        values,
        device=self.device,
        dtype=self.W_enc.dtype,
    )
```

### Thread Safety During Inference

```python
# Copy-on-read for thread safety during inference

def forward(self, x: Tensor) -> Tensor:
    """Thread-safe forward with steering."""
    # Encode
    feature_acts = torch.relu(x @ self.W_enc + self.b_enc)

    # Capture local reference (atomic read)
    steering_enabled = self._steering_enabled
    steering_vector = self._steering_vector

    # Apply steering if enabled
    if steering_enabled and steering_vector is not None:
        feature_acts = feature_acts + steering_vector

    # Decode
    return feature_acts @ self.W_dec + self.b_dec
```

---

## 12. Development Phases

### Phase 1: Service Layer (1 day)
- [ ] SteeringService implementation
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
