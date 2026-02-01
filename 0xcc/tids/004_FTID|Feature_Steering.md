# Technical Implementation Document: Feature Steering

## miLLM Feature 4

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Complete - Implemented as part of SAE Management (Feature 3)
**References:**
- Feature PRD: `004_FPRD|Feature_Steering.md`
- Feature TDD: `004_FTDD|Feature_Steering.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Overview

This Technical Implementation Document provides specific implementation guidance for Feature 4: Feature Steering. This feature builds on the LoadedSAE wrapper from Feature 3, adding a service layer and API endpoints for steering management.

### Implementation Philosophy
- **Leverage Existing:** LoadedSAE already has steering methods from Feature 3
- **Thin Service Layer:** SteeringService wraps SAE methods with validation
- **Real-Time Events:** WebSocket notifications for UI updates
- **Zero Overhead When Off:** No performance impact when steering disabled

---

## 2. File Structure

### Backend Organization

```
millm/
├── api/
│   ├── routes/
│   │   └── management/
│   │       └── steering.py              # Steering API routes
│   └── schemas/
│       └── steering.py                  # Pydantic schemas
│
├── services/
│   └── steering_service.py              # Steering service layer
│
├── sockets/
│   └── events.py                        # Add steering events
│
└── core/
    └── errors.py                        # Add steering-specific errors
```

### Test Organization

```
tests/
├── unit/
│   └── services/
│       └── test_steering_service.py     # Service unit tests
│
├── integration/
│   └── api/
│       └── test_steering_routes.py      # API integration tests
│
└── fixtures/
    └── steering/
        └── conftest.py                  # Steering test fixtures
```

---

## 3. Pydantic Schemas Implementation

```python
# millm/api/schemas/steering.py

"""
Pydantic schemas for Feature Steering API.

Implementation notes:
1. Use Field() for validation constraints
2. Round values to 1 decimal for consistency
3. Use string keys in SteeringState.values for JSON compatibility
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional


class SetFeatureRequest(BaseModel):
    """Request to set steering value for a single feature."""
    feature_index: int = Field(
        ...,
        ge=0,
        description="Feature index (0-indexed)"
    )
    value: float = Field(
        ...,
        ge=-10.0,
        le=10.0,
        description="Steering strength (-10 to +10)"
    )

    @field_validator("value", mode="after")
    @classmethod
    def round_to_one_decimal(cls, v: float) -> float:
        """Round to 1 decimal place for UI consistency."""
        return round(v, 1)


class BatchSetRequest(BaseModel):
    """Request to set multiple features at once."""
    steering: List[SetFeatureRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of feature-value pairs"
    )


class ToggleRequest(BaseModel):
    """Request to enable/disable steering globally."""
    enabled: bool = Field(..., description="Enable or disable steering")


class SteeringState(BaseModel):
    """Complete steering state response."""
    enabled: bool = Field(..., description="Whether steering is active")
    active_count: int = Field(
        ...,
        ge=0,
        description="Number of features with non-zero values"
    )
    values: Dict[str, float] = Field(
        default_factory=dict,
        description="Map of feature_index (as string) to value"
    )
    sae_id: Optional[str] = Field(
        None,
        description="ID of attached SAE, or null if none"
    )
    sae_feature_count: Optional[int] = Field(
        None,
        description="Total features in attached SAE"
    )

    @classmethod
    def from_sae(
        cls,
        enabled: bool,
        values: Dict[int, float],
        sae_id: Optional[str],
        feature_count: Optional[int],
    ) -> "SteeringState":
        """
        Create from SAE state.

        Converts integer keys to strings for JSON serialization.
        """
        return cls(
            enabled=enabled,
            active_count=len(values),
            values={str(k): v for k, v in values.items()},
            sae_id=sae_id,
            sae_feature_count=feature_count,
        )

    def to_integer_values(self) -> Dict[int, float]:
        """Convert string keys back to integers."""
        return {int(k): v for k, v in self.values.items()}


class SetFeatureResponse(BaseModel):
    """Response after setting a feature value."""
    feature_index: int
    value: float
    active_count: int = Field(..., description="Total active features after change")


class ClearResponse(BaseModel):
    """Response after clearing features."""
    cleared_count: int = Field(..., description="Number of features cleared")
    active_count: int = Field(
        ...,
        description="Active features remaining (should be 0 for clear all)"
    )
```

---

## 4. Steering Service Implementation

```python
# millm/services/steering_service.py

"""
Steering service for managing feature steering.

Implementation notes:
1. Wraps LoadedSAE steering methods with validation
2. Emits WebSocket events for UI updates
3. Handles edge cases (no SAE attached, etc.)
"""

from typing import Dict, Optional
import logging

from millm.services.sae_service import SAEService
from millm.api.schemas.steering import (
    SteeringState,
    SetFeatureResponse,
    ClearResponse,
)
from millm.core.errors import NoSAEAttachedError, InvalidFeatureIndexError

logger = logging.getLogger(__name__)


class SteeringService:
    """
    Service layer for feature steering operations.

    Depends on SAEService to access the loaded SAE.
    All steering state lives on the LoadedSAE instance.
    """

    def __init__(self, sae_service: SAEService):
        """
        Initialize steering service.

        Args:
            sae_service: SAE service for accessing loaded SAE
        """
        self._sae_service = sae_service

    @property
    def _sae(self):
        """
        Get currently loaded SAE.

        Returns None if no SAE attached.
        """
        return self._sae_service._loaded_sae

    @property
    def _sae_id(self) -> Optional[str]:
        """Get ID of attached SAE."""
        return self._sae_service._attached_sae_id

    def _require_sae(self):
        """
        Validate that an SAE is attached.

        Raises:
            NoSAEAttachedError: If no SAE is attached
        """
        if not self._sae_id or not self._sae:
            raise NoSAEAttachedError()

    def _validate_feature_index(self, index: int):
        """
        Validate feature index is in valid range.

        Raises:
            InvalidFeatureIndexError: If index out of range
        """
        self._require_sae()
        if not 0 <= index < self._sae.d_sae:
            raise InvalidFeatureIndexError(
                index=index,
                max_index=self._sae.d_sae - 1,
            )

    async def get_state(self) -> SteeringState:
        """
        Get current steering state.

        Returns empty state if no SAE attached.
        """
        if not self._sae_id or not self._sae:
            return SteeringState(
                enabled=False,
                active_count=0,
                values={},
                sae_id=None,
                sae_feature_count=None,
            )

        return SteeringState.from_sae(
            enabled=self._sae._steering_enabled,
            values=self._sae.get_steering_values(),
            sae_id=self._sae_id,
            feature_count=self._sae.d_sae,
        )

    async def set_enabled(self, enabled: bool) -> SteeringState:
        """
        Enable or disable steering globally.

        Steering values are preserved when disabled.
        """
        self._require_sae()

        self._sae.enable_steering(enabled)

        logger.info(f"Steering {'enabled' if enabled else 'disabled'}")
        self._emit_event(enabled=enabled)

        return await self.get_state()

    async def set_feature(
        self,
        feature_index: int,
        value: float,
    ) -> SetFeatureResponse:
        """
        Set steering value for a single feature.

        Setting value to 0 clears the feature.
        """
        self._validate_feature_index(feature_index)

        # Round to 1 decimal place
        value = round(value, 1)

        # Set or clear based on value
        if value == 0.0:
            self._sae.clear_steering(feature_index)
            logger.info(f"Cleared steering for feature {feature_index}")
        else:
            self._sae.set_steering(feature_index, value)
            logger.info(f"Set feature {feature_index} to {value}")

        # Emit event for UI
        self._emit_event(feature_index=feature_index, value=value)

        return SetFeatureResponse(
            feature_index=feature_index,
            value=value,
            active_count=len(self._sae.get_steering_values()),
        )

    async def batch_set(
        self,
        steering: Dict[int, float],
    ) -> SteeringState:
        """
        Set multiple feature values at once.

        Validates all indices before applying any changes.
        """
        self._require_sae()

        # Validate ALL indices first
        for index in steering:
            if not 0 <= index < self._sae.d_sae:
                raise InvalidFeatureIndexError(
                    index=index,
                    max_index=self._sae.d_sae - 1,
                )

        # Round all values
        rounded = {k: round(v, 1) for k, v in steering.items()}

        # Apply batch
        self._sae.set_steering_batch(rounded)

        logger.info(f"Batch set {len(steering)} features")
        self._emit_event(batch=True, count=len(steering))

        return await self.get_state()

    async def clear_feature(self, feature_index: int) -> SetFeatureResponse:
        """
        Clear steering for a specific feature.

        No validation needed - clearing non-existent feature is no-op.
        """
        self._require_sae()

        self._sae.clear_steering(feature_index)

        logger.info(f"Cleared feature {feature_index}")
        self._emit_event(feature_index=feature_index, removed=True)

        return SetFeatureResponse(
            feature_index=feature_index,
            value=0.0,
            active_count=len(self._sae.get_steering_values()),
        )

    async def clear_all(self) -> ClearResponse:
        """Clear all steering values."""
        self._require_sae()

        # Get count before clearing
        values = self._sae.get_steering_values()
        cleared_count = len(values)

        # Clear all
        self._sae.clear_steering()

        logger.info(f"Cleared all {cleared_count} steering values")
        self._emit_event(cleared=True, count=cleared_count)

        return ClearResponse(
            cleared_count=cleared_count,
            active_count=0,
        )

    def _emit_event(self, **kwargs):
        """
        Emit WebSocket event for steering change.

        Import here to avoid circular dependency.
        """
        try:
            from millm.sockets.events import emit_steering_changed
            emit_steering_changed(**kwargs)
        except ImportError:
            # WebSocket module may not be available in tests
            pass
```

---

## 5. API Routes Implementation

```python
# millm/api/routes/management/steering.py

"""
Feature Steering API routes.

Endpoints:
- GET  /api/steering              - Get current state
- POST /api/steering/enable       - Enable/disable steering
- POST /api/steering/features     - Set single feature
- POST /api/steering/features/batch - Set multiple features
- DELETE /api/steering/features/{index} - Clear single feature
- DELETE /api/steering/features   - Clear all features
"""

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
from millm.core.errors import NoSAEAttachedError, InvalidFeatureIndexError

router = APIRouter(prefix="/api/steering", tags=["Feature Steering"])


def handle_steering_error(e: Exception):
    """Convert steering errors to HTTP exceptions."""
    if isinstance(e, NoSAEAttachedError):
        raise HTTPException(
            status_code=400,
            detail="No SAE attached. Attach an SAE to use steering."
        )
    elif isinstance(e, InvalidFeatureIndexError):
        raise HTTPException(
            status_code=400,
            detail=str(e.message)
        )
    raise e


@router.get("", response_model=SteeringState)
async def get_steering_state(
    steering: SteeringService = Depends(get_steering_service),
):
    """
    Get current steering state.

    Returns enabled status, active feature count, and all values.
    Returns empty state if no SAE is attached.
    """
    return await steering.get_state()


@router.post("/enable", response_model=SteeringState)
async def toggle_steering(
    request: ToggleRequest,
    steering: SteeringService = Depends(get_steering_service),
):
    """
    Enable or disable steering globally.

    Steering values are preserved when disabled.
    Requires an SAE to be attached.
    """
    try:
        return await steering.set_enabled(request.enabled)
    except Exception as e:
        handle_steering_error(e)


@router.post("/features", response_model=SetFeatureResponse)
async def set_feature(
    request: SetFeatureRequest,
    steering: SteeringService = Depends(get_steering_service),
):
    """
    Set steering value for a single feature.

    Value range: -10.0 to +10.0
    Setting value to 0 clears the feature.
    """
    try:
        return await steering.set_feature(
            request.feature_index,
            request.value,
        )
    except Exception as e:
        handle_steering_error(e)


@router.post("/features/batch", response_model=SteeringState)
async def batch_set_features(
    request: BatchSetRequest,
    steering: SteeringService = Depends(get_steering_service),
):
    """
    Set multiple feature values at once.

    All indices are validated before any values are applied.
    Maximum 100 features per batch.
    """
    try:
        steering_dict = {
            item.feature_index: item.value
            for item in request.steering
        }
        return await steering.batch_set(steering_dict)
    except Exception as e:
        handle_steering_error(e)


@router.delete("/features/{feature_index}", response_model=SetFeatureResponse)
async def remove_feature(
    feature_index: int,
    steering: SteeringService = Depends(get_steering_service),
):
    """
    Remove steering for a specific feature.

    Sets the feature value to 0 (no steering effect).
    """
    try:
        return await steering.clear_feature(feature_index)
    except Exception as e:
        handle_steering_error(e)


@router.delete("/features", response_model=ClearResponse)
async def clear_all_features(
    steering: SteeringService = Depends(get_steering_service),
):
    """
    Clear all steering values.

    Resets all features to 0 (no steering effect).
    Steering enabled state is preserved.
    """
    try:
        return await steering.clear_all()
    except Exception as e:
        handle_steering_error(e)
```

---

## 6. Error Classes Implementation

```python
# millm/core/errors.py (add to existing file)

"""
Steering-specific error classes.
"""


class SteeringError(MiLLMError):
    """Base class for steering errors."""
    pass


class NoSAEAttachedError(SteeringError):
    """Raised when steering operation requires SAE but none attached."""

    def __init__(self):
        super().__init__(
            code="NO_SAE_ATTACHED",
            message="No SAE attached. Attach an SAE to enable steering."
        )


class InvalidFeatureIndexError(SteeringError):
    """Raised when feature index is out of valid range."""

    def __init__(self, index: int, max_index: int):
        super().__init__(
            code="INVALID_FEATURE_INDEX",
            message=f"Feature index {index} out of range. Valid range: 0-{max_index}"
        )
        self.index = index
        self.max_index = max_index


class InvalidSteeringValueError(SteeringError):
    """Raised when steering value is out of allowed range."""

    def __init__(self, value: float, min_val: float = -10.0, max_val: float = 10.0):
        super().__init__(
            code="INVALID_STEERING_VALUE",
            message=f"Steering value {value} out of range ({min_val} to {max_val})"
        )
        self.value = value
```

---

## 7. WebSocket Events Implementation

```python
# millm/sockets/events.py (add steering events)

"""
WebSocket event emitters for steering.
"""

from millm.sockets.manager import socket_manager


def emit_steering_changed(**kwargs):
    """
    Emit steering state change event.

    Event types:
    - {"enabled": bool} - Steering toggled
    - {"feature_index": int, "value": float} - Feature value changed
    - {"feature_index": int, "removed": True} - Feature cleared
    - {"batch": True, "count": int} - Batch update
    - {"cleared": True, "count": int} - All cleared

    Sent to "admin" room only (not inference clients).
    """
    socket_manager.emit(
        event="steering_changed",
        data=kwargs,
        room="admin",
    )
```

---

## 8. Dependency Setup

```python
# millm/api/dependencies.py (add steering dependency)

from fastapi import Request

from millm.services.steering_service import SteeringService


def get_steering_service(request: Request) -> SteeringService:
    """Get SteeringService from app state."""
    return request.app.state.steering_service


# In main.py lifespan:
async def lifespan(app):
    # ... existing setup ...

    # Initialize steering service
    app.state.steering_service = SteeringService(
        sae_service=app.state.sae_service
    )

    yield

    # ... cleanup ...
```

---

## 9. Testing Patterns

### Unit Test Example

```python
# tests/unit/services/test_steering_service.py

import pytest
from unittest.mock import Mock, MagicMock

from millm.services.steering_service import SteeringService
from millm.core.errors import NoSAEAttachedError, InvalidFeatureIndexError


@pytest.fixture
def mock_sae():
    """Create mock LoadedSAE."""
    sae = MagicMock()
    sae.d_sae = 1000
    sae._steering_enabled = False
    sae._steering_values = {}
    sae.get_steering_values.return_value = {}
    return sae


@pytest.fixture
def mock_sae_service(mock_sae):
    """Create mock SAE service with attached SAE."""
    service = Mock()
    service._attached_sae_id = "test-sae"
    service._loaded_sae = mock_sae
    return service


@pytest.fixture
def steering_service(mock_sae_service):
    """Create steering service with mocked dependencies."""
    return SteeringService(mock_sae_service)


class TestSteeringServiceState:
    """Tests for get_state."""

    @pytest.mark.asyncio
    async def test_returns_empty_state_when_no_sae(self, mock_sae_service):
        """Should return empty state when no SAE attached."""
        mock_sae_service._attached_sae_id = None
        mock_sae_service._loaded_sae = None
        service = SteeringService(mock_sae_service)

        state = await service.get_state()

        assert not state.enabled
        assert state.active_count == 0
        assert state.sae_id is None

    @pytest.mark.asyncio
    async def test_returns_sae_state(self, steering_service, mock_sae):
        """Should return SAE steering state."""
        mock_sae._steering_enabled = True
        mock_sae.get_steering_values.return_value = {100: 5.0, 200: -3.0}

        state = await steering_service.get_state()

        assert state.enabled
        assert state.active_count == 2
        assert state.values == {"100": 5.0, "200": -3.0}


class TestSteeringServiceSetFeature:
    """Tests for set_feature."""

    @pytest.mark.asyncio
    async def test_requires_sae(self, mock_sae_service):
        """Should raise when no SAE attached."""
        mock_sae_service._attached_sae_id = None
        service = SteeringService(mock_sae_service)

        with pytest.raises(NoSAEAttachedError):
            await service.set_feature(100, 5.0)

    @pytest.mark.asyncio
    async def test_validates_feature_index(self, steering_service):
        """Should reject out-of-range index."""
        with pytest.raises(InvalidFeatureIndexError):
            await steering_service.set_feature(9999, 5.0)

    @pytest.mark.asyncio
    async def test_sets_feature_on_sae(self, steering_service, mock_sae):
        """Should call SAE set_steering."""
        await steering_service.set_feature(100, 5.0)

        mock_sae.set_steering.assert_called_once_with(100, 5.0)

    @pytest.mark.asyncio
    async def test_clears_on_zero_value(self, steering_service, mock_sae):
        """Should clear feature when value is 0."""
        await steering_service.set_feature(100, 0.0)

        mock_sae.clear_steering.assert_called_once_with(100)
        mock_sae.set_steering.assert_not_called()

    @pytest.mark.asyncio
    async def test_rounds_value(self, steering_service, mock_sae):
        """Should round value to 1 decimal."""
        await steering_service.set_feature(100, 5.55)

        mock_sae.set_steering.assert_called_once_with(100, 5.6)


class TestSteeringServiceBatch:
    """Tests for batch_set."""

    @pytest.mark.asyncio
    async def test_validates_all_indices_first(self, steering_service, mock_sae):
        """Should validate all indices before applying any."""
        with pytest.raises(InvalidFeatureIndexError):
            await steering_service.batch_set({
                100: 5.0,   # Valid
                9999: 3.0,  # Invalid
            })

        # Verify nothing was applied
        mock_sae.set_steering_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_applies_batch(self, steering_service, mock_sae):
        """Should call SAE batch set."""
        await steering_service.batch_set({100: 5.0, 200: -3.0})

        mock_sae.set_steering_batch.assert_called_once_with({
            100: 5.0,
            200: -3.0,
        })
```

### Integration Test Example

```python
# tests/integration/api/test_steering_routes.py

import pytest
from httpx import AsyncClient


@pytest.fixture
async def client_with_sae(app, attached_sae):
    """Create test client with SAE attached."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


class TestSteeringRoutes:
    @pytest.mark.asyncio
    async def test_get_state(self, client_with_sae):
        """GET /api/steering returns current state."""
        response = await client_with_sae.get("/api/steering")

        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "active_count" in data
        assert "sae_id" in data

    @pytest.mark.asyncio
    async def test_set_feature(self, client_with_sae):
        """POST /api/steering/features sets feature."""
        response = await client_with_sae.post(
            "/api/steering/features",
            json={"feature_index": 100, "value": 5.0}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["feature_index"] == 100
        assert data["value"] == 5.0

    @pytest.mark.asyncio
    async def test_validation_error(self, client_with_sae):
        """Should return 400 for invalid value."""
        response = await client_with_sae.post(
            "/api/steering/features",
            json={"feature_index": 100, "value": 15.0}  # Out of range
        )

        assert response.status_code == 422  # Pydantic validation
```

---

## 10. Common Patterns

### DO: Validate Before Mutating

```python
# Correct: Validate all, then apply
async def batch_set(self, steering):
    for index in steering:
        self._validate_feature_index(index)  # Validate ALL first

    self._sae.set_steering_batch(steering)  # Then apply

# Incorrect: Apply one by one
async def batch_set(self, steering):
    for index, value in steering.items():
        self._sae.set_steering(index, value)  # May partially apply before error
```

### DO: Round Values Consistently

```python
# Correct: Round in service layer
async def set_feature(self, index, value):
    value = round(value, 1)
    self._sae.set_steering(index, value)

# Also in schema (belt and suspenders)
@field_validator("value")
def round_value(cls, v):
    return round(v, 1)
```

### DON'T: Expose Internal State Directly

```python
# Incorrect: Direct access to internal dict
@router.get("/features")
async def get_features():
    return steering._sae._steering_values  # Exposes internal state!

# Correct: Use accessor methods
@router.get("/features")
async def get_features():
    return steering._sae.get_steering_values()  # Returns copy
```

---

**Document Status:** Complete
**Next Document:** `004_FTASKS|Feature_Steering.md` (Task List)
**Instruction File:** `@0xcc/instruct/006_generate-tasks.md`
