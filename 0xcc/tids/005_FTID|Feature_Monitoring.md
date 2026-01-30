# Technical Implementation Document: Feature Monitoring

## miLLM Feature 5

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `005_FPRD|Feature_Monitoring.md`
- Feature TDD: `005_FTDD|Feature_Monitoring.md`

---

## 1. Overview

Feature Monitoring provides real-time visibility into SAE feature activations. This document provides implementation guidance building on LoadedSAE monitoring capabilities from Feature 3.

---

## 2. File Structure

```
millm/
├── api/
│   ├── routes/
│   │   └── management/
│   │       └── monitoring.py            # Monitoring API routes
│   └── schemas/
│       └── monitoring.py                # Pydantic schemas
│
├── services/
│   └── monitoring_service.py            # Monitoring service with history/stats
│
└── sockets/
    └── events.py                        # Add monitoring events
```

```
tests/
├── unit/
│   └── services/
│       └── test_monitoring_service.py   # Service unit tests
└── integration/
    └── api/
        └── test_monitoring_routes.py    # API tests
```

---

## 3. Pydantic Schemas

```python
# millm/api/schemas/monitoring.py

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class ConfigureMonitoringRequest(BaseModel):
    """Request to configure monitoring."""
    features: List[int] = Field(
        default_factory=list,
        description="Feature indices to monitor"
    )
    enabled: bool = Field(default=True)


class ToggleMonitoringRequest(BaseModel):
    """Request to enable/disable monitoring."""
    enabled: bool


class FeatureStatistics(BaseModel):
    """Statistics for a single feature."""
    min: float
    max: float
    mean: float
    count: int


class MonitoringState(BaseModel):
    """Current monitoring state."""
    enabled: bool
    monitored_features: List[int]
    history_size: int
    statistics: Dict[str, FeatureStatistics]
    sae_id: Optional[str]


class ActivationRecord(BaseModel):
    """Single activation record."""
    timestamp: datetime
    request_id: str
    features: Dict[str, float]  # String keys for JSON
    position: Optional[int]
    activation_type: str


class ActivationHistoryResponse(BaseModel):
    """Activation history response."""
    records: List[ActivationRecord]
    total_count: int


class ClearResponse(BaseModel):
    """Response for clear operations."""
    cleared: bool
    cleared_count: int
```

---

## 4. Monitoring Service Implementation

```python
# millm/services/monitoring_service.py

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import time
import logging

from millm.services.sae_service import SAEService

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Running statistics for a feature."""
    min_val: float = float('inf')
    max_val: float = float('-inf')
    sum_val: float = 0.0
    count: int = 0

    def update(self, value: float):
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.sum_val += value
        self.count += 1

    def to_dict(self) -> Dict[str, float]:
        return {
            "min": self.min_val if self.count > 0 else 0,
            "max": self.max_val if self.count > 0 else 0,
            "mean": self.sum_val / self.count if self.count > 0 else 0,
            "count": self.count,
        }


@dataclass
class ActivationEntry:
    """Single activation history entry."""
    timestamp: datetime
    request_id: str
    activations: Dict[int, float]
    position: Optional[int]
    activation_type: str


class MonitoringService:
    """
    Service for feature activation monitoring.

    Manages:
    - Feature selection for monitoring
    - Activation history (ring buffer)
    - Running statistics
    - WebSocket event throttling
    """

    def __init__(
        self,
        sae_service: SAEService,
        max_history: int = 100,
        max_updates_per_second: int = 10,
    ):
        self._sae_service = sae_service
        self._max_history = max_history

        # History buffer
        self._history: deque[ActivationEntry] = deque(maxlen=max_history)

        # Statistics
        self._stats: Dict[int, FeatureStats] = {}

        # Throttling
        self._emit_interval = 1.0 / max_updates_per_second
        self._last_emit_time = 0.0

        # State
        self._enabled = False
        self._monitored_features: List[int] = []

    @property
    def _sae(self):
        return self._sae_service._loaded_sae

    async def configure(
        self,
        features: List[int],
        enabled: bool = True,
    ):
        """Configure monitoring settings."""
        self._monitored_features = features
        self._enabled = enabled

        if self._sae:
            self._sae.enable_monitoring(enabled, features=features if features else None)

        logger.info(f"Monitoring configured: enabled={enabled}, features={len(features)}")
        return await self.get_state()

    async def set_enabled(self, enabled: bool):
        """Enable or disable monitoring."""
        self._enabled = enabled
        if self._sae:
            self._sae.enable_monitoring(enabled, features=self._monitored_features or None)
        return await self.get_state()

    async def get_state(self):
        """Get current monitoring state."""
        return {
            "enabled": self._enabled,
            "monitored_features": self._monitored_features,
            "history_size": len(self._history),
            "statistics": {
                str(k): v.to_dict()
                for k, v in self._stats.items()
            },
            "sae_id": self._sae_service._attached_sae_id,
        }

    def on_activation(
        self,
        activations: Dict[int, float],
        request_id: str,
        position: Optional[int] = None,
        activation_type: str = "token",
    ):
        """
        Called when new activations captured.

        This is called from inference path - keep it fast!
        """
        if not self._enabled:
            return

        # Add to history
        entry = ActivationEntry(
            timestamp=datetime.utcnow(),
            request_id=request_id,
            activations=activations,
            position=position,
            activation_type=activation_type,
        )
        self._history.append(entry)

        # Update statistics
        for idx, value in activations.items():
            if idx not in self._stats:
                self._stats[idx] = FeatureStats()
            self._stats[idx].update(value)

        # Emit (throttled)
        self._maybe_emit(activations, request_id, position, activation_type)

    def _maybe_emit(
        self,
        activations: Dict[int, float],
        request_id: str,
        position: Optional[int],
        activation_type: str,
    ):
        """Emit WebSocket update if not throttled."""
        now = time.time()
        if now - self._last_emit_time >= self._emit_interval:
            try:
                from millm.sockets.events import emit_activation_update
                emit_activation_update(
                    features=activations,
                    request_id=request_id,
                    position=position,
                    activation_type=activation_type,
                )
            except ImportError:
                pass
            self._last_emit_time = now

    async def get_history(self) -> List[dict]:
        """Get activation history."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "request_id": e.request_id,
                "features": {str(k): v for k, v in e.activations.items()},
                "position": e.position,
                "activation_type": e.activation_type,
            }
            for e in self._history
        ]

    async def clear_history(self) -> int:
        """Clear activation history."""
        count = len(self._history)
        self._history.clear()
        logger.info(f"Cleared {count} history entries")
        return count

    async def get_statistics(self) -> Dict[str, dict]:
        """Get feature statistics."""
        return {
            str(k): v.to_dict()
            for k, v in self._stats.items()
        }

    async def reset_statistics(self) -> int:
        """Reset statistics."""
        count = len(self._stats)
        self._stats.clear()
        logger.info(f"Reset statistics for {count} features")
        return count
```

---

## 5. API Routes Implementation

```python
# millm/api/routes/management/monitoring.py

from fastapi import APIRouter, Depends

from millm.api.schemas.monitoring import (
    ConfigureMonitoringRequest,
    ToggleMonitoringRequest,
    MonitoringState,
    ActivationHistoryResponse,
    ClearResponse,
)
from millm.services.monitoring_service import MonitoringService
from millm.api.dependencies import get_monitoring_service

router = APIRouter(prefix="/api/monitoring", tags=["Feature Monitoring"])


@router.get("", response_model=MonitoringState)
async def get_monitoring_state(
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Get current monitoring state."""
    return await monitoring.get_state()


@router.post("/configure", response_model=MonitoringState)
async def configure_monitoring(
    request: ConfigureMonitoringRequest,
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Configure monitoring settings."""
    return await monitoring.configure(
        features=request.features,
        enabled=request.enabled,
    )


@router.post("/enable", response_model=MonitoringState)
async def toggle_monitoring(
    request: ToggleMonitoringRequest,
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Enable or disable monitoring."""
    return await monitoring.set_enabled(request.enabled)


@router.get("/history", response_model=ActivationHistoryResponse)
async def get_history(
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Get activation history."""
    records = await monitoring.get_history()
    return ActivationHistoryResponse(
        records=records,
        total_count=len(records),
    )


@router.delete("/history", response_model=ClearResponse)
async def clear_history(
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Clear activation history."""
    count = await monitoring.clear_history()
    return ClearResponse(cleared=True, cleared_count=count)


@router.get("/statistics")
async def get_statistics(
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Get feature statistics."""
    return await monitoring.get_statistics()


@router.delete("/statistics", response_model=ClearResponse)
async def reset_statistics(
    monitoring: MonitoringService = Depends(get_monitoring_service),
):
    """Reset statistics."""
    count = await monitoring.reset_statistics()
    return ClearResponse(cleared=True, cleared_count=count)
```

---

## 6. WebSocket Events

```python
# millm/sockets/events.py (add)

def emit_activation_update(
    features: Dict[int, float],
    request_id: str,
    position: Optional[int],
    activation_type: str,
):
    """Emit activation update to monitoring clients."""
    socket_manager.emit(
        "activation_update",
        {
            "timestamp": datetime.utcnow().isoformat(),
            "features": {str(k): v for k, v in features.items()},
            "request_id": request_id,
            "position": position,
            "type": activation_type,
        },
        room="monitoring",
    )
```

---

## 7. Integration with InferenceService

```python
# millm/services/inference_service.py (modification)

class InferenceService:
    def __init__(self, ..., monitoring_service: MonitoringService = None):
        self._monitoring_service = monitoring_service

    async def _after_generation(self, request_id: str):
        """Called after generation to capture monitoring data."""
        if self._monitoring_service and self._sae:
            activations = self._sae.get_last_feature_activations()
            if activations is not None:
                # Convert tensor to dict
                act_dict = self._tensor_to_dict(activations)
                self._monitoring_service.on_activation(
                    activations=act_dict,
                    request_id=request_id,
                    activation_type="mean",
                )
```

---

## 8. Testing Patterns

```python
# tests/unit/services/test_monitoring_service.py

class TestMonitoringService:
    def test_history_ring_buffer(self):
        """History should act as ring buffer."""
        service = MonitoringService(mock_sae_service, max_history=3)

        for i in range(5):
            service.on_activation({0: float(i)}, f"req-{i}")

        history = service._history
        assert len(history) == 3
        assert history[0].activations[0] == 2.0  # Oldest

    def test_statistics_running_computation(self):
        """Statistics should track min/max/mean."""
        service = MonitoringService(mock_sae_service)

        service.on_activation({0: 1.0}, "req-1")
        service.on_activation({0: 3.0}, "req-2")
        service.on_activation({0: 5.0}, "req-3")

        stats = service._stats[0].to_dict()
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["count"] == 3

    def test_throttling(self):
        """Should throttle WebSocket emissions."""
        service = MonitoringService(mock_sae_service, max_updates_per_second=10)

        emit_count = 0
        # Rapid fire activations
        for i in range(100):
            service.on_activation({0: float(i)}, f"req-{i}")

        # Should have emitted << 100 times due to throttling
```

---

**Document Status:** Complete
**Next Document:** `005_FTASKS|Feature_Monitoring.md`
