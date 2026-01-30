# Technical Design Document: Feature Monitoring

## miLLM Feature 5

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `005_FPRD|Feature_Monitoring.md`
- ADR: `000_PADR|miLLM.md`

---

## 1. Executive Summary

Feature Monitoring provides real-time visibility into SAE feature activations during inference. The design leverages the monitoring capabilities already built into LoadedSAE (Feature 3), adding a service layer, WebSocket streaming, and history management.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Protocol | WebSocket for updates | Real-time, bidirectional |
| Throttling | Max 10 updates/sec | Prevent UI flooding |
| Storage | In-memory ring buffer | Fast access, bounded memory |
| Statistics | Running computation | O(1) memory for stats |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Admin UI                                    │
│    [Feature List]  [Activation Chart]  [Statistics Panel]       │
└────────────────────────────┬────────────────────────────────────┘
                             │ WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MonitoringService                           │
│  - configure()        - get_state()        - clear_history()    │
│  - emit_activations() - compute_statistics()                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   LoadedSAE     │ │  HistoryBuffer  │ │ StatisticsTracker│
│ _monitoring_    │ │ Ring buffer of  │ │ Running min/max │
│ _features[]     │ │ activations     │ │ mean computation│
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Data Flow: Activation Capture

```
Model forward pass triggers SAE hook
        │
        ▼
LoadedSAE.forward() captures activations
        │ (only if monitoring enabled)
        ▼
MonitoringService.on_activation() called
        │
        ├──► Add to HistoryBuffer
        │
        ├──► Update StatisticsTracker
        │
        └──► Emit via WebSocket (throttled)
```

---

## 3. Component Design

### MonitoringService

```python
# millm/services/monitoring_service.py

class MonitoringService:
    """Service for feature activation monitoring."""

    def __init__(self, sae_service: SAEService):
        self._sae_service = sae_service
        self._history = ActivationHistory(max_size=100)
        self._statistics = StatisticsTracker()
        self._last_emit_time = 0
        self._emit_interval = 0.1  # 100ms = 10/sec max

    async def configure(
        self,
        features: List[int],
        enabled: bool = True,
    ) -> MonitoringState:
        """Configure monitoring settings."""
        sae = self._sae_service._loaded_sae
        if sae:
            sae.enable_monitoring(enabled, features=features)
        return await self.get_state()

    def on_activation(
        self,
        activations: Dict[int, float],
        request_id: str,
        position: Optional[int] = None,
        activation_type: str = "token",
    ):
        """Called when new activations captured."""
        # Add to history
        self._history.add(activations, request_id, position, activation_type)

        # Update statistics
        self._statistics.update(activations)

        # Emit via WebSocket (throttled)
        self._maybe_emit(activations, request_id, position, activation_type)

    def _maybe_emit(self, activations, request_id, position, activation_type):
        """Emit update if not throttled."""
        now = time.time()
        if now - self._last_emit_time >= self._emit_interval:
            emit_activation_update(
                features=activations,
                request_id=request_id,
                position=position,
                activation_type=activation_type,
            )
            self._last_emit_time = now
```

### ActivationHistory

```python
# millm/services/monitoring_service.py

class ActivationHistory:
    """Ring buffer for activation history."""

    def __init__(self, max_size: int = 100):
        self._max_size = max_size
        self._buffer: List[ActivationRecord] = []

    def add(
        self,
        activations: Dict[int, float],
        request_id: str,
        position: Optional[int],
        activation_type: str,
    ):
        record = ActivationRecord(
            timestamp=datetime.utcnow(),
            activations=activations,
            request_id=request_id,
            position=position,
            activation_type=activation_type,
        )
        self._buffer.append(record)
        if len(self._buffer) > self._max_size:
            self._buffer.pop(0)

    def get_all(self) -> List[ActivationRecord]:
        return list(self._buffer)

    def clear(self) -> int:
        count = len(self._buffer)
        self._buffer.clear()
        return count
```

### StatisticsTracker

```python
# millm/services/monitoring_service.py

class StatisticsTracker:
    """Running statistics for monitored features."""

    def __init__(self):
        self._stats: Dict[int, FeatureStats] = {}

    def update(self, activations: Dict[int, float]):
        for feature_idx, value in activations.items():
            if feature_idx not in self._stats:
                self._stats[feature_idx] = FeatureStats()
            self._stats[feature_idx].update(value)

    def get_stats(self) -> Dict[int, Dict[str, float]]:
        return {
            idx: stats.to_dict()
            for idx, stats in self._stats.items()
        }

    def reset(self):
        self._stats.clear()


@dataclass
class FeatureStats:
    """Running statistics for a single feature."""
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
```

---

## 4. Integration with LoadedSAE

The LoadedSAE class (Feature 3) already has monitoring methods:

```python
# In LoadedSAE.forward()
if self._monitoring_enabled:
    self._capture_activations(feature_acts)

# Methods available:
- enable_monitoring(enabled, features=None)
- get_last_feature_activations()
- _capture_activations(feature_acts)
```

MonitoringService hooks into this by:
1. Configuring LoadedSAE with features to monitor
2. Polling or being called with activations after inference
3. Processing and distributing activation data

---

## 5. API Design

### Routes

```python
# millm/api/routes/management/monitoring.py

router = APIRouter(prefix="/api/monitoring", tags=["Feature Monitoring"])

@router.get("", response_model=MonitoringState)
async def get_monitoring_state(...)

@router.post("/configure", response_model=MonitoringState)
async def configure_monitoring(request: ConfigureMonitoringRequest, ...)

@router.post("/enable", response_model=MonitoringState)
async def toggle_monitoring(request: ToggleRequest, ...)

@router.get("/history", response_model=ActivationHistoryResponse)
async def get_history(...)

@router.delete("/history")
async def clear_history(...)

@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(...)

@router.delete("/statistics")
async def reset_statistics(...)
```

### WebSocket Events

```python
# millm/sockets/events.py

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
            "features": features,
            "request_id": request_id,
            "position": position,
            "type": activation_type,
        },
        room="monitoring",
    )
```

---

## 6. Performance Considerations

### Throttling Strategy

```python
class ThrottledEmitter:
    """Emit updates at controlled rate."""

    def __init__(self, max_per_second: int = 10):
        self._interval = 1.0 / max_per_second
        self._last_emit = 0
        self._pending: Optional[Dict] = None

    def maybe_emit(self, data: Dict):
        now = time.time()
        if now - self._last_emit >= self._interval:
            self._emit(data)
            self._last_emit = now
        else:
            self._pending = data  # Store for next window

    def flush(self):
        """Emit pending data."""
        if self._pending:
            self._emit(self._pending)
            self._pending = None
```

### Selective Capture

```python
# Only copy selected features, not full tensor
def _capture_activations(self, feature_acts: Tensor):
    if self._monitored_features:
        # Index only selected features
        selected = feature_acts[..., self._monitored_features]
        self._last_feature_acts = selected.detach().cpu()
    else:
        # Full capture (expensive!)
        self._last_feature_acts = feature_acts.detach().cpu()
```

---

## 7. Testing Strategy

### Unit Tests
- StatisticsTracker running computation
- ActivationHistory ring buffer behavior
- Throttling logic
- Configuration validation

### Integration Tests
- Full monitoring flow with mock SAE
- WebSocket event emission
- History accumulation and clearing
- Statistics accuracy

---

**Document Status:** Complete
**Next Document:** `005_FTID|Feature_Monitoring.md`
