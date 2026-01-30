# Task List: Feature Monitoring

## miLLM Feature 5

**Document Version:** 1.1
**Created:** January 30, 2026
**Status:** Complete - Core Implementation

---

## Relevant Files

### Backend
- `millm/api/routes/management/monitoring.py` - Monitoring API routes ✅
- `millm/api/schemas/monitoring.py` - Pydantic schemas ✅
- `millm/services/monitoring_service.py` - Monitoring service ✅
- `millm/sockets/progress.py` - Activation events ✅
- `millm/api/dependencies.py` - MonitoringServiceDep ✅

### Tests
- `tests/unit/services/test_monitoring_service.py` - Service tests ✅
- `tests/integration/api/test_monitoring_routes.py` - API tests (pending)

---

## Tasks

### Phase 1: Core Service

- [x] 1.0 Create monitoring schemas
  - [x] 1.1 Create `millm/api/schemas/monitoring.py`
  - [x] 1.2 Implement ConfigureMonitoringRequest
  - [x] 1.3 Implement ToggleMonitoringRequest
  - [x] 1.4 Implement FeatureStatistics
  - [x] 1.5 Implement MonitoringState
  - [x] 1.6 Implement ActivationRecord
  - [x] 1.7 Implement ActivationHistoryResponse
  - [x] 1.8 Implement ClearResponse
  - [x] 1.9 Implement TopFeaturesRequest/Response

- [x] 2.0 Create monitoring service
  - [x] 2.1 Create `millm/services/monitoring_service.py`
  - [x] 2.2 Implement FeatureStats dataclass
  - [x] 2.3 Implement ActivationEntry dataclass
  - [x] 2.4 Implement MonitoringService class
  - [x] 2.5 Implement history ring buffer with deque
  - [x] 2.6 Implement statistics tracking (mean, std, min, max, active_ratio)
  - [x] 2.7 Implement throttled WebSocket emission
  - [x] 2.8 Implement configure() method
  - [x] 2.9 Implement on_activation() method
  - [x] 2.10 Implement get_history() and clear_history()
  - [x] 2.11 Implement get_statistics() and reset_statistics()
  - [x] 2.12 Implement get_top_features() method

### Phase 2: API Routes

- [x] 3.0 Create monitoring routes
  - [x] 3.1 Create `millm/api/routes/management/monitoring.py`
  - [x] 3.2 Implement GET /api/monitoring
  - [x] 3.3 Implement POST /api/monitoring/configure
  - [x] 3.4 Implement POST /api/monitoring/enable
  - [x] 3.5 Implement GET /api/monitoring/history
  - [x] 3.6 Implement DELETE /api/monitoring/history
  - [x] 3.7 Implement GET /api/monitoring/statistics
  - [x] 3.8 Implement DELETE /api/monitoring/statistics
  - [x] 3.9 Implement POST /api/monitoring/statistics/top
  - [x] 3.10 Mount router in main app

### Phase 3: WebSocket Events

- [x] 4.0 Add monitoring WebSocket events
  - [x] 4.1 Add emit_activation_update to progress.py
  - [x] 4.2 Include timestamp, features, request_id, position
  - [x] 4.3 Add emit_monitoring_state_changed event

### Phase 4: Integration

- [x] 5.0 Integration setup
  - [x] 5.1 Add MonitoringServiceDep to dependencies.py
  - [x] 5.2 Export MonitoringService in services/__init__.py
  - [ ] 5.3 Full inference pipeline integration (future enhancement)

### Phase 5: Testing

- [x] 6.0 Unit tests
  - [x] 6.1 Test FeatureStats calculations (mean, std, min, max, active_ratio)
  - [x] 6.2 Test history ring buffer behavior
  - [x] 6.3 Test statistics tracking
  - [x] 6.4 Test throttling logic
  - [x] 6.5 Test configuration
  - [x] 6.6 Test get_top_features

- [ ] 7.0 Integration tests (deferred)
  - [ ] 7.1 Test API endpoints with TestClient
  - [ ] 7.2 Test with mock SAE
  - [ ] 7.3 Test history accumulation

---

## Implementation Notes

### Monitoring Flow

1. **Configure monitoring** via `/api/monitoring/configure`
2. **SAE captures activations** in LoadedSAE forward pass
3. **MonitoringService.on_activation()** records to history and updates statistics
4. **WebSocket events** emit real-time updates (throttled)
5. **Query history/statistics** via API endpoints

### Architecture Decisions

- **Singleton service**: MonitoringService is a singleton to preserve state across requests
- **Throttled events**: WebSocket emission throttled to prevent flooding (default 100ms)
- **Ring buffer history**: Fixed-size deque prevents unbounded memory growth
- **Running statistics**: Welford's algorithm for stable variance computation

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/monitoring` | GET | Get monitoring state |
| `/api/monitoring/configure` | POST | Configure monitoring |
| `/api/monitoring/enable` | POST | Enable/disable |
| `/api/monitoring/history` | GET | Get activation history |
| `/api/monitoring/history` | DELETE | Clear history |
| `/api/monitoring/statistics` | GET | Get feature statistics |
| `/api/monitoring/statistics` | DELETE | Reset statistics |
| `/api/monitoring/statistics/top` | POST | Get top features by metric |

---

**Document Status:** Complete
**Total Tasks:** 7 parent tasks, 40+ sub-tasks
**Next Feature:** Feature 6 - Profile Management
