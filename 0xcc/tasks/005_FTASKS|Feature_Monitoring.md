# Task List: Feature Monitoring

## miLLM Feature 5

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft

---

## Relevant Files

### Backend
- `millm/api/routes/management/monitoring.py` - Monitoring API routes
- `millm/api/schemas/monitoring.py` - Pydantic schemas
- `millm/services/monitoring_service.py` - Monitoring service
- `millm/sockets/events.py` - Add activation events
- `millm/api/dependencies.py` - Add get_monitoring_service

### Tests
- `tests/unit/services/test_monitoring_service.py` - Service tests
- `tests/integration/api/test_monitoring_routes.py` - API tests

---

## Tasks

### Phase 1: Core Service

- [ ] 1.0 Create monitoring schemas
  - [ ] 1.1 Create `millm/api/schemas/monitoring.py`
  - [ ] 1.2 Implement ConfigureMonitoringRequest
  - [ ] 1.3 Implement ToggleMonitoringRequest
  - [ ] 1.4 Implement FeatureStatistics
  - [ ] 1.5 Implement MonitoringState
  - [ ] 1.6 Implement ActivationRecord
  - [ ] 1.7 Implement ActivationHistoryResponse
  - [ ] 1.8 Implement ClearResponse

- [ ] 2.0 Create monitoring service
  - [ ] 2.1 Create `millm/services/monitoring_service.py`
  - [ ] 2.2 Implement FeatureStats dataclass
  - [ ] 2.3 Implement ActivationEntry dataclass
  - [ ] 2.4 Implement MonitoringService class
  - [ ] 2.5 Implement history ring buffer with deque
  - [ ] 2.6 Implement statistics tracking
  - [ ] 2.7 Implement throttled WebSocket emission
  - [ ] 2.8 Implement configure() method
  - [ ] 2.9 Implement on_activation() method
  - [ ] 2.10 Implement get_history() and clear_history()
  - [ ] 2.11 Implement get_statistics() and reset_statistics()

### Phase 2: API Routes

- [ ] 3.0 Create monitoring routes
  - [ ] 3.1 Create `millm/api/routes/management/monitoring.py`
  - [ ] 3.2 Implement GET /api/monitoring
  - [ ] 3.3 Implement POST /api/monitoring/configure
  - [ ] 3.4 Implement POST /api/monitoring/enable
  - [ ] 3.5 Implement GET /api/monitoring/history
  - [ ] 3.6 Implement DELETE /api/monitoring/history
  - [ ] 3.7 Implement GET /api/monitoring/statistics
  - [ ] 3.8 Implement DELETE /api/monitoring/statistics
  - [ ] 3.9 Mount router in main app

### Phase 3: WebSocket Events

- [ ] 4.0 Add monitoring WebSocket events
  - [ ] 4.1 Add emit_activation_update to events.py
  - [ ] 4.2 Include timestamp, features, request_id, position, type
  - [ ] 4.3 Send to "monitoring" room

### Phase 4: Integration

- [ ] 5.0 Integrate with inference
  - [ ] 5.1 Add monitoring_service to InferenceService
  - [ ] 5.2 Call on_activation after generation
  - [ ] 5.3 Convert tensor to dict for monitoring
  - [ ] 5.4 Add get_monitoring_service dependency
  - [ ] 5.5 Initialize in main.py lifespan

### Phase 5: Testing

- [ ] 6.0 Unit tests
  - [ ] 6.1 Test history ring buffer behavior
  - [ ] 6.2 Test statistics running computation
  - [ ] 6.3 Test throttling logic
  - [ ] 6.4 Test configuration

- [ ] 7.0 Integration tests
  - [ ] 7.1 Test API endpoints
  - [ ] 7.2 Test with mock SAE
  - [ ] 7.3 Test history accumulation

---

**Total Tasks:** 7 parent tasks, 45+ sub-tasks
**Estimated Timeline:** 3-4 days
