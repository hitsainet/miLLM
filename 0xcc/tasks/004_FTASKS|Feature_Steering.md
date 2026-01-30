# Task List: Feature Steering

## miLLM Feature 4

**Document Version:** 1.1
**Created:** January 30, 2026
**Status:** Complete - Implemented as part of Feature 3 (SAE Management)
**References:**
- Feature PRD: `004_FPRD|Feature_Steering.md`
- Feature TDD: `004_FTDD|Feature_Steering.md`
- Feature TID: `004_FTID|Feature_Steering.md`

---

## Implementation Notes

Feature 4 (Feature Steering) was implemented as an integrated part of Feature 3 (SAE Management) rather than as a separate module. This decision was made because:

1. **Tight Coupling:** Steering operates directly on LoadedSAE - no separate service needed
2. **Simpler Architecture:** One service (SAEService) handles attachment AND steering
3. **API Consistency:** All SAE-related operations under `/api/saes/*`

### What Was Implemented (in Feature 3)

**LoadedSAE Steering Methods:**
- `set_steering(feature_idx, value)` - Set single feature
- `set_steering_batch(steering)` - Set multiple features
- `clear_steering(feature_idx=None)` - Clear one or all
- `enable_steering(enabled)` - Toggle steering
- `get_steering_values()` - Get current values
- `is_steering_enabled` - Property

**SAEService Steering Delegation:**
- All steering methods delegate to attached LoadedSAE
- Thread-safe via attachment lock

**API Endpoints (at `/api/saes/steering`):**
- `GET /api/saes/steering` - Get current state
- `POST /api/saes/steering` - Set single feature
- `POST /api/saes/steering/batch` - Set multiple features
- `POST /api/saes/steering/enable` - Toggle enabled
- `DELETE /api/saes/steering/{feature_idx}` - Clear single feature
- `DELETE /api/saes/steering` - Clear all

**Pydantic Schemas:**
- `SteeringRequest` - Single feature request
- `SteeringBatchRequest` - Batch update request
- `SteeringStatus` - Response with enabled and values

**WebSocket Events (in `progress.py`):**
- Steering changes emitted via existing SAE event system

---

## Tasks (All Complete)

All tasks marked complete as functionality exists in Feature 3 implementation.

### Phase 1-9: All Complete

- [x] 1.0 Error classes - SAE errors cover steering validation
- [x] 2.0 Schemas - Implemented in `millm/api/schemas/sae.py`
- [x] 3.0-8.0 Service - Integrated into SAEService
- [x] 9.0 WebSocket events - In `millm/sockets/progress.py`
- [x] 10.0-11.0 API routes - In `millm/api/routes/management/saes.py`
- [x] 12.0 Dependencies - Handled by SAEService dependency
- [x] 13.0-15.0 Tests - Covered in SAE wrapper tests
- [x] 16.0-17.0 Integration - Complete

---

## API Path Difference

The PRD specified `/api/steering/*` but implementation uses `/api/saes/steering/*`.

**Rationale:** Steering is intrinsically tied to the attached SAE. Placing it under `/api/saes/` makes the relationship clear and keeps the API organized by resource (SAE operations, including steering, are grouped together).

---

**Document Status:** Complete
**Implementation Location:** Feature 3 (SAE Management)
**Next Feature:** Feature 5 - Feature Monitoring
