# Task List: Feature Steering

## miLLM Feature 4

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft
**References:**
- Feature PRD: `004_FPRD|Feature_Steering.md`
- Feature TDD: `004_FTDD|Feature_Steering.md`
- Feature TID: `004_FTID|Feature_Steering.md`

---

## Relevant Files

### Backend - API

- `millm/api/routes/management/steering.py` - Steering API routes
- `millm/api/schemas/steering.py` - Pydantic schemas for steering

### Backend - Services

- `millm/services/steering_service.py` - Steering service layer

### Backend - WebSocket

- `millm/sockets/events.py` - Add steering event emitters

### Backend - Errors

- `millm/core/errors.py` - Add steering-specific error classes

### Backend - Dependencies

- `millm/api/dependencies.py` - Add get_steering_service
- `millm/main.py` - Initialize SteeringService in lifespan

### Tests - Unit

- `tests/unit/services/test_steering_service.py` - Service unit tests
- `tests/unit/api/test_steering_schemas.py` - Schema validation tests

### Tests - Integration

- `tests/integration/api/test_steering_routes.py` - API endpoint tests

### Notes

- Feature Steering builds on LoadedSAE from Feature 3
- LoadedSAE already has steering methods (set_steering, clear_steering, etc.)
- This feature adds service layer and API exposure
- No database persistence - state lives in memory on LoadedSAE

---

## Tasks

### Phase 1: Error Classes

- [ ] 1.0 Add steering-specific error classes
  - [ ] 1.1 Add SteeringError base class to `millm/core/errors.py`
  - [ ] 1.2 Add NoSAEAttachedError class
  - [ ] 1.3 Add InvalidFeatureIndexError with index and max_index
  - [ ] 1.4 Add InvalidSteeringValueError with value and range

### Phase 2: Pydantic Schemas

- [ ] 2.0 Create steering schemas
  - [ ] 2.1 Create `millm/api/schemas/steering.py` file
  - [ ] 2.2 Implement SetFeatureRequest with feature_index and value
  - [ ] 2.3 Add value validator to round to 1 decimal
  - [ ] 2.4 Implement BatchSetRequest with list of SetFeatureRequest
  - [ ] 2.5 Add max_length=100 constraint to batch
  - [ ] 2.6 Implement ToggleRequest with enabled boolean
  - [ ] 2.7 Implement SteeringState response schema
  - [ ] 2.8 Add from_sae class method for conversion
  - [ ] 2.9 Add to_integer_values helper for reverse conversion
  - [ ] 2.10 Implement SetFeatureResponse schema
  - [ ] 2.11 Implement ClearResponse schema

### Phase 3: Steering Service

- [ ] 3.0 Create steering service
  - [ ] 3.1 Create `millm/services/steering_service.py` file
  - [ ] 3.2 Implement SteeringService class with sae_service dependency
  - [ ] 3.3 Add _sae property for accessing loaded SAE
  - [ ] 3.4 Add _sae_id property for SAE ID
  - [ ] 3.5 Implement _require_sae validation helper
  - [ ] 3.6 Implement _validate_feature_index helper

- [ ] 4.0 Implement state methods
  - [ ] 4.1 Implement get_state() returning SteeringState
  - [ ] 4.2 Handle case when no SAE attached (return empty state)
  - [ ] 4.3 Convert SAE state to response schema

- [ ] 5.0 Implement steering control methods
  - [ ] 5.1 Implement set_enabled(enabled) method
  - [ ] 5.2 Call SAE enable_steering method
  - [ ] 5.3 Emit WebSocket event on change
  - [ ] 5.4 Return updated state

- [ ] 6.0 Implement feature manipulation methods
  - [ ] 6.1 Implement set_feature(index, value) method
  - [ ] 6.2 Validate feature index
  - [ ] 6.3 Round value to 1 decimal
  - [ ] 6.4 Clear feature if value is 0
  - [ ] 6.5 Call SAE set_steering for non-zero values
  - [ ] 6.6 Emit WebSocket event
  - [ ] 6.7 Return SetFeatureResponse

- [ ] 7.0 Implement batch operations
  - [ ] 7.1 Implement batch_set(steering) method
  - [ ] 7.2 Validate ALL indices before applying
  - [ ] 7.3 Round all values
  - [ ] 7.4 Call SAE set_steering_batch
  - [ ] 7.5 Emit batch event
  - [ ] 7.6 Return updated state

- [ ] 8.0 Implement clear operations
  - [ ] 8.1 Implement clear_feature(index) method
  - [ ] 8.2 Call SAE clear_steering with index
  - [ ] 8.3 Emit event
  - [ ] 8.4 Implement clear_all() method
  - [ ] 8.5 Get count before clearing
  - [ ] 8.6 Call SAE clear_steering with no args
  - [ ] 8.7 Emit cleared event with count
  - [ ] 8.8 Return ClearResponse

### Phase 4: WebSocket Events

- [ ] 9.0 Add steering WebSocket events
  - [ ] 9.1 Add emit_steering_changed function to `millm/sockets/events.py`
  - [ ] 9.2 Support enabled toggle event
  - [ ] 9.3 Support feature value change event
  - [ ] 9.4 Support feature removed event
  - [ ] 9.5 Support batch update event
  - [ ] 9.6 Support cleared event
  - [ ] 9.7 Send to "admin" room only

### Phase 5: API Routes

- [ ] 10.0 Create steering routes
  - [ ] 10.1 Create `millm/api/routes/management/steering.py` file
  - [ ] 10.2 Create router with prefix="/api/steering"
  - [ ] 10.3 Add handle_steering_error helper function

- [ ] 11.0 Implement route handlers
  - [ ] 11.1 Implement GET /api/steering (get_steering_state)
  - [ ] 11.2 Implement POST /api/steering/enable (toggle_steering)
  - [ ] 11.3 Implement POST /api/steering/features (set_feature)
  - [ ] 11.4 Implement POST /api/steering/features/batch (batch_set_features)
  - [ ] 11.5 Implement DELETE /api/steering/features/{index} (remove_feature)
  - [ ] 11.6 Implement DELETE /api/steering/features (clear_all_features)

### Phase 6: Dependency Setup

- [ ] 12.0 Configure dependencies
  - [ ] 12.1 Add get_steering_service to `millm/api/dependencies.py`
  - [ ] 12.2 Update main.py lifespan to create SteeringService
  - [ ] 12.3 Pass sae_service to SteeringService constructor
  - [ ] 12.4 Store in app.state.steering_service
  - [ ] 12.5 Mount steering router in main app

### Phase 7: Unit Tests

- [ ] 13.0 Create schema tests
  - [ ] 13.1 Create `tests/unit/api/test_steering_schemas.py`
  - [ ] 13.2 Test SetFeatureRequest validation
  - [ ] 13.3 Test value rounding
  - [ ] 13.4 Test BatchSetRequest max_length
  - [ ] 13.5 Test SteeringState.from_sae conversion

- [ ] 14.0 Create service tests
  - [ ] 14.1 Create `tests/unit/services/test_steering_service.py`
  - [ ] 14.2 Create mock_sae fixture
  - [ ] 14.3 Create mock_sae_service fixture
  - [ ] 14.4 Create steering_service fixture
  - [ ] 14.5 Test get_state with no SAE
  - [ ] 14.6 Test get_state with SAE
  - [ ] 14.7 Test set_feature requires SAE
  - [ ] 14.8 Test set_feature validates index
  - [ ] 14.9 Test set_feature calls SAE method
  - [ ] 14.10 Test set_feature clears on zero
  - [ ] 14.11 Test batch_set validates all first
  - [ ] 14.12 Test clear_all returns count

### Phase 8: Integration Tests

- [ ] 15.0 Create API route tests
  - [ ] 15.1 Create `tests/integration/api/test_steering_routes.py`
  - [ ] 15.2 Create client_with_sae fixture
  - [ ] 15.3 Test GET /api/steering returns state
  - [ ] 15.4 Test POST /api/steering/enable toggles
  - [ ] 15.5 Test POST /api/steering/features sets value
  - [ ] 15.6 Test validation errors return 422
  - [ ] 15.7 Test no SAE returns 400
  - [ ] 15.8 Test DELETE endpoints clear values

### Phase 9: Integration and Polish

- [ ] 16.0 Final integration
  - [ ] 16.1 Verify routes are accessible
  - [ ] 16.2 Test WebSocket events are emitted
  - [ ] 16.3 Test with actual SAE attached
  - [ ] 16.4 Verify steering affects inference output

- [ ] 17.0 Documentation and cleanup
  - [ ] 17.1 Add docstrings to all public methods
  - [ ] 17.2 Update OpenAPI documentation
  - [ ] 17.3 Review error messages for clarity
  - [ ] 17.4 Remove any debug code
  - [ ] 17.5 Run full test suite

---

## Notes

### Development Order Recommendation

1. Error classes (Task 1) - foundation for validation
2. Schemas (Task 2) - API contract definition
3. Service layer (Tasks 3-8) - business logic
4. WebSocket events (Task 9) - real-time updates
5. API routes (Tasks 10-11) - external interface
6. Dependencies (Task 12) - wiring it together
7. Tests (Tasks 13-15) - validation
8. Integration (Tasks 16-17) - polish

### Key Dependencies

- Requires SAE Management (Feature 3) to be complete
- LoadedSAE must have steering methods implemented
- SAE must be attached before steering operations

### Testing Notes

- Unit tests should mock SAEService and LoadedSAE
- Integration tests need actual SAE attached
- No real inference needed - just verify SAE methods called
- WebSocket events tested separately

### Performance Considerations

- Steering vector pre-computed on value change
- No performance impact when steering disabled
- Batch operations for UI efficiency

---

**Document Status:** Complete
**Total Tasks:** 17 parent tasks, 85+ sub-tasks
**Estimated Timeline:** 1 week
**Next Feature:** Feature 5 - Feature Monitoring
