# Task List: Profile Management

## miLLM Feature 6

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft

---

## Relevant Files

### Backend - Database
- `millm/db/models/profile.py` - SQLAlchemy Profile model
- `millm/db/repositories/profile_repository.py` - Database operations
- `migrations/versions/xxx_add_profiles_table.py` - Alembic migration

### Backend - API
- `millm/api/routes/management/profiles.py` - Profile API routes
- `millm/api/schemas/profile.py` - Pydantic schemas

### Backend - Services
- `millm/services/profile_service.py` - Profile service

### Backend - Errors
- `millm/core/errors.py` - Add profile-specific errors

### Tests
- `tests/unit/services/test_profile_service.py` - Service tests
- `tests/integration/api/test_profile_routes.py` - API tests

---

## Tasks

### Phase 1: Database Layer

- [ ] 1.0 Create Profile database model
  - [ ] 1.1 Create `millm/db/models/profile.py`
  - [ ] 1.2 Define Profile model with all fields
  - [ ] 1.3 Add JSONB column for steering
  - [ ] 1.4 Add partial unique index for active profile
  - [ ] 1.5 Export in models __init__.py

- [ ] 2.0 Create Profile migration
  - [ ] 2.1 Generate Alembic migration
  - [ ] 2.2 Add profiles table
  - [ ] 2.3 Add indexes
  - [ ] 2.4 Test migration up/down

- [ ] 3.0 Create Profile repository
  - [ ] 3.1 Create `millm/db/repositories/profile_repository.py`
  - [ ] 3.2 Implement get_all, get, get_by_name
  - [ ] 3.3 Implement get_active
  - [ ] 3.4 Implement create, update, delete
  - [ ] 3.5 Implement deactivate_all, set_active

### Phase 2: Schemas

- [ ] 4.0 Create Pydantic schemas
  - [ ] 4.1 Create `millm/api/schemas/profile.py`
  - [ ] 4.2 Implement CreateProfileRequest
  - [ ] 4.3 Implement UpdateProfileRequest
  - [ ] 4.4 Implement ProfileResponse with from_model
  - [ ] 4.5 Implement ProfileListResponse
  - [ ] 4.6 Implement ProfileExport

### Phase 3: Service Layer

- [ ] 5.0 Create Profile service
  - [ ] 5.1 Create `millm/services/profile_service.py`
  - [ ] 5.2 Implement list_profiles
  - [ ] 5.3 Implement get_profile
  - [ ] 5.4 Implement create_profile (use current steering if none)
  - [ ] 5.5 Implement update_profile
  - [ ] 5.6 Implement delete_profile (prevent active deletion)
  - [ ] 5.7 Implement activate_profile (apply steering)
  - [ ] 5.8 Implement deactivate_profile
  - [ ] 5.9 Implement export_profile
  - [ ] 5.10 Implement import_profile with validation
  - [ ] 5.11 Implement _check_compatibility

### Phase 4: API Routes

- [ ] 6.0 Create Profile routes
  - [ ] 6.1 Create `millm/api/routes/management/profiles.py`
  - [ ] 6.2 Implement GET /api/profiles
  - [ ] 6.3 Implement POST /api/profiles
  - [ ] 6.4 Implement GET /api/profiles/{id}
  - [ ] 6.5 Implement PUT /api/profiles/{id}
  - [ ] 6.6 Implement DELETE /api/profiles/{id}
  - [ ] 6.7 Implement POST /api/profiles/{id}/activate
  - [ ] 6.8 Implement GET /api/profiles/{id}/export
  - [ ] 6.9 Implement POST /api/profiles/import
  - [ ] 6.10 Mount router in main app

### Phase 5: Error Handling

- [ ] 7.0 Add profile errors
  - [ ] 7.1 Add ProfileNotFoundError
  - [ ] 7.2 Add ProfileCompatibilityError
  - [ ] 7.3 Add InvalidProfileFormatError
  - [ ] 7.4 Register error handlers

### Phase 6: Integration

- [ ] 8.0 Integrate with inference
  - [ ] 8.1 Support profile parameter in ChatCompletionRequest
  - [ ] 8.2 Apply profile steering temporarily for request
  - [ ] 8.3 Restore original steering after request
  - [ ] 8.4 Add get_profile_service dependency
  - [ ] 8.5 Initialize in main.py lifespan

### Phase 7: Testing

- [ ] 9.0 Unit tests
  - [ ] 9.1 Test profile CRUD operations
  - [ ] 9.2 Test activation logic
  - [ ] 9.3 Test export/import format
  - [ ] 9.4 Test compatibility checking

- [ ] 10.0 Integration tests
  - [ ] 10.1 Test API endpoints
  - [ ] 10.2 Test profile activation with steering
  - [ ] 10.3 Test import/export round-trip

---

**Total Tasks:** 10 parent tasks, 55+ sub-tasks
**Estimated Timeline:** 1 week
