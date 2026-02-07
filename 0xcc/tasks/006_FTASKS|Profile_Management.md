# Task List: Profile Management

## miLLM Feature 6

**Document Version:** 1.1
**Created:** January 30, 2026
**Status:** Complete - Core Implementation Done
**References:**
- Feature PRD: `006_FPRD|Profile_Management.md`
- Feature TDD: `006_FTDD|Profile_Management.md`
- Feature TID: `006_FTID|Profile_Management.md`

---

## Implementation Notes

Feature 6 (Profile Management) implements the core profile CRUD operations, activation/deactivation with steering application, and save-current-steering functionality. Export/import functionality is deferred to a future enhancement.

### What Was Implemented

**Database Layer:**
- `millm/db/models/profile.py` - Profile SQLAlchemy model with JSONB steering storage
- `millm/db/migrations/versions/003_create_profiles_table.py` - Alembic migration
- `millm/db/repositories/profile_repository.py` - Full CRUD operations

**API Layer:**
- `millm/api/schemas/profile.py` - All request/response schemas
- `millm/api/routes/management/profiles.py` - REST API endpoints
- `millm/api/dependencies.py` - ProfileServiceDep and ProfileRepo dependencies

**Service Layer:**
- `millm/services/profile_service.py` - Complete profile orchestration

**Error Handling:**
- `millm/core/errors.py` - ProfileAlreadyExistsError added (others existed)

**Testing:**
- `tests/unit/services/test_profile_service.py` - Comprehensive unit tests

---

## Relevant Files

### Backend - Database
- `millm/db/models/profile.py` - SQLAlchemy Profile model
- `millm/db/models/__init__.py` - Profile export added
- `millm/db/repositories/profile_repository.py` - Database operations
- `millm/db/repositories/__init__.py` - ProfileRepository export added
- `millm/db/migrations/versions/003_create_profiles_table.py` - Alembic migration

### Backend - API
- `millm/api/routes/management/profiles.py` - Profile API routes
- `millm/api/routes/__init__.py` - Router registration
- `millm/api/schemas/profile.py` - Pydantic schemas
- `millm/api/schemas/__init__.py` - Schema exports added
- `millm/api/dependencies.py` - ProfileServiceDep added

### Backend - Services
- `millm/services/profile_service.py` - Profile service
- `millm/services/__init__.py` - ProfileService export added

### Backend - Errors
- `millm/core/errors.py` - ProfileAlreadyExistsError added

### Tests
- `tests/unit/services/test_profile_service.py` - Service unit tests

---

## Tasks

### Phase 1: Database Layer

- [x] 1.0 Create Profile database model
  - [x] 1.1 Create `millm/db/models/profile.py`
  - [x] 1.2 Define Profile model with all fields
  - [x] 1.3 Add JSONB column for steering
  - [x] 1.4 Add partial unique index for active profile
  - [x] 1.5 Export in models __init__.py

- [x] 2.0 Create Profile migration
  - [x] 2.1 Create Alembic migration 003
  - [x] 2.2 Add profiles table with all columns
  - [x] 2.3 Add indexes (name index, partial unique for active)
  - [x] 2.4 Include downgrade for rollback

- [x] 3.0 Create Profile repository
  - [x] 3.1 Create `millm/db/repositories/profile_repository.py`
  - [x] 3.2 Implement get_all, get, get_by_name
  - [x] 3.3 Implement get_active
  - [x] 3.4 Implement create, update, delete
  - [x] 3.5 Implement deactivate_all, set_active
  - [x] 3.6 Implement name_exists for validation

### Phase 2: Schemas

- [x] 4.0 Create Pydantic schemas
  - [x] 4.1 Create `millm/api/schemas/profile.py`
  - [x] 4.2 Implement CreateProfileRequest
  - [x] 4.3 Implement UpdateProfileRequest
  - [x] 4.4 Implement ProfileResponse with from_profile
  - [x] 4.5 Implement ProfileListResponse
  - [x] 4.6 Implement ActivateProfileRequest/Response
  - [x] 4.7 Implement DeactivateProfileResponse
  - [x] 4.8 Implement SaveCurrentRequest
  - [x] 4.9 Implement DeleteProfileResponse

### Phase 3: Service Layer

- [x] 5.0 Create Profile service
  - [x] 5.1 Create `millm/services/profile_service.py`
  - [x] 5.2 Implement list_profiles
  - [x] 5.3 Implement get_profile, get_active_profile
  - [x] 5.4 Implement create_profile
  - [x] 5.5 Implement save_current_steering
  - [x] 5.6 Implement update_profile
  - [x] 5.7 Implement delete_profile
  - [x] 5.8 Implement activate_profile (with steering application)
  - [x] 5.9 Implement deactivate_profile (with steering clear)

### Phase 4: API Routes

- [x] 6.0 Create Profile routes
  - [x] 6.1 Create `millm/api/routes/management/profiles.py`
  - [x] 6.2 Implement GET /api/profiles (list all)
  - [x] 6.3 Implement POST /api/profiles (create)
  - [x] 6.4 Implement POST /api/profiles/save-current (save steering)
  - [x] 6.5 Implement GET /api/profiles/active (get active)
  - [x] 6.6 Implement GET /api/profiles/{id} (get by ID)
  - [x] 6.7 Implement PATCH /api/profiles/{id} (update)
  - [x] 6.8 Implement DELETE /api/profiles/{id} (delete)
  - [x] 6.9 Implement POST /api/profiles/{id}/activate
  - [x] 6.10 Implement POST /api/profiles/{id}/deactivate
  - [x] 6.11 Mount router in routes/__init__.py
  - [x] 6.12 Add ProfileServiceDep in dependencies.py

### Phase 5: Error Handling

- [x] 7.0 Add profile errors
  - [x] 7.1 ProfileNotFoundError (existed)
  - [x] 7.2 ProfileCompatibilityError (existed)
  - [x] 7.3 InvalidProfileFormatError (existed)
  - [x] 7.4 ProfileAlreadyExistsError (added)

### Phase 6: Testing

- [x] 8.0 Unit tests
  - [x] 8.1 Test profile list operations
  - [x] 8.2 Test profile CRUD operations
  - [x] 8.3 Test save current steering
  - [x] 8.4 Test activation/deactivation logic
  - [x] 8.5 Test error cases

---

## Previously Deferred Tasks (Now Complete)

### Export/Import
- [x] Implement export_profile (JSON response with profile data)
- [x] Implement import_profile with validation (uses ProfileImportRequest BaseModel)
- [x] GET /api/profiles/{id}/export endpoint
- [x] POST /api/profiles/import endpoint
- [x] ProfileExportData schema (in routes/management/profiles.py)

### Inference Integration
- [x] Support profile parameter in ChatCompletionRequest
- [x] Apply profile steering for request (in chat.py route handler)
- [x] Steering applied via sae.set_steering_batch() + sae.enable_steering(True)

---

## API Endpoint Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/profiles | List all profiles |
| POST | /api/profiles | Create new profile |
| POST | /api/profiles/save-current | Save current steering as profile |
| GET | /api/profiles/active | Get currently active profile |
| GET | /api/profiles/{id} | Get profile by ID |
| PATCH | /api/profiles/{id} | Update profile |
| DELETE | /api/profiles/{id} | Delete profile |
| POST | /api/profiles/{id}/activate | Activate profile |
| POST | /api/profiles/{id}/deactivate | Deactivate profile |

---

**Document Status:** Complete
**Implementation Location:** Feature 6
**Next Feature:** Feature 7 or enhancement
