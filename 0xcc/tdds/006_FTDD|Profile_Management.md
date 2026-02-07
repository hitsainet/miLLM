# Technical Design Document: Profile Management

## miLLM Feature 6

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft

---

## 1. Executive Summary

Profile Management provides persistence and management for steering configurations. Profiles are stored in the database and can be activated to apply their steering values. The design supports API-based profile selection and import/export for sharing.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Storage | PostgreSQL with JSONB | Flexible steering storage |
| Active Profile | Partial unique index | Only one active at a time |
| Export Format | JSON with metadata | miStudio compatibility |
| API Selection | Profile override per-request | Flexible integration |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Admin UI                                    │
│    [Profile List]  [Create]  [Activate]  [Export/Import]        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ProfileService                              │
│  - create()   - activate()   - export()   - import()            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ProfileRepository                           │
│                    PostgreSQL (profiles table)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Database Design

```python
# millm/db/models/profile.py

class Profile(Base):
    __tablename__ = "profiles"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    model_id = Column(String(100))
    sae_id = Column(String(50))
    layer = Column(Integer)
    steering = Column(JSONB, nullable=False)  # {feature_idx: value}
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_active_profile', 'is_active', unique=True,
              postgresql_where=(is_active == True)),
    )
```

---

## 4. Service Design

```python
# millm/services/profile_service.py

class ProfileService:
    def __init__(
        self,
        repository: ProfileRepository,
        sae_service: SAEService,
    ):
        self._repository = repository
        self._sae = sae_service
        # Note: No separate SteeringService. Steering is managed
        # through SAEService and LoadedSAE methods directly.

    async def create(
        self,
        name: str,
        description: Optional[str] = None,
        steering: Optional[Dict[int, float]] = None,
    ) -> Profile:
        """Create profile from current or provided steering."""
        # Use current steering if not provided
        if steering is None:
            steering = self._sae.get_steering_values()

        profile = Profile(
            id=self._generate_id(),
            name=name,
            description=description,
            model_id=self._sae._model_service.get_current_model().id,
            sae_id=self._sae._attached_sae_id,
            layer=self._sae._attached_layer,
            steering=steering,
        )
        return await self._repository.create(profile)

    async def activate(self, profile_id: str) -> Profile:
        """Activate a profile, applying its steering."""
        profile = await self._repository.get(profile_id)
        if not profile:
            raise ProfileNotFoundError(profile_id)

        # Check compatibility
        await self._check_compatibility(profile)

        # Deactivate current
        await self._repository.deactivate_all()

        # Apply steering via SAEService
        self._sae.set_steering_batch(profile.steering)
        self._sae.enable_steering(True)

        # Mark active
        await self._repository.set_active(profile_id)

        return profile

    async def export(self, profile_id: str) -> dict:
        """Export profile as JSON."""
        profile = await self._repository.get(profile_id)
        return {
            "version": "1.0",
            "type": "millm_profile",
            "profile": {
                "name": profile.name,
                "description": profile.description,
                "model": profile.model_id,
                "sae": profile.sae_id,
                "layer": profile.layer,
                "steering": profile.steering,
            },
            "exported_at": datetime.utcnow().isoformat(),
            "exported_from": "miLLM v1.0",
        }

    async def import_profile(self, data: ProfileImportRequest) -> Profile:
        """Import profile from JSON.

        Uses ProfileImportRequest BaseModel (not file upload).
        Keys in the steering dict arrive as strings from JSON serialization
        and are converted to int on import.
        """
        # Validate format
        if data.type != "millm_profile":
            raise InvalidProfileFormat()

        profile_data = data.profile
        # Convert string keys to int (JSON only supports string keys)
        steering = {int(k): v for k, v in profile_data.steering.items()}
        return await self.create(
            name=profile_data.name,
            description=profile_data.description,
            steering=steering,
        )
```

---

## 5. API Design

```python
# millm/api/routes/management/profiles.py

router = APIRouter(prefix="/api/profiles", tags=["Profile Management"])

@router.get("", response_model=ProfileListResponse)
async def list_profiles(...)

@router.post("", response_model=ProfileResponse)
async def create_profile(request: CreateProfileRequest, ...)

@router.get("/active", response_model=ProfileResponse)
async def get_active_profile(...)
    """Get currently active profile."""

@router.post("/save-current", response_model=ProfileResponse)
async def save_current_steering(request: SaveCurrentRequest, ...)
    """Save current SAE steering configuration as a new profile."""

@router.post("/import", response_model=ProfileResponse)
async def import_profile(request: ProfileImportRequest, ...)
    """Import profile from JSON. Uses ProfileImportRequest BaseModel (not file upload).
    Keys in steering dict are strings (from JSON), converted to int on import."""

@router.get("/{profile_id}", response_model=ProfileResponse)
async def get_profile(profile_id: str, ...)

@router.put("/{profile_id}", response_model=ProfileResponse)
async def update_profile(profile_id: str, request: UpdateProfileRequest, ...)

@router.delete("/{profile_id}")
async def delete_profile(profile_id: str, ...)

@router.post("/{profile_id}/activate", response_model=ProfileResponse)
async def activate_profile(profile_id: str, ...)

@router.post("/{profile_id}/deactivate", response_model=ProfileResponse)
async def deactivate_profile(profile_id: str, clear_steering: bool = True, ...)
    """Deactivate profile. Query param clear_steering=true clears steering values."""

@router.get("/{profile_id}/export")
async def export_profile(profile_id: str, ...)
    """Export profile as miStudio-compatible JSON."""
```

---

## 6. API Integration

The profile parameter in OpenAI API requests:

> **Implementation Note:** When `request.profile` is set in `/v1/chat/completions`,
> the chat route loads the profile by name from the database, applies its steering
> values via `sae.set_steering_batch()`, and enables steering. This **permanently**
> sets the steering configuration -- it does NOT restore the previous steering after
> the request completes. The rationale is that selecting a profile is an intentional
> state change, not a per-request override.

```python
# millm/api/routes/openai/chat.py

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,  # Has profile: Optional[str]
    ...
):
    # Apply profile steering if specified (permanent, not per-request)
    if request.profile:
        profile = await profile_service.get_by_name(request.profile)
        if profile:
            sae = get_attached_sae()  # from AttachedSAEState singleton
            sae.set_steering_batch(
                {int(k): v for k, v in profile.steering.items()}
            )
            sae.enable_steering(True)
            # Note: steering remains active after this request

    # Normal generation with current steering (possibly just set by profile)
    return await inference_service.generate(...)
```

---

## 7. Testing Strategy

### Unit Tests
- Profile creation with various inputs
- Activation logic
- Export/import format validation
- Compatibility checking

### Integration Tests
- Full CRUD operations
- Profile activation with steering
- API profile parameter

---

**Document Status:** Complete
**Next Document:** `006_FTID|Profile_Management.md`
