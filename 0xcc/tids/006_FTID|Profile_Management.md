# Technical Implementation Document: Profile Management

## miLLM Feature 6

**Document Version:** 1.0
**Created:** January 30, 2026
**Status:** Draft

---

## 1. File Structure

```
millm/
├── api/
│   ├── routes/
│   │   └── management/
│   │       └── profiles.py              # Profile API routes
│   └── schemas/
│       └── profile.py                   # Pydantic schemas
│
├── services/
│   └── profile_service.py               # Profile service
│
├── db/
│   ├── models/
│   │   └── profile.py                   # SQLAlchemy model
│   └── repositories/
│       └── profile_repository.py        # Database operations
│
└── core/
    └── errors.py                        # Add profile errors
```

---

## 2. Database Model

```python
# millm/db/models/profile.py

from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

from millm.db.base import Base


class Profile(Base):
    """Steering configuration profile."""
    __tablename__ = "profiles"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    model_id = Column(String(100))
    sae_id = Column(String(50))
    layer = Column(Integer)
    steering = Column(JSONB, nullable=False, default={})
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index(
            'idx_single_active_profile',
            'is_active',
            unique=True,
            postgresql_where=(is_active == True)
        ),
    )
```

---

## 3. Pydantic Schemas

```python
# millm/api/schemas/profile.py

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class CreateProfileRequest(BaseModel):
    """Request to create a profile."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    steering: Optional[Dict[int, float]] = None  # None = use current


class UpdateProfileRequest(BaseModel):
    """Request to update a profile."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    steering: Optional[Dict[int, float]] = None


class ProfileResponse(BaseModel):
    """Profile response."""
    id: str
    name: str
    description: Optional[str]
    model_id: Optional[str]
    sae_id: Optional[str]
    layer: Optional[int]
    steering: Dict[str, float]  # String keys for JSON
    feature_count: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, profile) -> "ProfileResponse":
        return cls(
            id=profile.id,
            name=profile.name,
            description=profile.description,
            model_id=profile.model_id,
            sae_id=profile.sae_id,
            layer=profile.layer,
            steering={str(k): v for k, v in profile.steering.items()},
            feature_count=len(profile.steering),
            is_active=profile.is_active,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )


class ProfileListResponse(BaseModel):
    """Profile list response."""
    profiles: List[ProfileResponse]
    active_profile_id: Optional[str]


class ProfileExport(BaseModel):
    """Profile export format."""
    version: str = "1.0"
    type: str = "millm_profile"
    profile: dict
    exported_at: str
    exported_from: str = "miLLM v1.0"
```

---

## 4. Repository Implementation

```python
# millm/db/repositories/profile_repository.py

from typing import Optional, List
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.profile import Profile


class ProfileRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_all(self) -> List[Profile]:
        result = await self._session.execute(
            select(Profile).order_by(Profile.name)
        )
        return result.scalars().all()

    async def get(self, profile_id: str) -> Optional[Profile]:
        result = await self._session.execute(
            select(Profile).where(Profile.id == profile_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[Profile]:
        result = await self._session.execute(
            select(Profile).where(Profile.name == name)
        )
        return result.scalar_one_or_none()

    async def get_active(self) -> Optional[Profile]:
        result = await self._session.execute(
            select(Profile).where(Profile.is_active == True)
        )
        return result.scalar_one_or_none()

    async def create(self, profile: Profile) -> Profile:
        self._session.add(profile)
        await self._session.commit()
        return profile

    async def update(self, profile_id: str, **kwargs) -> Profile:
        await self._session.execute(
            update(Profile)
            .where(Profile.id == profile_id)
            .values(**kwargs)
        )
        await self._session.commit()
        return await self.get(profile_id)

    async def delete(self, profile_id: str):
        profile = await self.get(profile_id)
        if profile:
            await self._session.delete(profile)
            await self._session.commit()

    async def deactivate_all(self):
        await self._session.execute(
            update(Profile).values(is_active=False)
        )
        await self._session.commit()

    async def set_active(self, profile_id: str):
        await self.deactivate_all()
        await self._session.execute(
            update(Profile)
            .where(Profile.id == profile_id)
            .values(is_active=True)
        )
        await self._session.commit()
```

---

## 5. Service Implementation

```python
# millm/services/profile_service.py

from typing import Dict, Optional, List
import uuid
from datetime import datetime
import logging

from millm.db.repositories.profile_repository import ProfileRepository
from millm.db.models.profile import Profile
from millm.services.steering_service import SteeringService
from millm.services.sae_service import SAEService
from millm.core.errors import ProfileNotFoundError, ProfileCompatibilityError

logger = logging.getLogger(__name__)


class ProfileService:
    def __init__(
        self,
        repository: ProfileRepository,
        steering_service: SteeringService,
        sae_service: SAEService,
    ):
        self._repository = repository
        self._steering = steering_service
        self._sae = sae_service

    async def list_profiles(self) -> dict:
        profiles = await self._repository.get_all()
        active = await self._repository.get_active()
        return {
            "profiles": profiles,
            "active_profile_id": active.id if active else None,
        }

    async def get_profile(self, profile_id: str) -> Profile:
        profile = await self._repository.get(profile_id)
        if not profile:
            raise ProfileNotFoundError(profile_id)
        return profile

    async def create_profile(
        self,
        name: str,
        description: Optional[str] = None,
        steering: Optional[Dict[int, float]] = None,
    ) -> Profile:
        # Use current steering if not provided
        if steering is None and self._sae._loaded_sae:
            steering = self._sae._loaded_sae.get_steering_values()
        steering = steering or {}

        profile = Profile(
            id=f"prf-{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            model_id=self._sae._model_service.get_current_model().id if self._sae._model_service.is_loaded() else None,
            sae_id=self._sae._attached_sae_id,
            layer=self._sae._attached_layer,
            steering=steering,
        )
        return await self._repository.create(profile)

    async def update_profile(
        self,
        profile_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        steering: Optional[Dict[int, float]] = None,
    ) -> Profile:
        kwargs = {}
        if name is not None:
            kwargs["name"] = name
        if description is not None:
            kwargs["description"] = description
        if steering is not None:
            kwargs["steering"] = steering
        kwargs["updated_at"] = datetime.utcnow()

        return await self._repository.update(profile_id, **kwargs)

    async def delete_profile(self, profile_id: str):
        profile = await self._repository.get(profile_id)
        if profile and profile.is_active:
            raise ValueError("Cannot delete active profile")
        await self._repository.delete(profile_id)

    async def activate_profile(self, profile_id: str) -> Profile:
        profile = await self.get_profile(profile_id)

        # Check compatibility (warn but don't block)
        warnings = await self._check_compatibility(profile)

        # Apply steering
        steering = {int(k): v for k, v in profile.steering.items()}
        await self._steering.batch_set(steering)
        await self._steering.set_enabled(True)

        # Mark active
        await self._repository.set_active(profile_id)

        logger.info(f"Activated profile: {profile.name}")
        return profile

    async def deactivate_profile(self):
        await self._repository.deactivate_all()

    async def export_profile(self, profile_id: str) -> dict:
        profile = await self.get_profile(profile_id)
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

    async def import_profile(self, data: dict) -> Profile:
        if data.get("type") != "millm_profile":
            raise ValueError("Invalid profile format")

        profile_data = data["profile"]
        steering = {int(k): v for k, v in profile_data["steering"].items()}

        return await self.create_profile(
            name=profile_data["name"],
            description=profile_data.get("description"),
            steering=steering,
        )

    async def _check_compatibility(self, profile: Profile) -> List[str]:
        warnings = []
        if profile.sae_id != self._sae._attached_sae_id:
            warnings.append(
                f"Profile was created with different SAE "
                f"({profile.sae_id} vs {self._sae._attached_sae_id})"
            )
        return warnings
```

---

## 6. API Routes Implementation

```python
# millm/api/routes/management/profiles.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import json

from millm.api.schemas.profile import (
    CreateProfileRequest,
    UpdateProfileRequest,
    ProfileResponse,
    ProfileListResponse,
)
from millm.services.profile_service import ProfileService
from millm.api.dependencies import get_profile_service

router = APIRouter(prefix="/api/profiles", tags=["Profile Management"])


@router.get("", response_model=ProfileListResponse)
async def list_profiles(
    profile_service: ProfileService = Depends(get_profile_service),
):
    result = await profile_service.list_profiles()
    return ProfileListResponse(
        profiles=[ProfileResponse.from_model(p) for p in result["profiles"]],
        active_profile_id=result["active_profile_id"],
    )


@router.post("", response_model=ProfileResponse)
async def create_profile(
    request: CreateProfileRequest,
    profile_service: ProfileService = Depends(get_profile_service),
):
    profile = await profile_service.create_profile(
        name=request.name,
        description=request.description,
        steering=request.steering,
    )
    return ProfileResponse.from_model(profile)


@router.get("/{profile_id}", response_model=ProfileResponse)
async def get_profile(
    profile_id: str,
    profile_service: ProfileService = Depends(get_profile_service),
):
    profile = await profile_service.get_profile(profile_id)
    return ProfileResponse.from_model(profile)


@router.put("/{profile_id}", response_model=ProfileResponse)
async def update_profile(
    profile_id: str,
    request: UpdateProfileRequest,
    profile_service: ProfileService = Depends(get_profile_service),
):
    profile = await profile_service.update_profile(
        profile_id,
        name=request.name,
        description=request.description,
        steering=request.steering,
    )
    return ProfileResponse.from_model(profile)


@router.delete("/{profile_id}")
async def delete_profile(
    profile_id: str,
    profile_service: ProfileService = Depends(get_profile_service),
):
    await profile_service.delete_profile(profile_id)
    return {"deleted": True}


@router.post("/{profile_id}/activate", response_model=ProfileResponse)
async def activate_profile(
    profile_id: str,
    profile_service: ProfileService = Depends(get_profile_service),
):
    profile = await profile_service.activate_profile(profile_id)
    return ProfileResponse.from_model(profile)


@router.get("/{profile_id}/export")
async def export_profile(
    profile_id: str,
    profile_service: ProfileService = Depends(get_profile_service),
):
    data = await profile_service.export_profile(profile_id)
    return JSONResponse(
        content=data,
        headers={
            "Content-Disposition": f"attachment; filename={data['profile']['name']}.json"
        }
    )


@router.post("/import", response_model=ProfileResponse)
async def import_profile(
    file: UploadFile = File(...),
    profile_service: ProfileService = Depends(get_profile_service),
):
    content = await file.read()
    data = json.loads(content)
    profile = await profile_service.import_profile(data)
    return ProfileResponse.from_model(profile)
```

---

**Document Status:** Complete
**Next Document:** `006_FTASKS|Profile_Management.md`
