"""
Profile management API endpoints.

Provides endpoints for creating, updating, activating, and deleting
steering configuration profiles. Includes import/export for miStudio compatibility.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Path
from pydantic import BaseModel, Field

from millm.api.dependencies import ProfileServiceDep
from millm.api.schemas.common import ApiResponse
from millm.api.schemas.profile import (
    ActivateProfileRequest,
    ActivateProfileResponse,
    CreateProfileRequest,
    DeactivateProfileResponse,
    DeleteProfileResponse,
    ProfileListResponse,
    ProfileResponse,
    SaveCurrentRequest,
    UpdateProfileRequest,
)

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


# Type alias for profile ID path parameter
ProfileId = Annotated[str, Path(description="Profile ID")]


@router.get(
    "",
    response_model=ApiResponse[ProfileListResponse],
    summary="List all profiles",
    description="Get all steering configuration profiles.",
)
async def list_profiles(
    service: ProfileServiceDep,
) -> ApiResponse[ProfileListResponse]:
    """
    List all profiles.

    Returns all steering configuration profiles ordered by name.
    Includes the currently active profile ID if one is set.
    """
    profiles = await service.list_profiles()
    active = await service.get_active_profile()

    return ApiResponse.ok(ProfileListResponse(
        profiles=[ProfileResponse.from_profile(p) for p in profiles],
        total=len(profiles),
        active_profile_id=active.id if active else None,
    ))


@router.post(
    "",
    response_model=ApiResponse[ProfileResponse],
    summary="Create a profile",
    description="Create a new steering configuration profile.",
)
async def create_profile(
    request: CreateProfileRequest,
    service: ProfileServiceDep,
) -> ApiResponse[ProfileResponse]:
    """
    Create a new profile.

    Creates a profile with the specified name and optional steering values.
    The profile is not activated until explicitly requested.
    """
    profile = await service.create_profile(
        name=request.name,
        description=request.description,
        steering=request.steering,
        model_id=request.model_id,
        sae_id=request.sae_id,
        layer=request.layer,
    )

    return ApiResponse.ok(ProfileResponse.from_profile(profile))


@router.post(
    "/save-current",
    response_model=ApiResponse[ProfileResponse],
    summary="Save current steering as profile",
    description="Save the current SAE steering configuration as a new profile.",
)
async def save_current_steering(
    request: SaveCurrentRequest,
    service: ProfileServiceDep,
) -> ApiResponse[ProfileResponse]:
    """
    Save current steering as a profile.

    Creates a new profile from the currently active SAE's steering values.
    Requires an SAE to be attached.
    """
    profile = await service.save_current_steering(
        name=request.name,
        description=request.description,
    )

    return ApiResponse.ok(ProfileResponse.from_profile(profile))


@router.get(
    "/active",
    response_model=ApiResponse[ProfileResponse | None],
    summary="Get active profile",
    description="Get the currently active profile, if any.",
)
async def get_active_profile(
    service: ProfileServiceDep,
) -> ApiResponse[ProfileResponse | None]:
    """
    Get the currently active profile.

    Returns the active profile or null if no profile is active.
    """
    active = await service.get_active_profile()
    if active:
        return ApiResponse.ok(ProfileResponse.from_profile(active))
    return ApiResponse.ok(None)


@router.get(
    "/{profile_id}",
    response_model=ApiResponse[ProfileResponse],
    summary="Get a profile",
    description="Get a single profile by ID.",
)
async def get_profile(
    profile_id: ProfileId,
    service: ProfileServiceDep,
) -> ApiResponse[ProfileResponse]:
    """
    Get a profile by ID.

    Returns the profile with the specified ID.
    """
    profile = await service.get_profile(profile_id)
    return ApiResponse.ok(ProfileResponse.from_profile(profile))


@router.patch(
    "/{profile_id}",
    response_model=ApiResponse[ProfileResponse],
    summary="Update a profile",
    description="Update an existing profile's properties.",
)
async def update_profile(
    profile_id: ProfileId,
    request: UpdateProfileRequest,
    service: ProfileServiceDep,
) -> ApiResponse[ProfileResponse]:
    """
    Update a profile.

    Updates the specified properties of an existing profile.
    Only provided fields are updated.
    """
    profile = await service.update_profile(
        profile_id=profile_id,
        name=request.name,
        description=request.description,
        steering=request.steering,
        model_id=request.model_id,
        sae_id=request.sae_id,
        layer=request.layer,
    )

    return ApiResponse.ok(ProfileResponse.from_profile(profile))


@router.post(
    "/{profile_id}/activate",
    response_model=ApiResponse[ActivateProfileResponse],
    summary="Activate a profile",
    description="Activate a profile and optionally apply its steering values.",
)
async def activate_profile(
    profile_id: ProfileId,
    request: ActivateProfileRequest,
    service: ProfileServiceDep,
) -> ApiResponse[ActivateProfileResponse]:
    """
    Activate a profile.

    Sets the profile as active and optionally applies its steering
    values to the currently attached SAE.
    """
    result = await service.activate_profile(
        profile_id=profile_id,
        apply_steering=request.apply_steering,
    )

    return ApiResponse.ok(ActivateProfileResponse(
        profile_id=result["profile_id"],
        applied_steering=result["applied_steering"],
        feature_count=result["feature_count"],
    ))


@router.post(
    "/{profile_id}/deactivate",
    response_model=ApiResponse[DeactivateProfileResponse],
    summary="Deactivate a profile",
    description="Deactivate a profile and optionally clear steering values.",
)
async def deactivate_profile(
    profile_id: ProfileId,
    service: ProfileServiceDep,
    clear_steering: bool = True,
) -> ApiResponse[DeactivateProfileResponse]:
    """
    Deactivate a profile.

    Removes the profile from active status. Optionally clears the
    current SAE steering values.
    """
    result = await service.deactivate_profile(
        profile_id=profile_id,
        clear_steering=clear_steering,
    )

    return ApiResponse.ok(DeactivateProfileResponse(
        profile_id=result["profile_id"],
        cleared_steering=result["cleared_steering"],
    ))


@router.delete(
    "/{profile_id}",
    response_model=ApiResponse[DeleteProfileResponse],
    summary="Delete a profile",
    description="Permanently delete a profile.",
)
async def delete_profile(
    profile_id: ProfileId,
    service: ProfileServiceDep,
) -> ApiResponse[DeleteProfileResponse]:
    """
    Delete a profile.

    Permanently removes the profile from the database.
    If the profile is active, it will be deactivated first.
    """
    result = await service.delete_profile(profile_id)

    return ApiResponse.ok(DeleteProfileResponse(
        profile_id=result["profile_id"],
        was_active=result["was_active"],
    ))


# =============================================================================
# Export/Import endpoints (miStudio compatible)
# =============================================================================


class ProfileExportData(BaseModel):
    """miStudio-compatible profile export format."""

    version: str = "1.0"
    name: str
    description: str | None = None
    model_id: str | None = None
    sae_id: str | None = None
    layer: int | None = None
    steering: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class ProfileImportRequest(BaseModel):
    """Request schema for importing a profile."""

    version: str = Field(default="1.0", description="Export format version")
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    model_id: str | None = None
    sae_id: str | None = None
    layer: int | None = Field(default=None, ge=0)
    steering: dict[str, Any] = Field(default_factory=dict)


@router.get(
    "/{profile_id}/export",
    response_model=ApiResponse[ProfileExportData],
    summary="Export a profile",
    description="Export a profile in miStudio-compatible JSON format.",
)
async def export_profile(
    profile_id: ProfileId,
    service: ProfileServiceDep,
) -> ApiResponse[ProfileExportData]:
    """
    Export a profile as miStudio-compatible JSON.

    Returns the profile data in a portable format for sharing or miStudio import.
    """
    profile = await service.get_profile(profile_id)

    return ApiResponse.ok(ProfileExportData(
        name=profile.name,
        description=profile.description,
        model_id=profile.model_id,
        sae_id=profile.sae_id,
        layer=profile.layer,
        steering=profile.steering or {},
        created_at=profile.created_at.isoformat() if profile.created_at else None,
        updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
    ))


@router.post(
    "/import",
    response_model=ApiResponse[ProfileResponse],
    summary="Import a profile",
    description="Import a profile from miStudio-compatible JSON format.",
)
async def import_profile(
    request: ProfileImportRequest,
    service: ProfileServiceDep,
) -> ApiResponse[ProfileResponse]:
    """
    Import a profile from miStudio-compatible JSON.

    Creates a new profile from the imported data. Validates the format
    and steering values before creating.
    """
    # Convert string keys to int keys for steering
    steering = {}
    for k, v in request.steering.items():
        try:
            steering[int(k)] = float(v)
        except (ValueError, TypeError):
            pass

    profile = await service.create_profile(
        name=request.name,
        description=request.description,
        steering=steering,
        model_id=request.model_id,
        sae_id=request.sae_id,
        layer=request.layer,
    )

    return ApiResponse.ok(ProfileResponse.from_profile(profile))
