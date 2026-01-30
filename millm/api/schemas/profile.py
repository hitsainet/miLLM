"""
Pydantic schemas for Profile API endpoints.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CreateProfileRequest(BaseModel):
    """Request schema for creating a profile."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique display name for the profile",
        examples=["Creative Writing", "Factual Analysis", "Code Assistant"],
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Optional description of the profile's purpose",
    )
    steering: dict[int, float] = Field(
        default_factory=dict,
        description="Dictionary mapping feature indices to steering values",
    )
    model_id: str | None = Field(
        default=None,
        max_length=100,
        description="Optional model identifier this profile was designed for",
    )
    sae_id: str | None = Field(
        default=None,
        max_length=50,
        description="Optional SAE ID this profile targets",
    )
    layer: int | None = Field(
        default=None,
        ge=0,
        description="Optional layer the SAE targets",
    )

    @field_validator("steering")
    @classmethod
    def validate_steering(cls, v: dict[int, float]) -> dict[int, float]:
        """Validate all indices are non-negative."""
        for idx in v.keys():
            if idx < 0:
                raise ValueError(f"Feature index {idx} must be non-negative")
        return v


class UpdateProfileRequest(BaseModel):
    """Request schema for updating a profile."""

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="New display name for the profile",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="New description",
    )
    steering: dict[int, float] | None = Field(
        default=None,
        description="New steering values (replaces existing)",
    )
    model_id: str | None = Field(
        default=None,
        max_length=100,
        description="Model identifier this profile was designed for",
    )
    sae_id: str | None = Field(
        default=None,
        max_length=50,
        description="SAE ID this profile targets",
    )
    layer: int | None = Field(
        default=None,
        ge=0,
        description="Layer the SAE targets",
    )

    @field_validator("steering")
    @classmethod
    def validate_steering(cls, v: dict[int, float] | None) -> dict[int, float] | None:
        """Validate all indices are non-negative."""
        if v is not None:
            for idx in v.keys():
                if idx < 0:
                    raise ValueError(f"Feature index {idx} must be non-negative")
        return v


class ProfileResponse(BaseModel):
    """Response schema for a profile."""

    id: str = Field(..., description="Unique profile identifier")
    name: str = Field(..., description="Display name")
    description: str | None = Field(default=None, description="Profile description")
    model_id: str | None = Field(default=None, description="Target model identifier")
    sae_id: str | None = Field(default=None, description="Target SAE ID")
    layer: int | None = Field(default=None, description="Target layer")
    steering: dict[str, Any] = Field(
        default_factory=dict,
        description="Steering values (feature_idx -> value)",
    )
    is_active: bool = Field(..., description="Whether this profile is currently active")
    created_at: datetime = Field(..., description="When the profile was created")
    updated_at: datetime = Field(..., description="When the profile was last updated")

    model_config = {"from_attributes": True}

    @classmethod
    def from_profile(cls, profile: Any) -> "ProfileResponse":
        """
        Create a response from an ORM model.

        Args:
            profile: The Profile ORM instance.

        Returns:
            ProfileResponse populated from the ORM model.
        """
        return cls.model_validate(profile)


class ProfileListResponse(BaseModel):
    """Response schema for profile list endpoint."""

    profiles: list[ProfileResponse] = Field(
        ...,
        description="List of profiles",
    )
    total: int = Field(..., description="Total number of profiles")
    active_profile_id: str | None = Field(
        default=None,
        description="ID of currently active profile (if any)",
    )


class ActivateProfileRequest(BaseModel):
    """Request schema for activating a profile."""

    apply_steering: bool = Field(
        default=True,
        description="Whether to apply the profile's steering values to the current SAE",
    )


class ActivateProfileResponse(BaseModel):
    """Response schema for profile activation."""

    profile_id: str = Field(..., description="Activated profile ID")
    applied_steering: bool = Field(
        ...,
        description="Whether steering was applied",
    )
    feature_count: int = Field(
        default=0,
        description="Number of features steered",
    )


class DeactivateProfileResponse(BaseModel):
    """Response schema for profile deactivation."""

    profile_id: str = Field(..., description="Deactivated profile ID")
    cleared_steering: bool = Field(
        default=False,
        description="Whether steering was cleared",
    )


class SaveCurrentRequest(BaseModel):
    """Request schema for saving current steering as a profile."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name for the new profile",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Optional description",
    )


class DeleteProfileResponse(BaseModel):
    """Response schema for profile deletion."""

    profile_id: str = Field(..., description="Deleted profile ID")
    was_active: bool = Field(
        default=False,
        description="Whether the deleted profile was active",
    )
