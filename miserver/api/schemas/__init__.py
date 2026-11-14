"""
Pydantic schemas for API request/response validation.
"""

from miserver.api.schemas.common import ApiResponse, ErrorDetails
from miserver.api.schemas.profile import (
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

__all__ = [
    "ActivateProfileRequest",
    "ActivateProfileResponse",
    "ApiResponse",
    "CreateProfileRequest",
    "DeactivateProfileResponse",
    "DeleteProfileResponse",
    "ErrorDetails",
    "ProfileListResponse",
    "ProfileResponse",
    "SaveCurrentRequest",
    "UpdateProfileRequest",
]
