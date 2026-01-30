"""
Pydantic schemas for API request/response validation.
"""

from millm.api.schemas.common import ApiResponse, ErrorDetails
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
