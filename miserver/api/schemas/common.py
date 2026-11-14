"""
Common API response schemas.

Provides a consistent response format for all API endpoints.
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetails(BaseModel):
    """Error details for failed API responses."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error context",
    )


class ApiResponse(BaseModel, Generic[T]):
    """
    Standard API response wrapper.

    All API endpoints return this format for consistency.

    Success response:
        {
            "success": true,
            "data": <response data>
        }

    Error response:
        {
            "success": false,
            "error": {
                "code": "ERROR_CODE",
                "message": "Human readable message",
                "details": {}
            }
        }
    """

    success: bool = Field(..., description="Whether the request succeeded")
    data: T | None = Field(default=None, description="Response data (on success)")
    error: ErrorDetails | None = Field(
        default=None,
        description="Error details (on failure)",
    )

    @classmethod
    def ok(cls, data: T) -> "ApiResponse[T]":
        """
        Create a successful response.

        Args:
            data: The response data.

        Returns:
            ApiResponse with success=True and the provided data.
        """
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(
        cls,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> "ApiResponse[None]":
        """
        Create a failed response.

        Args:
            code: Machine-readable error code.
            message: Human-readable error message.
            details: Additional error context.

        Returns:
            ApiResponse with success=False and error details.
        """
        return cls(
            success=False,
            data=None,
            error=ErrorDetails(
                code=code,
                message=message,
                details=details or {},
            ),
        )

    model_config = {"json_schema_extra": {"examples": [{"success": True, "data": {}}]}}


class PageResponse(ApiResponse):
    """`ApiResponse` plus `meta` — the paginated lists only (039 FR-39.15, FTID I-4).

    `meta` is `{"next": cursor | None, "limit": n}`. A SUBCLASS, not a field on `ApiResponse`:
    the shared envelope's generated schema is vendored by other repos (the weights API contract),
    and every unpaginated route keeps its exact bytes. Each item's shape is unchanged.
    """

    meta: dict[str, Any] | None = Field(default=None, description="Pagination: next cursor, limit")

    @classmethod
    def page(cls, data: Any, meta: dict[str, Any]) -> "PageResponse":
        return cls(success=True, data=data, error=None, meta=meta)
