"""
Health check endpoint.

Provides a simple health check for load balancers and monitoring.
"""

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

from millm import __version__

router = APIRouter(prefix="/api/health", tags=["system"])


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Health status (healthy/unhealthy)")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(..., description="Current server time")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the server.",
)
async def health_check() -> HealthResponse:
    """
    Check if the server is healthy.

    Returns basic health information including version and timestamp.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow(),
    )
