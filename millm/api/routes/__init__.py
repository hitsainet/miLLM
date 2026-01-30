"""
API route registration.

Collects all route modules and registers them with the FastAPI application.
"""

from fastapi import FastAPI

from millm.api.routes.management.models import router as models_router
from millm.api.routes.management.monitoring import router as monitoring_router
from millm.api.routes.management.saes import router as saes_router
from millm.api.routes.openai import openai_router
from millm.api.routes.system.health import router as health_router


def register_routes(app: FastAPI) -> None:
    """
    Register all API routes with the application.

    Args:
        app: The FastAPI application instance.
    """
    # System routes (health, status, etc.)
    app.include_router(health_router)

    # Management API routes (models, SAE, steering, monitoring, etc.)
    app.include_router(models_router)
    app.include_router(saes_router)
    app.include_router(monitoring_router)

    # OpenAI-compatible API routes (mounted at /v1)
    app.include_router(openai_router, prefix="/v1")
