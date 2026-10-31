"""
API route registration.

Collects all route modules and registers them with the FastAPI application.
"""

from fastapi import FastAPI

from millm.api.approval_gate import mark_gated_routes
from millm.api.routes.management.approvals import router as approvals_router
from millm.api.routes.management.model_probe_capacity import router as model_probe_capacity_router
from millm.api.routes.management.models import router as models_router
from millm.api.routes.management.monitoring import router as monitoring_router
from millm.api.routes.management.probe_sessions import router as probe_sessions_router
from millm.api.routes.management.probes import router as probes_router
from millm.api.routes.management.rule_sets import router as rule_sets_router
from millm.api.routes.management.profiles import router as profiles_router
from millm.api.routes.management.clusters import router as clusters_router
from millm.api.routes.management.environment import router as environment_router
from millm.api.routes.management.circuits import router as circuits_router
from millm.api.routes.management.circuit_sensing import (
    router as circuit_sensing_router,
)
from millm.api.routes.management.sensing import router as sensing_router
from millm.api.routes.management.saes import router as saes_router
from millm.api.routes.management.settings import router as settings_router
from millm.api.routes.management.weights import router as weights_router
from millm.api.routes.management.weights_adapters import router as weights_adapters_router
from millm.api.routes.management.weights_checkpoints import (
    router as weights_checkpoints_router,
)
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

    # Management API routes (models, SAE, steering, monitoring, profiles, etc.)
    app.include_router(models_router)
    # Per-model probe limits and measured capacity (operator 2026-10-21).
    app.include_router(model_probe_capacity_router)
    app.include_router(saes_router)
    app.include_router(monitoring_router)
    # Feature 36: BEFORE the probes router, whose `GET /api/probes/{probe_id}` would otherwise
    # match `sessions` as a probe id (pinned by test_probe_session_routes_are_served).
    app.include_router(probe_sessions_router)
    app.include_router(probes_router)
    # 039 S2: rule sets as data (FR-39.17); the attachment routes ride `probes_router`.
    app.include_router(rule_sets_router)
    app.include_router(profiles_router)
    app.include_router(clusters_router)
    app.include_router(circuits_router)
    app.include_router(sensing_router)
    app.include_router(circuit_sensing_router)
    # Feature 34: operator settings (the saved Hugging Face token).
    app.include_router(settings_router)
    # Feature 35: environment settings, and the 30 that may change while miLLM runs.
    app.include_router(environment_router)
    # Attachable weights (Features 31, 32, 33): three routers under one /api/weights prefix,
    # one per feature, so no two features edit one router file.
    app.include_router(weights_router)
    app.include_router(weights_checkpoints_router)
    app.include_router(weights_adapters_router)
    # ENH-008: the approval queue an agent's gated request waits in.
    app.include_router(approvals_router)

    # OpenAI-compatible API routes (mounted at /v1)
    app.include_router(openai_router, prefix="/v1")

    # ENH-008: publish each gated route's action (`x-approval-action`) in the OpenAPI document,
    # from the live table; the coverage guard and `GET /api/approvals/actions` read it.
    mark_gated_routes(app)
