"""
Health check endpoints.

Provides health and readiness checks for load balancers, monitoring,
and container orchestration systems.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from millm import __version__
from millm.api.dependencies import get_inference_service, get_model_loader
from millm.core.resilience import CircuitBreaker, huggingface_circuit
from millm.services.sae_service import AttachedSAEState

router = APIRouter(prefix="/api/health", tags=["system"])

logger = structlog.get_logger()


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Component status")
    message: Optional[str] = Field(None, description="Additional status message")
    latency_ms: Optional[float] = Field(None, description="Check latency in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(..., description="Current server time (UTC)")
    uptime_seconds: Optional[float] = Field(None, description="Server uptime in seconds")


class ReadinessResponse(BaseModel):
    """Response schema for readiness check."""

    ready: bool = Field(..., description="Whether the service is ready to accept requests")
    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(..., description="Current server time (UTC)")
    components: list[ComponentHealth] = Field(
        default_factory=list,
        description="Health status of individual components",
    )
    model_loaded: bool = Field(False, description="Whether a model is currently loaded")
    sae_attached: bool = Field(False, description="Whether an SAE is currently attached")


class CircuitBreakerStatus(BaseModel):
    """Status of a circuit breaker."""

    name: str = Field(..., description="Circuit breaker name")
    state: str = Field(..., description="Current state (closed, open, half_open)")
    failure_count: int = Field(..., description="Current failure count")
    is_open: bool = Field(..., description="Whether circuit is currently blocking requests")


class ActiveProfileInfo(BaseModel):
    """The active steering profile, surfaced for the unified MCP server
    (Feature 9): agents answer 'what is steering right now?' from one
    detailed-health call instead of a second round-trip."""

    id: str = Field(..., description="Profile ID")
    name: str = Field(..., description="Profile display name")
    source_kind: str = Field("manual", description="'manual' or 'cluster'")
    intensity: float = Field(1.0, description="Current lambda dial (clusters)")
    sensing_enabled: bool = Field(False, description="Sensing intent (Feature 11)")


class DetailedHealthResponse(BaseModel):
    """Detailed health response with all system information."""

    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(..., description="Current server time (UTC)")
    components: list[ComponentHealth] = Field(
        default_factory=list,
        description="Health status of individual components",
    )
    circuit_breakers: list[CircuitBreakerStatus] = Field(
        default_factory=list,
        description="Circuit breaker statuses",
    )
    model_loaded: bool = Field(False, description="Whether a model is currently loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    sae_attached: bool = Field(False, description="Whether an SAE is currently attached")
    sae_id: Optional[str] = Field(
        None,
        description=(
            "ID of the FIRST attached SAE (back-compat singular view). With a "
            "multi-SAE circuit attached, see sae_count/sae_ids for the full set."
        ),
    )
    sae_count: int = Field(
        0, description="Number of attached (sae_id, layer) entries (Feature 12)"
    )
    sae_ids: list[str] = Field(
        default_factory=list,
        description="Every attached SAE id, in attachment order (Feature 12)",
    )
    inference: Optional[dict[str, Any]] = Field(None, description="Inference backend info")
    active_profile: Optional[ActiveProfileInfo] = Field(
        None, description="Currently active steering profile (null when none)"
    )


# Track server start time for uptime calculation
_start_time: Optional[datetime] = None


def get_start_time() -> datetime:
    """Get or initialize server start time."""
    global _start_time
    if _start_time is None:
        _start_time = datetime.now(timezone.utc)
    return _start_time


def get_circuit_breaker_status(circuit: CircuitBreaker) -> CircuitBreakerStatus:
    """Get status of a circuit breaker."""
    return CircuitBreakerStatus(
        name=circuit.name,
        state=circuit.state.state.value,
        failure_count=circuit.state.failure_count,
        is_open=circuit.is_open,
    )


@router.get(
    "",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Simple liveness check for container orchestration. Returns 200 if server is running.",
)
async def liveness_check() -> HealthResponse:
    """
    Liveness check endpoint.

    This is a lightweight check that confirms the server is running.
    Use /readiness for a more comprehensive check.
    """
    start_time = get_start_time()
    uptime = (datetime.now(timezone.utc) - start_time).total_seconds()

    return HealthResponse(
        status=HealthStatus.HEALTHY,
        version=__version__,
        timestamp=datetime.now(timezone.utc),
        uptime_seconds=uptime,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
    summary="Readiness check",
    description="Comprehensive readiness check. Returns 200 if service is ready to accept requests.",
)
async def readiness_check(
    inference_service=Depends(get_inference_service),
    model_loader=Depends(get_model_loader),
) -> JSONResponse:
    """
    Readiness check endpoint.

    Checks if the service is ready to accept inference requests.
    Returns 503 if any critical components are unhealthy.
    """
    components: list[ComponentHealth] = []
    overall_status = HealthStatus.HEALTHY

    # Check model loader
    try:
        is_loaded = model_loader.is_loaded
        model_name = model_loader.model_name if is_loaded else None
        components.append(ComponentHealth(
            name="model_loader",
            status=HealthStatus.HEALTHY if is_loaded else HealthStatus.DEGRADED,
            message=f"Model loaded: {model_name}" if is_loaded else "No model loaded",
        ))
        if not is_loaded:
            overall_status = HealthStatus.DEGRADED
    except Exception as e:
        logger.error("health_check_error", component="model_loader", error=str(e))
        components.append(ComponentHealth(
            name="model_loader",
            status=HealthStatus.UNHEALTHY,
            message="Component check failed — see server logs",
        ))
        overall_status = HealthStatus.UNHEALTHY

    # Check HuggingFace circuit breaker
    hf_circuit_status = get_circuit_breaker_status(huggingface_circuit)
    if hf_circuit_status.is_open:
        components.append(ComponentHealth(
            name="huggingface_circuit",
            status=HealthStatus.DEGRADED,
            message=f"Circuit open after {hf_circuit_status.failure_count} failures",
        ))
        if overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED
    else:
        components.append(ComponentHealth(
            name="huggingface_circuit",
            status=HealthStatus.HEALTHY,
            message=f"Circuit {hf_circuit_status.state}",
        ))

    # Determine if ready (healthy or degraded is acceptable)
    is_ready = overall_status != HealthStatus.UNHEALTHY

    # Get model and SAE status
    model_loaded = False
    try:
        model_loaded = model_loader.is_loaded
    except Exception:
        pass

    sae_attached = AttachedSAEState().is_attached

    response = ReadinessResponse(
        ready=is_ready,
        status=overall_status,
        version=__version__,
        timestamp=datetime.now(timezone.utc),
        components=components,
        model_loaded=model_loaded,
        sae_attached=sae_attached,
    )

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=response.model_dump(mode="json"), status_code=status_code)


@router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed health check",
    description="Detailed health status including all components and circuit breakers.",
)
async def detailed_health_check(
    inference_service=Depends(get_inference_service),
    model_loader=Depends(get_model_loader),
) -> DetailedHealthResponse:
    """
    Detailed health check endpoint.

    Returns comprehensive health information including:
    - Component health status
    - Circuit breaker states
    - Model and SAE status
    """
    components: list[ComponentHealth] = []
    circuit_breakers: list[CircuitBreakerStatus] = []
    overall_status = HealthStatus.HEALTHY

    # Check model loader
    model_loaded = False
    model_name = None
    try:
        model_loaded = model_loader.is_loaded
        if model_loaded:
            model_name = model_loader.model_name
        components.append(ComponentHealth(
            name="model_loader",
            status=HealthStatus.HEALTHY,
            message=f"Model: {model_name}" if model_loaded else "No model loaded",
        ))
    except Exception as e:
        logger.error("health_check_error", component="model_loader", error=str(e))
        components.append(ComponentHealth(
            name="model_loader",
            status=HealthStatus.UNHEALTHY,
            message="Component check failed — see server logs",
        ))
        overall_status = HealthStatus.UNHEALTHY

    # Get all circuit breaker statuses
    hf_status = get_circuit_breaker_status(huggingface_circuit)
    circuit_breakers.append(hf_status)

    if hf_status.is_open:
        if overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED

    # Inference backend info
    inference_info: dict[str, Any] = {}
    try:
        queue = inference_service.request_queue
        cbm_enabled = inference_service._cbm_backend is not None
        cbm_running = inference_service._use_cbm()
        inference_info = {
            "backend": "cbm" if cbm_running else "queue",
            "cbm_enabled": cbm_enabled,
            "cbm_running": cbm_running,
            "queue_pending": queue.pending_count,
            "queue_max_concurrent": queue.max_concurrent,
            "queue_max_pending": queue.max_pending,
        }
    except Exception:
        pass

    sae_state = AttachedSAEState()
    # Active steering profile (Feature 9: one-call status for MCP agents).
    # Best-effort: a DB hiccup must not fail the health endpoint.
    active_profile: Optional[ActiveProfileInfo] = None
    try:
        from millm.db.base import async_session_factory
        from millm.db.repositories.profile_repository import ProfileRepository

        async with async_session_factory() as session:
            active = await ProfileRepository(session).get_active()
        if active is not None:
            active_profile = ActiveProfileInfo(
                id=active.id,
                name=active.name,
                source_kind=active.source_kind or "manual",
                intensity=active.intensity if active.intensity is not None else 1.0,
                sensing_enabled=bool(active.sensing_enabled),
            )
    except Exception as e:
        logger.warning("health_active_profile_lookup_failed", error=str(e))

    return DetailedHealthResponse(
        status=overall_status,
        version=__version__,
        timestamp=datetime.now(timezone.utc),
        components=components,
        circuit_breakers=circuit_breakers,
        model_loaded=model_loaded,
        model_name=model_name,
        sae_attached=sae_state.is_attached,
        sae_id=sae_state.attached_sae_id,
        # Multi-SAE aware (Feature 12/13): a cross-layer circuit attaches
        # several SAEs, so the singular id alone under-reports what is live.
        sae_count=sae_state.count,
        sae_ids=[e.sae_id for e in sae_state.entries()],
        inference=inference_info or None,
        active_profile=active_profile,
    )


@router.get(
    "/circuits",
    response_model=list[CircuitBreakerStatus],
    summary="Circuit breaker status",
    description="Get status of all circuit breakers.",
)
async def get_circuit_breaker_statuses() -> list[CircuitBreakerStatus]:
    """
    Get circuit breaker statuses.

    Returns the current state of all circuit breakers in the system.
    """
    return [
        get_circuit_breaker_status(huggingface_circuit),
    ]


@router.post(
    "/circuits/{name}/reset",
    response_model=CircuitBreakerStatus,
    summary="Reset circuit breaker",
    description="Manually reset a circuit breaker to closed state.",
)
async def reset_circuit_breaker(name: str) -> CircuitBreakerStatus:
    """
    Reset a circuit breaker.

    Manually resets the circuit breaker to closed state.
    Use with caution - only reset if you know the underlying issue is resolved.
    """
    circuits = {
        "huggingface": huggingface_circuit,
    }

    if name not in circuits:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Circuit breaker '{name}' not found",
        )

    circuits[name].reset()
    logger.info("circuit_breaker_reset", name=name)

    return get_circuit_breaker_status(circuits[name])


# ============================================================================
# Metrics Endpoints
# ============================================================================


class MetricsResponse(BaseModel):
    """Application metrics for observability."""

    # Request metrics
    total_requests: int = Field(0, description="Total requests processed")
    active_requests: int = Field(0, description="Currently active requests")
    request_errors: int = Field(0, description="Total request errors")

    # Model metrics
    model_loaded: bool = Field(False, description="Whether a model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    model_load_count: int = Field(0, description="Total model loads since startup")
    model_unload_count: int = Field(0, description="Total model unloads since startup")

    # SAE metrics
    sae_attached: bool = Field(False, description="Whether an SAE is attached")
    sae_id: Optional[str] = Field(None, description="ID of attached SAE")

    # Steering metrics
    steering_enabled: bool = Field(False, description="Whether steering is enabled")
    #: Feature 19. How many circuits are steering right now, over how many
    #: layers, and how many of those layers carry MORE THAN ONE circuit.
    #: `circuit_layers_composed > 0` is the alertable condition: on a composed
    #: layer the rung header is suppressed, because no single circuit's
    #: evidence describes the response.
    circuits_serving: int = Field(0, description="Circuits currently steering")
    circuit_layers_served: int = Field(
        0, description="Distinct layers steered by circuits"
    )
    circuit_layers_composed: int = Field(
        0, description="Layers carrying more than one circuit (rung suppressed)"
    )
    #: R2-16: unrepaired claim/steering divergences. Alert on the RATE.
    circuit_claim_faults: int = Field(
        0, description="Claim/steering divergences that could not be repaired"
    )
    active_features: int = Field(0, description="Number of active steering features")

    # Monitoring metrics
    monitoring_enabled: bool = Field(False, description="Whether monitoring is enabled")
    monitored_features: int = Field(0, description="Number of monitored features")

    # Circuit breaker metrics
    circuit_breaker_open: bool = Field(False, description="Whether any circuit is open")
    circuit_breaker_trips: int = Field(0, description="Total circuit breaker trips")

    # System metrics
    uptime_seconds: float = Field(0.0, description="Server uptime in seconds")
    timestamp: datetime = Field(..., description="Current server time (UTC)")


# Simple in-memory counters for metrics
class MetricsCounter:
    """Simple metrics counter for application observability."""

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.active_requests: int = 0
        self.request_errors: int = 0
        self.model_load_count: int = 0
        self.model_unload_count: int = 0
        self.circuit_breaker_trips: int = 0
        #: F19 R2-16. Every one of F19's failure handlers logged and continued,
        #: so a claim leak, a failed rollback, or a bypassed gate was
        #: discoverable ONLY by grepping structured logs for a specific event
        #: name. An operator watching a dashboard saw a fully green system.
        #: These are the states where the DB and the runtime can disagree —
        #: exactly the divergence F19 exists to remove — so they need a number.
        self.circuit_claim_faults: int = 0

    def increment_requests(self) -> None:
        """Increment total and active request counts."""
        self.total_requests += 1
        self.active_requests += 1

    def decrement_active(self) -> None:
        """Decrement active request count."""
        self.active_requests = max(0, self.active_requests - 1)

    def increment_errors(self) -> None:
        """Increment error count."""
        self.request_errors += 1

    def increment_model_loads(self) -> None:
        """Increment model load count."""
        self.model_load_count += 1

    def increment_model_unloads(self) -> None:
        """Increment model unload count."""
        self.model_unload_count += 1

    def increment_circuit_trips(self) -> None:
        """Increment circuit breaker trip count."""
        self.circuit_breaker_trips += 1

    def increment_circuit_claim_faults(self) -> None:
        """A claim/steering divergence was detected and could not be repaired.

        Non-zero means the claims table and what is actually steering may
        disagree: a claim that could not be released, a rollback that could not
        complete, or the claim gate running without a session. Alert on the
        RATE, not the value — one fault during a DB blip is recoverable; a
        rising count means layers are leaking.
        """
        self.circuit_claim_faults += 1


# Global metrics counter instance
metrics_counter = MetricsCounter()


def get_metrics_counter() -> MetricsCounter:
    """Get the global metrics counter."""
    return metrics_counter


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Application metrics",
    description="Get application metrics for monitoring and observability.",
)
async def get_metrics(
    model_loader=Depends(get_model_loader),
) -> MetricsResponse:
    """
    Get application metrics.

    Returns metrics for monitoring dashboards and alerting systems.
    Includes request counts, model status, and circuit breaker state.
    """
    start_time = get_start_time()
    uptime = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Get model status
    model_loaded = False
    model_name = None
    try:
        model_loaded = model_loader.is_loaded
        if model_loaded:
            model_name = model_loader.model_name
    except Exception:
        pass

    # Get circuit breaker status
    hf_status = get_circuit_breaker_status(huggingface_circuit)
    circuit_open = hf_status.is_open

    sae_state = AttachedSAEState()
    sae_attached = sae_state.is_attached
    # Multi-SAE aware (Feature 12): aggregate steering/monitoring/active-feature
    # counts across ALL attached SAEs so a cross-layer circuit is not
    # under-reported to only its first entry.
    _entries = sae_state.entries()
    steering_enabled = any(e.sae.is_steering_enabled for e in _entries if e.sae)
    monitoring_enabled = any(e.sae.is_monitoring_enabled for e in _entries if e.sae)
    active_features = sum(
        len(e.sae.get_steering_values())
        for e in _entries
        if e.sae and e.sae.is_steering_enabled
    )

    # F19 R1-11: circuit-serving metrics. Composition — the state in which the
    # runtime deliberately STOPS making evidence claims — was invisible to
    # every dashboard and unalertable, and the only "circuit" metric on this
    # surface is the unrelated HuggingFace HTTP breaker, which is an active
    # trap for an operator alerting on the word.
    #
    # Read from the in-memory owner map rather than the claims table: /metrics
    # is scraped continuously and must not add a DB round-trip per scrape, and
    # the owner map is the authority on what is ACTUALLY steering right now.
    circuit_owners = sorted(
        owner for owner in sae_state._owners if owner.startswith("circuit:")
    )
    circuit_layers: set[int] = set()
    for owner in circuit_owners:
        circuit_layers.update(sae_state.owner_keys(owner))
    # F19 R2-11: count EVERY owner on a circuit-held layer, not just circuit
    # owners.
    #
    # Counting `circuit:` owners alone made the metric blind to the case it
    # most needed to catch: a layer whose co-tenant arrived through
    # SLICE-FALLBACK materialises a CLUSTER PROFILE, not a circuit owner. That
    # layer has one circuit owner, so `circuit_layers_composed` read 0 — the
    # documented alertable condition never firing — while `GET /circuits/claims`
    # badged the layer composed and the rung header WAS being suppressed.
    #
    # Two authorities disagreeing is worse than either being wrong: the metric
    # said "nothing composed" while the response headers said otherwise. The
    # metric now agrees with what actually suppresses the header — more than
    # one contributor on a layer a circuit is steering.
    layer_holders: dict[int, int] = {}
    circuit_held: set[int] = set()
    for owner in circuit_owners:
        for layer in sae_state.owner_keys(owner):
            circuit_held.add(layer)
    for owner in sae_state._owners:
        for layer in sae_state.owner_keys(owner):
            if layer in circuit_held:
                layer_holders[layer] = layer_holders.get(layer, 0) + 1
    composed_layers = sorted(l for l, n in layer_holders.items() if n > 1)

    return MetricsResponse(
        circuits_serving=len(circuit_owners),
        circuit_layers_served=len(circuit_layers),
        circuit_layers_composed=len(composed_layers),
        circuit_claim_faults=metrics_counter.circuit_claim_faults,
        total_requests=metrics_counter.total_requests,
        active_requests=metrics_counter.active_requests,
        request_errors=metrics_counter.request_errors,
        model_loaded=model_loaded,
        model_name=model_name,
        model_load_count=metrics_counter.model_load_count,
        model_unload_count=metrics_counter.model_unload_count,
        sae_attached=sae_attached,
        sae_id=sae_state.attached_sae_id,
        steering_enabled=steering_enabled,
        active_features=active_features,
        monitoring_enabled=monitoring_enabled,
        monitored_features=0,
        circuit_breaker_open=circuit_open,
        circuit_breaker_trips=metrics_counter.circuit_breaker_trips,
        uptime_seconds=uptime,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/metrics/prometheus",
    summary="Prometheus metrics",
    description="Get metrics in Prometheus text format.",
    response_class=JSONResponse,
)
async def get_prometheus_metrics(
    model_loader=Depends(get_model_loader),
) -> str:
    """
    Get metrics in Prometheus text exposition format.

    Returns metrics in the format expected by Prometheus scrapers.
    """
    from fastapi.responses import PlainTextResponse

    start_time = get_start_time()
    uptime = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Get model status
    model_loaded = 1 if model_loader.is_loaded else 0

    # Get circuit breaker status
    hf_status = get_circuit_breaker_status(huggingface_circuit)
    circuit_open = 1 if hf_status.is_open else 0

    # F19: circuit serving, from the in-memory owner map (no DB round-trip on
    # a continuously-scraped endpoint).
    _sae_state = AttachedSAEState()
    _circuit_owners = [
        owner for owner in _sae_state._owners if owner.startswith("circuit:")
    ]
    # R2-11: same rule as the JSON surface — every owner on a circuit-held
    # layer counts, so a slice-fallback co-tenant is not invisible.
    _circuit_held: set[int] = set()
    for _owner in _circuit_owners:
        _circuit_held.update(_sae_state.owner_keys(_owner))
    _holders: dict[int, int] = {}
    for _owner in _sae_state._owners:
        for _layer in _sae_state.owner_keys(_owner):
            if _layer in _circuit_held:
                _holders[_layer] = _holders.get(_layer, 0) + 1
    circuits_serving = len(_circuit_owners)
    circuit_layers_served = len(_circuit_held)
    circuit_layers_composed = sum(1 for n in _holders.values() if n > 1)

    # Build Prometheus format
    lines = [
        "# HELP millm_requests_total Total number of requests processed",
        "# TYPE millm_requests_total counter",
        f"millm_requests_total {metrics_counter.total_requests}",
        "",
        "# HELP millm_requests_active Currently active requests",
        "# TYPE millm_requests_active gauge",
        f"millm_requests_active {metrics_counter.active_requests}",
        "",
        # F19: composition is the alertable condition — on a composed layer
        # the rung header is suppressed because no single circuit's evidence
        # describes the response. NOTE for anyone alerting on "circuit":
        # `millm_circuit_breaker_*` below is the HuggingFace HTTP breaker and
        # is UNRELATED to circuit serving.
        "# HELP millm_circuits_serving Circuits currently steering",
        "# TYPE millm_circuits_serving gauge",
        f"millm_circuits_serving {circuits_serving}",
        "",
        "# HELP millm_circuit_layers_served Distinct layers steered by circuits",
        "# TYPE millm_circuit_layers_served gauge",
        f"millm_circuit_layers_served {circuit_layers_served}",
        "",
        "# HELP millm_circuit_layers_composed Layers carrying more than one circuit (rung header suppressed)",
        "# TYPE millm_circuit_layers_composed gauge",
        f"millm_circuit_layers_composed {circuit_layers_composed}",
        "",
        "# HELP millm_circuit_claim_faults_total Claim/steering divergences that could not be repaired",
        "# TYPE millm_circuit_claim_faults_total counter",
        f"millm_circuit_claim_faults_total {metrics_counter.circuit_claim_faults}",
        "",
        "# HELP millm_request_errors_total Total number of request errors",
        "# TYPE millm_request_errors_total counter",
        f"millm_request_errors_total {metrics_counter.request_errors}",
        "",
        "# HELP millm_model_loaded Whether a model is currently loaded",
        "# TYPE millm_model_loaded gauge",
        f"millm_model_loaded {model_loaded}",
        "",
        "# HELP millm_model_loads_total Total number of model loads",
        "# TYPE millm_model_loads_total counter",
        f"millm_model_loads_total {metrics_counter.model_load_count}",
        "",
        "# HELP millm_model_unloads_total Total number of model unloads",
        "# TYPE millm_model_unloads_total counter",
        f"millm_model_unloads_total {metrics_counter.model_unload_count}",
        "",
        "# HELP millm_circuit_breaker_open Whether circuit breaker is open",
        "# TYPE millm_circuit_breaker_open gauge",
        f'millm_circuit_breaker_open{{name="huggingface"}} {circuit_open}',
        "",
        "# HELP millm_circuit_breaker_trips_total Total circuit breaker trips",
        "# TYPE millm_circuit_breaker_trips_total counter",
        f"millm_circuit_breaker_trips_total {metrics_counter.circuit_breaker_trips}",
        "",
        "# HELP millm_uptime_seconds Server uptime in seconds",
        "# TYPE millm_uptime_seconds gauge",
        f"millm_uptime_seconds {uptime:.2f}",
        "",
    ]

    return PlainTextResponse(content="\n".join(lines), media_type="text/plain")


@router.get("/version", summary="Get application version")
def get_version():
    """Return application version from VERSION file, falling back to package __version__."""
    from pathlib import Path
    for candidate in [
        Path(__file__).parents[5] / "VERSION",
        Path(__file__).parents[4] / "VERSION",
        Path(__file__).parents[3] / "VERSION",
    ]:
        if candidate.exists():
            return {"version": candidate.read_text().strip(), "app": "miLLM"}
    return {"version": __version__, "app": "miLLM"}


@router.get(
    "/inference",
    summary="Inference backend status",
    description=(
        "Returns the active inference backend (serial queue or CBM), its "
        "capabilities, and known limitations.  Useful for diagnosing latency "
        "jitter when requests fall back from CBM to the serial path, and for "
        "understanding which features are available under each backend.  "
        "The X-miLLM-Backend response header on /v1/* endpoints reports the "
        "same information per request."
    ),
    tags=["system"],
)
def get_inference_status(
    inference=Depends(get_inference_service),
) -> dict[str, Any]:
    """
    Describe the active inference backend and its per-request capabilities.

    Serial backend capabilities:
      - Per-request temperature / top_p
      - Prefix KV-cache reuse across requests with the same system prompt
      - Per-request steering profile override (request.profile)
      - Speculative decoding (if SPECULATIVE_MODEL is configured)

    CBM (ContinuousBatchingManager) backend capabilities:
      - Higher throughput via request batching
      - Fixed temperature / top_p (set at manager creation, not per request)
      - No prefix cache, no profile override, no speculative decoding
      - Requests with non-matching sampling params fall back to serial silently
        (logged at INFO as cbm_routing_fallback_to_serial)
    """
    return inference.get_backend_info()
