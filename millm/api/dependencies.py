"""
FastAPI dependency injection.

Provides dependencies for database sessions and services.
"""

from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.base import async_session_factory
from millm.db.repositories.model_repository import ModelRepository


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.

    The session is automatically closed after the request.

    Yields:
        AsyncSession: Database session for the request.
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


# Type alias for injected database session
DbSession = Annotated[AsyncSession, Depends(get_db)]


async def get_model_repository(
    session: DbSession,
) -> ModelRepository:
    """
    Dependency that provides a ModelRepository.

    Args:
        session: Injected database session.

    Returns:
        ModelRepository instance for the request.
    """
    return ModelRepository(session)


# Type alias for injected ModelRepository
ModelRepo = Annotated[ModelRepository, Depends(get_model_repository)]


def get_socket_io(request: Request):
    """
    Get the Socket.IO server instance from app state.

    Args:
        request: FastAPI request object.

    Returns:
        The Socket.IO AsyncServer instance.
    """
    return request.app.state.sio


# Type alias for injected Socket.IO server
SocketIO = Annotated[object, Depends(get_socket_io)]


# =============================================================================
# Singleton dependencies (cached for application lifetime)
# =============================================================================


@lru_cache()
def get_model_downloader():
    """
    Singleton model downloader.

    Returns:
        ModelDownloader instance (cached).
    """
    from millm.ml.model_downloader import ModelDownloader

    return ModelDownloader()


@lru_cache()
def get_model_loader():
    """
    Singleton model loader.

    Returns:
        ModelLoader instance (cached).
    """
    from millm.ml.model_loader import ModelLoader

    return ModelLoader()


# Type alias for injected ModelDownloader
ModelDownloaderDep = Annotated[object, Depends(get_model_downloader)]


# =============================================================================
# Service dependencies
# =============================================================================


async def get_model_service(
    repository: ModelRepo,
    request: Request,
) -> "ModelService":
    """
    Dependency that provides a ModelService.

    Args:
        repository: Injected model repository.
        request: FastAPI request for accessing app state.

    Returns:
        ModelService instance for the request.
    """
    from millm.services.model_service import ModelService
    from millm.sockets.progress import progress_emitter

    return ModelService(
        repository=repository,
        downloader=get_model_downloader(),
        loader=get_model_loader(),
        emitter=progress_emitter,
        inference_service=get_inference_service(),
    )


# Type alias for injected ModelService
ModelServiceDep = Annotated["ModelService", Depends(get_model_service)]


# =============================================================================
# Inference Service dependency
# =============================================================================


@lru_cache()
def get_inference_service() -> "InferenceService":
    """
    Singleton inference service.

    Returns:
        InferenceService instance (cached).
    """
    from millm.core.config import settings
    from millm.services.inference_service import InferenceService

    return InferenceService(
        max_concurrent=settings.MAX_CONCURRENT_REQUESTS,
        max_pending=settings.MAX_PENDING_REQUESTS,
        kv_cache_mode=settings.KV_CACHE_MODE,
        speculative_model=settings.SPECULATIVE_MODEL,
        speculative_num_tokens=settings.SPECULATIVE_NUM_TOKENS,
        enable_cbm=settings.ENABLE_CONTINUOUS_BATCHING,
        cbm_config={
            "max_queue_size": settings.CBM_MAX_QUEUE_SIZE,
            "default_temperature": settings.CBM_DEFAULT_TEMPERATURE,
            "default_top_p": settings.CBM_DEFAULT_TOP_P,
            "default_max_tokens": settings.CBM_DEFAULT_MAX_TOKENS,
        },
        cbm_force_serial_monitoring=settings.CBM_FORCE_SERIAL_MONITORING,
    )


# Type alias for injected InferenceService
InferenceServiceDep = Annotated["InferenceService", Depends(get_inference_service)]


# =============================================================================
# SAE Service dependency
# =============================================================================


async def get_sae_repository(
    session: DbSession,
) -> "SAERepository":
    """
    Dependency that provides an SAERepository.

    Args:
        session: Injected database session.

    Returns:
        SAERepository instance for the request.
    """
    from millm.db.repositories.sae_repository import SAERepository

    return SAERepository(session)


# Type alias for injected SAERepository
SAERepo = Annotated["SAERepository", Depends(get_sae_repository)]


async def get_sae_service(
    repository: SAERepo,
    request: Request,
) -> "SAEService":
    """
    Dependency that provides an SAEService.

    Args:
        repository: Injected SAE repository.
        request: FastAPI request for accessing app state.

    Returns:
        SAEService instance for the request.
    """
    from millm.core.config import settings
    from millm.services.sae_service import SAEService
    from millm.sockets.progress import progress_emitter

    return SAEService(
        repository=repository,
        cache_dir=settings.SAE_CACHE_DIR,
        emitter=progress_emitter,
        inference_service=get_inference_service(),
    )


# Type alias for injected SAEService
SAEServiceDep = Annotated["SAEService", Depends(get_sae_service)]


# =============================================================================
# Monitoring Service dependency
# =============================================================================

# Singleton monitoring service (stored in app state)
_monitoring_service = None
_sensing_service = None


def get_sensing_service() -> "SensingService":
    """Singleton sensing service (Feature 11)."""
    global _sensing_service

    if _sensing_service is None:
        from millm.services.sensing_service import SensingService

        _sensing_service = SensingService()
    return _sensing_service


async def get_monitoring_service() -> "MonitoringService":
    """
    Dependency that provides a MonitoringService singleton.

    The singleton preserves activation history and statistics across requests.

    SAE state is accessed directly via AttachedSAEState (process singleton)
    rather than via a per-request SAEService, which would capture a DB session
    that closes after the first request and causes all subsequent monitoring
    operations to fail.
    """
    global _monitoring_service

    if _monitoring_service is None:
        from millm.services.monitoring_service import MonitoringService
        from millm.sockets.progress import progress_emitter

        _monitoring_service = MonitoringService(
            sae_service=None,  # Uses AttachedSAEState directly; see MonitoringService docs
            emitter=progress_emitter,
        )

    return _monitoring_service


# Type alias for injected MonitoringService
MonitoringServiceDep = Annotated["MonitoringService", Depends(get_monitoring_service)]


# =============================================================================
# Profile Service dependency
# =============================================================================


async def get_profile_repository(
    session: DbSession,
) -> "ProfileRepository":
    """
    Dependency that provides a ProfileRepository.

    Args:
        session: Injected database session.

    Returns:
        ProfileRepository instance for the request.
    """
    from millm.db.repositories.profile_repository import ProfileRepository

    return ProfileRepository(session)


# Type alias for injected ProfileRepository
ProfileRepo = Annotated["ProfileRepository", Depends(get_profile_repository)]


async def get_profile_service(
    repository: ProfileRepo,
    sae_service: SAEServiceDep,
) -> "ProfileService":
    """
    Dependency that provides a ProfileService.

    Args:
        repository: Injected profile repository.
        sae_service: Injected SAE service.

    Returns:
        ProfileService instance for the request.
    """
    from millm.services.profile_service import ProfileService

    return ProfileService(
        repository=repository,
        sae_service=sae_service,
    )


# Type alias for injected ProfileService
ProfileServiceDep = Annotated["ProfileService", Depends(get_profile_service)]

async def get_cluster_service(
    profile_service: ProfileServiceDep,
    repository: ProfileRepo,
    sae_service: SAEServiceDep,
) -> "ClusterService":
    """Dependency that provides a ClusterService (Feature 8)."""
    from millm.services.cluster_service import ClusterService

    return ClusterService(
        profile_service=profile_service,
        repository=repository,
        sae_service=sae_service,
    )


# Type alias for injected ClusterService
ClusterServiceDep = Annotated["ClusterService", Depends(get_cluster_service)]

# Module-level singleton so the Hub listing cache survives across requests.
_cluster_hub_service: "ClusterHubService | None" = None


def get_cluster_hub_service() -> "ClusterHubService":
    """Dependency that provides the shared ClusterHubService (Feature 8)."""
    from millm.services.cluster_hub_service import ClusterHubService

    global _cluster_hub_service
    if _cluster_hub_service is None:
        _cluster_hub_service = ClusterHubService()
    return _cluster_hub_service


# Type alias for injected ClusterHubService
ClusterHubServiceDep = Annotated["ClusterHubService", Depends(get_cluster_hub_service)]

