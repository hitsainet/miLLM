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
    from millm.services.inference_service import InferenceService

    return InferenceService()


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
    )


# Type alias for injected SAEService
SAEServiceDep = Annotated["SAEService", Depends(get_sae_service)]


# =============================================================================
# Monitoring Service dependency
# =============================================================================

# Singleton monitoring service (stored in app state)
_monitoring_service = None


async def get_monitoring_service(
    sae_service: SAEServiceDep,
    request: Request,
) -> "MonitoringService":
    """
    Dependency that provides a MonitoringService.

    The monitoring service is a singleton stored in app state to preserve
    history and statistics across requests.

    Args:
        sae_service: Injected SAE service.
        request: FastAPI request for accessing app state.

    Returns:
        MonitoringService instance.
    """
    global _monitoring_service

    if _monitoring_service is None:
        from millm.services.monitoring_service import MonitoringService
        from millm.sockets.progress import progress_emitter

        _monitoring_service = MonitoringService(
            sae_service=sae_service,
            emitter=progress_emitter,
        )

    return _monitoring_service


# Type alias for injected MonitoringService
MonitoringServiceDep = Annotated["MonitoringService", Depends(get_monitoring_service)]
