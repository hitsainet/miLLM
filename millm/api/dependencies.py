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
