"""
FastAPI application entry point.

Creates the combined FastAPI + Socket.IO application.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from millm import __version__
from millm.api.exception_handlers import generic_exception_handler, millm_error_handler
from millm.api.routes import register_routes
from millm.core.config import settings
from millm.core.errors import MiLLMError
from millm.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def _auto_load_model(model_identifier: str) -> None:
    """
    Auto-load a model on startup.

    Args:
        model_identifier: Model ID (numeric) or model name
    """
    from millm.db.base import async_session_factory
    from millm.db.repositories.model_repository import ModelRepository
    from millm.ml.model_downloader import ModelDownloader
    from millm.ml.model_loader import ModelLoader
    from millm.services.model_service import ModelService

    logger.info("auto_load_model_starting", model=model_identifier)

    async with async_session_factory() as session:
        repository = ModelRepository(session)
        downloader = ModelDownloader()
        loader = ModelLoader()
        service = ModelService(
            repository=repository,
            downloader=downloader,
            loader=loader,
            emitter=None,  # No progress emitter needed for auto-load
        )

        # Find model by ID or name
        model = None
        if model_identifier.isdigit():
            model = await service.get_model(int(model_identifier))
        else:
            # Search by name
            models = await service.list_models()
            for m in models:
                if m.name == model_identifier:
                    model = m
                    break

        if not model:
            logger.warning("auto_load_model_not_found", model=model_identifier)
            return

        if model.status.value == "loaded":
            logger.info("auto_load_model_already_loaded", model_id=model.id, name=model.name)
            return

        if model.status.value != "ready":
            logger.warning(
                "auto_load_model_not_ready",
                model_id=model.id,
                name=model.name,
                status=model.status.value,
            )
            return

        # Load the model
        logger.info("auto_load_model_loading", model_id=model.id, name=model.name)
        await service.load_model(model.id)

        # Wait for model to finish loading (poll status)
        import asyncio
        for _ in range(120):  # Max 2 minutes
            await asyncio.sleep(1)
            updated_model = await service.get_model(model.id)
            if updated_model.status.value == "loaded":
                logger.info("auto_load_model_complete", model_id=model.id, name=model.name)
                return
            elif updated_model.status.value == "error":
                logger.error(
                    "auto_load_model_error",
                    model_id=model.id,
                    error=updated_model.error_message,
                )
                return

        logger.warning("auto_load_model_timeout", model_id=model.id, name=model.name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Sets up logging on startup and cleans up resources on shutdown.
    """
    # Startup
    setup_logging()
    logger.info(
        "application_starting",
        version=__version__,
        debug=settings.DEBUG,
    )

    # Reset any models/SAEs marked as "loaded" since in-memory state is lost on restart
    try:
        from millm.db.base import async_session_factory
        from sqlalchemy import text

        async with async_session_factory() as session:
            # Reset models that were marked as loaded or loading
            # (in-memory state is lost on restart, so these are stale)
            result = await session.execute(
                text("UPDATE models SET status = 'ready', loaded_at = NULL WHERE status IN ('loaded', 'loading')")
            )
            if result.rowcount > 0:
                logger.info("reset_stale_model_status", count=result.rowcount)

            # Reset SAEs that were marked as attached (back to cached)
            result = await session.execute(
                text("UPDATE saes SET status = 'cached' WHERE status = 'attached'")
            )
            if result.rowcount > 0:
                logger.info("reset_stale_sae_status", count=result.rowcount)

            # Deactivate any active attachment records
            result = await session.execute(
                text("UPDATE sae_attachments SET is_active = false, detached_at = NOW() WHERE is_active = true")
            )
            if result.rowcount > 0:
                logger.info("deactivated_stale_attachments", count=result.rowcount)

            await session.commit()
    except Exception as e:
        logger.warning("failed_to_reset_stale_status", error=str(e))

    # Clear any stale in-memory SAE attachment state
    try:
        from millm.services.sae_service import AttachedSAEState
        sae_state = AttachedSAEState()
        if sae_state.is_attached:
            logger.info("clearing_stale_sae_attachment")
            sae_state.clear()
    except Exception as e:
        logger.warning("failed_to_clear_sae_state", error=str(e))

    # Auto-load model if configured
    if settings.AUTO_LOAD_MODEL:
        try:
            await _auto_load_model(settings.AUTO_LOAD_MODEL)
        except Exception as e:
            logger.error("auto_load_model_failed", error=str(e), model=settings.AUTO_LOAD_MODEL)

    yield

    # Shutdown
    logger.info("application_shutting_down")

    # Cleanup any loaded model
    try:
        from millm.ml.model_loader import LoadedModelState

        state = LoadedModelState()
        if state.is_loaded:
            logger.info("unloading_model_on_shutdown")
            state.clear()
    except ImportError:
        pass  # Model loader not yet implemented


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="miLLM API",
        description="""
## Mechanistic Interpretability LLM Server

miLLM provides a REST API for managing Large Language Models (LLMs) with support for
Sparse Autoencoders (SAEs) for mechanistic interpretability research.

### Features

- **Model Management**: Download, load, and manage LLM models from HuggingFace
- **Quantization Support**: FP16, Q4, and Q8 quantization for memory efficiency
- **Real-time Updates**: WebSocket events for download and load progress
- **SAE Integration**: Attach and configure Sparse Autoencoders for interpretability
- **Feature Steering**: Modify model behavior through feature interventions

### Authentication

Currently, the API does not require authentication for local use.
For gated HuggingFace models, provide a `hf_token` in the request.

### WebSocket Events

Connect to `/socket.io/` for real-time progress updates:

- `model:download:progress` - Download progress (bytes, speed)
- `model:download:complete` - Download finished
- `model:download:error` - Download failed
- `model:load:progress` - Load progress (stage, percentage)
- `model:load:complete` - Model loaded successfully
- `model:unload:complete` - Model unloaded

### Error Handling

All errors return a standard response format:
```json
{
    "success": false,
    "error": {
        "code": "ERROR_CODE",
        "message": "User-friendly message",
        "details": {"technical_message": "..."}
    }
}
```
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "miLLM Support",
            "url": "https://github.com/miLLM/miLLM",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=[
            {
                "name": "models",
                "description": "Model management operations - download, load, unload, delete",
            },
            {
                "name": "saes",
                "description": "SAE management operations - download, attach, steer, monitor",
            },
            {
                "name": "system",
                "description": "System health and status endpoints",
            },
            {
                "name": "openai",
                "description": "OpenAI-compatible API endpoints for chat completions, text completions, and embeddings",
            },
        ],
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(MiLLMError, millm_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Register routes
    register_routes(app)

    return app


def create_socket_io_server() -> socketio.AsyncServer:
    """
    Create and configure the Socket.IO server.

    Returns:
        Configured Socket.IO AsyncServer instance.
    """
    from millm.sockets.progress import create_socket_io

    # Use the sockets module's factory which configures progress_emitter
    return create_socket_io()


def create_combined_app() -> socketio.ASGIApp:
    """
    Create the combined FastAPI + Socket.IO ASGI application.

    Returns:
        Combined ASGI application that handles both HTTP and WebSocket.
    """
    fastapi_app = create_app()
    sio = create_socket_io_server()

    # Store Socket.IO reference in app state for dependency injection
    fastapi_app.state.sio = sio

    # Combine FastAPI and Socket.IO
    combined = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

    return combined


# The ASGI application
app = create_combined_app()


def run() -> None:
    """
    Run the application with Uvicorn.

    This is the entry point for the `millm` command.
    """
    import uvicorn

    uvicorn.run(
        "millm.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    run()
