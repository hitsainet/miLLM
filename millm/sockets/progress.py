"""
Socket.IO progress event handlers for miLLM.

Handles real-time progress updates for model downloads, loads, and other
long-running operations via WebSocket connections.
"""

from typing import Any, Optional

import socketio
import structlog

logger = structlog.get_logger()


def register_handlers(sio: socketio.AsyncServer) -> None:
    """
    Register Socket.IO event handlers.

    Args:
        sio: The Socket.IO async server instance
    """

    @sio.event
    async def connect(sid: str, environ: dict) -> None:
        """Handle client connection."""
        logger.info("socket_connected", sid=sid)

    @sio.event
    async def disconnect(sid: str) -> None:
        """Handle client disconnection."""
        logger.info("socket_disconnected", sid=sid)


class ProgressEmitter:
    """
    Emits progress events via Socket.IO.

    This class provides a convenient interface for emitting various
    progress events to connected clients.
    """

    def __init__(self, sio: Optional[socketio.AsyncServer] = None) -> None:
        """
        Initialize the progress emitter.

        Args:
            sio: The Socket.IO async server instance. Can be None for testing.
        """
        self._sio = sio

    def set_sio(self, sio: socketio.AsyncServer) -> None:
        """Set the Socket.IO server instance."""
        self._sio = sio

    async def emit_download_progress(
        self,
        model_id: int,
        progress: float,
        downloaded_bytes: int,
        total_bytes: int,
        speed_bps: Optional[int] = None,
    ) -> None:
        """
        Emit download progress event.

        Args:
            model_id: Database ID of the model being downloaded
            progress: Progress percentage (0-100)
            downloaded_bytes: Bytes downloaded so far
            total_bytes: Total bytes to download
            speed_bps: Download speed in bytes per second (optional)
        """
        if self._sio is None:
            return

        data = {
            "modelId": model_id,
            "progress": progress,
            "downloadedBytes": downloaded_bytes,
            "totalBytes": total_bytes,
        }
        if speed_bps is not None:
            data["speedBps"] = speed_bps

        await self._sio.emit("model:download:progress", data)

    async def emit_download_complete(
        self,
        model_id: int,
        local_path: str,
    ) -> None:
        """
        Emit download complete event.

        Args:
            model_id: Database ID of the model
            local_path: Path where the model was downloaded
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "model:download:complete",
            {
                "modelId": model_id,
                "localPath": local_path,
            },
        )
        logger.info("emitted_download_complete", model_id=model_id)

    async def emit_download_error(
        self,
        model_id: int,
        error_code: str,
        error_message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Emit download error event.

        Args:
            model_id: Database ID of the model
            error_code: Error code from MiLLMError
            error_message: Human-readable error message
            details: Additional error details (optional)
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "model:download:error",
            {
                "modelId": model_id,
                "error": {
                    "code": error_code,
                    "message": error_message,
                    "details": details or {},
                },
            },
        )
        logger.warning(
            "emitted_download_error",
            model_id=model_id,
            error_code=error_code,
        )

    async def emit_load_progress(
        self,
        model_id: int,
        stage: str,
        progress: float,
    ) -> None:
        """
        Emit model load progress event.

        Args:
            model_id: Database ID of the model being loaded
            stage: Current loading stage (e.g., "loading_weights", "loading_tokenizer")
            progress: Progress percentage (0-100)
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "model:load:progress",
            {
                "modelId": model_id,
                "stage": stage,
                "progress": progress,
            },
        )

    async def emit_load_complete(
        self,
        model_id: int,
        memory_used_mb: int,
    ) -> None:
        """
        Emit model load complete event.

        Args:
            model_id: Database ID of the model
            memory_used_mb: GPU memory used in megabytes
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "model:load:complete",
            {
                "modelId": model_id,
                "memoryUsedMb": memory_used_mb,
            },
        )
        logger.info("emitted_load_complete", model_id=model_id)

    async def emit_load_error(
        self,
        model_id: int,
        error_code: str,
        error_message: str,
    ) -> None:
        """
        Emit model load error event.

        Args:
            model_id: Database ID of the model
            error_code: Error code from MiLLMError
            error_message: Human-readable error message
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "model:load:error",
            {
                "modelId": model_id,
                "error": {
                    "code": error_code,
                    "message": error_message,
                },
            },
        )
        logger.warning(
            "emitted_load_error",
            model_id=model_id,
            error_code=error_code,
        )

    async def emit_unload_complete(self, model_id: int) -> None:
        """
        Emit model unload complete event.

        Args:
            model_id: Database ID of the model that was unloaded
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "model:unload:complete",
            {
                "modelId": model_id,
            },
        )
        logger.info("emitted_unload_complete", model_id=model_id)


# Global emitter instance - will be configured with sio on app startup
progress_emitter = ProgressEmitter()


def create_socket_io() -> socketio.AsyncServer:
    """
    Create and configure the Socket.IO async server.

    Returns:
        Configured Socket.IO AsyncServer instance
    """
    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins="*",
        logger=False,  # Use structlog instead
        engineio_logger=False,
    )

    # Register event handlers
    register_handlers(sio)

    # Configure global emitter
    progress_emitter.set_sio(sio)

    return sio
