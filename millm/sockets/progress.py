"""
Socket.IO progress event handlers for miLLM.

Handles real-time progress updates for model downloads, loads, and other
long-running operations via WebSocket connections.
"""

import asyncio
import subprocess
from typing import Any, Optional

import socketio
import structlog

logger = structlog.get_logger()

# Track clients subscribed to system metrics
_system_metrics_subscribers: set[str] = set()
_system_metrics_task: Optional[asyncio.Task] = None


def get_gpu_metrics() -> dict[str, Any]:
    """
    Get GPU metrics using nvidia-smi.

    Returns:
        Dictionary with GPU utilization, memory, and temperature.
        Returns zeros if nvidia-smi is not available.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                return {
                    "gpu_utilization": int(parts[0]),
                    "gpu_memory_used_mb": int(parts[1]),
                    "gpu_memory_total_mb": int(parts[2]),
                    "gpu_temperature": int(parts[3]),
                }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        logger.debug("nvidia_smi_failed", error=str(e))

    return {
        "gpu_utilization": 0,
        "gpu_memory_used_mb": 0,
        "gpu_memory_total_mb": 0,
        "gpu_temperature": 0,
    }


async def system_metrics_loop(sio: socketio.AsyncServer) -> None:
    """
    Background task that emits system metrics periodically.

    Args:
        sio: The Socket.IO async server instance
    """
    global _system_metrics_subscribers
    logger.info("system_metrics_loop_started")

    while True:
        try:
            await asyncio.sleep(2)  # Emit every 2 seconds

            if not _system_metrics_subscribers:
                continue

            metrics = get_gpu_metrics()
            await sio.emit("system:metrics", metrics)
        except asyncio.CancelledError:
            logger.info("system_metrics_loop_cancelled")
            break
        except Exception as e:
            logger.warning("system_metrics_loop_error", error=str(e))
            await asyncio.sleep(5)  # Back off on error


def register_handlers(sio: socketio.AsyncServer) -> None:
    """
    Register Socket.IO event handlers.

    Args:
        sio: The Socket.IO async server instance
    """
    global _system_metrics_task

    @sio.event
    async def connect(sid: str, environ: dict) -> None:
        """Handle client connection."""
        logger.info("socket_connected", sid=sid)

    @sio.event
    async def disconnect(sid: str) -> None:
        """Handle client disconnection."""
        global _system_metrics_subscribers
        _system_metrics_subscribers.discard(sid)
        logger.info("socket_disconnected", sid=sid)

    @sio.on("system:join")
    async def on_system_join(sid: str) -> None:
        """Handle client subscribing to system metrics."""
        global _system_metrics_task, _system_metrics_subscribers
        _system_metrics_subscribers.add(sid)
        logger.info("system_metrics_subscribed", sid=sid)

        # Start the metrics loop if not already running
        if _system_metrics_task is None or _system_metrics_task.done():
            _system_metrics_task = asyncio.create_task(system_metrics_loop(sio))

        # Send immediate metrics on join
        metrics = get_gpu_metrics()
        await sio.emit("system:metrics", metrics, to=sid)


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

    # =========================================================================
    # SAE Events
    # =========================================================================

    async def emit_sae_download_progress(
        self,
        sae_id: str,
        percent: int,
    ) -> None:
        """
        Emit SAE download progress event.

        Args:
            sae_id: ID of the SAE being downloaded
            percent: Progress percentage (0-100)
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "sae:download:progress",
            {
                "saeId": sae_id,
                "percent": percent,
            },
        )

    async def emit_sae_download_complete(self, sae_id: str) -> None:
        """
        Emit SAE download complete event.

        Args:
            sae_id: ID of the SAE that finished downloading
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "sae:download:complete",
            {
                "saeId": sae_id,
            },
        )
        logger.info("emitted_sae_download_complete", sae_id=sae_id)

    async def emit_sae_download_error(
        self,
        sae_id: str,
        error: str,
    ) -> None:
        """
        Emit SAE download error event.

        Args:
            sae_id: ID of the SAE
            error: Error message
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "sae:download:error",
            {
                "saeId": sae_id,
                "error": error,
            },
        )
        logger.warning("emitted_sae_download_error", sae_id=sae_id, error=error)

    async def emit_sae_attached(
        self,
        sae_id: str,
        layer: int,
        memory_mb: int,
    ) -> None:
        """
        Emit SAE attached event.

        Args:
            sae_id: ID of the attached SAE
            layer: Layer where SAE is attached
            memory_mb: Memory used in MB
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "sae:attached",
            {
                "saeId": sae_id,
                "layer": layer,
                "memoryMb": memory_mb,
            },
        )
        logger.info("emitted_sae_attached", sae_id=sae_id, layer=layer)

    async def emit_sae_detached(self, sae_id: str) -> None:
        """
        Emit SAE detached event.

        Args:
            sae_id: ID of the detached SAE
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "sae:detached",
            {
                "saeId": sae_id,
            },
        )
        logger.info("emitted_sae_detached", sae_id=sae_id)

    # =========================================================================
    # Steering Events
    # =========================================================================

    async def emit_steering_changed(
        self,
        enabled: bool,
        values: dict[str, float],
        active_count: int = 0,
    ) -> None:
        """
        Emit steering state changed event.

        Args:
            enabled: Whether steering is enabled
            values: Current steering values {feature_idx: strength}
            active_count: Number of active features
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "steering:update",
            {
                "enabled": enabled,
                "values": values,
                "activeCount": active_count,
            },
        )

    # =========================================================================
    # Monitoring Events
    # =========================================================================

    def emit_activation_update(
        self,
        timestamp: str,
        features: list[tuple[int, float]],
        request_id: Optional[str] = None,
        position: int = 0,
    ) -> None:
        """
        Emit feature activation update event.

        This is a synchronous method for use in the forward pass.
        Emits to connected clients watching the "monitoring" room.

        Args:
            timestamp: ISO format timestamp
            features: List of (feature_idx, value) tuples for top features
            request_id: Associated inference request ID
            position: Token position in sequence
        """
        if self._sio is None:
            return

        import asyncio

        data = {
            "timestamp": timestamp,
            "features": [{"idx": idx, "value": val} for idx, val in features],
            "requestId": request_id,
            "position": position,
        }

        # Schedule async emit from thread (don't block forward pass)
        try:
            # Called from a background thread, so use run_coroutine_threadsafe
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - create task directly
                asyncio.create_task(
                    self._sio.emit("monitoring:activation", data)
                )
            except RuntimeError:
                # No running loop in this thread - find the main loop
                # and schedule the coroutine there
                import threading
                if hasattr(self, "_main_loop") and self._main_loop:
                    asyncio.run_coroutine_threadsafe(
                        self._sio.emit("monitoring:activation", data),
                        self._main_loop,
                    )
        except Exception:
            # Don't let emission errors affect inference
            pass

    async def emit_monitoring_state_changed(
        self,
        enabled: bool,
        monitored_features: Optional[list[int]] = None,
    ) -> None:
        """
        Emit monitoring state changed event.

        Args:
            enabled: Whether monitoring is now enabled
            monitored_features: List of monitored feature indices
        """
        if self._sio is None:
            return

        await self._sio.emit(
            "monitoring:state",
            {
                "enabled": enabled,
                "monitoredFeatures": monitored_features,
            },
        )


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
