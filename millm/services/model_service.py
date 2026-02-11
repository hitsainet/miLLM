"""
Model service for orchestrating model operations.

This service coordinates between the repository, downloader, and loader
components to manage model lifecycle operations.
"""

import asyncio
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog

# Configuration constants
MAX_DOWNLOAD_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds
RETRY_MAX_DELAY = 30.0  # seconds
UNLOAD_TIMEOUT = 30.0  # seconds

from millm.api.schemas.model import ModelDownloadRequest, ModelPreviewRequest
from millm.core.errors import (
    DownloadCancelledError,
    ModelAlreadyExistsError,
    ModelAlreadyLoadedError,
    ModelBusyError,
    ModelLockedError,
    ModelNotFoundError,
    ModelNotLoadedError,
)
from millm.db.models.model import Model, ModelSource, ModelStatus
from millm.db.repositories.model_repository import ModelRepository
from millm.ml.memory_utils import estimate_memory_mb
from millm.ml.model_downloader import ModelDownloader
from millm.ml.model_loader import ModelLoader
from millm.sockets.progress import ProgressEmitter

logger = structlog.get_logger()


class ModelService:
    """
    Orchestration layer for model operations.

    Coordinates between repository, downloader, and loader components.
    Uses a thread pool for long-running operations like downloads and loads.
    """

    def __init__(
        self,
        repository: ModelRepository,
        downloader: ModelDownloader,
        loader: Optional[ModelLoader] = None,
        emitter: Optional[ProgressEmitter] = None,
    ) -> None:
        """
        Initialize the model service.

        Args:
            repository: Model database repository
            downloader: Model downloader for HuggingFace
            loader: Model loader for GPU loading
            emitter: Progress event emitter for WebSocket updates
        """
        self.repository = repository
        self.downloader = downloader
        self.loader = loader or ModelLoader()
        self.emitter = emitter

        # Thread pool for background tasks
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="millm-",
        )

        # Track active downloads for cancellation
        self._active_downloads: dict[int, Future[str]] = {}
        self._cancelled_downloads: set[int] = set()

        # Track active loads
        self._loading_model_id: Optional[int] = None

        # Reference to main event loop for thread-safe async operations
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Track download progress (in-memory, model_id -> progress percentage)
        self._download_progress: dict[int, int] = {}

    def _run_async_from_thread(self, coro: Any) -> Any:
        """
        Run an async coroutine from a background thread.

        Uses asyncio.run_coroutine_threadsafe() to safely schedule the coroutine
        on the main event loop and wait for its result.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        if self._main_loop is None:
            raise RuntimeError("Main event loop not set")
        future = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
        return future.result(timeout=30.0)  # 30 second timeout for DB operations

    def get_download_progress(self, model_id: int) -> int | None:
        """
        Get the current download progress for a model.

        Args:
            model_id: The model's database ID

        Returns:
            Progress percentage (0-100) if downloading, None otherwise.
        """
        return self._download_progress.get(model_id)

    async def list_models(self) -> list[Model]:
        """
        Get all models from the database.

        Returns:
            List of all models ordered by creation date descending.
        """
        return await self.repository.get_all()

    async def get_model(self, model_id: int) -> Model:
        """
        Get a single model by ID.

        Args:
            model_id: The model's database ID

        Returns:
            The model if found

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model = await self.repository.get_by_id(model_id)
        if not model:
            raise ModelNotFoundError(
                f"Model with ID {model_id} not found",
                details={"model_id": model_id},
            )
        return model

    async def preview_model(self, request: ModelPreviewRequest) -> dict[str, Any]:
        """
        Get model info from HuggingFace without downloading.

        Args:
            request: Preview request with repo_id and optional token

        Returns:
            Dict with model metadata (name, params, architecture, etc.)
        """
        return self.downloader.get_model_info(
            repo_id=request.repo_id,
            token=request.hf_token,
        )

    # =========================================================================
    # Download Operations
    # =========================================================================

    async def download_model(self, request: ModelDownloadRequest) -> Model:
        """
        Start downloading a model from HuggingFace.

        Creates a database record and starts the download in a background thread.
        Progress updates are sent via WebSocket.

        Args:
            request: Download request with repo_id, quantization, etc.

        Returns:
            The created model record (status will be DOWNLOADING)

        Raises:
            ModelAlreadyExistsError: If model with same repo_id/quantization exists
        """
        # Check for existing model with same repo_id and quantization
        if request.source == ModelSource.HUGGINGFACE and request.repo_id:
            existing = await self.repository.find_by_repo_quantization(
                repo_id=request.repo_id,
                quantization=request.quantization,
            )
            if existing:
                raise ModelAlreadyExistsError(
                    f"Model {request.repo_id} with {request.quantization.value} quantization already exists",
                    details={
                        "repo_id": request.repo_id,
                        "quantization": request.quantization.value,
                        "existing_model_id": existing.id,
                    },
                )

        # Get model info for metadata
        model_info = None
        if request.source == ModelSource.HUGGINGFACE and request.repo_id:
            try:
                model_info = self.downloader.get_model_info(
                    repo_id=request.repo_id,
                    token=request.hf_token,
                )
            except Exception as e:
                logger.warning(
                    "failed_to_get_model_info",
                    repo_id=request.repo_id,
                    error=str(e),
                )

        # Generate cache path or use local path
        cache_path = ""
        if request.source == ModelSource.HUGGINGFACE and request.repo_id:
            safe_name = request.repo_id.replace("/", "--") + f"--{request.quantization.value}"
            cache_path = f"huggingface/{safe_name}"
        elif request.source == ModelSource.LOCAL and request.local_path:
            # For local models, use the local path directly as cache_path
            cache_path = request.local_path

        # Extract name from repo_id or local_path
        name = ""
        if request.custom_name:
            name = request.custom_name
        elif request.repo_id:
            name = request.repo_id.split("/")[-1]
        elif request.local_path:
            name = request.local_path.rstrip("/").split("/")[-1]

        # Estimate memory requirement
        estimated_memory = 0
        if model_info and model_info.get("params"):
            estimated_memory = estimate_memory_mb(
                model_info["params"],
                request.quantization.value,
            )

        # Create database record in DOWNLOADING state
        model = await self.repository.create(
            name=name,
            source=request.source,
            repo_id=request.repo_id,
            local_path=request.local_path,
            quantization=request.quantization,
            cache_path=cache_path,
            trust_remote_code=request.trust_remote_code,
            status=ModelStatus.DOWNLOADING,
            params=model_info.get("params") if model_info else None,
            architecture=model_info.get("architecture") if model_info else None,
            estimated_memory_mb=estimated_memory if estimated_memory > 0 else None,
        )

        # For local models, mark as ready immediately (no download needed)
        if request.source == ModelSource.LOCAL and request.local_path:
            import os
            if not os.path.isdir(request.local_path):
                from millm.core.errors import InvalidLocalPathError
                raise InvalidLocalPathError(
                    f"Local path does not exist or is not a directory: {request.local_path}",
                    details={"path": request.local_path},
                )

            # Calculate disk size
            total_size = sum(
                f.stat().st_size
                for f in Path(request.local_path).rglob("*")
                if f.is_file()
            ) if Path(request.local_path).exists() else 0

            model = await self.repository.update(
                model.id,
                status=ModelStatus.READY,
                cache_path=request.local_path,
                disk_size_mb=int(total_size / (1024 * 1024)),
            )
            logger.info(
                "local_model_registered",
                model_id=model.id,
                local_path=request.local_path,
            )
            return model

        logger.info(
            "download_started",
            model_id=model.id,
            repo_id=request.repo_id,
            quantization=request.quantization.value,
        )

        # Start background download
        # Store the main loop for thread-safe async operations
        loop = asyncio.get_running_loop()
        self._main_loop = loop
        future = loop.run_in_executor(
            self._executor,
            self._download_worker,
            model.id,
            request,
        )
        self._active_downloads[model.id] = future

        return model

    def _download_worker(
        self,
        model_id: int,
        request: ModelDownloadRequest,
    ) -> str:
        """
        Background worker for downloading models.

        Runs in thread pool. Updates database and emits events upon completion/failure.
        Implements retry logic with exponential backoff for transient failures.
        """
        last_error: Optional[Exception] = None

        # Initialize progress tracking
        self._download_progress[model_id] = 0

        # Create progress callback that emits WebSocket events
        def on_progress(pct: float, downloaded: int, total: int) -> None:
            progress = int(pct)
            self._download_progress[model_id] = progress

            # Emit WebSocket progress event
            if self.emitter and self._main_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.emitter.emit_download_progress(
                            model_id=model_id,
                            progress=progress,
                            downloaded_bytes=downloaded,
                            total_bytes=total if total > 0 else None,
                        ),
                        self._main_loop,
                    )
                except Exception:
                    pass  # Don't let emit errors break downloading

            logger.debug(
                "download_progress_update",
                model_id=model_id,
                progress=progress,
                downloaded_bytes=downloaded,
                total_bytes=total,
            )

        try:
            for attempt in range(MAX_DOWNLOAD_RETRIES):
                try:
                    # Check if cancelled before starting
                    if model_id in self._cancelled_downloads:
                        self._cancelled_downloads.discard(model_id)
                        self._cleanup_partial_download(request.repo_id, request.quantization.value)
                        raise DownloadCancelledError("Download was cancelled")

                    logger.info(
                        "download_attempt",
                        model_id=model_id,
                        attempt=attempt + 1,
                        max_attempts=MAX_DOWNLOAD_RETRIES,
                    )

                    # Perform download with progress callback
                    cache_path = self.downloader.download(
                        repo_id=request.repo_id,
                        quantization=request.quantization.value,
                        progress_callback=on_progress,
                        token=request.hf_token,
                        trust_remote_code=request.trust_remote_code,
                    )

                    # Check if cancelled after download
                    if model_id in self._cancelled_downloads:
                        self._cancelled_downloads.discard(model_id)
                        self._cleanup_partial_download(request.repo_id, request.quantization.value)
                        raise DownloadCancelledError("Download was cancelled")

                    # Mark progress as 100%
                    self._download_progress[model_id] = 100

                    # Get disk size
                    disk_size_bytes = self.downloader.get_cache_size(
                        request.repo_id,
                        request.quantization.value,
                    )
                    disk_size_mb = disk_size_bytes // (1024 * 1024)

                    # Update database (thread-safe async call)
                    self._run_async_from_thread(
                        self._update_download_complete(
                            model_id=model_id,
                            cache_path=cache_path,
                            disk_size_mb=disk_size_mb,
                        )
                    )

                    logger.info(
                        "download_complete",
                        model_id=model_id,
                        cache_path=cache_path,
                        disk_size_mb=disk_size_mb,
                    )

                    return cache_path

                except DownloadCancelledError:
                    self._run_async_from_thread(self._update_download_cancelled(model_id))
                    raise

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "download_attempt_failed",
                        model_id=model_id,
                        attempt=attempt + 1,
                        error=str(e),
                        retrying=attempt < MAX_DOWNLOAD_RETRIES - 1,
                    )

                    # Check if error is retryable (network errors, timeouts)
                    if not self._is_retryable_error(e):
                        break

                    # Check if cancelled while waiting to retry
                    if model_id in self._cancelled_downloads:
                        self._cancelled_downloads.discard(model_id)
                        self._cleanup_partial_download(request.repo_id, request.quantization.value)
                        raise DownloadCancelledError("Download was cancelled")

                    # Calculate backoff delay
                    if attempt < MAX_DOWNLOAD_RETRIES - 1:
                        delay = min(
                            RETRY_BASE_DELAY * (2 ** attempt),
                            RETRY_MAX_DELAY,
                        )
                        logger.info(
                            "download_retry_delay",
                            model_id=model_id,
                            delay_seconds=delay,
                        )
                        time.sleep(delay)

                        # Clean up partial download before retry
                        self._cleanup_partial_download(request.repo_id, request.quantization.value)

            # All retries exhausted - clean up and report failure
            self._cleanup_partial_download(request.repo_id, request.quantization.value)

            if last_error:
                logger.error(
                    "download_failed",
                    model_id=model_id,
                    error=str(last_error),
                    attempts=MAX_DOWNLOAD_RETRIES,
                )
                self._run_async_from_thread(
                    self._update_download_error(
                        model_id=model_id,
                        error_code=getattr(last_error, "code", "DOWNLOAD_FAILED"),
                        error_message=str(last_error),
                    )
                )
                raise last_error

            return ""

        finally:
            # Clean up download tracking
            self._active_downloads.pop(model_id, None)
            self._cancelled_downloads.discard(model_id)
            self._download_progress.pop(model_id, None)

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable (transient network issues).

        Args:
            error: The exception to check

        Returns:
            True if the error is transient and should be retried
        """
        # Non-retryable errors
        non_retryable_codes = {
            "REPO_NOT_FOUND",
            "GATED_MODEL_NO_TOKEN",
            "INVALID_HF_TOKEN",
            "INVALID_LOCAL_PATH",
        }

        error_code = getattr(error, "code", "")
        if error_code in non_retryable_codes:
            return False

        # Check for common retryable error types
        error_name = type(error).__name__
        retryable_names = {
            "ConnectionError",
            "TimeoutError",
            "OSError",
            "IOError",
            "HTTPError",
            "RequestException",
        }

        return error_name in retryable_names or "timeout" in str(error).lower()

    def _cleanup_partial_download(self, repo_id: str, quantization: str) -> None:
        """
        Clean up partially downloaded files.

        Args:
            repo_id: HuggingFace repository ID
            quantization: Quantization type
        """
        try:
            self.downloader.delete_cached_model(repo_id, quantization)
            logger.info(
                "partial_download_cleaned",
                repo_id=repo_id,
                quantization=quantization,
            )
        except Exception as e:
            logger.warning(
                "partial_download_cleanup_failed",
                repo_id=repo_id,
                quantization=quantization,
                error=str(e),
            )

    async def _update_download_complete(
        self,
        model_id: int,
        cache_path: str,
        disk_size_mb: int,
    ) -> None:
        """Update database and emit event when download completes."""
        await self.repository.update(
            model_id,
            cache_path=cache_path,
            disk_size_mb=disk_size_mb,
            status=ModelStatus.READY,
        )

        if self.emitter:
            await self.emitter.emit_download_complete(
                model_id=model_id,
                local_path=cache_path,
            )

    async def _update_download_cancelled(self, model_id: int) -> None:
        """Update database and emit event when download is cancelled."""
        await self.repository.update_status(
            model_id,
            status=ModelStatus.ERROR,
            error_message="Download cancelled by user",
        )

        if self.emitter:
            await self.emitter.emit_download_error(
                model_id=model_id,
                error_code="DOWNLOAD_CANCELLED",
                error_message="Download cancelled by user",
            )

    async def _update_download_error(
        self,
        model_id: int,
        error_code: str,
        error_message: str,
    ) -> None:
        """Update database and emit event when download fails."""
        await self.repository.update_status(
            model_id,
            status=ModelStatus.ERROR,
            error_message=error_message,
        )

        if self.emitter:
            await self.emitter.emit_download_error(
                model_id=model_id,
                error_code=error_code,
                error_message=error_message,
            )

    async def cancel_download(self, model_id: int) -> Model:
        """
        Cancel an in-progress download.

        Args:
            model_id: The model's database ID

        Returns:
            The updated model record

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model = await self.get_model(model_id)

        if model.status != ModelStatus.DOWNLOADING:
            return model

        self._cancelled_downloads.add(model_id)

        future = self._active_downloads.get(model_id)
        if future and not future.done():
            future.cancel()

        model = await self.repository.update_status(
            model_id,
            status=ModelStatus.ERROR,
            error_message="Download cancelled by user",
        )

        logger.info("download_cancelled", model_id=model_id)

        return model

    # =========================================================================
    # Load/Unload Operations
    # =========================================================================

    async def load_model(self, model_id: int) -> Model:
        """
        Load a model into GPU memory.

        If another model is loaded, it will be automatically unloaded first.
        Progress updates are sent via WebSocket.

        Args:
            model_id: The model's database ID

        Returns:
            The updated model record (status will be LOADED)

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelBusyError: If another load is in progress
        """
        model = await self.get_model(model_id)

        # Check if model is ready to load
        if model.status == ModelStatus.DOWNLOADING:
            raise ModelBusyError(
                f"Model {model_id} is still downloading",
                details={"model_id": model_id, "status": model.status.value},
            )

        if model.status == ModelStatus.LOADING:
            raise ModelBusyError(
                f"Model {model_id} is already being loaded",
                details={"model_id": model_id},
            )

        if model.status == ModelStatus.LOADED:
            raise ModelAlreadyLoadedError(
                f"Model {model_id} is already loaded",
                details={"model_id": model_id},
            )

        if model.status == ModelStatus.ERROR:
            # Allow retry from error state - reset to ready first
            logger.info(
                "retrying_errored_model",
                model_id=model_id,
                previous_error=model.error_message,
            )
            model = await self.repository.update_status(
                model_id, status=ModelStatus.READY, error_message=None
            )

        # Check if another load is in progress
        if self._loading_model_id is not None:
            raise ModelBusyError(
                f"Another model ({self._loading_model_id}) is currently being loaded",
                details={"loading_model_id": self._loading_model_id},
            )

        # Unload any currently loaded model
        if self.loader.is_loaded:
            current_model_id = self.loader.loaded_model_id
            logger.info(
                "auto_unloading_model",
                current_model_id=current_model_id,
                new_model_id=model_id,
            )
            await self.unload_model(current_model_id)

        # Update status to LOADING
        model = await self.repository.update_status(model_id, status=ModelStatus.LOADING)
        self._loading_model_id = model_id

        logger.info(
            "load_started",
            model_id=model_id,
            cache_path=model.cache_path,
            quantization=model.quantization.value,
        )

        # Emit progress event
        if self.emitter:
            await self.emitter.emit_load_progress(
                model_id=model_id,
                stage="initializing",
                progress=0,
            )

        # Start background load
        # Store the main loop for thread-safe async operations
        loop = asyncio.get_running_loop()
        self._main_loop = loop
        loop.run_in_executor(
            self._executor,
            self._load_worker,
            model_id,
            model.name,
            model.cache_path,
            model.quantization.value,
            model.estimated_memory_mb or 0,
            model.trust_remote_code,
        )

        return model

    def _load_worker(
        self,
        model_id: int,
        model_name: str,
        cache_path: str,
        quantization: str,
        estimated_memory_mb: int,
        trust_remote_code: bool,
    ) -> None:
        """
        Background worker for loading models.

        Runs in thread pool. Updates database and emits events upon completion/failure.
        """
        try:
            # Load the model
            from millm.core.config import settings

            # Handle both absolute and relative cache paths
            # (database may store either depending on when model was downloaded)
            if os.path.isabs(cache_path):
                full_cache_path = cache_path
            else:
                full_cache_path = f"{settings.MODEL_CACHE_DIR}/{cache_path}"

            loaded = self.loader.load(
                model_id=model_id,
                model_name=model_name,
                cache_path=full_cache_path,
                quantization=quantization,
                estimated_memory_mb=estimated_memory_mb,
                trust_remote_code=trust_remote_code,
                torch_compile=settings.TORCH_COMPILE,
                torch_compile_mode=settings.TORCH_COMPILE_MODE,
            )

            # Update database (thread-safe async call)
            self._run_async_from_thread(
                self._update_load_complete(
                    model_id=model_id,
                    memory_used_mb=loaded.memory_used_mb,
                )
            )

            logger.info(
                "load_complete",
                model_id=model_id,
                memory_used_mb=loaded.memory_used_mb,
            )

            # Notify inference service (starts CBM if enabled)
            try:
                from millm.api.dependencies import get_inference_service
                get_inference_service().on_model_loaded()
            except Exception:
                pass

        except Exception as e:
            logger.error("load_failed", model_id=model_id, error=str(e))
            self._run_async_from_thread(
                self._update_load_error(
                    model_id=model_id,
                    error_code=getattr(e, "code", "MODEL_LOAD_FAILED"),
                    error_message=str(e),
                )
            )

        finally:
            self._loading_model_id = None

    async def _update_load_complete(
        self,
        model_id: int,
        memory_used_mb: int,
    ) -> None:
        """Update database and emit event when load completes."""
        await self.repository.update(
            model_id,
            status=ModelStatus.LOADED,
            loaded_at=datetime.utcnow(),
        )

        if self.emitter:
            await self.emitter.emit_load_complete(
                model_id=model_id,
                memory_used_mb=memory_used_mb,
            )

    async def _update_load_error(
        self,
        model_id: int,
        error_code: str,
        error_message: str,
    ) -> None:
        """Update database and emit event when load fails."""
        await self.repository.update_status(
            model_id,
            status=ModelStatus.ERROR,
            error_message=error_message,
        )

        if self.emitter:
            await self.emitter.emit_load_error(
                model_id=model_id,
                error_code=error_code,
                error_message=error_message,
            )

    async def unload_model(self, model_id: int, timeout: float = UNLOAD_TIMEOUT) -> Model:
        """
        Unload a model from GPU memory with graceful timeout.

        Args:
            model_id: The model's database ID
            timeout: Maximum time in seconds to wait for unload (default 30s)

        Returns:
            The updated model record (status will be READY)

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelNotLoadedError: If model is not loaded
        """
        model = await self.get_model(model_id)

        # Check if this model is actually loaded
        if self.loader.loaded_model_id != model_id:
            raise ModelNotLoadedError(
                f"Model {model_id} is not currently loaded",
                details={"model_id": model_id},
            )

        logger.info("unload_started", model_id=model_id, timeout=timeout)

        # Stop CBM before unloading model
        try:
            from millm.api.dependencies import get_inference_service
            get_inference_service().on_model_unloading()
        except Exception:
            pass

        # Wait for pending inference requests to drain
        try:
            from millm.api.dependencies import get_inference_service
            inference_svc = get_inference_service()
            queue = inference_svc.request_queue
            if queue.pending_count > 0:
                logger.info("waiting_for_pending_inference", pending=queue.pending_count)
                for _ in range(50):  # Wait up to 5 seconds
                    if queue.pending_count == 0:
                        break
                    await asyncio.sleep(0.1)
        except Exception:
            pass  # Don't block unload if queue check fails

        # Unload from GPU with timeout
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._unload_worker, model_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "unload_timeout",
                model_id=model_id,
                timeout=timeout,
            )
            # Force unload anyway
            self.loader.unload()

        # Auto-unlock on unload
        model = await self.repository.update(
            model_id,
            status=ModelStatus.READY,
            loaded_at=None,
            locked=False,
        )

        # Emit event
        if self.emitter:
            await self.emitter.emit_unload_complete(model_id=model_id)

        logger.info("unload_complete", model_id=model_id)

        return model

    def _unload_worker(self, model_id: int) -> None:
        """
        Background worker for unloading models.

        This allows the unload to be properly awaited with a timeout.
        """
        logger.debug("unload_worker_started", model_id=model_id)
        self.loader.unload()
        logger.debug("unload_worker_complete", model_id=model_id)

    async def get_loaded_model_id(self) -> Optional[int]:
        """
        Get the ID of the currently loaded model.

        Returns:
            Model ID if a model is loaded, None otherwise.
        """
        return self.loader.loaded_model_id

    def get_loaded_model_info(self) -> Optional[dict]:
        """
        Get runtime info for the currently loaded model.

        Returns:
            Dict with num_parameters, memory_footprint, device, dtype if loaded,
            None otherwise.
        """
        if not self.loader.is_loaded or self.loader.state.current is None:
            return None

        loaded = self.loader.state.current
        return {
            "model_id": loaded.model_id,
            "num_parameters": loaded.num_parameters,
            "memory_footprint": loaded.memory_used_mb * 1024 * 1024,  # Convert MB to bytes
            "device": loaded.device,
            "dtype": loaded.dtype,
        }

    # =========================================================================
    # Lock/Unlock Operations
    # =========================================================================

    async def lock_model(self, model_id: int) -> Model:
        """
        Lock a model to prevent auto-unload (used for steering).

        The model must be in LOADED state. Only one model can be locked at a time.

        Args:
            model_id: The model's database ID

        Returns:
            The updated model record

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelNotLoadedError: If model is not loaded
            ModelLockedError: If another model is already locked
        """
        model = await self.get_model(model_id)

        if model.status != ModelStatus.LOADED:
            raise ModelNotLoadedError(
                f"Model {model_id} must be loaded before it can be locked",
                details={"model_id": model_id, "status": model.status.value},
            )

        # Check if another model is already locked
        locked = await self.repository.get_locked_model()
        if locked and locked.id != model_id:
            raise ModelLockedError(
                f"Model '{locked.name}' (ID {locked.id}) is already locked. Unlock it first.",
                details={"locked_model_id": locked.id, "locked_model_name": locked.name},
            )

        model = await self.repository.update(model_id, locked=True)
        logger.info("model_locked", model_id=model_id)
        return model

    async def unlock_model(self, model_id: int) -> Model:
        """
        Unlock a model to allow auto-unload.

        Args:
            model_id: The model's database ID

        Returns:
            The updated model record

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model = await self.get_model(model_id)
        model = await self.repository.update(model_id, locked=False)
        logger.info("model_unlocked", model_id=model_id)
        return model

    async def get_locked_model(self) -> Model | None:
        """
        Get the currently locked model (if any).

        Returns:
            The locked model or None.
        """
        return await self.repository.get_locked_model()

    async def load_model_and_wait(
        self,
        model_id: int,
        timeout: float = 180.0,
    ) -> Model:
        """
        Load a model and wait for it to complete loading.

        Used by the OpenAI-compatible endpoints for auto-load on demand.
        Calls load_model() then polls until status becomes LOADED or ERROR.

        Args:
            model_id: The model's database ID
            timeout: Maximum time in seconds to wait for load (default 180s)

        Returns:
            The loaded model record

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelBusyError: If another load is in progress
            ModelLockedError: If a different model is locked
            asyncio.TimeoutError: If loading exceeds timeout
        """
        model = await self.get_model(model_id)

        # If already loaded, return immediately
        if model.status == ModelStatus.LOADED and self.loader.loaded_model_id == model_id:
            return model

        # Check if locked model prevents loading this one
        locked = await self.repository.get_locked_model()
        if locked and locked.id != model_id:
            raise ModelLockedError(
                f"Model '{locked.name}' is locked for steering. "
                f"Cannot auto-load '{model.name}'.",
                details={"locked_model_id": locked.id, "requested_model_id": model_id},
            )

        # Start loading (this returns immediately with status=LOADING)
        await self.load_model(model_id)

        # Poll until loaded or error
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            await asyncio.sleep(0.5)
            updated = await self.repository.get_by_id(model_id)
            if updated is None:
                raise ModelNotFoundError(
                    f"Model {model_id} disappeared during loading",
                    details={"model_id": model_id},
                )
            if updated.status == ModelStatus.LOADED:
                return updated
            if updated.status == ModelStatus.ERROR:
                raise ModelBusyError(
                    f"Model failed to load: {updated.error_message}",
                    details={"model_id": model_id, "error": updated.error_message},
                )

        raise asyncio.TimeoutError(
            f"Model {model_id} did not finish loading within {timeout}s"
        )

    async def find_model_by_name(self, name: str) -> Model | None:
        """
        Find a model by its display name.

        Args:
            name: The model's display name

        Returns:
            The model or None if not found
        """
        return await self.repository.find_by_name(name)

    async def get_available_models(self) -> list[Model]:
        """
        Get all models available for use (READY, LOADED, LOADING).

        Returns:
            List of available models
        """
        return await self.repository.get_available_models()

    # =========================================================================
    # Delete Operations
    # =========================================================================

    async def delete_model(self, model_id: int) -> bool:
        """
        Delete a model from the database and disk.

        Args:
            model_id: The model's database ID

        Returns:
            True if model was deleted

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelBusyError: If model is currently loaded
        """
        model = await self.get_model(model_id)

        # Check if model is loaded
        if model.status == ModelStatus.LOADED or self.loader.loaded_model_id == model_id:
            raise ModelBusyError(
                f"Cannot delete model {model_id} while it is loaded. Unload it first.",
                details={"model_id": model_id},
            )

        # Cancel any active download
        if model.status == ModelStatus.DOWNLOADING:
            await self.cancel_download(model_id)

        # Delete cached files if from HuggingFace
        if model.source == ModelSource.HUGGINGFACE and model.repo_id:
            self.downloader.delete_cached_model(
                model.repo_id,
                model.quantization.value,
            )

        # Delete from database
        deleted = await self.repository.delete(model_id)

        logger.info("model_deleted", model_id=model_id)

        return deleted

    def shutdown(self) -> None:
        """Clean up resources on application shutdown."""
        # Unload any loaded model
        if self.loader.is_loaded:
            self.loader.unload()

        # Shutdown executor
        self._executor.shutdown(wait=False)
