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
    GpuNotFoundError,
    ModelAlreadyExistsError,
    ModelAlreadyLoadedError,
    ModelBusyError,
    ModelLoadError,
    ModelLockedError,
    ModelNotFoundError,
    ModelNotLoadedError,
)
from millm.core.errors import GgufTensorSplitError
from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from millm.db.repositories.model_repository import ModelRepository
from millm.ml.memory_utils import estimate_memory_mb
from millm.ml.gpu_placement import (
    GpuRequest,
    list_gpus,
    parse_gpu_request,
    project_free_after_unload,
)
from millm.ml.gguf_catalog import coarse_quantization
from millm.ml.model_downloader import ModelDownloader, _safe_variant
from millm.ml.model_loader import (
    GGUF_MIN_CONTEXT,
    ModelLoader,
    _gguf_shard_rule,
    _gguf_tensor_split,
    plan_gguf_placement,
    plan_transformers_load,
    preflight_split,
)
from millm.sockets.progress import ProgressEmitter

logger = structlog.get_logger()


def resolve_cache_path(cache_path: str) -> str:
    """A model row's cache path as the loader opens it.

    The database stores either an absolute path or one relative to
    MODEL_CACHE_DIR, depending on when the model was downloaded. The pre-unload
    check and the background load must read the SAME checkpoint, so both resolve
    it here.
    """
    from millm.core.config import settings

    if os.path.isabs(cache_path):
        return cache_path
    return f"{settings.MODEL_CACHE_DIR}/{cache_path}"


#: Download progress, keyed by model id, SHARED ACROSS SERVICE INSTANCES.
#:
#: This has to live outside the instance because `get_model_service` constructs
#: a brand-new `ModelService` for every request. A dict on `self` is therefore
#: written by the instance that started the download and read by a different,
#: empty one — so `GET /api/models` reported `download_progress: null` for the
#: entire duration of every download that has ever run. Observed on a 51.8 GB
#: Qwen3.8-27B pull: the bytes were landing at 110 MB/s and the API said
#: nothing, which is indistinguishable from a download that has died.
#:
#: In-process is sufficient and durable enough. The download runs in a thread of
#: this same process, and a restart ends it anyway — startup resets any model
#: left in DOWNLOADING (STALE_STATE_RESETS in main.py; that claim was made here
#: long before anything implemented it, and a row sat stuck for half an hour
#: because of it). Persisting to a column would add a write every poll interval
#: to reconstruct state that cannot outlive the work it describes.
_DOWNLOAD_PROGRESS: dict[int, int] = {}


#: The model whose load is in progress, SHARED ACROSS SERVICE INSTANCES, for the
#: reason `_DOWNLOAD_PROGRESS` is: `get_model_service` builds a new ModelService
#: per request. The one-load-at-a-time slot lived on `self`, so it was only ever
#: seen by the request that claimed it. A double-clicked Switch, or two OpenAI
#: requests naming different models, each got a fresh instance whose slot was
#: empty, both passed the busy check and both loads ran at once. Review round 3,
#: 2026-09-14. Read and written only through `ModelService._loading_model_id`.
_LOAD_SLOT: dict[str, Optional[int]] = {"model_id": None}


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
        inference_service: Optional[Any] = None,
    ) -> None:
        """
        Initialize the model service.

        Args:
            repository: Model database repository
            downloader: Model downloader for HuggingFace
            loader: Model loader for GPU loading
            emitter: Progress event emitter for WebSocket updates
            inference_service: Optional InferenceService ref for CBM lifecycle hooks.
                When provided, avoids importing from millm.api (layer violation).
        """
        self.repository = repository
        self.downloader = downloader
        self.loader = loader or ModelLoader()
        self.emitter = emitter
        self._inference_service = inference_service

        # Thread pool for background tasks
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="millm-",
        )

        # Track active downloads for cancellation
        self._active_downloads: dict[int, Future[str]] = {}
        self._cancelled_downloads: set[int] = set()

        # Active loads are tracked in the module-level `_LOAD_SLOT` (see
        # `_loading_model_id`). NOT reset here: every request constructs a
        # service, so resetting in __init__ would release a load in progress.

        # Reference to main event loop for thread-safe async operations
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Progress lives in the module-level `_DOWNLOAD_PROGRESS`, not here —
        # see the note on that dict. An instance attribute is invisible to the
        # next request, which gets a different ModelService.

    @property
    def _loading_model_id(self) -> Optional[int]:
        """The model whose load is in progress, in ANY service instance."""
        return _LOAD_SLOT["model_id"]

    @_loading_model_id.setter
    def _loading_model_id(self, model_id: Optional[int]) -> None:
        _LOAD_SLOT["model_id"] = model_id

    def _release_load_slot(self, model_id: int) -> None:
        """Release the slot only if `model_id` still holds it."""
        if _LOAD_SLOT["model_id"] == model_id:
            _LOAD_SLOT["model_id"] = None

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
        return _DOWNLOAD_PROGRESS.get(model_id)

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
            revision=request.revision,
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
        # The coarse level is DERIVED from the GGUF label, never taken from the
        # request. It is a function of the label, so letting a caller send both
        # lets them disagree — and every GGUF download would otherwise be filed
        # under whatever the quantization dropdown defaulted to.
        if request.gguf_label:
            derived = QuantizationType(coarse_quantization(request.gguf_label))
            if derived != request.quantization:
                logger.info(
                    "gguf_quantization_derived",
                    gguf_label=request.gguf_label,
                    requested=request.quantization.value,
                    derived=derived.value,
                )
            request = request.model_copy(update={"quantization": derived})

        # Check for existing model with same repo_id and quantization
        if request.source == ModelSource.HUGGINGFACE and request.repo_id:
            existing = await self.repository.find_by_repo_quantization(
                repo_id=request.repo_id,
                quantization=request.quantization,
                gguf_label=request.gguf_label or "",
            )
            if existing:
                described = request.gguf_label or request.quantization.value
                raise ModelAlreadyExistsError(
                    f"Model {request.repo_id} with {described} quantization already exists",
                    details={
                        "repo_id": request.repo_id,
                        "quantization": request.quantization.value,
                        "gguf_label": request.gguf_label or "",
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
            # Must match ModelDownloader._get_local_dir, including the variant.
            # Q4_K_M, Q4_K_S and Q4_0 are all "Q4"; without the suffix they share
            # one directory and deleting any one destroys the others.
            if request.gguf_label:
                safe_name += f"--{_safe_variant(request.gguf_label)}"
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
            # `repo:QUANT`, the convention Ollama already taught everyone.
            #
            # Several quantizations of one repository can coexist since the
            # uniqueness constraint was widened, and they are DIFFERENT MODELS —
            # different sizes, different quality. Leaving them to share a name
            # made `find_by_name` raise "Multiple rows were found" and every
            # OpenAI request naming that model return 500, with the model
            # loaded and serving perfectly the whole time.
            #
            # It also makes both quantizations separately selectable in an
            # OpenAI client: previously one shadowed the other in /v1/models.
            if request.gguf_label:
                name = f"{name}:{request.gguf_label}"
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
            # '' not None — the column is part of uq_repo_quantization, and
            # Postgres treats NULLs as distinct, which would defeat it.
            gguf_label=request.gguf_label or "",
            gguf_files=list(request.gguf_files) if request.gguf_files else None,
            revision=request.revision,
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
        _DOWNLOAD_PROGRESS[model_id] = 0

        # Create progress callback that emits WebSocket events
        def on_progress(
            pct: float, downloaded: int, total: int, speed_bps: float = 0.0
        ) -> None:
            progress = int(pct)
            _DOWNLOAD_PROGRESS[model_id] = progress

            # Emit WebSocket progress event
            if self.emitter and self._main_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.emitter.emit_download_progress(
                            model_id=model_id,
                            progress=progress,
                            downloaded_bytes=downloaded,
                            total_bytes=total if total > 0 else None,
                            speed_bps=int(speed_bps) if speed_bps > 0 else None,
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
                speed_bps=int(speed_bps),
            )

        try:
            for attempt in range(MAX_DOWNLOAD_RETRIES):
                try:
                    # Check if cancelled before starting
                    if model_id in self._cancelled_downloads:
                        self._cancelled_downloads.discard(model_id)
                        self._cleanup_partial_download(
                            request.repo_id, request.quantization.value, request.gguf_label
                        )
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
                        file_paths=request.gguf_files,
                        variant=request.gguf_label,
                        revision=request.revision,
                    )

                    # Check if cancelled after download
                    if model_id in self._cancelled_downloads:
                        self._cancelled_downloads.discard(model_id)
                        self._cleanup_partial_download(
                            request.repo_id,
                            request.quantization.value,
                            request.gguf_label,
                        )
                        raise DownloadCancelledError("Download was cancelled")

                    # Mark progress as 100%
                    _DOWNLOAD_PROGRESS[model_id] = 100

                    # Get disk size
                    disk_size_bytes = self.downloader.get_cache_size(
                        request.repo_id,
                        request.quantization.value,
                        request.gguf_label,
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
                        self._cleanup_partial_download(
                            request.repo_id, request.quantization.value, request.gguf_label
                        )
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
                        self._cleanup_partial_download(
                            request.repo_id, request.quantization.value, request.gguf_label
                        )

            # All retries exhausted - clean up and report failure
            self._cleanup_partial_download(
                request.repo_id, request.quantization.value, request.gguf_label
            )

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
            _DOWNLOAD_PROGRESS.pop(model_id, None)

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

    def _cleanup_partial_download(
        self, repo_id: str, quantization: str, variant: Optional[str] = None
    ) -> None:
        """
        Clean up partially downloaded files.

        Args:
            repo_id: HuggingFace repository ID
            quantization: Quantization type
            variant: The quant label, when several downloads share a coarse
                quantization. This must reach delete_cached_model: without it
                a cancelled Q4_K_M download deletes whatever "Q4" resolves to,
                which is a DIFFERENT, complete model.
        """
        try:
            self.downloader.delete_cached_model(repo_id, quantization, variant)
            logger.info(
                "partial_download_cleaned",
                repo_id=repo_id,
                quantization=quantization,
                variant=variant,
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

    async def load_model(self, model_id: int, gpu: GpuRequest = None) -> Model:
        """
        Load a model into GPU memory.

        If another model is loaded, it will be automatically unloaded first.
        Progress updates are sent via WebSocket.

        Args:
            model_id: The model's database ID
            gpu: None or "auto" (the most-free card that fits), a CUDA index,
                or a GPU UUID. A named card that does not exist, or that cannot
                hold the model even with the resident model's memory given
                back, is refused here — before anything is unloaded.

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

        # One load at a time, and the slot is CLAIMED here, with no await
        # between the check and the claim. It used to be set only after the
        # resident model's unload was awaited (which drains pending inference
        # for up to five seconds), so a second load arriving in that window —
        # a double-clicked Switch, or two OpenAI requests naming different
        # models — passed this same check. Both were submitted to the
        # two-worker executor, ran at once, and the second replaced the first
        # in LoadedModelState without unloading it.
        if self._loading_model_id is not None:
            raise ModelBusyError(
                f"Another model ({self._loading_model_id}) is currently being loaded",
                details={"loading_model_id": self._loading_model_id},
            )
        self._loading_model_id = model_id
        # Whether this request has set the row to LOADING, so a failure before
        # the background load is submitted can put it back (see the except).
        marked_loading = False

        try:
            # The card, checked NOW: before any state below changes, and above
            # all before the resident model is unloaded to make room. The load
            # itself runs in the background, so a refusal found only there
            # arrived as an ERROR status after the served model was gone.
            try:
                wanted = parse_gpu_request(gpu)
            except ValueError as e:
                raise GpuNotFoundError(str(e), details={"requested": gpu}) from e

            def _precheck() -> None:
                self._precheck_placement(model, wanted)

            # IN A WORKER THREAD. The pre-check reads the cards through
            # nvidia-smi, which can take up to its five-second timeout; called
            # inline it stalled the event loop, and every request with it, on
            # each load. Review round 3, 2026-09-14.
            await asyncio.to_thread(_precheck)

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
            marked_loading = True

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
                # The chosen GGUF file, when this row is one. Its presence is
                # what routes the load to llama.cpp instead of transformers.
                (model.gguf_files or [None])[0],
                wanted,
            )
        except BaseException as exc:
            # Nothing was submitted, so no worker's `finally` will release the
            # slot. Without this a refused or failed load left the service
            # refusing every later load as busy.
            self._release_load_slot(model_id)
            if marked_loading:
                # ...and no worker will move the row off LOADING, which this
                # method refuses as busy: that model could not be loaded again
                # until miLLM restarted. A cancelled start is not a failure.
                # Review round 3, 2026-09-14.
                cancelled = isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                try:
                    await self.repository.update_status(
                        model_id,
                        status=ModelStatus.READY if cancelled else ModelStatus.ERROR,
                        error_message=None if cancelled else f"Load did not start: {exc}",
                    )
                except Exception as restore_error:  # noqa: BLE001 - never mask the original error
                    logger.warning(
                        "load_status_restore_failed",
                        model_id=model_id,
                        error=str(restore_error),
                    )
            raise

        return model

    def _precheck_placement(self, model: Model, wanted: Any) -> None:
        """Refuse, before unloading anything, a load that cannot be placed.

        Judged against what each card WILL have once the resident model is
        gone: live free memory plus that model's recorded usage on the card.
        Without the add-back, switching from a model on the 3090 to a bigger
        one on the same card would be refused while it fits.

        The background load decides again after the unload, against live
        memory, and that decision is authoritative — free memory can change in
        between. This check only refuses early what would be refused there.

        Raises:
            GpuNotFoundError: the named card is not visible.
            InsufficientMemoryError: the named card cannot hold the model, a
                requested split across every card ('all') cannot, no split
                across the cards can hold a transformers model, or the split's
                real device map (preflight_split) puts part of it on the CPU or
                disk.
            UnsupportedQuantizationError: Q2 on a transformers checkpoint that
                is not pre-quantized.
            GgufTensorSplitError: GGUF_TENSOR_SPLIT names fewer cards than the
                GGUF split must use, or more than could take part in any split.
        """
        from millm.core.config import parse_gguf_tensor_split, settings

        resident = self.loader.state.current if self.loader.is_loaded else None
        gpus = project_free_after_unload(
            list_gpus(), getattr(resident, "memory_by_device_mb", None)
        )
        # The checkpoint the background load will open, resolved the same way.
        cache_path = resolve_cache_path(model.cache_path) if model.cache_path else None

        gguf_file = (model.gguf_files or [None])[0]
        if gguf_file:
            # GGUF on Auto is refused here only for a GGUF_TENSOR_SPLIT no split
            # could match. With no card that has room it runs on the CPU; a
            # model every card together is short of is still attempted with
            # every layer on the GPUs (there is no partial CPU offload — the
            # claim that it "spills" was wrong; review round 2). A named card is
            # sized on weights and overhead at the smallest context with no KV
            # term — a lower bound on what the loader asks for, so this cannot
            # refuse a card the loader would accept. Reading the KV size would
            # load the file's metadata inside the request.
            weights_mb = 0
            if cache_path:
                path = Path(cache_path) / gguf_file
                if path.is_file():
                    weights_mb = int(path.stat().st_size / (1024 * 1024))
            placement = plan_gguf_placement(weights_mb, None, 0, requested=wanted, gpus=gpus)
            configured = parse_gguf_tensor_split(settings.GGUF_TENSOR_SPLIT)
            if configured is not None and placement.is_shard:
                # The loader sizes its split with the KV cache on top of these
                # weights, so it spans at least these cards and at most every
                # card with any room — and it refuses a GGUF_TENSOR_SPLIT whose
                # length differs from the cards it spans. Found there, the served
                # model was already unloaded. Review round 1 refused a list
                # SHORTER than the lower bound; a list LONGER than every card
                # that could take part ("1,1,1" on two GPUs) matches no load
                # either. Between the bounds the loader decides. Review round 2.
                card_limit_mb = _gguf_shard_rule(weights_mb, None, GGUF_MIN_CONTEXT).limit_mb
                most_cards = sum(1 for gpu in gpus if card_limit_mb(gpu) > 0)
                if len(configured) > most_cards:
                    raise GgufTensorSplitError(
                        f"GGUF_TENSOR_SPLIT has {len(configured)} value(s), but at most "
                        f"{most_cards} GPU(s) can take part in a split here. Give one "
                        "proportion per card used, in index order, or leave it empty to "
                        "split by free memory.",
                        details={
                            "tensor_split": configured,
                            "max_cards": most_cards,
                            "placement": placement.to_dict(),
                        },
                    )
                if len(configured) < len(placement.gpu_indices):
                    _gguf_tensor_split(placement, configured)
            return

        placement = plan_transformers_load(
            model.estimated_memory_mb or 0,
            model.quantization.value,
            requested=wanted,
            gpus=gpus,
            cache_path=cache_path,
            trust_remote_code=bool(getattr(model, "trust_remote_code", False)),
        )
        # The map itself, computed from the checkpoint's config with no weight
        # read: a split the estimate accepts can still map to disk, and the load
        # would find that only after the unload. Review round 2, 2026-09-14.
        preflight_split(
            model.name,
            cache_path,
            model.quantization.value,
            placement,
            trust_remote_code=bool(getattr(model, "trust_remote_code", False)),
        )

    def _load_worker(
        self,
        model_id: int,
        model_name: str,
        cache_path: str,
        quantization: str,
        estimated_memory_mb: int,
        trust_remote_code: bool,
        gguf_file: Optional[str] = None,
        gpu: GpuRequest = None,
    ) -> None:
        """
        Background worker for loading models.

        Runs in thread pool. Updates database and emits events upon completion/failure.
        """
        try:
            # Load the model
            from millm.core.config import settings

            # Absolute or relative to MODEL_CACHE_DIR; the pre-unload check
            # resolves it the same way, so both read the same checkpoint.
            full_cache_path = resolve_cache_path(cache_path)

            # Resolve TORCH_COMPILE=None (auto-detect) to a concrete bool.
            # Auto: enable for CUDA + non-bitsandbytes models; disable otherwise.
            # A model that ENDS UP split across devices is also not compiled,
            # but that is only knowable after from_pretrained, so the loader
            # (ModelLoadContext.load) enforces it rather than this resolution.
            torch_compile_setting = settings.TORCH_COMPILE
            if torch_compile_setting is None:
                try:
                    import torch as _torch
                    is_bnb = quantization.upper() in ("Q4", "Q8", "Q2")
                    torch_compile_resolved = _torch.cuda.is_available() and not is_bnb
                except Exception:
                    torch_compile_resolved = False
                if torch_compile_resolved:
                    logger.info(
                        "torch_compile_auto_enabled",
                        quantization=quantization,
                    )
                else:
                    logger.info(
                        "torch_compile_auto_disabled",
                        reason="bitsandbytes_quantization" if quantization.upper() in ("Q4", "Q8", "Q2") else "no_cuda",
                        quantization=quantization,
                    )
            else:
                torch_compile_resolved = torch_compile_setting

            loaded = self.loader.load(
                model_id=model_id,
                model_name=model_name,
                cache_path=full_cache_path,
                quantization=quantization,
                estimated_memory_mb=estimated_memory_mb,
                trust_remote_code=trust_remote_code,
                torch_compile=torch_compile_resolved,
                torch_compile_mode=settings.TORCH_COMPILE_MODE,
                gguf_file=gguf_file,
                gpu=gpu,
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
                svc = self._inference_service
                if svc is None:
                    from millm.api.dependencies import get_inference_service
                    svc = get_inference_service()
                svc.on_model_loaded()
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
            self._release_load_slot(model_id)

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
            svc = self._inference_service
            if svc is None:
                from millm.api.dependencies import get_inference_service
                svc = get_inference_service()
            svc.on_model_unloading()
        except Exception:
            pass

        # Wait for pending inference requests to drain
        try:
            svc = self._inference_service
            if svc is None:
                from millm.api.dependencies import get_inference_service
                svc = get_inference_service()
            inference_svc = svc
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
            # Additive: which card(s), how the choice was made, and what the
            # load consumed on each card.
            "gpu_indices": list(loaded.gpu_indices),
            "memory_by_device_mb": dict(loaded.memory_by_device_mb),
            "placement": (
                {**loaded.placement, "memory_by_device_mb": dict(loaded.memory_by_device_mb)}
                if loaded.placement
                else None
            ),
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

        # Through the exclusive writer even though the check above has already
        # ruled out a rival lock: that check now ignores locks held by models
        # that are not LOADED, so a leaked flag on a non-resident row can no
        # longer block a legitimate lock — and this is what then clears it.
        model = await self.repository.set_exclusive_lock(model_id)
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
                # A load that FAILED, not one still running: as ModelBusyError
                # the OpenAI routes answered it "Another model load is already
                # in progress; retry once it finishes" — for a load that had
                # already finished, unsuccessfully, and would fail again.
                # Review round 3, 2026-09-14.
                raise ModelLoadError(
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
            # The variant MUST be passed. The cache directory is
            # repo--quantization[--gguf_label], so resolving it without the
            # label computes a path that does not exist: deleting a GGUF model
            # would remove the row and leave the files — 5.4 GB in the case that
            # found this — orphaned on disk with nothing left pointing at them.
            self.downloader.delete_cached_model(
                model.repo_id,
                model.quantization.value,
                model.gguf_label or None,
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
