"""
SAE service for orchestrating SAE operations.

This service coordinates between the repository, downloader, loader, and hooker
components to manage SAE lifecycle operations including download, attach, and detach.
"""

import asyncio
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

import structlog
import torch

from millm.core.errors import (
    DownloadCancelledError,
    ModelNotLoadedError,
    SAEAlreadyAttachedError,
    SAEIncompatibleError,
    SAENotAttachedError,
    SAENotFoundError,
)
from millm.db.models.sae import SAE, SAEStatus
from millm.db.repositories.sae_repository import SAERepository
from millm.ml.model_loader import LoadedModelState
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_downloader import SAEDownloader
from millm.ml.sae_hooker import SAEHooker
from millm.ml.sae_loader import SAELoader
from millm.ml.sae_wrapper import LoadedSAE
from millm.sockets.progress import ProgressEmitter

logger = structlog.get_logger()


@dataclass
class AttachmentStatus:
    """Current SAE attachment status."""

    is_attached: bool
    sae_id: Optional[str] = None
    layer: Optional[int] = None
    memory_usage_mb: Optional[int] = None
    steering_enabled: bool = False
    monitoring_enabled: bool = False


@dataclass
class DownloadResult:
    """Result of a download request."""

    sae_id: str
    status: str  # "downloading", "cached", "attached", "already_downloading"
    message: str


@dataclass
class CompatibilityResult:
    """Result of SAE-model compatibility check."""

    compatible: bool
    errors: list[str]
    warnings: list[str]


class AttachedSAEState:
    """
    Singleton managing the currently attached SAE.

    This persists SAE attachment state across request boundaries since
    SAEService instances are created per-request via dependency injection.

    Thread-safe for access from executor threads.
    """

    _instance: Optional["AttachedSAEState"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AttachedSAEState":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._attached_sae: Optional[LoadedSAE] = None
                    cls._instance._attached_sae_id: Optional[str] = None
                    cls._instance._attached_layer: Optional[int] = None
                    cls._instance._hook_handle: Optional[Any] = None
        return cls._instance

    @property
    def attached_sae(self) -> Optional[LoadedSAE]:
        """Get the currently attached SAE."""
        return self._attached_sae

    @property
    def attached_sae_id(self) -> Optional[str]:
        """Get the ID of the attached SAE."""
        return self._attached_sae_id

    @property
    def attached_layer(self) -> Optional[int]:
        """Get the layer the SAE is attached to."""
        return self._attached_layer

    @property
    def hook_handle(self) -> Optional[Any]:
        """Get the hook handle for the attached SAE."""
        return self._hook_handle

    @property
    def is_attached(self) -> bool:
        """Check if an SAE is currently attached."""
        return self._attached_sae is not None

    def set(
        self,
        sae: LoadedSAE,
        sae_id: str,
        layer: int,
        hook_handle: Any,
    ) -> None:
        """Set the attached SAE state."""
        with self._lock:
            self._attached_sae = sae
            self._attached_sae_id = sae_id
            self._attached_layer = layer
            self._hook_handle = hook_handle

    def clear(self) -> None:
        """Clear the attached SAE state."""
        with self._lock:
            if self._hook_handle is not None:
                try:
                    self._hook_handle.remove()
                except Exception as e:
                    logger.warning("error_removing_hook", error=str(e))
            self._attached_sae = None
            self._attached_sae_id = None
            self._attached_layer = None
            self._hook_handle = None


class SAEService:
    """
    Orchestration layer for SAE operations.

    Coordinates between repository, downloader, loader, and hooker components.
    Manages SAE lifecycle: download, attach, steer, monitor, detach.

    Thread Safety:
        Uses _attachment_lock for state mutations during attach/detach.
        Forward pass through SAE is thread-safe.
    """

    def __init__(
        self,
        repository: SAERepository,
        cache_dir: str,
        emitter: Optional[ProgressEmitter] = None,
    ) -> None:
        """
        Initialize the SAE service.

        Args:
            repository: SAE database repository.
            cache_dir: Directory for SAE cache.
            emitter: Progress event emitter for WebSocket updates.
        """
        self.repository = repository
        self.emitter = emitter

        # Initialize components
        self._downloader = SAEDownloader(cache_dir)
        self._loader = SAELoader()
        self._hooker = SAEHooker()

        # Attachment state singleton (persists across requests)
        self._sae_state = AttachedSAEState()

        # Track active downloads for cancellation
        self._active_downloads: dict[str, asyncio.Task] = {}
        self._cancelled_downloads: set[str] = set()

        logger.debug("SAEService initialized", cache_dir=cache_dir)

    # =========================================================================
    # Listing Methods
    # =========================================================================

    async def list_saes(self) -> list[SAE]:
        """
        Get all SAEs from the database.

        Returns:
            List of all SAEs ordered by creation date descending.
        """
        return await self.repository.get_all()

    async def get_sae(self, sae_id: str) -> SAE:
        """
        Get a single SAE by ID.

        Args:
            sae_id: The SAE's database ID.

        Returns:
            The SAE if found.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
        """
        sae = await self.repository.get(sae_id)
        if not sae:
            raise SAENotFoundError(
                f"SAE with ID '{sae_id}' not found",
                details={"sae_id": sae_id},
            )
        return sae

    def get_attachment_status(self) -> AttachmentStatus:
        """
        Get current SAE attachment status.

        Returns:
            AttachmentStatus with current state.
        """
        if not self._sae_state.is_attached:
            return AttachmentStatus(is_attached=False)

        sae = self._sae_state.attached_sae
        return AttachmentStatus(
            is_attached=True,
            sae_id=self._sae_state.attached_sae_id,
            layer=self._sae_state.attached_layer,
            memory_usage_mb=int(sae.estimate_memory_mb()) if sae else None,
            steering_enabled=sae.is_steering_enabled if sae else False,
            monitoring_enabled=sae.is_monitoring_enabled if sae else False,
        )

    async def preview_repository(
        self,
        repository_id: str,
        revision: str = "main",
        token: str | None = None,
    ) -> dict:
        """
        Preview SAE files in a HuggingFace repository without downloading.

        Args:
            repository_id: HuggingFace repo (e.g., "google/gemma-scope-2b-pt-res").
            revision: Git revision (branch, tag, commit).
            token: HuggingFace access token for gated repositories.

        Returns:
            Dictionary with repository info and available SAE files.
        """
        return await self._downloader.list_repository_files(repository_id, revision, token)

    # =========================================================================
    # Download Methods
    # =========================================================================

    async def start_download(
        self,
        repository_id: str,
        revision: str = "main",
        file_path: str | None = None,
    ) -> DownloadResult:
        """
        Start downloading an SAE from HuggingFace.

        Creates a database record and starts the download asynchronously.

        Args:
            repository_id: HuggingFace repo (e.g., "jbloom/gemma-2-2b-res-jb").
            revision: Git revision (branch, tag, commit).
            file_path: Specific SAE file to download (e.g., "layer_12/width_16k/average_l0_50/params.npz").
                       If provided, only downloads that specific SAE directory.

        Returns:
            DownloadResult with SAE ID, status, and message.

        Raises:
            ValueError: If SAE already exists with same repo/revision.
        """
        # Generate SAE ID first (includes file_path for uniqueness)
        sae_id = self._downloader.generate_sae_id(repository_id, revision, file_path)

        # Check for existing SAE with this specific ID
        existing = await self.repository.get(sae_id)
        if existing:
            if existing.status == SAEStatus.CACHED:
                logger.info(
                    "sae_already_cached",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                return DownloadResult(
                    sae_id=existing.id,
                    status="cached",
                    message="SAE is already downloaded and cached",
                )
            elif existing.status == SAEStatus.ATTACHED:
                logger.info(
                    "sae_already_attached",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                return DownloadResult(
                    sae_id=existing.id,
                    status="attached",
                    message="SAE is already downloaded and attached to model",
                )
            elif existing.status == SAEStatus.DOWNLOADING:
                logger.info(
                    "sae_already_downloading",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                return DownloadResult(
                    sae_id=existing.id,
                    status="already_downloading",
                    message="SAE download is already in progress",
                )
            elif existing.status == SAEStatus.ERROR:
                # Delete the failed SAE and retry download
                logger.info(
                    "sae_retrying_failed_download",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                await self.repository.delete(existing.id)

        # Create database record in downloading state
        await self.repository.create_downloading(
            sae_id=sae_id,
            repository_id=repository_id,
            revision=revision,
            cache_path="",  # Updated after download
        )

        logger.info(
            "sae_download_started",
            sae_id=sae_id,
            repository_id=repository_id,
            revision=revision,
        )

        # Start background download and track it
        task = asyncio.create_task(self._download_task(sae_id, repository_id, revision, file_path))
        self._active_downloads[sae_id] = task

        return DownloadResult(
            sae_id=sae_id,
            status="downloading",
            message=f"Download started for {repository_id}",
        )

    async def _download_task(
        self,
        sae_id: str,
        repository_id: str,
        revision: str,
        file_path: str | None = None,
    ) -> None:
        """
        Background task for downloading SAE.

        Updates database on completion or error.
        """
        try:
            # Check if cancelled before starting
            if sae_id in self._cancelled_downloads:
                self._cancelled_downloads.discard(sae_id)
                raise DownloadCancelledError("Download was cancelled")

            # Download SAE
            cache_path = await self._downloader.download(
                repository_id=repository_id,
                revision=revision,
                file_path=file_path,
                progress_callback=self._make_progress_callback(sae_id),
            )

            # Check if cancelled after download
            if sae_id in self._cancelled_downloads:
                self._cancelled_downloads.discard(sae_id)
                raise DownloadCancelledError("Download was cancelled")

            # When downloading a specific file, the actual SAE is in a subdirectory
            # e.g., file_path="layer_20/width_16k/average_l0_71/params.npz"
            # cache_path is the root snapshot, but SAE files are in the subdirectory
            if file_path:
                # Extract directory from file path (e.g., "layer_20/width_16k/average_l0_71")
                sae_subdir = os.path.dirname(file_path)
                if sae_subdir:
                    cache_path = os.path.join(cache_path, sae_subdir)
                    logger.debug(
                        "sae_download_adjusted_path",
                        sae_id=sae_id,
                        file_path=file_path,
                        adjusted_cache_path=cache_path,
                    )

            # Load config to get dimensions
            config = self._loader.load_config(cache_path)

            # Calculate file size
            file_size = sum(
                os.path.getsize(os.path.join(cache_path, f))
                for f in os.listdir(cache_path)
                if os.path.isfile(os.path.join(cache_path, f))
            )

            # Extract width and average_l0 from file_path
            width = None
            average_l0 = None
            if file_path:
                width, average_l0 = self._parse_sae_path_metadata(file_path)

            # Update database with downloaded info
            await self.repository.update_downloaded(
                sae_id=sae_id,
                cache_path=cache_path,
                d_in=config.d_in,
                d_sae=config.d_sae,
                trained_on=config.model_name,
                trained_layer=config.hook_layer,
                file_size_bytes=file_size,
                width=width,
                average_l0=average_l0,
            )

            logger.info(
                "sae_download_complete",
                sae_id=sae_id,
                cache_path=cache_path,
                d_in=config.d_in,
                d_sae=config.d_sae,
            )

            # Emit completion event
            if self.emitter:
                await self.emitter.emit_sae_download_complete(sae_id=sae_id)

        except DownloadCancelledError:
            logger.info("sae_download_cancelled", sae_id=sae_id)
            await self.repository.update_status(
                sae_id=sae_id,
                status=SAEStatus.ERROR,
                error_message="Download cancelled by user",
            )
            if self.emitter:
                await self.emitter.emit_sae_download_error(
                    sae_id=sae_id,
                    error="Download cancelled by user",
                )

        except asyncio.CancelledError:
            logger.info("sae_download_cancelled", sae_id=sae_id)
            await self.repository.update_status(
                sae_id=sae_id,
                status=SAEStatus.ERROR,
                error_message="Download cancelled by user",
            )
            if self.emitter:
                await self.emitter.emit_sae_download_error(
                    sae_id=sae_id,
                    error="Download cancelled by user",
                )

        except Exception as e:
            logger.error(
                "sae_download_failed",
                sae_id=sae_id,
                error=str(e),
            )

            await self.repository.update_status(
                sae_id=sae_id,
                status=SAEStatus.ERROR,
                error_message=str(e),
            )

            if self.emitter:
                await self.emitter.emit_sae_download_error(
                    sae_id=sae_id,
                    error=str(e),
                )

        finally:
            # Clean up tracking
            self._active_downloads.pop(sae_id, None)
            self._cancelled_downloads.discard(sae_id)

    def _parse_sae_path_metadata(self, file_path: str) -> tuple[str | None, int | None]:
        """
        Extract width and average_l0 from SAE file path.

        Args:
            file_path: Path like "layer_20/width_16k/average_l0_38/params.npz"

        Returns:
            Tuple of (width, average_l0). E.g., ("16k", 38)
        """
        width = None
        average_l0 = None

        # Match width pattern: width_16k, width_65k, etc.
        width_match = re.search(r"width[_-]?(\d+k?)", file_path, re.IGNORECASE)
        if width_match:
            width = width_match.group(1)

        # Match average_l0 pattern: average_l0_38, l0_38, etc.
        l0_match = re.search(r"(?:average_)?l0[_-]?(\d+)", file_path, re.IGNORECASE)
        if l0_match:
            average_l0 = int(l0_match.group(1))

        return width, average_l0

    def _make_progress_callback(self, sae_id: str):
        """
        Create progress callback for download.

        The callback may be called from a thread executor, so we use
        run_coroutine_threadsafe to safely schedule async operations.
        """
        # Capture the event loop at callback creation time (when we're in async context)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def callback(progress: dict[str, Any]) -> None:
            if self.emitter and progress.get("status") == "downloading":
                coro = self.emitter.emit_sae_download_progress(
                    sae_id=sae_id,
                    percent=progress.get("percent", 0),
                )
                if loop and loop.is_running():
                    # Schedule from thread to main event loop
                    asyncio.run_coroutine_threadsafe(coro, loop)
                else:
                    # Fallback: try to create task if we're in async context
                    try:
                        asyncio.create_task(coro)
                    except RuntimeError:
                        # No event loop available, skip progress emission
                        logger.debug(
                            "skipping_progress_emit",
                            reason="no_event_loop",
                            sae_id=sae_id,
                        )
        return callback

    async def cancel_download(self, sae_id: str) -> SAE:
        """
        Cancel an in-progress SAE download.

        Args:
            sae_id: The SAE's ID.

        Returns:
            The SAE with updated status.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
        """
        sae = await self.get_sae(sae_id)

        # Only cancel if actually downloading
        if sae.status != SAEStatus.DOWNLOADING:
            return sae

        # Mark as cancelled
        self._cancelled_downloads.add(sae_id)

        # Cancel the task if it exists
        task = self._active_downloads.get(sae_id)
        if task and not task.done():
            task.cancel()

        # Update database status
        await self.repository.update_status(
            sae_id=sae_id,
            status=SAEStatus.ERROR,
            error_message="Download cancelled by user",
        )

        logger.info("sae_download_cancelled", sae_id=sae_id)

        # Return updated SAE
        return await self.get_sae(sae_id)

    # =========================================================================
    # Compatibility Methods
    # =========================================================================

    async def check_compatibility(
        self,
        sae_id: str,
        layer: int,
    ) -> CompatibilityResult:
        """
        Check if SAE is compatible with currently loaded model.

        Args:
            sae_id: The SAE's ID.
            layer: Target layer to attach.

        Returns:
            CompatibilityResult with compatibility status and any issues.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
            ModelNotLoadedError: If no model is loaded.
        """
        sae = await self.get_sae(sae_id)

        # Check model is loaded
        model_state = LoadedModelState()
        if not model_state.is_loaded:
            raise ModelNotLoadedError(
                "No model loaded. Load a model before checking SAE compatibility.",
            )

        errors: list[str] = []
        warnings: list[str] = []

        # Check SAE status
        if sae.status != SAEStatus.CACHED:
            errors.append(f"SAE is not ready (status: {sae.status.value})")

        # Check dimension compatibility
        model = model_state.current.model
        if hasattr(model, "config"):
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size and sae.d_in != hidden_size:
                errors.append(
                    f"Dimension mismatch: SAE d_in={sae.d_in}, "
                    f"model hidden_size={hidden_size}"
                )

        # Check layer range
        try:
            num_layers = self._hooker.get_layer_count(model)
            if not 0 <= layer < num_layers:
                errors.append(
                    f"Layer {layer} out of range [0, {num_layers-1}]"
                )
        except ValueError as e:
            warnings.append(f"Could not verify layer count: {e}")

        # Check trained layer match
        if sae.trained_layer is not None and sae.trained_layer != layer:
            warnings.append(
                f"SAE was trained on layer {sae.trained_layer}, "
                f"but attaching to layer {layer}"
            )

        # Check trained model match
        if sae.trained_on:
            model_name = getattr(model.config, "name_or_path", "") if hasattr(model, "config") else ""
            if model_name and sae.trained_on not in model_name and model_name not in sae.trained_on:
                warnings.append(
                    f"SAE was trained on '{sae.trained_on}', "
                    f"current model is '{model_name}'"
                )

        return CompatibilityResult(
            compatible=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # =========================================================================
    # Attachment Methods
    # =========================================================================

    async def attach_sae(
        self,
        sae_id: str,
        layer: int,
    ) -> dict[str, Any]:
        """
        Attach an SAE to the loaded model.

        Args:
            sae_id: The SAE's ID.
            layer: Layer to attach SAE to.

        Returns:
            Dict with attachment status and memory info.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
            SAEAlreadyAttachedError: If an SAE is already attached.
            SAEIncompatibleError: If SAE is incompatible with model.
            ModelNotLoadedError: If no model is loaded.
        """
        sae = await self.get_sae(sae_id)

        # Check model is loaded
        model_state = LoadedModelState()
        if not model_state.is_loaded:
            raise ModelNotLoadedError(
                "No model loaded. Load a model before attaching SAE.",
            )

        # Check no SAE already attached
        if self._sae_state.is_attached:
            raise SAEAlreadyAttachedError(
                f"SAE '{self._sae_state.attached_sae_id}' is already attached. "
                "Detach it first before attaching another.",
                details={"attached_sae_id": self._sae_state.attached_sae_id},
            )

        # Check compatibility
        compat = await self.check_compatibility(sae_id, layer)
        if not compat.compatible:
            raise SAEIncompatibleError(
                f"SAE incompatible with model: {compat.errors[0]}",
                details={"errors": compat.errors, "warnings": compat.warnings},
            )

        # Log warnings
        for warning in compat.warnings:
            logger.warning("sae_compatibility_warning", warning=warning)

        # Load SAE weights
        logger.info("loading_sae", sae_id=sae_id, cache_path=sae.cache_path)
        loaded_sae = self._loader.load(
            cache_path=sae.cache_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Install hook
        model = model_state.current.model
        handle = self._hooker.install(model, layer, loaded_sae)

        # Update state in singleton
        self._sae_state.set(loaded_sae, sae_id, layer, handle)

        # Update database
        await self.repository.update_status(sae_id, SAEStatus.ATTACHED)
        await self.repository.create_attachment(
            sae_id=sae_id,
            model_id=model_state.loaded_model_id,
            layer=layer,
            memory_usage_mb=int(loaded_sae.estimate_memory_mb()),
        )

        memory_mb = int(loaded_sae.estimate_memory_mb())

        logger.info(
            "sae_attached",
            sae_id=sae_id,
            layer=layer,
            memory_mb=memory_mb,
        )

        return {
            "status": "attached",
            "sae_id": sae_id,
            "layer": layer,
            "memory_usage_mb": memory_mb,
            "warnings": compat.warnings,
        }

    async def detach_sae(self, sae_id: str) -> dict[str, Any]:
        """
        Detach an SAE from the model.

        Args:
            sae_id: The SAE's ID.

        Returns:
            Dict with detachment status and freed memory info.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
            SAENotAttachedError: If SAE is not attached.
        """
        sae = await self.get_sae(sae_id)

        if self._sae_state.attached_sae_id != sae_id:
            raise SAENotAttachedError(
                f"SAE '{sae_id}' is not attached",
                details={"attached_sae_id": self._sae_state.attached_sae_id},
            )

        # Get memory before cleanup
        attached_sae = self._sae_state.attached_sae
        memory_freed_mb = int(attached_sae.estimate_memory_mb()) if attached_sae else 0

        # Remove hook
        hook_handle = self._sae_state.hook_handle
        if hook_handle:
            self._hooker.remove(hook_handle)

        # Move to CPU and cleanup
        if attached_sae:
            attached_sae.to_cpu()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear state in singleton (this also removes hook if not already done)
        self._sae_state.clear()

        # Update database
        await self.repository.update_status(sae_id, SAEStatus.CACHED)
        await self.repository.deactivate_attachment(sae_id)

        logger.info(
            "sae_detached",
            sae_id=sae_id,
            memory_freed_mb=memory_freed_mb,
        )

        return {
            "status": "detached",
            "sae_id": sae_id,
            "memory_freed_mb": memory_freed_mb,
        }

    # =========================================================================
    # Steering Methods
    # =========================================================================

    def set_steering(self, feature_idx: int, value: float) -> None:
        """
        Set steering value for a feature.

        Args:
            feature_idx: Feature index.
            value: Steering strength.

        Raises:
            SAENotAttachedError: If no SAE is attached.
            ValueError: If feature index is invalid.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        self._sae_state.attached_sae.set_steering(feature_idx, value)

    def set_steering_batch(self, steering: dict[int, float]) -> None:
        """
        Set multiple steering values at once.

        Args:
            steering: Dict mapping feature indices to values.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        self._sae_state.attached_sae.set_steering_batch(steering)

    def clear_steering(self, feature_idx: Optional[int] = None) -> None:
        """
        Clear steering for one or all features.

        Args:
            feature_idx: Specific feature to clear (None = clear all).

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        self._sae_state.attached_sae.clear_steering(feature_idx)

    def enable_steering(self, enabled: bool = True) -> None:
        """
        Enable or disable steering.

        Args:
            enabled: Whether to enable steering.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        self._sae_state.attached_sae.enable_steering(enabled)

    def get_steering_values(self) -> dict[int, float]:
        """
        Get current steering values.

        Returns:
            Dict mapping feature indices to steering values.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        return self._sae_state.attached_sae.get_steering_values()

    # =========================================================================
    # Monitoring Methods
    # =========================================================================

    def enable_monitoring(
        self,
        enabled: bool = True,
        features: Optional[list[int]] = None,
    ) -> None:
        """
        Enable or disable feature monitoring.

        Args:
            enabled: Whether to capture activations.
            features: Specific features to monitor (None = all).

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        self._sae_state.attached_sae.enable_monitoring(enabled, features)

    def get_last_activations(self) -> Optional[Any]:
        """
        Get feature activations from last forward pass.

        Returns:
            Activations tensor or None.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        if not self._sae_state.is_attached:
            raise SAENotAttachedError("No SAE attached")
        return self._sae_state.attached_sae.get_last_feature_activations()

    # =========================================================================
    # Delete Methods
    # =========================================================================

    async def delete_sae(self, sae_id: str) -> dict[str, Any]:
        """
        Delete an SAE from database and disk.

        Args:
            sae_id: The SAE's ID.

        Returns:
            Dict with deletion status and freed disk space.

        Raises:
            SAENotFoundError: If SAE doesn't exist.
            SAEAlreadyAttachedError: If SAE is currently attached.
        """
        sae = await self.get_sae(sae_id)

        # Check SAE is not attached
        if self._sae_state.attached_sae_id == sae_id:
            raise SAEAlreadyAttachedError(
                f"Cannot delete SAE '{sae_id}' while it is attached. "
                "Detach it first.",
                details={"sae_id": sae_id},
            )

        # Delete cache files
        freed_mb = 0.0
        if sae.cache_path:
            freed_mb = await self._downloader.delete(sae.cache_path)

        # Delete from database
        await self.repository.delete(sae_id)

        logger.info(
            "sae_deleted",
            sae_id=sae_id,
            freed_mb=freed_mb,
        )

        return {
            "status": "deleted",
            "sae_id": sae_id,
            "freed_disk_mb": freed_mb,
        }

    # =========================================================================
    # Cleanup
    # =========================================================================

    def shutdown(self) -> None:
        """Clean up resources on application shutdown."""
        if self._sae_state.is_attached:
            attached_sae = self._sae_state.attached_sae
            hook_handle = self._sae_state.hook_handle
            if hook_handle:
                self._hooker.remove(hook_handle)
            if attached_sae:
                attached_sae.to_cpu()
            self._sae_state.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("SAEService shutdown complete")
