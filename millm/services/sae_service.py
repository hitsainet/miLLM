"""
SAE service for orchestrating SAE operations.

This service coordinates between the repository, downloader, loader, and hooker
components to manage SAE lifecycle operations including download, attach, and detach.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Optional

import structlog
import torch

from millm.core.errors import (
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
class CompatibilityResult:
    """Result of SAE-model compatibility check."""

    compatible: bool
    errors: list[str]
    warnings: list[str]


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

        # Attachment state
        self._attachment_lock = threading.Lock()
        self._attached_sae: Optional[LoadedSAE] = None
        self._attached_sae_id: Optional[str] = None
        self._attached_layer: Optional[int] = None
        self._hook_handle: Optional[Any] = None

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
        with self._attachment_lock:
            if self._attached_sae is None:
                return AttachmentStatus(is_attached=False)

            return AttachmentStatus(
                is_attached=True,
                sae_id=self._attached_sae_id,
                layer=self._attached_layer,
                memory_usage_mb=int(self._attached_sae.estimate_memory_mb()),
                steering_enabled=self._attached_sae.is_steering_enabled,
                monitoring_enabled=self._attached_sae.is_monitoring_enabled,
            )

    # =========================================================================
    # Download Methods
    # =========================================================================

    async def start_download(
        self,
        repository_id: str,
        revision: str = "main",
    ) -> str:
        """
        Start downloading an SAE from HuggingFace.

        Creates a database record and starts the download asynchronously.

        Args:
            repository_id: HuggingFace repo (e.g., "jbloom/gemma-2-2b-res-jb").
            revision: Git revision (branch, tag, commit).

        Returns:
            The SAE ID.

        Raises:
            ValueError: If SAE already exists with same repo/revision.
        """
        # Check for existing SAE
        existing = await self.repository.get_by_repository(repository_id, revision)
        if existing:
            if existing.status == SAEStatus.CACHED:
                logger.info(
                    "sae_already_cached",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                return existing.id
            elif existing.status == SAEStatus.DOWNLOADING:
                logger.info(
                    "sae_already_downloading",
                    sae_id=existing.id,
                    repository_id=repository_id,
                )
                return existing.id

        # Generate SAE ID
        sae_id = self._downloader.generate_sae_id(repository_id, revision)

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

        # Start background download
        asyncio.create_task(self._download_task(sae_id, repository_id, revision))

        return sae_id

    async def _download_task(
        self,
        sae_id: str,
        repository_id: str,
        revision: str,
    ) -> None:
        """
        Background task for downloading SAE.

        Updates database on completion or error.
        """
        try:
            # Download SAE
            cache_path = await self._downloader.download(
                repository_id=repository_id,
                revision=revision,
                progress_callback=self._make_progress_callback(sae_id),
            )

            # Load config to get dimensions
            config = self._loader.load_config(cache_path)

            # Calculate file size
            import os
            file_size = sum(
                os.path.getsize(os.path.join(cache_path, f))
                for f in os.listdir(cache_path)
                if os.path.isfile(os.path.join(cache_path, f))
            )

            # Update database with downloaded info
            await self.repository.update_downloaded(
                sae_id=sae_id,
                cache_path=cache_path,
                d_in=config.d_in,
                d_sae=config.d_sae,
                trained_on=config.model_name,
                trained_layer=config.hook_layer,
                file_size_bytes=file_size,
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

    def _make_progress_callback(self, sae_id: str):
        """Create progress callback for download."""
        def callback(progress: dict[str, Any]) -> None:
            if self.emitter and progress.get("status") == "downloading":
                asyncio.create_task(
                    self.emitter.emit_sae_download_progress(
                        sae_id=sae_id,
                        percent=progress.get("percent", 0),
                    )
                )
        return callback

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
        model_state = LoadedModelState.get_instance()
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
        model = model_state.model
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
        model_state = LoadedModelState.get_instance()
        if not model_state.is_loaded:
            raise ModelNotLoadedError(
                "No model loaded. Load a model before attaching SAE.",
            )

        # Check no SAE already attached
        with self._attachment_lock:
            if self._attached_sae is not None:
                raise SAEAlreadyAttachedError(
                    f"SAE '{self._attached_sae_id}' is already attached. "
                    "Detach it first before attaching another.",
                    details={"attached_sae_id": self._attached_sae_id},
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
        model = model_state.model
        handle = self._hooker.install(model, layer, loaded_sae)

        # Update state
        with self._attachment_lock:
            self._attached_sae = loaded_sae
            self._attached_sae_id = sae_id
            self._attached_layer = layer
            self._hook_handle = handle

        # Update database
        await self.repository.update_status(sae_id, SAEStatus.ATTACHED)
        await self.repository.create_attachment(
            sae_id=sae_id,
            model_id=model_state.model_id,
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

        with self._attachment_lock:
            if self._attached_sae_id != sae_id:
                raise SAENotAttachedError(
                    f"SAE '{sae_id}' is not attached",
                    details={"attached_sae_id": self._attached_sae_id},
                )

            # Get memory before cleanup
            memory_freed_mb = int(self._attached_sae.estimate_memory_mb())

            # Remove hook
            if self._hook_handle:
                self._hooker.remove(self._hook_handle)

            # Move to CPU and cleanup
            self._attached_sae.to_cpu()
            del self._attached_sae

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clear state
            self._attached_sae = None
            self._attached_sae_id = None
            self._attached_layer = None
            self._hook_handle = None

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
        with self._attachment_lock:
            if self._attached_sae is None:
                raise SAENotAttachedError("No SAE attached")

            self._attached_sae.set_steering(feature_idx, value)

    def set_steering_batch(self, steering: dict[int, float]) -> None:
        """
        Set multiple steering values at once.

        Args:
            steering: Dict mapping feature indices to values.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        with self._attachment_lock:
            if self._attached_sae is None:
                raise SAENotAttachedError("No SAE attached")

            self._attached_sae.set_steering_batch(steering)

    def clear_steering(self, feature_idx: Optional[int] = None) -> None:
        """
        Clear steering for one or all features.

        Args:
            feature_idx: Specific feature to clear (None = clear all).

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        with self._attachment_lock:
            if self._attached_sae is None:
                raise SAENotAttachedError("No SAE attached")

            self._attached_sae.clear_steering(feature_idx)

    def enable_steering(self, enabled: bool = True) -> None:
        """
        Enable or disable steering.

        Args:
            enabled: Whether to enable steering.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        with self._attachment_lock:
            if self._attached_sae is None:
                raise SAENotAttachedError("No SAE attached")

            self._attached_sae.enable_steering(enabled)

    def get_steering_values(self) -> dict[int, float]:
        """
        Get current steering values.

        Returns:
            Dict mapping feature indices to steering values.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        with self._attachment_lock:
            if self._attached_sae is None:
                raise SAENotAttachedError("No SAE attached")

            return self._attached_sae.get_steering_values()

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
        with self._attachment_lock:
            if self._attached_sae is None:
                raise SAENotAttachedError("No SAE attached")

            self._attached_sae.enable_monitoring(enabled, features)

    def get_last_activations(self) -> Optional[Any]:
        """
        Get feature activations from last forward pass.

        Returns:
            Activations tensor or None.

        Raises:
            SAENotAttachedError: If no SAE is attached.
        """
        with self._attachment_lock:
            if self._attached_sae is None:
                raise SAENotAttachedError("No SAE attached")

            return self._attached_sae.get_last_feature_activations()

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
        with self._attachment_lock:
            if self._attached_sae_id == sae_id:
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
        with self._attachment_lock:
            if self._attached_sae is not None:
                if self._hook_handle:
                    self._hooker.remove(self._hook_handle)
                self._attached_sae.to_cpu()
                del self._attached_sae
                self._attached_sae = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info("SAEService shutdown complete")
