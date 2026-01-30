"""
Download SAEs from HuggingFace Hub.

Uses huggingface_hub for reliable downloading with resume support.
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

logger = logging.getLogger(__name__)


class SAEDownloader:
    """
    Download SAEs from HuggingFace Hub.

    Features:
    - Resume interrupted downloads
    - Progress tracking via callback
    - Local caching with configurable directory
    - Repository validation before download

    Usage:
        downloader = SAEDownloader(cache_dir="~/.cache/millm/saes")
        path = await downloader.download("jbloom/gemma-2-2b-res-jb")
    """

    def __init__(self, cache_dir: str | Path) -> None:
        """
        Initialize downloader.

        Args:
            cache_dir: Directory for SAE cache (e.g., ~/.cache/millm/saes).
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api = HfApi()

        logger.debug(f"SAEDownloader initialized with cache_dir={self.cache_dir}")

    async def download(
        self,
        repository_id: str,
        revision: str = "main",
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> str:
        """
        Download SAE repository.

        Args:
            repository_id: HuggingFace repo (e.g., "jbloom/gemma-2-2b-res-jb").
            revision: Git revision (branch, tag, commit).
            progress_callback: Called with progress updates:
                {"status": "downloading", "percent": 50}
                {"status": "complete", "percent": 100, "path": "/path/to/sae"}

        Returns:
            Local cache path.

        Raises:
            ValueError: If repository doesn't exist.
            HfHubHTTPError: If download fails.
        """
        loop = asyncio.get_event_loop()

        # Run in executor (HF download is blocking)
        cache_path = await loop.run_in_executor(
            None,
            self._download_sync,
            repository_id,
            revision,
            progress_callback,
        )

        return cache_path

    def _download_sync(
        self,
        repository_id: str,
        revision: str,
        progress_callback: Optional[Callable[[dict[str, Any]], None]],
    ) -> str:
        """
        Synchronous download implementation.

        Args:
            repository_id: HuggingFace repo ID.
            revision: Git revision.
            progress_callback: Optional progress callback.

        Returns:
            Local cache path.
        """
        logger.info(f"Downloading SAE from {repository_id}@{revision}")

        # Emit start
        if progress_callback:
            progress_callback({
                "status": "downloading",
                "percent": 0,
                "repository_id": repository_id,
            })

        try:
            # Validate repository exists
            self._validate_repository(repository_id, revision)

            # Download with huggingface_hub
            local_path = snapshot_download(
                repo_id=repository_id,
                revision=revision,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                # Only download essential files
                ignore_patterns=[
                    "*.md",
                    "*.txt",
                    ".git*",
                    "*.png",
                    "*.jpg",
                    "*.jpeg",
                    "README*",
                    "LICENSE*",
                ],
            )

            logger.info(f"SAE downloaded to {local_path}")

            # Emit completion
            if progress_callback:
                progress_callback({
                    "status": "complete",
                    "percent": 100,
                    "path": local_path,
                })

            return local_path

        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            logger.error(f"Repository not found: {repository_id}@{revision}")
            if progress_callback:
                progress_callback({
                    "status": "error",
                    "error": str(e),
                })
            raise ValueError(f"SAE repository not found: {repository_id}@{revision}") from e

        except HfHubHTTPError as e:
            logger.error(f"Download failed: {e}")
            if progress_callback:
                progress_callback({
                    "status": "error",
                    "error": str(e),
                })
            raise

    def _validate_repository(self, repository_id: str, revision: str) -> None:
        """
        Validate that repository exists and is accessible.

        Args:
            repository_id: HuggingFace repo ID.
            revision: Git revision.

        Raises:
            ValueError: If repository not found.
        """
        try:
            self._api.repo_info(repository_id, revision=revision)
        except RepositoryNotFoundError as e:
            raise ValueError(f"SAE repository not found: {repository_id}") from e
        except RevisionNotFoundError as e:
            raise ValueError(
                f"Revision '{revision}' not found for {repository_id}"
            ) from e

    async def delete(self, cache_path: str | Path) -> float:
        """
        Delete cached SAE directory.

        Args:
            cache_path: Path to SAE cache directory.

        Returns:
            Freed disk space in MB.
        """
        path = Path(cache_path)
        if not path.exists():
            return 0.0

        # Calculate size
        size_bytes = self._calculate_directory_size(path)

        # Delete directory
        shutil.rmtree(path)

        freed_mb = size_bytes / (1024 * 1024)
        logger.info(f"Deleted SAE cache: {cache_path} ({freed_mb:.1f}MB)")

        return freed_mb

    def get_cache_path(self, repository_id: str, revision: str = "main") -> Path | None:
        """
        Get local cache path for a repository if it exists.

        Args:
            repository_id: HuggingFace repo ID.
            revision: Git revision.

        Returns:
            Path to cached SAE or None if not cached.
        """
        # huggingface_hub uses a specific cache structure
        # Check for common patterns
        repo_folder_name = repository_id.replace("/", "--")

        # Check if snapshot exists
        snapshots_dir = self.cache_dir / "models--" + repo_folder_name / "snapshots"
        if snapshots_dir.exists():
            # Return first snapshot (most recent)
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                return snapshots[0]

        return None

    def is_cached(self, repository_id: str, revision: str = "main") -> bool:
        """
        Check if SAE is already cached.

        Args:
            repository_id: HuggingFace repo ID.
            revision: Git revision.

        Returns:
            True if SAE is cached locally.
        """
        return self.get_cache_path(repository_id, revision) is not None

    def get_cache_size(self) -> float:
        """
        Get total SAE cache size in MB.

        Returns:
            Total cache size in MB.
        """
        if not self.cache_dir.exists():
            return 0.0

        return self._calculate_directory_size(self.cache_dir) / (1024 * 1024)

    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of directory in bytes."""
        total = 0
        for f in path.glob("**/*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except (OSError, IOError):
                    pass
        return total

    def generate_sae_id(self, repository_id: str, revision: str = "main") -> str:
        """
        Generate a unique SAE ID from repository info.

        Args:
            repository_id: HuggingFace repo ID (e.g., "jbloom/gemma-2-2b-res-jb").
            revision: Git revision.

        Returns:
            Unique SAE ID (e.g., "jbloom--gemma-2-2b-res-jb").
        """
        # Replace / with -- for filesystem-safe ID
        base_id = repository_id.replace("/", "--")

        # Add revision if not main
        if revision != "main":
            base_id = f"{base_id}@{revision}"

        return base_id
