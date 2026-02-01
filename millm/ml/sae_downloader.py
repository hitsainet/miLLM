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
        file_path: str | None = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> str:
        """
        Download SAE from repository.

        Args:
            repository_id: HuggingFace repo (e.g., "jbloom/gemma-2-2b-res-jb").
            revision: Git revision (branch, tag, commit).
            file_path: Specific file to download (e.g., "layer_12/width_16k/average_l0_50/params.npz").
                       If provided, only downloads files in that directory.
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
            file_path,
            progress_callback,
        )

        return cache_path

    def _download_sync(
        self,
        repository_id: str,
        revision: str,
        file_path: str | None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]],
    ) -> str:
        """
        Synchronous download implementation.

        Args:
            repository_id: HuggingFace repo ID.
            revision: Git revision.
            file_path: Specific file/directory to download.
            progress_callback: Optional progress callback.

        Returns:
            Local cache path.
        """
        if file_path:
            logger.info(f"Downloading SAE file {file_path} from {repository_id}@{revision}")
        else:
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

            # Build allow_patterns if specific file requested
            allow_patterns = None
            if file_path:
                # Get the directory containing the SAE file
                # e.g., "layer_12/width_16k/average_l0_50/params.npz" -> "layer_12/width_16k/average_l0_50/*"
                from pathlib import PurePosixPath
                sae_dir = str(PurePosixPath(file_path).parent)
                allow_patterns = [f"{sae_dir}/*"]
                logger.info(f"Filtering download to: {allow_patterns}")

            # Download with huggingface_hub
            local_path = snapshot_download(
                repo_id=repository_id,
                revision=revision,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                allow_patterns=allow_patterns,
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

    def generate_sae_id(
        self,
        repository_id: str,
        revision: str = "main",
        file_path: str | None = None,
    ) -> str:
        """
        Generate a unique SAE ID from repository info.

        Args:
            repository_id: HuggingFace repo ID (e.g., "jbloom/gemma-2-2b-res-jb").
            revision: Git revision.
            file_path: Specific file path (e.g., "layer_20/width_16k/average_l0_71/params.npz").
                       If provided, the directory is included in the ID to make it unique.

        Returns:
            Unique SAE ID (e.g., "jbloom--gemma-2-2b-res-jb--layer_20--width_16k--average_l0_71").
        """
        # Replace / with -- for filesystem-safe ID
        base_id = repository_id.replace("/", "--")

        # Add revision if not main
        if revision != "main":
            base_id = f"{base_id}@{revision}"

        # Add file path directory if provided (makes ID unique per SAE in repo)
        if file_path:
            from pathlib import PurePosixPath
            sae_dir = str(PurePosixPath(file_path).parent)
            if sae_dir and sae_dir != ".":
                # Replace / with -- for filesystem-safe ID
                dir_id = sae_dir.replace("/", "--")
                base_id = f"{base_id}--{dir_id}"

        return base_id

    async def list_repository_files(
        self,
        repository_id: str,
        revision: str = "main",
        token: str | None = None,
    ) -> dict[str, Any]:
        """
        List SAE files in a HuggingFace repository.

        Args:
            repository_id: HuggingFace repo ID (e.g., "google/gemma-scope-2b-pt-res").
            revision: Git revision (branch, tag, commit).
            token: HuggingFace access token for gated repositories.

        Returns:
            Dictionary containing repository info and SAE files:
            {
                "repository_id": str,
                "revision": str,
                "model_id": str | None,  # Extracted from repo if identifiable
                "files": [
                    {
                        "path": str,  # e.g., "layer_0/width_16k/average_l0_13/params.npz"
                        "size_bytes": int,
                        "layer": int | None,  # Extracted from path if identifiable
                        "width": str | None,  # e.g., "16k"
                    },
                    ...
                ],
                "total_files": int,
            }

        Raises:
            ValueError: If repository doesn't exist.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._list_repository_files_sync,
            repository_id,
            revision,
            token,
        )

    def _list_repository_files_sync(
        self,
        repository_id: str,
        revision: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous implementation of list_repository_files."""
        import re

        logger.info(f"Listing files in SAE repository {repository_id}@{revision}")

        try:
            # Validate and get repo info
            repo_info = self._api.repo_info(repository_id, revision=revision, token=token)

            # Get all files
            files = self._api.list_repo_files(repository_id, revision=revision, token=token)

            # Filter and parse SAE files
            sae_files = []
            for file_path in files:
                # Skip non-SAE files
                if not self._is_sae_file(file_path):
                    continue

                # Get file size if available
                size_bytes = 0
                if repo_info.siblings:
                    for sibling in repo_info.siblings:
                        if sibling.rfilename == file_path:
                            size_bytes = sibling.size or 0
                            break

                # Parse layer and width from path
                layer, width = self._parse_sae_path(file_path)

                sae_files.append({
                    "path": file_path,
                    "size_bytes": size_bytes,
                    "layer": layer,
                    "width": width,
                })

            # Sort by layer, then path
            sae_files.sort(key=lambda x: (x["layer"] or 999, x["path"]))

            # Try to identify the model from repo name
            model_id = self._extract_model_id(repository_id)

            return {
                "repository_id": repository_id,
                "revision": revision,
                "model_id": model_id,
                "files": sae_files,
                "total_files": len(sae_files),
            }

        except RepositoryNotFoundError as e:
            raise ValueError(f"SAE repository not found: {repository_id}") from e
        except RevisionNotFoundError as e:
            raise ValueError(
                f"Revision '{revision}' not found for {repository_id}"
            ) from e

    def _is_sae_file(self, path: str) -> bool:
        """
        Check if file path is an SAE file (params.npz, weights, etc.).

        SAE files are typically:
        - params.npz (SAELens format)
        - sae_weights.safetensors / sae_weights.pt
        - cfg.json (SAE config)
        - sparsity.npz
        - Files in layer_N/width_Nk directories

        Model files to exclude:
        - model-NNNNN-of-NNNNN.safetensors (sharded model weights)
        - model.safetensors (single model weight file)
        - pytorch_model.bin / pytorch_model-*.bin
        - generation_config.json, tokenizer_config.json, etc.
        """
        import re

        path_lower = path.lower()

        # Explicit exclusions for model files
        model_patterns = [
            r"model-\d+-of-\d+\.safetensors",  # Sharded model weights
            r"model\.safetensors$",  # Single model weight file
            r"pytorch_model.*\.bin",  # PyTorch model files
            r"generation_config\.json",
            r"tokenizer_config\.json",
            r"tokenizer\.json",
            r"tokenizer\.model",
            r"special_tokens_map\.json",
            r"vocab\.json",
            r"merges\.txt",
            r"added_tokens\.json",
            r"model\.safetensors\.index\.json",  # Model index file
        ]
        for pattern in model_patterns:
            if re.search(pattern, path_lower):
                return False

        # Explicit SAE file patterns
        sae_specific_patterns = [
            "params.npz",
            "sae_weights.pt",
            "sae_weights.safetensors",
            "cfg.json",
            "sparsity.npz",
        ]
        if any(pattern in path_lower for pattern in sae_specific_patterns):
            return True

        # SAE files in layer/width directory structure
        # e.g., "layer_12/width_16k/average_l0_50/params.npz"
        if re.search(r"layer[_-]?\d+", path_lower) and (
            path_lower.endswith(".npz")
            or path_lower.endswith(".pt")
            or path_lower.endswith(".safetensors")
        ):
            return True

        # config.json in a layer directory is likely SAE config
        if "config.json" in path_lower and re.search(r"layer[_-]?\d+", path_lower):
            return True

        return False

    def _parse_sae_path(self, path: str) -> tuple[int | None, str | None]:
        """
        Extract layer number and width from SAE file path.

        Examples:
            "layer_0/width_16k/average_l0_13/params.npz" -> (0, "16k")
            "gemma-scope-2b-pt-res-canonical/layer_12/width_32k/..." -> (12, "32k")
        """
        import re

        layer = None
        width = None

        # Match layer_N or layer_NN
        layer_match = re.search(r"layer[_-]?(\d+)", path, re.IGNORECASE)
        if layer_match:
            layer = int(layer_match.group(1))

        # Match width_NNk or width_NNNk
        width_match = re.search(r"width[_-]?(\d+k?)", path, re.IGNORECASE)
        if width_match:
            width = width_match.group(1)

        return layer, width

    def _extract_model_id(self, repository_id: str) -> str | None:
        """
        Try to extract the model ID from repository name.

        Examples:
            "google/gemma-scope-2b-pt-res" -> "gemma"
            "jbloom/gemma-2-2b-res-jb" -> "gemma-2-2b"
        """
        repo_name = repository_id.split("/")[-1].lower()

        # Common model patterns
        if "gemma" in repo_name:
            if "2b" in repo_name:
                return "gemma-2-2b"
            if "9b" in repo_name:
                return "gemma-2-9b"
            return "gemma"
        if "gpt2" in repo_name:
            return "gpt2"
        if "llama" in repo_name:
            return "llama"
        if "tinystories" in repo_name:
            return "tinystories"

        return None
