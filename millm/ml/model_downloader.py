"""
Model downloader for HuggingFace Hub.

Handles downloading models from HuggingFace with progress tracking
and error handling for common scenarios (gated models, missing repos).
"""

import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import structlog
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)
from tqdm.auto import tqdm as tqdm_auto

from millm.core.config import settings
from millm.core.errors import (
    DownloadFailedError,
    GatedModelError,
    InvalidTokenError,
    RepoNotFoundError,
)
from millm.core.resilience import huggingface_circuit, CircuitOpenError

logger = structlog.get_logger()

# Type alias for progress callback: (progress_pct, downloaded_bytes, total_bytes)
ProgressCallback = Callable[[float, int, int], None]


class _DownloadProgressTqdm(tqdm_auto):
    """
    Custom tqdm that aggregates byte-level progress from parallel HF downloads.

    HuggingFace's snapshot_download creates one tqdm instance per file.
    This class aggregates all instances into a single progress percentage
    and reports it via a callback.
    """

    _lock = threading.Lock()
    _total_bytes: int = 0
    _downloaded_bytes: int = 0
    _callback: Optional[ProgressCallback] = None
    _last_pct: int = 0

    @classmethod
    def _reset(cls, callback: Optional[ProgressCallback] = None) -> None:
        """Reset shared state before a new download."""
        with cls._lock:
            cls._total_bytes = 0
            cls._downloaded_bytes = 0
            cls._callback = callback
            cls._last_pct = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Disable actual tqdm output
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
        # Each instance tracks one file; self.total is that file's size
        if self.total and self.total > 0:
            with _DownloadProgressTqdm._lock:
                _DownloadProgressTqdm._total_bytes += int(self.total)

    def update(self, n: int = 1) -> None:  # type: ignore[override]
        super().update(n)
        if n <= 0:
            return
        with _DownloadProgressTqdm._lock:
            _DownloadProgressTqdm._downloaded_bytes += int(n)
            total = _DownloadProgressTqdm._total_bytes
            downloaded = _DownloadProgressTqdm._downloaded_bytes
            cb = _DownloadProgressTqdm._callback
        if cb and total > 0:
            pct = min(int((downloaded / total) * 100), 99)
            # Only call back when percentage actually changes
            if pct > _DownloadProgressTqdm._last_pct:
                _DownloadProgressTqdm._last_pct = pct
                cb(float(pct), downloaded, total)


class ModelDownloader:
    """
    Downloads models from HuggingFace Hub.

    Uses huggingface_hub's snapshot_download for reliable downloads
    with automatic resume support and local caching.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """
        Initialize the downloader.

        Args:
            cache_dir: Directory for model cache. Defaults to settings.MODEL_CACHE_DIR.
        """
        self.cache_dir = Path(cache_dir or settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_api = HfApi()

    @staticmethod
    @huggingface_circuit
    def _snapshot_download_with_circuit(
        repo_id: str,
        local_dir: str,
        token: Optional[str],
        resume: bool,
        tqdm_class: Optional[type] = None,
    ) -> None:
        """Download with circuit breaker protection."""
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir": local_dir,
            "local_dir_use_symlinks": False,
            "token": token,
            "resume_download": resume,
        }
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class
        snapshot_download(**kwargs)

    def _get_local_dir(self, repo_id: str, quantization: str) -> Path:
        """Generate the local directory path for a model."""
        # Format: huggingface/owner--repo--quantization
        safe_name = repo_id.replace("/", "--") + f"--{quantization}"
        return self.cache_dir / "huggingface" / safe_name

    def exists(self, repo_id: str, quantization: str) -> bool:
        """
        Check if a model already exists in local cache.

        Args:
            repo_id: HuggingFace repo (e.g., "google/gemma-2-2b")
            quantization: Q4, Q8, or FP16

        Returns:
            True if model exists and appears complete.
        """
        local_dir = self._get_local_dir(repo_id, quantization)
        if not local_dir.exists():
            return False

        # Check for common model files that indicate a complete download
        indicators = [
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ]
        return any((local_dir / f).exists() for f in indicators)

    def download(
        self,
        repo_id: str,
        quantization: str,
        progress_callback: Optional[ProgressCallback] = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        resume: bool = True,
    ) -> str:
        """
        Download model to cache directory.

        Args:
            repo_id: HuggingFace repo (e.g., "google/gemma-2-2b")
            quantization: Q4, Q8, or FP16
            progress_callback: Called with (progress_pct, downloaded_bytes, total_bytes)
            token: HuggingFace access token for gated models
            trust_remote_code: Whether model requires trust_remote_code
            resume: Whether to resume partial downloads (default True)

        Returns:
            Path to downloaded model directory

        Raises:
            RepoNotFoundError: Repository doesn't exist
            GatedModelError: Model is gated and no valid token provided
            InvalidTokenError: Token is invalid
            DownloadFailedError: Download failed for other reasons
        """
        local_dir = self._get_local_dir(repo_id, quantization)

        logger.info(
            "download_started",
            repo_id=repo_id,
            quantization=quantization,
            local_dir=str(local_dir),
        )

        try:
            # Set up progress tracking via custom tqdm class
            tqdm_cls = None
            if progress_callback is not None:
                _DownloadProgressTqdm._reset(callback=progress_callback)
                tqdm_cls = _DownloadProgressTqdm

            # Use circuit-breaker-protected snapshot_download for reliable downloading
            # It handles resume, parallel downloads, caching, and failure detection
            self._snapshot_download_with_circuit(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token or settings.HF_TOKEN,
                resume=resume,
                tqdm_class=tqdm_cls,
            )

            logger.info(
                "download_complete",
                repo_id=repo_id,
                local_dir=str(local_dir),
            )
            return str(local_dir)

        except CircuitOpenError as e:
            logger.error(
                "circuit_open_download_blocked",
                repo_id=repo_id,
                error=str(e),
            )
            raise DownloadFailedError(
                "HuggingFace service is temporarily unavailable after multiple failures. "
                "Please try again later.",
                details={"repo_id": repo_id, "circuit_error": str(e)},
            )

        except GatedRepoError as e:
            # Must be before RepositoryNotFoundError (GatedRepoError is a subclass)
            logger.warning("gated_repo", repo_id=repo_id, error=str(e))
            raise GatedModelError(
                f"Model '{repo_id}' is gated. Please provide a valid access token "
                "and ensure you have accepted the model's terms of use.",
                details={"repo_id": repo_id},
            )

        except RepositoryNotFoundError as e:
            logger.warning("repo_not_found", repo_id=repo_id, error=str(e))
            raise RepoNotFoundError(
                f"Repository '{repo_id}' not found on HuggingFace",
                details={"repo_id": repo_id},
            )

        except EntryNotFoundError as e:
            logger.warning("entry_not_found", repo_id=repo_id, error=str(e))
            raise DownloadFailedError(
                f"Required file not found in repository '{repo_id}'",
                details={"repo_id": repo_id, "error": str(e)},
            )

        except Exception as e:
            error_msg = str(e).lower()

            # Check for token-related errors
            if "401" in error_msg or "unauthorized" in error_msg:
                logger.warning("invalid_token", repo_id=repo_id, error=str(e))
                raise InvalidTokenError(
                    "HuggingFace token is invalid or expired",
                    details={"repo_id": repo_id},
                )

            # Clean up partial download on unexpected error
            if local_dir.exists():
                logger.warning(
                    "cleaning_partial_download",
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                )
                shutil.rmtree(local_dir, ignore_errors=True)

            logger.error("download_failed", repo_id=repo_id, error=str(e))
            raise DownloadFailedError(
                f"Download failed: {str(e)}",
                details={"repo_id": repo_id, "error": str(e)},
            )

    @huggingface_circuit
    def _get_model_info_with_circuit(self, repo_id: str, token: Optional[str]) -> Any:
        """Get model info with circuit breaker protection."""
        return self.hf_api.model_info(repo_id, token=token)

    def get_model_info(
        self,
        repo_id: str,
        token: Optional[str] = None,
    ) -> dict:
        """
        Get model info without downloading.

        Args:
            repo_id: HuggingFace repo (e.g., "google/gemma-2-2b")
            token: HuggingFace access token for gated models

        Returns:
            Dict with model metadata

        Raises:
            RepoNotFoundError: Repository doesn't exist
            GatedModelError: Model is gated and no valid token provided
        """
        try:
            info = self._get_model_info_with_circuit(repo_id, token=token or settings.HF_TOKEN)

            # Extract config fields (model_type, architectures)
            config = getattr(info, "config", None) or {}
            if hasattr(config, "__dict__"):
                config = config.__dict__ if not isinstance(config, dict) else config

            # Extract card_data fields (license, language)
            card_data = getattr(info, "card_data", None) or {}
            if hasattr(card_data, "__dict__") and not isinstance(card_data, dict):
                card_data = card_data.__dict__

            return {
                "name": info.modelId.split("/")[-1] if info.modelId else repo_id.split("/")[-1],
                "repo_id": info.modelId,
                "params": self._extract_params(info),
                "architecture": getattr(info, "pipeline_tag", None) or "text-generation",
                "is_gated": bool(info.gated),
                "requires_trust_remote_code": self._check_trust_remote_code(info),
                "library_name": getattr(info, "library_name", None),
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "tags": list(getattr(info, "tags", None) or []),
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "model_type": config.get("model_type") if isinstance(config, dict) else None,
                "architectures": config.get("architectures") if isinstance(config, dict) else None,
                "license": card_data.get("license") if isinstance(card_data, dict) else None,
                "language": card_data.get("language") if isinstance(card_data, dict) else None,
            }

        except CircuitOpenError as e:
            logger.error("circuit_open_info_blocked", repo_id=repo_id, error=str(e))
            raise DownloadFailedError(
                "HuggingFace service is temporarily unavailable. Please try again later.",
                details={"repo_id": repo_id, "circuit_error": str(e)},
            )

        except RepositoryNotFoundError:
            raise RepoNotFoundError(
                f"Repository '{repo_id}' not found",
                details={"repo_id": repo_id},
            )

        except GatedRepoError:
            raise GatedModelError(
                f"Model '{repo_id}' is gated. Please provide a valid access token.",
                details={"repo_id": repo_id},
            )

    def _extract_params(self, info) -> str:
        """Extract parameter count from model info."""
        # Try to get from safetensors metadata
        safetensors = getattr(info, "safetensors", None)
        if safetensors:
            total = getattr(safetensors, "total", None)
            if total:
                if total >= 1e12:
                    return f"{total / 1e12:.1f}T"
                elif total >= 1e9:
                    return f"{total / 1e9:.1f}B"
                elif total >= 1e6:
                    return f"{total / 1e6:.0f}M"
                return str(total)

        # Try to extract from model name
        model_id = info.modelId or ""
        for pattern in ["70b", "13b", "7b", "3b", "2b", "1b", "405b", "8b"]:
            if pattern in model_id.lower():
                return pattern.upper()

        return "unknown"

    def _check_trust_remote_code(self, info) -> bool:
        """Check if model requires trust_remote_code."""
        # Check for custom modeling files that indicate custom code
        files = [f.rfilename for f in info.siblings] if info.siblings else []

        custom_code_indicators = [
            "modeling_",
            "_utils.py",
            "configuration_",
        ]

        for filename in files:
            for indicator in custom_code_indicators:
                if indicator in filename and filename.endswith(".py"):
                    return True

        return False

    def delete_cached_model(self, repo_id: str, quantization: str) -> bool:
        """
        Delete a cached model from local storage.

        Args:
            repo_id: HuggingFace repo
            quantization: Q4, Q8, or FP16

        Returns:
            True if model was deleted, False if it didn't exist
        """
        local_dir = self._get_local_dir(repo_id, quantization)

        if not local_dir.exists():
            return False

        logger.info(
            "deleting_cached_model",
            repo_id=repo_id,
            quantization=quantization,
            local_dir=str(local_dir),
        )

        shutil.rmtree(local_dir)
        return True

    def get_cache_size(self, repo_id: str, quantization: str) -> int:
        """
        Get the size of a cached model in bytes.

        Args:
            repo_id: HuggingFace repo
            quantization: Q4, Q8, or FP16

        Returns:
            Size in bytes, or 0 if not cached
        """
        local_dir = self._get_local_dir(repo_id, quantization)

        if not local_dir.exists():
            return 0

        total = 0
        for file in local_dir.rglob("*"):
            if file.is_file():
                total += file.stat().st_size

        return total

    def get_expected_download_size(
        self,
        repo_id: str,
        token: Optional[str] = None,
    ) -> int:
        """
        Get the expected total download size for a model repository.

        Args:
            repo_id: HuggingFace repo (e.g., "google/gemma-2-2b")
            token: HuggingFace access token for gated models

        Returns:
            Expected total size in bytes, or 0 if unable to determine
        """
        try:
            info = self.hf_api.model_info(repo_id, token=token or settings.HF_TOKEN, files_metadata=True)

            total_size = 0
            if info.siblings:
                for sibling in info.siblings:
                    if hasattr(sibling, "size") and sibling.size:
                        total_size += sibling.size

            return total_size

        except Exception as e:
            logger.warning(
                "failed_to_get_expected_size",
                repo_id=repo_id,
                error=str(e),
            )
            return 0

    def get_local_dir_path(self, repo_id: str, quantization: str) -> Path:
        """
        Get the local directory path for a model (public access).

        Args:
            repo_id: HuggingFace repo
            quantization: Q4, Q8, or FP16

        Returns:
            Path to local directory
        """
        return self._get_local_dir(repo_id, quantization)
