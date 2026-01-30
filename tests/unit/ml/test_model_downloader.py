"""Unit tests for ModelDownloader."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import (
    GatedRepoError,
    RepositoryNotFoundError,
)

from millm.core.errors import (
    DownloadFailedError,
    GatedModelError,
    RepoNotFoundError,
)
from millm.ml.model_downloader import ModelDownloader


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "model_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def downloader(temp_cache_dir):
    """Create a ModelDownloader with a temporary cache directory."""
    return ModelDownloader(cache_dir=str(temp_cache_dir))


class TestModelDownloaderInit:
    """Tests for ModelDownloader initialization."""

    def test_creates_cache_directory(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        ModelDownloader(cache_dir=str(cache_dir))

        assert cache_dir.exists()

    def test_uses_provided_cache_dir(self, temp_cache_dir):
        """Test that provided cache directory is used."""
        downloader = ModelDownloader(cache_dir=str(temp_cache_dir))

        assert downloader.cache_dir == temp_cache_dir


class TestModelDownloaderLocalDir:
    """Tests for local directory path generation."""

    def test_get_local_dir_format(self, downloader):
        """Test that local directory path follows expected format."""
        local_dir = downloader._get_local_dir("google/gemma-2-2b", "Q4")

        assert "huggingface" in str(local_dir)
        assert "google--gemma-2-2b--Q4" in str(local_dir)

    def test_get_local_dir_handles_special_chars(self, downloader):
        """Test that repo IDs with special characters are handled."""
        local_dir = downloader._get_local_dir("meta-llama/Llama-3.1-8B", "FP16")

        assert "meta-llama--Llama-3.1-8B--FP16" in str(local_dir)


class TestModelDownloaderExists:
    """Tests for checking if model exists in cache."""

    def test_exists_returns_false_when_no_dir(self, downloader):
        """Test that exists returns False when directory doesn't exist."""
        assert not downloader.exists("nonexistent/model", "Q4")

    def test_exists_returns_false_when_empty_dir(self, downloader, temp_cache_dir):
        """Test that exists returns False when directory is empty."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)

        assert not downloader.exists("google/gemma", "Q4")

    def test_exists_returns_true_with_config_json(self, downloader, temp_cache_dir):
        """Test that exists returns True when config.json is present."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").touch()

        assert downloader.exists("google/gemma", "Q4")

    def test_exists_returns_true_with_safetensors(self, downloader, temp_cache_dir):
        """Test that exists returns True when model.safetensors is present."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)
        (local_dir / "model.safetensors").touch()

        assert downloader.exists("google/gemma", "Q4")


class TestModelDownloaderDownload:
    """Tests for the download method."""

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_calls_snapshot_download(self, mock_snapshot, downloader, temp_cache_dir):
        """Test that download calls huggingface_hub's snapshot_download."""
        mock_snapshot.return_value = "/path/to/model"

        result = downloader.download("google/gemma-2-2b", "Q4")

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["repo_id"] == "google/gemma-2-2b"
        assert call_kwargs["local_dir_use_symlinks"] is False
        assert call_kwargs["resume_download"] is True

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_returns_local_path(self, mock_snapshot, downloader):
        """Test that download returns the local directory path."""
        result = downloader.download("google/gemma-2-2b", "Q4")

        expected_path = str(downloader._get_local_dir("google/gemma-2-2b", "Q4"))
        assert result == expected_path

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_passes_token(self, mock_snapshot, downloader):
        """Test that token is passed to snapshot_download."""
        downloader.download("google/gemma-2-2b", "Q4", token="hf_test_token")

        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["token"] == "hf_test_token"

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_raises_repo_not_found(self, mock_snapshot, downloader):
        """Test that RepoNotFoundError is raised for missing repos."""
        mock_snapshot.side_effect = RepositoryNotFoundError("Not found")

        with pytest.raises(RepoNotFoundError) as exc_info:
            downloader.download("nonexistent/model", "Q4")

        assert "nonexistent/model" in str(exc_info.value.message)
        assert exc_info.value.details["repo_id"] == "nonexistent/model"

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_raises_gated_model_error(self, mock_snapshot, downloader):
        """Test that GatedModelError is raised for gated repos."""
        mock_snapshot.side_effect = GatedRepoError("Gated")

        with pytest.raises(GatedModelError) as exc_info:
            downloader.download("meta-llama/Llama-2-7b", "Q4")

        assert "meta-llama/Llama-2-7b" in str(exc_info.value.message)
        assert "gated" in str(exc_info.value.message).lower()

    @patch("millm.ml.model_downloader.snapshot_download")
    @patch("millm.ml.model_downloader.shutil.rmtree")
    def test_download_cleans_up_on_failure(self, mock_rmtree, mock_snapshot, downloader, temp_cache_dir):
        """Test that partial downloads are cleaned up on failure."""
        mock_snapshot.side_effect = Exception("Network error")

        # Create the directory to simulate partial download
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)

        with pytest.raises(DownloadFailedError):
            downloader.download("google/gemma", "Q4")

        mock_rmtree.assert_called_once()


class TestModelDownloaderGetModelInfo:
    """Tests for get_model_info method."""

    @patch.object(ModelDownloader, "_extract_params", return_value="2B")
    @patch.object(ModelDownloader, "_check_trust_remote_code", return_value=False)
    def test_get_model_info_returns_dict(self, mock_trust, mock_params, downloader):
        """Test that get_model_info returns expected structure."""
        mock_info = MagicMock()
        mock_info.modelId = "google/gemma-2-2b"
        mock_info.gated = False
        mock_info.pipeline_tag = "text-generation"
        mock_info.library_name = "transformers"
        mock_info.downloads = 1000
        mock_info.likes = 50
        mock_info.siblings = []

        with patch.object(downloader.hf_api, "model_info", return_value=mock_info):
            result = downloader.get_model_info("google/gemma-2-2b")

        assert result["name"] == "gemma-2-2b"
        assert result["repo_id"] == "google/gemma-2-2b"
        assert result["params"] == "2B"
        assert result["architecture"] == "text-generation"
        assert result["is_gated"] is False
        assert result["requires_trust_remote_code"] is False

    def test_get_model_info_raises_repo_not_found(self, downloader):
        """Test that RepoNotFoundError is raised for missing repos."""
        with patch.object(downloader.hf_api, "model_info") as mock_info:
            mock_info.side_effect = RepositoryNotFoundError("Not found")

            with pytest.raises(RepoNotFoundError):
                downloader.get_model_info("nonexistent/model")

    def test_get_model_info_raises_gated_error(self, downloader):
        """Test that GatedModelError is raised for gated repos without token."""
        with patch.object(downloader.hf_api, "model_info") as mock_info:
            mock_info.side_effect = GatedRepoError("Gated")

            with pytest.raises(GatedModelError):
                downloader.get_model_info("meta-llama/Llama-2-7b")


class TestModelDownloaderExtractParams:
    """Tests for parameter extraction."""

    def test_extract_params_from_safetensors(self, downloader):
        """Test extraction from safetensors metadata."""
        mock_info = MagicMock()
        mock_safetensors = MagicMock()
        mock_safetensors.total = 2_000_000_000
        mock_info.safetensors = mock_safetensors
        mock_info.modelId = "test/model"

        result = downloader._extract_params(mock_info)

        assert result == "2.0B"

    def test_extract_params_trillions(self, downloader):
        """Test extraction for trillion-parameter models."""
        mock_info = MagicMock()
        mock_safetensors = MagicMock()
        mock_safetensors.total = 1_500_000_000_000
        mock_info.safetensors = mock_safetensors
        mock_info.modelId = "test/model"

        result = downloader._extract_params(mock_info)

        assert result == "1.5T"

    def test_extract_params_millions(self, downloader):
        """Test extraction for million-parameter models."""
        mock_info = MagicMock()
        mock_safetensors = MagicMock()
        mock_safetensors.total = 350_000_000
        mock_info.safetensors = mock_safetensors
        mock_info.modelId = "test/model"

        result = downloader._extract_params(mock_info)

        assert result == "350M"

    def test_extract_params_from_name(self, downloader):
        """Test extraction from model name when safetensors not available."""
        mock_info = MagicMock()
        mock_info.safetensors = None
        mock_info.modelId = "meta-llama/Llama-2-7b"

        result = downloader._extract_params(mock_info)

        assert result == "7B"

    def test_extract_params_unknown(self, downloader):
        """Test that unknown is returned when params can't be determined."""
        mock_info = MagicMock()
        mock_info.safetensors = None
        mock_info.modelId = "test/custom-model"

        result = downloader._extract_params(mock_info)

        assert result == "unknown"


class TestModelDownloaderTrustRemoteCode:
    """Tests for trust_remote_code detection."""

    def test_detects_custom_modeling_file(self, downloader):
        """Test that modeling_*.py files trigger trust_remote_code."""
        mock_info = MagicMock()
        mock_sibling = MagicMock()
        mock_sibling.rfilename = "modeling_custom.py"
        mock_info.siblings = [mock_sibling]

        result = downloader._check_trust_remote_code(mock_info)

        assert result is True

    def test_detects_configuration_file(self, downloader):
        """Test that configuration_*.py files trigger trust_remote_code."""
        mock_info = MagicMock()
        mock_sibling = MagicMock()
        mock_sibling.rfilename = "configuration_model.py"
        mock_info.siblings = [mock_sibling]

        result = downloader._check_trust_remote_code(mock_info)

        assert result is True

    def test_ignores_standard_files(self, downloader):
        """Test that standard files don't trigger trust_remote_code."""
        mock_info = MagicMock()
        mock_sibling1 = MagicMock()
        mock_sibling1.rfilename = "config.json"
        mock_sibling2 = MagicMock()
        mock_sibling2.rfilename = "model.safetensors"
        mock_info.siblings = [mock_sibling1, mock_sibling2]

        result = downloader._check_trust_remote_code(mock_info)

        assert result is False


class TestModelDownloaderDeleteCache:
    """Tests for delete_cached_model method."""

    def test_delete_returns_false_when_not_cached(self, downloader):
        """Test that delete returns False when model isn't cached."""
        result = downloader.delete_cached_model("nonexistent/model", "Q4")

        assert result is False

    def test_delete_removes_directory(self, downloader, temp_cache_dir):
        """Test that delete removes the cached model directory."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)
        (local_dir / "model.safetensors").touch()

        result = downloader.delete_cached_model("google/gemma", "Q4")

        assert result is True
        assert not local_dir.exists()


class TestModelDownloaderCacheSize:
    """Tests for get_cache_size method."""

    def test_cache_size_returns_zero_when_not_cached(self, downloader):
        """Test that cache size returns 0 for non-existent models."""
        result = downloader.get_cache_size("nonexistent/model", "Q4")

        assert result == 0

    def test_cache_size_calculates_total(self, downloader, temp_cache_dir):
        """Test that cache size calculates total file size."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)

        # Create files with known sizes
        file1 = local_dir / "config.json"
        file1.write_text("x" * 100)  # 100 bytes

        file2 = local_dir / "model.bin"
        file2.write_text("y" * 1000)  # 1000 bytes

        result = downloader.get_cache_size("google/gemma", "Q4")

        assert result == 1100
